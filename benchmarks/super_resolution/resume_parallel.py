"""Resume a stopped video run with isolated clip workers and audited assembly.

Completed videos are immutable inputs, never overwritten or recomputed. This is
an offline quality workflow: shared-resource timing is not viewer throughput.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from isharescreen.enhancement.models import sha256
from .video_report import audit
from .video_perceptual import audit_scores, summarize


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows, fields=None):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_key(row):
    return row["clip"], row["model"], int(row["frame"])


def partition(clips, workers):
    """Disjoint deterministic longest-first assignment, balanced by frame count."""
    buckets = [[] for _ in range(min(workers, len(clips)))]
    loads = [0] * len(buckets)
    for clip in sorted(clips, key=lambda c: (-c["frames"], c["id"])):
        index = min(range(len(buckets)), key=lambda i: (loads[i], i))
        buckets[index].append(clip)
        loads[index] += clip["frames"]
    return buckets


def snapshot(run, manifest_path):
    manifest = read_json(manifest_path)
    env = read_json(run / "environment.json")
    if sha256(manifest_path) != env["manifest_sha256"]:
        raise ValueError("Original manifest changed")
    results = read_json(run / "clips.json")
    keys = [(r["clip"], r["model"]) for r in results]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate published video")
    definitions = {c["id"]: c for c in manifest["clips"]}
    if len(definitions) != len(manifest["clips"]):
        raise ValueError("Duplicate input clip")
    raw = read_rows(run / "quality.csv")
    selected = [r for r in raw if (r["clip"], r["model"]) in set(keys)]
    expected = set()
    for result in results:
        clip = definitions[result["clip"]]
        if (not result["lossless_roundtrip_verified"] or result["frames"] != clip["frames"]
                or result["inference_calls"] + result["reused_frames"] != clip["frames"]):
            raise ValueError("Unverified published video")
        if sha256(run / result["output"]) != result["sha256"]:
            raise ValueError("Published video changed")
        indices = set(range(0, clip["frames"], env["metrics_every"])) | {clip["frames"] - 1}
        expected.update((clip["id"], result["model"], i) for i in indices)
    actual = [sample_key(r) for r in selected]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Incomplete published sample coverage")
    return manifest, env, results, selected


def subset_manifest(manifest, clips, source_parent):
    value = dict(manifest, clips=[])
    for clip in clips:
        item = dict(clip)
        for role in ("reference", "low"):
            item[role] = str((source_parent / clip[role]).resolve())
        value["clips"].append(item)
    return value


def validate_runtime(base, current):
    for field in ("python", "platform", "torch", "av", "cuda", "runner_sha256",
                  "adapter_sha256", "precision", "reuse_identical", "metrics_every", "gpu"):
        if base[field] != current[field]:
            raise ValueError(f"Shard runtime changed: {field}")


def assemble(target, manifest_path, base_env, status, segments):
    """Hard-link only verified outputs into a fresh self-contained run directory."""
    target.mkdir(parents=True, exist_ok=False)
    manifest = read_json(manifest_path)
    results, rows, seen, evidence = [], [], set(), []
    for folder, env, records, raw in segments:
        validate_runtime(base_env, env)
        evidence.append(dict(run=str(folder.resolve()),
                             environment_sha256=sha256(folder / "environment.json"),
                             results_sha256=sha256(folder / "clips.json"),
                             quality_sha256=sha256(folder / "quality.csv")))
        for result in records:
            key = result["clip"], result["model"]
            if key in seen:
                raise ValueError("Overlapping video shards")
            seen.add(key)
            source = folder / result["output"]
            if sha256(source) != result["sha256"]:
                raise ValueError("Shard output changed")
            relative = Path(result["model"]) / (result["clip"] + ".mkv")
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            results.append(dict(result, output=str(relative)))
        rows.extend(raw)
    required = {(c["id"], model) for c in manifest["clips"] for model in status
                if status[model]["state"] != "unavailable"}
    if seen != required:
        raise ValueError("Incomplete assembled video coverage")
    statuses = {key: dict(value, state="completed") if value["state"] != "unavailable"
                else value for key, value in status.items()}
    environment = dict(base_env, manifest_sha256=sha256(manifest_path),
                       pid=os.getpid(), started=time.time(), command=sys.argv,
                       execution="resumed_clip_shards", segments=evidence,
                       assembly_runner_sha256=sha256(Path(__file__)))
    write_json(target / "environment.json", environment)
    write_json(target / "clips.json", results)
    write_json(target / "status.json", statuses)
    write_rows(target / "quality.csv", rows)
    write_json(target / "COMPLETED.json", dict(finished=time.time(), videos=len(results),
               frames_written=sum(r["frames"] for r in results), quality_rows=len(rows), status=statuses))
    try:
        audit(target, manifest_path)
    except BaseException:
        # Never leave a positive completion marker after a failed assembly audit.
        (target / "COMPLETED.json").rename(target / "AUDIT_REJECTED.json")
        raise
    return environment


def derived_cache(target, old_cache, old_run, original_manifest, view, view_manifest):
    """Explicit provenance rebinding, followed by mandatory per-sample RGB checks.

    This is a derived cache, not a rewrite of the original experiment. Rebinding
    is allowed only for byte-identical published videos and matching references.
    The metric worker still checks its network state and every reused RGB hash.
    """
    previous = read_json(old_cache / "environment.json")
    if (previous["producer_environment_sha256"] != sha256(old_run / "environment.json")
            or previous["manifest_sha256"] != sha256(original_manifest)
            or previous["run"] != str(old_run.resolve())):
        raise ValueError("Original metric cache binding changed")
    old_results = {(r["clip"], r["model"]): r for r in read_json(old_run / "clips.json")}
    new_results = {(r["clip"], r["model"]): r for r in read_json(view / "clips.json")}
    original_clips = {c["id"]: c for c in read_json(original_manifest)["clips"]}
    for clip in read_json(view_manifest)["clips"]:
        for role in ("reference", "low"):
            if clip[f"{role}_sha256"] != original_clips[clip["id"]][f"{role}_sha256"]:
                raise ValueError("Cache reference/input changed")
    selected = []
    for row in read_rows(old_cache / "quality.csv"):
        key = row["clip"], row["model"]
        if key not in new_results:
            continue
        if key not in old_results or old_results[key]["sha256"] != new_results[key]["sha256"]:
            raise ValueError("Cached output not identical to published source")
        selected.append(row)
    target.mkdir(parents=True, exist_ok=False)
    write_rows(target / "quality.csv", selected, fields=list(read_rows(old_cache / "quality.csv")[0]))
    environment = dict(previous, manifest_sha256=sha256(view_manifest),
                       producer_environment_sha256=sha256(view / "environment.json"), run=str(view.resolve()),
                       derived_cache=dict(origin=str(old_cache.resolve()),
                         origin_environment_sha256=sha256(old_cache / "environment.json"),
                         origin_quality_sha256=sha256(old_cache / "quality.csv"),
                         origin_environment=previous,
                         rule="Identical reference/input and output file hashes; RGB hashes rechecked by scorer"))
    write_json(target / "environment.json", environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Stopped, preserved producer run")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Stopped LPIPS companion")
    parser.add_argument("--output", type=Path, required=True, help="Fresh campaign directory")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=21600)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout_seconds <= 0:
        parser.error("workers and timeout must be positive")
    source_env = read_json(args.run / "environment.json")
    cache_env = read_json(args.cache / "environment.json")
    for env in (source_env, cache_env):
        try:
            os.kill(env["pid"], 0)
        except ProcessLookupError:
            continue
        raise RuntimeError("Stop and preserve the original processes before resuming")
    args.output.mkdir(parents=True, exist_ok=False)
    preserved = {str(path): sha256(path) for path in
                 (args.run / "environment.json", args.run / "clips.json", args.run / "quality.csv",
                  args.cache / "environment.json", args.cache / "quality.csv", args.manifest)}
    manifest, original_env, original_results, original_rows = snapshot(args.run, args.manifest)
    status = read_json(args.run / "status.json")
    complete = {(r["clip"], r["model"]) for r in original_results}
    missing = [(model, clip) for model, state in status.items() if state["state"] != "unavailable"
               for clip in manifest["clips"] if (clip["id"], model) not in complete]
    missing_models = {model for model, _ in missing}
    if len(missing_models) > 1:
        raise ValueError("This resume path supports one remaining model; partition models separately")
    runner = Path(original_env["command"][0]).resolve()
    import isharescreen.enhancement.models as adapter
    if sha256(runner) != original_env["runner_sha256"] or sha256(Path(adapter.__file__)) != original_env["adapter_sha256"]:
        raise ValueError("Frozen inference code changed; do not silently resume")
    state = dict(pid=os.getpid(), started=time.time(), status="starting", jobs=[],
                 preserved_run=str(args.run.resolve()), preserved_cache=str(args.cache.resolve()))
    deadline = time.monotonic() + args.timeout_seconds
    def publish():
        write_json(args.output / "state.json", state)
    def launch(name, command):
        command = ["timeout", "--signal=TERM", "--kill-after=10s",
                   f"{max(1, int(deadline-time.monotonic()))}s", *command]
        log = (args.output / f"{name}.log").open("x")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        job = dict(name=name, command=command, pid=process.pid, started=time.time(), status="running")
        state["jobs"].append(job)
        publish()
        return process, log, job
    def wait_all(jobs):
        while True:
            running = False
            for process, log, job in jobs:
                code = process.poll()
                if code is None:
                    running = True
                elif job["status"] == "running":
                    job.update(returncode=code, status="completed" if code == 0 else "failed", finished=time.time())
                    log.close()
                    publish()
                    if code:
                        raise RuntimeError(f"{job['name']} failed ({code}); no automatic retry")
            if not running:
                return
            if time.monotonic() > deadline:
                print("HARD TIMEOUT: terminating this campaign's children", flush=True)
                for process, _, _ in jobs:
                    if process.poll() is None:
                        process.terminate()
                for process, _, _ in jobs:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                raise TimeoutError("Parallel campaign deadline")
            print("ALIVE", [(j["name"], j["pid"]) for p, _, j in jobs if p.poll() is None], flush=True)
            publish()
            time.sleep(15)
    publish()
    try:
        jobs, shards = [], []
        for i, clips in enumerate(partition([c for _, c in missing], args.workers)):
            path = args.output / f"sr-manifest-{i}.json"
            write_json(path, subset_manifest(manifest, clips, args.manifest.parent))
            output = args.output / f"sr-{i}"
            command = [sys.executable, str(runner), "--manifest", str(path), "--output", str(output),
                       "--models", next(iter(missing_models)), "--precision", original_env["precision"],
                       "--device", "cuda", "--metrics-every", str(original_env["metrics_every"])]
            if original_env["reuse_identical"]:
                command.append("--reuse-identical")
            jobs.append(launch(f"sr-{i}", command))
            shards.append((output, path))
        state["status"] = "upscaling"
        publish()
        wait_all(jobs)
        segments = [(args.run, original_env, original_results, original_rows)]
        for folder, path in shards:
            _, _, env, records = audit(folder, path)
            segments.append((folder, env, records, read_rows(folder / "quality.csv")))
        assembled = args.output / "reconstructions"
        assembled_env = assemble(assembled, args.manifest, original_env, status, segments)
        state["status"] = "perceptual"
        publish()
        jobs, companions = [], []
        full_records = read_json(assembled / "clips.json")
        full_rows = read_rows(assembled / "quality.csv")
        for i, clips in enumerate(partition(manifest["clips"], args.workers)):
            ids = {c["id"] for c in clips}
            path = args.output / f"metric-manifest-{i}.json"
            write_json(path, subset_manifest(manifest, clips, args.manifest.parent))
            view = args.output / f"metric-source-{i}"
            filtered = [(assembled, assembled_env, [r for r in full_records if r["clip"] in ids],
                         [r for r in full_rows if r["clip"] in ids])]
            assemble(view, path, assembled_env, status, filtered)
            cache = args.output / f"metric-cache-{i}"
            derived_cache(cache, args.cache, args.run, args.manifest, view, path)
            output = args.output / f"metric-{i}"
            command = [sys.executable, "-u", "-m", "benchmarks.super_resolution.video_perceptual",
                       "--run", str(view), "--manifest", str(path), "--output", str(output),
                       "--device", "cuda", "--threads", "2", "--pixel-metrics", "cuda",
                       "--reference-cache-mib", "3072", "--resume-from", str(cache),
                       "--timeout-seconds", str(max(1, deadline-time.monotonic()))]
            jobs.append(launch(f"metric-{i}", command))
            companions.append(output)
        wait_all(jobs)
        target = args.output / "perceptual"
        target.mkdir(exist_ok=False)
        rows, provenance, metric_env = [], [], None
        for folder in companions:
            marker = read_json(folder / "COMPLETED.json")
            for name in ("quality", "summary", "environment", "report"):
                file = folder / ({"quality":"quality.csv", "summary":"summary.json", "environment":"environment.json", "report":"REPORT.md"}[name])
                if sha256(file) != marker[f"{name}_sha256"]:
                    raise ValueError("Metric shard changed after completion")
            current = read_json(folder / "environment.json")
            if metric_env is not None:
                for key in ("state_sha256", "backbone_sha256", "linear_sha256", "torch", "torchvision", "device", "lpips_version"):
                    if current[key] != metric_env[key]:
                        raise ValueError(f"Metric shard identity changed: {key}")
            metric_env = current
            provenance.append(dict(run=str(folder.resolve()), marker=marker))
            rows.extend(read_rows(folder / "quality.csv"))
        summary = summarize(rows)
        expected = {sample_key(r):r for r in full_rows}
        audit_scores(rows, summary, expected)
        write_rows(target / "quality.csv", rows)
        write_json(target / "summary.json", summary)
        write_json(target / "environment.json", dict(execution="parallel_metric_shards", segments=provenance,
                   producer_environment_sha256=sha256(assembled / "environment.json"),
                   origin_cache_environment_sha256=sha256(args.cache / "environment.json"),
                   origin_cache_quality_sha256=sha256(args.cache / "quality.csv"),
                   manifest_sha256=sha256(args.manifest), metric_identity={k:metric_env[k] for k in
                       ("state_sha256", "backbone_sha256", "linear_sha256", "torch", "torchvision", "device", "lpips_version")}))
        lines = ["# Recorded-video reconstruction quality", "", "Equal-clip means; paired RGB samples; original pretrained models; no fine-tuning.",
                 "Saved FP32 LPIPS scores were reused only after source-file and sampled-RGB verification.",
                 "Shared-GPU timing is not isolated model latency or viewer FPS.", "",
                 "| Method | Clips | Samples | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | ΔPSNR vs bicubic | ΔLPIPS vs bicubic |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for model, value in summary.items():
            lines.append(f"| {model} | {value['clips']} | {value['samples']} | {value['psnr']:.3f} | {value['ssim']:.5f} | {value['lpips']:.5f} | {value['delta_psnr_vs_bicubic']:+.3f} | {value['delta_lpips_vs_bicubic']:+.5f} |")
        (target / "REPORT.md").write_text("\n".join(lines) + "\n")
        marker = dict(rows=len(rows), clips=len(manifest["clips"]), models=list(summary), finished=time.time())
        for filename, digest in preserved.items():
            if sha256(Path(filename)) != digest:
                raise ValueError("Preserved resume inputs changed during campaign")
        for name, filename in (("quality", "quality.csv"), ("summary", "summary.json"), ("environment", "environment.json"), ("report", "REPORT.md")):
            marker[f"{name}_sha256"] = sha256(target / filename)
        write_json(target / "COMPLETED.json", marker)
        state.update(status="completed", finished=time.time(), videos=len(full_records), quality_rows=len(rows))
        publish()
        print("COMPLETED parallel resume", flush=True)
    except BaseException as exc:
        state.update(status="failed", error=str(exc), finished=time.time())
        publish()
        raise


if __name__ == "__main__":
    main()
