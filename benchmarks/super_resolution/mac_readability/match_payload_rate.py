"""Match 1080p mean elementary payload to each existing 720p clip, on the Mac.

The target is measured six-second video bytes, not a claimed physical-link cap.
Select by absolute rate error only, without looking at OCR or image quality.
All attempts are preserved. No final-test material is accessed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def rate(run):
    return sum(int(p["size"]) for p in run["probe"]["packets"]) * 8 / 6000


def exact_pts(run):
    pts = [float(p["pts_time"]) for p in run["probe"]["packets"]]
    return all(any(abs(x - t) < 1e-5 for x in pts) for t in [2.8, 5.8])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    os.umask(0o077)
    a.output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((a.baseline / "MANIFEST.json").read_text())
    source = json.loads((a.source / "manifest.json").read_text())
    assert baseline["source_manifest_sha256"] == sha(a.source / "manifest.json")
    result = dict(
        schema="matched_actual_payload_v1",
        source_sha256=sha(a.source / "manifest.json"),
        baseline_sha256=sha(a.baseline / "MANIFEST.json"),
        script_sha256=sha(Path(__file__)),
        scope=__doc__,
        platform=platform.platform(),
        started=time.time(),
        complete=False,
        protocol=dict(
            tolerance_relative=0.03,
            max_attempts=6,
            reference="Existing720p600k clip, same document and two source states",
            matching="Each document independently; exact2.8/5.8s PTS required; nearest measured rate, never quality",
            encoder="Same actual Mac HEVC hardware,420,30fps,GOP60,bf0; only requested bitrate/maxrate/bufsize adjusted",
            caveat="Matching six-second mean video payload does not match instantaneous bursts or include network overhead",
        ),
        groups=[],
    )
    save(a.output / "MANIFEST.json", result)
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    try:
        for group in dict.fromkeys(s["group"] for s in source["states"]):
            lo = next(
                r
                for r in baseline["runs"]
                if r["group"] == group and r["height"] == 720
            )
            hi = next(
                r
                for r in baseline["runs"]
                if r["group"] == group and r["height"] == 1080
            )
            assert (
                sha(a.baseline / lo["path"]) == lo["sha256"]
                and sha(a.baseline / hi["path"]) == hi["sha256"]
            )
            target = rate(lo)
            lower = 20.0
            upper = 600.0
            request = min(599, max(21, round(600 * target / rate(hi))))
            entry = dict(
                group=group,
                target_kbps=target,
                reference_id=lo["id"],
                reference_sha256=lo["sha256"],
                attempts=[],
            )
            result["groups"].append(entry)
            for attempt in range(6):
                name = f"{group}-1080p-match-a{attempt + 1:02d}-{request}k"
                path = a.output / (name + ".mp4")
                command = list(hi["command"])
                command[-1] = str(path)
                for opt in ["-b:v", "-maxrate", "-bufsize"]:
                    command[command.index(opt) + 1] = f"{request}k"
                started = time.time()
                run = subprocess.run(
                    command, capture_output=True, text=True, timeout=180
                )
                (a.output / (name + ".log")).write_text(run.stderr)
                if run.returncode:
                    raise RuntimeError(run.stderr[-1000:])
                probe = json.loads(
                    subprocess.check_output(
                        [
                            "/opt/homebrew/bin/ffprobe",
                            "-v",
                            "error",
                            "-select_streams",
                            "v:0",
                            "-count_frames",
                            "-show_streams",
                            "-show_packets",
                            "-show_entries",
                            "packet=pts_time,dts_time,size,flags:stream=codec_name,width,height,pix_fmt,nb_read_frames,color_range,color_space",
                            "-of",
                            "json",
                            str(path),
                        ],
                        text=True,
                    )
                )
                row = dict(
                    id=name,
                    path=path.name,
                    sha256=sha(path),
                    requested_kbps=request,
                    probe=probe,
                    command=command,
                    state_ids=hi["state_ids"],
                    group=group,
                    category=hi["category"],
                    height=1080,
                    elapsed_s=time.time() - started,
                )
                row.update(actual_kbps=rate(row), exact_sample_pts=exact_pts(row))
                assert (
                    probe["streams"][0]["height"] == 1080
                    and probe["streams"][0]["pix_fmt"] == "yuv420p"
                )
                row["relative_rate_error"] = row["actual_kbps"] / target - 1
                entry["attempts"].append(row)
                save(a.output / "MANIFEST.json", result)
                print(
                    name,
                    round(row["actual_kbps"], 3),
                    "target",
                    round(target, 3),
                    flush=True,
                )
                if abs(row["relative_rate_error"]) <= 0.03 and row["exact_sample_pts"]:
                    break
                if row["actual_kbps"] > target:
                    upper = request
                else:
                    lower = request
                estimate = round(request * target / row["actual_kbps"])
                request = max(round(lower + 1), min(round(upper - 1), estimate))
                if request <= lower or request >= upper:
                    break
            eligible = [r for r in entry["attempts"] if r["exact_sample_pts"]]
            assert eligible, "No exact frozen samples"
            chosen = min(eligible, key=lambda r: abs(r["relative_rate_error"]))
            entry.update(
                selected_id=chosen["id"],
                matched=abs(chosen["relative_rate_error"]) <= 0.03,
            )
            save(a.output / "MANIFEST.json", result)
        result.update(complete=True, finished=time.time())
        save(a.output / "MANIFEST.json", result)
    finally:
        awake.terminate()
        awake.wait(timeout=5)


if __name__ == "__main__":
    main()
