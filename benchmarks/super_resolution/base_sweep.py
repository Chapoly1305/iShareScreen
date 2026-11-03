"""Reproducible low-bandwidth base-resolution sweep; selected development clips.

Uses existing lossless RGB excerpts and official native x2 adapters. All quality
scores use the same 1080p reference at matching timestamps. No OCR or tuning.
"""

import argparse
from collections import defaultdict
from fractions import Fraction
import gc
import html
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import av
import cv2
import numpy as np
from PIL import Image
import torch

from benchmarks.super_resolution.bitrate_poc import packet_stats, run
from benchmarks.super_resolution.gallery import crop_strip
from benchmarks.super_resolution.metrics import metrics
from benchmarks.super_resolution.prepare_video import probe
from benchmarks.super_resolution.train import perceptual_network
from benchmarks.super_resolution.training_state import atomic_json
from benchmarks.super_resolution.video_perceptual import normalized_tensor
from isharescreen.enhancement import candidates
from isharescreen.enhancement.models import sha256

MODELS = ("quicksrnet", "plainusr", "rt4ksr", "efrlfn")
TIMES = (1, 2, 3)


def configuration(height, bps, chroma="420", fps=30, group="primary"):
    return dict(
        id=f"{height}p-{bps // 1000}k-{chroma}-{fps}",
        height=height,
        width=height * 16 // 9,
        bps=bps,
        chroma=chroma,
        fps=fps,
        group=group,
    )


def configs():
    out = [
        configuration(h, b)
        for b in (64000, 128000, 256000, 600000, 1000000)
        for h in (540, 720, 1080)
    ]
    out += [
        configuration(h, b, "444", group="chroma-control")
        for b in (128000, 600000)
        for h in (540, 720, 1080)
    ]
    out += [
        configuration(720, b, fps=60, group="fps-control") for b in (128000, 600000)
    ]
    return out


def prepare(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    (root / "runner.py").write_bytes(Path(__file__).read_bytes())
    parent = json.loads((args.prepared / "manifest.json").read_text())
    receipt = json.loads((args.prepared / "PREPARED.json").read_text())
    assert receipt["manifest_sha256"] == sha256(args.prepared / "manifest.json")
    assert parent["frames"] == 240 and parent["fps"] == "60"
    manifest = dict(
        schema="sr_base_sweep_v1",
        command=sys.argv,
        started=time.time(),
        parent_manifest=str((args.prepared / "manifest.json").resolve()),
        parent_manifest_sha256=sha256(args.prepared / "manifest.json"),
        script_sha256=sha256(Path(__file__)),
        configs=[
            c for c in configs() if not args.primary_only or c["group"] == "primary"
        ],
        scenes=[],
        runs=[],
        sample_times_s=TIMES,
        reference=parent.get("reference", "Decoded RGB8 recording proxy"),
        encoder="libx265 fast + zerolatency; no B frames/lookahead; 2-second GOP, 1-second VBV, strict-cbr/HRD",
        color="8-bit full-range BT.709; Lanczos base reduction; explicit 420/444",
        display="1920x1080 RGB8; baseline bicubic; native x2 SR then INTER_AREA downsample when needed",
        scope=f"{len(parent['scenes'])} preselected 4-second development excerpts. Quality sampled at 1,2,3s; first second excluded from bitrate. Not live network, sustained FPS, temporal quality or text correctness.",
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[
            0
        ],
    )
    for scene in parent["scenes"]:
        ref = (args.prepared / scene["files"]["reference"]["path"]).resolve()
        assert sha256(ref) == scene["files"]["reference"]["sha256"]
        copied = {
            k: scene[k]
            for k in (
                "id",
                "title",
                "capture",
                "crops",
                "excerpt_start_frame",
                "source_sha256",
            )
        }
        copied.update(reference=str(ref), reference_sha256=sha256(ref))
        (root / scene["id"]).mkdir()
        count = 0
        with av.open(str(ref)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                assert (frame.width, frame.height) == (
                    1920,
                    1080,
                ) and not frame.is_corrupt
                assert abs(float(frame.pts * frame.time_base) - i / 60) < 0.0011
                if i in [t * 60 for t in TIMES]:
                    Image.fromarray(frame.to_ndarray(format="rgb24")).save(
                        root / scene["id"] / f"reference-{i // 60}s.png"
                    )
                count += 1
        assert count == 240
        manifest["scenes"].append(copied)
    atomic_json(root / "manifest.json", manifest)
    for cfg in manifest["configs"]:
        for scene in manifest["scenes"]:
            folder = root / scene["id"] / cfg["id"]
            folder.mkdir()
            target = folder / "encoded.mp4"
            kbps = cfg["bps"] // 1000
            fps = cfg["fps"]
            h = cfg["height"]
            w = cfg["width"]
            pix = "yuv" + cfg["chroma"] + "p"
            params = (
                f"pools=4:frame-threads=1:keyint={fps * 2}:min-keyint=1:bframes=0:rc-lookahead=0:"
                f"bitrate={kbps}:vbv-maxrate={kbps}:vbv-bufsize={kbps}:vbv-init=0.5:strict-cbr=1:hrd=1:log-level=info"
            )
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-nostdin",
                "-i",
                scene["reference"],
                "-an",
                "-vf",
                f"fps={fps},scale={w}:{h}:flags=lanczos:in_range=full:out_range=full:out_color_matrix=bt709",
                "-c:v",
                "libx265",
                "-preset",
                "fast",
                "-tune",
                "zerolatency",
                "-b:v",
                str(cfg["bps"]),
                "-minrate",
                str(cfg["bps"]),
                "-maxrate",
                str(cfg["bps"]),
                "-bufsize",
                str(cfg["bps"]),
                "-pix_fmt",
                pix,
                "-color_range",
                "pc",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-x265-params",
                params,
                "-tag:v",
                "hvc1",
                "-movflags",
                "+faststart",
                str(target),
            ]
            run(cmd, folder / "encode.log")
            stream = probe(target)
            assert (stream["width"], stream["height"]) == (w, h)
            assert stream["pix_fmt"] in (pix, "yuvj" + cfg["chroma"] + "p")
            assert (
                Fraction(stream["avg_frame_rate"]) == fps
                and int(stream["nb_read_frames"]) == fps * 4
            )
            assert (
                stream["color_range"] == "pc"
                and stream["color_space"] == "bt709"
                and stream["has_b_frames"] == 0
            )
            count = 0
            hashes = {}
            with av.open(str(target)) as container:
                for i, frame in enumerate(container.decode(video=0)):
                    assert not frame.is_corrupt and (frame.width, frame.height) == (
                        w,
                        h,
                    )
                    assert abs(float(frame.pts * frame.time_base) - i / fps) < 0.0011
                    if i in [t * fps for t in TIMES]:
                        path = folder / f"input-{i // fps}s.png"
                        Image.fromarray(frame.to_ndarray(format="rgb24")).save(path)
                        hashes[str(i // fps)] = sha256(path)
                    count += 1
            assert count == fps * 4
            rate = packet_stats(target, Fraction(fps), fps)
            underflows = (folder / "encode.log").read_text().count("VBV underflow")
            passed = rate["retained_video_bps"] <= cfg["bps"] * 1.05 and underflows == 0
            row = dict(
                config=cfg["id"],
                scene=scene["id"],
                encoded=str(target.relative_to(root)),
                sha256=sha256(target),
                command=cmd,
                stream=stream,
                rate=rate,
                vbv_underflows=underflows,
                budget_pass=passed,
                input_png_hashes=hashes,
                log_sha256=sha256(folder / "encode.log"),
            )
            manifest["runs"].append(row)
            atomic_json(root / "manifest.json", manifest)
            print(
                f"ENCODED {len(manifest['runs'])}/{len(manifest['configs']) * len(manifest['scenes'])} {scene['id']} {cfg['id']} {rate['retained_video_bps'] / 1000:.1f}k underflows={underflows}",
                flush=True,
            )
    manifest["finished"] = time.time()
    atomic_json(root / "manifest.json", manifest)
    atomic_json(
        root / "PREPARED.json",
        dict(
            manifest_sha256=sha256(root / "manifest.json"), runs=len(manifest["runs"])
        ),
    )


def evaluate(args):
    root = args.output
    manifest = json.loads((root / "manifest.json").read_text())
    assert json.loads((root / "PREPARED.json").read_text())[
        "manifest_sha256"
    ] == sha256(root / "manifest.json")
    for r in manifest["runs"]:
        assert sha256(root / r["encoded"]) == r["sha256"]
    references = {
        (s["id"], t): np.array(
            Image.open(root / s["id"] / f"reference-{t}s.png").convert("RGB")
        )
        for s in manifest["scenes"]
        for t in TIMES
    }
    network, identity = perceptual_network(args.device)
    path = root / "results.json"
    if path.exists():
        result = json.loads(path.read_text())
        assert result["manifest_sha256"] == sha256(root / "manifest.json")
    else:
        result = dict(
            manifest_sha256=sha256(root / "manifest.json"),
            script_sha256=sha256(Path(__file__)),
            started=time.time(),
            device=args.device,
            precision="fp32",
            platform=platform.platform(),
            torch=torch.__version__,
            lpips_identity=identity,
            models={},
            records=[],
            complete=False,
        )
    completed = {
        (r["config"], r["scene"], r["time_s"], r["method"]) for r in result["records"]
    }
    by_config = {c["id"]: c for c in manifest["configs"]}

    def score(runrow, t, method, output, timing):
        cfg = runrow["config"]
        scene = runrow["scene"]
        ref = references[(scene, t)]
        assert output.shape == ref.shape and output.dtype == np.uint8
        scores = metrics(ref, output)
        with torch.inference_mode():
            scores["lpips"] = float(
                network(
                    normalized_tensor(ref, args.device),
                    normalized_tensor(output, args.device),
                ).item()
            )
        image = None
        if t == 2:
            target = root / scene / cfg / f"{method}.png"
            Image.fromarray(output).save(target)
            image = dict(path=str(target.relative_to(root)), sha256=sha256(target))
        result["records"].append(
            dict(
                config=cfg,
                scene=scene,
                time_s=t,
                method=method,
                quality=scores,
                timing=timing,
                image=image,
            )
        )

    for method in ("baseline", *args.models):
        adapter = None
        if method != "baseline":
            adapter = candidates.TorchUpscaler(
                method, device=args.device, precision="fp32"
            )
            assert adapter.spec["scale"] == 2
            result["models"][method] = adapter.provenance
        warmed = set()
        for i, row in enumerate(manifest["runs"]):
            cfg = by_config[row["config"]]
            for t in TIMES:
                key = (row["config"], row["scene"], t, method)
                if key in completed:
                    continue
                file = root / row["scene"] / row["config"] / f"input-{t}s.png"
                assert sha256(file) == row["input_png_hashes"][str(t)]
                rgb = np.array(Image.open(file).convert("RGB"))
                if adapter:
                    if cfg["height"] not in warmed:
                        adapter(rgb)
                        warmed.add(cfg["height"])
                    output, timing = adapter(rgb)
                    assert output.shape == (cfg["height"] * 2, cfg["width"] * 2, 3)
                    started = time.perf_counter()
                    if cfg["height"] != 540:
                        output = cv2.resize(
                            output, (1920, 1080), interpolation=cv2.INTER_AREA
                        )
                    timing["display_resize_ms"] = (time.perf_counter() - started) * 1000
                    timing["display_total_ms"] = (
                        timing["total_ms"] + timing["display_resize_ms"]
                    )
                else:
                    started = time.perf_counter()
                    output = (
                        rgb
                        if cfg["height"] == 1080
                        else cv2.resize(
                            rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC
                        )
                    )
                    timing = dict(
                        display_total_ms=(time.perf_counter() - started) * 1000
                    )
                score(row, t, method, output, timing)
            atomic_json(path, result)
            print(
                f"SCORED {method} {i + 1}/{len(manifest['runs'])} {row['config']} {row['scene']}",
                flush=True,
            )
        del adapter
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
    expected = len(manifest["runs"]) * len(TIMES) * (1 + len(args.models))
    assert len(result["records"]) == expected
    assert (
        len(
            {
                (r["config"], r["scene"], r["time_s"], r["method"])
                for r in result["records"]
            }
        )
        == expected
    )
    result.update(complete=True, finished=time.time())
    atomic_json(path, result)
    atomic_json(
        root / "COMPLETED.json", dict(results_sha256=sha256(path), records=expected)
    )


def report(args):
    import csv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.output
    m = json.loads((root / "manifest.json").read_text())
    r = json.loads((root / "results.json").read_text())
    assert r["complete"] and json.loads((root / "COMPLETED.json").read_text())[
        "results_sha256"
    ] == sha256(root / "results.json")
    grouped = defaultdict(list)
    for row in r["records"]:
        if row["image"]:
            assert sha256(root / row["image"]["path"]) == row["image"]["sha256"]
        grouped[(row["config"], row["method"])].append(row)
    rows = []
    for cfg in m["configs"]:
        runs = [x for x in m["runs"] if x["config"] == cfg["id"]]
        for method in ("baseline", *r["models"]):
            values = grouped[(cfg["id"], method)]
            assert len(values) == len(m["scenes"]) * len(TIMES)
            base = grouped[(cfg["id"], "baseline")]
            row = dict(
                **cfg,
                method=method,
                samples=len(values),
                budget_pass=all(x["budget_pass"] for x in runs),
                max_actual_kbps=max(x["rate"]["retained_video_bps"] for x in runs)
                / 1000,
                mean_actual_kbps=float(
                    np.mean([x["rate"]["retained_video_bps"] for x in runs])
                )
                / 1000,
                underflows=sum(x["vbv_underflows"] for x in runs),
                **{
                    metric: float(np.mean([x["quality"][metric] for x in values]))
                    for metric in ("psnr", "ssim", "lpips")
                },
                display_ms_median=float(
                    np.median([x["timing"]["display_total_ms"] for x in values])
                ),
            )
            for metric in ("psnr", "ssim", "lpips"):
                row["delta_" + metric] = row[metric] - float(
                    np.mean([x["quality"][metric] for x in base])
                )
            # Positive values are improvement, normalized as fraction of baseline perceptual error.
            row["lpips_gain_pct"] = (
                -row["delta_lpips"]
                / float(np.mean([x["quality"]["lpips"] for x in base]))
                * 100
            )
            rows.append(row)
    for row in rows:
        peers = [x for x in rows if x["budget_pass"]]
        row["pareto_lpips"] = bool(
            row["budget_pass"]
            and not any(
                x["max_actual_kbps"] <= row["max_actual_kbps"]
                and x["lpips"] <= row["lpips"]
                and (
                    x["max_actual_kbps"] < row["max_actual_kbps"]
                    or x["lpips"] < row["lpips"]
                )
                for x in peers
            )
        )
    atomic_json(root / "summary.json", rows)
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Keep baseline eligible: improvement claims must survive comparison with doing no ML.
    winners = []
    for bps in (64000, 128000, 256000, 600000, 1000000):
        pool = [x for x in rows if x["bps"] == bps and x["budget_pass"]]
        if pool:
            lp = min(pool, key=lambda x: x["lpips"])
            ss = max(pool, key=lambda x: x["ssim"])
            ps = max(pool, key=lambda x: x["psnr"])
            winners.append(dict(target_bps=bps, lpips=lp, ssim=ss, psnr=ps))
        else:
            winners.append(
                dict(
                    target_bps=bps,
                    unavailable="No base passed encoder-budget gate on every selected scene",
                )
            )
    atomic_json(
        root / "selection.json",
        dict(
            criteria="All tested groups eligible; exploratory, no readability threshold or trained selector. LPIPS and SSIM/PSNR shown independently. Baseline eligible. Budget gate: every selected clip <=105% requested average after 1s and zero VBV underflows.",
            winners=winners,
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for h, col in ((540, "#2563eb"), (720, "#16a34a"), (1080, "#dc2626")):
        for method, marker in (
            ("baseline", "x"),
            ("quicksrnet", "o"),
            ("plainusr", "s"),
            ("rt4ksr", "^"),
            ("efrlfn", "D"),
        ):
            pts = sorted(
                [
                    x
                    for x in rows
                    if x["height"] == h
                    and x["method"] == method
                    and x["group"] == "primary"
                    and x["budget_pass"]
                ],
                key=lambda x: x["max_actual_kbps"],
            )
            for ax, metric in zip(axes, ("lpips", "ssim")):
                ax.plot(
                    [x["max_actual_kbps"] for x in pts],
                    [x[metric] for x in pts],
                    marker=marker,
                    color=col,
                    alpha=0.75,
                    label=f"{h}p {method}",
                )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Maximum scored video Kbps across selected scenes")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("LPIPS (lower is better)")
    axes[1].set_ylabel("SSIM (higher is better)")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(root / "quality-bandwidth.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Low-bandwidth base-resolution sweep",
        "",
        "All outputs scored at 1920×1080. 720p ×2 = 2560×1440, area-downsampled to 1080p here.",
        f"{len(m['scenes'])} existing development excerpts; 4s continuous HEVC encoding, matching frames at 1/2/3s. "
        "30 fps / 420 primary grid: 540p, 720p, 1080p × 64/128/256/600/1000 Kbps. "
        "444/30 controls at 128/600 Kbps for all bases; 720p/420/60 controls at 128/600 Kbps. "
        "x265 fast + zerolatency, no B frames/lookahead. Official pretrained x2 models, no fine-tuning. "
        "RGB PSNR/SSIM and existing AlexNet-v0.1 LPIPS; equal frames and equal scenes. Reference acquisition is documented in manifest.json.",
        "",
        "## Selection across all tested configurations (including chroma/FPS controls)",
        "",
        "| Requested Kbps | Lowest LPIPS | LPIPS | Highest SSIM | SSIM |",
        "|---:|---|---:|---|---:|",
    ]
    for winner in winners:
        if "lpips" not in winner:
            lines.append(
                f"| {winner['target_bps'] / 1000:g} | No all-scene budget pass | — | — | — |"
            )
            continue
        lp = winner["lpips"]
        ss = winner["ssim"]
        lines.append(
            f"| {winner['target_bps'] / 1000:g} | {lp['height']}p {lp['method']} / {lp['chroma']} / {lp['fps']}fps | {lp['lpips']:.5f} | {ss['height']}p {ss['method']} / {ss['chroma']} / {ss['fps']}fps | {ss['ssim']:.5f} |"
        )
    lines += [
        "",
        "Budget eligibility requires every scene to average at most 105% of the request after the first second and zero VBV underflow warnings. "
        "This is an encoder screening gate, not proof that a real 128 Kbps network works. Packet rates exclude IP/RTP/RTCP, audio and control. "
        "One-second packet bins, startup traffic and per-scene failures are in manifest.json. Ineligible scores remain in CSV for diagnosis but are excluded from winner/Pareto selection.",
        "",
        "There is no measured minimum readable bitrate: these selected clips have no human readability labels, and no OCR or temporal metric was added. "
        "Results are a development-set sweep, not a held-out generalization result or a live Apple encoder benchmark. "
        "Timings are warmed sampled PyTorch MPS FP32 adapter + output resize times, not Core ML, sustained FPS, or end-to-end latency.",
        "",
        "[x265 VBV/strict CBR documentation](https://x265.readthedocs.io/en/master/cli.html#cmdoption-strict-cbr).",
        "",
        "Full results: summary.csv / summary.json. Raw rows and model hashes: results.json. Exact encoder commands and input hashes: manifest.json.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    page = [
        '<!doctype html><html><meta charset="utf-8"><title>Base resolution sweep</title><style>body{font:15px system-ui;margin:28px;color:#102030}img{max-width:100%}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccc}section{margin:35px 0}</style><body>',
        "<h1>540p / 720p / 1080p — bandwidth and pretrained enhancement</h1>",
        "<p>Shared 1080p display; official native ×2 weights. Selected development excerpts, three paired timestamps each. No live-network or text-correctness claim.</p>",
        '<p><a href="REPORT.md">Protocol and selection</a> · <a href="summary.csv">All metrics CSV</a> · <a href="selection.json">Selection JSON</a> · <a href="manifest.json">Encoding and validity</a></p>',
        '<img src="quality-bandwidth.png"><h2>420 / 30 fps — each base and budget</h2>',
        "<table><tr><th>Kbps</th><th>Base</th><th>Method</th><th>Actual max Kbps</th><th>Budget</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>LPIPS gain %</th><th>Adapter ms</th></tr>",
    ]
    for row in rows:
        if row["group"] != "primary":
            continue
        page.append(
            f"<tr><td>{row['bps'] // 1000}</td><td>{row['height']}p</td><td>{row['method']}</td><td>{row['max_actual_kbps']:.1f}</td><td>{'PASS' if row['budget_pass'] else 'FAIL'}</td><td>{row['psnr']:.3f}</td><td>{row['ssim']:.5f}</td><td>{row['lpips']:.5f}</td><td>{row['lpips_gain_pct']:+.2f}</td><td>{row['display_ms_median']:.1f}</td></tr>"
        )
    page.append("</table>")
    for bps in (64000, 128000, 256000, 600000, 1000000):
        page.append(f"<h2>{bps // 1000} Kbps / 420 / 30 — matched detail crops</h2>")
        for scene in m["scenes"]:
            images = [
                Image.open(root / scene["id"] / "reference-2s.png").convert("RGB")
            ]
            labels = ["Original"]
            for h in (540, 720, 1080):
                cfg = configuration(h, bps)["id"]
                for method in ("baseline", *r["models"]):
                    images.append(
                        Image.open(root / scene["id"] / cfg / f"{method}.png").convert(
                            "RGB"
                        )
                    )
                    labels.append(f"{h}p {method}")
            crop = scene["crops"][0]
            target = root / scene["id"] / f"comparison-{bps // 1000}k.png"
            crop_strip(images, crop["xywh"], labels=labels, columns=4).save(target)
            page.append(
                f'<section><h3>{html.escape(scene["title"])}</h3><a href="{target.relative_to(root)}"><img loading="lazy" src="{target.relative_to(root)}"></a></section>'
            )
    page.append("</body></html>")
    (root / "index.html").write_text("\n".join(page))
    atomic_json(
        root / "DELIVERY.json",
        dict(
            report_sha256=sha256(root / "REPORT.md"),
            summary_sha256=sha256(root / "summary.json"),
            index_sha256=sha256(root / "index.html"),
            records=len(r["records"]),
            encoded_runs=len(m["runs"]),
            report_generator_sha256=sha256(Path(__file__)),
            completed=time.time(),
        ),
    )
    print(json.dumps(winners, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "evaluate", "report"))
    p.add_argument(
        "--prepared",
        type=Path,
        default=Path(
            "benchmarks/super_resolution/artifacts/bitrate-poc-600kbps-v1/inputs"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/super_resolution/artifacts/base-sweep-v1"),
    )
    p.add_argument("--device", default="mps")
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument(
        "--primary-only",
        action="store_true",
        help="Prepare only the 15 420/30 base-resolution and bitrate configurations.",
    )
    args = p.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    dict(prepare=prepare, evaluate=evaluate, report=report)[args.action](args)


if __name__ == "__main__":
    main()
