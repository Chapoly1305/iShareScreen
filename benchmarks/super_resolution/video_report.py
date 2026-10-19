"""Audit completed video reconstruction and export viewable original/model pairs."""
import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from benchmarks.super_resolution.prepare_video import probe
from isharescreen.enhancement.models import sha256


def audit(run,manifest_path):
    manifest=json.loads(manifest_path.read_text())
    done=json.loads((run/"COMPLETED.json").read_text())
    env=json.loads((run/"environment.json").read_text())
    results=json.loads((run/"clips.json").read_text())
    rows=list(csv.DictReader((run/"quality.csv").open()))
    assert sha256(manifest_path)==env["manifest_sha256"],"Manifest changed"
    definitions={c["id"]:c for c in manifest["clips"]}
    models={m for m,s in done["status"].items() if s["state"]=="completed"}
    keys=[(r["clip"],r["model"]) for r in results]
    assert len(keys)==len(set(keys)) and set(keys)=={(c,m) for c in definitions for m in models},"Missing/duplicate complete videos"
    assert done["videos"]==len(results) and done["frames_written"]==sum(r["frames"] for r in results)
    assert done["quality_rows"]==len(rows)
    lock=json.loads((Path(__file__).parent/"models.lock.json").read_text())
    for model in models & set(lock):
        assert done["status"][model]["checkpoint_sha256"]==lock[model]["sha256"]
        assert done["status"][model]["source_sha"]==lock[model]["revision"]
    for clip in definitions.values():
        for role in ["reference","low"]:
            assert sha256(manifest_path.parent/clip[role])==clip[f"{role}_sha256"],"Input video changed"
    all_raw=[]
    for result in results:
        clip=definitions[result["clip"]]
        assert result["frames"]==clip["frames"] and result["lossless_roundtrip_verified"]
        assert result["inference_calls"]+result["reused_frames"]==clip["frames"]
        if not env["reuse_identical"]:assert result["reused_frames"]==0
        assert sha256(run/result["output"])==result["sha256"],"Output video changed"
        raw=[r for r in rows if (r["clip"],r["model"])==(result["clip"],result["model"])]
        indices=list(range(0,clip["frames"],env["metrics_every"]))
        if clip["frames"]-1 not in indices:indices.append(clip["frames"]-1)
        assert [int(r["frame"]) for r in raw]==indices,"Missing/duplicate quality samples"
        assert result["sampled_quality_frames"]==len(raw)
        for row in raw:
            assert math.isclose(float(row["time_s"]),float(Fraction(int(row["frame"]),1)/Fraction(clip["fps"])),abs_tol=1e-9)
        for metric in ["psnr","ssim"]:
            assert math.isclose(result[metric],float(np.mean([float(r[metric]) for r in raw])),abs_tol=1e-9),"Stale quality mean"
        all_raw.extend(raw)
    assert len(all_raw)==len(rows)
    return manifest,done,env,results


def paired_quality(manifest, results):
    """Match sampled clip means by ID, never by row order or frame count weight."""
    definitions={clip["id"]:clip for clip in manifest["clips"]}
    baseline={r["clip"]:r for r in results if r["model"]=="bicubic"}
    if not baseline:return []
    assert set(baseline)==set(definitions),"Incomplete bicubic paired baseline"
    return [dict(clip=r["clip"],model=r["model"],
                 content=definitions[r["clip"]].get("content") or "unlabelled",
                 psnr_delta_db=r["psnr"]-baseline[r["clip"]]["psnr"],
                 ssim_delta=r["ssim"]-baseline[r["clip"]]["ssim"])
            for r in results if r["model"]!="bicubic"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--comparisons",type=Path,help="Fresh directory for 3840x1080 H264 comparison MP4s")
    p.add_argument("--comparison-clips",nargs="+")
    args=p.parse_args()
    manifest,done,env,results=audit(args.run,args.manifest)
    comparisons=[]
    if args.comparisons:
        args.comparisons.mkdir(parents=True,exist_ok=False)
        for result in results:
            if args.comparison_clips and result["clip"] not in args.comparison_clips:continue
            if result["model"]=="bicubic":continue
            clip=next(c for c in manifest["clips"] if c["id"]==result["clip"])
            output=args.comparisons/f"{clip['id']}--original-vs-{result['model']}.mp4"
            command=["ffmpeg","-hide_banner","-loglevel","error","-nostdin",
                     "-i",str(args.manifest.parent/clip["reference"]),"-i",str(args.run/result["output"]),
                     "-filter_complex","[0:v][1:v]hstack=inputs=2,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[v]",
                     "-map","[v]","-an","-c:v","libx264","-preset","fast","-crf","16","-threads","4",
                     "-color_range","tv","-colorspace","bt709","-color_primaries","bt709","-color_trc","bt709",
                     "-fps_mode","passthrough","-movflags","+faststart",str(output)]
            subprocess.run(command,check=True,capture_output=True)
            stream=probe(output)
            assert int(stream["nb_read_frames"])==clip["frames"] and Fraction(stream["avg_frame_rate"])==Fraction(clip["fps"])
            comparisons.append({"file":str(output),"sha256":sha256(output),"frames":clip["frames"],"command":command})
            print(f"COMPARISON {output}",flush=True)
        (args.comparisons/"manifest.json").write_text(json.dumps(comparisons,indent=2))
    models=[m for m,s in done["status"].items() if s["state"]=="completed"]
    lines=["# Complete recorded-video pretrained comparison", "", "## Material Passport", "",
           "- Verification Status: ANALYZED (complete video outputs and sampled metrics audited)",
           "- Version Label: recorded_video_full709_v1", "",
           f"Input clips: {len(manifest['clips'])}; source frames: {sum(c['frames'] for c in manifest['clips'])}.",
           f"All clips are complete 1920×1080 recordings at native frame rates. LR input is continuous",
           "960×540 HEVC RExt 4:4:4 CRF 28 with inter-frame prediction, full-range BT.709, Lanczos reduction.",
           f"Inference: original pretrained weights, {env['precision']}, {env['gpu']}; no fine-tuning.",
           f"Every frame is reconstructed; PSNR/SSIM are sampled every {env['metrics_every']} frames and at the final frame.",
           "References are RGB8 decoded proxies of original recordings, not lossless acquisition.", "",
           "| Model | Complete videos | Written frames | Quality samples | Mean clip PSNR dB | Mean clip SSIM | Reused exact frames |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for model in models:
        selected=[r for r in results if r["model"]==model]
        lines.append(f"| {model} | {len(selected)} | {sum(r['frames'] for r in selected)} | {sum(r['sampled_quality_frames'] for r in selected)} | {np.mean([r['psnr'] for r in selected]):.3f} | {np.mean([r['ssim'] for r in selected]):.5f} | {sum(r['reused_frames'] for r in selected)} |")
    paired=paired_quality(manifest,results)
    if paired:
        paired_path=args.output.with_suffix(".paired.csv")
        with paired_path.open("w",newline="") as output:
            writer=csv.DictWriter(output,fieldnames=list(paired[0]))
            writer.writeheader();writer.writerows(paired)
        lines += ["", "## Paired gain versus bicubic", "",
                  "Differences match the same clip and quality-sample indices. Each clip has equal",
                  "weight, regardless of duration. Positive means higher reconstruction quality,",
                  "not necessarily better text legibility. These are descriptive, not independent-session",
                  "or significance results. Min/max expose clips hidden by an overall average.", "",
                  "| Model | Clips | Mean PSNR gain dB | Min / max gain dB | Clips with PSNR loss | Mean SSIM gain |",
                  "|---|---:|---:|---:|---:|---:|"]
        for model in models:
            selected=[r for r in paired if r["model"]==model]
            if not selected:continue
            gains=[r["psnr_delta_db"] for r in selected]
            lines.append(f"| {model} | {len(selected)} | {np.mean(gains):+.3f} | {min(gains):+.3f} / {max(gains):+.3f} | {sum(v<0 for v in gains)} | {np.mean([r['ssim_delta'] for r in selected]):+.5f} |")
        lines += ["", "Provisional content groups (labels describe capture content, not separate sessions):", "",
                  "| Content | Model | Clips | Mean paired PSNR gain dB | Mean paired SSIM gain |",
                  "|---|---|---:|---:|---:|"]
        for content in sorted({r["content"] for r in paired}):
            for model in models:
                selected=[r for r in paired if r["model"]==model and r["content"]==content]
                if not selected:continue
                lines.append(f"| {content} | {model} | {len(selected)} | {np.mean([r['psnr_delta_db'] for r in selected]):+.3f} | {np.mean([r['ssim_delta'] for r in selected]):+.5f} |")
        lines += ["", f"Per-clip paired differences: `{paired_path}`."]
    elif any(model!="bicubic" for model in models):
        lines += ["", "Paired gains unavailable: this run contains no complete bicubic baseline."]
    lines += ["", "Each clip has equal weight in these descriptive means. Adjacent frames and clips from",
              "one capture session are not independent test sessions. Earlier single-frame protocols remain separate.",
              "Exact consecutive-input reuse is an offline optimization: every output frame is still written,",
              "but cache reuse and offline elapsed time must not be reported as live model throughput or FPS.",
              "Every saved lossless reconstruction was decoded in full and its RGB sequence hash matched the",
              "pre-encode model output. The audit also checked file hashes, frame/sample coverage, timestamps,",
              "checkpoint/source identity and raw-to-summary quality means.", "", "## Viewing", "",
              "Comparison MP4s: LEFT = original 1080p reference; RIGHT = model-upscaled 1080p.",
              "Both panels retain full resolution (3840×1080 combined); use a player at 1:1/zoom for small text.",
              "These H264/YUV420 CRF16 files are convenient viewing copies, not the images used for metrics.",
              "The lossless RGB videos and raw sampled metrics remain available for exact inspection.", ""]
    for item in comparisons:lines.append(f"- `{item['file']}`")
    for model,state in done["status"].items():
        if state["state"]=="unavailable":lines.append(f"- {model}: unavailable — {state['reason']}.")
    lines += ["", f"Run: `{args.run}`", f"Manifest: `{args.manifest}`",
              "Metrics: quality.csv; per-clip counts, timings and output hashes: clips.json.",
              "Offline results do not establish live Screen Sharing or input-to-display latency.", ""]
    args.output.write_text("\n".join(lines))
    print(f"AUDITED {len(results)} complete videos; wrote {args.output}",flush=True)


if __name__=="__main__":main()
