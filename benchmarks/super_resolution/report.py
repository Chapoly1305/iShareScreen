"""Audit a finished run and publish a numerical report from measured artifacts."""
import argparse
import csv
import json
import math
from pathlib import Path

from isharescreen.enhancement.models import sha256


def audit_aggregates(rows, summary):
    """Recompute aggregate claims from raw measurements, not only markers."""
    quality_metrics=["psnr","ssim","psnr_border2","ssim_border2","lpips"]
    by_model={}
    for row in rows:
        by_model.setdefault(row["model"],[]).append(row)
        stages=[float(row[k]) for k in ["prepare_ms","upload_ms","inference_ms","download_ms"]]
        assert all(math.isfinite(v) and v>=0 for v in stages), "Invalid stage timing"
        total=float(row["total_ms"])
        assert math.isfinite(total) and math.isclose(total,sum(stages),abs_tol=1e-6), "Stage timings do not sum to total"
    assert set(by_model)=={m for m,v in summary.items() if v}, "Summary/model coverage differs from raw rows"
    bicubic={r["sample"]:r for r in by_model.get("bicubic",[])}
    for model, model_rows in by_model.items():
        degradations={r["degradation"] for r in model_rows}
        assert set(summary[model])==degradations, "Summary degradation coverage differs"
        for degradation in degradations:
            selected=[r for r in model_rows if r["degradation"]==degradation]
            captures={r["capture"] for r in selected}
            reported=summary[model][degradation]
            assert reported["n_frames"]==len(selected) and reported["n_captures"]==len(captures), "Summary counts differ"
            for metric in quality_metrics:
                if not selected[0].get(metric):continue
                values=[float(r[metric]) for r in selected]
                assert all(math.isfinite(v) for v in values), f"Nonfinite {metric} in raw measurements"
                capture_means=[sum(float(r[metric]) for r in selected if r["capture"]==c)/sum(r["capture"]==c for r in selected) for c in captures]
                actual=sum(capture_means)/len(capture_means)
                assert math.isclose(reported[metric],actual,abs_tol=1e-9), f"Stale or incorrect {model}/{degradation}/{metric} summary"
            if bicubic:
                for metric in ["psnr","ssim"]:
                    differences=[sum(float(r[metric])-float(bicubic[r["sample"]][metric]) for r in selected if r["capture"]==c)/sum(r["capture"]==c for r in selected) for c in captures]
                    actual=sum(differences)/len(differences)
                    assert math.isclose(reported[f"delta_{metric}_vs_bicubic"],actual,abs_tol=1e-9), "Paired delta differs from raw measurements"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--pairs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if (args.run/"VALIDATION_FAILED.json").exists():raise RuntimeError("Run failed numerical validation")
    if not (args.run/"COMPLETED.json").is_file():raise RuntimeError("Run is not completed")
    completed=json.loads((args.run/"COMPLETED.json").read_text())
    status=json.loads((args.run/"status.json").read_text())
    env=json.loads((args.run/"environment.json").read_text())
    summary=json.loads((args.run/"summary.json").read_text())
    timing=json.loads((args.run/"timing.json").read_text())
    pairs=json.loads(args.pairs.read_text())
    rows=list(csv.DictReader((args.run/"per_frame.csv").open()))
    expected={p["id"] for p in pairs}
    assert len(expected)==len(pairs), "Duplicate input IDs"
    definitions={p["id"]:p for p in pairs}
    assert env["manifest_sha256"]==sha256(args.pairs),"Manifest differs from executed run"
    verified_inputs={}
    for pair in pairs:
        for role in ["lr","hr"]:
            path=args.pairs.parent/pair[role]
            if f"{role}_sha256" in pair:
                if path not in verified_inputs:verified_inputs[path]=sha256(path)
                assert verified_inputs[path]==pair[f"{role}_sha256"], f"Input changed: {path}"
    for model,s in status.items():
        if s["status"]!="completed":continue
        actual=[r["sample"] for r in rows if r["model"]==model]
        assert len(actual)==len(set(actual))==len(expected) and set(actual)==expected, f"Incomplete or duplicate {model} rows"
    assert completed["rows"]==len(rows)
    assert {r["model"] for r in rows}=={m for m,s in status.items() if s["status"]=="completed"}
    for row in rows:
        definition=definitions[row["sample"]]
        assert row["capture"]==definition["capture"] and row["degradation"]==definition["degradation"], "Raw row is mislabelled against input manifest"
    audit_aggregates(rows,summary)
    lock=json.loads((Path(__file__).parent/"models.lock.json").read_text())
    for model in ["edsr","imdn","safmn"]:
        if model in status and status[model]["status"]=="completed":
            assert status[model]["checkpoint_sha256"]==lock[model]["sha256"]
            assert status[model]["source_sha"]==lock[model]["revision"]
    models=[m for m in summary if summary[m]]
    degenerations=sorted({p["degradation"] for p in pairs})
    captures={p["capture"] for p in pairs}
    unique_references={sha256(args.pairs.parent/p["hr"]) for p in pairs}
    color_protocols={p.get("hevc_color","ffmpeg-default") for p in pairs}
    assert len(color_protocols)==1, "Cannot pool color protocols"
    color_protocol=next(iter(color_protocols))
    lines=["# Original pretrained ×2 evaluation: measured results", "",
           "## Material Passport", "",
           "- Verification Status: ANALYZED (completed artifacts audited; no independent repeat yet)",
           "- Version Label: pretrained_single_frame_v1", "",
           "## Coverage", "",
           f"Completed {len(rows)} model–image measurements: {len(models)} methods × {len(pairs)} aligned pairs.",
           f"The inputs represent {len(captures)} capture clips and {len(degenerations)} degradations.",
           f"There are {len(unique_references)} distinct extracted reference PNGs by SHA256.",
           f"Hardware: {env['gpu']}; Torch {env['torch']}, CUDA {env['cuda']}. Neural models use original checkpoints",
           f"and {env.get('precision',env.get('dtype','float32'))} inference; no fine-tuning was performed.", "",
           "Results use decoded recording references. Clips may share capture sessions and",
           "include static scenes; image-pair counts do not establish independent temporal evidence.",
           f"Synthetic intra HEVC 4:4:4 color protocol: {color_protocol}.",
           "Different color protocols and frame-extraction versions must be reported separately.", "",
           "## RGB quality", "",
           "Each cell is capture-mean RGB PSNR in dB / SSIM, full frame; higher is better.", "",
           "| Model | "+" | ".join(degenerations)+" |",
           "|---|"+"---:|"*len(degenerations)]
    for model in models:
        lines.append("| "+model+" | "+" | ".join(f"{summary[model][d]['psnr']:.3f} / {summary[model][d]['ssim']:.4f}" for d in degenerations)+" |")
    if "bicubic" in models:
        lines += ["", "Paired PSNR improvement versus bicubic (dB):", "",
                  "| Model | "+" | ".join(degenerations)+" |", "|---|"+"---:|"*len(degenerations)]
        for model in [m for m in models if m in ["edsr","imdn","safmn"]]:
            lines.append("| "+model+" | "+" | ".join(f"{summary[model][d]['delta_psnr_vs_bicubic']:+.3f}" for d in degenerations)+" |")
        neural=[m for m in models if m in ["edsr","imdn","safmn"]]
        if neural:
            lines += ["", "Observed pilot trade-offs:", ""]
            for deg in degenerations:
                best=max(neural,key=lambda m:summary[m][deg]["psnr"])
                delta=summary[best][deg]["delta_psnr_vs_bicubic"]
                lines.append(f"- {deg}: {best} has the highest neural mean RGB PSNR, {summary[best][deg]['psnr']:.3f} dB ({delta:+.3f} dB versus bicubic).")
            lines += ["", "Compression materially changes the reconstruction gains. This ranking describes",
                      "the current capture sample and RGB metric, not independent-session generalization."]
    review={"content": {p["capture"]: p.get("content", "unreviewed") for p in pairs}}
    content_types=sorted({review["content"].get(c,"unreviewed") for c in captures})
    lines += ["", "## Content groups", "",
              "Content labels are supplied by the input manifest. Each cell is capture-mean",
              "RGB PSNR at CRF 28; no independent-session or significance inference is made.", "",
              "| Content | Capture clips | "+" | ".join(models)+" |",
              "|---|---:|"+"---:|"*len(models)]
    for content in content_types:
        capture_ids={c for c in captures if review["content"].get(c,"unreviewed")==content}
        values=[]
        for model in models:
            per_capture=[]
            for capture in capture_ids:
                group=[float(r["psnr"]) for r in rows if r["model"]==model and r["capture"]==capture and r["degradation"]=="hevc28"]
                if group:per_capture.append(sum(group)/len(group))
            values.append(f"{sum(per_capture)/len(per_capture):.3f}" if per_capture else "—")
        lines.append(f"| {content} | {len(capture_ids)} | "+" | ".join(values)+" |")
    if "lpips" in summary[models[0]][degenerations[0]]:
        lines += ["", "LPIPS/AlexNet (lower is better):", "",
                  "| Model | "+" | ".join(degenerations)+" |", "|---|"+"---:|"*len(degenerations)]
        for model in models:
            lines.append("| "+model+" | "+" | ".join(f"{summary[model][d]['lpips']:.4f}" for d in degenerations)+" |")
    lines += ["", "## Warm single-frame processing", "",
              "960×540 input → 1920×1080 output. Synchronized measurements include preparation,",
              "transfers and quantization. They exclude disk I/O, canvas assembly, native rendering",
              "and interaction. Conventional baselines run on CPU OpenCV; their timing does not",
              "measure the native wgpu scaling path. Peak CUDA allocation includes resident LPIPS.", "",
              "| Model | Forward p50 ms | Upload p50 ms | Download/quantization p50 ms | Total p50 ms | Total p95 ms | Peak CUDA MiB |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model in models:
        t=timing[model]
        peak="—" if t["peak_cuda_mib"] is None else f"{t['peak_cuda_mib']:.1f}"
        lines.append(f"| {model} | {t['inference_ms']['p50']:.2f} | {t['upload_ms']['p50']:.2f} | {t['download_ms']['p50']:.2f} | {t['total_ms']['p50']:.2f} | {t['total_ms']['p95']:.2f} | {peak} |")
    lines += ["", "Stage medians need not add exactly to the total median. See timing.json for warmup,",
              "repeat count, first-call time and every stage. Offline measurements do not establish",
              "performance on other GPUs, sustained client FPS or input-to-display responsiveness."]
    lines += ["", "## Model availability and client evidence", ""]
    for model,s in status.items():
        if s["status"]=="unavailable":lines.append(f"- {model}: unavailable — {s['reason']}.")
    lines += ["- This report audits saved offline results; it does not certify viewer integration or live transport.",
              "- Run the adapter, control-path and opt-in GPU readback checks separately when changing runtime code.", "",
              "## Artifacts and audit", "",
              f"- Run: `{args.run}`", f"- Input manifest: `{args.pairs}`",
              "- Per-frame PSNR/SSIM, border-cropped variants, LPIPS and timing: `per_frame.csv`.",
              "- Capture means and paired deltas: `summary.json`; timing: `timing.json`.",
              "- Environment, exact command, package freeze and checkpoint/source hashes: run JSON/TXT artifacts.",
              "- Full-size outputs and comparison crops: `images/`; overview: `quality_latency.png`.",
              "- Audit checked completion marker, exact model/sample coverage, duplicate rows, manifest hash, official checkpoint hashes and source revisions.",
              "- Capture-weighted quality means, paired deltas and timing-stage sums were recomputed from raw measurements.",
              "- Input PNG hashes were rechecked where supplied by the manifest; older manifests did not record them.",
              "- No statistical significance or independent-session generalization is claimed.", ""]
    args.output.write_text("\n".join(lines))
    print(f"AUDITED {len(rows)} rows; wrote {args.output}")


if __name__=="__main__":main()
