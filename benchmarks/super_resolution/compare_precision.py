"""Compare FP16/FP32 results only on matching model/image pairs and weights."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(folder):
    if (folder/"VALIDATION_FAILED.json").exists():raise RuntimeError(f"Run failed numerical validation: {folder}")
    if not (folder/"COMPLETED.json").exists():raise RuntimeError(f"Incomplete run: {folder}")
    result={"rows":list(csv.DictReader((folder/"per_frame.csv").open())),
            "environment":json.loads((folder/"environment.json").read_text()),
            "status":json.loads((folder/"status.json").read_text()),
            "timing":json.loads((folder/"timing.json").read_text())}
    marker=json.loads((folder/"COMPLETED.json").read_text())
    if len(result["rows"])!=marker["rows"]:raise RuntimeError(f"Row count differs from completion marker: {folder}")
    keys=[(r["model"],r["sample"]) for r in result["rows"]]
    if len(set(keys))!=len(keys):raise RuntimeError(f"Duplicate model/sample rows: {folder}")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fp32",type=Path,required=True);p.add_argument("--fp16",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--allow-subset",action="store_true",help="Only for explicitly labelled smoke comparisons")
    args=p.parse_args()
    a,b=load(args.fp32),load(args.fp16)
    assert b["environment"]["precision"] in ["fp16","amp-fp16"]
    assert a["environment"]["gpu"]==b["environment"]["gpu"]
    assert a["environment"]["torch"]==b["environment"]["torch"]
    assert a["environment"]["cuda"]==b["environment"]["cuda"]
    precision=b["environment"]["precision"]
    bound_sources=all(a["environment"].get(k) and b["environment"].get(k)
                      for k in ["runner_sha256","adapter_sha256"])
    if bound_sources:
        for key in ["runner_sha256","adapter_sha256"]:
            assert a["environment"][key]==b["environment"][key], f"Different {key} in matched runs"
    lines=[f"# CUDA {precision} versus FP32: matched-input comparison","",
           "## Material Passport", "",
           "- Verification Status: ANALYZED (matched completed runs; no independent repeat)",
           "- Version Label: matched_precision_v1", "",
           f"FP32 run: `{args.fp32}`; FP16 run: `{args.fp16}`.",
           f"Original architecture and checkpoint hashes match. Differences below are {precision} minus FP32.",
           "Manifest, GPU, Torch/CUDA versions and source/checkpoint hashes match. Warm timing uses the same first input geometry.","",
           "| Model | Matched images | Mean ΔPSNR dB | Worst ΔPSNR dB | Mean ΔSSIM | FP32 p50 ms | AMP p50 ms | AMP p95 ms |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    paired_rows=[]
    statistics={}
    for model in ["edsr","imdn","safmn"]:
        for key in ["source_sha","checkpoint_sha256"]:
            assert a["status"][model][key]==b["status"][model][key]
        first={r["sample"]:r for r in a["rows"] if r["model"]==model}
        second=[r for r in b["rows"] if r["model"]==model]
        assert len(second)==len({r["sample"] for r in second})
        assert all(r["sample"] in first for r in second)
        if not args.allow_subset:
            assert {r["sample"] for r in second}==set(first), f"Incomplete matched coverage for {model}"
        assert a["timing"][model]["input_wh"]==b["timing"][model]["input_wh"]
        for key in ["warmup","repeats"]:
            assert a["timing"][model][key]==b["timing"][model][key], f"Different timing {key}"
        # For complete runs the manifest hash must also match. Smoke runs bind
        # a prefix of the identical manifest via its hash plus sample coverage.
        assert a["environment"]["manifest_sha256"]==b["environment"]["manifest_sha256"]
        differences=[]
        for r in second:
            base=first[r["sample"]]
            assert r["capture"]==base["capture"] and r["degradation"]==base["degradation"]
            row={"model":model,"sample":r["sample"],"capture":r["capture"],"degradation":r["degradation"]}
            for metric in ["psnr","ssim"]:
                first_value,second_value=float(base[metric]),float(r[metric])
                assert np.isfinite(first_value) and np.isfinite(second_value), "Cannot compare nonfinite quality scores"
                row[f"fp32_{metric}"]=first_value;row[f"amp_{metric}"]=second_value
                row[f"delta_{metric}"]=second_value-first_value
            paired_rows.append(row);differences.append(row)
        dp=float(np.mean([r["delta_psnr"] for r in differences]))
        ds=float(np.mean([r["delta_ssim"] for r in differences]))
        worst=min(differences,key=lambda r:r["delta_psnr"])
        statistics[model]={"n":len(second),"mean_delta_psnr":dp,"mean_delta_ssim":ds,
                           "worst":worst,"max_abs_delta_ssim":float(max(abs(r["delta_ssim"]) for r in differences))}
        t32=a["timing"][model]["total_ms"]["p50"];t16=b["timing"][model]["total_ms"]
        lines.append(f"| {model} | {len(second)} | {dp:+.5f} | {worst['delta_psnr']:+.5f} | {ds:+.6f} | {t32:.2f} | {t16['p50']:.2f} | {t16['p95']:.2f} |")
    lines += ["","Total processing includes preparation, synchronized transfers, forward inference and output quantization.",
              "This excludes canvas assembly, native rendering and input latency.",
              ("Exact recorded runner and adapter hashes match in these two runs. Both include the finite-output guard."
               if bound_sources else
               "The first FP32 pilot predates the per-output finite-value guard and runner/adapter hash recording; AMP includes that extra check."),
              "This is a measured prototype-path comparison, not a pure kernel speedup.",
              "These H100 times do not establish live receiver FPS.","",
              "## Per-degradation mean PSNR difference", "",
              "| Model | clean | hevc18 | hevc28 | hevc38 |", "|---|---:|---:|---:|---:|"]
    for model in statistics:
        values=[]
        for degradation in ["clean","hevc18","hevc28","hevc38"]:
            selected=[r["delta_psnr"] for r in paired_rows if r["model"]==model and r["degradation"]==degradation]
            values.append(f"{np.mean(selected):+.5f}" if selected else "—")
        lines.append("| "+model+" | "+" | ".join(values)+" |")
    lines += ["", "Worst individual sample per model:", ""]
    for model,stats in statistics.items():
        lines.append(f"- {model}: `{stats['worst']['sample']}`, ΔPSNR {stats['worst']['delta_psnr']:+.5f} dB; maximum absolute ΔSSIM {stats['max_abs_delta_ssim']:.6f}.")
    lines += ["", "Smoke subsets are diagnostic only. Full matched coverage is required unless --allow-subset is explicit.",
              "All current clips share one capture session; no statistical significance or independent-session generalization is claimed.", ""]
    with args.output.with_suffix(".csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(paired_rows[0]));writer.writeheader();writer.writerows(paired_rows)
    args.output.with_suffix(".json").write_text(json.dumps(statistics,indent=2))
    args.output.write_text("\n".join(lines));print(f"Wrote {args.output}")


if __name__=="__main__":main()
