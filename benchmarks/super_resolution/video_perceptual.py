"""Add full-resolution LPIPS to an audited completed video run, without rerunning SR."""
import argparse
from collections import OrderedDict
import csv
from fractions import Fraction
import hashlib
import importlib.metadata
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import av
import numpy as np
import torch

from benchmarks.super_resolution.benchmark import metrics
from benchmarks.super_resolution.video_report import audit
from isharescreen.enhancement.models import sha256

BACKBONE_SHA256="7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
LINEAR_SHA256="df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"


class ReferenceCache:
    """Bounded RGB sample cache; decoding validation is never skipped on a cache miss."""
    def __init__(self,limit):
        self.limit=limit;self.bytes=0;self.values=OrderedDict()

    def get(self,key,load):
        if key in self.values:
            self.values.move_to_end(key);return self.values[key][0]
        value=load();size=sum(rgb.nbytes for _,_,rgb in value)
        if size<=self.limit:
            while self.bytes+size>self.limit and self.values:
                _,(_,removed)=self.values.popitem(last=False);self.bytes-=removed
            self.values[key]=(value,size);self.bytes+=size
        return value


def source_snapshot(run, manifest_path, follow_pid=None):
    """Only published, verified clips are eligible; completion still needs full audit."""
    if (run/"COMPLETED.json").exists():
        for attempt in range(5):
            try:
                manifest,done,producer,results=audit(run,manifest_path);break
            except json.JSONDecodeError:
                if attempt==4:raise
                time.sleep(.05)
    else:
        if follow_pid is None:raise ValueError("Source is incomplete; --follow-pid is required")
        producer=json.loads((run/"environment.json").read_text())
        if producer["pid"]!=follow_pid:raise ValueError("Producer PID does not match source provenance")
        try:os.kill(follow_pid,0)
        except ProcessLookupError:
            if (run/"COMPLETED.json").exists():return source_snapshot(run,manifest_path)
            raise RuntimeError("Producer exited without completing; no automatic retry")
        cmdline=Path(f"/proc/{follow_pid}/cmdline")
        if cmdline.exists() and "command" in producer:
            command=producer["command"];output_arg=command[command.index("--output")+1]
            actual=cmdline.read_bytes().decode().split("\0")
            if "benchmarks.super_resolution.video_benchmark" not in actual or output_arg not in actual:
                raise ValueError("Producer PID changed ownership")
        manifest=json.loads(manifest_path.read_text());done=None
        if sha256(manifest_path)!=producer["manifest_sha256"]:raise ValueError("Manifest changed")
        # clips.json is a published list, but its non-atomic rewrite can be observed mid-write.
        for attempt in range(5):
            try:
                results=json.loads((run/"clips.json").read_text()) if (run/"clips.json").exists() else [];break
            except json.JSONDecodeError:
                if attempt==4:raise
                time.sleep(.05)
    expected={}
    keys=[(r["clip"],r["model"]) for r in results]
    if len(keys)!=len(set(keys)):raise ValueError("Duplicate completed clip")
    ready=set(keys);definitions={c["id"]:c for c in manifest["clips"]}
    raw_rows=csv.DictReader((run/"quality.csv").open()) if (run/"quality.csv").exists() else []
    for row in raw_rows:
        if (row["clip"],row["model"]) not in ready:continue
        key=(row["clip"],row["model"],int(row["frame"]))
        if key in expected:raise ValueError("Duplicate producer quality sample")
        expected[key]=row
    for result in results:
        clip=definitions[result["clip"]]
        if (result["frames"]!=clip["frames"] or not result["lossless_roundtrip_verified"]
                or result["inference_calls"]+result["reused_frames"]!=clip["frames"]):
            raise ValueError("Source clip has not passed complete reconstruction checks")
        indices=[k[2] for k in expected if k[:2]==(result["clip"],result["model"])]
        if indices!=sample_indices(clip["frames"],producer["metrics_every"]):
            raise ValueError("Incomplete published quality samples")
    return manifest,done,producer,results,expected


def pending_results(results, processed):
    for result in results:
        key=(result["clip"],result["model"])
        if key in processed:
            if processed[key]!=result:raise ValueError("Previously scored source clip changed")
        else:yield result


def sample_indices(count, every):
    if count<1 or every<1:raise ValueError("Frame count and sampling interval must be positive")
    indices=list(range(0,count,every))
    if indices[-1]!=count-1:indices.append(count-1)
    return indices


def sampled_rgb(path, count, fps, every, size=(1920,1080)):
    """Decode every frame, validate timing/count/geometry, convert only selected frames."""
    wanted=set(sample_indices(count,every));decoded=0
    with av.open(str(path)) as video:
        video.streams.video[0].codec_context.thread_count=2
        for i,frame in enumerate(video.decode(video=0)):
            if i>=count:raise ValueError("More decoded frames than expected")
            target=float(Fraction(i,1)/fps)
            if frame.pts is None or abs(float(frame.pts*frame.time_base)-target)>max(.0011,float(frame.time_base)*1.1):
                raise ValueError(f"Timestamp misalignment at frame {i}")
            if (frame.width,frame.height)!=size:raise ValueError("Video geometry changed")
            decoded+=1
            if i in wanted:yield i,target,frame.to_ndarray(format="rgb24")
        if decoded!=count:raise ValueError(f"Expected {count} frames, decoded {decoded}")


def normalized_tensor(rgb, device):
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError("LPIPS requires RGB uint8 HWC input")
    return torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).float().to(device)/127.5-1


def tensor_rgb_metrics(reference, output, device):
    """FP64 PSNR and skimage-default SSIM: valid 7x7 box, sample covariance, RGB mean."""
    if (reference.shape!=output.shape or reference.dtype!=np.uint8 or output.dtype!=np.uint8
            or reference.ndim!=3 or reference.shape[2]!=3 or min(reference.shape[:2])<7):
        raise ValueError("Metrics require equal RGB uint8 images at least 7x7")
    import torch.nn.functional as F
    def tensor(rgb):
        return torch.from_numpy(np.ascontiguousarray(rgb)).to(device=device,dtype=torch.float64).permute(2,0,1).unsqueeze(0).contiguous()
    with torch.inference_mode():
        x,y=tensor(reference),tensor(output)
        mse=(x-y).square().mean()
        psnr=10*torch.log10(65025/mse)
        ux,uy=F.avg_pool2d(x,7,stride=1),F.avg_pool2d(y,7,stride=1)
        vx=(F.avg_pool2d(x*x,7,stride=1)-ux*ux)*(49/48)
        vy=(F.avg_pool2d(y*y,7,stride=1)-uy*uy)*(49/48)
        vxy=(F.avg_pool2d(x*y,7,stride=1)-ux*uy)*(49/48)
        ssim=((2*ux*uy+6.5025)*(2*vxy+58.5225)/((ux.square()+uy.square()+6.5025)*(vx+vy+58.5225))).mean()
        return dict(psnr=float(psnr.item()),ssim=float(ssim.item()))


def resume_scores(path, environment, expected):
    previous=json.loads((path/"environment.json").read_text())
    for field in ["manifest_sha256","producer_environment_sha256","run","state_sha256",
                  "backbone_sha256","linear_sha256","lpips_version","network","metric_version",
                  "torch","torchvision","device"]:
        if previous[field]!=environment[field]:raise ValueError(f"Resume provenance changed: {field}")
    cached={}
    for row in csv.DictReader((path/"quality.csv").open()):
        key=(row["clip"],row["model"],int(row["frame"]))
        if key in cached or key not in expected:raise ValueError("Invalid resumed sample coverage")
        for field in ["psnr","ssim","time_s"]:
            if not math.isclose(float(row[field]),float(expected[key][field]),abs_tol=1e-9,rel_tol=1e-10):
                raise ValueError("Resumed sample differs from producer")
        if not math.isfinite(float(row["lpips"])):raise ValueError("Nonfinite resumed LPIPS")
        cached[key]=row
    return cached


def summarize(rows):
    if not rows:raise ValueError("No perceptual measurements")
    keys=[(r["clip"],r["model"],r["frame"]) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError("Duplicate perceptual sample")
    baseline={(r["clip"],r["frame"]):r for r in rows if r["model"]=="bicubic"}
    if not baseline:raise ValueError("A bicubic baseline is required")
    output={}
    for model in dict.fromkeys(r["model"] for r in rows):
        selected=[r for r in rows if r["model"]==model]
        if {(r["clip"],r["frame"]) for r in selected}!=set(baseline):
            raise ValueError("Unequal paired sample coverage")
        clips=sorted({r["clip"] for r in selected})
        result=dict(clips=len(clips),samples=len(selected))
        for metric in ["psnr","ssim","lpips"]:
            if not all(math.isfinite(float(r[metric])) for r in selected):
                raise ValueError("Nonfinite perceptual measurement")
            result[metric]=float(np.mean([np.mean([float(r[metric]) for r in selected if r["clip"]==clip]) for clip in clips]))
            result[f"delta_{metric}_vs_bicubic"]=float(np.mean([
                np.mean([float(r[metric])-float(baseline[(r["clip"],r["frame"])][metric])
                         for r in selected if r["clip"]==clip]) for clip in clips]))
        output[model]=result
    return output


def audit_scores(rows, summary, expected):
    """Recheck saved raw coverage, producer metrics and clip-weighted aggregates."""
    typed=[dict(r,frame=int(r["frame"])) for r in rows]
    keys=[(r["clip"],r["model"],r["frame"]) for r in typed]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):
        raise ValueError("Incomplete or duplicate saved perceptual coverage")
    refs={}
    for row,key in zip(typed,keys):
        previous=expected[key]
        for metric in ["psnr","ssim","time_s"]:
            if not math.isclose(float(row[metric]),float(previous[metric]),abs_tol=1e-9,rel_tol=1e-10):
                raise ValueError("Saved perceptual row differs from producer metrics/timestamp")
        refkey=(row["clip"],row["frame"])
        for field in ["reference_rgb_sha256","output_rgb_sha256"]:
            value=row[field]
            if len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid sampled RGB hash")
        if refs.setdefault(refkey,row["reference_rgb_sha256"])!=row["reference_rgb_sha256"]:
            raise ValueError("Models used different sampled reference pixels")
    rebuilt=summarize(typed)
    if set(rebuilt)!=set(summary):raise ValueError("Stale perceptual summary model coverage")
    for model,values in rebuilt.items():
        for field,value in values.items():
            if not math.isclose(value,summary[model][field],abs_tol=1e-10,rel_tol=1e-10):
                raise ValueError("Stale perceptual summary aggregate")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True,help="Fresh companion directory, never the SR run directory")
    p.add_argument("--device",default="auto")
    p.add_argument("--follow-pid",type=int,help="Score already-verified clips while this producer is live; never certify a partial run")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--timeout-seconds",type=float,default=21600)
    p.add_argument("--pixel-metrics",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--resume-from",type=Path,help="Preserved partial companion; verify sample pixels before reusing LPIPS")
    p.add_argument("--reference-cache-mib",type=int,default=512,help="Bounded cache of decoded sampled reference pixels")
    args=p.parse_args()
    if args.threads<1 or args.timeout_seconds<=0 or args.reference_cache_mib<0:p.error("Invalid threads, timeout or cache limit")
    if args.output.exists():raise FileExistsError("Choose a fresh perceptual output directory")
    if args.pixel_metrics=="cuda" and not torch.cuda.is_available():raise ValueError("CUDA metrics requested without CUDA")
    started=time.monotonic()
    manifest,done,producer,results,expected_by_key=source_snapshot(args.run,args.manifest,args.follow_pid)
    import lpips
    import torchvision
    backbone=Path(torch.hub.get_dir())/"checkpoints"/Path(torchvision.models.AlexNet_Weights.IMAGENET1K_V1.url).name
    linear=Path(lpips.__file__).parent/"weights/v0.1/alex.pth"
    if not backbone.is_file():raise FileNotFoundError(f"Pre-cache the official AlexNet weights; implicit downloads disabled: {backbone}")
    if sha256(backbone)!=BACKBONE_SHA256 or sha256(linear)!=LINEAR_SHA256:
        raise ValueError("LPIPS backbone or calibrated weights changed")
    device=args.device
    if device=="auto":device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(args.threads);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    network=lpips.LPIPS(net="alex",version="0.1",pretrained=True,pnet_rand=False,verbose=False).eval()
    state=hashlib.sha256()
    for name,tensor in sorted(network.state_dict().items()):
        state.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        state.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    network=network.to(device)
    args.output.mkdir(parents=True,exist_ok=False)
    environment=dict(command=sys.argv,started=time.time(),python=sys.version,platform=platform.platform(),
        device=device,torch=torch.__version__,torchvision=torchvision.__version__,av=av.__version__,
        lpips_version=importlib.metadata.version("lpips"),network="alex",metric_version="0.1",
        backbone_sha256=BACKBONE_SHA256,linear_sha256=LINEAR_SHA256,state_sha256=state.hexdigest(),
        runner_sha256=sha256(Path(__file__)),manifest_sha256=sha256(args.manifest),
        producer_environment_sha256=sha256(args.run/"environment.json"),
        producer_results_sha256=None,producer_quality_sha256=None,
        producer=producer,model_status=None,source_results=[],producer_complete=False,pid=os.getpid(),
        pixel_metrics_backend=args.pixel_metrics,pixel_metrics_precision="float64",
        reference_cache_limit_mib=args.reference_cache_mib,
        run=str(args.run.resolve()),manifest=str(args.manifest.resolve()),
        protocol="RGB uint8 -> [-1,1]; full resolution; no resize/crop/tiling; same sample indices as producer")
    cached=resume_scores(args.resume_from,environment,expected_by_key) if args.resume_from else {}
    if args.resume_from:
        environment.update(resume_from=str(args.resume_from.resolve()),resume_samples=len(cached),
            resume_environment_sha256=sha256(args.resume_from/"environment.json"),
            resume_quality_sha256=sha256(args.resume_from/"quality.csv"))
    (args.output/"environment.json").write_text(json.dumps(environment,indent=2))
    rows=[];self_distance=None;processed={};checked_inputs=set()
    reference_cache=ReferenceCache(args.reference_cache_mib*1024**2)
    clip_order={clip["id"]:i for i,clip in enumerate(manifest["clips"])}
    fields=["clip","model","frame","time_s","psnr","ssim","lpips","reference_rgb_sha256","output_rgb_sha256"]
    try:
        with (args.output/"quality.csv").open("x",newline="") as output:
            writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
            while True:
                if time.monotonic()-started>args.timeout_seconds:raise TimeoutError("Perceptual evaluation timeout")
                new=0
                for result in sorted(pending_results(results,processed),key=lambda r:clip_order[r["clip"]]):
                    clip=next(c for c in manifest["clips"] if c["id"]==result["clip"])
                    if clip["id"] not in checked_inputs:
                        for role in ["reference","low"]:
                            if sha256(args.manifest.parent/clip[role])!=clip[f"{role}_sha256"]:
                                raise ValueError("Source input video changed")
                        checked_inputs.add(clip["id"])
                    if sha256(args.run/result["output"])!=result["sha256"]:raise ValueError("Source output video changed")
                    fps=Fraction(clip["fps"]);every=producer["metrics_every"]
                    references=reference_cache.get(clip["id"],lambda:list(sampled_rgb(
                        args.manifest.parent/clip["reference"],clip["frames"],fps,every)))
                    samples=sampled_rgb(args.run/result["output"],clip["frames"],fps,every)
                    for reference,sample in zip_longest(references,samples):
                        if time.monotonic()-started>args.timeout_seconds:raise TimeoutError("Perceptual evaluation timeout")
                        if reference is None or sample is None:raise ValueError("Unequal reference/output sample coverage")
                        index,target,hr=reference;i,t,rgb=sample
                        if (index,target)!=(i,t):raise ValueError("Reference/output sample misalignment")
                        raw=expected_by_key[(clip["id"],result["model"],i)]
                        quality=tensor_rgb_metrics(hr,rgb,"cuda") if args.pixel_metrics=="cuda" else metrics(hr,rgb)
                        for metric in ["psnr","ssim"]:
                            if not math.isclose(quality[metric],float(raw[metric]),abs_tol=1e-9,rel_tol=1e-10):
                                raise ValueError(f"Re-decoded {metric} differs from original run")
                        quality={metric:float(raw[metric]) for metric in ["psnr","ssim"]}
                        a,b=normalized_tensor(hr,device),normalized_tensor(rgb,device)
                        reference_hash=hashlib.sha256(hr.tobytes()).hexdigest()
                        output_hash=hashlib.sha256(rgb.tobytes()).hexdigest()
                        with torch.inference_mode():
                            if self_distance is None:
                                self_distance=float(network(a,a).item())
                                if abs(self_distance)>1e-8:raise ValueError("LPIPS identical-image self-check failed")
                            cached_row=cached.get((clip["id"],result["model"],i))
                            if cached_row:
                                if (cached_row["reference_rgb_sha256"]!=reference_hash
                                        or cached_row["output_rgb_sha256"]!=output_hash):
                                    raise ValueError("Resumed sample pixels changed")
                                score=float(cached_row["lpips"])
                            else:score=float(network(a,b).item())
                        if not math.isfinite(score):raise ValueError("Nonfinite LPIPS score")
                        row=dict(clip=clip["id"],model=result["model"],frame=i,time_s=t,**quality,lpips=score,
                                 reference_rgb_sha256=reference_hash,output_rgb_sha256=output_hash)
                        writer.writerow(row);output.flush();rows.append(row)
                    print(f"LPIPS {clip['id']} {result['model']} complete: {len(references)} samples",flush=True)
                    processed[(result["clip"],result["model"])]=dict(result);new+=1
                if done is not None:break
                if not new:
                    print(f"FOLLOW waiting for producer {args.follow_pid}; {len(processed)} clips scored",flush=True)
                    time.sleep(30)
                current,done,current_producer,results,expected_by_key=source_snapshot(args.run,args.manifest,args.follow_pid)
                if current!=manifest or current_producer!=producer:raise ValueError("Source environment or manifest changed")
            if sha256(args.run/"environment.json")!=environment["producer_environment_sha256"]:
                raise ValueError("Source environment file changed")
            if args.resume_from and (sha256(args.resume_from/"quality.csv")!=environment["resume_quality_sha256"]
                    or sha256(args.resume_from/"environment.json")!=environment["resume_environment_sha256"]):
                raise ValueError("Resumed companion changed during evaluation")
            environment.update(producer_results_sha256=sha256(args.run/"clips.json"),
                producer_quality_sha256=sha256(args.run/"quality.csv"),model_status=done["status"],
                source_results=results,producer_complete=True)
            (args.output/"environment.json").write_text(json.dumps(environment,indent=2))
        if {(r["clip"],r["model"],r["frame"]) for r in rows}!=set(expected_by_key):
            raise ValueError("Incomplete perceptual coverage")
        summary=summarize(rows)
        (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
        audit_scores(list(csv.DictReader((args.output/"quality.csv").open())),
                     json.loads((args.output/"summary.json").read_text()),expected_by_key)
        lines=["# Recorded-video reconstruction quality", "", "## Material Passport", "",
            "- Version Label: recorded_video_rgb_lpips_v1", "", "## Experiment Result", "",
            "- Type: analysis", "- Status: completed", f"- Command: `{' '.join(sys.argv)}`",
            f"- Device for metric computation: {device}; not the SR producer's performance device.",
            f"- Coverage: {len(manifest['clips'])} complete clips; {len(rows)} sampled model/frame measurements.",
            "- No SR model was rerun. Every output was decoded in full for count/timestamp checks.",
            "- PSNR/SSIM were recomputed and matched the original sampled metrics before LPIPS was added.",
            f"- Pixel-metric verification backend: {args.pixel_metrics}, float64; source PSNR/SSIM rows retained after matching.",
            f"- Reused partial LPIPS scores: {len(cached)}; model-state provenance and re-decoded sample pixels verified.",
            "- LPIPS/AlexNet v0.1, FP32, RGB [-1,1], native 1920x1080; no resizing or metric crops.",
            "- Backbone/calibrated weight hashes, loaded-state hash, source/output hashes and per-sample RGB hashes are recorded.",
            f"- Identical-image LPIPS self-check: {self_distance}.", "", "## Quality", "",
            "| Method | Clips | Samples | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | ΔPSNR vs bicubic | ΔLPIPS vs bicubic ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for model,s in summary.items():
            lines.append(f"| {model} | {s['clips']} | {s['samples']} | {s['psnr']:.3f} | {s['ssim']:.5f} | {s['lpips']:.5f} | {s['delta_psnr_vs_bicubic']:+.3f} | {s['delta_lpips_vs_bicubic']:+.5f} |")
        lines += ["", "Each clip has equal weight; frames within one capture session are not independent replicates.",
                  "These are sampled full-reference reconstruction scores, not a temporal metric or a claim about text correctness.",
                  "OCR CER is excluded from the main evaluation by user choice. No OCR labels or predictions are generated.",
                  "The reference is an RGB8 decoded recording proxy, not lossless screen acquisition.",
                  "Metric computation time is not SR inference latency, sustained client FPS or input-to-display latency.",
                  "Raw rows: quality.csv; equal-clip summaries and paired deltas: summary.json; provenance: environment.json.", ""]
        (args.output/"REPORT.md").write_text("\n".join(lines))
        marker=dict(rows=len(rows),clips=len(manifest["clips"]),models=list(summary),finished=time.time(),
                    quality_sha256=sha256(args.output/"quality.csv"),summary_sha256=sha256(args.output/"summary.json"),
                    report_sha256=sha256(args.output/"REPORT.md"),self_distance=self_distance)
        marker["environment_sha256"]=sha256(args.output/"environment.json")
        (args.output/"COMPLETED.json").write_text(json.dumps(marker,indent=2))
        print(f"COMPLETED perceptual video evaluation: {len(rows)} samples",flush=True)
    except Exception as exc:
        (args.output/"FAILED.json").write_text(json.dumps(dict(error=str(exc),time=time.time()),indent=2))
        raise


if __name__=="__main__":main()
