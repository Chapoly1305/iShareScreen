"""Decode complete low-resolution videos, upscale every frame, save lossless output."""
import argparse
import csv
from fractions import Fraction
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import av
import cv2
import numpy as np
import torch

from benchmarks.super_resolution.benchmark import metrics
from isharescreen.enhancement.models import Upscaler, availability, REPOS, project_home, sha256


def aligned_frames(reference, low, expected, fps):
    """No silent truncation, dropped frames, timestamp shifts or geometry changes."""
    with av.open(str(reference)) as hr, av.open(str(low)) as lr:
        count=0
        for i,(a,b) in enumerate(zip_longest(hr.decode(video=0),lr.decode(video=0))):
            if a is None or b is None:raise ValueError("Unequal decoded frame counts")
            target=float(Fraction(i,1)/fps)
            for frame in [a,b]:
                if frame.pts is None:raise ValueError("Missing frame timestamp")
                tolerance=max(.0011,float(frame.time_base)*1.1)
                if abs(float(frame.pts*frame.time_base)-target)>tolerance:
                    raise ValueError(f"Timestamp misalignment at frame {i}: {frame.time} vs {target}")
            if (a.width,a.height)!=(2*b.width,2*b.height):raise ValueError("Not an aligned x2 pair")
            yield i,a.to_ndarray(format="rgb24"),b.to_ndarray(format="rgb24")
            count+=1
        if count!=expected:raise ValueError(f"Expected {expected} frames, decoded {count}")


class LosslessWriter:
    def __init__(self,path,width,height,fps):
        self.container=av.open(str(path),"w")
        self.stream=self.container.add_stream("libx264rgb",rate=fps)
        self.stream.width=width;self.stream.height=height;self.stream.pix_fmt="rgb24"
        self.stream.codec_context.thread_count=4
        self.stream.options={"crf":"0","preset":"fast","g":"120"}
        self.fps=fps;self.count=0;self.digest=hashlib.sha256()

    def write(self,rgb):
        frame=av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb),format="rgb24")
        frame.pts=self.count;frame.time_base=1/self.fps
        self.digest.update(rgb.tobytes())
        for packet in self.stream.encode(frame):self.container.mux(packet)
        self.count+=1

    def close(self):
        try:
            for packet in self.stream.encode():self.container.mux(packet)
        finally:self.container.close()


def verify_lossless(path,expected_count,expected_digest):
    digest=hashlib.sha256();count=0
    with av.open(str(path)) as video:
        for frame in video.decode(video=0):
            digest.update(frame.to_ndarray(format="rgb24").tobytes());count+=1
    if count!=expected_count or digest.hexdigest()!=expected_digest:
        raise ValueError(f"Lossless output round-trip failed: {path}")
    return count


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--models",nargs="+",default=["bicubic","edsr","imdn","safmn","ecbsr"])
    p.add_argument("--device",default="auto")
    p.add_argument("--precision",choices=["fp32","amp-fp16"],default="fp32")
    p.add_argument("--metrics-every",type=int,default=60,help="Sample quality every N frames; all frames are upscaled")
    p.add_argument("--reuse-identical",action="store_true",help="Offline-only reuse of exact consecutive RGB inputs; not a throughput benchmark")
    args=p.parse_args()
    if args.metrics_every<1:p.error("metrics-every must be positive")
    args.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(4)
    torch.manual_seed(0);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads(args.manifest.read_text())
    import isharescreen.enhancement.models as adapters
    environment={"command":sys.argv,"python":sys.version,"platform":platform.platform(),
                 "torch":torch.__version__,"av":av.__version__,"cuda":torch.version.cuda,
                 "manifest_sha256":sha256(args.manifest),"runner_sha256":sha256(Path(__file__)),
                 "adapter_sha256":sha256(Path(adapters.__file__)),"precision":args.precision,
                 "reuse_identical":args.reuse_identical,"metrics_every":args.metrics_every,
                 "gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                 "started":time.time(),"pid":__import__("os").getpid()}
    (args.output/"environment.json").write_text(json.dumps(environment,indent=2))
    for clip in manifest["clips"]:
        for role in ["reference","low"]:
            if sha256(args.manifest.parent/clip[role])!=clip[f"{role}_sha256"]:
                raise ValueError(f"Video input changed: {clip['id']}/{role}")
    results=[];status={};rows=[]
    fields=["clip","model","frame","time_s","psnr","ssim","reused","processing_ms"]
    with (args.output/"quality.csv").open("w",newline="") as csv_file:
        writer=csv.DictWriter(csv_file,fieldnames=fields);writer.writeheader()
        for key in args.models:
            info=next((o for o in availability() if o["id"]==key),None)
            if not info or not info["available"]:
                status[key]={"state":"unavailable","reason":info["reason"] if info else "unknown"}
                continue
            model=Upscaler(key,device=args.device,precision=args.precision)
            provenance=dict(model.provenance)
            if key in REPOS:
                provenance["source_sha"]=subprocess.check_output(
                    ["git","-C",str(project_home()/"vendor"/REPOS[key]),"rev-parse","HEAD"],text=True).strip()
            status[key]={"state":"running",**provenance}
            (args.output/"status.json").write_text(json.dumps(status,indent=2))
            for clip in manifest["clips"]:
                folder=args.output/clip["id"];folder.mkdir(exist_ok=True)
                output=folder/f"{key}-1080p-lossless.mkv"
                fps=Fraction(clip["fps"])
                sink=LosslessWriter(output,1920,1080,fps)
                selected=[];timings=[];reused=0;previous=None;last_out=None
                started=time.monotonic()
                try:
                    for i,hr,lr in aligned_frames(args.manifest.parent/clip["reference"],args.manifest.parent/clip["low"],clip["frames"],fps):
                        repeat=args.reuse_identical and previous is not None and np.array_equal(previous,lr)
                        if repeat:out=last_out;elapsed=0;reused+=1
                        else:
                            out,timing=model(lr);elapsed=timing["total_ms"];timings.append(elapsed)
                        if out.shape!=hr.shape or out.dtype!=np.uint8:raise ValueError("Invalid reconstruction geometry/type")
                        sink.write(out)
                        previous,last_out=lr,out
                        if i%args.metrics_every==0 or i==clip["frames"]-1:
                            quality=metrics(hr,out)
                            row=dict(clip=clip["id"],model=key,frame=i,time_s=float(Fraction(i,1)/fps),
                                     **quality,reused=bool(repeat),processing_ms=elapsed)
                            writer.writerow(row);csv_file.flush();rows.append(row);selected.append(quality)
                        if (i+1)%60==0:print(f"VIDEO {key} {clip['id']} {i+1}/{clip['frames']} reused={reused}",flush=True)
                finally:sink.close()
                digest=sink.digest.hexdigest()
                verify_lossless(output,clip["frames"],digest)
                result={"clip":clip["id"],"model":key,"frames":sink.count,"fps":str(fps),
                        "output":str(output.relative_to(args.output)),"sha256":sha256(output),
                        "decoded_rgb_sequence_sha256":digest,"lossless_roundtrip_verified":True,
                        "inference_calls":len(timings),"reused_frames":reused,
                        "sampled_quality_frames":len(selected),
                        "psnr":float(np.mean([s["psnr"] for s in selected])),
                        "ssim":float(np.mean([s["ssim"] for s in selected])),
                        "processing_p50_ms":float(np.percentile(timings,50)),
                        "processing_p95_ms":float(np.percentile(timings,95)),
                        "offline_elapsed_s":time.monotonic()-started}
                results.append(result)
                (args.output/"clips.json").write_text(json.dumps(results,indent=2))
                print(f"VERIFIED VIDEO {key} {clip['id']}: {sink.count} lossless frames, PSNR={result['psnr']:.3f}",flush=True)
            status[key]["state"]="completed"
            (args.output/"status.json").write_text(json.dumps(status,indent=2))
            del model
            if torch.cuda.is_available():torch.cuda.empty_cache()
    (args.output/"COMPLETED.json").write_text(json.dumps({"finished":time.time(),"videos":len(results),
        "frames_written":sum(r["frames"] for r in results),"quality_rows":len(rows),"status":status},indent=2))
    print(f"COMPLETED {len(results)} full upscaled videos",flush=True)


if __name__=="__main__":main()
