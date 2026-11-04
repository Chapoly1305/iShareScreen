"""Local error and OCR-agreement audit, not ground-truth readability accuracy."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import torch

from benchmarks.super_resolution.profile_edsr_tiles import Engine
from benchmarks.super_resolution.profile_inference_optimization import digest,guard,save


def rgb(engine,x):
    engine.input.copy_(x);y=engine.run()
    native=y[0].mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
    return cv2.resize(native,(1920,1080),interpolation=cv2.INTER_AREA)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fp16",type=Path,required=True);p.add_argument("--int8",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);dest=a.output/"DETAIL.json"
    if dest.exists():raise FileExistsError(dest)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    result=dict(complete=False,started=time.time(),gpu=guard(),script_sha256=digest(Path(__file__)),
        engine_sha256=dict(fp16=digest(a.fp16),int8=digest(a.int8)),records=[],scope=__doc__,
        selection="Top five frames by maximum absolute RGB difference; crop centered at worst pixel. Development validation only. OCR uses identical preprocessing for both outputs and clean reference; OCR agreement is not transcription accuracy.")
    save(dest,result)
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    engines={"fp16":Engine(a.fp16),"int8":Engine(a.int8)}
    runs=[r for r in json.loads((a.validation/"INPUTS.json").read_text())["runs"] if r["config"]=="720p-600k-420"]
    top=[];total=np.zeros(256,dtype=np.int64)
    for run in runs:
        for s in run["samples"]:
            for k in ("lr","hr"):assert digest(a.validation/s[k])==s[k+"_sha256"]
            low=np.asarray(Image.open(a.validation/s["lr"]).convert("RGB"))
            x=torch.from_numpy(np.ascontiguousarray(low.transpose(2,0,1))).unsqueeze(0).cuda().float()/255
            images={k:rgb(e,x) for k,e in engines.items()}
            delta=np.abs(images["int8"].astype(np.int16)-images["fp16"].astype(np.int16))
            histogram=np.bincount(delta.ravel(),minlength=256);total+=histogram
            y,xpos,c=np.unravel_index(np.argmax(delta),delta.shape)
            row=dict(clip=run["clip"],frame=s["frame"],content=run["content"],max_rgb=int(delta.max()),
                mean_abs_rgb=float(delta.mean()),max_location=[int(xpos),int(y),int(c)],
                fractions={str(t):float((delta>t).mean()) for t in (0,1,2,5,10)})
            result["records"].append(row)
            if len(top)<5 or row["max_rgb"]>min(x[0]["max_rgb"] for x in top):
                images["clean"]=np.asarray(Image.open(a.validation/s["hr"]).convert("RGB"))
                top.append((row,images));top=sorted(top,key=lambda x:-x[0]["max_rgb"])[:5]
    result["histogram_rgb_abs_difference"]=total.tolist()
    result["fractions_over"]=dict((str(t),float(total[t+1:].sum()/total.sum())) for t in (0,1,2,5,10))
    result["crops"]=[]
    for i,(row,images) in enumerate(top):
        centerx,centery,_=row["max_location"]
        left=max(0,min(centerx-160,1920-320));up=max(0,min(centery-80,1080-160))
        entry=dict(clip=row["clip"],frame=row["frame"],max_rgb=row["max_rgb"],xywh=[left,up,320,160],images={},ocr={})
        for name,array in images.items():
            crop=Image.fromarray(array[up:up+160,left:left+320])
            path=a.output/f"crop-{i}-{name}.png";crop.save(path);entry["images"][name]=path.name
            scaled=a.output/f"ocr-{i}-{name}.png";crop.resize((960,480),Image.Resampling.LANCZOS).save(scaled)
            entry["ocr"][name]=subprocess.check_output(["tesseract",str(scaled),"stdout","-l","eng","--oem","1","--psm","6"],
                text=True,stderr=subprocess.DEVNULL,env=dict(os.environ,OMP_THREAD_LIMIT="1"))
        entry["ocr_fp16_int8_identical"]="".join(entry["ocr"]["fp16"].split())=="".join(entry["ocr"]["int8"].split())
        result["crops"].append(entry)
    result.update(complete=True,finished=time.time(),gpu_end=guard())
    save(dest,result);print(json.dumps(dict(fractions=result["fractions_over"],ocr_identical=[x["ocr_fp16_int8_identical"] for x in result["crops"]])),flush=True)


if __name__=="__main__":main()
