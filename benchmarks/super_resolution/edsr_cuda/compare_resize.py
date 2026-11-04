"""Paired resize-only experiment on the frozen EDSR engine; never changes runtime."""
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
import triton
import triton.language as tl

from benchmarks.super_resolution.edsr_cuda.common import Engine, digest, guard, save, stats
from isharescreen.enhancement.cuda_kernels import pack_area


@triton.jit
def area_float(X, Y, CLAMP: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    ch = i % 4
    col, row = i // 4 % 1920, i // (4 * 1920)
    sx, sy = col * 4 // 3, row * 4 // 3
    wx, wy = (3-col % 3)*0.25, (3-row % 3)*0.25
    offset = ch*1440*2560+sy*2560+sx
    valid = (i<N) & (ch<3)
    a = tl.load(X+offset, valid, 0)*255
    b = tl.load(X+offset+1, valid, 0)*255
    c = tl.load(X+offset+2560, valid, 0)*255
    d = tl.load(X+offset+2561, valid, 0)*255
    if CLAMP:
        a=tl.minimum(tl.maximum(a,0),255)
        b=tl.minimum(tl.maximum(b,0),255)
        c=tl.minimum(tl.maximum(c,0),255)
        d=tl.minimum(tl.maximum(d,0),255)
    value=(a*wx+b*(1-wx))*wy+(c*wx+d*(1-wx))*(1-wy)
    value=tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(value),0),255)
    tl.store(Y+i,tl.where(ch==3,255,value).to(tl.uint8),i<N)


@triton.jit
def resample_h(X,T,INDEX,WEIGHT,K:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=i%1920; y=i//1920%1440; c=i//(1920*1440)
    value=tl.full((B,),0,tl.float32)
    for k in range(K):
        sx=tl.load(INDEX+x*K+k)
        w=tl.load(WEIGHT+x*K+k)
        value+=tl.load(X+c*1440*2560+y*2560+sx,i<N,0)*w
    tl.store(T+i,value,i<N)


@triton.jit
def resample_v(T,Y,INDEX,WEIGHT,K:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    c=i%4; x=i//4%1920; y=i//(4*1920)
    value=tl.full((B,),0,tl.float32)
    for k in range(K):
        sy=tl.load(INDEX+y*K+k)
        w=tl.load(WEIGHT+y*K+k)
        value+=tl.load(T+c*1440*1920+sy*1920+x,(i<N)&(c<3),0)*w
    value=tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(value*255),0),255)
    tl.store(Y+i,tl.where(c==3,255,value).to(tl.uint8),i<N)


def coefficients(source,target,kind):
    # OpenCV half-pixel coordinates, replicated boundary, no widened AA filter.
    k=4 if kind=='cubic' else 8
    pos=(np.arange(target,dtype=np.float64)+.5)*source/target-.5
    indices=np.floor(pos).astype(np.int32)[:,None]+np.arange(k)-(k//2-1)
    distance=np.abs(pos[:,None]-indices)
    if kind=='cubic':
        a=-.75
        w=np.where(distance<=1,(a+2)*distance**3-(a+3)*distance**2+1,
                   np.where(distance<2,a*distance**3-5*a*distance**2+8*a*distance-4*a,0))
    else:
        w=np.where(distance<4,np.sinc(distance)*np.sinc(distance/4),0)
    w/=w.sum(axis=1,keepdims=True)
    return (torch.from_numpy(np.clip(indices,0,source-1).astype(np.int32)).cuda(),
            torch.from_numpy(w.astype(np.float32)).cuda())


def quality(hr,rgb):
    a,b=hr.astype(np.float64),rgb.astype(np.float64)
    psnr=float(10*np.log10(255**2/np.mean((a-b)**2)))
    blur=lambda v:cv2.boxFilter(v,-1,(7,7),normalize=True)[3:-3,3:-3]
    u,v=blur(a),blur(b)
    va,vb,cov=(blur(a*a)-u*u)*49/48,(blur(b*b)-v*v)*49/48,(blur(a*b)-u*v)*49/48
    ssim=float(np.mean(((2*u*v+6.5025)*(2*cov+58.5225))/((u*u+v*v+6.5025)*(va+vb+58.5225))))
    return dict(psnr=psnr,ssim=ssim)


def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]


def ocr(rgb):
    ok,png=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));assert ok
    env=dict(os.environ,OMP_THREAD_LIMIT='1')
    result=subprocess.run(['tesseract','stdin','stdout','-l','eng','--psm','6'],input=png.tobytes(),capture_output=True,check=True,env=env)
    return ''.join(result.stdout.decode().split())


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--expected-gpu-uuid',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ['ISS_EXPECTED_GPU_UUID']=a.expected_gpu_uuid
    torch.set_num_threads(1);cv2.setNumThreads(1)
    sha=digest(a.engine);assert sha=='74f418fb7ad0cd03ae25d6ff701fead1414a6f180e334d13e9281263007dde74'
    r=dict(complete=False,started=time.time(),gpu=guard(),engine_sha256=sha,script_sha256=digest(__file__),
           input_manifest_sha256=digest(a.validation/'INPUTS.json'),torch=torch.__version__,triton=triton.__version__,opencv=cv2.__version__,
           scope='Same accepted EDSR float output for all methods; stored RGB frozen validation input, not native NVDEC. 72 frames / 24 clips. No test data, retraining or runtime change.',
           timing_scope='GPU resize plus RGBA8 packing only; resident FP32 CHW1440 -> RGBA1080. 50 warmups, 1000 CUDA-event measurements per method, two reverse-order rounds. Candidate kernels are prototypes; excludes inference/decode/transfers/display.',
           ocr_scope='Exploratory English OCR agreement against OCR of HR, not human transcription: fixed central 960x540 crop on first frame of every third clip (8 clips); whitespace removed; PSM6. No Chinese/readability claim.',
           filter_scope='Cubic a=-0.75, Lanczos4 8 taps per axis, half-pixel centers and replicated edges; OpenCV-style filters without scale-widened antialiasing.',rows=[],timings=[],ocr=[])
    save(a.output/'RESULTS.json',r)
    engine=Engine(a.engine);x=engine.output
    output=torch.empty((1080,1920,4),dtype=torch.uint8,device='cuda');n=output.numel()
    temp=torch.empty((3,1440,1920),device='cuda')
    plans={k:(coefficients(2560,1920,k),coefficients(1440,1080,k)) for k in ['cubic','lanczos4']}
    names=['legacy_area','clamp_float_area','float_area','cubic','lanczos4']
    def run(name):
        if name=='legacy_area':pack_area[(triton.cdiv(n,256),)](x,output,1440*2560,2560,1,4,n,256,enable_fp_fusion=False)
        elif name.endswith('area'):area_float[(triton.cdiv(n,256),)](x,output,name=='clamp_float_area',n,256,enable_fp_fusion=False)
        else:
            h,v=plans[name];k=4 if name=='cubic' else 8
            resample_h[(triton.cdiv(temp.numel(),256),)](x,temp,*h,k,temp.numel(),256,enable_fp_fusion=False)
            resample_v[(triton.cdiv(n,256),)](temp,output,*v,k,n,256,enable_fp_fusion=False)
        return output
    runs=[v for v in json.loads((a.validation/'INPUTS.json').read_text())['runs'] if v['config']=='720p-600k-420']
    assert len(runs)==24
    for ci,clip in enumerate(runs):
        for si,sample in enumerate(clip['samples']):
            for k in ['lr','hr']:assert digest(a.validation/sample[k])==sample[k+'_sha256']
            lr=np.array(Image.open(a.validation/sample['lr']).convert('RGB'))
            hr=np.array(Image.open(a.validation/sample['hr']).convert('RGB'));assert hr.shape==(1080,1920,3)
            engine.input.copy_(torch.from_numpy(lr.copy()).permute(2,0,1).unsqueeze(0).cuda().float()/255)
            engine.run();native=x[0].permute(1,2,0).contiguous().cpu().numpy()*255
            assert np.isfinite(native).all()
            cpus={
                'legacy_area':cv2.resize(np.clip(np.rint(native),0,255).astype('uint8'),(1920,1080),interpolation=cv2.INTER_AREA),
                'clamp_float_area':cv2.resize(np.clip(native,0,255),(1920,1080),interpolation=cv2.INTER_AREA),
                'float_area':cv2.resize(native,(1920,1080),interpolation=cv2.INTER_AREA),
                'cubic':cv2.resize(native,(1920,1080),interpolation=cv2.INTER_CUBIC),
                'lanczos4':cv2.resize(native,(1920,1080),interpolation=cv2.INTER_LANCZOS4)}
            selected=ci%3==0 and si==0
            if selected:
                crop=(slice(270,810),slice(480,1440));truth=ocr(hr[crop]);Image.fromarray(hr[crop]).save(a.output/f'crop-{ci:02d}-hr.png')
            for name in names:
                rgba=run(name).cpu().numpy();assert np.all(rgba[:,:,3]==255)
                rgb=rgba[:,:,:3].copy();ref=np.clip(np.rint(cpus[name]),0,255).astype('uint8')
                error=np.abs(rgb.astype('int16')-ref.astype('int16'))
                assert error.max()<=(0 if name=='legacy_area' else 1),(name,error.max())
                row=dict(clip=clip['clip'],frame=sample['frame'],method=name,**quality(hr,rgb),
                         cpu_reference_max_error=int(error.max()),cpu_reference_different_fraction=float(np.mean(error!=0)))
                r['rows'].append(row)
                if selected:
                    text=ocr(rgb[crop]);r['ocr'].append(dict(clip=clip['clip'],method=name,hr_text=truth,text=text,reference_chars=len(truth),edits=edit_distance(truth,text)))
                    Image.fromarray(rgb[crop]).save(a.output/f'crop-{ci:02d}-{name}.png')
            save(a.output/'RESULTS.json',r)
        print('quality',ci+1,'/24 clips',flush=True)
    # Fixed last-frame output, all variants interleaved by round; no concurrent quality work.
    for round_index,order in enumerate([names,list(reversed(names))]):
        for name in order:
            guard()
            for _ in range(50):run(name)
            torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            times=[]
            for _ in range(1000):
                start.record();run(name);end.record();end.synchronize();times.append(start.elapsed_time(end))
            r['timings'].append(dict(round=round_index,method=name,**stats(times)))
            save(a.output/'RESULTS.json',r);print('timing',round_index,name,stats(times)['p95_ms'],flush=True)
    base=[v for v in r['rows'] if v['method']=='legacy_area'];r['summary']={}
    for name in names:
        rows=[v for v in r['rows'] if v['method']==name]
        delta=[v['psnr']-b['psnr'] for v,b in zip(rows,base)]
        texts=[v for v in r['ocr'] if v['method']==name];chars=sum(v['reference_chars'] for v in texts)
        r['summary'][name]=dict(psnr=float(np.mean([v['psnr'] for v in rows])),ssim=float(np.mean([v['ssim'] for v in rows])),
            mean_psnr_delta=float(np.mean(delta)),worst_psnr_delta=min(delta),best_psnr_delta=max(delta),
            improved_frames=sum(v>1e-6 for v in delta),regressed_frames=sum(v< -1e-6 for v in delta),
            ocr_reference_chars=chars,ocr_edits=sum(v['edits'] for v in texts),
            ocr_proxy_cer=sum(v['edits'] for v in texts)/chars if chars else None,
            p95_ms=[v['p95_ms'] for v in r['timings'] if v['method']==name])
    r.update(complete=True,finished=time.time(),gpu_end=guard())
    save(a.output/'RESULTS.json',r);print(json.dumps(r['summary'],indent=2),flush=True)


if __name__=='__main__':main()
