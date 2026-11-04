"""Check fused NV12 normalization, decoder equivalence and 72-frame EDSR quality."""
import argparse
import gc
import json
from pathlib import Path
import time
import weakref

import cv2
import numpy as np
from PIL import Image
import torch
import PyNvVideoCodec as nvc

from isharescreen.enhancement.cuda_kernels import normalize_nv12_709_full
from isharescreen.enhancement.cuda_surface import CudaSurfaceEDSR, SurfaceBusy
from benchmarks.super_resolution.profile_cuda_surface_runtime import DlpackOwner, rejects
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save


def reference_rgb(nv12):
    """Independent eager BT.709 full-range, nearest chroma and RGB8 truncation."""
    y = nv12[:720].float()
    uv = nv12[720:].reshape(360,640,2).float().repeat_interleave(2,0).repeat_interleave(2,1)
    u, v = uv[:,:,0]-128, uv[:,:,1]-128
    kg = 1 - .2126 - .0722
    return torch.stack((y+1.5748*v,
                        y-(2*.0722*(1-.0722)/kg)*u-(2*.2126*(1-.2126)/kg)*v,
                        y+1.8556*u)).clamp(0,255).byte()


def synthetic_check():
    records = []
    out = torch.empty((1,3,720,1280), device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(1729)
    for layout in ("contiguous", "padded", "column_stride"):
        for kind in ("random", "black", "white", "chroma_extreme"):
            width = 1280 if layout == "contiguous" else 1312 if layout == "padded" else 2560
            storage = torch.randint(0,256,(1080,width),device="cuda",dtype=torch.uint8,generator=generator)
            x = storage[:,:1280] if layout != "column_stride" else storage[:,::2]
            if kind in ("black","white"):
                x[:720].fill_(0 if kind == "black" else 255); x[720:].fill_(128)
            elif kind == "chroma_extreme":
                x[720:,::2].zero_(); x[720:,1::2].fill_(255)
            expected = reference_rgb(x).unsqueeze(0).float()/255
            normalize_nv12_709_full[((out.numel()+255)//256,)](x,out,*x.stride(),out.numel(),256,enable_fp_fusion=False)
            assert torch.equal(out,expected), (layout,kind,float((out-expected).abs().max()))
            records.append(dict(layout=layout,kind=kind,normalization_bit_exact=True))
    return records


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--reference-quality",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    r = dict(complete=False,started=time.time(),gpu=guard(),engine_sha256=digest(a.engine),
             source_sha256=digest(Path(__file__)),runtime_sha256=digest(Path("src/isharescreen/enhancement/cuda_surface.py")),
             kernel_sha256=digest(Path("src/isharescreen/enhancement/cuda_kernels.py")),records=[],
             scope="Correctness only: downloads are outside all performance measurements.")
    save(a.output,r)
    producer=torch.cuda.Stream(); torch.cuda.set_stream(producer)
    consumer=torch.cuda.Stream()
    r["synthetic_checks"]=synthetic_check(); producer.synchronize(); save(a.output,r)
    refs=read_refs(a.reference_quality,r["engine_sha256"])
    runtime=CudaSurfaceEDSR(a.engine,engine_sha256=r["engine_sha256"],input_format="nv12_bt709_full")
    runs=[x for x in json.loads((a.validation/"INPUTS.json").read_text())["runs"] if x["config"]=="720p-600k-420"]
    for run in runs:
        assert run["stream"]["color_space"]=="bt709" and run["stream"]["color_range"]=="pc"
        path=a.validation/run["video"]; assert digest(path)==run["video_sha256"]
        native=nvc.SimpleDecoder(str(path),gpu_id=0,cuda_stream=producer.cuda_stream,use_device_memory=True,output_color_type=nvc.OutputColorType.NATIVE)
        rgb=nvc.SimpleDecoder(str(path),gpu_id=0,cuda_stream=producer.cuda_stream,use_device_memory=True,output_color_type=nvc.OutputColorType.RGBP)
        for sample in run["samples"]:
            for kind in ("lr","hr"): assert digest(a.validation/sample[kind])==sample[kind+"_sha256"]
            frame=native[sample["frame"]]; rgb_frame=rgb[sample["frame"]]
            tensor=torch.from_dlpack(frame); original=torch.from_dlpack(rgb_frame)
            assert tensor.data_ptr()==frame.GetPtrToPlane(0) and tuple(tensor.shape)==(1080,1280)
            expected=reference_rgb(tensor)
            delta=(expected.short()-original.short()).abs()
            conversion=dict(max_rgb=int(delta.max()),mean_abs_rgb=float(delta.float().mean()),different_fraction=float((delta!=0).float().mean()))
            lease=runtime.submit(frame); lease.wait_input_reuse_on(producer); output=lease.wait_on(consumer)
            with torch.cuda.stream(consumer): snapshot=output.clone()
            lease.release(); consumer.synchronize()
            assert torch.equal(runtime._input,expected.unsqueeze(0).float()/255), "Fused normalization differs from independent reference"
            pixels=snapshot.cpu().numpy(); assert np.all(pixels[:,:,3]==255)
            hr=np.asarray(Image.open(a.validation/sample["hr"]).convert("RGB"))
            psnr=float(10*np.log10(255**2/np.mean((pixels[:,:,:3].astype(np.float64)-hr.astype(np.float64))**2)))
            reference=refs[(run["clip"],sample["frame"])]
            r["records"].append(dict(clip=run["clip"],frame=sample["frame"],conversion=conversion,
                normalized_bit_exact=True,psnr=dict(reference["psnr"],native_fused=psnr),
                deltas={k:psnr-v for k,v in reference["psnr"].items()}))
            del frame,rgb_frame,tensor,original
        assert runtime.collect_completed()
        producer.synchronize(); del native,rgb
        save(a.output,r); print("validated",len(r["records"]),flush=True)
    # Cross-stream ownership and input lifetime on the new NV12 path.
    dark=torch.zeros((1080,1280),device="cuda",dtype=torch.uint8); dark[720:].fill_(128)
    bright=dark.clone(); bright[:720].fill_(255)
    first=runtime.submit(dark); runtime.stream.synchronize()
    held=first.wait_on(consumer)
    with torch.cuda.stream(consumer): saved=held.clone()
    consumer.synchronize()
    second=runtime.submit(bright); runtime.stream.synchronize()
    rejects(SurfaceBusy,lambda:runtime.submit(dark)); rejects(SurfaceBusy,runtime.close)
    second.discard()
    third=runtime.submit(bright); runtime.stream.synchronize()
    assert torch.equal(held,saved)
    third.discard(); first.release(); consumer.synchronize()
    rejects(RuntimeError,first.release)
    rejects(RuntimeError,lambda:first.wait_input_reuse_on(producer))
    owner=DlpackOwner(bright); retained=weakref.ref(owner)
    lease=runtime.submit(owner); del owner; gc.collect()
    assert retained() is not None
    lease.discard(); runtime.stream.synchronize(); assert runtime.collect_completed()
    gc.collect(); assert retained() is None
    for invalid in (torch.empty((1080,1280),dtype=torch.uint8),torch.empty((1080,1280),device="cuda"),torch.empty((720,1280,3),device="cuda",dtype=torch.uint8)):
        rejects(ValueError,lambda invalid=invalid:runtime.submit(invalid))
    # Delay the GPU read so immediate producer reuse would otherwise overwrite
    # its source. The event must protect preprocessing, without a CPU wait.
    dark[:720].zero_(); producer.synchronize()
    with torch.cuda.stream(runtime.stream): torch.cuda._sleep(20_000_000)
    lease=runtime.submit(dark)
    lease.wait_input_reuse_on(producer)
    dark[:720].fill_(255)
    lease.wait_on(consumer); lease.release(); consumer.synchronize(); producer.synchronize()
    assert torch.count_nonzero(runtime._input).item()==0, "Producer overwrote input before GPU preprocessing completed"
    r["ownership"]=dict(held_output_preserved=True,two_slot_bound=True,owner_retained_until_completion=True,
                        invalid_inputs_rejected=True,input_reuse_event_protects_delayed_read=True)
    runtime.close()
    assert len(r["records"])==72
    metrics={k:dict(mean_delta=float(np.mean([x["deltas"][k] for x in r["records"]])),worst_delta=min(x["deltas"][k] for x in r["records"])) for k in ("fp32","baseline_amp","optimized")}
    r.update(complete=True,finished=time.time(),gpu_end=guard(),metrics=metrics,
        passed=all(metrics[k]["mean_delta"]>=-.05 and metrics[k]["worst_delta"]>=-.1 for k in ("fp32","baseline_amp")))
    save(a.output,r); print(json.dumps(dict(metrics=metrics,passed=r["passed"])),flush=True)


def read_refs(path,engine_sha):
    r=json.loads(path.read_text()); assert r["complete"] and r["engine_sha256"]==engine_sha
    return {(x["clip"],x["frame"]):x for x in r["quality"]["records"]}


if __name__=="__main__": main()
