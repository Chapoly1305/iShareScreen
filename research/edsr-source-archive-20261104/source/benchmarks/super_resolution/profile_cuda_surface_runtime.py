"""Validate GPU surface ownership, stream ordering, pixels and actual latency."""
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

from isharescreen.enhancement.cuda_surface import CudaSurfaceEDSR, SurfaceBusy
from benchmarks.super_resolution.profile_edsr_tiles import Engine
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save, stats


class DlpackOwner:
    def __init__(self, tensor): self.tensor=tensor
    def __dlpack_device__(self): return self.tensor.__dlpack_device__()
    def __dlpack__(self, stream=None): return self.tensor.__dlpack__(stream=stream)


def rejects(error, fn):
    try: fn()
    except error: return
    raise AssertionError("Expected "+error.__name__)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine",type=Path,required=True);p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--samples",type=int,default=1000)
    p.add_argument("--capture",action="store_true")
    p.add_argument("--engine-sha256",default="d051effa8afd66fea52704749b665e39c9b6d2664db83dc56d03d9997101fe66")
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    result=dict(complete=False,started=time.time(),gpu=guard(),script_sha256=digest(Path(__file__)),
        runtime_sha256=digest(Path("src/isharescreen/enhancement/cuda_surface.py")),
        kernel_sha256=digest(Path("src/isharescreen/enhancement/cuda_kernels.py")),
        samples=a.samples,capture=a.capture,scope="GPU RGB720 -> EDSR -> GPU RGBA1080 through production surface API; includes submit, stream handoff and consumer release; no decoder or presentation")
    save(a.output,result)
    try:
        expected=a.engine_sha256
        runtime=CudaSurfaceEDSR(a.engine,engine_sha256=expected)
        result["provenance"]=runtime.provenance
        reference=Engine(a.engine)
        producer=torch.cuda.current_stream();consumer=torch.cuda.Stream()
        runs=[r for r in json.loads((a.validation/"INPUTS.json").read_text())["runs"] if r["config"]=="720p-600k-420"]
        rows=[]
        for run in runs:
            for sample in run["samples"]:
                path=a.validation/sample["lr"];assert digest(path)==sample["lr_sha256"]
                rgb=np.asarray(Image.open(path).convert("RGB")).copy()
                tensor=torch.from_numpy(rgb).cuda()
                # Alternate layout, use padded rows, and exercise DLPack handoff.
                if len(rows)%2:
                    padded=torch.empty((3,720,1312),dtype=torch.uint8,device="cuda")
                    padded[:,:,:1280].copy_(tensor.permute(2,0,1));tensor=padded[:,:,:1280]
                    inp=DlpackOwner(tensor)
                    chw=tensor
                else:
                    inp=tensor;chw=tensor.permute(2,0,1)
                lease=runtime.submit(inp)
                actual=lease.wait_on(consumer)
                with torch.cuda.stream(consumer):
                    snapshot=actual.clone()
                lease.release();consumer.synchronize()
                reference.input.copy_(chw.unsqueeze(0).float()/255)
                y=reference.run()
                native=y[0].mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
                target=cv2.resize(native,(1920,1080),interpolation=cv2.INTER_AREA)
                image=snapshot.cpu().numpy()
                assert np.array_equal(image[:,:,:3],target) and np.all(image[:,:,3]==255)
                rows.append(dict(clip=run["clip"],frame=sample["frame"],max_rgb=0,alpha=255,
                                 layout="padded CHW DLPack" if len(rows)%2 else "HWC tensor"))
        assert len(rows)==72
        result["pixel_validation"]=rows;save(a.output,result)

        # Holding output A must prevent its storage being overwritten by B/C.
        black=torch.zeros((720,1280,3),dtype=torch.uint8,device="cuda")
        white=torch.full_like(black,255)
        first=runtime.submit(black);runtime.stream.synchronize()
        first_tensor=first.wait_on(consumer)
        with torch.cuda.stream(consumer): before=first_tensor.clone()
        second=runtime.submit(white);runtime.stream.synchronize()
        rejects(SurfaceBusy,lambda:runtime.submit(black))
        rejects(SurfaceBusy,runtime.close)
        second.discard()
        third=runtime.submit(white);runtime.stream.synchronize()
        third_tensor=third.wait_on(consumer)
        assert first_tensor.data_ptr()!=third_tensor.data_ptr()
        with torch.cuda.stream(consumer): unchanged=torch.equal(first_tensor,before)
        assert unchanged
        first.release();third.release();consumer.synchronize()
        rejects(RuntimeError,first.release);rejects(RuntimeError,first.ready)
        result["ownership"]=dict(held_output_preserved=True,two_slot_bound=True,stale_release_rejected=True,close_with_held_output_rejected=True)

        # The runtime retains a decoder owner through asynchronous execution.
        owner=DlpackOwner(white);owner_ref=weakref.ref(owner)
        lease=runtime.submit(owner);del owner;gc.collect()
        assert owner_ref() is not None
        lease.discard();runtime.stream.synchronize()
        following=runtime.submit(black);gc.collect()
        assert owner_ref() is None
        following.discard();runtime.stream.synchronize()
        result["ownership"]["decoder_owner_retained_until_completion"]=True
        for bad in (torch.empty((720,1280,3),dtype=torch.uint8),torch.empty((720,1280,3),device="cuda"),torch.empty((2,2,3),dtype=torch.uint8,device="cuda")):
            rejects(ValueError,lambda bad=bad:runtime.submit(bad))
        result["invalid_inputs_rejected"]=True
        save(a.output,result)
        sample=runs[0]["samples"][1]
        x=torch.from_numpy(np.asarray(Image.open(a.validation/sample["lr"]).convert("RGB")).copy()).cuda()
        def once():
            lease=runtime.submit(x)
            lease.wait_on(consumer)
            lease.release()
        for _ in range(50):
            once();consumer.synchronize()
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        times=[];device=[]
        if a.capture:torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("phase_surface_gpu")
        for _ in range(a.samples):
            torch.cuda.nvtx.range_push("frame_surface_gpu")
            start=time.perf_counter();begin.record(runtime.stream)
            once();end.record(consumer);end.synchronize()
            times.append((time.perf_counter()-start)*1000);device.append(begin.elapsed_time(end))
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        if a.capture:torch.cuda.cudart().cudaProfilerStop()
        result.update(timings=dict(wall=stats(times),cuda_event=stats(device)),context_memory_bytes=runtime.context_memory_bytes)
        runtime.close();rejects(RuntimeError,lambda:runtime.submit(x))
        result.update(complete=True,finished=time.time(),gpu_end=guard())
        save(a.output,result);print(json.dumps({k:v["p95_ms"] for k,v in result["timings"].items()}),flush=True)
    except BaseException as exc:
        result.update(error=repr(exc),failed=time.time());save(a.output,result);raise


if __name__=="__main__":main()
