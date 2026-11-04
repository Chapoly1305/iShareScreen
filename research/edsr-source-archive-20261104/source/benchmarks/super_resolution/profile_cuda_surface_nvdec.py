"""Actual NVDEC frames through the bounded framework GPU-surface interface."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import PyNvVideoCodec as nvc

from isharescreen.enhancement.cuda_surface import CudaSurfaceEDSR
from benchmarks.super_resolution.profile_edsr_tiles import Engine
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save, stats


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine",type=Path,required=True);p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--samples",type=int,default=1000)
    p.add_argument("--capture",action="store_true")
    p.add_argument("--input-format",choices=("rgb","nv12_bt709_full"),default="rgb")
    p.add_argument("--decoder-latency",choices=("LOW","ZERO"),default="LOW")
    p.add_argument("--engine-sha256",default="d051effa8afd66fea52704749b665e39c9b6d2664db83dc56d03d9997101fe66")
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    result=dict(complete=False,started=time.time(),gpu=guard(),script_sha256=digest(Path(__file__)),
        runtime_sha256=digest(Path("src/isharescreen/enhancement/cuda_surface.py")),nvc=nvc.__version__,
        samples=a.samples,profiler=a.capture,validation=[],input_format=a.input_format,decoder_latency=a.decoder_latency,
        kernel_sha256=digest(Path("src/isharescreen/enhancement/cuda_kernels.py")),
        scope="Compressed packet submit -> GPU decode/color -> DLPack GPU surface API -> EDSR -> GPU RGBA1080 -> consumer stream handoff/release. No CPU pixels during measurement. Excludes demux/network/pacing/presentation/encode.")
    save(a.output,result)
    try:
        runtime=CudaSurfaceEDSR(a.engine,engine_sha256=a.engine_sha256,input_format=a.input_format)
        decode_stream=torch.cuda.Stream();torch.cuda.set_stream(decode_stream)
        consumer=torch.cuda.Stream();reference=Engine(a.engine)
        runs=[r for r in json.loads((a.validation/"INPUTS.json").read_text())["runs"] if r["config"]=="720p-600k-420"]
        run=runs[0];path=a.validation/run["video"]
        assert digest(path)==run["video_sha256"] and run["stream"]["has_b_frames"]==0
        if a.input_format=="nv12_bt709_full":
            assert run["stream"]["color_space"]=="bt709" and run["stream"]["color_range"]=="pc"
        result["source"]={k:run[k] for k in ("clip","video_sha256","stream")}
        wall=[];compute=[];decode=[];passes=0
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        while len(wall)<a.samples:
            demuxer=nvc.CreateDemuxer(filename=str(path))
            decoder=nvc.CreateDecoder(gpuid=0,codec=demuxer.GetNvCodecId(),cudacontext=0,
                cudastream=decode_stream.cuda_stream,usedevicememory=True,
                outputColorType=nvc.OutputColorType.RGBP if a.input_format=="rgb" else nvc.OutputColorType.NATIVE,
                latency=getattr(nvc.DisplayDecodeLatencyType,a.decoder_latency))
            index=0;captured=False
            for packet in demuxer:
                if index==50 and a.capture and not captured:
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push("phase_surface_nvdec");captured=True
                torch.cuda.nvtx.range_push("frame_surface_nvdec")
                t0=time.perf_counter()
                torch.cuda.nvtx.range_push("decode_api")
                frames=decoder.Decode(packet)
                torch.cuda.nvtx.range_pop();t1=time.perf_counter()
                if not frames:
                    torch.cuda.nvtx.range_pop();continue
                assert len(frames)==1
                frame=frames[0]
                begin.record(runtime.stream)
                lease=runtime.submit(frame)
                lease.wait_input_reuse_on(decode_stream)
                output=lease.wait_on(consumer)
                checking=index<50 and index in (0,24,49)
                if checking:
                    with torch.cuda.stream(consumer):snapshot=output.clone()
                lease.release();end.record(consumer);end.synchronize();t2=time.perf_counter()
                torch.cuda.nvtx.range_pop()
                if index>=50:
                    wall.append((t2-t0)*1000);compute.append(begin.elapsed_time(end));decode.append((t1-t0)*1000)
                elif checking:
                    tensor=torch.from_dlpack(frame)
                    assert tensor.data_ptr()==frame.GetPtrToPlane(0)
                    if a.input_format=="nv12_bt709_full":
                        from benchmarks.super_resolution.profile_nv12_quality import reference_rgb
                        rgb=reference_rgb(tensor)
                    else:
                        rgb=tensor
                    reference.input.copy_(rgb.unsqueeze(0).float()/255)
                    y=reference.run()
                    native=y[0].mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
                    expected=cv2.resize(native,(1920,1080),interpolation=cv2.INTER_AREA)
                    actual=snapshot.cpu().numpy()
                    assert np.array_equal(actual[:,:,:3],expected) and np.all(actual[:,:,3]==255)
                    result["validation"].append(dict(pass_index=passes,frame=index,pointer_shared=True,max_rgb=0,alpha=255))
                    del tensor,rgb
                del frame,frames
                index+=1
                if len(wall)>=a.samples:break
            if captured:
                torch.cuda.nvtx.range_pop();torch.cuda.cudart().cudaProfilerStop()
            assert index>50 and runtime.collect_completed()
            del decoder,demuxer
            passes+=1
        runtime.close()
        result.update(complete=True,finished=time.time(),gpu_end=guard(),passes=passes,
            timings=dict(packet_to_consumer_gpu_wall=stats(wall),rgb_to_consumer_gpu_event=stats(compute),decode_api_wall=stats(decode)),
            event_timing_note="rgb_to_consumer_gpu_event is a legacy field name; for NV12 mode it starts before native-surface preprocessing.",
            quality_scope="Actual decoder-frame pixel equivalence checked before timing; runtime has separate 72-frame pixel equivalence and decoder RGB has separate 72-frame PSNR validation. This is not a new independent quality dataset.")
        save(a.output,result);print(json.dumps({k:v["p95_ms"] for k,v in result["timings"].items()}),flush=True)
    except BaseException as exc:
        result.update(error=repr(exc),failed=time.time());save(a.output,result);raise


if __name__=="__main__":main()
