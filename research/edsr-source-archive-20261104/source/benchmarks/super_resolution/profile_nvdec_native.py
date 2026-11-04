"""Inspect native NV12 device surfaces and isolate decode/handoff latency."""
import argparse
import json
from pathlib import Path
import time

import torch
import PyNvVideoCodec as nvc

from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save, stats


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=300)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    r = dict(complete=False, started=time.time(), gpu=guard(), source_sha256=digest(Path(__file__)),
             nvc=nvc.__version__, scope=__doc__, modes={})
    save(a.output, r)
    torch.set_num_threads(1)
    runs = [x for x in json.loads((a.validation / "INPUTS.json").read_text())["runs"] if x["config"] == "720p-600k-420"]
    r["stream_colors"] = sorted({str({k:x["stream"].get(k) for k in ("pix_fmt","color_range","color_space","chroma_location")}) for x in runs})
    run = runs[0]; path = a.validation / run["video"]
    assert digest(path) == run["video_sha256"] and run["stream"]["has_b_frames"] == 0
    r["source"] = {k:run[k] for k in ("clip","video_sha256","stream")}
    snapshots = {}
    for mode in ("RGBP", "NATIVE"):
        stream = torch.cuda.Stream(); consumer = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        demux = nvc.CreateDemuxer(filename=str(path))
        decoder = nvc.CreateDecoder(gpuid=0, codec=demux.GetNvCodecId(), cudacontext=0,
            cudastream=stream.cuda_stream, usedevicememory=True, outputColorType=getattr(nvc.OutputColorType, mode),
            latency=nvc.DisplayDecodeLatencyType.LOW)
        api, handoff, total = [], [], []
        event = torch.cuda.Event()
        record = {}
        index = 0
        for packet in demux:
            start = time.perf_counter()
            frames = decoder.Decode(packet)
            after_decode = time.perf_counter()
            if not frames: continue
            assert len(frames) == 1
            frame = frames[0]
            with torch.cuda.stream(consumer):
                tensor = torch.from_dlpack(frame)
                after_handoff = time.perf_counter()
                event.record()
            event.synchronize()
            stop = time.perf_counter()
            assert tensor.is_cuda and tensor.data_ptr() == frame.GetPtrToPlane(0)
            if index == 0:
                record.update(format=str(frame.format), shape=list(tensor.shape), strides=list(tensor.stride()),
                              decoder_shape=list(frame.shape), decoder_strides=list(frame.strides), dtype=str(tensor.dtype),
                              pointer_shared=True, plane1_byte_offset=frame.GetPtrToPlane(1)-frame.GetPtrToPlane(0))
                with torch.cuda.stream(consumer): snapshots[mode] = tensor.clone()
                consumer.synchronize()
            if index >= 50:
                api.append((after_decode-start)*1000)
                handoff.append((after_handoff-after_decode)*1000)
                total.append((stop-start)*1000)
            del tensor,frame,frames
            index += 1
            if len(api) == a.samples: break
        assert len(api) == a.samples
        consumer.synchronize(); stream.synchronize()
        del decoder, demux
        record.update(api_wall=stats(api), handoff_wall=stats(handoff), decoded_ready_wall=stats(total))
        r["modes"][mode] = record
        print(mode,record["shape"],record["strides"],{k:record[k]["p95_ms"] for k in ("api_wall","decoded_ready_wall")},flush=True)
        save(a.output,r)
    torch.cuda.synchronize()
    nv12 = snapshots["NATIVE"]
    assert tuple(nv12.shape) == (1080,1280) and nv12.dtype == torch.uint8
    y = nv12[:720].float()
    uv = nv12[720:].reshape(360,640,2).float().repeat_interleave(2,0).repeat_interleave(2,1)
    u, v = uv[:,:,0]-128, uv[:,:,1]-128
    rgb = snapshots["RGBP"]
    r["conversion_screen"] = []
    for kr,kb,name in ((.2126,.0722,"bt709"),(.299,.114,"bt601")):
        for full in (True,False):
            yy = y if full else (y-16)*(255/219)
            uu, vv = (u,v) if full else (u*(255/224),v*(255/224))
            values = torch.stack((yy+2*(1-kr)*vv,
                yy-2*kb*(1-kb)/(1-kr-kb)*uu-2*kr*(1-kr)/(1-kr-kb)*vv,
                yy+2*(1-kb)*uu))
            for rounding in ("nearest","truncate"):
                output = (values.round() if rounding == "nearest" else values).clamp(0,255).byte()
                delta = (output.short()-rgb.short()).abs()
                r["conversion_screen"].append(dict(matrix=name,full_range=full,rounding=rounding,
                    max_rgb=int(delta.max()),mean_abs_rgb=float(delta.float().mean()),different_fraction=float((delta!=0).float().mean())))
    r.update(complete=True,finished=time.time(),gpu_end=guard())
    save(a.output,r)
    print(json.dumps(r["conversion_screen"],indent=2),flush=True)


if __name__ == "__main__": main()
