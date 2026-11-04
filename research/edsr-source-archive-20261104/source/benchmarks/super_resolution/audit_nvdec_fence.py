"""Verify GPU event ordering protects real recycled decoder buffers."""
import argparse
import json
from pathlib import Path
import time
import torch
import PyNvVideoCodec as nvc
from isharescreen.enhancement.cuda_surface import CudaSurfaceEDSR
from benchmarks.super_resolution.profile_nv12_quality import reference_rgb
from benchmarks.super_resolution.profile_inference_optimization import guard,save,digest


def next_frame(decoder, packets):
    # Demux may emit codec-configuration packets before the first picture.
    for packet in packets:
        frames=decoder.Decode(packet)
        if frames:
            assert len(frames)==1
            return frames
    raise RuntimeError("Stream ended before another decoded picture")


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine",type=Path,required=True);p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=dict(complete=False,started=time.time(),gpu=guard(),source_sha256=digest(Path(__file__)),
        engine_sha256=digest(a.engine),runtime_sha256=digest(Path("src/isharescreen/enhancement/cuda_surface.py")),modes={})
    save(a.output,r);torch.set_num_threads(1)
    run=next(x for x in json.loads((a.validation/"INPUTS.json").read_text())["runs"] if x["config"]=="720p-600k-420")
    path=a.validation/run["video"];assert digest(path)==run["video_sha256"]
    for mode in ("rgb","nv12_bt709_full"):
        producer=torch.cuda.Stream();consumer=torch.cuda.Stream();torch.cuda.set_stream(producer)
        runtime=CudaSurfaceEDSR(a.engine,engine_sha256=r["engine_sha256"],input_format=mode)
        demux=nvc.CreateDemuxer(filename=str(path));packets=iter(demux)
        decoder=nvc.CreateDecoder(gpuid=0,codec=demux.GetNvCodecId(),cudastream=producer.cuda_stream,
            usedevicememory=True,outputColorType=nvc.OutputColorType.RGBP if mode=="rgb" else nvc.OutputColorType.NATIVE,
            latency=nvc.DisplayDecodeLatencyType.LOW)
        frames=next_frame(decoder,packets)
        frame=frames[0];tensor=torch.from_dlpack(frame)
        expected=(tensor if mode=="rgb" else reference_rgb(tensor)).unsqueeze(0).float()/255
        expected=expected.clone();producer.synchronize()
        with torch.cuda.stream(runtime.stream):torch.cuda._sleep(20_000_000)
        lease=runtime.submit(frame)
        lease.wait_input_reuse_on(producer)
        following=next_frame(decoder,packets)
        next_tensor=torch.from_dlpack(following[0])
        same_pointer=next_tensor.data_ptr()==tensor.data_ptr()
        still_inference=not lease.ready()
        lease.wait_on(consumer);lease.release();consumer.synchronize();producer.synchronize()
        assert same_pointer, "Test did not exercise decoder storage reuse"
        assert torch.equal(runtime._input,expected), "A following decode overwrote in-flight preprocessing input"
        r["modes"][mode]=dict(recycled_pointer=True,normalized_original_frame_preserved=True,
                              next_decode_returned_before_inference_completed=still_inference)
        assert runtime.collect_completed()
        del next_tensor,following,tensor,frame,frames,decoder,demux
        runtime.close();save(a.output,r)
    r.update(complete=True,finished=time.time(),gpu_end=guard());save(a.output,r)
    print(json.dumps(r["modes"]),flush=True)


if __name__=="__main__":main()
