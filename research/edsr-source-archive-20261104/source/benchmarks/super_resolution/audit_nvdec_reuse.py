"""Check whether retaining DLPack frames prevents decoder output-buffer reuse."""
import argparse
import json
from pathlib import Path
import time
import torch
import PyNvVideoCodec as nvc
from benchmarks.super_resolution.profile_inference_optimization import guard,save,digest


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=dict(complete=False,started=time.time(),gpu=guard(),nvc=nvc.__version__,source_sha256=digest(Path(__file__)),modes={})
    save(a.output,r);torch.set_num_threads(1)
    run=next(x for x in json.loads((a.validation/"INPUTS.json").read_text())["runs"] if x["config"]=="720p-600k-420")
    path=a.validation/run["video"];assert digest(path)==run["video_sha256"]
    for mode in ("RGBP","NATIVE"):
        stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
        demux=nvc.CreateDemuxer(filename=str(path))
        decoder=nvc.CreateDecoder(gpuid=0,codec=demux.GetNvCodecId(),cudastream=stream.cuda_stream,
            usedevicememory=True,outputColorType=getattr(nvc.OutputColorType,mode),latency=nvc.DisplayDecodeLatencyType.LOW)
        old_frame=old_tensor=snapshot=None;records=[];index=0
        for packet in demux:
            frames=decoder.Decode(packet)
            if not frames:continue
            assert len(frames)==1
            current=torch.from_dlpack(frames[0]);stream.synchronize()
            if old_tensor is None:
                old_frame=frames[0];old_tensor=current;snapshot=current.clone();stream.synchronize()
            else:
                changed=int((old_tensor!=snapshot).sum())
                records.append(dict(frame=index,same_pointer=current.data_ptr()==old_tensor.data_ptr(),held_frame_changed_components=changed))
            del current,frames
            index+=1
            if index==20:break
        stream.synchronize()
        r["modes"][mode]=records;save(a.output,r)
        del old_tensor,old_frame,snapshot,decoder,demux
    r.update(complete=True,finished=time.time(),gpu_end=guard());save(a.output,r)
    print(json.dumps({k:dict(reused=any(x["same_pointer"] for x in v),changed_despite_owner=any(x["held_frame_changed_components"] for x in v)) for k,v in r["modes"].items()}),flush=True)


if __name__=="__main__":main()
