"""Test cache reuse across two fused convolutions without feature-copy kernels.

Full-width strips bind directly into the original feature image. Only the
first convolution's one-row halo is materialized in a small scratch tensor;
the second writes its core straight into the destination frame.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from benchmarks.super_resolution.edsr_winograd import transform_weights,winograd
from benchmarks.super_resolution.profile_inference_optimization import digest,guard,save,stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(1);torch.manual_seed(1729)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    result=dict(complete=False,started=time.time(),gpu=guard(),script_sha256=digest(Path(__file__)),
        kernel_sha256=digest(Path(__file__).with_name("edsr_winograd.py")),rows=[],
        scope="One 64-channel residual block on full 720p CUDA features. Random data/weights. Every pixel evaluated. No host pixels or feature crop/stitch copies; no full EDSR performance or quality claim.")
    save(a.output,result)
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    weights=[torch.randn((64,64,3,3))*.02 for _ in range(2)]
    transformed=[transform_weights(w).to(device="cuda",dtype=torch.float16) for w in weights]
    biases=[torch.randn(64,device="cuda")*.01 for _ in range(2)]
    with torch.inference_mode():
        x=torch.randn((1,64,720,1280),device="cuda",dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        out=torch.empty_like(x)
        expected=F.conv2d(F.conv2d(x.float(),weights[0].cuda(),biases[0],padding=1).relu(),weights[1].cuda(),biases[1],padding=1)+x.float()
        for rows in [1,4,8,12,20]:
            core=720//rows;halo_height=720 if rows==1 else core+2
            scratch=torch.empty((1,64,halo_height,1280),device="cuda",dtype=torch.float16,memory_format=torch.channels_last)
            for block_m,warps,unroll in [(32,4,4),(64,8,1)]:
                def predict():
                    if rows==1:
                        winograd(x,transformed[0],biases[0],scratch,relu=True,block_m=block_m,warps=warps,unroll=unroll)
                        winograd(scratch,transformed[1],biases[1],out,skip=x,block_m=block_m,warps=warps,unroll=unroll)
                    else:
                        for start in range(0,720,core):
                            winograd(x,transformed[0],biases[0],scratch,relu=True,block_m=block_m,warps=warps,unroll=unroll,
                                dst_y0=start-1,out_y0=start-1,out_h=core+2,global_h=720)
                            winograd(scratch,transformed[1],biases[1],out,skip=x,block_m=block_m,warps=warps,unroll=unroll,
                                src_y0=start-1,out_y0=start,out_h=core,global_h=720)
                    return out

                t0=time.time();predict();stream.synchronize()
                delta=(out.float()-expected).abs()
                row=dict(rows=rows,core_height=core,scratch_height=halo_height,block_m=block_m,warps=warps,unroll=unroll,
                    setup_s=time.time()-t0,max_abs_error=float(delta.max()),mean_abs_error=float(delta.mean()))
                assert torch.isfinite(out).all().item() and row["max_abs_error"]<.01,row
                for _ in range(50):predict()
                stream.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):predict()
                for _ in range(50):graph.replay()
                stream.synchronize();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                samples=[]
                for _ in range(200):
                    begin.record();graph.replay();end.record();end.synchronize();samples.append(begin.elapsed_time(end))
                row["latency"]=stats(samples);result["rows"].append(row);save(a.output,result)
                print(json.dumps({k:v for k,v in row.items() if k!="latency"}),"p95",row["latency"]["p95_ms"],flush=True)
                del graph,delta
        result.update(complete=True,finished=time.time(),gpu_end=guard());save(a.output,result)


if __name__=="__main__":main()
