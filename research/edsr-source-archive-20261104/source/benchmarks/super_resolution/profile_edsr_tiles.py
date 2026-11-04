"""Same-checkpoint EDSR cache/tiling experiments with full-frame validation."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
import tensorrt as trt

from benchmarks.super_resolution.profile_inference_optimization import adapter, amp, digest, forward, guard, save, stats


class NormalizedEDSR(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model

    def forward(self,x):
        return self.model(x*255)/255


class Engine:
    def __init__(self,path):
        self.logger=trt.Logger(trt.Logger.WARNING)
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(path.read_bytes())
        self.context=self.engine.create_execution_context()
        self.input=torch.empty(tuple(self.engine.get_tensor_shape("input")),device="cuda",dtype=torch.float32)
        self.output=torch.empty(tuple(self.engine.get_tensor_shape("output")),device="cuda",dtype=torch.float32)
        self.context.set_tensor_address("input",self.input.data_ptr())
        self.context.set_tensor_address("output",self.output.data_ptr())

    def run(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):raise RuntimeError("TensorRT failed")
        return self.output


def tile_coordinates(rows,cols,halo=36):
    assert 720%rows==0 and 1280%cols==0
    ch,cw=720//rows,1280//cols
    th=ch+min(rows-1,2)*halo;tw=cw+min(cols-1,2)*halo
    coords=[]
    for row in range(rows):
        for col in range(cols):
            dy,dx=row*ch,col*cw
            sy=min(max(dy-halo,0),720-th);sx=min(max(dx-halo,0),1280-tw)
            coords.append(dict(sy=sy,sx=sx,dy=dy,dx=dx,cy=dy-sy,cx=dx-sx))
    return th,tw,ch,cw,coords


class TypedNormalizedEDSR(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model.half()

    def forward(self,x):
        return self.model((x*255).half()).float()/255


def build(model,shape,out,native_tiling="none",l2_mib=0,fold_tail=False,strongly_typed=False,max_tactics=-1):
    strongly_typed=strongly_typed or int(trt.__version__.split('.')[0])>=11
    if strongly_typed:model=copy.deepcopy(model)
    if fold_tail:
        from benchmarks.super_resolution.edsr_folded_tail import FoldedTail
        model=copy.deepcopy(model)
        model.tail=FoldedTail(model.tail)
    onnx=out/"edsr.onnx"
    with torch.inference_mode():
        wrapper=TypedNormalizedEDSR(model) if strongly_typed else NormalizedEDSR(model)
        dynamic={"input":{2:"h",3:"w"},"output":{2:"oh",3:"ow"}} if strongly_typed else None
        torch.onnx.export(wrapper,torch.zeros(1,3,8,8) if strongly_typed else torch.zeros(1,3,*shape),str(onnx),
            opset_version=17,input_names=["input"],output_names=["output"],dynamo=False,do_constant_folding=True,dynamic_axes=dynamic)
    logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
    flags=0
    if strongly_typed and int(trt.__version__.split('.')[0])<11:flags=1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network=builder.create_network(flags)
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(onnx.read_bytes()):raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    if strongly_typed:network.get_input(0).shape=(1,3,*shape)
    config=builder.create_builder_config()
    if not strongly_typed:config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(trt.BuilderFlag,"TF32"):config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level=5;config.avg_timing_iterations=8
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,4<<30)
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    config.tiling_optimization_level=getattr(trt.TilingOptimizationLevel,native_tiling.upper())
    if l2_mib:config.l2_limit_for_tiling=l2_mib<<20
    if max_tactics>=0:config.max_num_tactics=max_tactics
    blob=builder.build_serialized_network(network,config)
    if blob is None:raise RuntimeError("Build failed")
    path=out/"edsr.engine";path.write_bytes(bytes(blob));return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights",type=Path,required=True);p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--reference-engine",type=Path)
    p.add_argument("--engine",type=Path);p.add_argument("--rows",type=int,default=1);p.add_argument("--cols",type=int,default=1)
    p.add_argument("--samples",type=int,default=200);p.add_argument("--quality",action="store_true")
    p.add_argument("--native-tiling",choices=["none","fast","moderate","full"],default="none")
    p.add_argument("--l2-mib",type=int,default=0)
    p.add_argument("--fold-tail",action="store_true")
    p.add_argument("--strongly-typed",action="store_true")
    p.add_argument("--max-tactics",type=int,default=-1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);dest=a.output/"RESULTS.json"
    if dest.exists():raise FileExistsError(dest)
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    plan=json.loads((a.weights/"PLAN.json").read_text());item=next(r for r in plan["models"] if r["model"]=="edsr")
    ckpt=a.weights/"checkpoints/edsr.pt";assert digest(ckpt)==item["sha256"]
    th,tw,ch,cw,coords=tile_coordinates(a.rows,a.cols)
    result=dict(complete=False,checkpoint=item,started=time.time(),gpu_start=guard(),pid=os.getpid(),
        torch=str(torch.__version__),trt=trt.__version__,script_sha256=digest(Path(__file__)),
        grid=[a.rows,a.cols],halo=36,tile_shape=[th,tw],coords=coords,
        pixel_work_ratio=a.rows*a.cols*th*tw/(720*1280),command=os.sys.argv,
        scope="Whole 720p resident RGB ->1440p RGB. Includes tile input copies and output stitching, TensorRT (precision determined by the hashed engine/build record), unchanged checkpoint. No decode/display. Graph capture still runs every tile for each call.")
    save(dest,result)
    try:
        model=adapter("edsr","cpu").model.eval().requires_grad_(False)
        torch.nn.Module.load_state_dict(model,torch.load(ckpt,map_location="cpu",weights_only=True)["state_dict"],strict=True)
        t0=time.time();path=a.engine or build(model,(th,tw),a.output,a.native_tiling,a.l2_mib,a.fold_tail,a.strongly_typed,a.max_tactics)
        result.update(build_s=time.time()-t0,engine_sha256=digest(path),engine=str(path))
        engine=Engine(path);assert engine.input.shape==(1,3,th,tw)
        inspector=engine.engine.create_engine_inspector()
        (a.output/"engine-layers.json").write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        x=torch.linspace(0,1,1280*720*3,device="cuda").reshape(1,3,720,1280)
        output=torch.empty((1,3,1440,2560),device="cuda",dtype=torch.float32)

        def tiled():
            for c in coords:
                engine.input.copy_(x[:,:,c["sy"]:c["sy"]+th,c["sx"]:c["sx"]+tw])
                y=engine.run()
                output[:,:,c["dy"]*2:(c["dy"]+ch)*2,c["dx"]*2:(c["dx"]+cw)*2].copy_(
                    y[:,:,c["cy"]*2:(c["cy"]+ch)*2,c["cx"]*2:(c["cx"]+cw)*2])
            return output

        with torch.inference_mode():
            for _ in range(50):y=tiled()
            torch.cuda.synchronize()
            eager_result=y.clone()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):tiled()
            graph.replay();torch.cuda.synchronize()
            assert torch.equal(output,eager_result)
            result["graph_matches_eager"]=True
            for label,fn in [("eager",tiled),("graph",graph.replay)]:
                for _ in range(50):fn()
                torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                times=[];wall=[]
                for _ in range(a.samples):
                    t0=time.perf_counter();start.record();fn();end.record();end.synchronize()
                    times.append(start.elapsed_time(end));wall.append((time.perf_counter()-t0)*1000)
                result[label]=stats(times);result[label+"_wall"]=stats(wall)
            result["gpu_after"]=guard();save(dest,result)
            print("grid",a.rows,a.cols,"eager",result["eager"]["p95_ms"],"graph",result["graph"]["p95_ms"],flush=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(5):graph.replay()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(a.output/"trace.json"))
            if a.quality:
                reference_engine=Engine(a.reference_engine) if a.reference_engine else None
                model.to("cuda")
                runs=[r for r in json.loads((a.validation/"INPUTS.json").read_text())["runs"] if r["config"]=="720p-600k-420"]
                mask=np.zeros((1080,1920),dtype=bool)
                for iy in range(1,a.rows):mask[iy*1080//a.rows-4:iy*1080//a.rows+4,:]=True
                for ix in range(1,a.cols):mask[:,ix*1920//a.cols-4:ix*1920//a.cols+4]=True
                records=[]
                for run in runs:
                    for s in run["samples"]:
                        for k in ("lr","hr"):assert digest(a.validation/s[k])==s[k+"_sha256"]
                        low=np.asarray(Image.open(a.validation/s["lr"]).convert("RGB"));hr=np.asarray(Image.open(a.validation/s["hr"]).convert("RGB"))
                        t=torch.from_numpy(np.ascontiguousarray(low.transpose(2,0,1))).unsqueeze(0).to("cuda").float()/255
                        fp32=forward(model,"edsr",t)
                        with amp():baseline=forward(model,"edsr",t)
                        if reference_engine:
                            reference_engine.input.copy_(t);full=reference_engine.run()
                        x.copy_(t);graph.replay()
                        assert torch.isfinite(output).all().item()
                        images={}
                        tensors=[("fp32",fp32),("baseline_amp",baseline),("optimized",output)]
                        if reference_engine:tensors.append(("full_trt",full))
                        for k,v in tensors:
                            rgb=v[0].float().mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
                            images[k]=cv2.resize(rgb,(1920,1080),interpolation=cv2.INTER_AREA)
                        row=dict(clip=run["clip"],frame=s["frame"],psnr={})
                        for k,v in images.items():row["psnr"][k]=float(10*np.log10(255**2/np.mean((v.astype(np.float64)-hr.astype(np.float64))**2)))
                        for k in (["fp32","baseline_amp","full_trt"] if reference_engine else ["fp32","baseline_amp"]):
                            diff=np.abs(images["optimized"].astype(np.int16)-images[k].astype(np.int16))
                            row["vs_"+k]=dict(max_rgb=int(diff.max()),mean_abs_rgb=float(diff.mean()),
                                seam_max_rgb=int(diff[mask].max()) if mask.any() else None,
                                seam_mean_abs_rgb=float(diff[mask].mean()) if mask.any() else None)
                        records.append(row)
                result["quality"]=dict(samples=len(records),mean_psnr={k:float(np.mean([r["psnr"][k] for r in records])) for k in images},records=records)
            result.update(complete=True,finished=time.time(),gpu_end=guard());save(dest,result)
    except BaseException as exc:
        result.update(error=repr(exc),failed=time.time());save(dest,result);raise


if __name__=="__main__":main()
