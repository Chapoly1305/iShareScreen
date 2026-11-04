"""Cache-oriented EDSR execution: spatial strips within groups of residual blocks.

Unlike whole-model tiling, a four-block group needs only eight LR halo pixels.
FP16 HWC8 intermediate buffers keep copies and the entire execution on CUDA.
Each group completes the full frame before the next group. One CUDA Graph
contains every crop/copy/convolution/stitch; there is no reuse of frame outputs.
"""
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

from benchmarks.super_resolution.edsr_folded_tail import FoldedTail
from benchmarks.super_resolution.profile_inference_optimization import adapter, amp, digest, forward, guard, save, stats


class Tail(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.final=model.body[-1];self.tail=FoldedTail(model.tail);self.add_mean=model.add_mean

    def forward(self,activation,skip):
        return self.add_mean(self.tail(self.final(activation)+skip))/255


def strip_coordinates(rows,halo):
    assert rows>0 and 720%rows==0
    core=720//rows;tile=core+min(rows-1,2)*halo
    assert tile<=720
    return tile,[(dy,min(max(dy-halo,0),720-tile),core) for dy in range(0,720,core)]


def convolution(net,x,module,name):
    layer=net.add_convolution_nd(x,module.out_channels,module.kernel_size,
        np.ascontiguousarray(module.weight.detach().numpy()),np.ascontiguousarray(module.bias.detach().numpy()))
    layer.name=name;layer.padding_nd=module.padding;layer.stride_nd=module.stride
    return layer.get_output(0)


def compile_engine(builder,net,path,cache):
    config=builder.create_builder_config();config.set_flag(trt.BuilderFlag.FP16);config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level=5;config.avg_timing_iterations=8
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,4<<30)
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    timing=config.create_timing_cache(cache.read_bytes() if cache.exists() else b"")
    config.set_timing_cache(timing,False)
    blob=builder.build_serialized_network(net,config)
    if blob is None:raise RuntimeError("Engine build failed: "+str(path))
    path.write_bytes(bytes(blob));cache.write_bytes(bytes(config.get_timing_cache().serialize()))


def set_half_hwc8(tensor):
    tensor.dtype=trt.float16;tensor.allowed_formats=1<<int(trt.TensorFormat.HWC8)


def build(model,root,blocks,tile):
    logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
    cache=root/"timing.cache";flag=1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    net=builder.create_network(flag)
    x=net.add_input("input",trt.float32,(1,3,720,1280))
    scale=net.add_constant((1,1,1,1),np.array([255],dtype=np.float32)).get_output(0)
    y=net.add_elementwise(x,scale,trt.ElementWiseOperation.PROD).get_output(0)
    y=convolution(net,y,model.sub_mean,"sub_mean")
    y=convolution(net,y,model.head[0],"head")
    y.name="output";net.mark_output(y);set_half_hwc8(y)
    compile_engine(builder,net,root/"head.engine",cache)
    paths=[]
    for first in range(0,16,blocks):
        net=builder.create_network(flag)
        x=net.add_input("input",trt.float16,(1,64,tile,1280));set_half_hwc8(x);y=x
        for idx in range(first,first+blocks):
            block=model.body[idx]
            z=convolution(net,y,block.body[0],f"block{idx}_conv1")
            z=net.add_activation(z,trt.ActivationType.RELU).get_output(0)
            z=convolution(net,z,block.body[2],f"block{idx}_conv2")
            if block.res_scale!=1:
                scale=net.add_constant((1,1,1,1),np.array([block.res_scale],dtype=np.float32)).get_output(0)
                z=net.add_elementwise(z,scale,trt.ElementWiseOperation.PROD).get_output(0)
            y=net.add_elementwise(y,z,trt.ElementWiseOperation.SUM).get_output(0)
        y.name="output";net.mark_output(y);set_half_hwc8(y)
        path=root/f"stage-{first:02d}.engine";compile_engine(builder,net,path,cache);paths.append(path)
        print("built",path.name,flush=True)
    # Trace a small dynamic graph; no CPU full-resolution convolution is needed.
    onnx=root/"tail.onnx"
    with torch.inference_mode():
        torch.onnx.export(Tail(copy.deepcopy(model)),(torch.zeros(1,64,8,8),torch.zeros(1,64,8,8)),str(onnx),
            opset_version=17,input_names=["activation","skip"],output_names=["output"],dynamo=False,
            dynamic_axes={"activation":{2:"h",3:"w"},"skip":{2:"h",3:"w"},"output":{2:"oh",3:"ow"}})
    net=builder.create_network(flag);parser=trt.OnnxParser(net,logger)
    if not parser.parse(onnx.read_bytes()):raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    for i in range(net.num_inputs):
        x=net.get_input(i);x.shape=(1,64,720,1280);set_half_hwc8(x)
    compile_engine(builder,net,root/"tail.engine",cache)
    return paths


class Engine:
    def __init__(self,path):
        self.logger=trt.Logger(trt.Logger.WARNING);self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(path.read_bytes());self.context=self.engine.create_execution_context()
        self.tensors={}
        for i in range(self.engine.num_io_tensors):
            name=self.engine.get_tensor_name(i);shape=tuple(self.engine.get_tensor_shape(name))
            dtype={trt.float16:torch.float16,trt.float32:torch.float32}[self.engine.get_tensor_dtype(name)]
            fmt=self.engine.get_tensor_format(name)
            assert fmt in (trt.TensorFormat.HWC8,trt.TensorFormat.LINEAR),(name,fmt)
            memory_format=torch.channels_last if fmt==trt.TensorFormat.HWC8 else torch.contiguous_format
            self.tensors[name]=torch.empty(shape,dtype=dtype,device="cuda",memory_format=memory_format)
            self.context.set_tensor_address(name,self.tensors[name].data_ptr())

    def bind(self,name,tensor):
        assert tensor.shape==self.tensors[name].shape and tensor.dtype==self.tensors[name].dtype
        assert tensor.stride()==self.tensors[name].stride()
        self.tensors[name]=tensor;self.context.set_tensor_address(name,tensor.data_ptr())

    def run(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):raise RuntimeError("TensorRT failed")
        return self.tensors["output"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights",type=Path,required=True);p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--blocks",type=int,default=4)
    p.add_argument("--rows",type=int,default=8);p.add_argument("--samples",type=int,default=200)
    p.add_argument("--quality",action="store_true");p.add_argument("--reuse",action="store_true")
    a=p.parse_args();assert 16%a.blocks==0
    a.output.mkdir(parents=True,exist_ok=True);dest=a.output/("CONFIRM.json" if a.reuse else "RESULTS.json")
    if dest.exists():raise FileExistsError(dest)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    item=next(r for r in json.loads((a.weights/"PLAN.json").read_text())["models"] if r["model"]=="edsr")
    checkpoint=a.weights/"checkpoints/edsr.pt";assert digest(checkpoint)==item["sha256"]
    tile,coords=strip_coordinates(a.rows,2*a.blocks)
    result=dict(complete=False,started=time.time(),pid=os.getpid(),gpu_start=guard(),checkpoint=item,
        command=os.sys.argv,script_sha256=digest(Path(__file__)),fold_source_sha256=digest(Path(__file__).with_name("edsr_folded_tail.py")),
        blocks_per_group=a.blocks,rows=a.rows,halo=2*a.blocks,tile_height=tile,coordinates=coords,
        convolution_pixel_ratio=a.rows*tile/720,
        scope="GPU resident full 720p RGB -> native1440p RGB. All feature crops/copies/stitching and all residual blocks execute each frame; FP16 HWC8 stage IO. No decode/display/host pixel transfer.")
    save(dest,result)
    try:
        model=adapter("edsr","cpu").model.eval().requires_grad_(False)
        torch.nn.Module.load_state_dict(model,torch.load(checkpoint,map_location="cpu",weights_only=True)["state_dict"],strict=True)
        paths=[a.output/f"stage-{i:02d}.engine" for i in range(0,16,a.blocks)]
        t0=time.time()
        if not a.reuse:build(model,a.output,a.blocks,tile)
        result["build_s"]=time.time()-t0
        head=Engine(a.output/"head.engine");stages=[Engine(p) for p in paths];tail=Engine(a.output/"tail.engine")
        result["engine_sha256"]={p.name:digest(p) for p in [a.output/"head.engine",*paths,a.output/"tail.engine"]}
        for i,e in enumerate([head,*stages,tail]):
            (a.output/f"layers-{i}.json").write_text(e.engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
        frame_a=torch.empty_like(head.tensors["output"]);frame_b=torch.empty_like(frame_a)
        final=frame_a if len(stages)%2 else frame_b
        tail.bind("activation",final);tail.bind("skip",head.tensors["output"])

        def predict():
            current=head.run()
            for idx,stage in enumerate(stages):
                target=frame_a if idx%2==0 else frame_b
                for dy,sy,core in coords:
                    stage.tensors["input"].copy_(current[:,:,sy:sy+tile,:])
                    out=stage.run()
                    target[:,:,dy:dy+core,:].copy_(out[:,:,dy-sy:dy-sy+core,:])
                current=target
            return tail.run()

        with torch.inference_mode():
            head.tensors["input"].copy_(torch.linspace(0,1,720*1280*3,device="cuda").reshape(1,3,720,1280))
            for _ in range(50):predict()
            stream.synchronize();eager=tail.tensors["output"].clone()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):predict()
            graph.replay();stream.synchronize();assert torch.equal(eager,tail.tensors["output"])
            for _ in range(50):graph.replay()
            stream.synchronize();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            device=[];wall=[]
            for _ in range(a.samples):
                t0=time.perf_counter();begin.record();graph.replay();end.record();end.synchronize()
                device.append(begin.elapsed_time(end));wall.append((time.perf_counter()-t0)*1000)
            result.update(graph=stats(device),wall=stats(wall),graph_matches_eager=True);save(dest,result)
            print("graph p95",result["graph"]["p95_ms"],flush=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(5):graph.replay()
                stream.synchronize()
            prof.export_chrome_trace(str(a.output/("confirm-trace.json" if a.reuse else "trace.json")))
            if a.quality:
                model.to("cuda");records=[]
                runs=[r for r in json.loads((a.validation/"INPUTS.json").read_text())["runs"] if r["config"]=="720p-600k-420"]
                seam=np.zeros((1080,1920),dtype=bool)
                for row in range(1,a.rows):seam[row*1080//a.rows-4:row*1080//a.rows+4,:]=True
                for run in runs:
                    for sample in run["samples"]:
                        for k in ("lr","hr"):assert digest(a.validation/sample[k])==sample[k+"_sha256"]
                        rgb=np.asarray(Image.open(a.validation/sample["lr"]).convert("RGB")).copy()
                        hr=np.asarray(Image.open(a.validation/sample["hr"]).convert("RGB"))
                        x=torch.from_numpy(rgb).to("cuda").permute(2,0,1).unsqueeze(0).float()/255
                        fp32=forward(model,"edsr",x)
                        with amp():baseline=forward(model,"edsr",x)
                        head.tensors["input"].copy_(x);graph.replay()
                        out=tail.tensors["output"];assert torch.isfinite(out).all().item()
                        images={}
                        for k,v in [("fp32",fp32),("baseline_amp",baseline),("optimized",out)]:
                            native=v[0].float().mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
                            images[k]=cv2.resize(native,(1920,1080),interpolation=cv2.INTER_AREA)
                        row=dict(clip=run["clip"],frame=sample["frame"],psnr={})
                        for k,img in images.items():row["psnr"][k]=float(10*np.log10(255**2/np.mean((img.astype(np.float64)-hr)**2)))
                        for k in ("fp32","baseline_amp"):
                            delta=np.abs(images["optimized"].astype(np.int16)-images[k].astype(np.int16))
                            row["vs_"+k]=dict(max_rgb=int(delta.max()),mean_abs_rgb=float(delta.mean()),seam_max_rgb=int(delta[seam].max()) if seam.any() else None)
                        records.append(row)
                result["quality"]=dict(samples=len(records),mean_psnr={k:float(np.mean([r["psnr"][k] for r in records])) for k in images},records=records)
            result.update(complete=True,finished=time.time(),gpu_end=guard());save(dest,result)
    except BaseException as exc:
        result.update(error=repr(exc),failed=time.time());save(dest,result);raise


if __name__=="__main__":main()
