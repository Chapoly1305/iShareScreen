"""FP8 four-block strips with direct GPU feature views and compiled output crops.

Keeps FP16 residuals and FP32 quantization scales. Same weights; full-width
strips include the exact group receptive-field halo. No input feature copies
or Python output stitching: each engine binds directly to a view of shared
full-frame storage. Three crop positions handle the first/middle/last strips.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import onnx
from onnx import helper, numpy_helper, TensorProto
import tensorrt as trt
import torch

from benchmarks.super_resolution.profile_edsr_segmented import Engine, strip_coordinates
from benchmarks.super_resolution.profile_inference_optimization import adapter, amp, digest, forward, guard, save, stats


def extract(model, inputs, output_name, output_shape, crop=None):
    needed = {output_name}; nodes = []
    for node in reversed(model.graph.node):
        if needed.intersection(node.output):
            if set(node.output).intersection(inputs): continue
            nodes.append(copy.deepcopy(node)); needed.update(node.input)
    nodes.reverse()
    for node in nodes:
        for i,n in enumerate(node.input):
            if n in inputs: node.input[i] = inputs[n][0]
        for i,n in enumerate(node.output):
            if n == output_name: node.output[i] = "uncropped" if crop is not None else "output"
    initializers = [copy.deepcopy(t) for t in model.graph.initializer if t.name in needed]
    if crop is not None:
        for name,val in [("starts",[crop]),("ends",[crop+output_shape[2]]),("axes",[2]),("steps",[1])]:
            initializers.append(numpy_helper.from_array(np.array(val,dtype=np.int64),"crop_"+name))
        nodes.append(helper.make_node("Slice",["uncropped","crop_starts","crop_ends","crop_axes","crop_steps"],["output"],name="core_slice"))
    graph = helper.make_graph(nodes,"edsr_stage",
        [helper.make_tensor_value_info(name,dtype,shape) for name,dtype,shape in inputs.values()],
        [helper.make_tensor_value_info("output",TensorProto.FLOAT if output_shape[1]==3 else TensorProto.FLOAT16,output_shape)], initializers)
    out=helper.make_model(graph,opset_imports=model.opset_import);onnx.checker.check_model(out);return out


def compile_stage(model,path,cache):
    onnx.save(model,path.with_suffix(".onnx"))
    logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
    net=builder.create_network(0);parser=trt.OnnxParser(net,logger)
    if not parser.parse(model.SerializeToString()):raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    for tensor in [net.get_input(i) for i in range(net.num_inputs)]+[net.get_output(0)]:
        if tensor.dtype==trt.float16 and tensor.shape[1]==64:tensor.allowed_formats=1<<int(trt.TensorFormat.HWC8)
    config=builder.create_builder_config();config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level=5;config.avg_timing_iterations=8
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,4<<30)
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    timing=config.create_timing_cache(cache.read_bytes() if cache.exists() else b"")
    config.set_timing_cache(timing,False)
    blob=builder.build_serialized_network(net,config)
    if blob is None:raise RuntimeError("Failed building "+str(path))
    path.write_bytes(bytes(blob));cache.write_bytes(bytes(config.get_timing_cache().serialize()))
    print("built",path.name,flush=True)


def direct_view(tensor,row,height):
    return tensor.as_strided((1,64,height,1280),(height*1280*64,1,1280*64,64),tensor.storage_offset()+row*1280*64)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True);p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--samples",type=int,default=200);p.add_argument("--blocks",type=int,default=4)
    p.add_argument("--rows",type=int,default=8)
    p.add_argument("--precision",choices=["fp8","int8"],default="fp8",help="Validate and label the supplied ONNX quantized type")
    a=p.parse_args();assert 16%a.blocks==0
    a.output.mkdir(parents=True,exist_ok=True);dest=a.output/"RESULTS.json"
    if dest.exists():raise FileExistsError(dest)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    tile,coords=strip_coordinates(a.rows,2*a.blocks);core=720//a.rows
    result=dict(complete=False,started=time.time(),gpu=guard(),trt=trt.__version__,scope=__doc__.replace("FP8",a.precision.upper()),precision=a.precision,
        script_sha256=digest(Path(__file__)),source_onnx_sha256=digest(a.onnx),blocks=a.blocks,rows=a.rows,
        tile_height=tile,coordinates=coords,convolution_pixel_ratio=tile*a.rows/720)
    save(dest,result)
    try:
        model=onnx.load(a.onnx);cache=a.output/"timing.cache"
        types={x.name:x.data_type for x in model.graph.initializer}
        expected_type=TensorProto.FLOAT8E4M3FN if a.precision=="fp8" else TensorProto.INT8
        assert all(types[n.input[2]]==expected_type for n in model.graph.node if n.op_type=="QuantizeLinear")
        headname="/model/head/head.0/Conv_output_0"
        headpath=a.output/"head.engine"
        compile_stage(extract(model,{"input":("input",TensorProto.FLOAT,[1,3,720,1280])},headname,[1,64,720,1280]),headpath,cache)
        tailpath=a.output/"tail.engine"
        compile_stage(extract(model,{"/model/body/body.15/Add_output_0":("activation",TensorProto.FLOAT16,[1,64,720,1280]),
            headname:("skip",TensorProto.FLOAT16,[1,64,720,1280])},"output",[1,3,1440,2560]),tailpath,cache)
        paths={};crops=sorted({dy-sy for dy,sy,_ in coords})
        for first in range(0,16,a.blocks):
            start=headname if first==0 else f"/model/body/body.{first-1}/Add_output_0"
            end=f"/model/body/body.{first+a.blocks-1}/Add_output_0"
            for crop in crops:
                path=a.output/f"stage-{first:02d}-crop{crop}.engine"
                compile_stage(extract(model,{start:("input",TensorProto.FLOAT16,[1,64,tile,1280])},end,[1,64,core,1280],crop),path,cache)
                paths[first,crop]=path
        stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
        head=Engine(headpath);tail=Engine(tailpath);stages={k:Engine(p) for k,p in paths.items()}
        storage=[torch.empty_like(head.tensors["output"]) for _ in range(2)]
        calls=[];current=head.tensors["output"]
        for index,first in enumerate(range(0,16,a.blocks)):
            target=storage[index%2]
            for dy,sy,_ in coords:
                calls.append((stages[first,dy-sy],direct_view(current,sy,tile),direct_view(target,dy,core)))
            current=target
        tail.bind("activation",current);tail.bind("skip",head.tensors["output"])
        def run():
            head.run()
            for stage,inp,out in calls:
                stage.bind("input",inp);stage.bind("output",out);stage.run()
            return tail.run()
        head.tensors["input"].copy_(torch.linspace(0,1,720*1280*3,device="cuda").reshape(1,3,720,1280))
        for _ in range(20):run()
        stream.synchronize();expected=tail.tensors["output"].clone();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):run()
        graph.replay();stream.synchronize();assert torch.equal(expected,tail.tensors["output"])
        for _ in range(50):graph.replay()
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);values=[]
        for _ in range(a.samples):
            begin.record();graph.replay();end.record();end.synchronize();values.append(begin.elapsed_time(end))
        result.update(graph=stats(values),graph_matches_eager=True,
            engine_sha256={p.name:digest(p) for p in [headpath,tailpath,*paths.values()]})
        save(dest,result);print("graph p95",result["graph"]["p95_ms"],flush=True)
        reference=adapter("edsr","cpu").model.eval().requires_grad_(False)
        ckpt=a.weights/"checkpoints/edsr.pt"
        item=next(r for r in json.loads((a.weights/"PLAN.json").read_text())["models"] if r["model"]=="edsr")
        assert digest(ckpt)==item["sha256"];result["checkpoint"]=item
        torch.nn.Module.load_state_dict(reference,torch.load(ckpt,map_location="cpu",weights_only=True)["state_dict"],strict=True)
        reference.cuda();records=[]
        runs=[r for r in json.loads((a.validation/"INPUTS.json").read_text())["runs"] if r["config"]=="720p-600k-420"]
        for clip in runs:
            for s in clip["samples"]:
                for k in ("lr","hr"):assert digest(a.validation/s[k])==s[k+"_sha256"]
                low=np.asarray(Image.open(a.validation/s["lr"]).convert("RGB"));hr=np.asarray(Image.open(a.validation/s["hr"]).convert("RGB"))
                x=torch.from_numpy(np.ascontiguousarray(low.transpose(2,0,1))).unsqueeze(0).cuda().float()/255
                fp32=forward(reference,"edsr",x)
                with amp():baseline=forward(reference,"edsr",x)
                head.tensors["input"].copy_(x);graph.replay();out=tail.tensors["output"]
                assert torch.isfinite(out).all().item()
                images={}
                for k,v in [("fp32",fp32),("baseline_amp",baseline),("optimized",out)]:
                    native=v[0].float().mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
                    images[k]=cv2.resize(native,(1920,1080),interpolation=cv2.INTER_AREA)
                row=dict(clip=clip["clip"],frame=s["frame"],psnr={k:float(10*np.log10(255**2/np.mean((v.astype(np.float64)-hr.astype(np.float64))**2))) for k,v in images.items()})
                for k in ("fp32","baseline_amp"):row["vs_"+k]=dict(max_rgb=int(np.abs(images["optimized"].astype(np.int16)-images[k].astype(np.int16)).max()))
                records.append(row)
            print("quality",len(records),flush=True)
        result.update(complete=True,finished=time.time(),quality=dict(samples=len(records),records=records,
            mean_psnr={k:float(np.mean([r["psnr"][k] for r in records])) for k in images}))
        save(dest,result)
    except BaseException as exc:
        result.update(error=repr(exc),failed=time.time());save(dest,result);raise


if __name__=="__main__":main()
