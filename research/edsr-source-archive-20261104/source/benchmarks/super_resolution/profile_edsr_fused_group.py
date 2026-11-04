"""One four-block fused INT8 group: full-frame versus cache-local strips.

This is a component experiment, not an EDSR or GPU-decoding latency claim.
Every run processes the full 720p feature map. Strip buffers bind directly to
GPU views; internal entry conversion and output cropping remain timed.
"""
import argparse
import copy
from pathlib import Path
import time

import onnx
from onnx import helper,TensorProto
import tensorrt as trt
import torch

from benchmarks.super_resolution import edsr_entry_plugin,edsr_int8_plugin
from benchmarks.super_resolution.profile_edsr_fp8_strips import extract,direct_view
from benchmarks.super_resolution.profile_edsr_segmented import Engine,strip_coordinates
from benchmarks.super_resolution.profile_inference_optimization import digest,guard,save,stats


def compile_group(base,path,height,core,crop,cache):
    model=copy.deepcopy(base)
    for n in model.graph.node:
        if n.domain=='iss':
            keep=[copy.deepcopy(a)for a in n.attribute if a.name not in ('height','width')]
            del n.attribute[:];n.attribute.extend(keep)
            n.attribute.extend([helper.make_attribute('height',height),helper.make_attribute('width',1280)])
    source='/model/head/head.0/Conv_output_0';end='/model/body/body.3/Add_output_0'
    model=extract(model,{source:('input',TensorProto.FLOAT16,[1,64,height,1280])},end,[1,64,core,1280],crop)
    onnx.save(model,path.with_suffix('.onnx'))
    logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
    net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS))
    parser=trt.OnnxParser(net,logger)
    if not parser.parse(model.SerializeToString()):raise RuntimeError([str(parser.get_error(i))for i in range(parser.num_errors)])
    for tensor in (net.get_input(0),net.get_output(0)):
        assert tensor.dtype==trt.float16
        tensor.allowed_formats=1<<int(trt.TensorFormat.HWC8)
    config=builder.create_builder_config();config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level=5;config.avg_timing_iterations=8
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,4<<30)
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    timing=config.create_timing_cache(cache.read_bytes()if cache.exists()else b'');config.set_timing_cache(timing,False)
    blob=builder.build_serialized_network(net,config)
    if blob is None:raise RuntimeError('Failed group build: '+str(path))
    path.write_bytes(bytes(blob));cache.write_bytes(bytes(config.get_timing_cache().serialize()))
    print('built',path.name,flush=True)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rows',type=int,default=8);p.add_argument('--samples',type=int,default=1000)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);dest=a.output/'RESULTS.json'
    if dest.exists():raise FileExistsError(dest)
    assert digest(a.onnx)=='b80f2afdbd8b32404b1aa713c2d403f8d71777a7237aa48d1170f5f20962e26c'
    tile,coords=strip_coordinates(a.rows,8);core=720//a.rows
    r=dict(complete=False,scope=__doc__,started=time.time(),gpu=guard(),source_onnx_sha256=digest(a.onnx),
           source_sha256=digest(Path(__file__)),plugin_sha256={x.__name__:digest(Path(x.__file__))for x in (edsr_entry_plugin,edsr_int8_plugin)},
           rows=a.rows,tile_height=tile,core_height=core,halo=8,pixel_work_ratio=tile*a.rows/720,coordinates=coords)
    save(dest,r)
    try:
        torch.set_num_threads(1);torch.manual_seed(1729);stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
        base=onnx.load(a.onnx);cache=a.output/'timing.cache'
        fullpath=a.output/'full.engine';compile_group(base,fullpath,720,720,None,cache)
        paths={}
        for crop in sorted({dy-sy for dy,sy,_ in coords}):
            path=a.output/f'crop-{crop}.engine';compile_group(base,path,tile,core,crop,cache);paths[crop]=path
        full=Engine(fullpath);engines={k:Engine(p)for k,p in paths.items()}
        source=torch.randn_like(full.tensors['input'])*20;target=torch.empty_like(full.tensors['output'])
        full.bind('input',source)
        calls=[(engines[dy-sy],direct_view(source,sy,tile),direct_view(target,dy,core))for dy,sy,_ in coords]
        def strips():
            for engine,inp,out in calls:
                engine.bind('input',inp);engine.bind('output',out);engine.run()
            return target
        graph_runners={};outputs={}
        for label,run in (('full',full.run),('strips',strips)):
            for _ in range(10):out=run()
            stream.synchronize();expected=out.clone();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):run()
            graph.replay();stream.synchronize();assert torch.equal(expected,out)
            graph_runners[label]=graph;outputs[label]=out
        r['group_comparison']={}
        for kind in ('random','constant','corner'):
            if kind=='constant':source.fill_(1)
            if kind=='corner':source.zero_();source[:,:,0,0]=1;source[:,:,-1,-1]=-1
            for graph in graph_runners.values():graph.replay()
            stream.synchronize()
            assert all(torch.isfinite(x).all().item()for x in outputs.values())
            delta=(outputs['full'].float()-outputs['strips'].float()).abs()
            r['group_comparison'][kind]=dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),
                reference_abs_max=float(outputs['full'].abs().max()),bit_exact=bool(torch.equal(outputs['full'],outputs['strips'])))
        source.normal_(mean=0,std=20);r['timings']={}
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for label,graph in graph_runners.items():
            for _ in range(50):graph.replay()
            values=[]
            for _ in range(a.samples):
                begin.record();graph.replay();end.record();end.synchronize();values.append(begin.elapsed_time(end))
            r['timings'][label]=stats(values);print(label,r['timings'][label]['p95_ms'],flush=True)
        r['engine_sha256']={path.name:digest(path)for path in [fullpath,*paths.values()]}
        r.update(complete=True,finished=time.time(),gpu_end=guard())
        save(dest,r)
    except BaseException as exc:r.update(error=repr(exc),failed=time.time());save(dest,r);raise


if __name__=='__main__':main()
