"""Same-engine TensorRT activation memory: ordinary VMM versus compression.

Compression is requested and checked on the returned CUDA allocation. All
pixels remain on GPU during timing. CPU validation-image loading is setup only.
Both modes use the identical engine/tactics, input buffer, and CUDA graph path.
"""
import argparse
import ctypes as c
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import tensorrt as trt
import torch

from benchmarks.super_resolution.profile_inference_optimization import digest,guard,save,stats


class Allocation:
    def __init__(self,library,size,compressed,device=0):
        self.lib=c.CDLL(str(library.resolve()))
        u64=c.c_uint64;p64=c.POINTER(u64)
        self.lib.iss_vmm_alloc.argtypes=[u64,c.c_int,c.c_int,p64,p64,p64,c.POINTER(c.c_int)]
        self.lib.iss_vmm_alloc.restype=c.c_int
        self.lib.iss_vmm_free.argtypes=[u64,u64,u64];self.lib.iss_vmm_free.restype=c.c_int
        pointer,handle,allocated,actual=u64(),u64(),u64(),c.c_int()
        status=self.lib.iss_vmm_alloc(size,device,int(compressed),c.byref(pointer),c.byref(handle),c.byref(allocated),c.byref(actual))
        if status:raise RuntimeError('cuMem allocation failed: '+str(status))
        self.pointer,self.handle,self.size=pointer.value,handle.value,allocated.value
        self.actual_compression=actual.value;self.closed=False
        if actual.value!=int(compressed):
            self.close();raise RuntimeError('Driver did not grant requested compression: '+str(actual.value))
        assert self.pointer%256==0 and self.size>=size

    def close(self):
        if not self.closed:
            status=self.lib.iss_vmm_free(self.pointer,self.handle,self.size)
            if status:raise RuntimeError('cuMem release failed: '+str(status))
            self.closed=True


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('engine','library','validation','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--samples',type=int,default=300)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    assert digest(a.engine)=='677cd5444c60ab68e6460b52f7dcc39efdf6654d189884458ae31f2ef7f67633'
    r=dict(complete=False,started=time.time(),gpu=guard(),scope=__doc__,source_sha256=digest(Path(__file__)),
        library_sha256=digest(a.library),engine_sha256=digest(a.engine),trt=trt.__version__,records=[],validation=[])
    save(a.output,r);torch.set_num_threads(1);torch.manual_seed(1729)
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    x=torch.empty((1,3,720,1280),device='cuda',dtype=torch.float32)
    logger=trt.Logger(trt.Logger.WARNING);runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(a.engine.read_bytes())
    size=engine.device_memory_size_v2;r['required_context_bytes']=size
    modes={}
    try:
        for name,compressed in (('ordinary',False),('compressed',True)):
            allocation=Allocation(a.library,size,compressed)
            context=engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            context.set_device_memory(allocation.pointer,allocation.size)
            output=torch.empty((1,3,1440,2560),device='cuda',dtype=torch.float32)
            context.set_tensor_address('input',x.data_ptr());context.set_tensor_address('output',output.data_ptr())
            x.zero_()
            for _ in range(5):assert context.execute_async_v3(stream.cuda_stream)
            stream.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):assert context.execute_async_v3(stream.cuda_stream)
            modes[name]=dict(allocation=allocation,context=context,output=output,graph=graph)
        r['allocations']={k:dict(bytes=v['allocation'].size,actual_compression=v['allocation'].actual_compression,pointer_alignment_256=v['allocation'].pointer%256==0)for k,v in modes.items()}
        runs=[v for v in json.loads((a.validation/'INPUTS.json').read_text())['runs']if v['config']=='720p-600k-420']
        selected=[(runs[i],runs[i]['samples'][1])for i in (0,len(runs)//2,len(runs)-1)]
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        cases=[('ramp',None,None),('random_noise',None,None)]+[(run['clip'],run,s)for run,s in selected]
        for label,run,sample in cases:
            if label=='ramp':x.copy_(torch.linspace(0,1,x.numel(),device='cuda').reshape(x.shape))
            elif label=='random_noise':x.uniform_(0,1)
            else:
                path=a.validation/sample['lr'];assert digest(path)==sample['lr_sha256']
                rgb=np.asarray(Image.open(path).convert('RGB')).copy()
                x.copy_(torch.from_numpy(rgb).cuda().permute(2,0,1).unsqueeze(0).float()/255)
            row=dict(input=label,frame=sample['frame']if sample else None,timings={})
            for name,v in modes.items():
                for _ in range(50):v['graph'].replay()
                values=[]
                for _ in range(a.samples):
                    begin.record();v['graph'].replay();end.record();end.synchronize();values.append(begin.elapsed_time(end))
                row['timings'][name]=stats(values)
            assert torch.equal(modes['ordinary']['output'],modes['compressed']['output'])
            row['bit_exact']=True;r['records'].append(row);save(a.output,r)
            print(label,{k:v['p95_ms']for k,v in row['timings'].items()},flush=True)
        # Independent from timed samples: all 72 frozen development frames must
        # have bit-identical float outputs, before RGB rounding or resizing.
        for run in runs:
            for sample in run['samples']:
                path=a.validation/sample['lr'];assert digest(path)==sample['lr_sha256']
                rgb=np.asarray(Image.open(path).convert('RGB')).copy()
                x.copy_(torch.from_numpy(rgb).cuda().permute(2,0,1).unsqueeze(0).float()/255)
                for v in modes.values():v['graph'].replay()
                stream.synchronize()
                assert all(torch.isfinite(v['output']).all().item()for v in modes.values())
                assert torch.equal(modes['ordinary']['output'],modes['compressed']['output'])
                r['validation'].append(dict(clip=run['clip'],frame=sample['frame'],float_output_bit_exact=True))
        stream.synchronize()
        for v in modes.values():
            v['graph'].reset();v['context']=None
        context=None
        for v in modes.values():v['allocation'].close()
        r.update(complete=True,finished=time.time(),gpu_end=guard());save(a.output,r)
    except BaseException as exc:
        stream.synchronize()
        r.update(error=repr(exc),failed=time.time());save(a.output,r);raise


if __name__=='__main__':main()
