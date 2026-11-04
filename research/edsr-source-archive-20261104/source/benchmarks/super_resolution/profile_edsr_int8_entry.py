"""Fuse EDSR's first INT8 quantization and two feature-layout conversions.

Two TensorRT stages and one Triton entry kernel share fixed GPU buffers. The
entry kernel produces INT8 and FP16 CHW32 views directly from the head's HWC8
output, without a full-resolution FP32 intermediate. Same calibrated INT8
ONNX graph, no change of basis or re-training. Decode/display are excluded.
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
import triton
import triton.language as tl

from benchmarks.super_resolution.profile_edsr_fp8_strips import extract
from benchmarks.super_resolution.profile_inference_optimization import adapter, amp, digest, forward, guard, save, stats


@triton.jit
def entry_kernel(X, Q, S, Factor, N: tl.constexpr, PIXELS: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = (pos // (PIXELS * 32)) * 32 + pos % 32
    pixel = pos // 32 % PIXELS
    value = tl.load(X + pixel * 64 + channel, pos < N, 0)
    factor = tl.load(Factor + channel, pos < N, 0)
    q = tl.extra.cuda.libdevice.nearbyint(value.to(tl.float32) * factor)
    q = tl.maximum(tl.minimum(q, 127), -128).to(tl.int8)
    tl.store(Q + pos, q, pos < N)
    tl.store(S + pos, value, pos < N)


def compile_stage(model, path, formats):
    onnx.save(model, path.with_suffix(".onnx"))
    logger = trt.Logger(trt.Logger.WARNING); builder = trt.Builder(logger)
    net = builder.create_network(0); parser = trt.OnnxParser(net, logger)
    if not parser.parse(model.SerializeToString()):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    for tensor in [net.get_input(i) for i in range(net.num_inputs)] + [net.get_output(0)]:
        if tensor.name in formats:
            tensor.allowed_formats = 1 << int(formats[tensor.name])
    config = builder.create_builder_config(); config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = 5; config.avg_timing_iterations = 8
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    blob = builder.build_serialized_network(net, config)
    if blob is None: raise RuntimeError("Entry fusion stage failed: " + str(path))
    path.write_bytes(bytes(blob)); print("built", path.name, flush=True)


class Engine:
    def __init__(self, path):
        self.logger = trt.Logger(trt.Logger.WARNING); self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        self.context = self.engine.create_execution_context(); self.tensors = {}; self.formats = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = {trt.float16:torch.float16, trt.float32:torch.float32, trt.int8:torch.int8}[self.engine.get_tensor_dtype(name)]
            fmt = self.engine.get_tensor_format(name)
            assert fmt in (trt.TensorFormat.CHW32, trt.TensorFormat.HWC8, trt.TensorFormat.LINEAR)
            if fmt == trt.TensorFormat.CHW32:
                assert shape[1] == 64
                physical = (shape[0], 2, shape[2], shape[3], 32)
                tensor = torch.zeros(physical, device="cuda", dtype=dtype)
            else:
                tensor = torch.empty(shape, device="cuda", dtype=dtype,
                    memory_format=torch.channels_last if fmt == trt.TensorFormat.HWC8 else torch.contiguous_format).zero_()
            self.tensors[name] = tensor; self.formats[name] = str(fmt)
            self.context.set_tensor_address(name, tensor.data_ptr())
        path.with_suffix(".layers.json").write_text(self.engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))

    def bind(self, name, tensor):
        old = self.tensors[name]
        assert old.shape == tensor.shape and old.stride() == tensor.stride() and old.dtype == tensor.dtype
        self.tensors[name] = tensor; self.context.set_tensor_address(name, tensor.data_ptr())

    def run(self):
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("Entry stage execution failed")
        return self.tensors["output"]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("onnx", "weights", "validation", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--samples", type=int, default=1000)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    dest = a.output / "RESULTS.json"
    if dest.exists(): raise FileExistsError(dest)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(complete=False, started=time.time(), scope=__doc__, gpu=guard(), trt=trt.__version__,
                  source_onnx_sha256=digest(a.onnx), script_sha256=digest(Path(__file__)), samples=a.samples)
    save(dest, result)
    try:
        model = onnx.load(a.onnx)
        head = "/model/head/head.0/Conv_output_0"
        quant = "i8_body.0.body.0_activation_q"
        weights = {v.name:numpy_helper.to_array(v).copy() for v in model.graph.initializer}
        factor = weights["i8_body.0.body.0_inverse"].reshape(64) / weights["i8_body.0.body.0_activation_scale"]
        headpath = a.output / "head.engine"; bodypath = a.output / "body.engine"
        compile_stage(extract(model, {"input":("input", TensorProto.FLOAT, [1,3,720,1280])}, head, [1,64,720,1280]),
                      headpath, {"output":trt.TensorFormat.HWC8})
        # Only the first residual addition needs a CHW32 copy of the head.
        body = copy.deepcopy(model)
        changed = 0
        for node in body.graph.node:
            if node.name == "/model/body/body.0/Add":
                for i, name in enumerate(node.input):
                    if name == head: node.input[i] = "entry_skip"; changed += 1
        assert changed == 1
        inputs = {quant:("quant", TensorProto.INT8, [1,64,720,1280]),
                  "entry_skip":("skip", TensorProto.FLOAT16, [1,64,720,1280]),
                  head:("long_skip", TensorProto.FLOAT16, [1,64,720,1280])}
        compile_stage(extract(body, inputs, "output", [1,3,1440,2560]), bodypath,
                      {"quant":trt.TensorFormat.CHW32, "skip":trt.TensorFormat.CHW32, "long_skip":trt.TensorFormat.HWC8})
        stream = torch.cuda.Stream(); torch.cuda.set_stream(stream)
        h = Engine(headpath); b = Engine(bodypath)
        b.bind("long_skip", h.tensors["output"])
        factor = torch.from_numpy(factor.copy()).cuda()
        count = 720 * 1280 * 64
        def entry():
            entry_kernel[(triton.cdiv(count, 1024),)](h.tensors["output"], b.tensors["quant"], b.tensors["skip"],
                factor, count, 720*1280, 1024, enable_fp_fusion=False)
        def run(): h.run(); entry(); return b.run()
        h.tensors["input"].copy_(torch.linspace(0,1,720*1280*3,device="cuda").reshape(1,3,720,1280))
        for _ in range(20): run()
        stream.synchronize()
        raw = h.tensors["output"]
        expected = (raw.float() * factor[None,:,None,None]).round().clamp(-128,127).to(torch.int8)
        expected = expected.reshape(1,2,32,720,1280).permute(0,1,3,4,2).contiguous()
        assert torch.equal(expected, b.tensors["quant"])
        assert torch.equal(raw.reshape(1,2,32,720,1280).permute(0,1,3,4,2), b.tensors["skip"])
        result["entry_pixel_layout_validation"] = True
        expected = b.tensors["output"].clone(); graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream): run()
        graph.replay(); stream.synchronize()
        assert torch.equal(expected, b.tensors["output"])
        result["graph_matches_eager"] = True
        begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for _ in range(50): graph.replay()
        times=[]
        for _ in range(a.samples):
            begin.record(); graph.replay(); end.record(); end.synchronize(); times.append(begin.elapsed_time(end))
        result.update(graph=stats(times), formats=dict(head=h.formats, body=b.formats),
                      engine_sha256=dict(head=digest(headpath), body=digest(bodypath)))
        save(dest,result); print("entry fusion p95", result["graph"]["p95_ms"], flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(5): graph.replay()
            stream.synchronize()
        prof.export_chrome_trace(str(a.output / "trace.json"))
        reference = adapter("edsr","cpu").model.eval().requires_grad_(False)
        ckpt = a.weights / "checkpoints/edsr.pt"
        item = next(r for r in json.loads((a.weights/"PLAN.json").read_text())["models"] if r["model"]=="edsr")
        assert digest(ckpt)==item["sha256"]; result["checkpoint"]=item
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
                h.tensors["input"].copy_(x);graph.replay();out=b.tensors["output"]
                assert torch.isfinite(out).all().item();images={}
                for k,v in [("fp32",fp32),("baseline_amp",baseline),("optimized",out)]:
                    native=v[0].float().mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
                    images[k]=cv2.resize(native,(1920,1080),interpolation=cv2.INTER_AREA)
                row=dict(clip=clip["clip"],frame=s["frame"],psnr={k:float(10*np.log10(255**2/np.mean((v.astype(np.float64)-hr.astype(np.float64))**2))) for k,v in images.items()})
                for k in ("fp32","baseline_amp"):
                    row["vs_"+k]=dict(max_rgb=int(np.abs(images["optimized"].astype(np.int16)-images[k].astype(np.int16)).max()))
                records.append(row)
            print("entry quality",len(records),flush=True)
        result.update(complete=True,finished=time.time(),gpu_end=guard(),quality=dict(samples=len(records),records=records,
            mean_psnr={k:float(np.mean([r["psnr"][k] for r in records])) for k in images}))
        save(dest,result)
    except BaseException as exc:
        result.update(error=repr(exc),failed=time.time());save(dest,result);raise


if __name__ == "__main__": main()
