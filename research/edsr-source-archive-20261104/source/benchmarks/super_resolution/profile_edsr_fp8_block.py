"""Isolate an FP8 residual block to test fusion of its FP16 residual addition."""
import argparse
import copy
import json
from pathlib import Path
import time

import onnx
from onnx import helper, TensorProto
import tensorrt as trt
import torch

from benchmarks.super_resolution.profile_edsr_segmented import Engine
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save, stats


def extract_block(model, index, height=720):
    start = "/model/head/head.0/Conv_output_0" if index == 0 else f"/model/body/body.{index-1}/Add_output_0"
    end = f"/model/body/body.{index}/Add_output_0"
    needed = {end}; nodes = []
    for node in reversed(model.graph.node):
        if needed.intersection(node.output):
            if start in node.output:
                continue
            nodes.append(copy.deepcopy(node)); needed.update(node.input)
    nodes.reverse()
    for node in nodes:
        for k, name in enumerate(node.input):
            if name == start: node.input[k] = "input"
        for k, name in enumerate(node.output):
            if name == end: node.output[k] = "output"
    initializers = [copy.deepcopy(t) for t in model.graph.initializer if t.name in needed]
    shape = [1, 64, height, 1280]
    graph = helper.make_graph(nodes, f"fp8_block_{index}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, shape)], initializers)
    out = helper.make_model(graph, opset_imports=model.opset_import)
    onnx.checker.check_model(out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--height", type=int, default=720)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    dest = a.output / "BLOCK.json"
    if dest.exists(): raise FileExistsError(dest)
    result = dict(complete=False, gpu=guard(), started=time.time(), scope=__doc__, input_height=a.height, source_onnx_sha256=digest(a.onnx), script_sha256=digest(Path(__file__)))
    save(dest, result)
    try:
        model = extract_block(onnx.load(a.onnx), 0, a.height)
        onnx.save(model, a.output / "block.onnx")
        logger = trt.Logger(trt.Logger.WARNING); builder = trt.Builder(logger)
        net = builder.create_network(0); parser = trt.OnnxParser(net, logger)
        if not parser.parse(model.SerializeToString()):
            raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
        net.get_input(0).allowed_formats = 1 << int(trt.TensorFormat.HWC8)
        net.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.HWC8)
        config = builder.create_builder_config(); config.clear_flag(trt.BuilderFlag.TF32)
        config.builder_optimization_level = 5; config.avg_timing_iterations = 8
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        blob = builder.build_serialized_network(net, config)
        if blob is None: raise RuntimeError("Block build failed")
        path = a.output / "block.engine"; path.write_bytes(bytes(blob))
        stream = torch.cuda.Stream(); torch.cuda.set_stream(stream)
        engine = Engine(path)
        (a.output / "layers.json").write_text(engine.engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
        engine.tensors["input"].normal_()
        for _ in range(50): engine.run()
        stream.synchronize(); graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream): engine.run()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(200):
            begin.record(); graph.replay(); end.record(); end.synchronize()
            times.append(begin.elapsed_time(end))
        result.update(complete=True, finished=time.time(), engine_sha256=digest(path), latency=stats(times))
        save(dest, result); print("block p95", result["latency"]["p95_ms"], flush=True)
    except BaseException as exc:
        result.update(error=repr(exc)); save(dest, result); raise


if __name__ == "__main__": main()
