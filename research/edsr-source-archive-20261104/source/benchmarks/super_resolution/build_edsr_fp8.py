"""Explicit FP8 Q/DQ on EDSR residual convolutions, keeping FP16 skips/tail."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import tensorrt as trt

from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fp32-scales", action="store_true", help="Keep calibration/weight scales FP32 with explicit surrounding casts")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    dest = a.output / "BUILD.json"
    if dest.exists():
        raise FileExistsError(dest)
    calibration = json.loads(a.calibration.read_text())
    assert calibration["complete"] and calibration["calibration"]["split"] == "train"
    assert calibration["modes"]["fp8"]["passed"]
    result = dict(complete=False, started=time.time(), gpu=guard(), trt=trt.__version__,
        source_onnx_sha256=digest(a.onnx), calibration_sha256=digest(a.calibration),
        script_sha256=digest(Path(__file__)), scope=__doc__,
        scale_precision="FP32 scales with explicit FP32 Q/DQ and FP16 conv operands" if a.fp32_scales else "FP16 scales to match FP16 ONNX activation type; actual engine quality must be independently validated")
    save(dest, result)
    try:
        model = onnx.load(a.onnx)
        for opset in model.opset_import:
            if opset.domain == "":
                assert opset.version == 17
                opset.version = 19
        weights = {w.name: numpy_helper.to_array(w) for w in model.graph.initializer}
        maxima = calibration["calibration"]["maxima"]
        nodes = []; quantized = []
        for node in model.graph.node:
            name = node.input[1].removeprefix("model.").removesuffix(".weight") if node.op_type == "Conv" else ""
            if name in maxima:
                weight = weights[node.input[1]]
                dtype = np.float32 if a.fp32_scales else np.float16
                scales = (np.array(maxima[name] / 448, dtype=dtype),
                          (np.abs(weight.astype(np.float32)).max(axis=(1, 2, 3)) / 448).astype(dtype))
                assert all(np.all(s > 0) for s in scales)
                for i, scale in enumerate(scales):
                    prefix = "fp8_" + name + ("_activation" if i == 0 else "_weight")
                    model.graph.initializer.extend([
                        numpy_helper.from_array(scale, prefix + "_scale"),
                        helper.make_tensor(prefix + "_zero", TensorProto.FLOAT8E4M3FN, list(scale.shape), [0] * scale.size),
                    ])
                    attrs = {} if i == 0 else dict(axis=0)
                    operand = node.input[i]
                    if a.fp32_scales:
                        nodes.append(helper.make_node("Cast", [operand], [prefix + "_float"], name=prefix + "_cast_in", to=TensorProto.FLOAT))
                        operand = prefix + "_float"
                    nodes.extend([
                        helper.make_node("QuantizeLinear", [operand, prefix + "_scale", prefix + "_zero"], [prefix + "_q"], name=prefix + "_Q", **attrs),
                        helper.make_node("DequantizeLinear", [prefix + "_q", prefix + "_scale", prefix + "_zero"], [prefix + "_dq"], name=prefix + "_DQ", **attrs),
                    ])
                    node.input[i] = prefix + "_dq"
                    if a.fp32_scales:
                        nodes.append(helper.make_node("Cast", [prefix + "_dq"], [prefix + "_half"], name=prefix + "_cast_out", to=TensorProto.FLOAT16))
                        node.input[i] = prefix + "_half"
                quantized.append(name)
            nodes.append(node)
        assert len(quantized) == 32
        del model.graph.node[:]; model.graph.node.extend(nodes)
        onnx.checker.check_model(model)
        path = a.output / "edsr.onnx"; onnx.save(model, path)
        result.update(onnx_sha256=digest(path), quantized_layers=quantized)
        save(dest, result)
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(path.read_bytes()):
            raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
        network.get_input(0).shape = (1, 3, 720, 1280)
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.builder_optimization_level = 5
        config.avg_timing_iterations = 8
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        t0 = time.time()
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError("FP8 engine build failed")
        engine = a.output / "edsr.engine"; engine.write_bytes(bytes(blob))
        result.update(complete=True, finished=time.time(), build_s=time.time() - t0, engine_sha256=digest(engine))
        save(dest, result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        result.update(error=repr(exc), failed=time.time()); save(dest, result); raise


if __name__ == "__main__":
    main()
