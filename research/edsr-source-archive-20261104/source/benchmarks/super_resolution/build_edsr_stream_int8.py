"""Keep EDSR residual features in one train-derived channel basis.

For positive diagonal D and S, each residual block is represented as
z + D^-1 W2 S ReLU(S^-1 W1 D z + S^-1 b1) + D^-1 b2,
where z=D^-1 x. This is an exact real-arithmetic change of basis before
quantization. INT8 weight rounding is performed before the output-channel
division, then that division is folded into per-output-channel DQ scales.
This experiment requires independent floating-point and image-quality checks.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import tensorrt as trt
import torch
from torch.nn import functional as F

from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save


def check_basis(weights, common, hidden):
    """Verify the unquantized transformation, including padded borders."""
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(1729)
    d = torch.from_numpy(common.astype(np.float64))
    records = []
    for shape in ((2, 3), (7, 11)):
        for kind in ("random", "constant", "corner"):
            x = torch.randn((1, 64, *shape), dtype=torch.float64, generator=rng)
            if kind == "constant": x.fill_(1)
            if kind == "corner": x.zero_(); x[:, :, 0, 0] = 1; x[:, :, -1, -1] = -1
            z = x / d[None, :, None, None]
            for i in range(16):
                s = torch.from_numpy(hidden[i].astype(np.float64))
                w1, b1, w2, b2 = [torch.from_numpy(weights[f"model.body.{i}.body.{layer}.{param}"].astype(np.float64))
                                  for layer, param in ((0, "weight"), (0, "bias"), (2, "weight"), (2, "bias"))]
                x = x + F.conv2d(F.relu(F.conv2d(x, w1, b1, padding=1)), w2, b2, padding=1)
                z = z + F.conv2d(F.relu(F.conv2d(z, w1 * d[None, :, None, None] / s[:, None, None, None],
                    b1 / s, padding=1)), w2 * s[None, :, None, None] / d[:, None, None, None], b2 / d, padding=1)
            restored = z * d[None, :, None, None]
            torch.testing.assert_close(restored, x, atol=1e-8, rtol=1e-10)
            records.append(dict(shape=shape, kind=kind, maximum_error=float((restored - x).abs().max())))
    return dict(passed=True, dtype="float64", excludes_quantization=True, records=records)


def transform(model, maxima, alpha, margin):
    weights = {w.name: numpy_helper.to_array(w).copy() for w in model.graph.initializer}
    first = [f"body.{i}.body.0" for i in range(15)]
    activation = np.max([maxima[n] for n in first], axis=0).astype(np.float32)
    magnitude = np.max([np.abs(weights["model." + n + ".weight"].astype(np.float32)).max(axis=(0, 2, 3)) for n in first], axis=0)
    common = np.maximum(activation, 1e-12)**alpha / np.maximum(magnitude, 1e-12)**(1-alpha)
    hidden = {}
    for i in range(16):
        name = f"body.{i}.body.2"
        a = np.asarray(maxima[name], dtype=np.float32)
        w = weights["model." + name + ".weight"].astype(np.float32)
        hidden[i] = np.maximum(a, 1e-12)**alpha / np.maximum(np.abs(w).max(axis=(0, 2, 3)), 1e-12)**(1-alpha) if i < 15 else np.ones(64, dtype=np.float32)
    evidence = dict(common=common.tolist(), hidden={str(k):v.tolist() for k,v in hidden.items()},
                    math=check_basis(weights, common, hidden), margin=margin)
    nodes = []
    extra = []
    head = "/model/head/head.0/Conv_output_0"
    newhead = "balanced_head"
    for original in model.graph.node:
        node = copy.deepcopy(original)
        name = node.input[1].removeprefix("model.").removesuffix(".weight") if node.op_type == "Conv" else ""
        if name == "head.0":
            nodes.append(node)
            scaled = copy.deepcopy(node)
            scaled.name += "_balanced"
            scaled.output[0] = newhead
            for j, arr in ((1, weights[node.input[1]].astype(np.float32) / common[:, None, None, None]),
                           (2, weights[node.input[2]].astype(np.float32) / common)):
                scaled.input[j] += "_balanced"
                extra.append(numpy_helper.from_array(arr.astype(np.float16), scaled.input[j]))
            nodes.append(scaled)
            continue
        # Only the first block sees the new head. The long skip remains in its
        # original basis, avoiding any changes to the folded upsampling tail.
        if "/model/body/body.0/" in node.name:
            for j, value in enumerate(node.input):
                if value == head: node.input[j] = newhead
        if name in maxima:
            i = int(name.split(".")[1]); first_conv = name.endswith(".0")
            input_basis = common if first_conv else hidden[i]
            output_basis = hidden[i] if first_conv else common
            w = weights[original.input[1]].astype(np.float32)
            b = weights[original.input[2]].astype(np.float32)
            prefix = "stream_" + name
            extra.append(numpy_helper.from_array((b / output_basis).astype(np.float16), prefix + "_bias"))
            node.input[2] = prefix + "_bias"
            balanced = w * input_basis[None, :, None, None]
            if i < 15:
                a = np.asarray(maxima[name], dtype=np.float32)
                scale = np.array(np.max(a / input_basis) * margin / 127, dtype=np.float32)
                weight_scale = np.max(np.abs(balanced), axis=(1, 2, 3)) / 127
                # This quantization uses the same symmetric INT8 weight rule
                # as the prior candidate; the DQ output scale absorbs division.
                quant_weight = np.clip(np.rint(balanced / weight_scale[:, None, None, None]), -127, 127).astype(np.int8)
                dq_scale = (weight_scale / output_basis).astype(np.float32)
                extra.extend([numpy_helper.from_array(scale, prefix + "_as"),
                    numpy_helper.from_array(np.array(0, dtype=np.int8), prefix + "_az"),
                    numpy_helper.from_array(quant_weight, prefix + "_wq"),
                    numpy_helper.from_array(dq_scale, prefix + "_ws"),
                    numpy_helper.from_array(np.zeros(64, dtype=np.int8), prefix + "_wz")])
                nodes.extend([
                    helper.make_node("Cast", [node.input[0]], [prefix + "_af"], to=TensorProto.FLOAT, name=prefix + "_input_float"),
                    helper.make_node("QuantizeLinear", [prefix + "_af", prefix + "_as", prefix + "_az"], [prefix + "_aq"], name=prefix + "_Q"),
                    helper.make_node("DequantizeLinear", [prefix + "_aq", prefix + "_as", prefix + "_az"], [prefix + "_adq"], name=prefix + "_DQ"),
                    helper.make_node("Cast", [prefix + "_adq"], [prefix + "_ah"], to=TensorProto.FLOAT16, name=prefix + "_input_half"),
                    helper.make_node("DequantizeLinear", [prefix + "_wq", prefix + "_ws", prefix + "_wz"], [prefix + "_wdq"], axis=0, name=prefix + "_weight_DQ"),
                    helper.make_node("Cast", [prefix + "_wdq"], [prefix + "_wh"], to=TensorProto.FLOAT16, name=prefix + "_weight_half")])
                node.input[0] = prefix + "_ah"; node.input[1] = prefix + "_wh"
            else:
                extra.append(numpy_helper.from_array((balanced / output_basis[:, None, None, None]).astype(np.float16), prefix + "_weight"))
                node.input[1] = prefix + "_weight"
        elif name == "body.16":
            extra.append(numpy_helper.from_array((weights[node.input[1]].astype(np.float32) * common[None, :, None, None]).astype(np.float16), "stream_body16_weight"))
            node.input[1] = "stream_body16_weight"
        nodes.append(node)
    model.graph.initializer.extend(extra)
    del model.graph.node[:]; model.graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input}
    retained = [copy.deepcopy(w) for w in model.graph.initializer if w.name in used]
    del model.graph.initializer[:]; model.graph.initializer.extend(retained)
    onnx.checker.check_model(model)
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("onnx", "calibration", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--alpha", type=float, default=.75)
    p.add_argument("--margin", type=float, default=1.)
    a = p.parse_args()
    assert a.margin >= 1
    a.output.mkdir(parents=True, exist_ok=True)
    dest = a.output / "BUILD.json"
    if dest.exists(): raise FileExistsError(dest)
    result = dict(complete=False, started=time.time(), gpu=guard(), scope=__doc__, trt=trt.__version__,
        source_onnx_sha256=digest(a.onnx), calibration_sha256=digest(a.calibration),
        script_sha256=digest(Path(__file__)), alpha=a.alpha, margin=a.margin)
    save(dest, result)
    try:
        assert digest(a.onnx) == "dcaaae0d8d988f7a5298bc2cceeef68b4854288740b7a6ef140594be96031c24"
        cal = json.loads(a.calibration.read_text())
        assert cal["complete"] and cal["calibration"]["split"] == "train"
        model = onnx.load(a.onnx)
        result["basis"] = transform(model, cal["calibration"]["maxima"], a.alpha, a.margin)
        path = a.output / "edsr.onnx"; onnx.save(model, path)
        result["onnx_sha256"] = digest(path); save(dest, result)
        logger = trt.Logger(trt.Logger.WARNING); builder = trt.Builder(logger)
        net = builder.create_network(0); parser = trt.OnnxParser(net, logger)
        if not parser.parse(model.SerializeToString()):
            raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
        net.get_input(0).shape = (1, 3, 720, 1280)
        config = builder.create_builder_config(); config.clear_flag(trt.BuilderFlag.TF32)
        config.builder_optimization_level = 5; config.avg_timing_iterations = 8
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        t0 = time.time(); blob = builder.build_serialized_network(net, config)
        if blob is None: raise RuntimeError("Stream INT8 build failed")
        path = a.output / "edsr.engine"; path.write_bytes(bytes(blob))
        result.update(complete=True, finished=time.time(), build_s=time.time()-t0, engine_sha256=digest(path))
        save(dest, result); print("built stream INT8", result["build_s"], flush=True)
    except BaseException as exc:
        result.update(error=repr(exc), failed=time.time()); save(dest, result); raise


if __name__ == "__main__":
    main()
