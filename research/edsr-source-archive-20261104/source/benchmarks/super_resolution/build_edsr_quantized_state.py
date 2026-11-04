"""Quality screen for recovering FP16 residual state from existing INT8 codes.

Keeps the accepted dual-output plugin unchanged; adds explicit dequantization
only on selected residual skips. This intentionally does not measure the
performance of a future single-output INT8-state kernel. No retraining.
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

from benchmarks.super_resolution import edsr_entry_plugin, edsr_int8_plugin, edsr_final_plugin
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save


def transform(model, blocks, correction="none"):
    data = {x.name: numpy_helper.to_array(x).copy() for x in model.graph.initializer}
    selected = {f"dual_conv_block_{i}": i for i in blocks}
    replacements = {}
    nodes = []
    for source in model.graph.node:
        node = copy.deepcopy(source)
        # The frozen accepted ONNX predates fixed-shape plugin attributes.
        # Current plugin implementations require explicit dimensions.
        if node.domain == "iss":
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            for name, value in (("height", 720), ("width", 1280)):
                if name in attrs:
                    assert attrs[name] == value
                else:
                    node.attribute.append(helper.make_attribute(name, value))
        if node.domain == "iss" and node.op_type in ("edsr_dual", "edsr_final"):
            if node.input[2] in replacements:
                node.input[2] = replacements[node.input[2]]
        nodes.append(node)
        if node.name in selected:
            assert node.op_type == "edsr_dual" and node.domain == "iss"
            index = selected[node.name]
            factor = data[node.input[5]]
            assert factor.shape == (64,) and np.isfinite(factor).all() and (factor > 0).all()
            scale = (np.float32(1) / factor).reshape(1, 64, 1, 1)
            prefix = f"quantized_state_{index}"
            model.graph.initializer.append(numpy_helper.from_array(scale, prefix + "_scale"))
            nodes.append(helper.make_node("Cast", [node.output[1]], [prefix + "_codes"],
                                          to=TensorProto.FLOAT, name=prefix + "_float"))
            codes = prefix + "_codes"
            if correction == "nibble_escape":
                constants = dict(factor=factor.reshape(1, 64, 1, 1), fourteen=14., inverse_fourteen=1./14.,
                                 low=-7., high=7., escape_bound=.5)
                for label, value in constants.items():
                    model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), prefix + "_" + label))
                # Physical representation: 0..14 encode signed correction -7..7;
                # code 15 escapes to the original FP16 value when INT8 clips.
                nodes.extend([
                    helper.make_node("Cast", [node.output[0]], [prefix + "_state_float"], to=TensorProto.FLOAT),
                    helper.make_node("Mul", [prefix + "_state_float", prefix + "_factor"], [prefix + "_position"]),
                    helper.make_node("Sub", [prefix + "_position", codes], [prefix + "_error"]),
                    helper.make_node("Mul", [prefix + "_error", prefix + "_fourteen"], [prefix + "_fine_position"]),
                    helper.make_node("Round", [prefix + "_fine_position"], [prefix + "_rounded"]),
                    helper.make_node("Clip", [prefix + "_rounded", prefix + "_low", prefix + "_high"], [prefix + "_signed_nibble"]),
                    helper.make_node("Mul", [prefix + "_signed_nibble", prefix + "_inverse_fourteen"], [prefix + "_correction"]),
                    helper.make_node("Add", [codes, prefix + "_correction"], [prefix + "_corrected_codes"]),
                    helper.make_node("Abs", [prefix + "_error"], [prefix + "_abs_error"]),
                    helper.make_node("Greater", [prefix + "_abs_error", prefix + "_escape_bound"], [prefix + "_escape"]),
                ])
                codes = prefix + "_corrected_codes"
            nodes.extend([
                helper.make_node("Mul", [codes, prefix + "_scale"], [prefix + "_dequantized"], name=prefix + "_dequant"),
                helper.make_node("Cast", [prefix + "_dequantized"], [prefix + "_approx_half"], to=TensorProto.FLOAT16, name=prefix + "_half_cast")])
            if correction == "nibble_escape":
                nodes.append(helper.make_node("Where", [prefix + "_escape", node.output[0], prefix + "_approx_half"], [prefix + "_half"]))
            else:
                nodes.append(helper.make_node("Identity", [prefix + "_approx_half"], [prefix + "_half"]))
            replacements[node.output[0]] = prefix + "_half"
    assert len(replacements) == len(blocks)
    for old, new in replacements.items():
        consumers = [n for n in nodes if new in n.input]
        assert len(consumers) == 1 and consumers[0].domain == "iss"
        if correction == "none":
            assert not any(old in n.input for n in nodes), old
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return replacements


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--blocks", default=",".join(map(str, range(14))))
    p.add_argument("--correction", choices=("none", "nibble_escape"), default="none")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    dest = a.output / "BUILD.json"
    if dest.exists():
        raise FileExistsError(dest)
    assert digest(a.onnx) == "b80f2afdbd8b32404b1aa713c2d403f8d71777a7237aa48d1170f5f20962e26c"
    blocks = list(map(int, a.blocks.split(",")))
    assert blocks and len(set(blocks)) == len(blocks) and all(0 <= b < 14 for b in blocks)
    r = dict(complete=False, started=time.time(), gpu=guard(), scope=__doc__, trt=trt.__version__,
             source_sha256=digest(Path(__file__)), original_onnx_sha256=digest(a.onnx), blocks=blocks, correction=a.correction,
             plugin_sha256={m.__name__: digest(Path(m.__file__)) for m in (edsr_entry_plugin, edsr_int8_plugin, edsr_final_plugin)})
    save(dest, r)
    try:
        model = onnx.load(a.onnx)
        r["replacements"] = transform(model, blocks, a.correction)
        path = a.output / "edsr.onnx"
        onnx.save(model, path)
        r["onnx_sha256"] = digest(path)
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS))
        parser = trt.OnnxParser(net, logger)
        if not parser.parse(model.SerializeToString()):
            raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
        assert net.num_inputs == 1
        net.get_input(0).shape = (1, 3, 720, 1280)
        r["input_shape"] = list(net.get_input(0).shape)
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.builder_optimization_level = 5
        config.avg_timing_iterations = 8
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        blob = builder.build_serialized_network(net, config)
        if blob is None:
            raise RuntimeError("Quantized residual-state build failed")
        path = a.output / "edsr.engine"
        path.write_bytes(bytes(blob))
        r.update(complete=True, finished=time.time(), engine_sha256=digest(path), gpu_end=guard())
        save(dest, r)
    except BaseException as exc:
        r.update(error=repr(exc), failed=time.time())
        save(dest, r)
        raise


if __name__ == "__main__":
    main()
