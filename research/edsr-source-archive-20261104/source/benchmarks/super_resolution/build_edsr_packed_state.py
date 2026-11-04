"""Replace residual FP16 state traffic with byte corrections and masked escapes."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper
import tensorrt as trt

from benchmarks.super_resolution import edsr_entry_plugin, edsr_packed_state_plugin
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save


def transform(model, block_count=15):
    assert block_count in (15, 16)
    data = {x.name: numpy_helper.to_array(x).copy() for x in model.graph.initializer}
    replacements = {}
    nodes = []
    count = 0
    final_count = 0
    for original in model.graph.node:
        node = copy.deepcopy(original)
        if node.domain == "iss" and node.op_type == "edsr_dual":
            assert node.name == f"dual_conv_block_{count}"
            x, weight, skip, alpha, bias, factor = node.input
            half, codes = node.output
            correction = half + "_correction"
            escape = half + "_escape"
            inverse = half + "_inverse"
            f = data[factor]
            assert f.shape == (64,) and np.isfinite(f).all() and (f > 0).all()
            model.graph.initializer.append(numpy_helper.from_array(np.float32(1) / f, inverse))
            if count == 0:
                assert skip not in replacements
                op = "edsr_state_first"
                inputs = [x, weight, skip, alpha, bias, factor]
            else:
                assert skip in replacements
                op = "edsr_state_middle"
                inputs = [x, weight, *replacements[skip], alpha, bias, factor]
            replacements[half] = [codes, correction, escape, inverse]
            node = helper.make_node(op, inputs, [codes, correction, escape],
                                    name=f"state_block_{count}", domain="iss", height=720, width=1280)
            count += 1
        elif node.domain == "iss" and node.op_type == "edsr_final":
            x, weight, skip, alpha, bias = node.input
            assert count == block_count - 1 and skip in replacements
            node = helper.make_node("edsr_state_last", [x, weight, *replacements[skip], alpha, bias],
                                    list(node.output), name=f"state_block_{block_count - 1}", domain="iss", height=720, width=1280)
            final_count += 1
        elif node.domain == "iss":
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            for name, value in (("height", 720), ("width", 1280)):
                if name not in attrs:
                    node.attribute.append(helper.make_attribute(name, value))
                else:
                    assert attrs[name] == value
        nodes.append(node)
    assert count == block_count - 1 and final_count == 1
    assert not any(old in n.input for old in replacements for n in nodes)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Remove unused inverse for the terminal producer if the graph changes.
    used = {name for n in nodes for name in n.input}
    initializers = [x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    # Existing graph value_info may name outputs removed by this transformation.
    names = {v for n in nodes for v in (*n.input, *n.output)}
    value_info = [v for v in model.graph.value_info if v.name in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)
    onnx.checker.check_model(model)
    return replacements


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--editable-cache", action="store_true")
    p.add_argument("--timing-cache", type=Path)
    p.add_argument("--terminal-int8", action="store_true",
                   help="Use the separately frozen 32-convolution INT8 graph; requires fresh full quality validation.")
    p.add_argument("--restore-int8-tactic", action="store_true",
                   help="Replace the measured 64x128 INT8 tactic with the prior accepted 64x256 tactic.")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    dest = a.output / "BUILD.json"
    if dest.exists(): raise FileExistsError(dest)
    expected = ("b98d8a4a0131d78b7e1f13bdd643bb52ef1ad01be011b23d3096e52187c3b3f8" if a.terminal_int8
                else "b80f2afdbd8b32404b1aa713c2d403f8d71777a7237aa48d1170f5f20962e26c")
    assert digest(a.onnx) == expected
    manifest = edsr_packed_state_plugin.artifacts()
    r = dict(complete=False, started=time.time(), gpu=guard(), trt=trt.__version__, scope=__doc__,
             source_sha256=digest(Path(__file__)), original_onnx_sha256=digest(a.onnx),
             quantized_residual_blocks=16 if a.terminal_int8 else 15,
             plugin_sha256={m.__name__: digest(Path(m.__file__)) for m in (edsr_entry_plugin, edsr_packed_state_plugin)},
             kernel_config=manifest["config"], kernel_triton=manifest["triton"],
             kernel_source_sha256=manifest["kernel_source_sha256"],
             ptx_sha256={k: v["ptx_sha256"] for k, v in manifest["kernels"].items()})
    save(dest, r)
    try:
        model = onnx.load(a.onnx)
        r["replacements"] = transform(model, 16 if a.terminal_int8 else 15)
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
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.builder_optimization_level = 5
        config.avg_timing_iterations = 8
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if a.editable_cache or a.timing_cache:
            config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)
            cache = config.create_timing_cache(a.timing_cache.read_bytes() if a.timing_cache else b"")
            r["timing_cache_input_sha256"] = digest(a.timing_cache) if a.timing_cache else None
            if a.restore_int8_tactic:
                assert a.timing_cache is not None
                changes = []
                for key in cache.queryKeys():
                    value = cache.query(key)
                    if value.tacticHash == 0x214f03e23f252333:
                        assert cache.update(key, trt.TimingCacheValue(0x9dafb2758560cc1d, value.timingMSec))
                        changes.append(dict(key=str(key), previous=hex(value.tacticHash),
                                            replacement="0x9dafb2758560cc1d"))
                assert changes, "No matching INT8 tactic found in cache; no fallback changes allowed"
                r["tactic_changes"] = changes
            assert config.set_timing_cache(cache, False)
            save(dest, r)
        else:
            assert not a.restore_int8_tactic
        blob = builder.build_serialized_network(net, config)
        if blob is None: raise RuntimeError("Packed residual-state build failed")
        path = a.output / "edsr.engine"
        path.write_bytes(bytes(blob))
        if a.editable_cache or a.timing_cache:
            cache = config.get_timing_cache()
            cache_path = a.output / "timing.cache"
            cache_path.write_bytes(bytes(cache.serialize()))
            r["timing_cache_sha256"] = digest(cache_path)
            r["cache_entries"] = [dict(key=str(key), tactic=hex(cache.query(key).tacticHash),
                                       time_ms=cache.query(key).timingMSec) for key in cache.queryKeys()]
        r.update(complete=True, finished=time.time(), engine_sha256=digest(path), gpu_end=guard())
        save(dest, r)
    except BaseException as exc:
        r.update(error=repr(exc), failed=time.time()); save(dest, r)
        raise


if __name__ == "__main__": main()
