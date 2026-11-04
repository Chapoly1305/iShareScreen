"""AOT residual operators using byte correction codes and masked FP16 escapes.

PTX is compiled and independently checked with Triton 3.5, then embedded in
TensorRT. This avoids changing the compiler used by existing entry operators.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Tuple

import tensorrt.plugin as trtp


@lru_cache(maxsize=1)
def artifacts():
    root = Path(os.environ["ISS_EDSR_STATE_PTX"])
    data = json.loads((root / "MANIFEST.json").read_text())
    assert data["config"]["PACK_NIBBLES"] is False
    for mode, info in data["kernels"].items():
        ptx = (root / (mode + ".ptx")).read_bytes()
        assert hashlib.sha256(ptx).hexdigest() == info["ptx_sha256"]
        assert info["full_resolution_bit_exact"] and info["spills"] == 0
        info["ptx"] = ptx.decode()
    return data


def implementation(mode, height, width):
    data = artifacts()
    assert (height, width) == (data["height"], data["width"])
    entry = data["kernels"][mode]
    launch = trtp.KernelLaunchParams()
    launch.grid_x = entry["grid"]
    launch.block_x = entry["block"]
    launch.shared_mem = entry["shared"]
    return entry["name"], entry["ptx"], launch, trtp.SymIntExprs(0)


@trtp.register("iss::edsr_state_first")
def first(x: trtp.TensorDesc, weight: trtp.TensorDesc, skip: trtp.TensorDesc,
          alpha: trtp.TensorDesc, bias: trtp.TensorDesc, factor: trtp.TensorDesc,
          height: int, width: int) -> Tuple[trtp.TensorDesc, trtp.TensorDesc, trtp.TensorDesc]:
    return x.like(), x.like(), skip.like()


@trtp.autotune("iss::edsr_state_first")
def first_tune(x, weight, skip, alpha, bias, factor, height, width, outputs):
    return [trtp.AutoTuneCombination(
        "INT8,INT8,FP16,FP32,FP32,FP32,INT8,INT8,FP16",
        "CHW32,LINEAR,CHW32,LINEAR,LINEAR,LINEAR,CHW32,CHW32,CHW32")]


@trtp.aot_impl("iss::edsr_state_first")
def first_aot(x, weight, skip, alpha, bias, factor, height, width, outputs, tactic
              ) -> Tuple[str, str, trtp.KernelLaunchParams, trtp.SymIntExprs]:
    return implementation("first", height, width)


@trtp.register("iss::edsr_state_middle")
def middle(x: trtp.TensorDesc, weight: trtp.TensorDesc, skip_codes: trtp.TensorDesc,
           skip_bits: trtp.TensorDesc, skip_escape: trtp.TensorDesc, inverse: trtp.TensorDesc,
           alpha: trtp.TensorDesc, bias: trtp.TensorDesc, factor: trtp.TensorDesc,
           height: int, width: int) -> Tuple[trtp.TensorDesc, trtp.TensorDesc, trtp.TensorDesc]:
    return x.like(), x.like(), skip_escape.like()


@trtp.autotune("iss::edsr_state_middle")
def middle_tune(x, weight, skip_codes, skip_bits, skip_escape, inverse, alpha, bias, factor,
                height, width, outputs):
    return [trtp.AutoTuneCombination(
        "INT8,INT8,INT8,INT8,FP16,FP32,FP32,FP32,FP32,INT8,INT8,FP16",
        "CHW32,LINEAR,CHW32,CHW32,CHW32,LINEAR,LINEAR,LINEAR,LINEAR,CHW32,CHW32,CHW32")]


@trtp.aot_impl("iss::edsr_state_middle")
def middle_aot(x, weight, skip_codes, skip_bits, skip_escape, inverse, alpha, bias, factor,
               height, width, outputs, tactic
               ) -> Tuple[str, str, trtp.KernelLaunchParams, trtp.SymIntExprs]:
    return implementation("middle", height, width)


@trtp.register("iss::edsr_state_last")
def last(x: trtp.TensorDesc, weight: trtp.TensorDesc, skip_codes: trtp.TensorDesc,
         skip_bits: trtp.TensorDesc, skip_escape: trtp.TensorDesc, inverse: trtp.TensorDesc,
         alpha: trtp.TensorDesc, bias: trtp.TensorDesc, height: int, width: int) -> trtp.TensorDesc:
    return skip_escape.like()


@trtp.autotune("iss::edsr_state_last")
def last_tune(x, weight, skip_codes, skip_bits, skip_escape, inverse, alpha, bias,
              height, width, outputs):
    return [trtp.AutoTuneCombination(
        "INT8,INT8,INT8,INT8,FP16,FP32,FP32,FP32,FP16",
        "CHW32,LINEAR,CHW32,CHW32,CHW32,LINEAR,LINEAR,LINEAR,CHW32")]


@trtp.aot_impl("iss::edsr_state_last")
def last_aot(x, weight, skip_codes, skip_bits, skip_escape, inverse, alpha, bias,
             height, width, outputs, tactic
             ) -> Tuple[str, str, trtp.KernelLaunchParams, trtp.SymIntExprs]:
    return implementation("last", height, width)
