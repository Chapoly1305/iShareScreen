"""Residual convolution with INT8 codes, correction codes and FP16 escapes.

Physical correction buffer: [H, W, 32] bytes, low nibble channels 0..31,
high nibble channels 32..63. Code 15 reads a separately stored FP16 escape.
Escapes are intentionally uninitialized where the code is not 15.
Byte mode also supports signed corrections -127..127, with code 255 as escape.
"""
import argparse
from pathlib import Path
import time

import torch
from torch.nn import functional as F
import triton
import triton.language as tl

from benchmarks.super_resolution.profile_edsr_cutlass import timing
from benchmarks.super_resolution.profile_edsr_int8_epilogue import conv_dual, packed, unpacked
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save


@triton.jit
def packed_core(X, W, SkipCodes, SkipBits, SkipEscape, PreviousInverse,
                Alpha, Bias, Factor, OutCodes, OutBits, OutEscape,
                H: tl.constexpr, WIDTH: tl.constexpr, M: tl.constexpr,
                INPUT_HALF: tl.constexpr, FINAL: tl.constexpr,
                PACK_NIBBLES: tl.constexpr = True, CODE_MAX: tl.constexpr = 7):
    tl.static_assert(CODE_MAX == 7 or (CODE_MAX == 127 and not PACK_NIBBLES))
    pixel = tl.program_id(0) * M + tl.arange(0, M)
    channel = tl.arange(0, 64)
    output_channel = tl.arange(0, 64)
    y, x = pixel // WIDTH, pixel % WIDTH
    acc = tl.full((M, 64), 0, tl.int32)
    for rs in range(9):
        iy, ix = y + rs // 3 - 1, x + rs % 3 - 1
        address = channel[None, :] // 32 * (H * WIDTH * 32) + (iy[:, None] * WIDTH + ix[:, None]) * 32 + channel[None, :] % 32
        value = tl.load(X + address, (pixel[:, None] < H * WIDTH) &
            (iy[:, None] >= 0) & (iy[:, None] < H) & (ix[:, None] >= 0) & (ix[:, None] < WIDTH), 0)
        weight = tl.load(W + (rs * 64 + channel[:, None]) * 64 + output_channel[None, :])
        acc = tl.dot(value, weight, acc, out_dtype=tl.int32)
    offset = output_channel[None, :] // 32 * (H * WIDTH * 32) + pixel[:, None] * 32 + output_channel[None, :] % 32
    valid = pixel[:, None] < H * WIDTH
    if INPUT_HALF:
        skip = tl.load(SkipEscape + offset, valid, 0).to(tl.float32)
    else:
        q_previous = tl.load(SkipCodes + offset, valid, 0).to(tl.float32)
        if PACK_NIBBLES:
            bits = tl.load(SkipBits + pixel[:, None] * 32 + output_channel[None, :] % 32, valid, 0).to(tl.uint8)
            code = ((bits >> ((output_channel[None, :] // 32) * 4)) & 15).to(tl.int32)
        else:
            code = tl.load(SkipBits + offset, valid, 0).to(tl.int32)
        inverse = tl.load(PreviousInverse + output_channel)
        decoded = ((q_previous + (code - CODE_MAX).to(tl.float32) * (1.0 / (2 * CODE_MAX))) * inverse[None, :]).to(tl.float16)
        escaped = tl.load(SkipEscape + offset, valid & (code == 2 * CODE_MAX + 1), 0)
        skip = tl.where(code == 2 * CODE_MAX + 1, escaped, decoded).to(tl.float32)
    alpha = tl.load(Alpha + output_channel)
    bias = tl.load(Bias + output_channel)
    state = (acc.to(tl.float32) * alpha[None, :] + bias[None, :] + skip).to(tl.float16)
    if FINAL:
        tl.store(OutEscape + offset, state, valid)
    else:
        factor = tl.load(Factor + output_channel)
        position = state.to(tl.float32) * factor[None, :]
        q = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(position), -128), 127).to(tl.int8)
        error = position - q.to(tl.float32)
        signed = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(error * (2.0 * CODE_MAX)), -CODE_MAX), CODE_MAX).to(tl.int32)
        escape = tl.abs(error) > 0.5
        code = tl.where(escape, 2 * CODE_MAX + 1, signed + CODE_MAX).to(tl.uint8)
        if PACK_NIBBLES:
            grouped = tl.reshape(code, (M, 2, 32))
            low, high = tl.split(tl.permute(grouped, (0, 2, 1)))
            bits = low | (high << 4)
            tl.store(OutBits + pixel[:, None] * 32 + tl.arange(0, 32)[None, :], bits, valid)
        else:
            tl.store(OutBits + offset, code, valid)
        tl.store(OutCodes + offset, q, valid)
        tl.store(OutEscape + offset, state, valid & escape)


@triton.jit
def first_kernel(X, W, Skip, Alpha, Bias, Factor, Q, Bits, Escape,
                 H: tl.constexpr, WIDTH: tl.constexpr, M: tl.constexpr,
                 PACK_NIBBLES: tl.constexpr = True, CODE_MAX: tl.constexpr = 7):
    packed_core(X, W, X, X, Skip, Alpha, Alpha, Bias, Factor, Q, Bits, Escape,
                H, WIDTH, M, True, False, PACK_NIBBLES, CODE_MAX)


@triton.jit
def middle_kernel(X, W, SkipCodes, SkipBits, SkipEscape, PreviousInverse,
                  Alpha, Bias, Factor, Q, Bits, Escape,
                  H: tl.constexpr, WIDTH: tl.constexpr, M: tl.constexpr,
                  PACK_NIBBLES: tl.constexpr = True, CODE_MAX: tl.constexpr = 7):
    packed_core(X, W, SkipCodes, SkipBits, SkipEscape, PreviousInverse,
                Alpha, Bias, Factor, Q, Bits, Escape, H, WIDTH, M, False, False, PACK_NIBBLES, CODE_MAX)


@triton.jit
def last_kernel(X, W, SkipCodes, SkipBits, SkipEscape, PreviousInverse, Alpha, Bias, Output,
                H: tl.constexpr, WIDTH: tl.constexpr, M: tl.constexpr,
                PACK_NIBBLES: tl.constexpr = True, CODE_MAX: tl.constexpr = 7):
    packed_core(X, W, SkipCodes, SkipBits, SkipEscape, PreviousInverse,
                Alpha, Bias, Alpha, X, X, Output, H, WIDTH, M, False, True, PACK_NIBBLES, CODE_MAX)


def encode(state, factor, pack_nibbles=True, code_max=7):
    """Independent eager reference using NCHW tensors, not the GPU bit layout."""
    position = state.float() * factor[None, :, None, None]
    q = position.round().clamp(-128, 127).to(torch.int8)
    error = position - q.float()
    assert code_max == 7 or (code_max == 127 and not pack_nibbles)
    code = (error * (2 * code_max)).round().clamp(-code_max, code_max).int() + code_max
    code = torch.where(error.abs() > .5, 2 * code_max + 1, code).to(torch.uint8)
    bits = ((code[:, :32] | (code[:, 32:] << 4)).permute(0, 2, 3, 1).contiguous()
            if pack_nibbles else packed(code))
    mask = code == 2 * code_max + 1
    # Dense selection avoids the unrelated CUB nonzero helper used by boolean
    # indexing, whose PTX-version probe raises CUDA API sanitizer diagnostics.
    escape = torch.where(mask, state, torch.full_like(state, float("nan")))
    return packed(q), bits, packed(escape), packed(mask)


def decode(q, bits, escape, inverse, pack_nibbles=True, code_max=7):
    if pack_nibbles:
        low = bits & 15
        high = bits >> 4
        code = torch.cat((low.permute(0, 3, 1, 2), high.permute(0, 3, 1, 2)), 1)
    else:
        code = unpacked(bits)
    decoded = ((unpacked(q).float() + (code.int() - code_max).float() * (1./(2*code_max))) * inverse[None, :, None, None]).half()
    return torch.where(code == 2*code_max+1, unpacked(escape), decoded)


def equal_escapes(actual, expected, mask):
    return torch.equal(torch.where(mask, actual, 0), torch.where(mask, expected, 0))


def untouched_escapes(actual, mask):
    return torch.where(mask, True, torch.isnan(actual)).all().item()


def launch(args, h, w, config, mode):
    return packed_core[(triton.cdiv(h*w, config["M"]),)](*args, h, w, **config,
        INPUT_HALF=mode == "first", FINAL=mode == "last", enable_fp_fusion=False)


@torch.inference_mode()
def check(config):
    torch.manual_seed(1729)
    pack_nibbles = config.get("PACK_NIBBLES", True)
    code_max = config.get("CODE_MAX", 7)
    records = []
    for h, w in ((1, 1), (2, 3), (17, 35)):
        for kind in ("random", "constant", "corner"):
            x = torch.randint(-128, 128, (1, 64, h, w), device="cuda", dtype=torch.int8)
            state = torch.randn((1, 64, h, w), device="cuda", dtype=torch.float16) * 20
            if kind == "constant":
                x.fill_(1)
                state.zero_()
            if kind == "corner":
                x.zero_(); x[:, :, 0, 0] = 127; x[:, :, -1, -1] = -128
                state[:, :32, 0, 0] = 60000
                state[:, 32:, -1, -1] = -60000
            weight = torch.randint(-128, 128, (64, 64, 3, 3), device="cuda", dtype=torch.int8)
            previous_factor = torch.rand(64, device="cuda") * 2 + .01
            inverse = 1. / previous_factor
            sq, sb, se, _ = encode(state, previous_factor, pack_nibbles, code_max)
            decoded = decode(sq, sb, se, inverse, pack_nibbles, code_max)
            alpha = torch.rand(64, device="cuda") * .001
            bias = torch.randn(64, device="cuda")
            factor = torch.rand(64, device="cuda") * .4 + .01
            raw = F.conv2d(x.float(), weight.float(), padding=1) * alpha[None, :, None, None] + bias[None, :, None, None]
            for mode in ("first", "middle", "last"):
                expected_state = (raw + (state if mode == "first" else decoded).float()).half()
                # Leave headroom below FP16 overflow while exercising INT8
                # clipping with large, finite FP16 escape values.
                assert torch.isfinite(expected_state).all().item(), (h, w, kind)
                oq = torch.empty_like(sq)
                ob = torch.empty_like(sb)
                oe = torch.full_like(se, float("nan"))
                args = (packed(x), weight.permute(2, 3, 1, 0).contiguous(), sq, sb,
                        packed(state) if mode == "first" else se, inverse, alpha, bias, factor, oq, ob, oe)
                launch(args, h, w, config, mode)
                torch.cuda.synchronize()
                if mode == "last":
                    assert torch.equal(oe, packed(expected_state)), (h, w, kind, mode, "half")
                else:
                    eq, eb, ee, em = encode(expected_state, factor, pack_nibbles, code_max)
                    assert torch.equal(oq, eq), (h, w, kind, mode, "int8")
                    assert torch.equal(ob, eb), (h, w, kind, mode, "nibbles")
                    assert equal_escapes(oe, ee, em), (h, w, kind, mode, "escapes")
                    assert untouched_escapes(oe, em), "Unexpected write to unused escape storage"
                records.append(dict(shape=[h, w], kind=kind, mode=mode, bit_exact=True))
    return records


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--capture-only", action="store_true")
    p.add_argument("--byte-corrections", action="store_true",
                   help="Keep identical correction math but store one code per byte, avoiding bit packing.")
    p.add_argument("--code-max", type=int, choices=(7,127), default=7,
                   help="Signed correction range; 127 uses all byte codes with 255 reserved for FP16 escape.")
    p.add_argument("--configs", default="32:4:2,64:4:2,128:4:2,128:8:2,64:2:3",
                   help="Comma-separated M:warps:stages; capture-only uses the first.")
    p.add_argument("--export-plugins", type=Path,
                   help="Export checked first/middle/last PTX for TensorRT; uses first config and byte codes.")
    a = p.parse_args()
    assert a.code_max == 7 or a.byte_corrections
    configs = [tuple(map(int, s.split(":"))) for s in a.configs.split(",")]
    assert all(len(c) == 3 for c in configs)
    if a.output.exists(): raise FileExistsError(a.output)
    r = dict(complete=False, started=time.time(), gpu=guard(), source_sha256=digest(Path(__file__)),
             scope="Packed-state kernel microbenchmark only; not a full EDSR timing.", triton=triton.__version__,
             pack_nibbles=not a.byte_corrections, code_max=a.code_max, records=[])
    save(a.output, r)
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    stream = torch.cuda.Stream(); torch.cuda.set_stream(stream); torch.manual_seed(1729)
    h, w = 720, 1280
    x = torch.randint(-128, 128, (1, 2, h, w, 32), device="cuda", dtype=torch.int8)
    weight = torch.randint(-128, 128, (9, 64, 64), device="cuda", dtype=torch.int8)
    state = torch.randn((1, 64, h, w), device="cuda", dtype=torch.float16) * 20
    previous_factor = torch.rand(64, device="cuda") * 2 + .01
    inverse = 1. / previous_factor
    sq, sb, se, mask = encode(state, previous_factor, not a.byte_corrections, a.code_max)
    decoded = packed(decode(sq, sb, se, inverse, not a.byte_corrections, a.code_max))
    alpha = torch.rand(64, device="cuda") * .001
    bias = torch.randn(64, device="cuda")
    factor = torch.rand(64, device="cuda") * .2 + .01
    oq, ob, oe = torch.empty_like(sq), torch.empty_like(sb), torch.empty_like(se)
    reference, reference_q = torch.empty_like(se), torch.empty_like(sq)
    args = (x, weight, sq, sb, se, inverse, alpha, bias, factor, oq, ob, oe)
    if a.capture_only:
        m, warps, stages = configs[0]
        config = dict(M=m, num_warps=warps, num_stages=stages, PACK_NIBBLES=not a.byte_corrections, CODE_MAX=a.code_max)
        r["correctness"] = check(config)
        kernel = launch(args, h, w, config, "middle")
        stream.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        launch(args, h, w, config, "middle")
        stream.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        r.update(complete=True, finished=time.time(), registers=kernel.n_regs,
                 spills=kernel.n_spills, shared_bytes=kernel.metadata.shared,
                 gpu_end=guard())
        save(a.output, r)
        return
    def baseline():
        return conv_dual[(triton.cdiv(h*w, 64),)](x, weight, decoded, alpha, bias, factor,
            reference, reference_q, h, w, 64, num_warps=4, num_stages=2, enable_fp_fusion=False)
    r["baseline_before"] = timing(baseline, stream, a.samples)
    r["input_escape_fraction"] = float(mask.float().mean())
    baseline(); stream.synchronize()
    eq, eb, ee, em = encode(unpacked(reference), factor, not a.byte_corrections, a.code_max)
    r["output_escape_fraction"] = float(em.float().mean())
    if a.export_plugins:
        assert a.byte_corrections, "TensorRT candidate uses byte codes"
        a.export_plugins.mkdir(parents=True, exist_ok=True)
        m, warps, stages = configs[0]
        config = dict(M=m, num_warps=warps, num_stages=stages, PACK_NIBBLES=False, CODE_MAX=a.code_max)
        r["correctness"] = check(config)
        manifest = dict(height=h, width=w, config=config, triton=triton.__version__,
                        kernel_source_sha256=digest(Path(__file__)), kernels={})
        for mode, fn, arguments in (
            ("first", first_kernel, (x, weight, decoded, alpha, bias, factor, oq, ob, oe)),
            ("middle", middle_kernel, args),
            ("last", last_kernel, (x, weight, sq, sb, se, inverse, alpha, bias, oe)),
        ):
            def run():
                return fn[(triton.cdiv(h*w, m),)](*arguments, h, w, **config, enable_fp_fusion=False)
            oe.fill_(float("nan"))
            kernel = run(); stream.synchronize()
            if mode == "last":
                assert torch.equal(oe, reference), "Final wrapper differs"
            else:
                assert torch.equal(oq, eq) and torch.equal(ob, eb) and equal_escapes(oe, ee, em), mode
                assert untouched_escapes(oe, em), "Wrapper wrote unused escapes"
            latency = timing(run, stream, a.samples)
            path = a.export_plugins / (mode + ".ptx")
            assert not path.exists(), path
            path.write_text(kernel.asm["ptx"])
            entry = dict(name=kernel.metadata.name, registers=kernel.n_regs, spills=kernel.n_spills,
                         block=warps*32, grid=triton.cdiv(h*w, m), shared=kernel.metadata.shared,
                         full_resolution_bit_exact=True, ptx_sha256=digest(path), latency=latency)
            manifest["kernels"][mode] = entry
            print(mode, latency["p95_ms"], kernel.n_regs, kernel.n_spills, flush=True)
        save(a.export_plugins / "MANIFEST.json", manifest)
        r.update(export=manifest, baseline_after=timing(baseline, stream, a.samples), complete=True,
                 finished=time.time(), gpu_end=guard())
        save(a.output, r)
        return
    for m, warps, stages in configs:
        config = dict(M=m, num_warps=warps, num_stages=stages, PACK_NIBBLES=not a.byte_corrections, CODE_MAX=a.code_max)
        row = dict(config=config)
        try:
            row["correctness"] = check(config)
            kernel = launch(args, h, w, config, "middle"); stream.synchronize()
            assert torch.equal(oq, eq) and torch.equal(ob, eb) and equal_escapes(oe, ee, em), "Full-resolution outputs differ"
            row.update(full_resolution_bit_exact=True, registers=kernel.n_regs, spills=kernel.n_spills,
                       shared_bytes=kernel.metadata.shared)
            row["latency"] = timing(lambda: launch(args, h, w, config, "middle"), stream, a.samples)
            print(config, row["latency"]["p95_ms"], "regs", kernel.n_regs, "spills", kernel.n_spills, flush=True)
        except Exception as exc:
            row["error"] = repr(exc); print(config, repr(exc), flush=True)
        r["records"].append(row); save(a.output, r)
    r["baseline_after"] = timing(baseline, stream, a.samples)
    r.update(complete=True, finished=time.time(), gpu_end=guard()); save(a.output, r)


if __name__ == "__main__": main()
