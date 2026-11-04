"""Adjacent-channel nibble storage for the accepted EDSR residual state.

Correction storage is [2,H,W,16] bytes: each byte holds channels (2c,2c+1)
within one CHW32 group. Math and escape semantics match the accepted byte
implementation. This is a component experiment until integrated and measured.
"""
import argparse
from pathlib import Path
import time

import torch
from torch.nn import functional as F
import triton
import triton.language as tl

from benchmarks.super_resolution import profile_edsr_packed_state as previous
from benchmarks.super_resolution.profile_edsr_cutlass import timing
from benchmarks.super_resolution.profile_edsr_int8_epilogue import packed, unpacked
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save


@triton.jit
def adjacent_core(X, W, SkipCodes, SkipBits, SkipEscape, PreviousInverse,
                  Alpha, Bias, Factor, OutCodes, OutBits, OutEscape,
                  H: tl.constexpr, WIDTH: tl.constexpr, M: tl.constexpr,
                  INPUT_HALF: tl.constexpr, FINAL: tl.constexpr,
                  UNROLL: tl.constexpr = 1, PAIR_GATHER: tl.constexpr = False):
    pixel = tl.program_id(0) * M + tl.arange(0, M)
    channel = tl.arange(0, 64)
    output_channel = tl.arange(0, 64)
    y, x = pixel // WIDTH, pixel % WIDTH
    acc = tl.full((M, 64), 0, tl.int32)
    for rs in tl.range(9, loop_unroll_factor=UNROLL):
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
        bit_offset = output_channel[None, :] // 32 * (H * WIDTH * 16) + pixel[:, None] * 16 + (output_channel[None, :] % 32) // 2
        bits = tl.load(SkipBits + bit_offset, valid, 0).to(tl.uint8)
        code = ((bits >> ((output_channel[None, :] % 2) * 4)) & 15).to(tl.int32)
        inverse = tl.load(PreviousInverse + output_channel)
        decoded = ((q_previous + (code - 7).to(tl.float32) * (1.0 / 14.0)) * inverse[None, :]).to(tl.float16)
        escaped = tl.load(SkipEscape + offset, valid & (code == 15), 0)
        skip = tl.where(code == 15, escaped, decoded).to(tl.float32)
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
        signed = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(error * 14.0), -7), 7).to(tl.int32)
        escape = tl.abs(error) > 0.5
        code = tl.where(escape, 15, signed + 7).to(tl.uint8)
        if PAIR_GATHER:
            neighbor = tl.gather(code, tl.broadcast_to((output_channel[None, :] ^ 1), (M, 64)), axis=1)
            bit_offset = output_channel[None, :] // 32 * (H * WIDTH * 16) + pixel[:, None] * 16 + (output_channel[None, :] % 32) // 2
            tl.store(OutBits + bit_offset, code | (neighbor << 4), valid & (output_channel[None, :] % 2 == 0))
        else:
            low, high = tl.split(tl.reshape(code, (M, 32, 2)))
            pairs = tl.arange(0, 32)
            bit_offset = pairs[None, :] // 16 * (H * WIDTH * 16) + pixel[:, None] * 16 + pairs[None, :] % 16
            tl.store(OutBits + bit_offset, low | (high << 4), valid)
        tl.store(OutCodes + offset, q, valid)
        tl.store(OutEscape + offset, state, valid & escape)


@triton.jit
def first_kernel(X, W, Skip, Alpha, Bias, Factor, Q, Bits, Escape,
                 H: tl.constexpr, WIDTH: tl.constexpr, M: tl.constexpr, UNROLL: tl.constexpr = 1, PAIR_GATHER: tl.constexpr = False):
    adjacent_core(X, W, X, X, Skip, Alpha, Alpha, Bias, Factor, Q, Bits, Escape,
                  H, WIDTH, M, True, False, UNROLL, PAIR_GATHER)


@triton.jit
def middle_kernel(X, W, SkipCodes, SkipBits, SkipEscape, PreviousInverse,
                  Alpha, Bias, Factor, Q, Bits, Escape,
                  H: tl.constexpr, WIDTH: tl.constexpr, M: tl.constexpr, UNROLL: tl.constexpr = 1, PAIR_GATHER: tl.constexpr = False):
    adjacent_core(X, W, SkipCodes, SkipBits, SkipEscape, PreviousInverse,
                  Alpha, Bias, Factor, Q, Bits, Escape, H, WIDTH, M, False, False, UNROLL, PAIR_GATHER)


@triton.jit
def last_kernel(X, W, SkipCodes, SkipBits, SkipEscape, PreviousInverse, Alpha, Bias, Output,
                H: tl.constexpr, WIDTH: tl.constexpr, M: tl.constexpr, UNROLL: tl.constexpr = 1, PAIR_GATHER: tl.constexpr = False):
    adjacent_core(X, W, SkipCodes, SkipBits, SkipEscape, PreviousInverse,
                  Alpha, Bias, Alpha, X, X, Output, H, WIDTH, M, False, True, UNROLL, PAIR_GATHER)


def pair_codes(byte_codes):
    a = byte_codes.reshape(*byte_codes.shape[:-1], 16, 2)
    return (a[..., 0] | (a[..., 1] << 4)).contiguous()


def launch(args, h, w, config, mode):
    return adjacent_core[(triton.cdiv(h*w, config["M"]),)](*args, h, w, **config,
        INPUT_HALF=mode == "first", FINAL=mode == "last", enable_fp_fusion=False)


@torch.inference_mode()
def check(config):
    torch.manual_seed(1729)
    records = []
    for h, w in ((1, 1), (2, 3), (17, 35), (5, 1280)):
        for kind in ("random", "constant", "corner"):
            x = torch.randint(-128, 128, (1, 64, h, w), device="cuda", dtype=torch.int8)
            state = torch.randn((1, 64, h, w), device="cuda", dtype=torch.float16) * 20
            if kind == "constant":
                x.fill_(1); state.zero_()
            elif kind == "corner":
                x.zero_(); x[:, :, 0, 0] = 127; x[:, :, -1, -1] = -128
                state[:, :32, 0, 0] = 60000; state[:, 32:, -1, -1] = -60000
            weight = torch.randint(-128, 128, (64, 64, 3, 3), device="cuda", dtype=torch.int8)
            previous_factor = torch.rand(64, device="cuda") * 2 + .01
            inverse = 1. / previous_factor
            sq, sb, se, _ = previous.encode(state, previous_factor, False)
            decoded = previous.decode(sq, sb, se, inverse, False)
            paired = pair_codes(sb)
            alpha = torch.rand(64, device="cuda") * .001
            bias = torch.randn(64, device="cuda")
            factor = torch.rand(64, device="cuda") * .4 + .01
            raw = F.conv2d(x.float(), weight.float(), padding=1) * alpha[None, :, None, None] + bias[None, :, None, None]
            for mode in ("first", "middle", "last"):
                expected = (raw + (state if mode == "first" else decoded).float()).half()
                assert torch.isfinite(expected).all().item()
                oq, ob, oe = torch.empty_like(sq), torch.empty_like(paired), torch.full_like(se, float("nan"))
                args = (packed(x), weight.permute(2,3,1,0).contiguous(), sq, paired,
                        packed(state) if mode == "first" else se, inverse, alpha, bias, factor, oq, ob, oe)
                launch(args,h,w,config,mode); torch.cuda.synchronize()
                if mode == "last":
                    assert torch.equal(oe, packed(expected)), (h,w,kind,mode)
                else:
                    eq, eb, ee, mask = previous.encode(expected,factor,False)
                    assert torch.equal(oq,eq) and torch.equal(ob,pair_codes(eb)), (h,w,kind,mode)
                    assert torch.equal(oe[mask],ee[mask]) and torch.isnan(oe[mask == False]).all().item()
                records.append(dict(shape=[h,w],kind=kind,mode=mode,bit_exact=True))
    return records


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--samples",type=int,default=300)
    p.add_argument("--configs",default="64:2:2:1,128:4:1:1,64:4:2:1,32:2:2:1,64:2:1:1,64:2:2:3,64:2:2:9")
    p.add_argument("--export-plugins",type=Path)
    p.add_argument("--capture-only",action="store_true")
    p.add_argument("--pair-gather",action="store_true",help="Pack using adjacent gather and masked stores without changing the accumulator shape.")
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    configs=[dict(M=v[0],num_warps=v[1],num_stages=v[2],UNROLL=v[3],PAIR_GATHER=a.pair_gather) for v in
             [tuple(map(int,s.split(":"))) for s in a.configs.split(",")]]
    r=dict(complete=False,started=time.time(),gpu=guard(),source_sha256=digest(Path(__file__)),
           reference_sha256=digest(Path(previous.__file__)),triton=triton.__version__,records=[],
           scope="Component only; numerically identical adjacent nibble packing vs accepted byte correction state.")
    save(a.output,r)
    torch.set_num_threads(1);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream);torch.manual_seed(1729)
    h,w=720,1280
    x=torch.randint(-128,128,(1,2,h,w,32),device="cuda",dtype=torch.int8)
    weight=torch.randint(-128,128,(9,64,64),device="cuda",dtype=torch.int8)
    state=torch.randn((1,64,h,w),device="cuda",dtype=torch.float16)*20
    prevfactor=torch.rand(64,device="cuda")*2+.01;inverse=1./prevfactor
    sq,sb,se,mask=previous.encode(state,prevfactor,False)
    paired=pair_codes(sb);decoded=packed(previous.decode(sq,sb,se,inverse,False))
    alpha=torch.rand(64,device="cuda")*.001;bias=torch.randn(64,device="cuda");factor=torch.rand(64,device="cuda")*.2+.01
    oq,ob,oe=torch.empty_like(sq),torch.empty_like(paired),torch.empty_like(se)
    rq,rb,re=torch.empty_like(sq),torch.empty_like(sb),torch.empty_like(se)
    args=(x,weight,sq,paired,se,inverse,alpha,bias,factor,oq,ob,oe)
    refargs=(x,weight,sq,sb,se,inverse,alpha,bias,factor,rq,rb,re)
    refconfig=dict(M=64,num_warps=2,num_stages=2,PACK_NIBBLES=False)
    def baseline():return previous.launch(refargs,h,w,refconfig,"middle")
    r["baseline_before"]=timing(baseline,stream,a.samples)
    baseline();stream.synchronize();paired_reference=pair_codes(rb);escaped=rb==15
    for config in configs[:1] if a.capture_only or a.export_plugins else configs:
        row=dict(config=config);r["records"].append(row)
        try:
            row["correctness"]=check(config)
            kernel=launch(args,h,w,config,"middle");stream.synchronize()
            assert torch.equal(oq,rq) and torch.equal(ob,paired_reference) and torch.equal(oe[escaped],re[escaped])
            row.update(full_resolution_bit_exact=True,registers=kernel.n_regs,spills=kernel.n_spills,shared_bytes=kernel.metadata.shared)
            if a.capture_only:
                torch.cuda.cudart().cudaProfilerStart();launch(args,h,w,config,"middle");stream.synchronize();torch.cuda.cudart().cudaProfilerStop()
            row["latency"]=timing(lambda:launch(args,h,w,config,"middle"),stream,a.samples)
            print(config,row["latency"]["p95_ms"],kernel.n_regs,kernel.n_spills,flush=True)
            if a.export_plugins:
                a.export_plugins.mkdir(parents=True,exist_ok=False)
                manifest=dict(height=h,width=w,config=dict(config,PACK_NIBBLES=True),correction_layout="adjacent_nibble",
                              triton=triton.__version__,kernel_source_sha256=digest(Path(__file__)),kernels={})
                for mode,fn,arguments in (
                    ("first",first_kernel,(x,weight,decoded,alpha,bias,factor,oq,ob,oe)),
                    ("middle",middle_kernel,args),
                    ("last",last_kernel,(x,weight,sq,paired,se,inverse,alpha,bias,oe))):
                    refmodeargs=list(refargs)
                    if mode=="first":refmodeargs[4]=decoded
                    previous.launch(tuple(refmodeargs),h,w,refconfig,mode);stream.synchronize()
                    def run():return fn[(triton.cdiv(h*w,config['M']),)](*arguments,h,w,**config,enable_fp_fusion=False)
                    oe.fill_(float("nan"));k=run();stream.synchronize()
                    if mode=="last":assert torch.equal(oe,re)
                    else:
                        esc=rb==15
                        assert torch.equal(oq,rq) and torch.equal(ob,pair_codes(rb)) and torch.equal(oe[esc],re[esc])
                        assert torch.isnan(oe[esc==False]).all().item()
                    lat=timing(run,stream,a.samples);path=a.export_plugins/(mode+'.ptx');path.write_text(k.asm['ptx'])
                    manifest['kernels'][mode]=dict(name=k.metadata.name,registers=k.n_regs,spills=k.n_spills,
                        block=config['num_warps']*32,grid=triton.cdiv(h*w,config['M']),shared=k.metadata.shared,
                        full_resolution_bit_exact=True,ptx_sha256=digest(path),latency=lat)
                save(a.export_plugins/'MANIFEST.json',manifest);r['export']=manifest
        except Exception as exc:
            row['error']=repr(exc)
            if a.export_plugins or a.capture_only:save(a.output,r);raise
        save(a.output,r)
    r.update(baseline_after=timing(baseline,stream,a.samples),complete=True,finished=time.time(),gpu_end=guard())
    save(a.output,r)


if __name__=="__main__":main()
