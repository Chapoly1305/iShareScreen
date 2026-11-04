"""Supplement resize comparison with graph replay, avoiding Python launch gaps."""
import argparse
import os
import time
from pathlib import Path

import torch
import triton

from benchmarks.super_resolution.edsr_cuda.common import digest, guard, save, stats
from compare_resize import (
    area_float, coefficients, resample_h, resample_v,
)
from isharescreen.enhancement.cuda_kernels import pack_area


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-gpu-uuid', required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    os.environ['ISS_EXPECTED_GPU_UUID'] = a.expected_gpu_uuid
    torch.set_num_threads(1)
    r = dict(complete=False, started=time.time(), gpu=guard(),
             script_sha256=digest(__file__), torch=torch.__version__,
             triton=triton.__version__,
             scope='Resize plus RGBA8 packing only, one operation per CUDA graph. '
             'Synthetic seed-42 resident FP32 CHW 1440x2560 input; kernels have '
             'no data-dependent branching. No EDSR inference or quality measurements. '
             '100 warmups, 1000 CUDA-event samples per method per round, three '
             'rotated orders. No inference/decode/transfers/display.', timings=[])
    torch.manual_seed(42)
    x = torch.rand((3, 1440, 2560), device='cuda')
    output = torch.empty((1080, 1920, 4), dtype=torch.uint8, device='cuda')
    temp = torch.empty((3, 1440, 1920), device='cuda')
    n = output.numel()
    plans = {k: (coefficients(2560, 1920, k), coefficients(1440, 1080, k))
             for k in ['cubic', 'lanczos4']}
    names = ['legacy_area', 'clamp_float_area', 'float_area', 'cubic', 'lanczos4']

    def run(name):
        if name == 'legacy_area':
            pack_area[(triton.cdiv(n, 256),)](
                x, output, 1440*2560, 2560, 1, 4, n, 256, enable_fp_fusion=False)
        elif name.endswith('area'):
            area_float[(triton.cdiv(n, 256),)](
                x, output, name == 'clamp_float_area', n, 256, enable_fp_fusion=False)
        else:
            h, v = plans[name]
            k = 4 if name == 'cubic' else 8
            resample_h[(triton.cdiv(temp.numel(), 256),)](
                x, temp, *h, k, temp.numel(), 256, enable_fp_fusion=False)
            resample_v[(triton.cdiv(n, 256),)](
                temp, output, *v, k, n, 256, enable_fp_fusion=False)

    graphs = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name in names:
            for _ in range(10):
                run(name)
            torch.cuda.synchronize()
            graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[name], stream=stream):
                run(name)
    torch.cuda.synchronize()
    for ri, order in enumerate([names, list(reversed(names)), names[2:]+names[:2]]):
        for name in order:
            guard()
            for _ in range(100):
                graphs[name].replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(1000):
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            record = dict(round=ri, method=name, **stats(samples))
            r['timings'].append(record)
            save(a.output, r)
            print(ri, name, record['p50_ms'], record['p95_ms'], flush=True)
    r.update(complete=True, finished=time.time(), gpu_end=guard())
    save(a.output, r)


if __name__ == '__main__':
    main()
