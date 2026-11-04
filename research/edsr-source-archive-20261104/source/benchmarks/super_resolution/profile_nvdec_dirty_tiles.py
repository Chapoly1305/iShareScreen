"""Exact GPU-decoded changes expanded by EDSR's 36-LR-pixel dependency halo.

Feasibility census only: estimates the number of fixed-size tiles requiring
inference. It does not execute a cached EDSR or establish a latency speedup.
Only tile flags, not image pixels, are copied to CPU for the analysis record.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import triton
import triton.language as tl
import PyNvVideoCodec as nvc

from benchmarks.super_resolution.profile_edsr_tiles import tile_coordinates
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save


@triton.jit
def mark_dirty(Current, Previous, Dirty, SC: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
               H: tl.constexpr, W: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
               HALO: tl.constexpr, B: tl.constexpr):
    tile = tl.program_id(0)
    ch, cw = H // ROWS, W // COLS
    y0 = tl.maximum(tile // COLS * ch - HALO, 0)
    x0 = tl.maximum(tile % COLS * cw - HALO, 0)
    y1 = tl.minimum((tile // COLS + 1) * ch + HALO, H)
    x1 = tl.minimum((tile % COLS + 1) * cw + HALO, W)
    size = (y1 - y0) * (x1 - x0)
    offsets = tl.arange(0, B)
    base = 0
    changed = tl.full((), 0, tl.int32)
    while (base < size) & (changed == 0):
        index = base + offsets
        y, x = y0 + index // (x1 - x0), x0 + index % (x1 - x0)
        valid = index < size
        source, prior = y * SH + x * SW, y * W + x
        differs = tl.full((B,), False, tl.int1)
        for channel in tl.static_range(3):
            a = tl.load(Current + channel * SC + source, valid, 0)
            b = tl.load(Previous + channel * H * W + prior, valid, 0)
            differs = differs | (a != b)
        changed = tl.max(differs.to(tl.int32), 0)
        base += B
    tl.store(Dirty + tile, changed)


def metadata(rows, cols):
    h, w, _, _, coordinates = tile_coordinates(rows, cols, 36)
    return dict(rows=rows, cols=cols, core=[720 // rows, 1280 // cols], halo=36,
                tile=[h, w], pixels_per_tile=h * w,
                all_tiles_pixel_ratio=h * w * rows * cols / (720 * 1280))


@torch.inference_mode()
def check(grids):
    """Single-pixel changes must mark exactly those cores whose halo sees them."""
    previous = torch.zeros((3, 720, 1280), dtype=torch.uint8, device="cuda")
    current = torch.zeros_like(previous)
    records = []
    for rows, cols in grids:
        dirty = torch.empty(rows * cols, dtype=torch.int32, device="cuda")
        for y, x in ((0, 0), (719, 1279), (35, 35), (36, 36), (120, 128), (359, 639)):
            current.zero_()
            current[1, y, x] = 1
            mark_dirty[(rows * cols,)](current, previous, dirty, *current.stride(),
                720, 1280, rows, cols, 36, 256)
            expected = [int(max(0, r * (720 // rows) - 36) <= y < min(720, (r + 1) * (720 // rows) + 36)
                            and max(0, c * (1280 // cols) - 36) <= x < min(1280, (c + 1) * (1280 // cols) + 36))
                        for r in range(rows) for c in range(cols)]
            assert dirty.cpu().tolist() == expected, (rows, cols, y, x)
            records.append(dict(grid=[rows, cols], pixel=[y, x], exact=True))
        current.zero_()
        mark_dirty[(rows * cols,)](current, previous, dirty, *current.stride(),
            720, 1280, rows, cols, 36, 256)
        assert not dirty.any().item()
        records.append(dict(grid=[rows, cols], kind="identical", exact=True))
    return records


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frame-step", type=int, default=1,
                   help="Analyze every Nth adjacent-frame pair; still decode every frame")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    assert a.frame_step >= 1
    grids = [(6, 10), (3, 5), (4, 4), (4, 2), (8, 1), (4, 1)]
    names = [f"{r}x{c}" for r, c in grids]
    r = dict(complete=False, started=time.time(), gpu=guard(), scope=__doc__,
             source_sha256=digest(Path(__file__)), nvc=nvc.__version__, frame_step=a.frame_step,
             grids={n: metadata(*g) for n, g in zip(names, grids)}, clips=[])
    save(a.output, r)
    torch.set_num_threads(1)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    r["kernel_checks"] = check(grids)
    runs = [x for x in json.loads((a.validation / "INPUTS.json").read_text())["runs"]
            if x["config"] == "720p-600k-420"]
    for run in runs:
        path = a.validation / run["video"]
        assert digest(path) == run["video_sha256"] and run["stream"]["has_b_frames"] == 0
        demuxer = nvc.CreateDemuxer(filename=str(path))
        decoder = nvc.CreateDecoder(gpuid=0, codec=demuxer.GetNvCodecId(), cudacontext=0,
            cudastream=stream.cuda_stream, usedevicememory=True, outputColorType=nvc.OutputColorType.RGBP,
            latency=nvc.DisplayDecodeLatencyType.LOW)
        previous = torch.empty((3, 720, 1280), device="cuda", dtype=torch.uint8)
        flags = [torch.empty(a * b, device="cuda", dtype=torch.int32) for a, b in grids]
        frames = 0
        records = []
        for packet in demuxer:
            for frame in decoder.Decode(packet):
                tensor = torch.from_dlpack(frame)
                assert tensor.data_ptr() == frame.GetPtrToPlane(0) and tuple(tensor.shape) == (3, 720, 1280)
                if frames and frames % a.frame_step == 0:
                    for (rows, cols), flag in zip(grids, flags):
                        mark_dirty[(rows * cols,)](tensor, previous, flag, *tensor.stride(),
                            720, 1280, rows, cols, 36, 256)
                    masks = {}
                    for name, flag in zip(names, flags):
                        values = flag.cpu().tolist()
                        masks[name] = hex(sum(int(value) << i for i, value in enumerate(values)))
                    records.append(dict(frame=frames, dirty_masks=masks))
                previous.copy_(tensor)
                stream.synchronize()
                frames += 1
                del tensor
        assert frames == int(run["stream"]["nb_frames"])
        r["clips"].append(dict(clip=run["clip"], video_sha256=run["video_sha256"],
                              frames=frames, records=records))
        save(a.output, r)
        print(run["clip"], "decoded", frames, "pairs analyzed", len(records), flush=True)
        del decoder, demuxer, previous, frame
    summary = {}
    for name in names:
        counts = [int(x["dirty_masks"][name], 16).bit_count() for clip in r["clips"] for x in clip["records"]]
        work = np.asarray(counts) * r["grids"][name]["pixels_per_tile"] / (720 * 1280)
        summary[name] = dict(pairs=len(counts), mean_compute_pixel_ratio=float(work.mean()),
                            p50_compute_pixel_ratio=float(np.percentile(work, 50)),
                            p95_compute_pixel_ratio=float(np.percentile(work, 95)),
                            fraction_under_half_frame=float(np.mean(work < .5)),
                            fraction_under_full_frame=float(np.mean(work < 1.0)),
                            ideal_full_frame_fallback_mean_ratio=float(np.minimum(work, 1).mean()))
    r.update(complete=True, finished=time.time(), gpu_end=guard(), summary=summary)
    save(a.output, r)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
