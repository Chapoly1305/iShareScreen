"""Count exact adjacent RGB frame repeats after actual GPU decoding.

This is dataset characterization, not a latency benchmark. Pixels stay on GPU;
only one changed-component count per frame is read for the analysis record.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import PyNvVideoCodec as nvc

from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    r = dict(complete=False, started=time.time(), gpu=guard(), scope=__doc__,
             source_sha256=digest(Path(__file__)), nvc=nvc.__version__, clips=[])
    save(a.output, r)
    torch.set_num_threads(1)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
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
        changes = []

        def consume(frames):
            for frame in frames:
                tensor = torch.from_dlpack(frame)
                assert tensor.data_ptr() == frame.GetPtrToPlane(0) and tuple(tensor.shape) == (3, 720, 1280)
                count = int(torch.count_nonzero(tensor != previous).item()) if changes else None
                changes.append(count)
                previous.copy_(tensor)
                # Decoder-owned storage must remain live until the comparison
                # and previous-frame copy finish. This analysis is not timed.
                stream.synchronize()

        for packet in demuxer:
            consume(decoder.Decode(packet))
        # Installed PyNvVideoCodec 2.2.4 has no low-level Flush method.
        # These frozen streams have no B-frames and use LOW latency. Require
        # the full container frame count after demuxing; never silently omit
        # buffered frames or estimate repetition from an incomplete sequence.
        assert len(changes) == int(run["stream"]["nb_frames"]), (run["clip"], len(changes))
        repeats = sum(x == 0 for x in changes if x is not None)
        row = dict(clip=run["clip"], video_sha256=run["video_sha256"], frames=len(changes),
                   exact_repeats=repeats, repeat_fraction=repeats / len(changes),
                   changed_rgb_components=changes)
        r["clips"].append(row)
        save(a.output, r)
        print(run["clip"], len(changes), "repeats", repeats, f"{row['repeat_fraction']:.3%}", flush=True)
        del decoder, demuxer, previous
    frames = sum(x["frames"] for x in r["clips"])
    repeats = sum(x["exact_repeats"] for x in r["clips"])
    r.update(complete=True, finished=time.time(), gpu_end=guard(), total_frames=frames,
             exact_repeats=repeats, repeat_fraction=repeats / frames)
    save(a.output, r)


if __name__ == "__main__":
    main()
