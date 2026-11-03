"""Measure ITSRN encoder/query cost and exact per-frame reuse on RTX4090.

Only redundant feature unfolding and identical coordinate-grid construction are
cached. Caches are cleared for every frame. Original architecture/weights stay
unchanged. This isolated single-thread experiment never patches production code.
"""

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

from benchmarks.super_resolution.mac_readability.itsrn_probe import run, save, sha


@contextmanager
def frame_cache(module, enabled):
    old_unfold = module.F.unfold
    old_coord = module.make_coord
    cache = {}
    counts = dict(
        unfold_calls=0, unfold_computations=0, coord_calls=0, coord_computations=0
    )

    def unfold(x, *args, **kwargs):
        counts["unfold_calls"] += 1
        if not enabled:
            counts["unfold_computations"] += 1
            return old_unfold(x, *args, **kwargs)
        if "unfold" not in cache:
            cache["input"] = x
            cache["unfold_args"] = (args, kwargs)
            cache["unfold"] = old_unfold(x, *args, **kwargs)
            counts["unfold_computations"] += 1
        else:
            assert cache["input"] is x and cache["unfold_args"] == (args, kwargs)
        return cache["unfold"]

    def coord(shape, ranges=None, flatten=True):
        counts["coord_calls"] += 1
        key = (tuple(shape), ranges, flatten)
        if not enabled:
            counts["coord_computations"] += 1
            return old_coord(shape, ranges, flatten)
        if "coord" not in cache:
            cache["coord_key"] = key
            cache["coord"] = old_coord(shape, ranges, flatten).cuda()
            counts["coord_computations"] += 1
        else:
            assert cache["coord_key"] == key
        return cache["coord"]

    module.F.unfold = unfold
    module.make_coord = coord
    try:
        yield counts
    finally:
        module.F.unfold = old_unfold
        module.make_coord = old_coord
        cache.clear()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    r = a.root.resolve()
    out = r / "itsrn-cost-v1"
    out.mkdir(exist_ok=False)
    repo = r / "external-models/ITSRN"
    weight = r / "external-models/ITSRN_last.pth"
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    assert "GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a" in gpu
    owners = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    assert not any(
        x.strip().isdigit() and int(x) != os.getpid() for x in owners.splitlines()
    )
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert (
        sha(weight)
        == "2f3fadbebc0cbdc4583b4935929570655101bf34de2a77fd2fa6771478daf98b"
    )
    sys.path.insert(0, str(repo / "code"))
    import models
    from utils import make_coord

    module = importlib.import_module("models.ITNSR")
    payload = torch.load(weight, map_location="cpu", weights_only=True)
    model = models.make(payload["model"])
    model.load_state_dict(payload["model"]["sd"], strict=True)
    model = model.cuda().eval().requires_grad_(False)
    original_gen = model.gen_feat
    events = []

    def timed_gen(x):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_gen(x)
        end.record()
        events.append((start, end))
        return result

    model.gen_feat = timed_gen
    inp = r / "dense-inference-inputs/INPUTS.json"
    index = json.loads(inp.read_text())
    record = dict(
        schema="itsrn_cost_decomposition_v1",
        complete=False,
        started=time.time(),
        script_sha256=sha(Path(__file__)),
        probe_sha256=sha(Path(__file__).with_name("itsrn_probe.py")),
        inputs_sha256=sha(inp),
        checkpoint_sha256=sha(weight),
        host=socket.gethostname(),
        gpu=gpu,
        scope=__doc__,
        samples=[],
        comparisons=[],
        test_used=False,
    )
    save(out / "RESULTS.json", record)
    # One document from each category, fixed before any quality scores.
    for sid in [1, 9, 17, 25]:
        row = next(x for x in index["images"] if x["state_id"] == sid)
        path = inp.parent / row["input_path"]
        assert sha(path) == row["decoded_sha256"]
        rgb = np.asarray(Image.open(path).convert("RGB"))
        for precision in ["fp32", "amp-fp16"]:
            reference = None
            for cached in [False, True]:
                for rep in range(3):
                    events.clear()
                    with frame_cache(module, cached) as counts:
                        pixels, timing = run(model, rgb, make_coord, precision)
                    assert len(events) == 1
                    encoder = events[0][0].elapsed_time(events[0][1])
                    sample = dict(
                        state_id=sid,
                        precision=precision,
                        cached=cached,
                        repetition=rep,
                        warm=rep > 0,
                        encoder_cuda_ms=encoder,
                        query_and_dispatch_ms=timing["inference_ms"] - encoder,
                        counts=counts,
                        **timing,
                    )
                    record["samples"].append(sample)
                    if not cached:
                        reference = pixels
                    else:
                        difference = np.abs(
                            pixels.astype("i2") - reference.astype("i2")
                        )
                        record["comparisons"].append(
                            dict(
                                state_id=sid,
                                precision=precision,
                                repetition=rep,
                                max_delta=int(difference.max()),
                                mean_delta=float(difference.mean()),
                                exact=bool(np.array_equal(pixels, reference)),
                            )
                        )
                        assert np.array_equal(pixels, reference), (
                            "Cache changed output pixels"
                        )
                    save(out / "RESULTS.json", record)
                    print(
                        sid,
                        precision,
                        "cached",
                        cached,
                        "rep",
                        rep,
                        "encoder_ms",
                        round(encoder, 2),
                        "total_ms",
                        round(timing["total_ms"], 2),
                        flush=True,
                    )
    record.update(complete=True, finished=time.time())
    save(out / "RESULTS.json", record)


if __name__ == "__main__":
    main()
