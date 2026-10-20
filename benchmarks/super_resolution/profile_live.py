"""Profile warmed live adapters and optionally locate synchronized leaf hotspots."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import time

import numpy as np

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement import variants
from isharescreen.enhancement.models import sha256


def measure(adapter, rgb, warmup, iterations):
    if warmup < 1 or iterations < 2:
        raise ValueError("Use at least one warm-up and two measured frames")
    for _ in range(warmup):
        adapter(rgb)
    rows = [adapter(rgb)[1] for _ in range(iterations)]
    return dict(samples=rows, p50_ms={k: float(np.median([row[k] for row in rows])) for k in rows[0]},
                p95_total_ms=float(np.percentile([row["total_ms"] for row in rows], 95)))


def leaf_hotspots(adapter, rgb):
    """Diagnostic only: barriers perturb execution, and functional ops are omitted."""
    costs, starts, handles = defaultdict(float), {}, []
    if adapter.model is None or not hasattr(adapter.model, "named_modules"):
        return {}
    try:
        for name, module in adapter.model.named_modules():
            if any(module.children()):
                continue

            def before(module, inputs, name=name):
                adapter.sync()
                starts[name] = time.perf_counter()

            def after(module, inputs, output, name=name):
                adapter.sync()
                costs[type(module).__name__] += (time.perf_counter() - starts[name]) * 1000

            handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))
        adapter(rgb)
    finally:
        for handle in handles:
            handle.remove()
    return dict(sorted(costs.items(), key=lambda row: -row[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=list(variants.MODELS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--leaf-hotspots", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh report path")
    if args.warmup < 1 or args.iterations < 2:
        parser.error("Use at least one warm-up and two measured frames")
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    result = dict(schema="sr_live_profile_v1", platform=platform.platform(), torch=torch.__version__,
                  generator_sha256=sha256(Path(__file__)), records=[],
                  scope="Warmed adapter latency; excludes decode, tile assembly and display. Not sustained viewer FPS.",
                  leaf_scope="Synchronized leaf-module diagnostics omit functional ops and perturb execution.")
    for key in args.models:
        started = time.perf_counter()
        adapter = variants.load(key, device=args.device)
        load_ms = (time.perf_counter() - started) * 1000
        for path in args.images:
            with Image.open(path) as image:
                rgb = np.array(image.convert("RGB"))
            record = dict(model=key, image=path.name, image_sha256=sha256(path), shape=list(rgb.shape),
                          provenance=dict(adapter.provenance), load_ms=load_ms,
                          **measure(adapter, rgb, args.warmup, args.iterations))
            if args.leaf_hotspots:
                record["synchronized_leaf_ms"] = leaf_hotspots(adapter, rgb)
            result["records"].append(record)
            print(json.dumps(dict(model=key, image=str(path), p50_ms=record["p50_ms"])), flush=True)
        del adapter
        if args.device == "mps":
            torch.mps.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
