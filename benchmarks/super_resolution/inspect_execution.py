"""Audit Core ML planned placement and GPU-only Torch forward tensor devices."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from benchmarks.super_resolution.training_state import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("Start Python with PYTORCH_ENABLE_MPS_FALLBACK=0")
    import numpy as np
    from PIL import Image
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves
    from isharescreen.enhancement import candidates, coreml, execution, models, variants
    torch.set_num_threads(1)
    rgb = np.array(Image.open(args.image).convert("RGB"))
    if rgb.shape != (540,960,3):
        raise ValueError("Use RGB 960x540 input")
    report = dict(input_sha256=models.sha256(args.image), mps_cpu_fallback="0",
                  torch_forwards=[], coreml_plans=[], scope="Torch operator tensor-device audit; Core ML anticipated plans, not runtime traces")

    class GPUAudit(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            values = tree_leaves((args, kwargs or {}))
            if any(isinstance(v, torch.Tensor) and v.device.type != "mps" for v in values):
                raise ValueError(f"Non-MPS input inside model forward: {func}")
            output = func(*args, **(kwargs or {}))
            if any(isinstance(v, torch.Tensor) and v.device.type != "mps" for v in tree_leaves(output)):
                raise ValueError(f"Non-MPS output inside model forward: {func}")
            self.operations[str(func)] += 1
            return output

    for key in (*candidates.IDS, "edsr", "imdn", "safmn"):
        adapter = (candidates.TorchUpscaler(key, device="mps") if key in candidates.IDS
                   else variants.load(key, device="mps"))
        execution.validate_adapter(adapter)
        array = rgb.astype(np.float32)
        if key != "edsr":
            array /= 255
        value = torch.from_numpy(np.ascontiguousarray(array.transpose(2,0,1)[None])).to("mps")
        audit = GPUAudit()
        with torch.inference_mode(), audit:
            output = adapter.model(value)
        torch.mps.synchronize()
        if not torch.isfinite(output).all().item():
            raise ValueError(f"Nonfinite output: {key}")
        row = dict(model=key, device=str(output.device), operations=dict(audit.operations),
                   cpu_tensor_operations=0, provenance=adapter.provenance)
        report["torch_forwards"].append(row)
        print(f"GPU VERIFIED {key}: {sum(audit.operations.values())} operations, no CPU tensors", flush=True)
        del adapter, output, value
        torch.mps.empty_cache()
        atomic_json(args.output, report)

    from coremltools.models.compute_plan import MLComputePlan
    for key in ("quicksrnet", *coreml.CANDIDATE_IDS):
        adapter = coreml.CoreMLUpscaler(candidate=None if key == "quicksrnet" else key)
        plan = MLComputePlan.load_from_path(adapter.model.get_compiled_model_path())
        operations = []

        def visit(block):
            for op in block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                operations.append(dict(operator=op.operator_name,
                    preferred_device=type(usage.preferred_compute_device).__name__ if usage else None))
                for nested in op.blocks:
                    visit(nested)

        for function in plan.model_structure.program.functions.values():
            visit(function.block)
        counts = Counter(row["preferred_device"] or "unassigned" for row in operations)
        report["coreml_plans"].append(dict(model=key, operations=operations, counts=dict(counts),
            evidence="anticipated compute plan; not a runtime hardware trace"))
        print(f"CORE ML PLAN {key}: {dict(counts)}", flush=True)
        del adapter, plan
        atomic_json(args.output, report)


if __name__ == "__main__":
    main()
