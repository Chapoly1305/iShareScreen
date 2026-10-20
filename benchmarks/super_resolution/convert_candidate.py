"""Convert official x2 candidates and validate CPU agreement before installation."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from isharescreen.enhancement import candidates, coreml
from isharescreen.enhancement.models import project_home, sha256


def convert(key, images, home, units="ALL", precision="fp16"):
    if key not in coreml.CANDIDATE_IDS:
        raise ValueError("Use install_coreml for QuickSRNet; other conversions require supported native x2 candidates")
    import torch
    import coremltools as ct
    from PIL import Image
    spec = candidates.specification(key)
    if spec["scale"] != 2:
        raise ValueError("Live conversion requires official x2 weights; do not resize x4 outputs")
    if len(images) < 3:
        raise ValueError("Provide at least three distinct validation images")
    hashes = [sha256(path) for path in images]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Validation images must be distinct")
    root = candidates.asset_home(key, home)
    target = root / "coreml"
    if target.exists() or target.is_symlink():
        raise ValueError("Core ML installation already exists; preserve it and use a different asset home")
    model, spec = candidates.load_network(key, home)
    values = []
    for path in images:
        rgb = np.array(Image.open(path).convert("RGB"))
        if rgb.shape != (540, 960, 3):
            raise ValueError("Validation inputs must be RGB 960x540")
        values.append(np.ascontiguousarray((rgb.astype(np.float32)/255).transpose(2, 0, 1)[None]))
    with torch.inference_mode():
        traced = torch.jit.trace(model, torch.from_numpy(values[0]), check_trace=False)
    with tempfile.TemporaryDirectory(prefix=".coreml-", dir=root) as temp:
        staged = Path(temp)/"bundle"
        staged.mkdir()
        package = staged/"model.mlpackage"
        converted = ct.convert(traced, inputs=[ct.TensorType(name="image", shape=(1,3,540,960), dtype=np.float32)],
            outputs=[ct.TensorType(name="output", dtype=np.float32)],
            compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS13, skip_model_load=True)
        converted.save(str(package))
        engine = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, units))
        validation = []
        for digest, value in zip(hashes, values):
            with torch.inference_mode():
                reference = model(torch.from_numpy(value)).numpy()
            actual = engine.predict({"image": value})["output"]
            if actual.shape != (1, 3, 1080, 1920) or not np.isfinite(actual).all():
                raise ValueError("Core ML conversion returned invalid pixels")
            ref_rgb = np.clip(np.rint(reference*255), 0, 255).astype(np.uint8)
            actual_rgb = np.clip(np.rint(actual*255), 0, 255).astype(np.uint8)
            delta = np.abs(ref_rgb.astype(np.int16)-actual_rgb.astype(np.int16))
            row = dict(image_sha256=digest, max_rgb_delta=int(delta.max()), mean_rgb_delta=float(delta.mean()))
            print(json.dumps(dict(model=key, validation=row)), flush=True)
            if row["max_rgb_delta"] > 1:
                raise ValueError(f"{key} {units} conversion exceeds one RGB level: {row}")
            validation.append(row)
        manifest = dict(schema="isharescreen_candidate_coreml_v1", model=key, scale=2,
            source_revision=spec["revision"], official_checkpoint_sha256=spec["files"]["weights.pth"]["sha256"],
            input=[1,3,540,960], output=[1,3,1080,1920], precision=precision, compute_units=units,
            parameters=sum(p.numel() for p in model.parameters()), coremltools=ct.__version__,
            generator_sha256=sha256(Path(__file__)), validation=validation,
            package_sha256={p.relative_to(package).as_posix(): sha256(p) for p in package.rglob("*") if p.is_file()})
        (staged/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        staged.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=coreml.CANDIDATE_IDS, required=True)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--home", type=Path, default=project_home())
    parser.add_argument("--compute-units", choices=("ALL", "CPU_AND_GPU", "CPU_AND_NE"), default="ALL")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    print(f"VALIDATED {args.model}: {convert(args.model, args.images, args.home, args.compute_units, args.precision)} ({time.monotonic()-started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
