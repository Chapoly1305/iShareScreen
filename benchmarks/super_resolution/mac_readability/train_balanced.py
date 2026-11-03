"""Separate two-arm native-glyph sampling experiment; preserve v1 andv2 queues."""

import argparse
import fcntl
import os
from pathlib import Path
import platform
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from benchmarks.super_resolution.mac_readability import train_ablations as base
from benchmarks.super_resolution.mac_readability import stroke_loss
from benchmarks.super_resolution.training_state import (
    atomic_json,
    atomic_weights,
    cpu_state,
    fingerprint,
)
from isharescreen.enhancement import models


class Cache:
    def __init__(self, project):
        self.rows, self.bindings = base.training_data(project)
        root = project / "datasets/derived/mac-steady-720p-420-v1"
        data = base.read(root / "DATA.json")
        frozen = base.read(root / "FROZEN.json")
        assert (
            data["complete"]
            and data["role"] == "train_only"
            and base.sha(root / "DATA.json") == frozen["data_sha256"]
        )
        assert base.read(root / "SMOKE-VALIDATION.json")["passed"]
        self.steady_sha = frozen["data_sha256"]
        lookup = {r["id"]: r for r in data["rows"]}
        self.pairs = []
        for old in self.rows:
            row = lookup[old["id"]]
            assert row["split"] == "train" and row["hr_sha256"] == old["hr_sha256"]
            for role in ["hr", "lr"]:
                assert base.sha(root / row[role]) == row[role + "_sha256"]
            codec = np.asarray(
                Image.open(old["root"] / old["lr"]).convert("RGB")
            ).copy()
            steady = np.asarray(Image.open(root / row["lr"]).convert("RGB")).copy()
            high = np.asarray(Image.open(root / row["hr"]).convert("RGB")).copy()
            assert codec.shape == steady.shape == (720, 1280, 3) and high.shape == (
                1080,
                1920,
                3,
            )
            self.pairs.append((codec, steady, high))
        assert len(self.pairs) == len(lookup) == 512

    def batch(self, arm, attempt, seed):
        rng = np.random.default_rng(seed + attempt)
        mix = np.random.default_rng(seed + 100000000 + attempt)
        lows = []
        highs = []
        probability = arm["native_glyph_probability"]
        native = np.array([r["dataset"] == base.DATASETS[1] for r in self.rows])
        assert native.sum() == 128
        weights = np.where(native, probability / 128, (1 - probability) / 384)
        for i in rng.choice(512, size=16, replace=True, p=weights):
            row = self.rows[i]
            box = row["transform"]["content_box"]
            lines = row["transform"]["line_boxes"]
            if lines and rng.random() < 0.8:
                box = lines[int(rng.integers(len(lines)))]
            cx = rng.uniform(box[0], box[2]) / 1.5
            cy = rng.uniform(box[1], box[3]) / 1.5
            x = int(np.clip(cx - 64, 0, 1152)) // 2 * 2
            y = int(np.clip(cy - 64, 0, 592)) // 2 * 2
            codec, steady, high = self.pairs[i]
            low = steady if mix.random() < arm["steady_probability"] else codec
            lows.append(low[y : y + 128, x : x + 128])
            highs.append(
                high[y * 3 // 2 : y * 3 // 2 + 192, x * 3 // 2 : x * 3 // 2 + 192]
            )
        tensor = lambda values: (
            torch.from_numpy(np.stack(values).transpose(0, 3, 1, 2).copy()).float()
            / 255
        )
        return tensor(lows), tensor(highs)


def update(model, optimizer, scaler, arm, low, high):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values = np.zeros(3)
    for k in range(0, 16, 4):
        x = low[k : k + 4].to("cuda:0")
        truth = high[k : k + 4].to("cuda:0")
        with torch.autocast("cuda", dtype=torch.float16):
            native = model(x * 255)
        pred = base.area_4_to_3(native.float() / 255)
        loss, rgb, structure = stroke_loss.objective(pred, truth, arm["structural"])
        assert torch.isfinite(loss)
        scaler.scale(loss / 4).backward()
        values += np.array([float(v.detach()) for v in [loss, rgb, structure]]) / 4
    scaler.unscale_(optimizer)
    finite = all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for group in optimizer.param_groups
        for p in group["params"]
    )
    old = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    applied = scaler.get_scale() >= old
    assert not applied or finite
    return dict(
        loss=float(values[0]),
        rgb_l1=float(values[1]),
        structure_loss=float(values[2]),
        applied=applied,
        finite=finite,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args()
    root = (
        a.project
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102"
    )
    out = root / "training-v3"
    out.mkdir(exist_ok=True)
    lock = (out / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == base.GPU_UUID
    gpu = base.verify_device()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    plan = base.read(root / "TRAINING-PLAN-V4.json")
    assert base.sha(Path(stroke_loss.__file__)) == plan["loss_code_sha256"]
    assert plan["batch"] == 16 and plan["microbatch"] == 4 and plan["updates"] == 1500
    status = out / "STATUS.json"

    def state(**kw):
        atomic_json(status, dict(pid=os.getpid(), updated=time.time(), **kw))

    state(state="loading_train_pairs")
    cache = Cache(a.project)
    cases, source_hash = base.validation_inputs(
        root / "dense-inference-inputs",
        a.project
        / "datasets/sources/mac-readability-v2-20261102/dense-development-v1/reference",
    )
    for train_key, source_key in [
        ("group_id", "group_id"),
        ("repository", "repository"),
        ("source_text_sha256", "source_sha256"),
    ]:
        held = {
            r["state"]["origin"][source_key]
            for r in cases
            if source_key in r["state"]["origin"]
        }
        assert not {r[train_key] for r in cache.rows} & held
    policy = dict(
        plan=plan,
        runner_sha256=base.sha(Path(__file__)),
        base_sha256=base.sha(Path(base.__file__)),
        model_runner_sha256=base.sha(Path(models.__file__)),
        data=cache.bindings,
        steady_sha256=cache.steady_sha,
        source_sha256=source_hash,
    )
    binding = fingerprint(policy)
    if (out / "PROTOCOL.json").exists():
        assert base.read(out / "PROTOCOL.json") == policy
    else:
        atomic_json(out / "PROTOCOL.json", policy)
    atomic_json(
        out / "HOST.json",
        dict(
            hostname=platform.node(),
            gpu=gpu,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            pid=os.getpid(),
        ),
    )
    baseline = base.read(root / "dense-models-v1/RESULTS.json")
    baseline["root"] = root / "dense-models-v1"
    try:
        if not (out / "SMOKE-PASSED.json").exists():
            checks = []
            for arm in plan["arms"]:
                torch.manual_seed(plan["seed"])
                wrapper = models.Upscaler("edsr", device="cuda:0", precision="amp-fp16")
                model = wrapper.model
                params = base.configure_parameters(model, arm["id"])
                fixed = base.frozen_tensors(model)
                optimizer = torch.optim.Adam(params, lr=plan["lr"])
                scaler = torch.amp.GradScaler("cuda", init_scale=1024)
                low, high = cache.batch(arm, 0, plan["seed"])
                before = [p.detach().clone() for p in params]
                result = update(model, optimizer, scaler, arm, low, high)
                assert result["applied"] and result["finite"]
                assert any(not torch.equal(x, y) for x, y in zip(before, params))
                base.assert_frozen(model, fixed)
                folder = out / "smoke" / arm["id"]
                folder.mkdir(parents=True)
                path = folder / "checkpoint.pt"
                atomic_weights(
                    path,
                    dict(
                        state_dict=cpu_state(model),
                        optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(),
                    ),
                )
                model.eval()
                with torch.inference_mode():
                    expected = model(low[:1].to("cuda:0") * 255).cpu()
                saved = torch.load(path, map_location="cpu", weights_only=True)
                torch.nn.Module.load_state_dict(model, saved["state_dict"], strict=True)
                optimizer.load_state_dict(saved["optimizer"])
                scaler.load_state_dict(saved["scaler"])
                with torch.inference_mode():
                    actual = model(low[:1].to("cuda:0") * 255).cpu()
                assert torch.equal(expected, actual)
                v = base.evaluate(
                    wrapper,
                    cases,
                    source_hash,
                    folder / "validation",
                    arm["id"],
                    1,
                    smoke=True,
                )
                checks.append(
                    dict(
                        arm=arm["id"],
                        finite_update=True,
                        parameters_changed=True,
                        frozen_parameters_unchanged=True,
                        checkpoint_reload_exact=True,
                        images=len(v["images"]),
                        loss=result,
                    )
                )
                del wrapper, model, params, optimizer, before
                torch.cuda.empty_cache()
                print("SMOKE", arm["id"], "passed", flush=True)
            atomic_json(
                out / "SMOKE-PASSED.json",
                dict(binding=binding, passed=True, checks=checks),
            )
        assert base.read(out / "SMOKE-PASSED.json")["binding"] == binding
        if a.smoke_only:
            state(state="smoke_passed")
            return
        for arm in plan["arms"]:
            folder = out / "models" / arm["id"]
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "COMPLETED.json").exists():
                assert base.read(folder / "COMPLETED.json")["binding"] == binding
                continue
            torch.manual_seed(plan["seed"])
            wrapper = models.Upscaler("edsr", device="cuda:0", precision="amp-fp16")
            model = wrapper.model
            params = base.configure_parameters(model, arm["id"])
            fixed = base.frozen_tensors(model)
            optimizer = torch.optim.Adam(params, lr=plan["lr"])
            scaler = torch.amp.GradScaler("cuda", init_scale=1024)
            step = attempt = skipped = 0
            history = []
            validations = []
            if (folder / "last.pt").exists():
                saved = torch.load(
                    folder / "last.pt", map_location="cpu", weights_only=True
                )
                assert saved["binding"] == binding
                torch.nn.Module.load_state_dict(model, saved["state_dict"], strict=True)
                optimizer.load_state_dict(saved["optimizer"])
                scaler.load_state_dict(saved["scaler"])
                step = saved["step"]
                attempt = saved["attempt"]
                skipped = saved["skipped"]
                history = saved["history"]
                validations = saved["validations"]
                base.assert_frozen(model, fixed)

            def checkpoint():
                base.assert_frozen(model, fixed)
                atomic_weights(
                    folder / "last.pt",
                    dict(
                        binding=binding,
                        step=step,
                        attempt=attempt,
                        skipped=skipped,
                        state_dict=cpu_state(model),
                        optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(),
                        history=history,
                        validations=validations,
                    ),
                )

            def validate():
                state(state="validating", arm=arm["id"], step=step)
                path = folder / f"checkpoint-{step:06d}.pt"
                atomic_weights(
                    path,
                    dict(
                        binding=binding,
                        step=step,
                        arm=arm["id"],
                        state_dict=cpu_state(model),
                    ),
                )
                v = base.evaluate(
                    wrapper,
                    cases,
                    source_hash,
                    folder / f"validation-{step:06d}",
                    arm["id"],
                    step,
                    baseline=baseline,
                )
                validations.append(
                    dict(
                        step=step,
                        macro_psnr=v["macro_psnr"],
                        performance=v["performance"],
                        checkpoint_sha256=base.sha(path),
                    )
                )
                checkpoint()
                base.curves(folder, history, validations)

            if step in plan["checkpoint_steps"] and step not in {
                v["step"] for v in validations
            }:
                validate()
            while step < plan["updates"]:
                start = time.monotonic()
                low, high = cache.batch(arm, attempt, plan["seed"])
                attempt += 1
                result = update(model, optimizer, scaler, arm, low, high)
                if not result["applied"]:
                    skipped += 1
                    assert skipped <= 20
                    continue
                step += 1
                history.append(
                    dict(
                        step=step,
                        attempt=attempt,
                        **result,
                        update_wall_s=time.monotonic() - start,
                    )
                )
                if step % 100 == 0:
                    state(state="training", arm=arm["id"], step=step)
                    checkpoint()
                    base.curves(folder, history, validations)
                    print(arm["id"], step, result["loss"], flush=True)
                if step in plan["checkpoint_steps"]:
                    validate()
            atomic_json(
                folder / "COMPLETED.json",
                dict(
                    binding=binding,
                    updates=step,
                    skipped_updates=skipped,
                    checkpoints=validations,
                    selection_pending="FrozenOCR",
                ),
            )
            wrapper = model = params = optimizer = None
            torch.cuda.empty_cache()
        state(state="training_completed", ocr_selection_pending=True)
    except BaseException as error:
        state(state="failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
