"""Frozen three-arm continuation: original HR versus auxiliary ITSRN targets.

All arms share initialization, sampler, fresh optimizer and learning rate.
Teacher predictions are training-only auxiliary targets; the deployed EDSR
architecture and original HR truth are unchanged. Development selects checkpoints
only through the existing frozen OCR protocol, never through training gradients.
"""

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
import torch.nn.functional as F

from benchmarks.super_resolution.mac_readability import train_ablations as base
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
        steady = project / "datasets/derived/mac-steady-720p-420-v1"
        teacher = (
            project
            / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102/teacher-itsrn-steady-v1"
        )
        data = base.read(steady / "DATA.json")
        frozen = base.read(steady / "FROZEN.json")
        targets = base.read(teacher / "DATA.json")
        target_freeze = base.read(teacher / "FROZEN.json")
        assert data["complete"] and data["role"] == "train_only"
        assert targets["complete"] and targets["role"] == "train_only_auxiliary_targets"
        assert (
            base.sha(steady / "DATA.json")
            == frozen["data_sha256"]
            == targets["source_sha256"]
        )
        assert base.sha(teacher / "DATA.json") == target_freeze["data_sha256"]
        assert targets["binding"] == target_freeze["binding"]
        assert base.read(teacher / "SMOKE-PASSED.json")["passed"]
        self.steady_sha = frozen["data_sha256"]
        self.teacher_sha = target_freeze["data_sha256"]
        low_lookup = {r["id"]: r for r in data["rows"]}
        target_lookup = {r["id"]: r for r in targets["rows"]}
        assert len(low_lookup) == len(target_lookup) == len(self.rows) == 512
        assert set(low_lookup) == set(target_lookup) == {r["id"] for r in self.rows}
        self.pairs = []
        for original in self.rows:
            row = low_lookup[original["id"]]
            t = target_lookup[original["id"]]
            assert row["split"] == t["split"] == original["split"] == "train"
            assert row["hr_sha256"] == t["hr_sha256"] == original["hr_sha256"]
            assert (
                row["lr_sha256"] == t["lr_sha256"]
                and row["group_id"] == t["group_id"] == original["group_id"]
            )
            for role in ["lr", "hr"]:
                assert base.sha(steady / row[role]) == row[role + "_sha256"]
            assert base.sha(teacher / t["teacher"]) == t["teacher_sha256"]
            low = np.asarray(Image.open(steady / row["lr"]).convert("RGB")).copy()
            high = np.asarray(Image.open(steady / row["hr"]).convert("RGB")).copy()
            target = np.asarray(
                Image.open(teacher / t["teacher"]).convert("RGB")
            ).copy()
            assert low.shape == (720, 1280, 3) and high.shape == target.shape == (
                1080,
                1920,
                3,
            )
            self.pairs.append((low, high, target))

    def batch(self, attempt, seed, size=16):
        rng = np.random.default_rng(seed + attempt)
        lows = []
        highs = []
        targets = []
        for i in rng.choice(len(self.rows), size=size, replace=True):
            row = self.rows[i]
            box = row["transform"]["content_box"]
            lines = row["transform"]["line_boxes"]
            if lines and rng.random() < 0.8:
                box = lines[int(rng.integers(len(lines)))]
            cx = rng.uniform(box[0], box[2]) / 1.5
            cy = rng.uniform(box[1], box[3]) / 1.5
            x = int(np.clip(cx - 64, 0, 1152)) // 2 * 2
            y = int(np.clip(cy - 64, 0, 592)) // 2 * 2
            low, high, target = self.pairs[i]
            rx = x * 3 // 2
            ry = y * 3 // 2
            lows.append(low[y : y + 128, x : x + 128])
            highs.append(high[ry : ry + 192, rx : rx + 192])
            targets.append(target[ry : ry + 192, rx : rx + 192])
        tensor = lambda v: (
            torch.from_numpy(np.stack(v).transpose(0, 3, 1, 2).copy()).float() / 255
        )
        return tensor(lows), tensor(highs), tensor(targets)


def update(model, optimizer, scaler, weight, low, high, target):
    assert 0 <= weight <= 0.5
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values = np.zeros(3)
    for k in range(0, len(low), 4):
        x = low[k : k + 4].cuda()
        truth = high[k : k + 4].cuda()
        teacher = target[k : k + 4].cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            native = model(x * 255)
        pred = base.area_4_to_3(native.float() / 255)
        gt_loss = F.l1_loss(pred, truth)
        teacher_loss = F.l1_loss(pred, teacher)
        loss = (1 - weight) * gt_loss + weight * teacher_loss
        assert torch.isfinite(loss)
        fraction = len(x) / len(low)
        scaler.scale(loss * fraction).backward()
        values += (
            np.array([float(v.detach()) for v in [loss, gt_loss, teacher_loss]])
            * fraction
        )
    scaler.unscale_(optimizer)
    finite = all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for g in optimizer.param_groups
        for p in g["params"]
    )
    old = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    applied = scaler.get_scale() >= old
    assert not applied or finite
    return dict(
        loss=float(values[0]),
        rgb_l1=float(values[1]),
        teacher_l1=float(values[2]),
        applied=applied,
        finite=finite,
    )


def initialize(root, plan):
    wrapper = models.Upscaler("edsr", device="cuda:0", precision="amp-fp16")
    path = root / "selected-v2/checkpoint-001000.pt"
    assert base.sha(path) == plan["initialization"]["sha256"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["step"] == 1000 and payload["arm"] == "steady_l1"
    torch.nn.Module.load_state_dict(wrapper.model, payload["state_dict"], strict=True)
    return wrapper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args()
    root = (
        a.project
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102"
    )
    out = root / "training-v4"
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
    plan = base.read(root / "TRAINING-PLAN-V5.json")
    assert plan["batch"] == 16 and plan["microbatch"] == 4 and plan["updates"] == 1000

    def state(**kw):
        atomic_json(
            out / "STATUS.json", dict(pid=os.getpid(), updated=time.time(), **kw)
        )

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
        teacher_sha256=cache.teacher_sha,
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
    try:
        if not (out / "SMOKE-PASSED.json").exists():
            checks = []
            for arm in plan["arms"]:
                torch.manual_seed(plan["seed"])
                wrapper = initialize(root, plan)
                model = wrapper.model
                params = base.configure_parameters(model, arm["id"])
                fixed = base.frozen_tensors(model)
                optimizer = torch.optim.Adam(params, lr=plan["lr"])
                scaler = torch.amp.GradScaler("cuda", init_scale=1024)
                low, high, target = cache.batch(0, plan["seed"])
                before = [p.detach().clone() for p in params]
                result = update(
                    model, optimizer, scaler, arm["teacher_weight"], low, high, target
                )
                assert (
                    result["applied"]
                    and result["finite"]
                    and any(not torch.equal(x, y) for x, y in zip(before, params))
                )
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
                    expected = model(low[:1].cuda() * 255).cpu()
                saved = torch.load(path, map_location="cpu", weights_only=True)
                torch.nn.Module.load_state_dict(model, saved["state_dict"], strict=True)
                optimizer.load_state_dict(saved["optimizer"])
                scaler.load_state_dict(saved["scaler"])
                with torch.inference_mode():
                    actual = model(low[:1].cuda() * 255).cpu()
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
            wrapper = initialize(root, plan)
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
                )
                if step == 0:
                    original = base.read(root / "TRAINING-V2-SELECTED-IMAGES.json")
                    checks = []
                    for row in v["images"]:
                        old = next(
                            x
                            for x in original["images"]
                            if x["state_id"] == row["state_id"]
                        )
                        old_path = (
                            root
                            / "training-v2/models/steady_l1/validation-001000"
                            / Path(old["path"]).name
                        )
                        assert base.sha(old_path) == old["sha256"]
                        delta = np.abs(
                            np.asarray(Image.open(old_path)).astype("i2")
                            - np.asarray(
                                Image.open(
                                    folder / f"validation-{step:06d}" / row["path"]
                                )
                            ).astype("i2")
                        )
                        assert delta.max() <= 1
                        checks.append(
                            dict(
                                state_id=row["state_id"],
                                max_delta=int(delta.max()),
                                exact=bool(not delta.any()),
                            )
                        )
                    atomic_json(
                        folder / "INITIALIZATION-PARITY.json",
                        dict(passed=True, checks=checks),
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
                low, high, target = cache.batch(attempt, plan["seed"])
                attempt += 1
                result = update(
                    model, optimizer, scaler, arm["teacher_weight"], low, high, target
                )
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
