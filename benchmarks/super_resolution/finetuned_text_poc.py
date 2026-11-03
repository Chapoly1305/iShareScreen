"""Low-bitrate text audit of existing grouped-fold checkpoints; no training.

Dense recordings use their held-out fold. Source-rendered PDF viewports report
all five folds separately and average scores, never cherry-pick a fold or claim
an ensemble image. Baselines include the real viewer shader on decoded planes.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import csv
import gc
import html
from io import BytesIO
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import av
import cv2
import numpy as np
from PIL import Image
import torch

from benchmarks.super_resolution.base_text_quality import ocr
from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement import candidates
from isharescreen.enhancement.models import project_home, sha256

MODELS = ("quicksrnet", "plainusr", "rt4ksr", "efrlfn")
TIMES = (2, 3)


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checkpoint_identity(folder, model, fold, folds_sha, pairs_sha):
    """Verify a selected checkpoint, including content-fold and data identity."""
    done, policy = read(folder / "COMPLETED.json"), read(folder / "protocol.json")
    require(
        (done["model"], done["fold"], done["epochs"]) == (model, fold, 1500),
        "Wrong completed model/fold",
    )
    require(
        policy["schema"] == "sr_candidate_true_epoch_protocol_v1",
        "Wrong training protocol family",
    )
    require(
        policy["schedule"]["model"] == model and policy["schedule"]["fold"] == fold,
        "Wrong protocol model/fold",
    )
    require(
        policy["folds_sha256"] == folds_sha and policy["pairs_sha256"] == pairs_sha,
        "Wrong frozen data/folds",
    )
    require(
        done["protocol_binding"] == fingerprint(policy), "Unbound checkpoint receipt"
    )
    path = folder / "best.pt"
    require(sha256(path) == done["checkpoint_sha256"], "Changed selected checkpoint")
    return dict(
        model=model,
        fold=fold,
        epoch=done["best_epoch"],
        step=done["best_step"],
        path=str(path.resolve()),
        sha256=done["checkpoint_sha256"],
        protocol_binding=done["protocol_binding"],
        official_sha256=policy["official_checkpoint_sha256"],
        protocol_sha256=sha256(folder / "protocol.json"),
        recorded_sources=policy["sources"],
    )


def aggregate_group(rows):
    """Pool regions within a scene/fold, then equal-weight folds and scenes."""
    scenes = []
    for scene in sorted({r["scene"] for r in rows}):
        selected = [r for r in rows if r["scene"] == scene]
        folds = []
        for fold in sorted(
            {r["fold"] for r in selected}, key=lambda f: -1 if f is None else f
        ):
            values = [r for r in selected if r["fold"] == fold]
            require(
                len({r["region"] for r in values}) == len(values),
                "Duplicate OCR region",
            )
            errors = sum(r["edit_distance"] for r in values)
            chars = sum(r["characters"] for r in values)
            folds.append(
                dict(fold=fold, errors=errors, characters=chars, cer=errors / chars)
            )
        scenes.append(
            dict(
                scene=scene, folds=folds, cer=float(np.mean([f["cer"] for f in folds]))
            )
        )
    return dict(
        cer=float(np.mean([s["cer"] for s in scenes])),
        scenes=scenes,
        worst_scene_cer=max(s["cer"] for s in scenes),
        worst_scene_fold_cer=max(f["cer"] for s in scenes for f in s["folds"]),
    )


def source_definitions(args):
    base = args.artifacts.resolve()
    return [
        dict(
            id="pdf",
            run=str(base / "pdf-single-width-poc-v1/run"),
            truths={
                str(t): str(
                    base
                    / "pdf-single-width-poc-v1"
                    / f"text-ground-truth{'-3s' if t == 3 else ''}.json"
                )
                for t in TIMES
            },
            fold_policy="All five folds; mean of fold scores, no ensemble or final-test claim",
        ),
        dict(
            id="dense",
            run=str(base / "base-sweep-dense-v1/run"),
            truths={
                str(t): str(
                    base
                    / "base-sweep-dense-v1"
                    / f"text-ground-truth{'-3s' if t == 3 else ''}.json"
                )
                for t in TIMES
            },
            fold_policy="Each recording uses exactly its held-out content fold",
        ),
    ]


def infer(args):
    import wgpu
    from isharescreen.frontend.desktop.gpu import Renderer
    from isharescreen.proxy.media.decode_common import _av_frame_to_tile

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "sources").mkdir()
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    folds_path = args.artifacts / "frozen-curated-v1/folds.json"
    pairs_path = args.artifacts / "frozen-curated-packed-v1/manifest.json"
    folds = read(folds_path)
    membership = {r["id"]: r["fold"] for r in folds["clips"]}
    checkpoints = []
    for model in args.models:
        for fold in range(5):
            found = []
            for p in args.runs.rglob("COMPLETED.json"):
                d = read(p)
                if (d.get("model"), d.get("fold")) == (model, fold):
                    found.append(p.parent)
            require(
                len(found) == 1, f"Missing or duplicate checkpoint for {model}/{fold}"
            )
            checkpoints.append(
                checkpoint_identity(
                    found[0], model, fold, sha256(folds_path), sha256(pairs_path)
                )
            )
    plan = dict(
        started=time.time(),
        scope="Development text POC of existing fine-tuned models; not a new training run, independent final test, live network, temporal stability or performance test.",
        command=sys.argv,
        device=args.device,
        precision="fp32",
        torch=torch.__version__,
        platform=platform.platform(),
        checkpoints=checkpoints,
        datasets=source_definitions(args),
        inputs=[],
        reference_records=[],
        records=[],
        gallery_pdf_fold=0,
        source_sha256={},
        eligible={},
        script_sha256=sha256(Path(__file__)),
    )
    for path in [
        Path(__file__),
        Path(ocr.__code__.co_filename),
        Path(candidates.__file__),
        Path("src/isharescreen/frontend/desktop/gpu.py"),
        Path("src/isharescreen/proxy/media/decode_common.py"),
    ]:
        path = path.resolve()
        plan["source_sha256"][str(path)] = sha256(path)
        (
            out / "sources" / str(path.relative_to(Path.cwd())).replace("/", "__")
        ).write_bytes(path.read_bytes())
    source_results = {}
    for ds in plan["datasets"]:
        run = Path(ds["run"])
        m = read(run / "manifest.json")
        r = read(run / "results.json")
        require(
            r["complete"]
            and read(run / "COMPLETED.json")["results_sha256"]
            == sha256(run / "results.json"),
            "Incomplete parent run",
        )
        require(
            r["manifest_sha256"] == sha256(run / "manifest.json"),
            "Parent manifest changed",
        )
        ds.update(
            manifest_sha256=sha256(run / "manifest.json"),
            results_sha256=sha256(run / "results.json"),
            truth_sha256={t: sha256(Path(p)) for t, p in ds["truths"].items()},
            scenes=m["scenes"],
        )
        source_results[ds["id"]] = r
        configs = {
            c["id"]: c
            for c in m["configs"]
            if c["group"] == "primary"
            and c["bps"] in args.budgets
            and c["height"] in args.heights
        }
        require(
            len(configs) == len(args.budgets) * len(args.heights),
            "Parent grid missing requested configuration",
        )
        for cfg in configs.values():
            values = [v for v in m["runs"] if v["config"] == cfg["id"]]
            plan["eligible"][ds["id"] + "__" + cfg["id"]] = dict(
                parameters=cfg,
                passed=all(v["budget_pass"] for v in values),
                max_actual_kbps=max(v["rate"]["retained_video_bps"] for v in values)
                / 1000,
                rates=[
                    dict(
                        scene=v["scene"],
                        actual_kbps=v["rate"]["retained_video_bps"] / 1000,
                        underflows=v["vbv_underflows"],
                    )
                    for v in values
                ],
            )
        for scene in m["scenes"]:
            actual_fold = (
                membership.get(scene["capture"]) if ds["id"] == "dense" else None
            )
            require(
                ds["id"] != "dense" or actual_fold is not None,
                "Recording has no held-out fold",
            )
            for t in TIMES:
                ref = run / scene["id"] / f"reference-{t}s.png"
                environment = read(
                    run / ("text" if t == 2 else "text-3s") / "environment.json"
                )
                require(
                    sha256(ref) == environment["references"][scene["id"]],
                    "Reference differs from audited parent OCR",
                )
                require(
                    ds["truth_sha256"][str(t)] == environment["truth_sha256"],
                    "Ground truth differs from audited parent OCR",
                )
                plan["reference_records"].append(
                    dict(
                        dataset=ds["id"],
                        scene=scene["id"],
                        time_s=t,
                        config="reference",
                        method="reference",
                        model="reference",
                        variant="reference",
                        fold=None,
                        image=str(ref),
                        sha256=sha256(ref),
                    )
                )
            for row in [
                v
                for v in m["runs"]
                if v["scene"] == scene["id"] and v["config"] in configs
            ]:
                cfg = configs[row["config"]]
                encoded = run / row["encoded"]
                require(sha256(encoded) == row["sha256"], "Encoded source changed")
                for t in TIMES:
                    image = run / scene["id"] / cfg["id"] / f"input-{t}s.png"
                    require(
                        sha256(image) == row["input_png_hashes"][str(t)],
                        "Input pixels changed",
                    )
                    plan["inputs"].append(
                        dict(
                            dataset=ds["id"],
                            scene=scene["id"],
                            config=cfg["id"],
                            time_s=t,
                            image=str(image),
                            sha256=sha256(image),
                            encoded=str(encoded),
                            encoded_sha256=row["sha256"],
                            width=cfg["width"],
                            height=cfg["height"],
                            fps=cfg["fps"],
                            folds=list(range(5))
                            if actual_fold is None
                            else [actual_fold],
                        )
                    )
    atomic_json(out / "plan.json", plan)
    gpu = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = gpu.request_device_sync()
    plan["renderer"] = dict(gpu.info)
    renderers = {}
    target = device.create_texture(
        size=(1920, 1080, 1),
        format="rgba8unorm",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    view = target.create_view()

    def save(row, method, model, variant, fold, pixels, **extra):
        require(
            pixels.shape == (1080, 1920, 3) and pixels.dtype == np.uint8,
            "Bad output geometry/type",
        )
        folder = (
            out
            / "images"
            / row["dataset"]
            / row["scene"]
            / row["config"]
            / f"{row['time_s']}s"
        )
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{method}.png"
        Image.fromarray(pixels).save(path)
        plan["records"].append(
            dict(
                dataset=row["dataset"],
                scene=row["scene"],
                config=row["config"],
                time_s=row["time_s"],
                method=method,
                model=model,
                variant=variant,
                fold=fold,
                image=str(path.relative_to(out)),
                sha256=sha256(path),
                **extra,
            )
        )

    by_encoded = defaultdict(list)
    for row in plan["inputs"]:
        by_encoded[row["encoded"]].append(row)
    for encoded, rows in by_encoded.items():
        wanted = {r["time_s"] * r["fps"]: r for r in rows}
        with av.open(encoded) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i not in wanted:
                    continue
                row = wanted[i]
                rgb = frame.to_ndarray(format="rgb24")
                require(
                    np.array_equal(
                        rgb, np.array(Image.open(row["image"]).convert("RGB"))
                    ),
                    "Re-decoded RGB pixels differ",
                )
                require(
                    not frame.is_corrupt
                    and frame.color_range == 2
                    and frame.colorspace == 1,
                    "Unexpected source color/corruption",
                )
                tile, error = _av_frame_to_tile(frame, [None], set())
                require(tile is not None and not error, "Cannot render decoded source")
                h = row["height"]
                w = row["width"]
                if h not in renderers:
                    renderers[h] = Renderer(device, "rgba8unorm", w, h)
                renderer = renderers[h]
                renderer.upload_tile(0, tile, tile.height)
                renderer.draw(view, 1920, 1080)
                pixels = np.frombuffer(
                    device.queue.read_texture(
                        {"texture": target},
                        {"bytes_per_row": 1920 * 4},
                        (1920, 1080, 1),
                    ),
                    np.uint8,
                ).reshape(1080, 1920, 4)
                save(
                    row,
                    "original-shader",
                    "original-shader",
                    "baseline",
                    None,
                    pixels[..., :3],
                )
                for name, kind in [
                    ("bilinear", cv2.INTER_LINEAR),
                    ("bicubic", cv2.INTER_CUBIC),
                ]:
                    save(
                        row,
                        name,
                        name,
                        "baseline",
                        None,
                        rgb
                        if h == 1080
                        else cv2.resize(rgb, (1920, 1080), interpolation=kind),
                    )
        atomic_json(out / "plan.json", plan)
    print("BASELINES", len(plan["records"]), flush=True)
    for key in args.models:
        for fold in [None, *range(5)]:
            adapter = candidates.TorchUpscaler(
                key, device=args.device, home=args.home, precision="fp32"
            )
            cp = None
            if fold is not None:
                cp = next(
                    c for c in checkpoints if c["model"] == key and c["fold"] == fold
                )
                data = Path(cp["path"]).read_bytes()
                import hashlib

                require(
                    hashlib.sha256(data).hexdigest() == cp["sha256"],
                    "Checkpoint changed before load",
                )
                payload = torch.load(
                    BytesIO(data), map_location="cpu", weights_only=True
                )
                require(
                    (
                        payload["model"],
                        payload["fold"],
                        payload["epoch"],
                        payload["step"],
                        payload["protocol_binding"],
                    )
                    == (key, fold, cp["epoch"], cp["step"], cp["protocol_binding"]),
                    "Checkpoint payload identity mismatch",
                )
                require(
                    adapter.provenance["checkpoint_sha256"] == cp["official_sha256"],
                    "Official initializer mismatch",
                )
                require(
                    all(
                        isinstance(v, torch.Tensor) and torch.isfinite(v).all().item()
                        for v in payload["state_dict"].values()
                    ),
                    "Nonfinite checkpoint",
                )
                torch.nn.Module.load_state_dict(
                    adapter.model, payload["state_dict"], strict=True
                )
            method = (
                f"{key}-pretrained" if fold is None else f"{key}-finetuned-fold-{fold}"
            )
            selected = [r for r in plan["inputs"] if fold is None or fold in r["folds"]]
            for j, row in enumerate(selected, 1):
                rgb = np.array(Image.open(row["image"]).convert("RGB"))
                pixels, _ = adapter(rgb)
                require(
                    pixels.shape == (row["height"] * 2, row["width"] * 2, 3),
                    "Wrong native x2 output",
                )
                if row["height"] != 540:
                    pixels = cv2.resize(
                        pixels, (1920, 1080), interpolation=cv2.INTER_AREA
                    )
                save(
                    row,
                    method,
                    key,
                    "pretrained" if fold is None else "finetuned",
                    fold,
                    pixels,
                    checkpoint_sha256=adapter.provenance["checkpoint_sha256"]
                    if cp is None
                    else cp["sha256"],
                    selected_epoch=0 if cp is None else cp["epoch"],
                )
                if j % 12 == 0:
                    atomic_json(out / "plan.json", plan)
                    print("INFER", method, j, "/", len(selected), flush=True)
            del adapter
            gc.collect()
            if args.device == "mps":
                torch.mps.empty_cache()
            atomic_json(out / "plan.json", plan)
            print("FINISHED", method, len(selected), flush=True)
    expected = len(plan["inputs"]) * (3 + len(args.models)) + sum(
        len(r["folds"]) * len(args.models) for r in plan["inputs"]
    )
    require(len(plan["records"]) == expected, "Missing output records")
    require(
        len(
            {
                (r["dataset"], r["scene"], r["config"], r["time_s"], r["method"])
                for r in plan["records"]
            }
        )
        == expected,
        "Duplicate output",
    )
    plan.update(
        finished=time.time(),
        sources_after={p: sha256(Path(p)) for p in plan["source_sha256"]},
        complete=True,
    )
    atomic_json(out / "plan.json", plan)
    atomic_json(
        out / "INFERRED.json",
        dict(plan_sha256=sha256(out / "plan.json"), records=expected),
    )


def score(args):
    out = args.output
    plan = read(out / "plan.json")
    require(
        plan.get("complete")
        and read(out / "INFERRED.json")["plan_sha256"] == sha256(out / "plan.json"),
        "Incomplete inference",
    )
    folder = out / "ocr"
    folder.mkdir(exist_ok=False)
    (folder / "runner.py").write_bytes(Path(ocr.__code__.co_filename).read_bytes())
    atomic_json(
        folder / "environment.json",
        dict(
            started=time.time(),
            command=sys.argv,
            driver_sha256=sha256(Path(__file__)),
            scorer_sha256=sha256(Path(ocr.__code__.co_filename)),
            plan_sha256=sha256(out / "plan.json"),
            tesseract=subprocess.check_output(
                ["tesseract", "--version"], stderr=subprocess.STDOUT, text=True
            ),
            english_model_sha256=sha256(
                Path("/opt/homebrew/share/tessdata/eng.traineddata")
            ),
            preprocessing="Native FHD crop; Lanczos x3; 12px white border; eng/OEM1/PSM6/dpi300; no text correction.",
            aggregation="Pool regions inside each scene/fold; average all folds inside a scene, then scenes equally.",
        ),
    )
    (folder / "driver.py").write_bytes(Path(__file__).read_bytes())
    tasks = []
    for ds in plan["datasets"]:
        for t in TIMES:
            truth = read(ds["truths"][str(t)])
            require(
                truth["time_s"] == t
                and sha256(Path(ds["truths"][str(t)])) == ds["truth_sha256"][str(t)],
                "Ground truth changed",
            )
            for region in truth["regions"]:
                reference = next(
                    r
                    for r in plan["reference_records"]
                    if r["dataset"] == ds["id"]
                    and r["scene"] == region["scene"]
                    and r["time_s"] == t
                )
                records = [reference] + [
                    r
                    for r in plan["records"]
                    if r["dataset"] == ds["id"]
                    and r["scene"] == region["scene"]
                    and r["time_s"] == t
                ]
                for row in records:
                    image = (
                        Path(row["image"])
                        if row["variant"] == "reference"
                        else out / row["image"]
                    )
                    require(sha256(image) == row["sha256"], "Changed reconstruction")
                    crop = (
                        folder
                        / "crops"
                        / ds["id"]
                        / row["scene"]
                        / row["config"]
                        / f"{t}s"
                        / f"{row['method']}-{region['id']}.png"
                    )
                    crop.parent.mkdir(parents=True, exist_ok=True)
                    tasks.append(
                        dict(
                            region,
                            image=str(image),
                            reference=reference["image"],
                            config=row["config"],
                            method=row["method"],
                            crop=str(crop),
                            root=str(out),
                            dataset=ds["id"],
                            time_s=t,
                            model=row["model"],
                            variant=row["variant"],
                            fold=row["fold"],
                        )
                    )
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for task, row in zip(tasks, pool.map(ocr, tasks)):
            row.update(
                {k: task[k] for k in ["dataset", "time_s", "model", "variant", "fold"]}
            )
            rows.append(row)
            if len(rows) % 40 == 0:
                print("OCR", len(rows), "/", len(tasks), flush=True)
    require(len(rows) == len(tasks), "Missing OCR rows")
    atomic_json(folder / "rows.json", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[
            (row["dataset"], row["time_s"], row["config"], row["model"], row["variant"])
        ].append(row)
    summary = []
    for (dataset, t, config, model, variant), values in groups.items():
        data = aggregate_group(values)
        for scene in data["scenes"]:
            expected_folds = 5 if dataset == "pdf" and variant == "finetuned" else 1
            require(
                len(scene["folds"]) == expected_folds, "Missing or cherry-picked fold"
            )
        item = dict(
            dataset=dataset,
            time_s=t,
            config=config,
            model=model,
            variant=variant,
            **data,
        )
        if config != "reference":
            item.update(plan["eligible"][dataset + "__" + config])
        summary.append(item)
    atomic_json(folder / "summary.json", summary)
    atomic_json(
        folder / "COMPLETED.json",
        dict(
            rows=len(rows),
            rows_sha256=sha256(folder / "rows.json"),
            summary_sha256=sha256(folder / "summary.json"),
            finished=time.time(),
        ),
    )


def choose(summary, dataset, bps):
    pool = []
    for r in summary:
        if (
            r["dataset"] != dataset
            or r["time_s"] != 3
            or r["config"] == "reference"
            or not r["passed"]
            or r["parameters"]["bps"] != bps
        ):
            continue
        earlier = next(
            x
            for x in summary
            if (x["dataset"], x["time_s"], x["config"], x["model"], x["variant"])
            == (dataset, 2, r["config"], r["model"], r["variant"])
        )
        pool.append(
            dict(
                r,
                cer_2s=earlier["cer"],
                cer_3s=r["cer"],
                max_sampled_cer=max(earlier["cer"], r["cer"]),
            )
        )
    return (
        min(pool, key=lambda r: (r["max_sampled_cer"], r["variant"] != "baseline"))
        if pool
        else None
    )


def plot_comparison(summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [
        ("original-shader", "baseline", "Original shader", "#374151", "-"),
        ("quicksrnet", "pretrained", "Quick pretrained", "#1d77b5", "--"),
        ("quicksrnet", "finetuned", "Quick fine-tuned", "#1d77b5", "-"),
        ("plainusr", "pretrained", "Plain pretrained", "#d76e13", "--"),
        ("plainusr", "finetuned", "Plain fine-tuned", "#d76e13", "-"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for i, dataset in enumerate(["pdf", "dense"]):
        for j, t in enumerate(TIMES):
            ax = axes[i, j]
            for model, variant, label, color, style in methods:
                values = sorted(
                    [
                        r
                        for r in summary
                        if r["dataset"] == dataset
                        and r["time_s"] == t
                        and r["model"] == model
                        and r["variant"] == variant
                        and r["parameters"]["bps"] == 600000
                    ],
                    key=lambda r: r["parameters"]["height"],
                )
                if values:
                    ax.plot(
                        [r["parameters"]["height"] for r in values],
                        [r["cer"] * 100 for r in values],
                        marker="o",
                        label=label,
                        color=color,
                        linestyle=style,
                    )
            ax.set_title(f"{dataset}: {t}s ({'keyframe' if t == 2 else 'inter frame'})")
            ax.set_xlabel("Base height (pixels); final output 1080p")
            ax.set_ylabel("Mean scene OCR CER (%) ↓")
            ax.set_xticks([540, 720, 1080])
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "600 Kbps requested HEVC video — identical inputs and FP32 adapters\nPDF: mean of all five folds; dense: held-out recording folds; panel scales differ"
    )
    fig.savefig(out / "600k-base-comparison.png", dpi=180)
    fig.savefig(out / "600k-base-comparison.svg")
    plt.close(fig)


def report(args):
    out = args.output
    plan = read(out / "plan.json")
    summary = read(out / "ocr/summary.json")
    receipt = read(out / "ocr/COMPLETED.json")
    require(
        plan.get("complete")
        and read(out / "INFERRED.json")["plan_sha256"] == sha256(out / "plan.json")
        and read(out / "ocr/environment.json")["plan_sha256"]
        == sha256(out / "plan.json"),
        "Inference/OCR identity changed",
    )
    require(
        receipt["summary_sha256"] == sha256(out / "ocr/summary.json"),
        "OCR summary changed",
    )
    require(
        receipt["rows_sha256"] == sha256(out / "ocr/rows.json"),
        "OCR predictions changed",
    )
    rows = read(out / "ocr/rows.json")
    plot_comparison(summary, out)
    choices = []
    # Config identity is kept as a string; encoder parameters are separate.
    for dataset in ["pdf", "dense"]:
        for bps in sorted(
            {
                r["parameters"]["bps"]
                for r in summary
                if r["dataset"] == dataset and r["config"] != "reference"
            }
        ):
            choices.append(
                dict(dataset=dataset, bps=bps, best=choose(summary, dataset, bps))
            )
    atomic_json(out / "selection.json", choices)
    lines = [
        "# Existing fine-tuned models: low-bandwidth text POC",
        "",
        "No new training or deployment changes. Same saved compressed inputs are reconstructed by bilinear, bicubic, the actual viewer Original shader, official pretrained models and existing 1500-epoch selected checkpoints. All outputs are 1920x1080. Model inference is FP32 on "
        + plan["device"]
        + ".",
        "",
        "Dense recordings use their held-out folds (paper 3, Markdown 1). Source-rendered PDF scenes score all five folds separately, then average fold scores within each page and pages equally. This is not an ensemble image, a selected best fold, or an independent final-test claim. Current PDF source layouts are development scenes; document-level independence across the legacy capture pool has not been established.",
        "",
        "Original-shader uses the shared native GPU renderer on the actual decoded full-range BT.709 420 planes in a single canvas, with RGBA8 readback. It is the renderer pixel-path baseline, not a live network capture or synchronized four-stream composition. Bilinear/bicubic operate on the same decoded RGB inputs.",
        "",
        "Primary metric is case-sensitive non-whitespace OCR CER. Case, digits and punctuation retained; whitespace/indentation and human reading performance excluded. Identical fixed crop + Lanczos x3 + white border, Tesseract eng OEM1/PSM6/dpi300. Clean reference floor is measured.",
        "",
        "| Dataset | Requested Kbps | Max actual video Kbps | Best eligible configuration/method | CER 2s % | CER 3s % | Worse sampled CER % |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for choice in choices:
        r = choice["best"]
        if r:
            lines.append(
                f"| {choice['dataset']} | {choice['bps'] / 1000:g} | {r['max_actual_kbps']:.1f} | {r['config']} / {r['model']} {r['variant']} | {r['cer_2s'] * 100:.3f} | {r['cer_3s'] * 100:.3f} | {r['max_sampled_cer'] * 100:.3f} |"
            )
    lines += [
        "",
        "![600 Kbps base-resolution comparison](600k-base-comparison.png)",
        "",
        "## Paired official-pretrained versus fine-tuned CER",
        "",
        "| Dataset | Config | Model | Pretrained 2s % | Fine-tuned 2s % | Pretrained 3s % | Fine-tuned 3s % | Encoder eligible |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    paired = []
    for row in summary:
        if row["variant"] != "finetuned" or row["time_s"] != 3:
            continue

        def find(variant, t):
            return next(
                r
                for r in summary
                if (r["dataset"], r["config"], r["model"], r["variant"], r["time_s"])
                == (row["dataset"], row["config"], row["model"], variant, t)
            )

        pre2, ft2, pre3 = (
            find("pretrained", 2),
            find("finetuned", 2),
            find("pretrained", 3),
        )
        paired.append(
            dict(
                dataset=row["dataset"],
                config=row["config"],
                model=row["model"],
                passed=row["passed"],
                pretrained_2s=pre2["cer"],
                finetuned_2s=ft2["cer"],
                pretrained_3s=pre3["cer"],
                finetuned_3s=row["cer"],
            )
        )
        lines.append(
            f"| {row['dataset']} | {row['config']} | {row['model']} | {pre2['cer'] * 100:.3f} | {ft2['cer'] * 100:.3f} | {pre3['cer'] * 100:.3f} | {row['cer'] * 100:.3f} | {row['passed']} |"
        )
    atomic_json(out / "paired.json", paired)
    lines += [
        "",
        "## Actual viewer baseline versus fine-tuning",
        "",
        "| Dataset | Config | Time | Original shader CER % | Quick pretrained % | Quick fine-tuned % | Plain pretrained % | Plain fine-tuned % |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    viewer_comparison = []
    for original in [r for r in summary if r["model"] == "original-shader"]:
        values = {}
        for model in ["quicksrnet", "plainusr"]:
            if model not in {r["model"] for r in summary}:
                continue
            for variant in ["pretrained", "finetuned"]:
                values[model + "_" + variant] = next(
                    r["cer"]
                    for r in summary
                    if (
                        r["dataset"],
                        r["config"],
                        r["time_s"],
                        r["model"],
                        r["variant"],
                    )
                    == (
                        original["dataset"],
                        original["config"],
                        original["time_s"],
                        model,
                        variant,
                    )
                )
        viewer_comparison.append(
            dict(
                dataset=original["dataset"],
                config=original["config"],
                time_s=original["time_s"],
                original_shader=original["cer"],
                **values,
            )
        )
        cells = [
            f"{values[k] * 100:.3f}" if k in values else "—"
            for k in [
                "quicksrnet_pretrained",
                "quicksrnet_finetuned",
                "plainusr_pretrained",
                "plainusr_finetuned",
            ]
        ]
        lines.append(
            f"| {original['dataset']} | {original['config']} | {original['time_s']}s | {original['cer'] * 100:.3f} | "
            + " | ".join(cells)
            + " |"
        )
    atomic_json(out / "viewer-comparison.json", viewer_comparison)
    lines += ["", "## Clean-reference OCR floor", ""]
    for r in summary:
        if r["variant"] == "reference":
            lines.append(f"- {r['dataset']}, {r['time_s']}s: {r['cer'] * 100:.3f}%.")
    critical = []
    for r in summary:
        if r["variant"] == "reference":
            continue
        blocks = [
            v
            for v in rows
            if (v["dataset"], v["time_s"], v["config"], v["model"], v["variant"])
            == (r["dataset"], r["time_s"], r["config"], r["model"], r["variant"])
        ]
        worst = max(blocks, key=lambda v: v["cer"])
        critical.append(
            dict(
                dataset=r["dataset"],
                time_s=r["time_s"],
                config=r["config"],
                model=r["model"],
                variant=r["variant"],
                mean_scene_cer=r["cer"],
                worst_block={
                    k: worst[k]
                    for k in [
                        "scene",
                        "region",
                        "fold",
                        "cer",
                        "digit_errors",
                        "digits",
                        "punctuation_errors",
                        "punctuation",
                        "crop_png",
                    ]
                },
                exact_blocks=sum(v["exact_nonspace"] for v in blocks),
                evaluated_blocks=len(blocks),
            )
        )
    atomic_json(out / "critical-blocks.json", critical)
    lines += [
        "",
        "## Worst individual block in each selected configuration",
        "",
        "Mean scores can hide damaged numbers or paths. These are maximum individual crop CERs across scenes and, for PDF fine-tuned models, all folds. Digit/punctuation counts are sequence-projection diagnostics, not aligned per-character classification.",
        "",
        "| Dataset | Kbps | Time | Worst block | Block CER % | Digit edits / GT digits | Punctuation edits / GT punctuation |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for choice in choices:
        best = choice["best"]
        if not best:
            continue
        for t in TIMES:
            item = next(
                v
                for v in critical
                if (v["dataset"], v["time_s"], v["config"], v["model"], v["variant"])
                == (
                    choice["dataset"],
                    t,
                    best["config"],
                    best["model"],
                    best["variant"],
                )
            )
            w = item["worst_block"]
            lines.append(
                f"| {choice['dataset']} | {choice['bps'] / 1000:g} | {t}s | {w['scene']}/{w['region']} | {w['cer'] * 100:.3f} | {w['digit_errors']}/{w['digits']} | {w['punctuation_errors']}/{w['punctuation']} |"
            )
    lines += [
        "",
        "Eligibility copies the parent encoder gate: both scenes in a dataset must pass retained video average <=105% requested and zero VBV underflow warnings. Transport/control overhead is excluded. Requested and actual video rates are separate; this is not a constrained-network feasibility proof. 2s is a GOP keyframe, 3s an inter frame; no long-term or interaction-latency claim. No input generation or source-model files are changed.",
        "",
        "[Every OCR prediction](ocr/rows.json) · [Fold-level summaries and measured rates](ocr/summary.json) · [Checkpoint/source/input identities](plan.json) · [Selection](selection.json)",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    page = [
        '<!doctype html><html><meta charset="utf-8"><title>Fine-tuned low-bandwidth text</title><style>body{font:16px/1.6 system-ui;max-width:1300px;margin:35px auto;padding:0 22px;color:#172334;background:#f5f7fa}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #cbd5e1}img{max-width:100%}a{color:#1766a2}</style><h1>Existing fine-tuned models — low-bandwidth text</h1><p>Same decoded inputs; source truth; held-out recording folds; PDF scores average all five folds. No new training, network or deployment claim.</p><p><a href="REPORT.md">Full report</a> · <a href="ocr/summary.json">All configurations and fold scores</a> · <a href="ocr/rows.json">Exact OCR outputs</a> · <a href="plan.json">Provenance</a></p><table><tr><th>Dataset</th><th>Kbps</th><th>Best at sampled times</th><th>CER 2s</th><th>CER 3s</th></tr>'
    ]
    for c in choices:
        r = c["best"]
        if r:
            page.append(
                f"<tr><td>{c['dataset']}</td><td>{c['bps'] / 1000:g}</td><td>{html.escape(str(r['config']))} / {r['model']} {r['variant']}</td><td>{r['cer_2s'] * 100:.3f}%</td><td>{r['cer_3s'] * 100:.3f}%</td></tr>"
            )
    page.append(
        '</table><p>Choose the worse of the two sampled mean-scene CERs. Baselines remain eligible. This is a small development experiment.</p><img src="600k-base-comparison.png" alt="600 Kbps base resolution and model CER comparison">'
    )
    page.append(
        '<p><a href="critical-blocks.json">Worst-block, digit, punctuation and exact-block diagnostics</a>: a low average can hide incorrect table numbers or paths. Inspect the individual crops before judging usefulness.</p><h2>600 Kbps text crops: fixed 540p and 720p comparisons</h2><p>Every selected region, both sampled times, all four scenes. Images below use the identical OCR enlargement. PDF fine-tuned images always use the predeclared fold 0; reported PDF scores average all five folds. Dense recordings use their held-out folds. Individual crop CER is shown, not the aggregate score.</p>'
    )
    gallery = []
    for original in rows:
        if original["method"] != "original-shader":
            continue
        cfg = plan["eligible"][original["dataset"] + "__" + original["config"]][
            "parameters"
        ]
        if cfg["bps"] != 600000 or cfg["height"] not in (540, 720):
            continue
        selected = []
        for method in [
            "reference",
            "original-shader",
            "quicksrnet-pretrained",
            "quicksrnet-finetuned",
            "plainusr-pretrained",
            "plainusr-finetuned",
        ]:
            matches = [
                r
                for r in rows
                if (r["dataset"], r["scene"], r["region"], r["time_s"])
                == (
                    original["dataset"],
                    original["scene"],
                    original["region"],
                    original["time_s"],
                )
                and (
                    r["config"] == original["config"]
                    or method == "reference"
                    and r["config"] == "reference"
                )
                and (
                    r["method"] == method
                    or method.endswith("-finetuned")
                    and r["model"] == method.removesuffix("-finetuned")
                    and r["variant"] == "finetuned"
                    and (r["dataset"] != "pdf" or r["fold"] == plan["gallery_pdf_fold"])
                )
            ]
            require(len(matches) <= 1, "Ambiguous gallery fold")
            if matches:
                selected.append(matches[0])
        title = f"{original['dataset']} / {original['scene']} / {original['region']} / {original['config']} / {original['time_s']}s"
        page.append(
            f"<details><summary>{html.escape(title)}</summary><pre>{html.escape(original['text_gt'])}</pre>"
        )
        for row in selected:
            require(
                sha256(out / row["crop_png"]) == row["crop_sha256"],
                "Gallery crop changed",
            )
            page.append(
                f'<h3>{html.escape(row["method"])} — crop CER {row["cer"] * 100:.3f}%</h3><a href="{row["crop_png"]}"><img loading="lazy" src="{row["crop_png"]}"></a><details><summary>Exact OCR prediction</summary><pre>{html.escape(row["text_predicted"])}</pre></details>'
            )
            gallery.append(
                dict(
                    dataset=row["dataset"],
                    config=original["config"],
                    scene=row["scene"],
                    region=row["region"],
                    time_s=row["time_s"],
                    method=row["method"],
                    crop_png=row["crop_png"],
                    sha256=row["crop_sha256"],
                )
            )
        page.append("</details>")
    atomic_json(out / "gallery.json", gallery)
    page.append("</html>")
    (out / "index.html").write_text("\n".join(page) + "\n")
    (out / "report-runner.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(
        out / "DELIVERY.json",
        dict(
            files={
                p: sha256(out / p)
                for p in [
                    "REPORT.md",
                    "index.html",
                    "600k-base-comparison.png",
                    "600k-base-comparison.svg",
                    "selection.json",
                    "paired.json",
                    "viewer-comparison.json",
                    "critical-blocks.json",
                    "gallery.json",
                    "plan.json",
                    "ocr/summary.json",
                    "ocr/rows.json",
                    "report-runner.py",
                ]
            },
            finished=time.time(),
        ),
    )
    print(json.dumps(choices, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["infer", "ocr", "report", "all"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--artifacts", type=Path, default=Path("benchmarks/super_resolution/artifacts")
    )
    p.add_argument(
        "--runs",
        type=Path,
        default=Path("benchmarks/super_resolution/artifacts/candidate-training-1500"),
    )
    p.add_argument("--home", type=Path, default=project_home())
    p.add_argument("--device", default="mps")
    p.add_argument(
        "--models", nargs="+", choices=MODELS, default=["quicksrnet", "plainusr"]
    )
    p.add_argument("--budgets", nargs="+", type=int, default=[256000, 600000, 1000000])
    p.add_argument("--heights", nargs="+", type=int, default=[540, 720, 1080])
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    args.output = args.output.resolve()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    for action in ["infer", "ocr", "report"] if args.action == "all" else [args.action]:
        dict(infer=infer, ocr=score, report=report)[action](args)


if __name__ == "__main__":
    main()
