"""Conservative, local FiftyOne curation proposals; never launch or resume training."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from benchmarks.super_resolution.folds import validate
from benchmarks.super_resolution.prepare_curation import inside
from isharescreen.enhancement.models import sha256


def pixel_difference(a, b, tile=16):
    """Native-resolution RGB MAE and worst local tile, in uint8 intensity units."""
    import numpy as np
    if a.dtype != np.uint8 or b.dtype != np.uint8 or a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("Pixel comparison requires matching RGB uint8 arrays")
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
    height, width = diff.shape
    # Include partial edge tiles rather than cropping away screen borders.
    sums = np.add.reduceat(np.add.reduceat(diff, np.arange(0, height, tile), axis=0),
                           np.arange(0, width, tile), axis=1)
    counts = np.minimum(tile, height - np.arange(0, height, tile))[:, None] * \
             np.minimum(tile, width - np.arange(0, width, tile))[None, :]
    return dict(mae=float(diff.mean()), max_tile_mae=float((sums / counts).max()))


def safe_duplicate(differences, mean_limit, tile_limit):
    return all(d["mae"] <= mean_limit and d["max_tile_mae"] <= tile_limit
               for d in differences.values())


def nearest_representatives(array, ids, unique_ids, duplicate_ids, threshold):
    """Match tool-selected IDs using sklearn in the full, centered feature space.

    Brain 0.25.0 centers features again when querying a representative-only view.
    Its neighbors_map distances can therefore disagree even for identical vectors.
    Keep its selection, but independently match representatives and check distance.
    """
    import numpy as np
    from sklearn.metrics import pairwise_distances
    positions = {sid: i for i, sid in enumerate(ids)}
    if not duplicate_ids:
        return {}, {}
    unique = [positions[sid] for sid in unique_ids]
    duplicates = [positions[sid] for sid in duplicate_ids]
    if not unique:
        raise ValueError("Tool returned duplicates without representatives")
    centered = array.astype(np.float64) - array.astype(np.float64).mean(axis=0, keepdims=True)
    distances = pairwise_distances(centered[duplicates], centered[unique], metric="cosine")
    suggestions, scores = {}, {}
    for duplicate, distance_row in zip(duplicates, distances):
        nearest = int(np.argmin(distance_row))
        distance = float(distance_row[nearest])
        # Degenerate/zero features or numeric edge cases must not remove frames.
        if math.isfinite(distance) and distance <= threshold + 1e-7:
            suggestions[duplicate] = unique[nearest]
            scores[duplicate] = distance
    return suggestions, scores


def shared_references(rows):
    owners = defaultdict(set)
    for row in rows:
        owners[row["hr_rgb_sha256"]].add(row["group"])
    return {digest for digest, groups in owners.items() if len(groups) > 1}


def load_rgb(root, row, role):
    import numpy as np
    from PIL import Image
    path = inside(root, row[role])
    if sha256(path) != row[f"{role}_sha256"]:
        raise ValueError(f"Changed candidate image: {path}")
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError("Candidate is not an RGB image")
        rgb = np.array(image)
    if hashlib.sha256(rgb.tobytes()).hexdigest() != row[f"{role}_rgb_sha256"]:
        raise ValueError("Decoded candidate pixels changed")
    return rgb


def verify_candidates(data, root, plan, baseline):
    membership = {c["id"]: c for c in plan["clips"]}
    originals = {(r["capture"], r["frame"]): r for r in baseline["pairs"]}
    seen, found = set(), set()
    for row in data["candidates"]:
        capture, frame = row["capture"], row["frame"]
        if capture not in membership:
            raise ValueError("Unknown candidate capture")
        clip = membership[capture]
        if type(frame) is not int or not 0 <= frame < clip["frames"] or (capture, frame) in seen:
            raise ValueError("Duplicate or invalid candidate frame")
        if (row["group"], row["fold"]) != (clip["group"], clip["fold"]):
            raise ValueError("Candidate moved from its frozen group/fold")
        if not math.isclose(row["time_s"], float(Fraction(frame, 1) / Fraction(clip["fps"])), abs_tol=1e-9):
            raise ValueError("Candidate timestamp differs from source index")
        seen.add((capture, frame))
        original = originals.get((capture, frame))
        if row["baseline"] != (original is not None):
            raise ValueError("Candidate baseline identity changed")
        hr, lr = load_rgb(root, row, "hr"), load_rgb(root, row, "lr")
        if hr.shape != (2 * lr.shape[0], 2 * lr.shape[1], 3):
            raise ValueError("Candidate is not an aligned x2 pair")
        if original:
            found.add((capture, frame))
            for role in ("hr", "lr"):
                if original[f"{role}_rgb_sha256"] != row[f"{role}_rgb_sha256"]:
                    raise ValueError("Candidate changed frozen baseline pixels")
    if found != set(originals) or {capture for capture, _ in seen} != set(membership):
        raise ValueError("Candidates omit baseline frames or a source capture")


def embeddings(rows, root, output, device, batch_size, allow_download):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import models, transforms
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    cache = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    if not cache.exists() and not allow_download:
        raise FileNotFoundError("Embedding weights absent; pass --download-weights explicitly to download public weights")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    model = models.mobilenet_v3_small(weights=weights).eval().to(device)
    # Keep the WHOLE desktop rather than applying the classifier's center crop.
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            images = []
            for row in rows[start:start + batch_size]:
                with Image.open(inside(root, row["hr"])) as image:
                    images.append(transform(image.convert("RGB")))
            x = torch.stack(images).to(device)
            x = model.avgpool(model.features(x)).flatten(1)
            result.append(x.cpu().numpy())
            if start % (batch_size * 10) == 0:
                print(f"EMBEDDINGS {min(start + batch_size, len(rows))}/{len(rows)} device={device}", flush=True)
    array = np.concatenate(result)
    if not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) == 0).any():
        raise ValueError("Invalid embedding values")
    np.save(output / "embeddings.npy", array, allow_pickle=False)
    return array, dict(model="torchvision/mobilenet_v3_small", weights="IMAGENET1K_V1",
                       weights_url=weights.url, weights_sha256=sha256(cache), device=device,
                       preprocessing="whole RGB frame resized to 224x224, ImageNet normalization; no center crop",
                       limitation="Generic global embeddings are not text/detail fidelity metrics")


def decide(rows, suggestions, root, mean_limit, tile_limit, workers=1):
    """Accept a tool suggestion only when BOTH native-resolution paired images agree."""
    # Denser sampling can reveal shared targets absent from the old scan.
    # Quarantine new cross-group copies, but do not silently change old anchors.
    # Within ONE group, identical HR with meaningfully different LR can survive.
    protected = shared_references(rows)
    eligible = {i for i, r in enumerate(rows)
                if r["baseline"] or r["hr_rgb_sha256"] not in protected}
    paired = defaultdict(list)
    for i in sorted(eligible):
        row = rows[i]
        paired[row["capture"], row["hr_rgb_sha256"], row["lr_rgb_sha256"]].append(i)
    exact = {}
    for group in paired.values():
        representative = min(group, key=lambda i: (not rows[i]["baseline"], rows[i]["frame"]))
        exact.update({i: representative for i in group if i != representative and not rows[i]["baseline"]})
    # A tool representative may itself be an exact copy. Resolve it to the
    # retained paired-hash representative before applying native pixel guards.
    suggestions = {i: exact.get(reference, reference) for i, reference in suggestions.items()}
    # FiftyOne maps duplicates directly to retained, non-duplicate representatives.
    # Never drop a candidate in favor of an excluded or itself dropped reference.
    exact_representatives = set(exact.values())
    references = set(suggestions.values()) | exact_representatives
    def decision(item):
        i, row = item
        entry = dict(index=i, selected=True, reason="distinct_representative")
        if row["baseline"]:
            entry["reason"] = "frozen_baseline_anchor"
        elif i not in eligible:
            entry.update(selected=False, reason="quarantined_reference_overlap")
        elif i in exact:
            entry.update(selected=False, reason="exact_pair_duplicate", reference_index=exact[i])
        elif i in exact_representatives:
            entry["reason"] = "exact_pair_representative"
        elif i in suggestions and i not in references:
            reference = suggestions[i]
            if reference in eligible:
                other = rows[reference]
                if row["capture"] != other["capture"]:
                    raise ValueError("Duplicate suggestion crosses captures")
                differences = {role: pixel_difference(load_rgb(root, row, role), load_rgb(root, other, role))
                               for role in ("hr", "lr")}
                duplicate = safe_duplicate(differences, mean_limit, tile_limit)
                entry.update(selected=not duplicate,
                             reason="near_duplicate_confirmed" if duplicate else "native_detail_or_degradation_preserved",
                             reference_index=reference, differences=differences)
        return entry
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(decision, enumerate(rows)))


def write_report(rows, decisions, output, root):
    from PIL import Image, ImageDraw
    groups = defaultdict(list)
    for decision in decisions:
        groups[rows[decision["index"]]["capture"]].append(decision)
    overview = []
    pages = []
    for ordinal, (capture, group) in enumerate(sorted(groups.items())):
        summary = dict(capture=capture, fold=rows[group[0]["index"]]["fold"],
                       content=rows[group[0]["index"]]["content"], candidates=len(group),
                       selected=sum(d["selected"] for d in group),
                       baseline=sum(rows[d["index"]]["baseline"] for d in group),
                       reasons=dict(Counter(d["reason"] for d in group)))
        summary["new_selected"] = summary["selected"] - summary["baseline"]
        overview.append(summary)
        # Preview each decision type, spread across the clip, not just its opening.
        preview = []
        for reason in sorted({d["reason"] for d in group}):
            cases = [d for d in group if d["reason"] == reason]
            indices = sorted({round(i * (len(cases) - 1) / 2) for i in range(3)})
            preview.extend(cases[i] for i in indices)
        sheet = Image.new("RGB", (960, math.ceil(len(preview) / 3) * 224), "#20242a")
        draw = ImageDraw.Draw(sheet)
        for j, decision in enumerate(preview):
            row = rows[decision["index"]]
            with Image.open(inside(root, row["hr"])) as image:
                image = image.convert("RGB"); image.thumbnail((320, 180))
                x, y = (j % 3) * 320, (j // 3) * 224
                sheet.paste(image, (x, y))
            label = "KEEP" if decision["selected"] else "SKIP"
            draw.text((x + 4, y + 182), f"{label} frame {row['frame']} / {row['time_s']:.3f}s", fill="white")
            draw.text((x + 4, y + 198), decision["reason"][:43], fill="#a8d8ff")
        filename = f"contact-{ordinal:02d}.jpg"
        sheet.save(output / filename, quality=90)
        pages.append(f"<h2>{html.escape(capture)}</h2><p>{html.escape(str(summary))}</p>"
                     f'<a href="{filename}"><img width="960" src="{filename}" loading="lazy"></a>')
    headers = ["capture", "fold", "content", "candidates", "baseline", "new_selected", "selected"]
    table = "<table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    for summary in overview:
        table += "<tr>" + "".join(f"<td>{html.escape(str(summary[h]))}</td>" for h in headers) + "</tr>"
    table += "</table>"
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Frame curation review</title>'
            '<style>body{font:15px system-ui;margin:32px;background:#fafafa;color:#20242a}'
            'td,th{padding:6px 12px;border:1px solid #ccc}table{border-collapse:collapse}'
            'img{max-width:100%;height:auto}p{max-width:1100px;overflow-wrap:anywhere}</style>'
            '<h1>Frame curation review — proposal only</h1>'
            '<p>Original training/validation data and all baseline frames remain unchanged. '
            'Previews are lossy viewing copies, never training targets. Near duplicates are '
            'screened within each capture, with native HR and LR pixel guards. '
            'Quarantined overlap is not a quality failure. No OCR, model training, or quality improvement claim.</p>'
            + table + "".join(pages) + "</html>")
    (output / "index.html").write_text(page)
    return overview


def run(args):
    started = time.monotonic()
    data = json.loads(args.candidates.read_text())
    plan = validate(json.loads(args.folds.read_text()))
    baseline_path = Path(data["baseline_manifest"])
    baseline = json.loads(baseline_path.read_text())
    if (sha256(args.folds) != data["folds_sha256"] or sha256(baseline_path) != data["baseline_sha256"]
            or plan["manifest_sha256"] != data["manifest_sha256"]
            or sha256(Path(data["source_manifest"])) != data["manifest_sha256"]):
        raise ValueError("Candidate/frozen input provenance mismatch")
    rows, root = data["candidates"], args.candidates.parent
    verify_candidates(data, root, plan, baseline)
    args.output.mkdir(parents=True, exist_ok=False)
    # Isolate Mongo state and disable telemetry BEFORE importing FiftyOne.
    os.environ["FIFTYONE_DO_NOT_TRACK"] = "true"
    os.environ["FIFTYONE_DATABASE_DIR"] = str((args.output / "database").resolve())
    # Ignore a user's default remote database configuration for this LOCAL tool.
    config_path = args.output / "fiftyone-config.json"
    config_path.write_text(json.dumps(dict(database_dir=os.environ["FIFTYONE_DATABASE_DIR"],
                                          do_not_track=True)) + "\n")
    os.environ["FIFTYONE_CONFIG_PATH"] = str(config_path.resolve())
    os.environ.pop("FIFTYONE_DATABASE_URI", None)
    import fiftyone as fo
    import fiftyone.brain as fob
    if args.embedding_cache:
        import numpy as np
        cached = json.loads((args.embedding_cache / "proposal.json").read_text())
        cached_audit = json.loads(args.cache_audit.read_text())
        cached_array = args.embedding_cache / "embeddings.npy"
        if (cached["candidates_sha256"] != sha256(args.candidates)
                or sha256(args.embedding_cache / "proposal.json") != cached_audit["proposal_sha256"]
                or sha256(cached_array) != cached_audit["embeddings_sha256"]):
            raise ValueError("Embedding cache is not bound to the audited candidate manifest")
        array = np.load(cached_array, allow_pickle=False)
        if array.shape != (len(rows), 576) or not np.isfinite(array).all():
            raise ValueError("Embedding cache shape or values changed")
        embedding_info = dict(cached["embedding"], reused_from=str(args.embedding_cache.resolve()),
                              cache_audit_sha256=sha256(args.cache_audit))
        np.save(args.output / "embeddings.npy", array, allow_pickle=False)
        print(f"REUSED {len(rows)} hash-audited embeddings; no model inference", flush=True)
    else:
        array, embedding_info = embeddings(rows, root, args.output, args.device,
                                           args.batch_size, args.download_weights)
    dataset = fo.Dataset(name="desktop-frame-curation-" + args.output.name)
    dataset.persistent = True
    samples = [fo.Sample(filepath=str(inside(root, r["hr"])),
                         source_frame=r["frame"], capture=r["capture"], fold=r["fold"],
                         content_group=r["group"], baseline=r["baseline"], candidate_index=i)
               for i, r in enumerate(rows)]
    dataset.add_samples(samples)
    ids = [s.id for s in samples]
    by_capture = defaultdict(list)
    for i, row in enumerate(rows):
        by_capture[row["capture"]].append(i)
    suggestions, distances = {}, {}
    for capture, group in by_capture.items():
        view = dataset.select([ids[i] for i in group], ordered=True)
        index = fob.compute_near_duplicates(view, embeddings=array[group].copy(), threshold=args.threshold,
                                           skip_failures=False)
        local, scores = nearest_representatives(array[group], [ids[i] for i in group],
                                                index.unique_ids, index.duplicate_ids, args.threshold)
        suggestions.update({group[i]: group[reference] for i, reference in local.items()})
        distances.update({group[i]: distance for i, distance in scores.items()})
        print(f"SIMILARITY {capture}: {len(index.duplicate_ids)}/{len(group)} suggestions", flush=True)
    decisions = decide(rows, suggestions, root, args.mean_limit, args.tile_limit, workers=args.workers)
    for entry in decisions:
        if entry["index"] in distances:
            entry["embedding_distance"] = distances[entry["index"]]
    selected = [r for r, d in zip(rows, decisions) if d["selected"]]
    dataset.set_values("curation_selected", [d["selected"] for d in decisions])
    dataset.set_values("curation_reason", [d["reason"] for d in decisions])
    overview = write_report(rows, decisions, args.output, root)
    result = dict(schema="sr_curation_proposal_v1", training_ready=False,
                  status="proposal_requires_visual_review", dataset_name=dataset.name,
                  candidates_manifest=str(args.candidates.resolve()), candidates_sha256=sha256(args.candidates),
                  folds_sha256=data["folds_sha256"], baseline_sha256=data["baseline_sha256"],
                  manifest_sha256=data["manifest_sha256"], generator_sha256=sha256(Path(__file__)),
                  selected=selected, decisions=decisions, overview=overview,
                  embedding=embedding_info, parameters=dict(threshold=args.threshold,
                  mean_limit=args.mean_limit, tile_limit=args.tile_limit, tile_size=16,
                  metric="cosine after per-capture mean centering; independently checked representative matching"),
                  reasons=dict(Counter(d["reason"] for d in decisions)),
                  versions={name: importlib.metadata.version(name) for name in
                            ("fiftyone", "fiftyone-brain", "torch", "torchvision", "numpy", "Pillow", "scikit-learn")},
                  elapsed_s=time.monotonic() - started,
                  limitations=["No downstream training comparison; no demonstrated model-quality benefit",
                               "Candidate scan is denser sampling, not exhaustive frame selection",
                               "Generic embeddings plus conservative pixel guards; not text correctness or difficulty scores",
                               "Existing baseline anchors preserved even when denser scan reveals cross-group overlap",
                               "Validation frame membership must remain frozen in any future per-fold training dataset"])
    (args.output / "proposal.json").write_text(json.dumps(result, indent=2) + "\n")
    additions = [r for r in selected if not r["baseline"]]
    fold_plan = [dict(held_out_fold=fold,
                     training_additions=[dict(capture=r["capture"], frame=r["frame"])
                                         for r in additions if r["fold"] != fold],
                     validation_unchanged=True,
                     validation_frames=[dict(capture=r["capture"], frame=r["frame"])
                                        for r in baseline["pairs"] if r["fold"] == fold])
                 for fold in range(plan["n_folds"])]
    (args.output / "fold-additions.json").write_text(json.dumps(fold_plan, indent=2) + "\n")
    print(f"PROPOSED {len(selected)}/{len(rows)} frames, including {len(additions)} additions; training unchanged", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=.005)
    p.add_argument("--mean-limit", type=float, default=.25)
    p.add_argument("--tile-limit", type=float, default=1.0)
    p.add_argument("--download-weights", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--embedding-cache", type=Path, help="Prior review directory; requires its completed automated audit")
    p.add_argument("--cache-audit", type=Path)
    args = p.parse_args()
    if bool(args.embedding_cache) != bool(args.cache_audit):
        p.error("embedding-cache and cache-audit must be supplied together")
    if args.batch_size < 1 or args.workers < 1 or not all(math.isfinite(v) and v >= 0 for v in
            (args.threshold, args.mean_limit, args.tile_limit)):
        p.error("Batch size must be positive and thresholds must be finite/nonnegative")
    run(args)


if __name__ == "__main__":
    main()
