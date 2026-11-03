"""Fill explicitly rejected layouts and freeze exactly 10,000 accepted views."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

from benchmarks.super_resolution.text_corpus import render
from benchmarks.super_resolution.text_corpus.catalog import document_family, Union
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def content(job):
    return {k: v for k, v in job.items() if k != "id"}


def run(args):
    root = args.root.resolve()
    original = render.read(root / "render-plan.json")
    primary = render.read(root / "views.json")
    if primary["plan_sha256"] != sha256(root / "render-plan.json") or primary[
        "processed"
    ] != len(original["jobs"]):
        raise ValueError("Primary render has not finished its entire requested scope")
    if original["script_sha256"] != sha256(Path(render.__file__)):
        raise ValueError("Frozen renderer changed")
    rows = primary["views"].copy()
    target = Counter(original["counts"])
    counts = Counter(r["kind"] for r in rows)
    seen = {r["hr_rgb_sha256"] for r in rows}
    rejected = primary["rejected"].copy()
    path = root / "supplement-plan.json"
    if not path.exists():
        candidates = render.make_plan(
            root, dict(paper=8000, specification=1800, code=3000)
        )
        old = {render.stable_order(content(j)) for j in original["jobs"]}
        jobs = []
        for kind, requested in target.items():
            need = requested - counts[kind]
            if need <= 0:
                continue
            pool = [
                j
                for j in candidates["jobs"]
                if j["kind"] == kind and render.stable_order(content(j)) not in old
            ]
            selected = pool[: need * 5 + 30]
            if len(selected) < need:
                raise ValueError("Insufficient replacement layouts")
            for j in selected:
                j["id"] = "extra-" + render.stable_order(content(j))[:24]
            jobs.extend(selected)
        atomic_json(
            path,
            dict(
                primary_sha256=sha256(root / "views.json"),
                renderer_sha256=sha256(Path(render.__file__)),
                jobs=jobs,
            ),
        )
    supplement = render.read(path)
    if supplement["primary_sha256"] != sha256(root / "views.json"):
        raise ValueError("Primary layouts changed")
    sources = original["source_map"]
    pdf = [j for j in supplement["jobs"] if j["kind"] in ("paper", "specification")]

    def accept(job, row):
        if counts[job["kind"]] >= target[job["kind"]]:
            return
        if row["hr_rgb_sha256"] in seen:
            rejected.append(dict(id=job["id"], reason="duplicate_render_pixels"))
            return
        rows.append(row)
        seen.add(row["hr_rgb_sha256"])
        counts[job["kind"]] += 1

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(render.pdf_view, (str(root), j, sources[j["source"]]))
            for j in pdf
        ]
        for n, (job, future) in enumerate(zip(pdf, futures), 1):
            try:
                accept(job, future.result())
            except (ValueError, RuntimeError) as error:
                rejected.append(dict(id=job["id"], reason=str(error)))
            if n % 50 == 0:
                print("SUPPLEMENT", n, "/", len(pdf), "accepted", len(rows), flush=True)
    html_jobs = [
        j for j in supplement["jobs"] if j["kind"] not in ("paper", "specification")
    ]
    for job, row in zip(html_jobs, render.html_views(root, html_jobs, sources)):
        accept(job, row)
    if counts != target:
        raise ValueError(f"Accepted strata short: {counts} != {target}")
    aliases = Union()
    for source in sources.values():
        old = source["source_group"]
        aliases.find(old)
        if source["kind"] != "code":
            aliases.join(
                old,
                document_family(source["title"], source.get("doi") or "", source["id"]),
            )
    group_aliases = {
        s["source_group"]: aliases.find(s["source_group"]) for s in sources.values()
    }
    splits = {}
    for row in rows:
        group = group_aliases.get(row["source_group"], row["source_group"])
        if group in splits and splits[group] != row["split"]:
            raise ValueError("Source group leaks across splits")
        splits[group] = row["split"]
    result = dict(
        schema="isharescreen_text_views_v1",
        complete=True,
        views=rows,
        count=len(rows),
        counts=dict(counts),
        split_counts=dict(Counter(r["split"] for r in rows)),
        groups=len(splits),
        rejected=rejected,
        group_aliases=group_aliases,
        primary_sha256=sha256(root / "views.json"),
        supplement_sha256=sha256(path),
        sources_sha256=sha256(root / "sources.json"),
        scope="10,000 unique rendered RGB views, not 10,000 independent documents. Same-source pages/variants share a split. Legacy videos are retained separately.",
        finished=time.time(),
    )
    if len(rows) != 10000 or len(seen) != 10000:
        raise ValueError("Expected exactly 10,000 unique views")
    atomic_json(root / "dataset-views.json", result)
    print(
        result["counts"], result["split_counts"], "groups", result["groups"], flush=True
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    run(p.parse_args())


if __name__ == "__main__":
    main()
