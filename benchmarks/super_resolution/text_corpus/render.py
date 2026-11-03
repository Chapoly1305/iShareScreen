"""Render source-bound FHD desktop views and geometric text labels.

PDF labels are extracted from the same page/clip transformation as the pixels.
HTML contains escaped source text, is offline, and uses measured browser boxes.
No OCR output is used as a training label.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image, ImageDraw

from benchmarks.super_resolution.text_corpus.catalog import split_for, text_identity
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256
from benchmarks.super_resolution.text_corpus.runtime_paths import chrome

WIDTH, HEIGHT = 1920, 1080
CHROME = chrome()


def read(path):
    return json.loads(path.read_text())


def stable_order(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def round_robin(groups):
    groups = [list(v) for v in groups]
    for i in range(max(map(len, groups), default=0)):
        for group in groups:
            if i < len(group):
                yield group[i]


def make_plan(root, counts):
    catalog = read(root / "sources.json")
    sources = catalog["sources"]
    jobs = []
    for kind, count in counts.items():
        groups = []
        for src in sorted(
            [s for s in sources if s["kind"] == kind],
            key=lambda s: stable_order(s["id"]),
        ):
            variants = []
            if kind in ("paper", "specification"):
                for page in sorted(
                    src["pages"], key=lambda p: stable_order((src["id"], p["page"]))
                ):
                    for width in [1840, 1120, 1480, 816]:
                        span = 1000 / (width / page["width"])
                        maximum = max(0, page["height"] - span)
                        for fraction in [0, 0.5, 1] if maximum > 30 else [0]:
                            variants.append(
                                dict(
                                    source=src["id"],
                                    kind=kind,
                                    page=page["page"],
                                    page_width=width,
                                    top_points=round(maximum * fraction, 4),
                                )
                            )
                cap = 240 if kind == "specification" else 48
            elif kind == "code":
                for offset in range(0, src["lines"] - 16, 26):
                    for theme in ["light", "dark"]:
                        variants.append(
                            dict(
                                source=src["id"],
                                kind=kind,
                                line_start=offset,
                                theme=theme,
                                font_size=[14, 16, 18, 20, 24][
                                    int(
                                        stable_order((src["id"], offset, theme))[:8], 16
                                    )
                                    % 5
                                ],
                            )
                        )
                cap = 10
            else:
                raise ValueError(kind)
            # Spread views over pages/locations rather than taking adjacent prefixes.
            variants.sort(key=stable_order)
            groups.append(variants[:cap])
        available = list(round_robin(groups))
        if len(available) < count:
            raise ValueError(
                f"Not enough unique {kind} views: {len(available)} < {count}"
            )
        jobs.extend(available[:count])
    # New rendered web/table/terminal documents have explicit source strings.
    for i in range(1000):
        theme = "dark" if i % 3 == 0 else "light"
        if i % 2 == 0:
            choices = sorted(
                [s for s in sources if s["kind"] == "paper"],
                key=lambda s: stable_order(s["id"]),
            )
            source = choices[(i // 2) % len(choices)]
            jobs.append(
                dict(
                    source=source["id"],
                    kind="web",
                    page=source["pages"][(i // len(choices)) % len(source["pages"])][
                        "page"
                    ],
                    theme=theme,
                    font_size=[14, 16, 18, 22, 26][i % 5],
                    variant=i,
                )
            )
        else:
            # Different strings per document; every view of a document stays in one split.
            family = "generated-document:" + str(i)
            jobs.append(
                dict(
                    source=None,
                    kind="table" if i % 4 == 1 else "terminal",
                    theme=theme,
                    font_size=[14, 16, 18, 20, 24][i % 5],
                    variant=i,
                    source_group=family,
                    split=split_for(family, catalog["seed"]),
                )
            )
    source_map = {s["id"]: s for s in sources}
    for j, job in enumerate(jobs):
        if job["source"]:
            src = source_map[job["source"]]
            job.update(source_group=src["source_group"], split=src["split"])
        job["id"] = f"view-{j:05d}-" + stable_order(job)[:12]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate render identity")
    return dict(
        schema="isharescreen_text_views_plan_v1",
        created=time.time(),
        sources_sha256=sha256(root / "sources.json"),
        script_sha256=sha256(Path(__file__)),
        canvas=[WIDTH, HEIGHT],
        seed=catalog["seed"],
        jobs=jobs,
        counts=dict(Counter(j["kind"] for j in jobs)),
        splits=dict(Counter(j["split"] for j in jobs)),
        truth="Fully visible source words/tokens with screenshot-space boxes. PDF text-layer extraction may preserve ligatures/reading-order quirks; no OCR pseudo-labels.",
        source_map=source_map,
    )


def save_view(root, job, pixels, words, metadata):
    if pixels.shape != (HEIGHT, WIDTH, 3) or pixels.dtype != np.uint8:
        raise ValueError("Invalid FHD RGB pixels")
    if len(words) < 30 or sum(len(w["text"]) for w in words) < 160:
        raise ValueError("Insufficient visible source text")
    for word in words:
        x, y, w, h = word["xywh"]
        if not (
            word["text"]
            and 0 <= x < WIDTH
            and 0 <= y < HEIGHT
            and w > 0
            and h > 0
            and x + w <= WIDTH + 0.01
            and y + h <= HEIGHT + 0.01
        ):
            raise ValueError("Text geometry outside rendered canvas")
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    for word in words:
        x, y, w, h = word["xywh"]
        mask[
            max(0, int(y)) : min(HEIGHT, math.ceil(y + h)),
            max(0, int(x)) : min(WIDTH, math.ceil(x + w)),
        ] = 1
    coverage = float(mask.mean())
    if coverage < 0.015 or pixels.std() < 8:
        raise ValueError("Blank or insufficient text coverage")
    folder = root / "views" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / "hr.png"
    Image.fromarray(pixels).save(png, compress_level=3)
    identity = hashlib.sha256(pixels.tobytes()).hexdigest()
    row = dict(
        job,
        hr=str(png.relative_to(root)),
        hr_sha256=sha256(png),
        hr_rgb_sha256=identity,
        words=words,
        characters=sum(len(w["text"]) for w in words),
        width=WIDTH,
        height=HEIGHT,
        metadata=dict(metadata, text_box_coverage=coverage),
    )
    atomic_json(folder / "view.json", row)
    return dict(
        id=job["id"],
        kind=job["kind"],
        split=job["split"],
        source_group=job["source_group"],
        hr=row["hr"],
        hr_sha256=row["hr_sha256"],
        hr_rgb_sha256=identity,
        labels=str((folder / "view.json").relative_to(root)),
        labels_sha256=sha256(folder / "view.json"),
        characters=row["characters"],
    )


def pdf_view(task):
    import pymupdf

    root, job, src = task
    root = Path(root)
    folder = root / "views" / job["id"]
    if (folder / "view.json").exists():
        row = read(folder / "view.json")
        if sha256(root / row["hr"]) != row["hr_sha256"]:
            raise ValueError("Changed existing render")
        return dict(
            id=job["id"],
            kind=job["kind"],
            split=job["split"],
            source_group=job["source_group"],
            hr=row["hr"],
            hr_sha256=row["hr_sha256"],
            hr_rgb_sha256=row["hr_rgb_sha256"],
            labels=str((folder / "view.json").relative_to(root)),
            labels_sha256=sha256(folder / "view.json"),
            characters=row["characters"],
        )
    with pymupdf.open(root / src["path"]) as doc:
        page = doc[job["page"] - 1]
        scale = job["page_width"] / page.rect.width
        top = job["top_points"]
        clip = pymupdf.Rect(
            0, top, page.rect.width, min(page.rect.height, top + 1000 / scale)
        )
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale),
            clip=clip,
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), (225, 228, 233))
        x0 = (WIDTH - pix.width) // 2
        y0 = 48
        canvas.paste(image, (x0, y0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, WIDTH, 31), fill=(246, 247, 249))
        draw.text(
            (18, 9),
            "Document viewer  |  " + str(job["page"]) + " / " + str(len(doc)),
            fill=(50, 55, 65),
        )
        words = []
        # Pixmap origins include the floor-rounded device clip origin.
        for x1, y1, x2, y2, text, block, line, index in page.get_text(
            "words", sort=True
        ):
            rect = pymupdf.Rect(x1, y1, x2, y2) * page.rotation_matrix
            x1, y1, x2, y2 = rect.x0, rect.y0, rect.x1, rect.y1
            if (
                x1 < clip.x0 + 0.25
                or x2 > clip.x1 - 0.25
                or y1 < clip.y0 + 0.25
                or y2 > clip.y1 - 0.25
            ):
                continue
            x = x0 + x1 * scale - pix.x
            y = y0 + y1 * scale - pix.y
            w = (x2 - x1) * scale
            h = (y2 - y1) * scale
            if x < 0 or y < 0 or x + w > WIDTH or y + h > HEIGHT:
                continue
            if not text.strip() or "\ufffd" in text:
                continue
            words.append(
                dict(
                    text=text,
                    xywh=[round(v, 3) for v in [x, y, w, h]],
                    block=block,
                    line=line,
                    index=index,
                )
            )
        return save_view(
            root,
            job,
            np.array(canvas),
            words,
            dict(
                renderer="PyMuPDF",
                version=pymupdf.__version__,
                source_sha256=src["sha256"],
                page=job["page"],
                scale=scale,
                clip_points=list(clip),
                pixmap_origin=[pix.x, pix.y],
                truth_origin="PDF embedded text layer; only fully visible word boxes",
            ),
        )


def token_markup(text):
    return "".join(
        html.escape(s)
        if s.isspace()
        else '<span data-gt="1">' + html.escape(s) + "</span>"
        for s in re.findall(r"\s+|\S+", text)
    )


def html_document(root, job, src):
    """Return escaped offline HTML, literal text and rendering identity."""
    size = job["font_size"]
    dark = job["theme"] == "dark"
    bg, fg, panel, accent = (
        ("#181c24", "#e4e9f0", "#242b38", "#94c9ff")
        if dark
        else ("#f2f4f7", "#202b39", "#ffffff", "#245ea6")
    )
    if job["kind"] == "code":
        from pygments import lex
        from pygments.lexers import get_lexer_for_filename
        from pygments.token import Token

        lines = (root / src["path"]).read_text().splitlines()
        start = job["line_start"]
        selected = lines[start : start + max(36, math.ceil(1000 / (size * 1.42)))]
        text = "\n".join(selected)
        lexer = get_lexer_for_filename(src["title"], stripnl=False)
        rendered = []
        for token, part in lex(text, lexer):
            color = (
                accent
                if token in Token.Keyword
                else ("#83ba88" if dark else "#347045")
                if token in Token.Literal.String
                else fg
            )
            rendered.append(
                f'<span style="color:{color}">' + token_markup(part) + "</span>"
            )
        body = '<pre id="content">' + "".join(rendered) + "</pre>"
        title = src["title"]
        font = "Menlo, 'Courier New', monospace"
        extra = "pre{white-space:pre;tab-size:4;margin:0;line-height:1.42}"
    elif job["kind"] == "web":
        import pymupdf

        with pymupdf.open(root / src["path"]) as doc:
            text = doc[job["page"] - 1].get_text("text", sort=True)
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        if not paragraphs:
            paragraphs = [" ".join(text.split())]
        body = (
            '<article id="content"><h1>'
            + token_markup(src["title"])
            + "</h1>"
            + "".join(
                "<p>" + token_markup(p.replace("\n", " ")) + "</p>" for p in paragraphs
            )
            + "</article>"
        )
        title = "Reader / documentation"
        font = ["Arial", "Georgia", "Verdana", "Times New Roman"][job["variant"] % 4]
        extra = (
            "article{max-width:"
            + str([840, 1200, 1740][job["variant"] % 3])
            + "px;margin:0 auto;line-height:1.5}p{margin:0 0 1em}h1{font-size:1.5em}"
        )
    elif job["kind"] == "table":
        rng = random.Random(682000 + job["variant"])
        headers = [
            "Case",
            "Requests",
            "Elapsed (ms)",
            "Error rate",
            "Version",
            "Status",
        ]
        values = []
        for i in range(35):
            values.append(
                [
                    f"CASE-{rng.randrange(100000, 999999)}",
                    f"{rng.randrange(1, 999999):,}",
                    f"{rng.uniform(0.01, 3000):.3f}",
                    f"{rng.uniform(0, 9):.4f}%",
                    f"{rng.randrange(1, 10)}.{rng.randrange(0, 30)}.{rng.randrange(0, 99)}",
                    rng.choice(["PASS", "FAIL", "RETRY", "PENDING"]),
                ]
            )
        text = "\n".join("\t".join(row) for row in [headers, *values])
        body = (
            '<div id="content"><h1>'
            + token_markup("Service diagnostics / numerical results")
            + "</h1><table>"
            + "".join(
                "<tr>"
                + "".join("<td>" + token_markup(v) + "</td>" for v in row)
                + "</tr>"
                for row in [headers, *values]
            )
            + "</table></div>"
        )
        title = "Diagnostics"
        font = ["Arial", "Georgia", "Verdana"][job["variant"] % 3]
        extra = (
            "table{border-collapse:collapse;width:100%}td{padding:7px 18px;border:1px solid #8993a0;font-variant-numeric:tabular-nums}tr:nth-child(even){background:"
            + bg
            + "}h1{font-size:1.4em}"
        )
    else:
        rng = random.Random(682000 + job["variant"])
        lines = []
        for i in range(60):
            path = f"/srv/customer-{rng.randrange(100, 999)}/config/device_{rng.randrange(10000):04d}.json"
            lines.append(
                f"2026-10-21 10:{i % 60:02d}:{rng.randrange(60):02d} {rng.choice(['INFO ', 'WARN ', 'ERROR'])} session={rng.randrange(16**8):08x} {path} code=0x{rng.randrange(65536):04X}"
            )
        text = "\n".join(lines)
        body = '<pre id="content">' + token_markup(text) + "</pre>"
        title = "Terminal / service session"
        font = "Menlo, 'Courier New', monospace"
        extra = "pre{white-space:pre;margin:0;line-height:1.4}"
    page = f"""<!doctype html><html><meta charset="utf-8"><style>*{{box-sizing:border-box}}html,body{{margin:0;width:1920px;height:1080px;overflow:hidden;background:{bg};color:{fg};font-family:{font};font-size:{size}px}}header{{height:38px;background:{panel};padding:9px 18px;font:15px Arial;border-bottom:1px solid #7f899655}}main{{position:absolute;left:32px;top:60px;width:1856px;height:992px;overflow:hidden;background:{panel};padding:18px 24px}}{extra}</style><header>{html.escape(title)}</header><main>{body}</main></html>"""
    return (
        page,
        text,
        dict(
            renderer="Chrome",
            font_css=font,
            font_size=size,
            theme=job["theme"],
            source_sha256=None if src is None else src["sha256"],
        ),
    )


def html_views(root, jobs, sources):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME, headless=True, args=["--disable-lcd-text"]
        )
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=1,
            locale="en-US",
            timezone_id="UTC",
        )
        context.route("**/*", lambda route: route.abort())
        page = context.new_page()
        for n, job in enumerate(jobs, 1):
            folder = root / "views" / job["id"]
            if (folder / "view.json").exists():
                row = read(folder / "view.json")
                if sha256(root / row["hr"]) != row["hr_sha256"]:
                    raise ValueError("Changed existing HTML render")
                yield dict(
                    id=job["id"],
                    kind=job["kind"],
                    split=job["split"],
                    source_group=job["source_group"],
                    hr=row["hr"],
                    hr_sha256=row["hr_sha256"],
                    hr_rgb_sha256=row["hr_rgb_sha256"],
                    labels=str((folder / "view.json").relative_to(root)),
                    labels_sha256=sha256(folder / "view.json"),
                    characters=row["characters"],
                )
                continue
            source = sources[job["source"]] if job["source"] else None
            document, text, metadata = html_document(root, job, source)
            page.set_content(document, wait_until="load")
            page.evaluate("document.fonts.ready")
            boxes = page.evaluate(
                """() => {let main=document.querySelector('main').getBoundingClientRect();return [...document.querySelectorAll('[data-gt]')].map((n,i)=>{let r=n.getBoundingClientRect();return {text:n.textContent,xywh:[r.x,r.y,r.width,r.height],index:i}}).filter(w=>{let [x,y,a,b]=w.xywh;return a>0&&b>0&&x>=main.left&&y>=main.top&&x+a<=main.right&&y+b<=main.bottom})}"""
            )
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "source.html").write_text(document)
            (folder / "source.txt").write_text(text)
            from io import BytesIO

            pixels = np.array(
                Image.open(BytesIO(page.screenshot(type="png"))).convert("RGB")
            )
            metadata.update(
                browser_version=browser.version,
                html_sha256=sha256(folder / "source.html"),
                text_sha256=sha256(folder / "source.txt"),
            )
            yield save_view(root, job, pixels, boxes, metadata)
            if n % 50 == 0:
                print("HTML", n, "/", len(jobs), flush=True)
        browser.close()


def run(args):
    root = args.root.resolve()
    plan_path = root / "render-plan.json"
    if not plan_path.exists():
        plan = make_plan(
            root,
            dict(paper=args.papers, specification=args.specifications, code=args.code),
        )
        atomic_json(plan_path, plan)
        (root / "render-runner.py").write_bytes(Path(__file__).read_bytes())
    plan = read(plan_path)
    if plan["sources_sha256"] != sha256(root / "sources.json"):
        raise ValueError("Source catalogue changed")
    if plan["script_sha256"] != sha256(Path(__file__)):
        raise ValueError("Renderer differs from frozen plan")
    jobs = (
        list(
            round_robin(
                [
                    [j for j in plan["jobs"] if j["kind"] == kind]
                    for kind in plan["counts"]
                ]
            )
        )[: args.limit]
        if args.limit
        else plan["jobs"]
    )
    pdf = [j for j in jobs if j["kind"] in ("paper", "specification")]
    source_map = plan["source_map"]
    rows = []
    rejected = []

    # A failed layout is explicit and never silently replaced by another split.
    def accept(job, result):
        if result["hr_rgb_sha256"] in pixel_ids:
            rejected.append(
                dict(
                    id=job["id"],
                    reason="duplicate_render_pixels",
                    duplicate=pixel_ids[result["hr_rgb_sha256"]],
                )
            )
        else:
            rows.append(result)
            pixel_ids[result["hr_rgb_sha256"]] = job["id"]

    pixel_ids = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(pdf_view, (str(root), j, source_map[j["source"]])) for j in pdf
        ]
        for n, (job, future) in enumerate(zip(pdf, futures), 1):
            try:
                accept(job, future.result())
            except (ValueError, RuntimeError) as error:
                rejected.append(dict(id=job["id"], reason=str(error)))
            if n % 100 == 0:
                print(
                    "PDF VIEWS",
                    n,
                    "/",
                    len(pdf),
                    "accepted",
                    len(rows),
                    "rejected",
                    len(rejected),
                    flush=True,
                )
    html_jobs = [j for j in jobs if j["kind"] not in ("paper", "specification")]
    for job, result in zip(html_jobs, html_views(root, html_jobs, source_map)):
        accept(job, result)
    result = dict(
        schema="isharescreen_text_views_v1",
        plan_sha256=sha256(plan_path),
        sources_sha256=sha256(root / "sources.json"),
        views=rows,
        rejected=rejected,
        complete=len(jobs) == len(plan["jobs"]) and len(rows) == len(jobs),
        requested=len(plan["jobs"]),
        processed=len(jobs),
        accepted=len(rows),
        counts=dict(Counter(r["kind"] for r in rows)),
        split_counts=dict(Counter(r["split"] for r in rows)),
        finished=time.time(),
    )
    atomic_json(root / ("views-pilot.json" if args.limit else "views.json"), result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("views", "rejected")},
            indent=2,
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--papers", type=int, default=6000)
    p.add_argument("--specifications", type=int, default=1000)
    p.add_argument("--code", type=int, default=2000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    run(p.parse_args())


if __name__ == "__main__":
    main()
