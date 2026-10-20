"""Lossless comparison panels and HTML galleries, independent of model inference."""
from __future__ import annotations

from copy import deepcopy
import html
import json
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw, ImageFont

from isharescreen.enhancement.models import sha256

IMAGE_SEPARATOR = 6
MODEL_HEADER_HEIGHT = 40
SEPARATOR_COLOR = "#94A3B8"
DEFAULT_LABELS = ("Original", "Bicubic", "EDSR", "IMDN", "SAFMN")
REFERENCE_SIZE = (1920, 1080)

_PAGE_HEAD = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Super-Resolution Comparison</title>
<style>
body {
    font-family: system-ui;
    max-width: 1600px;
    margin: 32px auto;
    padding: 0 24px;
    background: #f7f8fa;
    color: #111827;
}
video, img { width: 100%; height: auto; background: white; }
img { display: block; box-sizing: border-box; border: 1px solid #94a3b8; }
section { margin: 32px 0; }
h1, h2 { font-weight: 600; }
</style>
<body>
<h1>Super-Resolution Comparison</h1>"""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _asset_path(root, relative):
    """Do not let a supplied manifest read or replace files outside its gallery."""
    relative = Path(relative)
    path = root / relative
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and path.resolve().is_relative_to(root.resolve()),
        "Gallery asset escapes its output directory",
    )
    return path


def panel_grid(panels, columns, separator=IMAGE_SEPARATOR):
    _require(
        bool(panels)
        and type(columns) is int
        and columns > 0
        and len(panels) % columns == 0
        and type(separator) is int
        and separator >= 0
        and all(panel.size == panels[0].size for panel in panels),
        "Invalid panel grid",
    )
    width, height = panels[0].size
    rows = len(panels) // columns
    canvas = Image.new(
        "RGB",
        (columns * width + (columns - 1) * separator,
         rows * height + (rows - 1) * separator),
        SEPARATOR_COLOR,
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, (
            (index % columns) * (width + separator),
            (index // columns) * (height + separator),
        ))
    return canvas


def crop_strip(images, xywh, labels=None, columns=None):
    """Use identical native-reference coordinates and nearest-neighbor x2 display."""
    x, y, width, height = xywh
    labels = DEFAULT_LABELS if labels is None else labels
    columns = len(labels) if columns is None else columns
    _require(
        len(images) == len(labels)
        and type(columns) is int
        and columns > 0
        and len(images) % columns == 0
        and all(type(value) is int for value in xywh)
        and min(x, y) >= 0
        and min(width, height) > 0
        and all(image.size == REFERENCE_SIZE for image in images)
        and x + width <= REFERENCE_SIZE[0]
        and y + height <= REFERENCE_SIZE[1],
        "Crop must be the same valid reference-space rectangle for all models",
    )
    panel_width, panel_height = 2 * width, 2 * height
    font = ImageFont.load_default(size=24)
    panels = []
    for image, label in zip(images, labels):
        panel = Image.new("RGB", (panel_width, panel_height + MODEL_HEADER_HEIGHT), "#111827")
        ImageDraw.Draw(panel).text((14, 8), label, font=font, fill="white")
        crop = image.crop((x, y, x + width, y + height)).resize(
            (panel_width, panel_height), Image.Resampling.NEAREST
        )
        panel.paste(crop, (0, MODEL_HEADER_HEIGHT))
        panels.append(panel)
    return panel_grid(panels, columns)


def write_gallery(output, manifest):
    """Keep descriptive titles in HTML, not repeated inside the image pixels."""
    parts = [
        _PAGE_HEAD,
        f'<p>Fine-tuned models: epoch {html.escape(str(manifest["epoch"]))}.</p>',
        f'<video controls preload="metadata" src="{html.escape(manifest["video"], quote=True)}"></video>',
    ]
    for picture in manifest["pictures"]:
        filename = html.escape(picture["file"], quote=True)
        title = html.escape(picture["title"])
        alt = html.escape(picture["title"], quote=True)
        version = picture["sha256"][:12]
        parts.append(
            f'<section><h2>{title}</h2><a href="{filename}">'
            f'<img src="{filename}?v={version}" alt="{alt}" loading="lazy">'
            '</a></section>'
        )
    parts.append("</body></html>\n")
    (Path(output) / "index.html").write_text("\n".join(parts))


def _existing_panels(image, picture, columns, count, layout):
    separator = layout.get("separator_px", 0)
    title_height = MODEL_HEADER_HEIGHT if layout.get("embedded_titles", True) else 0
    _require(
        type(columns) is int and columns > 0 and count % columns == 0
        and type(separator) is int and separator >= 0,
        "Invalid existing image layout",
    )
    width = 2 * picture["xywh"][2]
    height = 2 * picture["xywh"][3] + MODEL_HEADER_HEIGHT
    rows = count // columns
    expected = (
        columns * width + (columns - 1) * separator,
        title_height + rows * height + (rows - 1) * separator,
    )
    _require(image.size == expected, "Unknown image layout")
    panels = []
    for index in range(count):
        x = (index % columns) * (width + separator)
        y = title_height + (index // columns) * (height + separator)
        panels.append(image.crop((x, y, x + width, y + height)))
    return panels


def refresh_gallery(output):
    """Restyle verified PNGs without changing their panel pixels or the video.

    Preflight all inputs and finish rendering/serialization in a temporary
    directory before replacing any existing gallery files. Failed image saves
    therefore leave the existing gallery usable. No persistent backup is made.
    """
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    video = _asset_path(output, manifest["video"])
    _require(sha256(video) == manifest["video_sha256"], "Comparison video changed")
    layout = manifest.get("image_layout", {})
    is_training_comparison = "pretrained_outputs" in manifest
    columns = layout.get("columns", 4 if is_training_comparison else 5)
    count = 8 if is_training_comparison else 5
    updates, seen = [], set()
    for picture in manifest["pictures"]:
        path = _asset_path(output, picture["file"])
        _require(path.suffix.lower() == ".png" and path.resolve() not in seen,
                 "Duplicate or invalid comparison image")
        seen.add(path.resolve())
        _require(sha256(path) == picture["sha256"], "Comparison image changed")
        with Image.open(path) as image:
            panels = _existing_panels(image, picture, columns, count, layout)
        updates.append(panel_grid(panels, columns))

    updated = deepcopy(manifest)
    updated["image_layout"] = dict(
        columns=columns, separator_px=IMAGE_SEPARATOR, embedded_titles=False
    )
    updated["gallery_generator_sha256"] = sha256(Path(__file__))
    with tempfile.TemporaryDirectory(prefix=".gallery-", dir=output) as temporary:
        staged = Path(temporary)
        for picture, image in zip(updated["pictures"], updates):
            target = _asset_path(staged, picture["file"])
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            picture["sha256"] = sha256(target)
        (staged / "manifest.json").write_text(json.dumps(updated, indent=2, allow_nan=False) + "\n")
        write_gallery(staged, updated)
        for relative in [p["file"] for p in updated["pictures"]] + ["manifest.json", "index.html"]:
            (staged / relative).replace(output / relative)
    print(f"UPDATED {len(updates)} images and gallery; video unchanged", flush=True)
