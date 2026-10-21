"""Extract one midpoint frame per recording for a human scene inventory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont


def extract(source, output, fraction=0.5):
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
        "-show_format", "-of", "json", str(source)], text=True))
    stream = probe["streams"][0]
    duration = float(stream.get("duration") or probe["format"]["duration"])
    timestamp = duration * fraction
    suffix = "" if fraction == .5 else f"-p{round(fraction * 100):02d}"
    target = output / "frames" / f"{source.stem}{suffix}.png"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-threads", "2", "-ss", str(timestamp), "-i", str(source), "-map", "0:v:0",
        "-frames:v", "1", "-threads", "1", str(target)], check=True)
    with Image.open(target) as image:
        histogram = image.convert("L").histogram()
        pixels = image.width * image.height
        light_fraction = sum(histogram[192:]) / pixels
        dark_fraction = sum(histogram[:64]) / pixels
    return dict(id=source.stem, source=str(source), duration_s=duration,
                sample_s=timestamp, fraction=fraction, width=stream["width"], height=stream["height"],
                codec=stream["codec_name"], profile=stream.get("profile"),
                image=str(target.relative_to(output)),
                light_pixel_fraction=light_fraction, dark_pixel_fraction=dark_fraction)


def gallery(output, rows, notes=None):
    notes = notes or {}
    page = ['<!doctype html><html lang="en"><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>Recording Scene Review</title>',
            '<style>body{font-family:system-ui;max-width:1400px;margin:32px auto;padding:0 24px;background:#f7f8fa;color:#111827}'
            'img{width:100%;height:auto}section{margin:40px 0}p{line-height:1.5}h2{font-size:20px}</style>',
            '<body><h1>Recording Scene Review</h1>',
            '<p>Sampled recording frames. Click any image to inspect the full-resolution frame.</p>']
    for number, row in enumerate(rows, 1):
        note = notes.get(row["id"], {})
        title = note.get("scene", "")
        description = note.get("assessment", "")
        image = html.escape(row["image"], quote=True)
        page += [f'<section id="{row["id"]}"><h2>{number:02d} · {html.escape(row["id"])}'
                 f'{" · " + html.escape(title) if title else ""}</h2>',
                 f'<p>{row["sample_s"]:.2f} / {row["duration_s"]:.2f} s · '
                 f'{row["width"]} × {row["height"]} · {html.escape(row["codec"])}</p>',
                 f'<p>{html.escape(description)}</p>' if description else '',
                 f'<a href="{image}"><img src="{image}" loading="lazy" alt="{html.escape(title or row["id"], quote=True)}"></a></section>']
    page.append('</body></html>')
    (output / "index.html").write_text("\n".join(page))


def contact_sheets(output, rows):
    width, height, header = 960, 540, 46
    font = ImageFont.load_default(size=26)
    for start in range(0, len(rows), 4):
        sheet = Image.new("RGB", (width * 2, (height + header) * 2), "#111827")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(rows[start:start + 4]):
            x, y = index % 2 * width, index // 2 * (height + header)
            draw.text((x + 12, y + 9), f'{start + index + 1:02d}  {row["id"]}  @ {row["sample_s"]:.1f}s',
                      fill="white", font=font)
            with Image.open(output / row["image"]) as source:
                source.thumbnail((width, height))
                sheet.paste(source, (x + (width - source.width) // 2, y + header + (height - source.height) // 2))
        sheet.save(output / f"sheet-{start // 4 + 1:02d}.jpg", quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", nargs="+", type=float, default=[.5])
    parser.add_argument("--captures", nargs="+")
    args = parser.parse_args()
    sources = sorted(args.dataset.glob("*.mov"))
    if args.captures:
        sources = [source for source in sources if source.stem in args.captures]
        if len(sources) != len(args.captures):
            raise ValueError("Requested recording missing")
    if not all(0 < value < 1 for value in args.fractions) or len(set(args.fractions)) != len(args.fractions):
        raise ValueError("Use unique fractions strictly between zero and one")
    if not sources or args.output.exists():
        raise ValueError("Supply a nonempty dataset and a fresh output directory")
    (args.output / "frames").mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        requests = [(source, fraction) for source in sources for fraction in args.fractions]
        rows = list(pool.map(lambda item: extract(item[0], args.output, item[1]), requests))
    (args.output / "manifest.json").write_text(json.dumps(dict(dataset=str(args.dataset),
        fractions=args.fractions, recordings=rows), indent=2) + "\n")
    contact_sheets(args.output, rows)
    gallery(args.output, rows)
    print(f"Extracted {len(rows)} midpoint frames", flush=True)


if __name__ == "__main__":
    main()
