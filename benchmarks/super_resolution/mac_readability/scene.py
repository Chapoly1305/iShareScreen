"""Display a short, source-labeled Mac text stimulus and capture its window.

All content is project-authored diagnostic text, not personal application data.
This is development calibration; it must not enter train or the final test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

import AppKit as A
import Foundation as F
import Quartz as Q


def scenes():
    code = [
        "def read_frame(packet: bytes, sequence: int) -> Frame:",
        "    if sequence <= last_sequence:",
        "        return cached_frame",
        "    payload = packet[12:]  # preserve punctuation and case",
        "    scale = 1.5; bitrate_kbps = 600",
        '    label = "O0 I1 l| :; , . [] {} ()"',
        "    return Frame(payload, timestamp_ns=clock_ns())",
        "",
        "assert total_bytes <= 75_000",
        'assert result["status"] == "ready"',
    ]
    table = [
        "Experiment       Width   Height   Bitrate   Error rate   p95 (ms)",
        "Native           1280       720       600       0.125       18.40",
        "Candidate A      1280       720       600       0.094       19.25",
        "Candidate B      1920      1080      1000       0.071       24.80",
        "Quarter   Revenue ($)   Cost ($)   Difference ($)   Valid",
        "2026-Q1    12,345.67   9,876.54         2,469.13    true",
        "2026-Q2    18,201.09   8,110.02        10,091.07    true",
        "2026-Q3     1,008.00   1,080.00           -72.00   false",
    ]
    paper = [
        "Low-bandwidth remote reading: fidelity before smooth motion.",
        "A sharp-looking letter can still be the wrong letter.",
        "Compare punctuation, digits, case and line boundaries separately.",
        "Keep the first response prompt while refining stable text.",
        "Small text: O0, I1, l|, rn/m, 0.01/0.10, 1280/1920.",
        "源画面与解码画面必须对应同一内容、同一时间和同一显示比例。",
        "文字、代码、表格：先保证正确可读，再提高静止后的清晰度。",
        "不要把锐化产生的错误笔画误认为真实细节。",
    ]
    terminal = [
        "$ python evaluate.py --height 720 --bitrate 600 --chroma 420",
        "frames: 120    dropped: 0    queue: 1    device: cuda:0",
        "[validation] cer=0.0942  exact_lines=52/64  psnr=31.5039",
        "Traceback (most recent call last):",
        '  File "reader.py", line 128, in decode_packet',
        '    value = table[row_id]["amount"]',
        'KeyError: "amount"',
        "$ git diff --stat",
        " viewer.py | 48 insertions(+), 14 deletions(-)",
    ]
    result = []
    for kind, lines in [
        ("code", code),
        ("table", table),
        ("paper", paper),
        ("terminal", terminal),
    ]:
        for version in range(2):
            content = list(lines)
            if version:
                content = [
                    s.replace("600", "601")
                    .replace("ready", "retry")
                    .replace("2,469.13", "2,469.18")
                    .replace("0.0942", "0.0842")
                    .replace("correct", "exact")
                    .replace("正确", "准确")
                    for s in content
                ]
            result.append(
                dict(
                    id=f"{kind}-{version}",
                    group=kind,
                    dark=kind in ("code", "terminal"),
                    lines=content,
                )
            )
    return result


def color(rgb):
    return A.NSColor.colorWithSRGBRed_green_blue_alpha_(*rgb, 1.0)


class TextView(A.NSView):
    def isFlipped(self):
        return True

    def drawRect_(self, dirty):
        if getattr(self, "drawn_state_id", None) != self.state_id:
            self.draw_start_ns = time.monotonic_ns()
        scene = self.scene
        color((0.06, 0.07, 0.09) if scene["dark"] else (1.0, 1.0, 1.0)).setFill()
        A.NSRectFill(self.bounds())
        foreground = color((0.94, 0.95, 0.96) if scene["dark"] else (0.04, 0.05, 0.06))
        self.labels = []
        for i, line in enumerate(scene["lines"]):
            size = 10 if scene["group"] != "table" else 9
            font = (
                A.NSFont.systemFontOfSize_(size)
                if scene["group"] == "paper"
                else A.NSFont.fontWithName_size_("Menlo", size)
            )
            attrs = {
                A.NSFontAttributeName: font,
                A.NSForegroundColorAttributeName: foreground,
            }
            x, y, w, h = 32, 52 + i * 24, 890, 22
            F.NSString.stringWithString_(line).drawInRect_withAttributes_(
                ((x, y), (w, h)), attrs
            )
            if line:
                width = float(
                    F.NSString.stringWithString_(line).sizeWithAttributes_(attrs).width
                )
                self.labels.append(
                    dict(
                        text=line,
                        rect_points=[x - 2, y - 2, min(w + 4, width + 4), h],
                        font_points=size,
                    )
                )
        # Font-size ladder is fixed before any model is evaluated.
        for j, size in enumerate((6.0, 7.0, 8.0, 10.0)):
            line = (
                f"{int(size * 2):02d}px: Read 0.01, 0.10; I1l| O0; [a != b] {{x: 1280}}"
            )
            attrs = {
                A.NSFontAttributeName: A.NSFont.fontWithName_size_("Menlo", size),
                A.NSForegroundColorAttributeName: foreground,
            }
            x, y, w, h = 32, 350 + j * 34, 890, 28
            F.NSString.stringWithString_(line).drawInRect_withAttributes_(
                ((x, y), (w, h)), attrs
            )
            width = float(
                F.NSString.stringWithString_(line).sizeWithAttributes_(attrs).width
            )
            self.labels.append(
                dict(
                    text=line,
                    rect_points=[x - 2, y - 2, width + 4, h],
                    font_points=size,
                )
            )
        for (x, y), rgb in zip(
            ((4, 4), (940, 4), (4, 520), (940, 520)),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
        ):
            color(rgb).setFill()
            A.NSRectFill(((x, y), (16, 16)))
        # Duplicate state markers at both ends detect mixed-state tile canvases.
        for y in (8, 520):
            for bit in range(8):
                v = 1.0 if (self.state_id >> bit) & 1 else 0.0
                color((v, v, v)).setFill()
                A.NSRectFill(((440 + bit * 10, y), (8, 8)))
        self.drawn_state_id = self.state_id
        self.draw_return_ns = time.monotonic_ns()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds-per-state", type=float, default=3)
    p.add_argument("--external-capture", action="store_true")
    p.add_argument(
        "--view-raster",
        action="store_true",
        help="Save the app's own rendered view, distinct from a screen capture",
    )
    args = p.parse_args()
    if not 2 <= args.seconds_per_state <= 10:
        raise SystemExit("Use 2..10 seconds per state")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    app = A.NSApplication.sharedApplication()
    app.setActivationPolicy_(A.NSApplicationActivationPolicyAccessory)
    app.finishLaunching()
    window = A.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((40, 120), (960, 540)),
        A.NSWindowStyleMaskBorderless,
        A.NSBackingStoreBuffered,
        False,
    )
    window.setTitle_("iShareScreen text evaluation")
    window.setHasShadow_(False)
    window.setCollectionBehavior_(
        A.NSWindowCollectionBehaviorCanJoinAllSpaces
        | A.NSWindowCollectionBehaviorFullScreenAuxiliary
    )
    window.setColorSpace_(A.NSColorSpace.sRGBColorSpace())
    view = TextView.alloc().initWithFrame_(((0, 0), (960, 540)))
    definitions = scenes()
    view.scene = definitions[0]
    view.state_id = 1
    view.labels = []
    window.setContentView_(view)
    window.orderFrontRegardless()
    report = dict(
        schema="mac_native_text_stimulus_v2",
        role="development_diagnostics_only",
        source="project-authored text, native AppKit rendering",
        window_id=int(window.windowNumber()),
        window_frame_points=str(window.frame()),
        backing_scale=float(window.backingScaleFactor()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        states=[],
        complete=False,
        external_capture=args.external_capture,
        pid=os.getpid(),
        view_raster=args.view_raster,
    )
    path = args.output / "manifest.json"
    start = time.monotonic()
    previous = -1
    captured = set()

    def stop(*unused):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        while time.monotonic() - start < len(definitions) * args.seconds_per_state:
            elapsed = time.monotonic() - start
            index = min(len(definitions) - 1, int(elapsed // args.seconds_per_state))
            if index != previous:
                view.scene = definitions[index]
                view.state_id = index + 1
                before = time.monotonic_ns()
                view.setNeedsDisplay_(True)
                window.displayIfNeeded()
                report["states"].append(
                    dict(
                        definitions[index],
                        state_id=index + 1,
                        draw_start_ns=before,
                        draw_return_ns=time.monotonic_ns(),
                        labels=list(view.labels),
                    )
                )
                report["window_visible"] = bool(window.isVisible())
                report["own_windows"] = [
                    dict(
                        number=int(w.get("kCGWindowNumber", 0)),
                        bounds={
                            str(k): float(v)
                            for k, v in w.get("kCGWindowBounds", {}).items()
                        },
                        onscreen=bool(w.get("kCGWindowIsOnscreen", False)),
                        owner_pid=int(w.get("kCGWindowOwnerPID", 0)),
                    )
                    for w in (
                        Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionAll, 0) or []
                    )
                    if w.get("kCGWindowOwnerPID") == os.getpid()
                ]
                save(path, report)
                previous = index
            event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                A.NSEventMaskAny,
                F.NSDate.dateWithTimeIntervalSinceNow_(0.02),
                F.NSDefaultRunLoopMode,
                True,
            )
            if event is not None:
                app.sendEvent_(event)
            app.updateWindows()
            if (
                elapsed - index * args.seconds_per_state >= 0.7
                and index not in captured
            ):
                if getattr(view, "drawn_state_id", None) != index + 1:
                    raise RuntimeError("Window has not drawn the expected source state")
                row = report["states"][-1]
                row.update(
                    labels=list(view.labels),
                    labels_state_id=view.drawn_state_id,
                    draw_start_ns=view.draw_start_ns,
                    draw_return_ns=view.draw_return_ns,
                )
                assert [label["text"] for label in row["labels"][:-4]] == [
                    line for line in row["lines"] if line
                ]
                if args.external_capture:
                    captured.add(index)
                    save(path, report)
                    continue
                output = args.output / (definitions[index]["id"] + ".png")
                before = time.monotonic_ns()
                if args.view_raster:
                    bitmap = view.bitmapImageRepForCachingDisplayInRect_(view.bounds())
                    view.cacheDisplayInRect_toBitmapImageRep_(view.bounds(), bitmap)
                    encoded = bitmap.representationUsingType_properties_(
                        A.NSBitmapImageFileTypePNG, {}
                    )
                    output.write_bytes(bytes(encoded))
                    row = report["states"][-1]
                    with output.open("rb") as stream:
                        header = stream.read(24)
                    row["reference_capture"] = dict(
                        start_ns=before,
                        end_ns=time.monotonic_ns(),
                        returncode=0,
                        kind="AppKit own-view raster; not a WindowServer screenshot",
                    )
                    row["reference"] = dict(
                        path=output.name,
                        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                        dimensions=list(struct.unpack(">II", header[16:24])),
                    )
                    captured.add(index)
                    save(path, report)
                    continue
                run = subprocess.run(
                    [
                        "/usr/sbin/screencapture",
                        "-x",
                        "-o",
                        "-l",
                        str(window.windowNumber()),
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                row = report["states"][-1]
                row["reference_capture"] = dict(
                    start_ns=before,
                    end_ns=time.monotonic_ns(),
                    returncode=run.returncode,
                )
                if run.returncode == 0:
                    with output.open("rb") as stream:
                        header = stream.read(24)
                    row["reference"] = dict(
                        path=output.name,
                        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                        dimensions=list(struct.unpack(">II", header[16:24])),
                    )
                else:
                    row["reference_capture"]["error"] = run.stderr.strip()
                captured.add(index)
                save(path, report)
        report["stimulus_complete"] = len(captured) == len(definitions)
        report["references_complete"] = all("reference" in s for s in report["states"])
        report["complete"] = (
            report["stimulus_complete"] and report["references_complete"]
        )
    except KeyboardInterrupt:
        report["interrupted"] = True
    finally:
        window.orderOut_(None)
        report["finished_ns"] = time.monotonic_ns()
        save(path, report)
    return (
        0
        if report["complete"]
        or (args.external_capture and report.get("stimulus_complete"))
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
