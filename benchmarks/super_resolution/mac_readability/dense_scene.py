"""Render frozen development documents with native Mac fonts and capture the window."""

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

from scene import color, save


class DenseView(A.NSView):
    def isFlipped(self):
        return True

    def drawRect_(self, dirty):
        self.draw_start_ns = time.monotonic_ns()
        state = self.state
        color((0.06, 0.07, 0.09) if state["dark"] else (1.0, 1.0, 1.0)).setFill()
        A.NSRectFill(self.bounds())
        font = A.NSFont.fontWithName_size_(state["font"], state["font_points"])
        if font is None:
            raise RuntimeError("Missing native font " + state["font"])
        attrs = {
            A.NSFontAttributeName: font,
            A.NSForegroundColorAttributeName: color(
                (0.94, 0.95, 0.96) if state["dark"] else (0.04, 0.05, 0.06)
            ),
        }
        self.labels = []
        for index, line in enumerate(state["lines"]):
            column, row = divmod(index, 24)
            x, y = 24 + column * 468, 44 + row * 18
            width = float(
                F.NSString.stringWithString_(line).sizeWithAttributes_(attrs).width
            )
            if column > 1 or width > 440:
                raise RuntimeError("Source text would be clipped")
            F.NSString.stringWithString_(line).drawInRect_withAttributes_(
                ((x, y), (440, 17)), attrs
            )
            self.labels.append(
                dict(
                    text=line,
                    rect_points=[x - 2, y - 1, width + 4, 17],
                    font_points=state["font_points"],
                    column=column,
                    row=row,
                    unit="line",
                    label_id=f"line-{index:02d}",
                )
            )
        for (x, y), rgb in zip(
            ((4, 4), (940, 4), (4, 520), (940, 520)),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
        ):
            color(rgb).setFill()
            A.NSRectFill(((x, y), (16, 16)))
        for y in [8, 520]:
            for bit in range(8):
                v = float(bool(state["state_id"] >> bit & 1))
                color((v, v, v)).setFill()
                A.NSRectFill(((440 + bit * 10, y), (8, 8)))
        self.drawn_state_id = state["state_id"]
        self.draw_return_ns = time.monotonic_ns()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--definitions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    definitions = json.loads(args.definitions.read_text())
    assert definitions["complete"] and len(definitions["states"]) == 32
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    if (Q.CGSessionCopyCurrentDictionary() or {}).get("CGSSessionScreenIsLocked"):
        raise RuntimeError("Mac locked")
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-di", "-w", str(os.getpid())])
    app = A.NSApplication.sharedApplication()
    app.setActivationPolicy_(A.NSApplicationActivationPolicyAccessory)
    app.finishLaunching()
    # Fail before entering an Objective-C drawing callback: exceptions raised
    # inside drawRect may not propagate to the Python capture loop.
    for state in definitions["states"]:
        font = A.NSFont.fontWithName_size_(state["font"], state["font_points"])
        if font is None:
            raise RuntimeError("Missing native font " + state["font"])
        attrs = {A.NSFontAttributeName: font}
        for index, line in enumerate(state["lines"]):
            width = float(
                F.NSString.stringWithString_(line).sizeWithAttributes_(attrs).width
            )
            if index >= 48 or width > 440:
                raise RuntimeError(
                    f"Layout overflow before capture: {state['id']} line{index}, width{width}"
                )
    window = A.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((40, 120), (960, 540)),
        A.NSWindowStyleMaskBorderless,
        A.NSBackingStoreBuffered,
        False,
    )
    window.setTitle_("iShareScreen development text evaluation")
    window.setHasShadow_(False)
    window.setColorSpace_(A.NSColorSpace.sRGBColorSpace())
    window.setCollectionBehavior_(
        A.NSWindowCollectionBehaviorCanJoinAllSpaces
        | A.NSWindowCollectionBehaviorFullScreenAuxiliary
    )
    view = DenseView.alloc().initWithFrame_(((0, 0), (960, 540)))
    view.state = definitions["states"][0]
    view.labels = []
    window.setContentView_(view)
    window.orderFrontRegardless()
    report = dict(
        schema="mac_dense_native_reference_v1",
        role=definitions["role"],
        definitions_sha256=hashlib.sha256(args.definitions.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        window_id=int(window.windowNumber()),
        backing_scale=float(window.backingScaleFactor()),
        source="Real WindowServer window screenshot, native AppKit fonts",
        states=[],
        complete=False,
        pid=os.getpid(),
    )
    path = args.output / "manifest.json"

    def stop(*unused):
        raise TimeoutError("Bounded capture stopped")

    signal.signal(signal.SIGTERM, stop)
    try:
        for state in definitions["states"]:
            if (Q.CGSessionCopyCurrentDictionary() or {}).get(
                "CGSSessionScreenIsLocked"
            ):
                raise RuntimeError("Mac locked during capture")
            view.state = state
            view.setNeedsDisplay_(True)
            window.displayIfNeeded()
            until = time.monotonic() + 0.8
            while time.monotonic() < until:
                event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                    A.NSEventMaskAny,
                    F.NSDate.dateWithTimeIntervalSinceNow_(0.02),
                    F.NSDefaultRunLoopMode,
                    True,
                )
                if event is not None:
                    app.sendEvent_(event)
                app.updateWindows()
            assert (
                view.drawn_state_id == state["state_id"]
                and [l["text"] for l in view.labels] == state["lines"]
            )
            output = args.output / (state["id"] + ".png")
            before = time.monotonic_ns()
            subprocess.run(
                [
                    "/usr/sbin/screencapture",
                    "-x",
                    "-o",
                    "-l",
                    str(window.windowNumber()),
                    str(output),
                ],
                check=True,
                capture_output=True,
                timeout=15,
            )
            header = output.read_bytes()[:24]
            dimensions = list(struct.unpack(">II", header[16:24]))
            assert dimensions == [1920, 1080]
            report["states"].append(
                dict(
                    state,
                    labels=list(view.labels),
                    labels_state_id=view.drawn_state_id,
                    draw_start_ns=view.draw_start_ns,
                    draw_return_ns=view.draw_return_ns,
                    reference_capture=dict(start_ns=before, end_ns=time.monotonic_ns()),
                    reference=dict(
                        path=output.name,
                        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                        dimensions=dimensions,
                    ),
                )
            )
            save(path, report)
        report["complete"] = len(report["states"]) == 32
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
    finally:
        window.orderOut_(None)
        awake.terminate()
        awake.wait(timeout=5)
        report["finished_ns"] = time.monotonic_ns()
        save(path, report)
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "states": len(report["states"]),
                "error": report.get("error"),
            }
        )
    )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
