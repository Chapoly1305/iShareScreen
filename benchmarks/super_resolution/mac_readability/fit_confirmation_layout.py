"""Native-font width preflight; explicit excerpt adjustments before scoring."""

import argparse
import hashlib
import json
from pathlib import Path
import AppKit as A
import Foundation as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--definitions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.definitions.read_text())
    adjustments = []
    for group in data["groups"]:
        font = A.NSFont.fontWithName_size_(group["font"], 9)
        attrs = {A.NSFontAttributeName: font}

        def width(text):
            return float(
                F.NSString.stringWithString_(text).sizeWithAttributes_(attrs).width
            )

        for index, original in enumerate(group["lines"]):
            text = original
            while width(text) > 440:
                text = text[:-1]
            assert text
            if text != original:
                adjustments.append(
                    dict(
                        group=group["id"],
                        line=index,
                        original=original,
                        retained=text,
                        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                        original_width_points=width(original),
                        retained_width_points=width(text),
                        font=group["font"],
                        font_points=9,
                    )
                )
    result = dict(
        source_definitions_sha256=hashlib.sha256(
            args.definitions.read_bytes()
        ).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        adjustments=adjustments,
        rule="Keep maximal prefix fitting440nativepoints at largest9pointfont; identical text forbothfontsizevariants; noOCR/modelresults",
        test_used=False,
    )
    with args.output.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("Native width adjustments:", len(adjustments))


if __name__ == "__main__":
    main()
