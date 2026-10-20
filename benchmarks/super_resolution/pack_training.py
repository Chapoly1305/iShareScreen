"""Pack audited PNG pairs into shared, read-only, RGB uint8 memory maps."""
import argparse
import hashlib
import json
from pathlib import Path

from isharescreen.enhancement.models import sha256


def pack(source, output):
    import numpy as np
    from PIL import Image
    data = json.loads(source.read_text())
    rows = data["pairs"]
    if not rows:
        raise ValueError("No retained pairs")
    output.mkdir(parents=True, exist_ok=False)
    packed = {}
    for role in ("lr", "hr"):
        array = None
        for index, row in enumerate(rows):
            path = (source.parent / row[role]).resolve()
            if not path.is_relative_to(source.parent.resolve()) or sha256(path) != row[f"{role}_sha256"]:
                raise ValueError("Changed or escaping PNG input")
            with Image.open(path) as image:
                rgb = np.array(image.convert("RGB"))
            if hashlib.sha256(rgb.tobytes()).hexdigest() != row[f"{role}_rgb_sha256"]:
                raise ValueError("Changed decoded pixels")
            if array is None:
                array = np.lib.format.open_memmap(output / f"{role}.npy", mode="w+", dtype=np.uint8,
                                                  shape=(len(rows), *rgb.shape))
            if rgb.shape != array.shape[1:]:
                raise ValueError("Packed pairs must share geometry")
            array[index] = rgb
        array.flush()
        packed[role] = dict(path=f"{role}.npy", sha256=sha256(output / f"{role}.npy"), shape=list(array.shape))
        del array
        print(f"PACKED {role}: {len(rows)} RGB frames", flush=True)
    if packed["hr"]["shape"][1:3] != [2*x for x in packed["lr"]["shape"][1:3]]:
        raise ValueError("Packed geometry is not x2")
    result = {**data, "packed": packed, "png_manifest_sha256": sha256(source),
              "packer_sha256": sha256(Path(__file__))}
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    pack(args.pairs, args.output)


if __name__ == "__main__":
    main()
