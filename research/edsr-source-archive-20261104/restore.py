"""Verify frozen source hashes, optionally restore them into a separate checkout."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-only", action="store_true")
    group.add_argument("--destination", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "MANIFEST.json").read_text())
    prepared = []
    if args.destination and args.destination.resolve() == root.parent.parent:
        raise ValueError("Use a separate checkout for historical sources")
    for original, entry in manifest["files"].items():
        relative = Path(original)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe archived path")
        data = (root / entry["source"]).read_bytes()
        if (
            len(data) != entry["bytes"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            raise ValueError("Archive mismatch: " + original)
        if args.destination:
            target = args.destination / relative
            if target.exists() and target.read_bytes() != data:
                raise FileExistsError(target)
            prepared.append((target, data))
    # Check every collision before writing any source.
    for target, data in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(manifest["count"], "sources verified" + (" and restored" if prepared else ""))


if __name__ == "__main__":
    main()
