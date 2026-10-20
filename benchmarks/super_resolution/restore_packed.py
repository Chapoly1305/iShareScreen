"""Restore losslessly transported packed arrays and verify their frozen hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha256(path):
    # Keep transport usable with a system Python, without viewer/ML dependencies.
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore(manifest, archives):
    data = json.loads(manifest.read_text())
    verified = {}
    for role in ("lr", "hr"):
        spec = data["packed"][role]
        if spec["path"] != f"{role}.npy":
            raise ValueError("Only portable role-named packed arrays are supported")
        target = manifest.parent / spec["path"]
        if not target.exists():
            archive = archives / f"{role}.npy.zst"
            temporary = target.with_suffix(".npy.restoring")
            with temporary.open("wb") as output:
                if shutil.which("zstd"):
                    subprocess.run(["zstd", "-d", "--long=27", "--stdout", str(archive)], stdout=output, check=True)
                else:
                    import zstandard  # Optional transport dependency, not a viewer dependency.
                    with archive.open("rb") as source:
                        zstandard.ZstdDecompressor().copy_stream(source, output)
            if sha256(temporary) != spec["sha256"]:
                raise ValueError("Restored pixels differ from frozen packed array")
            temporary.replace(target)
        elif sha256(target) != spec["sha256"]:
            raise ValueError("Existing packed array differs; refusing overwrite")
        verified[role] = spec["sha256"]
        print(f"VERIFIED {role}: {target}", flush=True)
    receipt = dict(schema="sr_packed_transport_receipt_v1", manifest_sha256=sha256(manifest),
                   verified=verified, transport="lossless zstd; decoded array hashes checked")
    (manifest.parent / "TRANSPORT-VERIFIED.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--archives", type=Path, required=True)
    args = p.parse_args()
    restore(args.manifest, args.archives)


if __name__ == "__main__":
    main()
