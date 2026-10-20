"""Download only hash-pinned official candidate assets; never train or execute."""
import argparse
from pathlib import Path
import shutil
import tempfile
import urllib.request

from isharescreen.enhancement import candidates
from isharescreen.enhancement.models import project_home


def install(key, home, allow_unlicensed=False):
    spec = candidates.specification(key)
    if spec["license_id"] == "Unspecified" and not allow_unlicensed:
        raise ValueError(f"{key} has no identified license; explicitly use --allow-unlicensed for private evaluation")
    root = candidates.asset_home(key, home)
    if root.exists() or root.is_symlink():
        candidates.verify_assets(key, home)
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{key}-", dir=root.parent) as temp:
        staged_home = Path(temp)
        staged = candidates.asset_home(key, staged_home)
        for name, asset in spec["files"].items():
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(asset["url"], timeout=60) as response, target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        candidates.verify_assets(key, staged_home)
        staged.rename(root)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=candidates.IDS, default=["plainusr", "quicksrnet", "rt4ksr", "efrlfn", "spanv2"])
    parser.add_argument("--home", type=Path, default=project_home())
    parser.add_argument("--allow-unlicensed", action="store_true")
    args = parser.parse_args()
    for key in args.models:
        print(f"VERIFIED {key}: {install(key, args.home, args.allow_unlicensed)}", flush=True)


if __name__ == "__main__":
    main()
