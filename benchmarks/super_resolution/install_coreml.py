"""Install the validated QuickSRNet Small x2 Core ML conversion for live use."""
import argparse
from pathlib import Path
import shutil
import tempfile

from isharescreen.enhancement.coreml import bundle_path, verify_package
from isharescreen.enhancement.models import project_home


def install(package: Path, home: Path, license_path: Path) -> Path:
    """Copy a pinned package atomically; never overwrite a different install."""
    verify_package(package)
    if not license_path.is_file():
        raise ValueError("The source distribution's LICENSE.md is required")
    target = bundle_path(home)
    if target.exists() or target.is_symlink():
        verify_package(target / "model.mlpackage")
        if not (target / "LICENSE.md").is_file():
            raise ValueError("Existing Core ML installation is missing its license")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".quicksrnet-", dir=target.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        shutil.copytree(package, staged / "model.mlpackage")
        shutil.copyfile(license_path, staged / "LICENSE.md")
        verify_package(staged / "model.mlpackage")
        staged.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="Validated model.mlpackage from the pretrained probe")
    parser.add_argument("--license", type=Path, help="Defaults to LICENSE.md beside the package")
    parser.add_argument("--home", type=Path, default=project_home())
    args = parser.parse_args()
    result = install(args.package, args.home, args.license or args.package.parent / "LICENSE.md")
    print(f"VERIFIED QuickSRNet Core ML: {result}")


if __name__ == "__main__":
    main()
