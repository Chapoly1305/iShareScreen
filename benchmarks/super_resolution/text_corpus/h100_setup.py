"""Build the independent Linux development environment in the migrated repo."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from urllib.request import urlopen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    meta = root / ".migration"
    meta.mkdir(exist_ok=True)
    temporary = root / ".tmp"
    temporary.mkdir(exist_ok=True)
    env = dict(
        os.environ,
        TMPDIR=str(temporary),
        PIP_CACHE_DIR=str(root / ".cache/pip"),
        PLAYWRIGHT_BROWSERS_PATH=str(root / ".cache/ms-playwright"),
    )

    def status(phase, **more):
        (meta / "environment-status.json").write_text(
            json.dumps(dict(phase=phase, updated=time.time(), **more), indent=2)
        )

    def run(command):
        subprocess.run(list(map(str, command)), cwd=root, env=env, check=True)

    old = Path("/home/jchen73/cs682-isharescreen/.venv")
    python = old / "bin/python"
    venv = root / ".venv"
    status("creating_independent_python")
    if venv.exists():
        raise ValueError("Environment already exists; inspect before retry")
    run([python, "-m", "venv", "--without-pip", venv])
    newpython = venv / "bin/python"
    oldsite = old / "lib/python3.10/site-packages"
    newsite = venv / "lib/python3.10/site-packages"
    # Copy installed Linux wheels locally, preserving CUDA versions. Remove the
    # old editable project binding before installing this checkout independently.
    shutil.copytree(oldsite, newsite, dirs_exist_ok=True)
    for path in list(newsite.glob("*isharescreen*")):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for path in newsite.glob("*.pth"):
        if "/home/jchen73/cs682-isharescreen" in path.read_text(errors="ignore"):
            path.unlink()
    status("installing_project_and_tools")
    run([newpython, "-m", "pip", "install", "--no-deps", "-e", str(root)])
    run(
        [
            newpython,
            "-m",
            "pip",
            "install",
            "keyring>=25,<26",
            "pymupdf",
            "playwright",
            "gdown==5.2.0",
            "lmdb==1.7.5",
            "paramiko",
            "zstandard",
        ]
    )
    run([newpython, "-m", "pip", "check"])
    (meta / "python-freeze.txt").write_bytes(
        subprocess.check_output([str(newpython), "-m", "pip", "freeze"], env=env)
    )
    status("installing_linux_media_ocr")
    archive = meta / "micromamba.tar.bz2"
    with (
        urlopen(
            "https://micro.mamba.pm/api/micromamba/linux-64/latest", timeout=90
        ) as source,
        archive.open("wb") as target,
    ):
        shutil.copyfileobj(source, target)
    binary = root / "tools-runtime/bin/micromamba"
    binary.parent.mkdir(parents=True)
    with tarfile.open(archive) as pack:
        with pack.extractfile("bin/micromamba") as source, binary.open("wb") as target:
            shutil.copyfileobj(source, target)
    binary.chmod(0o755)
    env["MAMBA_ROOT_PREFIX"] = str(root / ".cache/mamba")
    run(
        [
            binary,
            "create",
            "--yes",
            "--prefix",
            root / "tools-runtime/env",
            "--channel",
            "conda-forge",
            "ffmpeg",
            "tesseract",
            "fonts-conda-ecosystem",
        ]
    )
    status("installing_headless_chromium")
    run([newpython, "-m", "playwright", "install", "chromium"])
    activation = (
        """# Source this file from an interactive shell; all heavy state stays on H100.
export ISS_WORKSPACE=%s
export ISS_SR_HOME="$ISS_WORKSPACE/.cache/super_resolution"
export ISS_TESSDATA="$ISS_WORKSPACE/tools-runtime/tessdata"
export TESSDATA_PREFIX="$ISS_TESSDATA"
export TORCH_HOME="$ISS_WORKSPACE/.cache/torch"
export PLAYWRIGHT_BROWSERS_PATH="$ISS_WORKSPACE/.cache/ms-playwright"
export HF_HOME="$ISS_WORKSPACE/.cache/huggingface"
export TMPDIR="$ISS_WORKSPACE/.tmp"
export PATH="$ISS_WORKSPACE/.venv/bin:$ISS_WORKSPACE/tools-runtime/env/bin:$PATH"
export PYTHONPATH="$ISS_WORKSPACE:$ISS_WORKSPACE/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ISS_WORKSPACE"
"""
        % root
    )
    (root / "dev-environment.sh").write_text(activation)
    # English traineddata is copied byte-for-byte from the current benchmark.
    # Linux Tesseract version is recorded separately; historical engine bindings
    # are retained and must not be silently rewritten.
    status(
        "ready_for_validation",
        python=str(newpython),
        micromamba_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    main()
