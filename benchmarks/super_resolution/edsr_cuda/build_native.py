"""Build the optional Linux NVDEC shim against installed NVIDIA driver libraries."""

import argparse
import json
from pathlib import Path
import subprocess

from isharescreen.enhancement import cuda_decoder
from .common import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cuda-include", type=Path, default=Path("/usr/local/cuda/include")
    )
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    native = Path(cuda_decoder.__file__).parent / "native"
    command = [
        args.compiler,
        "-std=c++17",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-shared",
        "-fPIC",
        "-I" + str(args.cuda_include),
        "-I" + str(native / "include"),
        str(native / "nvdec.cpp"),
        "-l:libcuda.so.1",
        "-l:libnvcuvid.so.1",
        "-o",
        str(args.output),
    ]
    subprocess.run(command, check=True)
    files = [
        native / "nvdec.cpp",
        native / "include/nvcuvid.h",
        native / "include/cuviddec.h",
        args.cuda_include / "cuda.h",
        args.output,
    ]
    args.output.with_suffix(".build.json").write_text(
        json.dumps(
            dict(command=command, files={str(p): digest(p) for p in files}), indent=2
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
