"""Snapshot a worker's effective Python environment without requiring pip."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys


def version(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, text=True
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    distributions = []
    for package in importlib.metadata.distributions():
        name = package.metadata.get("Name")
        if name:
            distributions.append(
                dict(
                    name=name,
                    version=package.version,
                    location=str(package.locate_file("")),
                )
            )
    distributions.sort(
        key=lambda item: (item["name"].lower(), item["version"], item["location"])
    )
    data = dict(
        schema="isharescreen_low_bandwidth_environment_v1",
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python=sys.version,
        executable=sys.executable,
        prefix=sys.prefix,
        sys_path=sys.path,
        distributions=distributions,
        ffmpeg=version(["ffmpeg", "-version"]),
        tesseract=version(["tesseract", "--version"]),
        nvidia_smi=version(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ]
        ),
    )
    with open(args.output, "w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    print("ENVIRONMENT", data["hostname"], len(distributions), "packages")


if __name__ == "__main__":
    main()
