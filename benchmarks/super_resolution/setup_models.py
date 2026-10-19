"""Fetch locked official source/checkpoints and verify provenance. No model execution."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request

from isharescreen.enhancement.models import project_home, sha256


def verify_source(repo, key, revision):
    actual=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if actual != revision:
        raise RuntimeError(f"{key} source differs from lock: {actual}. Preserve local edits; resolve manually.")
    dirty=subprocess.check_output(["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],text=True)
    if dirty.strip():
        raise RuntimeError(f"{key} source has tracked edits. Preserve local edits; resolve manually.\n{dirty}")
    return actual


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--home",type=Path,default=project_home())
    args=parser.parse_args()
    lock=json.loads((Path(__file__).parent/"models.lock.json").read_text())
    for key, spec in lock.items():
        repo=args.home/"vendor"/spec["directory"]
        if not repo.exists():
            repo.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(["git","clone","--no-checkout",spec["repo"],str(repo)],check=True)
            subprocess.run(["git","-C",str(repo),"checkout",spec["revision"]],check=True)
        revision=verify_source(repo,key,spec["revision"])
        if "url" not in spec:
            print(f"UNAVAILABLE {key}: {spec['status']}",flush=True)
            continue
        target=args.home/"checkpoints"/spec["checkpoint"]
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            partial=target.with_suffix(target.suffix+".part")
            with urllib.request.urlopen(spec["url"], timeout=60) as response, partial.open("wb") as f:
                import shutil
                shutil.copyfileobj(response,f)
            if sha256(partial) != spec["sha256"]:
                raise RuntimeError(f"{key} download hash mismatch: {partial}")
            partial.replace(target)
        if sha256(target) != spec["sha256"]:
            raise RuntimeError(f"{key} existing checkpoint differs from lock: {target}")
        print(f"VERIFIED {key}: {revision} / {spec['sha256']}",flush=True)


if __name__=="__main__":
    main()
