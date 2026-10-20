"""Pinned model-source verification without model downloads."""
import subprocess

import pytest

from benchmarks.super_resolution.setup_models import verify_source


def test_original_source_verification_rejects_edits_without_overwriting(tmp_path):
    repo=tmp_path/"vendor"
    subprocess.run(["git","init","-q",str(repo)],check=True)
    source=repo/"architecture.py"
    source.write_text("original\n")
    subprocess.run(["git","-C",str(repo),"add","architecture.py"],check=True)
    subprocess.run(["git","-C",str(repo),"-c","user.name=Fixture","-c","user.email=fixture@example.invalid",
                    "commit","-qm","Original fixture"],check=True)
    revision=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    assert verify_source(repo,"fixture",revision)==revision
    (repo/"cache.pyc").write_bytes(b"untracked cache")
    assert verify_source(repo,"fixture",revision)==revision
    source.write_text("user modification\n")
    with pytest.raises(RuntimeError,match="tracked edits"):
        verify_source(repo,"fixture",revision)
    assert source.read_text()=="user modification\n"
    with pytest.raises(RuntimeError,match="differs from lock"):
        verify_source(repo,"fixture","0"*40)
