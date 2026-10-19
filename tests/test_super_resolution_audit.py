"""Raw-data regression checks for experiment reporting, without model downloads."""
from copy import deepcopy
import subprocess

import pytest

from benchmarks.super_resolution.report import audit_aggregates
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


def test_benchmark_refuses_existing_output_and_preserves_marker(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    pytest.importorskip("skimage")
    from benchmarks.super_resolution.benchmark import create_output
    output=tmp_path/"results"
    create_output(output)
    with pytest.raises(FileExistsError,match="fresh output directory"):
        create_output(output)
    marker=output/"COMPLETED.json"
    marker.write_text('{"rows": 123}\n')
    with pytest.raises(FileExistsError,match="fresh output directory"):
        create_output(output)
    assert marker.read_text()=='{"rows": 123}\n'


def test_quality_aggregate_is_capture_weighted_and_timing_is_audited():
    rows=[]
    for i,(capture,psnr) in enumerate([("a",20),("a",24),("b",30)]):
        rows.append(dict(model="bicubic",sample=str(i),capture=capture,degradation="clean",
                         psnr=str(psnr),ssim="0.8",psnr_border2=str(psnr+1),ssim_border2="0.9",lpips="",
                         prepare_ms="1",upload_ms="2",inference_ms="3",download_ms="4",total_ms="10"))
    summary={"bicubic":{"clean":dict(n_frames=3,n_captures=2,psnr=26,ssim=.8,
                                        psnr_border2=27,ssim_border2=.9,
                                        delta_psnr_vs_bicubic=0,delta_ssim_vs_bicubic=0)}}
    audit_aggregates(rows,summary)
    stale=deepcopy(summary)
    stale["bicubic"]["clean"]["psnr"]=sum(float(r["psnr"]) for r in rows)/3
    with pytest.raises(AssertionError,match="Stale or incorrect"):
        audit_aggregates(rows,stale)
    bad_timing=deepcopy(rows)
    bad_timing[0]["total_ms"]="9"
    with pytest.raises(AssertionError,match="timings do not sum"):
        audit_aggregates(bad_timing,summary)
