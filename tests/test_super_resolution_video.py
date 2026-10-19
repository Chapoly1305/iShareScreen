"""Real encoded-video alignment and RGB round-trip checks, no neural weights."""
from fractions import Fraction
import csv
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("skimage")
pytest.importorskip("PIL")
from benchmarks.super_resolution.video_benchmark import LosslessWriter, aligned_frames, verify_lossless
from benchmarks.super_resolution.video_report import audit, paired_quality
from isharescreen.enhancement.models import sha256


def make_video(path,size,count,fps=Fraction(60)):
    writer=LosslessWriter(path,*size,fps)
    frames=[]
    try:
        for i in range(count):
            rgb=np.zeros((size[1],size[0],3),np.uint8)
            rgb[:]=[17+i*13,83+i*7,149-i*9]
            rgb[::2,::2]=[250,1,127]
            frames.append(rgb)
            writer.write(rgb)
    finally:writer.close()
    return frames,writer.digest.hexdigest()


def test_video_lossless_rgb_and_native_60fps_alignment(tmp_path):
    hr,lr=tmp_path/"hr.mkv",tmp_path/"lr.mkv"
    refs,digest=make_video(hr,(32,32),7)
    lows,_=make_video(lr,(16,16),7)
    assert verify_lossless(hr,7,digest)==7
    decoded=list(aligned_frames(hr,lr,7,Fraction(60)))
    assert len(decoded)==7
    for (i,reference,low),a,b in zip(decoded,refs,lows):
        np.testing.assert_array_equal(reference,a)
        np.testing.assert_array_equal(low,b)
    with pytest.raises(ValueError,match="round-trip"):
        verify_lossless(hr,7,hashlib.sha256(b"wrong").hexdigest())


def test_video_rejects_frame_count_or_timestamp_mismatch(tmp_path):
    hr,lr=tmp_path/"hr.mkv",tmp_path/"lr.mkv"
    make_video(hr,(32,32),7)
    make_video(lr,(16,16),6)
    with pytest.raises(ValueError,match="Unequal decoded"):
        list(aligned_frames(hr,lr,7,Fraction(60)))
    other=tmp_path/"wrong-rate.mkv"
    make_video(other,(16,16),7,Fraction(30))
    with pytest.raises(ValueError,match="Timestamp misalignment"):
        list(aligned_frames(hr,other,7,Fraction(60)))


def test_video_report_audits_raw_coverage_and_aggregates(tmp_path):
    pairs=tmp_path/"pairs";pairs.mkdir()
    run=tmp_path/"run";run.mkdir()
    make_video(pairs/"hr.mkv",(32,32),7)
    make_video(pairs/"lr.mkv",(16,16),7)
    _,digest=make_video(run/"out.mkv",(32,32),7)
    clip=dict(id="clip",frames=7,fps="60/1",reference="hr.mkv",low="lr.mkv",
              reference_sha256=sha256(pairs/"hr.mkv"),low_sha256=sha256(pairs/"lr.mkv"))
    manifest=pairs/"manifest.json";manifest.write_text(json.dumps({"clips":[clip]}))
    result=dict(clip="clip",model="bicubic",frames=7,lossless_roundtrip_verified=True,
                inference_calls=7,reused_frames=0,output="out.mkv",sha256=sha256(run/"out.mkv"),
                decoded_rgb_sequence_sha256=digest,sampled_quality_frames=3,psnr=30,ssim=.8)
    (run/"clips.json").write_text(json.dumps([result]))
    (run/"environment.json").write_text(json.dumps(dict(manifest_sha256=sha256(manifest),metrics_every=3,reuse_identical=False)))
    (run/"COMPLETED.json").write_text(json.dumps(dict(videos=1,frames_written=7,quality_rows=3,status={"bicubic":{"state":"completed"}})))
    def write_rows(indices):
        with (run/"quality.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=["clip","model","frame","time_s","psnr","ssim"])
            writer.writeheader()
            for index in indices:writer.writerow(dict(clip="clip",model="bicubic",frame=index,time_s=index/60,psnr=30,ssim=.8))
    write_rows([0,3,6])
    audit(run,manifest)
    write_rows([0,3,3])
    with pytest.raises(AssertionError,match="quality samples"):
        audit(run,manifest)
    write_rows([0,3,6])
    result["psnr"]=31
    (run/"clips.json").write_text(json.dumps([result]))
    with pytest.raises(AssertionError,match="Stale quality mean"):
        audit(run,manifest)


def test_video_paired_quality_matches_clip_ids_not_result_order():
    manifest={"clips":[dict(id="short",content="code"),dict(id="long",content="web")]}
    results=[dict(clip="long",model="edsr",psnr=28,ssim=.91,frames=900),
             dict(clip="short",model="bicubic",psnr=30,ssim=.90,frames=60),
             dict(clip="short",model="edsr",psnr=29,ssim=.89,frames=60),
             dict(clip="long",model="bicubic",psnr=25,ssim=.85,frames=900)]
    paired=paired_quality(manifest,results)
    assert [(r["clip"],r["content"],r["psnr_delta_db"]) for r in paired]==[
        ("long","web",3),("short","code",-1)]
    assert np.mean([r["psnr_delta_db"] for r in paired])==1  # Equal clip weight.
    assert [r["ssim_delta"] for r in paired]==pytest.approx([.06,-.01])


def test_video_paired_quality_does_not_invent_missing_baseline():
    manifest={"clips":[dict(id="a"),dict(id="b")]}
    assert paired_quality(manifest,[dict(clip="a",model="edsr",psnr=30,ssim=.9)])==[]
    with pytest.raises(AssertionError,match="Incomplete bicubic"):
        paired_quality(manifest,[dict(clip="a",model="bicubic",psnr=25,ssim=.8)])
