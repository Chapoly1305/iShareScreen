"""Repeat checks require identical inputs, code, pixels and sampled qualities."""
from copy import deepcopy

import pytest

from benchmarks.super_resolution.verify_video_repeat import compare


def repeat_pair():
    environment=dict(python="3.10",platform="linux",torch="2.6",av="17",cuda="12.4",
                     gpu="H100",runner_sha256="runner",adapter_sha256="adapter",precision="fp32",
                     reuse_identical=True,metrics_every=60,pid=100,started=1)
    result=dict(frames=120,decoded_rgb_sequence_sha256="pixels",inference_calls=100,
                reused_frames=20,sampled_quality_frames=3,psnr=30,ssim=.9,
                sha256="container-one",processing_p50_ms=50)
    first=dict(environment=environment,clip=dict(id="clip",frames=120,fps="60/1",
               source_sha256="source",low_sha256="low"),
               models={"edsr":dict(result=result,qualities=[dict(frame="0",psnr="30")])})
    second=deepcopy(first)
    second["environment"].update(pid=200,started=2)
    second["models"]["edsr"]["result"].update(sha256="container-two",processing_p50_ms=70)
    return first,second


def test_video_repeat_compares_pixels_not_container_metadata_or_timing():
    compare(*repeat_pair())


@pytest.mark.parametrize("change,match",[
    (lambda r:r["environment"].update(torch="different"),"Environment changed"),
    (lambda r:r["clip"].update(low_sha256="different"),"Input changed"),
    (lambda r:r["models"]["edsr"]["result"].update(decoded_rgb_sequence_sha256="different"),"Repeat mismatch"),
    (lambda r:r["models"]["edsr"]["qualities"][0].update(psnr="31"),"Sampled quality mismatch"),
    (lambda r:r["environment"].update(pid=100),"Not separate producer processes"),
])
def test_video_repeat_rejects_changed_evidence(change,match):
    first,second=repeat_pair()
    change(second)
    with pytest.raises(AssertionError,match=match):compare(first,second)
