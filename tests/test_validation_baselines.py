from fractions import Fraction
import json
from pathlib import Path

import av
import numpy as np

from benchmarks.super_resolution.validation_baselines import configurations, pixel_quality, sample_video


def test_default_matrix_uses_only_420():
    matrix=configurations()
    assert len(matrix)==len({c['id'] for c in matrix})==8
    assert {c['chroma'] for c in matrix}=={'420'}
    assert {(c['width'],c['height']) for c in matrix}=={(960,540),(1024,576),(1280,720),(1920,1080)}


def test_explicit_chroma_comparison_preserves_original_matrix():
    matrix=configurations(('420','444'))
    assert len(matrix)==16
    for h in (540,576,720,1080):
        for b in (600000,1000000):
            assert {c['chroma'] for c in matrix if c['height']==h and c['bitrate']==b}=={'420','444'}


def test_recorded_policy_matches_default():
    policy=json.loads((Path(__file__).parents[1]/'benchmarks/super_resolution/evaluation-policy.json').read_text())
    assert policy['chroma']=='420'
    assert policy['pixel_format']=='yuv420p'


def test_sample_video_preserves_requested_frame_phase(tmp_path):
    path=tmp_path/'phase.mkv'
    with av.open(str(path),'w') as sink:
        stream=sink.add_stream('libx264rgb',rate=60)
        stream.width=32;stream.height=24;stream.pix_fmt='rgb24'
        stream.options={'crf':'0','preset':'ultrafast','g':'4'}
        for i in range(12):
            rgb=np.full((24,32,3),(i*10,40,170),dtype=np.uint8)
            frame=av.VideoFrame.from_ndarray(rgb,format='rgb24')
            frame.pts=i;frame.time_base=Fraction(1,60)
            for packet in stream.encode(frame):sink.mux(packet)
        for packet in stream.encode():sink.mux(packet)
    samples=list(sample_video(path,[0,4,10],60,(32,24)))
    assert [i for i,_ in samples]==[0,4,10]
    for i,rgb in samples:
        assert np.all(rgb==np.array([i*10,40,170],dtype=np.uint8))


def test_pixel_quality_detects_image_changes():
    rng=np.random.default_rng(682)
    reference=rng.integers(0,256,(24,32,3),dtype=np.uint8)
    result=pixel_quality(reference,np.roll(reference,1,axis=0))
    assert 0<result['psnr']<30
    assert result['ssim']<.5
