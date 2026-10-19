"""Perceptual evaluation gates, without downloading or loading neural weights."""
from copy import deepcopy
from fractions import Fraction
import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("skimage")
pytest.importorskip("PIL")
from benchmarks.super_resolution.video_perceptual import audit_scores, normalized_tensor, sample_indices, sampled_rgb, summarize
from benchmarks.super_resolution.video_benchmark import LosslessWriter
import benchmarks.super_resolution.video_perceptual as perceptual


def video(path,count=7,fps=Fraction(60)):
    writer=LosslessWriter(path,32,32,fps)
    try:
        for i in range(count):writer.write(np.full((32,32,3),i*20,np.uint8))
    finally:writer.close()


def test_perceptual_sample_indices_include_last_once():
    assert sample_indices(1,60)==[0]
    assert sample_indices(121,60)==[0,60,120]
    assert sample_indices(122,60)==[0,60,120,121]
    with pytest.raises(ValueError,match="positive"):sample_indices(0,60)
    with pytest.raises(ValueError,match="positive"):sample_indices(7,0)


def test_perceptual_sampler_preserves_pixels_indices_and_native_timing(tmp_path):
    path=tmp_path/"rgb.mkv";video(path)
    frames=list(sampled_rgb(path,7,Fraction(60),3,size=(32,32)))
    assert [(i,t) for i,t,_ in frames]==[(0,0),(3,.05),(6,.1)]
    for i,_,rgb in frames:np.testing.assert_array_equal(rgb,np.full((32,32,3),i*20,np.uint8))
    with pytest.raises(ValueError,match="Expected 8 frames"):
        list(sampled_rgb(path,8,Fraction(60),3,size=(32,32)))
    with pytest.raises(ValueError,match="More decoded"):
        list(sampled_rgb(path,6,Fraction(60),3,size=(32,32)))
    with pytest.raises(ValueError,match="Timestamp misalignment"):
        list(sampled_rgb(path,7,Fraction(30),3,size=(32,32)))
    with pytest.raises(ValueError,match="geometry"):
        list(sampled_rgb(path,7,Fraction(60),3,size=(64,64)))


def test_perceptual_rgb_normalization_is_exact_and_not_resized():
    rgb=np.array([[[0,127,255],[255,0,127]]],np.uint8)
    tensor=normalized_tensor(rgb,"cpu")
    assert tuple(tensor.shape)==(1,3,1,2)
    np.testing.assert_allclose(tensor.numpy()[0],rgb.transpose(2,0,1)/127.5-1,atol=1e-7)
    with pytest.raises(ValueError,match="RGB uint8"):
        normalized_tensor(rgb.astype(np.float32),"cpu")


def rows_fixture():
    rows=[]
    # A has three sampled frames, B only one. They must still have equal clip weight.
    for clip,indices in [("a",[0,60,120]),("b",[0])]:
        for model in ["bicubic","edsr"]:
            for frame in indices:
                score=.4 if clip=="a" else .8
                if model=="edsr":score-=.1 if clip=="a" else .3
                rows.append(dict(clip=clip,model=model,frame=frame,time_s=frame/60,
                    psnr=25+int(model=="edsr")*2,ssim=.8+int(model=="edsr")*.02,lpips=score,
                    reference_rgb_sha256="a"*64,output_rgb_sha256="b"*64))
    return list(reversed(rows))


def test_perceptual_aggregation_is_clip_weighted_with_paired_deltas():
    summary=summarize(rows_fixture())
    assert summary["bicubic"]["lpips"]==pytest.approx(.6)
    assert summary["edsr"]["lpips"]==pytest.approx(.4)
    assert summary["edsr"]["delta_lpips_vs_bicubic"]==pytest.approx(-.2)
    assert summary["edsr"]["delta_psnr_vs_bicubic"]==2


def test_perceptual_aggregation_rejects_missing_duplicate_or_nonfinite_data():
    rows=rows_fixture()
    with pytest.raises(ValueError,match="Duplicate"):summarize(rows+[dict(rows[0])])
    with pytest.raises(ValueError,match="baseline"):summarize([r for r in rows if r["model"]=="edsr"])
    with pytest.raises(ValueError,match="Unequal paired"):summarize(rows[1:])
    rows[0]["lpips"]=float("nan")
    with pytest.raises(ValueError,match="Nonfinite"):summarize(rows)


def test_saved_perceptual_raw_rows_and_summary_are_independently_audited():
    rows=rows_fixture();summary=summarize(rows)
    expected={(r["clip"],r["model"],r["frame"]):r for r in rows}
    audit_scores(rows,summary,expected)
    stale=deepcopy(summary);stale["edsr"]["lpips"]+=.1
    with pytest.raises(ValueError,match="Stale perceptual summary"):audit_scores(rows,stale,expected)
    changed=deepcopy(rows);changed[0]["psnr"]+=1
    with pytest.raises(ValueError,match="producer metrics"):audit_scores(changed,summary,expected)
    changed=deepcopy(rows);changed[0]["reference_rgb_sha256"]="c"*64
    with pytest.raises(ValueError,match="reference pixels"):audit_scores(changed,summary,expected)
    with pytest.raises(ValueError,match="coverage"):audit_scores(rows[1:],summary,expected)


def published_source(tmp_path):
    run=tmp_path/"run";run.mkdir()
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps({'clips':[{'id':'a','frames':121,'fps':'60'}]}))
    producer={'pid':123,'manifest_sha256':perceptual.sha256(manifest),'metrics_every':60}
    (run/'environment.json').write_text(json.dumps(producer))
    result={'clip':'a','model':'bicubic','frames':121,'inference_calls':100,'reused_frames':21,
            'lossless_roundtrip_verified':True}
    (run/'clips.json').write_text(json.dumps([result]))
    with (run/'quality.csv').open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=['clip','model','frame']);writer.writeheader()
        for frame in [0,60,120]:writer.writerow(dict(clip='a',model='bicubic',frame=frame))
        writer.writerow(dict(clip='a',model='edsr',frame=0))  # In-flight clip is never eligible.
    return run,manifest,producer,result


def test_follow_snapshot_requires_actual_matching_live_producer(tmp_path,monkeypatch):
    run,manifest,producer,result=published_source(tmp_path)
    calls=[];monkeypatch.setattr(perceptual.os,'kill',lambda pid,signal:calls.append((pid,signal)))
    _,done,_,results,expected=perceptual.source_snapshot(run,manifest,123)
    assert done is None and results==[result] and len(expected)==3
    assert calls==[(123,0)]
    with pytest.raises(ValueError,match='incomplete'):perceptual.source_snapshot(run,manifest)
    with pytest.raises(ValueError,match='PID'):perceptual.source_snapshot(run,manifest,124)
    def gone(*args):raise ProcessLookupError
    monkeypatch.setattr(perceptual.os,'kill',gone)
    with pytest.raises(RuntimeError,match='exited without completing'):
        perceptual.source_snapshot(run,manifest,123)
    assert not (run/'COMPLETED.json').exists()


def test_follow_snapshot_rejects_partial_or_mutated_published_clips(tmp_path,monkeypatch):
    run,manifest,producer,result=published_source(tmp_path)
    monkeypatch.setattr(perceptual.os,'kill',lambda *args:None)
    for field,value in [('lossless_roundtrip_verified',False),('frames',120),('inference_calls',99)]:
        changed=dict(result,**{field:value});(run/'clips.json').write_text(json.dumps([changed]))
        with pytest.raises(ValueError,match='reconstruction checks'):
            perceptual.source_snapshot(run,manifest,123)
    (run/'clips.json').write_text(json.dumps([result]))
    raw=(run/'quality.csv').read_text();(run/'quality.csv').write_text(raw.replace('a,bicubic,60\n',''))
    with pytest.raises(ValueError,match='Incomplete published'):
        perceptual.source_snapshot(run,manifest,123)
    assert list(perceptual.pending_results([result],{}))==[result]
    assert list(perceptual.pending_results([result],{('a','bicubic'):result}))==[]
    with pytest.raises(ValueError,match='Previously scored'):
        list(perceptual.pending_results([dict(result,frames=122)],{('a','bicubic'):result}))


def test_follow_completion_always_enters_full_source_audit(tmp_path,monkeypatch):
    run,manifest,producer,result=published_source(tmp_path)
    (run/'COMPLETED.json').write_text('{}');calls=[]
    def complete_audit(*args):
        calls.append(args)
        return json.loads(manifest.read_text()),{'status':{'bicubic':{'state':'completed'}}},producer,[result]
    monkeypatch.setattr(perceptual,'audit',complete_audit)
    def no_pid_check(*args):raise AssertionError('Completed producer need not remain alive')
    monkeypatch.setattr(perceptual.os,'kill',no_pid_check)
    assert perceptual.source_snapshot(run,manifest,123)[1] is not None
    assert calls==[(run,manifest)]


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('kind',['random','flat','identical','checkerboard','edges'])
def test_fp64_tensor_pixel_metrics_match_skimage(device,kind):
    import torch
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(0)
    a=rng.integers(0,256,(33,37,3),dtype=np.uint8)
    b=np.clip(a.astype(np.int16)+rng.integers(-20,21,a.shape),0,255).astype(np.uint8)
    if kind=='flat':a[:]=128;b[:]=129
    elif kind=='identical':a=a[:7,:7];b=a.copy()
    elif kind=='checkerboard':
        a[:]=(np.indices(a.shape[:2]).sum(0)%2*255)[...,None];b=255-a
    elif kind=='edges':a[:]=255;a[3:8,4:7]=0;b=a.copy();b[:,0]=0
    expected=perceptual.metrics(a,b);got=perceptual.tensor_rgb_metrics(a,b,device)
    assert got['psnr']==pytest.approx(expected['psnr'],abs=1e-10)
    assert got['ssim']==pytest.approx(expected['ssim'],abs=1e-10)
    with pytest.raises(ValueError,match='equal RGB'):
        perceptual.tensor_rgb_metrics(a[:6],b[:6],device)


@pytest.mark.parametrize('outcome',['completed','producer_failed','changed_result','changed_environment',
                                    'resumed','resumed_pixels_changed'])
def test_follow_runner_never_certifies_partial_or_changed_source(tmp_path,monkeypatch,outcome):
    import lpips
    import torch
    import torchvision
    run=tmp_path/'source';run.mkdir();out=tmp_path/'scores'
    manifest_path=tmp_path/'manifest.json'
    for name in ['reference.mkv','low.mp4']:(tmp_path/name).write_bytes(name.encode())
    (run/'bicubic.mkv').write_bytes(b'output')
    clip=dict(id='a',frames=1,fps='60',reference='reference.mkv',low='low.mp4',
              reference_sha256=perceptual.sha256(tmp_path/'reference.mkv'),
              low_sha256=perceptual.sha256(tmp_path/'low.mp4'))
    manifest={'clips':[clip]};manifest_path.write_text(json.dumps(manifest))
    producer={'pid':123,'metrics_every':60};(run/'environment.json').write_text(json.dumps(producer))
    (run/'clips.json').write_text('[]');(run/'quality.csv').write_text('producer rows')
    result=dict(clip='a',model='bicubic',output='bicubic.mkv',sha256=perceptual.sha256(run/'bicubic.mkv'))
    expected={('a','bicubic',0):dict(psnr=25,ssim=.8,time_s=0)}
    done={'status':{'bicubic':{'state':'completed'}}};calls=[]
    def snapshot(*args):
        calls.append(args)
        assert not (out/'COMPLETED.json').exists()
        if len(calls)==1:return manifest,None,producer,[result],expected
        if outcome=='producer_failed':raise RuntimeError('Producer exited without completing')
        if outcome=='changed_environment':(run/'environment.json').write_text('changed')
        final=dict(result,sha256='0'*64) if outcome=='changed_result' else result
        return manifest,done,producer,[final],expected
    monkeypatch.setattr(perceptual,'source_snapshot',snapshot)
    monkeypatch.setattr(perceptual,'sampled_rgb',lambda *args:iter([(0,0.,np.zeros((32,32,3),np.uint8))]))
    monkeypatch.setattr(perceptual,'metrics',lambda *args:dict(psnr=25,ssim=.8))
    net_calls=[]
    class Network:
        def eval(self):return self
        def to(self,device):return self
        def state_dict(self):return {}
        def __call__(self,*args):net_calls.append(args);return torch.tensor(0.)
    monkeypatch.setattr(lpips,'LPIPS',lambda **kwargs:Network())
    cache=tmp_path/'torch';(cache/'checkpoints').mkdir(parents=True)
    backbone=cache/'checkpoints'/Path(torchvision.models.AlexNet_Weights.IMAGENET1K_V1.url).name
    backbone.write_bytes(b'cached backbone')
    linear=tmp_path/'weights/v0.1/alex.pth';linear.parent.mkdir(parents=True);linear.write_bytes(b'cached calibration')
    monkeypatch.setattr(torch.hub,'get_dir',lambda:str(cache))
    monkeypatch.setattr(lpips,'__file__',str(tmp_path/'fake_lpips.py'))
    monkeypatch.setattr(perceptual,'BACKBONE_SHA256',perceptual.sha256(backbone))
    monkeypatch.setattr(perceptual,'LINEAR_SHA256',perceptual.sha256(linear))
    monkeypatch.setattr(perceptual.sys,'argv',['video_perceptual','--run',str(run),'--manifest',str(manifest_path),
        '--output',str(out),'--device','cpu','--threads','1','--follow-pid','123'])
    if outcome.startswith('resumed'):
        previous=tmp_path/'previous';previous.mkdir()
        provenance=dict(manifest_sha256=perceptual.sha256(manifest_path),
            producer_environment_sha256=perceptual.sha256(run/'environment.json'),run=str(run.resolve()),
            state_sha256=perceptual.hashlib.sha256().hexdigest(),backbone_sha256=perceptual.BACKBONE_SHA256,
            linear_sha256=perceptual.LINEAR_SHA256,lpips_version=perceptual.importlib.metadata.version('lpips'),
            network='alex',metric_version='0.1',torch=torch.__version__,torchvision=torchvision.__version__,device='cpu')
        (previous/'environment.json').write_text(json.dumps(provenance))
        rgb_hash=perceptual.hashlib.sha256(np.zeros((32,32,3),np.uint8).tobytes()).hexdigest()
        cached=dict(clip='a',model='bicubic',frame=0,time_s=0,psnr=25,ssim=.8,lpips=.125,
            reference_rgb_sha256=rgb_hash,output_rgb_sha256=rgb_hash if outcome=='resumed' else '0'*64)
        with (previous/'quality.csv').open('w',newline='') as output:
            writer=csv.DictWriter(output,fieldnames=list(cached));writer.writeheader();writer.writerow(cached)
        perceptual.sys.argv.extend(['--resume-from',str(previous)])
    threads=torch.get_num_threads()
    try:
        if outcome in ['completed','resumed']:
            perceptual.main()
            assert json.loads((out/'COMPLETED.json').read_text())['rows']==1
            assert json.loads((out/'environment.json').read_text())['producer_complete']
            if outcome=='resumed':
                saved=list(csv.DictReader((out/'quality.csv').open()))
                assert float(saved[0]['lpips'])==.125 and len(net_calls)==1  # Self-check only.
        else:
            with pytest.raises((RuntimeError,ValueError)):perceptual.main()
            assert (out/'FAILED.json').is_file() and not (out/'COMPLETED.json').exists()
    finally:torch.set_num_threads(threads)
    assert len(calls)==(1 if outcome=='resumed_pixels_changed' else 2)


def test_resume_rejects_different_metric_state(tmp_path):
    path=tmp_path/'previous';path.mkdir()
    names=['manifest_sha256','producer_environment_sha256','run','state_sha256','backbone_sha256',
           'linear_sha256','lpips_version','network','metric_version','torch','torchvision','device']
    env={name:name for name in names};(path/'environment.json').write_text(json.dumps(env))
    changed=dict(env,state_sha256='changed')
    with pytest.raises(ValueError,match='state_sha256'):perceptual.resume_scores(path,changed,{})


def test_reference_sample_cache_is_bounded_and_reuses_validated_pixels():
    cache=perceptual.ReferenceCache(6);calls=[]
    def load(key):
        calls.append(key);return [(0,0,np.zeros((1,1,3),np.uint8))]
    first=cache.get('a',lambda:load('a'))
    cache.get('b',lambda:load('b'))
    assert cache.get('a',lambda:load('a')) is first
    cache.get('c',lambda:load('c'))
    assert set(cache.values)=={'a','c'} and cache.bytes==6 and calls==['a','b','c']
    cache.get('b',lambda:load('b'))
    assert calls==['a','b','c','b'] and cache.bytes==6
    disabled=perceptual.ReferenceCache(0)
    disabled.get('a',lambda:load('a'));assert disabled.bytes==0 and not disabled.values
