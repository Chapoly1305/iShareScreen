"""Score completed recordings while other recordings are still encoding."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

from benchmarks.super_resolution.validation_baselines import read, evaluate, report
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def worker(args):
    import benchmarks.super_resolution.validation_baselines as runner
    from benchmarks.super_resolution.validation_gpu_quality import quality
    runner.pixel_quality=quality
    root=args.root.resolve();p=read(root/'PROTOCOL.json');pending=[c['id'] for c in p['source_clips']]
    started=time.time()
    while pending:
        ready=[]
        for clip in pending:
            receipts=[root/'encodes'/clip/c['id']/'COMPLETED.json' for c in p['configurations']]
            if all(f.is_file() for f in receipts):ready.append((clip,receipts))
        if not ready:
            if time.time()-started>21600:raise TimeoutError('Six hours waiting for completed encodes')
            time.sleep(10)
            continue
        clip,receipts=ready[0];local=root/'by-clip'/clip;local.mkdir(parents=True,exist_ok=True)
        for name in ('PROTOCOL.json','encodes','references'):
            path=local/name
            if not path.is_symlink():path.symlink_to(os.path.relpath(root/name,local),target_is_directory=name!='PROTOCOL.json')
        runs=[read(f) for f in receipts]
        atomic_json(local/'INPUTS.json',dict(protocol_sha256=sha256(root/'PROTOCOL.json'),runs=runs))
        atomic_json(local/'ENCODING-COMPLETED.json',dict(inputs_sha256=sha256(local/'INPUTS.json')))
        settings=argparse.Namespace(root=local,models=p['models'],device=args.device,workers=args.workers)
        evaluate(settings)
        for method in p['models']:
            identity=local/'evaluations'/method/'identity.json'
            metadata=read(identity)
            metadata['pixel_metric_acceleration']=dict(
                source_sha256=sha256(Path(__file__).with_name('validation_gpu_quality.py')),
                definition='Same full-resolution skimage-default uniform 7x7 SSIM; float64 exact integral RGB moments. Previously completed CPU rows retained.',
                validation='See GPU-METRIC-VALIDATION.json; full-frame SSIM agreement verified to 1e-10.')
            atomic_json(identity,metadata)
        atomic_json(local/'COMPLETED.json',dict(clip=clip,methods=p['models'],finished=time.time()))
        pending.remove(clip)
        atomic_json(root/'evaluation-progress.json',dict(completed_clips=len(p['source_clips'])-len(pending),total_clips=len(p['source_clips']),latest_clip=clip,elapsed_s=time.time()-started))
        print('CLIP COMPLETED',clip,len(p['source_clips'])-len(pending),'/',len(p['source_clips']),flush=True)
    atomic_json(root/'PIPELINE-COMPLETED.json',dict(clips=len(p['source_clips']),finished=time.time()))


def merge(args):
    root=args.root.resolve();p=read(root/'PROTOCOL.json');all_runs=[]
    for clip in p['source_clips']:
        child=root/'by-clip'/clip['id'];assert (child/'COMPLETED.json').is_file()
        inputs=read(child/'INPUTS.json')
        assert inputs['protocol_sha256']==sha256(root/'PROTOCOL.json')
        assert read(child/'ENCODING-COMPLETED.json')['inputs_sha256']==sha256(child/'INPUTS.json')
        for run in inputs['runs']:
            assert run['clip']==clip['id']
            assert sha256(root/run['video'])==run['video_sha256']
            for sample in run['samples']:
                for field in ('lr','hr'):assert sha256(root/sample[field])==sample[field+'_sha256']
        all_runs+=inputs['runs']
    assert len(all_runs)==len(p['source_clips'])*len(p['configurations'])
    atomic_json(root/'INPUTS.json',dict(protocol_sha256=sha256(root/'PROTOCOL.json'),runs=all_runs))
    atomic_json(root/'ENCODING-COMPLETED.json',dict(inputs_sha256=sha256(root/'INPUTS.json'),runs=len(all_runs),finished=time.time()))
    for method in p['models']:
        rows=[];identities=[]
        for clip in p['source_clips']:
            child=root/'by-clip'/clip['id'];folder=child/'evaluations'/method;path=folder/'records.jsonl'
            receipt=read(folder/'COMPLETED.json');identity=read(folder/'identity.json')
            assert receipt['records_sha256']==sha256(path)
            assert identity['binding']['protocol_sha256']==sha256(root/'PROTOCOL.json')
            assert identity['binding']['inputs_sha256']==sha256(child/'INPUTS.json')
            part=[__import__('json').loads(s) for s in path.read_text().splitlines()]
            assert len(part)==len(p['configurations'])*3
            for row in part:
                assert row['method']==method and row['clip']==clip['id']
                if row['image']:row['image']=str(Path('by-clip')/clip['id']/row['image'])
            rows+=part;identities.append(identity)
        checkpoints={v['model'].get('checkpoint_sha256') for v in identities}
        assert len(checkpoints)==1
        assert len(rows)==len({(r['clip'],r['config'],r['frame']) for r in rows})==len(all_runs)*3
        folder=root/'evaluations'/method;folder.mkdir(parents=True,exist_ok=True)
        path=folder/'records.jsonl';path.write_text(''.join(__import__('json').dumps(r)+'\n' for r in rows))
        atomic_json(folder/'identity.json',dict(model=identities[0]['model'],per_clip_identities=identities))
        atomic_json(folder/'COMPLETED.json',dict(records=len(rows),records_sha256=sha256(path),finished=time.time()))
    report(args)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('worker','merge'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();globals()[args.action](args)


if __name__=='__main__':main()
