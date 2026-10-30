"""Encode disjoint recording shards on CPU; preserve one common protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import platform
import time

from benchmarks.super_resolution.validation_baselines import read, encode_one
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--project',type=Path,default=Path.cwd())
    parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--shards',type=int,default=2)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();root=args.root.resolve();p=read(root/'PROTOCOL.json')
    assert 0 <= args.shard < args.shards
    clips=[]
    for i,original in enumerate(p['source_clips']):
        if i % args.shards != args.shard:continue
        clip=dict(original)
        clip['reference']=str((args.project/'datasets/validation/mac/processed_recordings'/clip['id']/'reference-rgb8.mkv').resolve())
        assert sha256(Path(clip['reference']))==clip['reference_sha256']
        clips.append(clip)
    started=time.time();rows=[]
    progress=root/f'encode-shard-{args.shard}.json'
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(encode_one,root,p,clip,cfg) for clip in clips for cfg in p['configurations']]
        for future in as_completed(tasks):
            row=future.result();rows.append(row)
            atomic_json(progress,dict(host=platform.node(),shard=args.shard,completed=len(rows),total=len(tasks),elapsed_s=time.time()-started))
            print('ENCODED',args.shard,len(rows),'/',len(tasks),row['clip'],row['config'],flush=True)
    atomic_json(root/f'ENCODE-SHARD-{args.shard}-COMPLETED.json',dict(
        protocol_sha256=sha256(root/'PROTOCOL.json'),runner_sha256=sha256(Path(__file__)),
        host=platform.node(),shard=args.shard,runs=len(rows),elapsed_s=time.time()-started))


if __name__=='__main__':main()
