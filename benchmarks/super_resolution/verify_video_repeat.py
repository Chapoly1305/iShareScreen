"""Verify selected completed clips across separate video runs, even a partial run.

This does not certify completion of either whole run. Container hashes can differ;
the strict reconstruction comparison uses full decoded RGB-sequence hashes that
each producer verified by lossless re-decoding, plus all sampled quality rows.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from isharescreen.enhancement.models import sha256


def inspect(run, manifest_path, clip_id, models, redecode=False):
    manifest=json.loads(manifest_path.read_text())
    env=json.loads((run/"environment.json").read_text())
    assert env["manifest_sha256"]==sha256(manifest_path),"Manifest changed"
    definitions=[c for c in manifest["clips"] if c["id"]==clip_id]
    assert len(definitions)==1,"Missing/duplicate clip"
    clip=definitions[0]
    for role in ["reference","low"]:
        assert sha256(manifest_path.parent/clip[role])==clip[f"{role}_sha256"],"Input file changed"
    results=json.loads((run/"clips.json").read_text())
    rows=list(csv.DictReader((run/"quality.csv").open()))
    status=json.loads((run/"status.json").read_text())
    lock=json.loads((Path(__file__).parent/"models.lock.json").read_text())
    selected={}
    for model in models:
        matching=[r for r in results if (r["clip"],r["model"])==(clip_id,model)]
        assert len(matching)==1,"Selected clip/model is not complete or is duplicated"
        result=matching[0]
        assert result["frames"]==clip["frames"] and result["lossless_roundtrip_verified"]
        assert Fraction(result["fps"])==Fraction(clip["fps"])
        assert sha256(run/result["output"])==result["sha256"],"Output changed"
        if redecode:
            import av
            digest=hashlib.sha256();count=0
            with av.open(str(run/result["output"])) as video:
                assert Fraction(video.streams.video[0].average_rate)==Fraction(clip["fps"])
                for frame in video.decode(video=0):
                    assert (frame.width,frame.height)==(1920,1080)
                    digest.update(frame.to_ndarray(format="rgb24").tobytes());count+=1
            assert count==result["frames"],"Re-decode frame count mismatch"
            assert digest.hexdigest()==result["decoded_rgb_sequence_sha256"],"Re-decode RGB mismatch"
        if model in lock:
            assert status[model]["source_sha"]==lock[model]["revision"]
            assert status[model]["checkpoint_sha256"]==lock[model]["sha256"]
        raw=[r for r in rows if (r["clip"],r["model"])==(clip_id,model)]
        indices=list(range(0,clip["frames"],env["metrics_every"]))
        if clip["frames"]-1 not in indices:indices.append(clip["frames"]-1)
        assert [int(r["frame"]) for r in raw]==indices,"Incomplete sampled quality"
        # Processing time is environment-sensitive and explicitly not compared.
        qualities=[{k:r[k] for k in ["frame","time_s","psnr","ssim","reused"]} for r in raw]
        selected[model]={"result":result,"qualities":qualities}
    return dict(environment=env,clip=clip,models=selected)


def compare(first, second):
    same_environment=["python","platform","torch","av","cuda","gpu","runner_sha256",
                      "adapter_sha256","precision","reuse_identical","metrics_every"]
    for key in same_environment:
        assert first["environment"][key]==second["environment"][key],f"Environment changed: {key}"
    for key in ["id","frames","fps","source_sha256","low_sha256"]:
        assert first["clip"][key]==second["clip"][key],f"Input changed: {key}"
    assert first["environment"]["pid"]!=second["environment"]["pid"],"Not separate producer processes"
    assert first["environment"]["started"]!=second["environment"]["started"],"Not separate runs"
    assert set(first["models"])==set(second["models"])
    fields=["frames","decoded_rgb_sequence_sha256","inference_calls","reused_frames",
            "sampled_quality_frames","psnr","ssim"]
    for model in first["models"]:
        a,b=first["models"][model],second["models"][model]
        for key in fields:
            assert a["result"][key]==b["result"][key],f"Repeat mismatch: {model}/{key}"
        assert a["qualities"]==b["qualities"],f"Sampled quality mismatch: {model}"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first-run",type=Path,required=True)
    p.add_argument("--first-manifest",type=Path,required=True)
    p.add_argument("--second-run",type=Path,required=True)
    p.add_argument("--second-manifest",type=Path,required=True)
    p.add_argument("--clip",required=True)
    p.add_argument("--models",nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--redecode",action="store_true",help="Independently decode and hash every selected output again")
    args=p.parse_args()
    first=inspect(args.first_run,args.first_manifest,args.clip,args.models,args.redecode)
    second=inspect(args.second_run,args.second_manifest,args.clip,args.models,args.redecode)
    compare(first,second)
    report=dict(status="EXACT_MATCH_SELECTED_COMPLETED_CLIP",clip=args.clip,models=args.models,
                independently_redecoded=args.redecode,
                frames_per_model=first["clip"]["frames"],first=first,second=second,
                limitations=["Not a certificate of full-batch completion or an independent capture session.",
                    "Timing is not compared. Container byte hashes may differ without RGB pixel changes.",
                    "Reference containers differ; sampled quality matches, but full reference RGB equality is not certified here."])
    with args.output.open("x") as output:json.dump(report,output,indent=2)
    print(f"EXACT MATCH: {args.clip}, {args.models}, {first['clip']['frames']} frames/model",flush=True)


if __name__=="__main__":main()
