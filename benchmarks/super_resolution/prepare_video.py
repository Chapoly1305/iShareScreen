"""Export complete aligned 1080p references and inter-frame 540p HEVC videos."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

from isharescreen.enhancement.models import sha256


def run(command):
    subprocess.run(command, check=True, capture_output=True)


def probe(path):
    data=json.loads(subprocess.check_output([
        "ffprobe","-v","error","-select_streams","v:0","-count_frames",
        "-show_streams","-of","json",str(path)],text=True))
    return data["streams"][0]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--captures",nargs="+")
    p.add_argument("--review",type=Path,help="Optional JSON with excluded, content and session_group fields")
    p.add_argument("--crf",type=int,default=28)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    review=json.loads(args.review.read_text()) if args.review else {"excluded": {}, "content": {}, "session_group": "unspecified"}
    videos=sorted(args.dataset.glob("*.mov"))
    if args.captures:
        videos=[v for v in videos if v.stem in args.captures]
        if {v.stem for v in videos}!=set(args.captures):raise ValueError("Requested capture not found")
    else:
        videos=[v for v in videos if v.stem not in review["excluded"]]
    if not videos:raise ValueError("No eligible videos")
    manifest={"protocol":"complete_video_full709_lanczos_x2_v1","command":sys.argv,
              "generator_sha256":sha256(Path(__file__)),"ffmpeg":subprocess.check_output(
                  ["ffmpeg","-version"],text=True).splitlines()[0],
              "reference":"Decoded RGB8 proxy from original recording, not lossless acquisition",
              "excluded":review["excluded"],"clips":[]}
    for source in videos:
        folder=args.output/source.stem;folder.mkdir()
        original=probe(source)
        if (original["width"],original["height"])!=(1920,1080):
            raise ValueError(f"Not a 1920x1080 capture: {source}")
        reference=folder/"reference-rgb8.mkv"
        low=folder/"low-540p-hevc.mp4"
        commands=[
            ["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-i",str(source),
             "-map","0:v:0","-an","-sn","-vf","setpts=PTS-STARTPTS,format=rgb24",
             "-fps_mode","passthrough","-c:v","libx264rgb","-crf","0","-preset","fast","-g","120",
             "-threads","4","-pix_fmt","rgb24",str(reference)],
            ["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-i",str(reference),
             "-map","0:v:0","-an","-vf",
             "scale=960:540:flags=lanczos:in_range=full:out_range=full:out_color_matrix=bt709",
             "-fps_mode","passthrough","-c:v","libx265","-preset","fast","-crf",str(args.crf),
             "-pix_fmt","yuv444p","-color_range","pc","-colorspace","bt709",
             "-color_primaries","bt709","-color_trc","bt709",
             "-x265-params","pools=4:frame-threads=2:keyint=120:min-keyint=1:log-level=error",
             "-tag:v","hvc1","-movflags","+faststart",str(low)]]
        for command in commands:run(command)
        hr,lr=probe(reference),probe(low)
        count=int(hr["nb_read_frames"])
        assert count==int(original["nb_read_frames"])==int(lr["nb_read_frames"]),"Frame count changed"
        assert Fraction(hr["avg_frame_rate"])==Fraction(original["avg_frame_rate"])==Fraction(lr["avg_frame_rate"]),"Frame rate changed"
        assert (lr["width"],lr["height"])==(960,540) and lr["pix_fmt"]=="yuv444p"
        clip={"id":source.stem,"source":str(source),"source_sha256":sha256(source),
              "session_group":review["session_group"],"content":review["content"].get(source.stem),
              "reference":str(reference.relative_to(args.output)),"reference_sha256":sha256(reference),
              "low":str(low.relative_to(args.output)),"low_sha256":sha256(low),
              "frames":count,"fps":original["avg_frame_rate"],"original_stream":original,
              "reference_stream":hr,"low_stream":lr,"crf":args.crf,"commands":commands}
        manifest["clips"].append(clip)
        (folder/"clip.json").write_text(json.dumps(clip,indent=2))
        print(f"EXPORTED {source.stem}: {count} frames at {clip['fps']}, complete clip",flush=True)
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"COMPLETED {len(videos)} complete video pairs",flush=True)


if __name__=="__main__":main()
