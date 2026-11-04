"""Check the actual NVDEC RGB conversion against 72 frozen validation frames."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
import PyNvVideoCodec as nvc

from benchmarks.super_resolution.profile_edsr_tiles import Engine
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save


def pixels(tensor):
    native = tensor[0].float().mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
    return cv2.resize(native, (1920,1080), interpolation=cv2.INTER_AREA)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--reference-quality", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    result = dict(complete=False, started=time.time(), gpu=guard(), engine_sha256=digest(a.engine),
        script_sha256=digest(Path(__file__)), nvc_version=nvc.__version__, records=[],
        scope="Quality only; GPU decoder converted RGB versus original frozen FFmpeg RGB. Same engine on both inputs. Pixel downloads are outside performance runs.")
    save(a.output, result)
    try:
        ref = json.loads(a.reference_quality.read_text())
        assert ref["complete"] and ref["engine_sha256"] == result["engine_sha256"]
        refs = {(r["clip"],r["frame"]):r for r in ref["quality"]["records"]}
        stream = torch.cuda.Stream(); torch.cuda.set_stream(stream)
        engine = Engine(a.engine)
        runs = [r for r in json.loads((a.validation / "INPUTS.json").read_text())["runs"] if r["config"] == "720p-600k-420"]
        for run in runs:
            path = a.validation / run["video"]; assert digest(path) == run["video_sha256"]
            decoder = nvc.SimpleDecoder(str(path), gpu_id=0, cuda_context=0, cuda_stream=stream.cuda_stream,
                use_device_memory=True, output_color_type=nvc.OutputColorType.RGBP)
            for sample in run["samples"]:
                for kind in ("lr","hr"):
                    assert digest(a.validation / sample[kind]) == sample[kind+"_sha256"]
                lr = np.asarray(Image.open(a.validation / sample["lr"]).convert("RGB"))
                hr = np.asarray(Image.open(a.validation / sample["hr"]).convert("RGB"))
                frame = decoder[sample["frame"]]
                tensor = torch.from_dlpack(frame)
                assert tensor.device.type == "cuda" and tensor.data_ptr() == frame.GetPtrToPlane(0)
                assert tuple(tensor.shape) == (3,720,1280)
                engine.input.copy_(tensor.unsqueeze(0).float()/255)
                enhanced = pixels(engine.run())
                gpu_rgb = tensor.permute(1,2,0).cpu().numpy()
                diff = np.abs(gpu_rgb.astype(np.int16)-lr.astype(np.int16))
                psnr = float(10*np.log10(255**2/np.mean((enhanced.astype(np.float64)-hr.astype(np.float64))**2)))
                baseline = refs[(run["clip"],sample["frame"])]
                result["records"].append(dict(clip=run["clip"],frame=sample["frame"], input_max_rgb=int(diff.max()),
                    input_mean_abs=float(diff.mean()), input_different_fraction=float(np.mean(diff!=0)),
                    psnr=dict(baseline["psnr"], nvdec=psnr), deltas={k:psnr-v for k,v in baseline["psnr"].items()}))
                del tensor, frame
            stream.synchronize(); del decoder
            save(a.output, result); print("validated",len(result["records"]),flush=True)
        assert len(result["records"]) == 72
        metrics = {k:dict(mean_delta=float(np.mean([r["deltas"][k] for r in result["records"]])),
                         worst_delta=min(r["deltas"][k] for r in result["records"])) for k in ("fp32","baseline_amp","optimized")}
        result.update(complete=True, finished=time.time(), metrics=metrics,
            passed=all(metrics[k]["mean_delta"]>=-.05 and metrics[k]["worst_delta"]>=-.1 for k in ("fp32","baseline_amp")))
        save(a.output, result); print(json.dumps(dict(metrics=metrics,passed=result["passed"]),indent=2),flush=True)
    except BaseException as exc:
        result.update(error=repr(exc)); save(a.output,result); raise


if __name__ == "__main__": main()
