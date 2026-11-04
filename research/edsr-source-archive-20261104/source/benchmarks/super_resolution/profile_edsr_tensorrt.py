"""Hash-bound EDSR TensorRT deployment experiment, independent of the viewer."""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
import tensorrt as trt

from benchmarks.super_resolution.profile_inference_optimization import adapter, amp, digest, forward, guard, save, stats


class NormalizedEDSR(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x * 255) / 255


class EngineRunner:
    def __init__(self, path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.output = torch.empty((1, 3, 1440, 2560), device="cuda", dtype=torch.float32)
        self.context.set_tensor_address("output", self.output.data_ptr())

    def __call__(self, x):
        assert x.shape == (1, 3, 720, 1280) and x.dtype == torch.float32 and x.is_contiguous()
        self.context.set_tensor_address("input", x.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return self.output


class TrainingCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, root, output):
        super().__init__()
        self.root, self.output = root, output
        manifest = root / "DATA.json"
        data = json.loads(manifest.read_text())
        assert data["complete"]
        languages = {}
        for row in data["rows"]:
            if row["split"] != "train": continue
            language = row["sample"].split("/")[0]
            languages.setdefault(language, []).append(row)
        self.rows = [r for lang in sorted(languages) for r in sorted(languages[lang], key=lambda r:r["id"])[:3]]
        self.index = 0
        self.tensor = torch.empty((1, 3, 720, 1280), dtype=torch.float32, device="cuda")
        save(output / "CALIBRATION.json", dict(source_sha256=digest(manifest), split="train", count=len(self.rows),
             selection="First three IDs in each language, sorted deterministically; no quality-based selection",
             rows=[{k:r[k] for k in ("id","sample","split","lr","lr_sha256")} for r in self.rows]))

    def get_batch_size(self): return 1

    def get_batch(self, names):
        if self.index == len(self.rows): return None
        row = self.rows[self.index]; self.index += 1
        path = self.root / row["lr"]
        assert digest(path) == row["lr_sha256"]
        rgb = np.asarray(Image.open(path).convert("RGB"))
        self.tensor.copy_(torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0)).div_(255)
        torch.cuda.synchronize()
        return [self.tensor.data_ptr()]

    def read_calibration_cache(self): return None

    def write_calibration_cache(self, cache):
        (self.output / "calibration.cache").write_bytes(cache)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--engine", type=Path, help="Reuse an existing engine instead of rebuilding")
    p.add_argument("--int8-calibration-data", type=Path)
    p.add_argument("--quality", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    dest = a.output / "RESULTS.json"
    if dest.exists(): raise FileExistsError(dest)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    compute_stream = torch.cuda.Stream()
    torch.cuda.set_stream(compute_stream)
    item = next(i for i in json.loads((a.weights / "PLAN.json").read_text())["models"] if i["model"] == "edsr")
    ckpt = a.weights / "checkpoints/edsr.pt"
    assert digest(ckpt) == item["sha256"]
    result = dict(complete=False, checkpoint=item, gpu_start=guard(), started=time.time(),
                  pid=os.getpid(), trt=trt.__version__, torch=str(torch.__version__),
                  script_sha256=digest(Path(__file__)), optimization_level=a.level,
                  scope="TensorRT FP16 permitted, TF32 disabled, FP32 input/output, unchanged FP32 checkpoint exported to ONNX. Static batch1 720p -> native1440p, CUDA-event resident forward. No decode/transfer/resize/display.")
    save(dest, result)
    try:
        model = adapter("edsr", "cpu").model.eval().requires_grad_(False)
        torch.nn.Module.load_state_dict(model, torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"], strict=True)
        onnx = a.output / "edsr.onnx"
        with torch.inference_mode():
            torch.onnx.export(NormalizedEDSR(model), torch.zeros(1, 3, 720, 1280), str(onnx),
                              opset_version=17, input_names=["input"], output_names=["output"],
                              do_constant_folding=True, dynamo=False)
        result["onnx_sha256"] = digest(onnx)
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx.read_bytes()):
            raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.TF32)
        if a.int8_calibration_data:
            calibrator = TrainingCalibrator(a.int8_calibration_data, a.output)
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            protected = []
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if layer.type == trt.LayerType.CONVOLUTION:
                    if layer.get_input(0).shape[1] == 3 or layer.get_output(0).shape[1] == 3:
                        layer.precision = trt.float16
                        layer.set_output_type(0, trt.float16)
                        protected.append(layer.name)
            result.update(precision="INT8 allowed internally; FP16 RGB-facing convolutions; FP32 IO; entropy calibration on train only",
                          protected_fp16_layers=protected)
            result["scope"] = result["scope"].replace("TensorRT FP16 permitted", "TensorRT INT8/FP16 mixed precision permitted")
            save(dest,result)
        config.builder_optimization_level = a.level
        config.avg_timing_iterations = 8
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        t0 = time.time()
        engine_blob = (a.engine.read_bytes() if a.engine else builder.build_serialized_network(network, config))
        if engine_blob is None: raise RuntimeError("TensorRT engine build failed")
        engine_path = a.output / "edsr.engine"
        engine_path.write_bytes(bytes(engine_blob))
        result.update(build_s=time.time()-t0, engine_sha256=digest(engine_path), network_layers=network.num_layers,
                      reused_engine=str(a.engine) if a.engine else None, nondefault_cuda_stream=True)
        print("Built EDSR TensorRT", result["build_s"], flush=True)
        runner = EngineRunner(engine_path)
        inspector = runner.engine.create_engine_inspector()
        (a.output / "engine-layers.json").write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        x = torch.linspace(0, 1, 1280*720*3, device="cuda").reshape(1, 3, 720, 1280)
        with torch.inference_mode():
            for _ in range(50): y=runner(x)
            torch.cuda.synchronize()
            assert torch.isfinite(y).all().item()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            times=[];wall=[]
            for _ in range(a.samples):
                t0=time.perf_counter();begin.record();y=runner(x);end.record();end.synchronize()
                times.append(begin.elapsed_time(end));wall.append((time.perf_counter()-t0)*1000)
            result.update(forward=stats(times),host_wall=stats(wall),gpu_after=guard())
            save(dest,result)
            print("TensorRT EDSR p95",result["forward"]["p95_ms"],flush=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(5):y=runner(x)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(a.output/"edsr-trt.trace.json"))
            if a.quality:
                model.to("cuda")
                runs=[r for r in json.loads((a.validation/"INPUTS.json").read_text())["runs"] if r["config"]=="720p-600k-420"]
                records=[]
                for run in runs:
                    for s in run["samples"]:
                        for k in ("lr","hr"): assert digest(a.validation/s[k])==s[k+"_sha256"]
                        low=np.asarray(Image.open(a.validation/s["lr"]).convert("RGB"))
                        hr=np.asarray(Image.open(a.validation/s["hr"]).convert("RGB"))
                        t=torch.from_numpy(np.ascontiguousarray(low.transpose(2,0,1))).unsqueeze(0).to("cuda").float()/255
                        ref=forward(model,"edsr",t)
                        with amp(): baseline=forward(model,"edsr",t)
                        output=runner(t)
                        assert torch.isfinite(output).all().item()
                        images={}
                        for k,v in [("fp32",ref),("baseline_amp",baseline),("optimized",output)]:
                            pixels=v[0].float().mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
                            images[k]=cv2.resize(pixels,(1920,1080),interpolation=cv2.INTER_AREA)
                        row=dict(clip=run["clip"],frame=s["frame"],psnr={})
                        for k,v in images.items():row["psnr"][k]=float(10*np.log10(255**2/np.mean((v.astype(np.float64)-hr.astype(np.float64))**2)))
                        for k in ("fp32","baseline_amp"):
                            diff=np.abs(images["optimized"].astype(np.int16)-images[k].astype(np.int16))
                            row["vs_"+k]=dict(max_rgb=int(diff.max()),mean_abs_rgb=float(diff.mean()),fraction_gt1=float((diff>1).mean()))
                        records.append(row)
                result["quality"]=dict(samples=len(records),mean_psnr={k:float(np.mean([r["psnr"][k] for r in records])) for k in images},records=records)
        result.update(complete=True,finished=time.time(),gpu_end=guard())
        save(dest,result)
    except BaseException as e:
        result.update(error=repr(e),failed=time.time());save(dest,result);raise


if __name__=="__main__":main()
