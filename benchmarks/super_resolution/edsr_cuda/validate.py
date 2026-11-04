"""Actual frozen-frame output equivalence and native-callback ownership checks."""

import argparse
import inspect
import os
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import PyNvVideoCodec as nvc
from isharescreen.enhancement.cuda_decoder import NvdecDecoder as Decoder
from isharescreen.enhancement.cuda_native import (
    NativeCudaSurfaceEDSR as DirectSurfaceEDSR,
)
from isharescreen.enhancement.cuda_surface import SurfaceBusy
from benchmarks.super_resolution.edsr_cuda.common import (
    Engine,
    reference_rgb,
    guard,
    save,
    digest,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    root, engine, sha = args.validation, args.engine, args.engine_sha256
    os.environ["ISS_NVDEC_LIBRARY"] = str(args.native_library.resolve())
    os.environ["ISS_EXPECTED_GPU_UUID"] = args.expected_gpu_uuid
    result = dict(
        complete=False,
        gpu=guard(),
        engine_sha256=sha,
        frames=[],
        ownership={},
        runtime_sha256=digest(Path(inspect.getfile(DirectSurfaceEDSR))),
        script_sha256=digest(Path(__file__)),
        decoder_source_sha256=digest(Path(inspect.getfile(Decoder))),
        native_library_sha256=digest(args.native_library),
    )
    save(out, result)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    runtime = DirectSurfaceEDSR(
        engine, engine_sha256=sha, input_format="nv12_bt709_full"
    )
    stream = runtime.stream
    torch.cuda.set_stream(stream)
    consumer = torch.cuda.Stream()
    end = torch.cuda.Event()
    reference = Engine(engine)
    runs = [
        x
        for x in json.loads((root / "INPUTS.json").read_text())["runs"]
        if x["config"] == "720p-600k-420"
    ]
    for run in runs:
        path = root / run["video"]
        assert digest(path) == run["video_sha256"]
        demux = nvc.CreateDemuxer(filename=str(path))
        decoder = Decoder(stream)
        selected = {x["frame"] for x in run["samples"]}
        index = 0
        for packet in demux:
            if index in selected or index == 0:
                submitted = runtime.decode_submit(decoder, packet)
                if submitted is None:
                    continue
                frame, lease = submitted
                if index == 0 and index not in selected:
                    lease.discard()
                    stream.synchronize()
            else:
                frame = decoder.decode(packet)
                if frame is None:
                    continue
            assert frame.pts == packet.pts
            if index in selected:
                assert frame.metadata.reserved == 1, (
                    "Frozen frame did not exercise native graph launch"
                )
                output = lease.wait_on(consumer)
                with torch.cuda.stream(consumer):
                    snapshot = output.clone()
                lease.release()
                consumer.synchronize()
                rgb = reference_rgb(frame.as_tensor())
                expected_input = rgb.unsqueeze(0).float() / 255
                assert torch.equal(runtime._input, expected_input), (
                    run["clip"],
                    index,
                    "input",
                )
                reference.input.copy_(expected_input)
                y = reference.run()
                native = (
                    y[0]
                    .mul(255)
                    .round()
                    .clamp(0, 255)
                    .byte()
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                expected = cv2.resize(
                    native, (1920, 1080), interpolation=cv2.INTER_AREA
                )
                actual = snapshot.cpu().numpy()
                assert np.array_equal(expected, actual[:, :, :3]) and np.all(
                    actual[:, :, 3] == 255
                ), (run["clip"], index)
                result["frames"].append(
                    dict(
                        clip=run["clip"],
                        frame=index,
                        input_bit_exact=True,
                        rgba_bit_exact=True,
                        graph_cache_addresses=len(runtime._address_graphs),
                        callback_launched=True,
                    )
                )
            end.record(stream)
            end.synchronize()
            decoder.release(end)
            index += 1
            if index > max(selected):
                break
        assert runtime.collect_completed() and len(runtime._address_graphs) <= 2
        decoder.close()
        save(out, result)
    assert len(result["frames"]) == 72
    from benchmarks.super_resolution.edsr_cuda.ownership import check_ownership

    result["ownership"] = check_ownership(runtime, root / runs[0]["video"], engine, sha)
    from benchmarks.super_resolution.edsr_cuda.ticket_checks import check_tickets

    result["tickets"] = check_tickets(runtime, root / runs[0]["video"])
    result["callback_launches"] = runtime.callback_launches
    result["cold_launches"] = runtime.cold_launches
    result["graph_builds"] = runtime.graph_builds
    runtime.close()
    result["complete"] = True
    save(out, result)
    print("PASS: 72 exact outputs and native-callback ownership checks", flush=True)


if __name__ == "__main__":
    main()
