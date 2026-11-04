"""Optional GPU-surface EDSR runtime, independent of the CPU image adapter.

Accepts CUDA uint8 RGB or explicitly configured BT.709 full-range NV12 tensors
and decoder DLPack frames. Pixels never
leave the GPU. One inference may be in flight and two output slots are retained
until consumers release them. A busy runtime rejects submission so the caller
can replace its pending frame rather than build a latency-producing GPU queue.

This is the inference handoff API, not a renderer or a registered viewer model.
Output uses float Area resize to 1080p, followed by RGB8 clipping/quantization.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import threading


class SurfaceBusy(RuntimeError):
    """The caller should retain only its newest pending frame and retry later."""


class SurfaceLease:
    """Exclusive ownership of one GPU output until its consumer finishes.

    Call wait_on(consumer_stream), enqueue all reads there, then release().
    Dropped frames use discard() without borrowing. No method downloads pixels.
    """

    def __init__(self, runtime, slot, serial):
        self._runtime, self._slot, self._serial = runtime, slot, serial
        self._consumer = None
        self._released = False

    def _check(self):
        if self._released:
            raise RuntimeError("GPU surface has already been released")

    def ready(self):
        self._check()
        return self._slot["ready"].query()

    def wait_on(self, stream):
        """Enqueue a GPU wait and return the tensor, without blocking the CPU."""
        self._check()
        if stream.device != self._runtime.device:
            raise ValueError("Consumer must use the same CUDA device")
        if self._consumer is not None and stream.cuda_stream != self._consumer.cuda_stream:
            raise RuntimeError("A surface lease supports one consumer stream")
        stream.wait_event(self._slot["ready"])
        self._consumer = stream
        return self._slot["output"]

    def wait_input_reuse_on(self, stream):
        """Order writes to a reusable input buffer after preprocessing reads.

        Some decoder pools reuse frame memory even while DLPack/Python owners
        still exist. Call this on that decoder's CUDA stream before its next
        Decode call. The wait is on GPU and does not wait for EDSR completion.
        Keep the decoder itself alive until its input buffer is no longer read.
        """
        self._check()
        if stream.device != self._runtime.device:
            raise ValueError("Input producer must use the same CUDA device")
        stream.wait_event(self._slot["input_consumed"])

    def release(self):
        self._check()
        if self._consumer is None:
            raise RuntimeError("Borrow with wait_on() first, or discard an unused surface")
        self._runtime._release(self._slot, self._serial, self._consumer)
        self._released = True

    def discard(self):
        self._check()
        if self._consumer is not None:
            raise RuntimeError("A borrowed surface must be released after its consumer reads")
        self._runtime._release(self._slot, self._serial, None)
        self._released = True


class CudaSurfaceEDSR:
    """Hash-bound TensorRT batch-one 720p -> 1440p -> 1080p GPU runtime.

    Construction and submit/close belong to one producer thread. Leases may be
    consumed on another thread using a CUDA stream on the same device. The
    caller owns color interpretation. RGB8 is the default. The explicit
    nv12_bt709_full input mode requires 8-bit BT.709 full-range NV12; other
    color matrices, ranges, chroma layouts and HDR are not accepted by it.
    """

    def __init__(self, engine_path, *, engine_sha256, device=0, output_format="rgba", input_format="rgb"):
        if not re.fullmatch(r"[0-9a-f]{64}", engine_sha256):
            raise ValueError("A valid expected engine SHA256 is required")
        if output_format not in ("rgb", "rgba"):
            raise ValueError("Output format must be rgb or rgba")
        if input_format not in ("rgb", "nv12_bt709_full"):
            raise ValueError("Input format must be rgb or nv12_bt709_full")
        blob = Path(engine_path).read_bytes()
        if hashlib.sha256(blob).hexdigest() != engine_sha256:
            raise ValueError("TensorRT engine hash mismatch")
        import torch
        import tensorrt as trt
        from .cuda_kernels import normalize_rgb, normalize_nv12_709_full, pack_area

        self._torch, self._normalize, self._pack = torch, normalize_rgb, pack_area
        self._normalize_nv12 = normalize_nv12_709_full
        self._input_format = input_format
        self.device = torch.device("cuda", device)
        self._thread = threading.get_ident()
        self._lock = threading.Lock()
        self._closed = False
        self._slots = []
        self._pending = None
        self._last_ready = None
        self.provenance = dict(engine_sha256=engine_sha256, tensorrt=trt.__version__,
            input=("CUDA RGB8 1280x720" if input_format == "rgb" else "CUDA NV12 BT.709 full-range 1280x720"),
            output=f"CUDA {output_format.upper()}8 1920x1080",
            gpu_pixels_only=True, output_slots=2, max_inflight_inference=1)
        with torch.cuda.device(self.device), torch.inference_mode():
            self.stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(self.stream):
                self._logger = trt.Logger(trt.Logger.WARNING)
                self._runtime = trt.Runtime(self._logger)
                self._engine = self._runtime.deserialize_cuda_engine(blob)
                if self._engine is None:
                    raise RuntimeError("TensorRT could not load this engine on the selected device")
                expected = {"input": (1,3,720,1280), "output": (1,3,1440,2560)}
                if self._engine.num_io_tensors != 2:
                    raise ValueError("Expected the validated two-tensor EDSR engine")
                for name, shape in expected.items():
                    if (tuple(self._engine.get_tensor_shape(name)) != shape
                            or self._engine.get_tensor_dtype(name) != trt.float32
                            or self._engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                        raise ValueError("Unexpected EDSR tensor contract: " + name)
                self._context = self._engine.create_execution_context()
                self._input = torch.zeros(expected["input"], device=self.device)
                self._native = torch.empty(expected["output"], device=self.device)
                self._context.set_tensor_address("input", self._input.data_ptr())
                self._context.set_tensor_address("output", self._native.data_ptr())
                channels = 4 if output_format == "rgba" else 3
                self._channels = channels
                for _ in range(2):
                    output = torch.empty((1080,1920,channels), dtype=torch.uint8, device=self.device)
                    for _ in range(3):
                        self._compute(output)
                    self.stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self.stream):
                        self._compute(output)
                    ready, consumed, input_consumed = torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()
                    ready.record(self.stream); consumed.record(self.stream); input_consumed.record(self.stream)
                    self._slots.append(dict(output=output, graph=graph, ready=ready, consumed=consumed,input_consumed=input_consumed,
                                            busy=False, serial=0, discarded=False))
                # Compile the selected format before any timed submission.
                shapes = [(3,720,1280),(720,1280,3)] if input_format == "rgb" else [(1080,1280)]
                for shape in shapes:
                    scratch = torch.zeros(shape,dtype=torch.uint8,device=self.device)
                    self._preprocess(scratch)
                self.stream.synchronize()
        self.context_memory_bytes = self._engine.device_memory_size_v2

    def _compute(self, output):
        if not self._context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        self._pack[((output.numel()+255)//256,)](self._native,output,*self._native.stride()[1:],
            self._channels,output.numel(),256,QUANTIZE_FIRST=False,enable_fp_fusion=False)

    def _producer(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError("GPU runtime must be submitted/closed by its producer thread")
        if self._closed:
            raise RuntimeError("GPU runtime is closed")

    def _preprocess(self, tensor):
        if tensor.device != self.device or tensor.dtype != self._torch.uint8:
            raise ValueError("Input must be uint8 pixels on the selected CUDA device")
        n = self._input.numel()
        if self._input_format == "nv12_bt709_full":
            if tuple(tensor.shape) != (1080,1280):
                raise ValueError("NV12 input must contain 720 Y rows followed by 360 interleaved UV rows")
            strides = tensor.stride()
            if any(s <= 0 for s in strides):
                raise ValueError("Input pixel strides must be positive")
            self._normalize_nv12[((n+255)//256,)](tensor,self._input,*strides,n,256,enable_fp_fusion=False)
        else:
            if tuple(tensor.shape) == (3,720,1280):
                strides = tensor.stride()
            elif tuple(tensor.shape) == (720,1280,3):
                strides = (tensor.stride(2),tensor.stride(0),tensor.stride(1))
            else:
                raise ValueError("Input must be RGB8 1280x720 in CHW or HWC layout")
            if any(s <= 0 for s in strides):
                raise ValueError("Input pixel strides must be positive")
            self._normalize[((n+255)//256,)](tensor,self._input,*strides,n,256)

    def submit(self, frame):
        """Submit a CUDA frame in the configured format, or raise SurfaceBusy.

        A tensor producer must submit on its producer CUDA stream. DLPack frames
        supply producer/consumer ordering through their __dlpack__ protocol.
        The frame and tensor remain referenced until inference completion, but
        retaining references does not prevent a decoder pool from overwriting
        their storage. Before such reuse, call lease.wait_input_reuse_on() on
        the decoder stream and keep the decoder itself alive.
        """
        self._producer()
        torch = self._torch
        with self._lock, torch.cuda.device(self.device), torch.inference_mode():
            if self._last_ready is not None and not self._last_ready.query():
                raise SurfaceBusy("Previous GPU inference is still running")
            self._pending = None
            available = [s for s in self._slots if not s["busy"]
                         and (s["ready"] if s["discarded"] else s["consumed"]).query()]
            if not available:
                raise SurfaceBusy("Both GPU outputs are still owned by consumers")
            slot = available[0]
            producer = torch.cuda.current_stream(self.device)
            with torch.cuda.stream(self.stream):
                if isinstance(frame, torch.Tensor):
                    tensor = frame
                    self.stream.wait_stream(producer)
                else:
                    tensor = torch.from_dlpack(frame)
                try:
                    self._preprocess(tensor)
                    slot["input_consumed"].record(self.stream)
                    slot["graph"].replay()
                    slot["ready"].record(self.stream)
                except BaseException:
                    # Keep externally owned decoder memory valid even after a
                    # partial enqueue. This wait is an error-path cleanup only.
                    self.stream.synchronize()
                    raise
                slot.update(busy=True,serial=slot["serial"]+1,discarded=False)
                self._last_ready = slot["ready"]
                self._pending = (frame,tensor)
                return SurfaceLease(self,slot,slot["serial"])

    def _release(self, slot, serial, consumer):
        with self._lock:
            if not slot["busy"] or slot["serial"] != serial:
                raise RuntimeError("Stale GPU surface lease")
            if consumer is not None:
                slot["consumed"].record(consumer)
            slot.update(busy=False,discarded=consumer is None)

    def collect_completed(self):
        """Release a completed input owner, e.g. before replacing its decoder.

        Returns False while inference is running. Output leases remain owned
        by their consumers and are unaffected. This never waits on the CPU.
        """
        self._producer()
        with self._lock:
            if self._last_ready is not None and not self._last_ready.query():
                return False
            self._pending = None
            return True

    def close(self):
        self._producer()
        with self._lock:
            if any(s["busy"] for s in self._slots):
                raise SurfaceBusy("Release or discard output leases before closing")
            with self._torch.cuda.device(self.device):
                self.stream.synchronize()
                for slot in self._slots:
                    if not slot["discarded"]:
                        slot["consumed"].synchronize()
                self._pending = None
                self._slots.clear()
                self._context = self._engine = self._runtime = None
                self._input = self._native = None
                self._closed = True

    def __del__(self):
        # Explicit close is preferred. A live lease retains this runtime, so a
        # destructor cannot run while an external consumer still owns output.
        try:
            if hasattr(self, "stream") and not self._closed:
                self.stream.synchronize()
                for slot in self._slots:
                    if not slot["discarded"]:
                        slot["consumed"].synchronize()
        except Exception:
            pass
