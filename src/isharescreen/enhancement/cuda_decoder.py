"""Owned fixed-format GPU mappings from a separately built native NVDEC shim.

Linux/NVIDIA only. The parser accepts complete progressive HEVC8 access units
at 1280x720, full-range BT.709 4:2:0. No implicit host pixel copies occur.
Call release() with a completed GPU reader event before decoding another frame.
"""

import ctypes as c
import os
from pathlib import Path


class Frame(c.Structure):
    _fields_ = [
        ("pointer", c.c_uint64),
        ("pitch", c.c_uint32),
        ("width", c.c_uint32),
        ("height", c.c_uint32),
        ("reserved", c.c_uint32),
        ("pts", c.c_int64),
        ("queries", c.c_uint64),
        ("wait_ms", c.c_double),
        ("map_ms", c.c_double),
    ]


class GraphEntry(c.Structure):
    _fields_ = [
        ("pointer", c.c_uint64),
        ("pitch", c.c_uint32),
        ("reserved", c.c_uint32),
        ("executable", c.c_uint64 * 2),
    ]


assert c.sizeof(Frame) == 56 and c.sizeof(GraphEntry) == 32


class MappedFrame:
    def __init__(self, owner, metadata, generation):
        self.owner, self.metadata, self.generation = owner, metadata, generation
        self.pointer, self.pitch, self.pts = (
            metadata.pointer,
            metadata.pitch,
            metadata.pts,
        )
        self.stream = owner.stream

    def validate(self):
        if not self.owner.mapped or self.generation != self.owner.generation:
            raise RuntimeError("Stale or released decoder mapping")
        if (
            self.metadata.width != 1280
            or self.metadata.height != 720
            or self.pitch < 1280
            or not self.pointer
        ):
            raise ValueError("Unexpected native NV12 surface")

    def as_tensor(self):
        """Correctness inspection only; the timed fast path never calls this."""
        import torch

        self.validate()

        class Surface:
            __cuda_array_interface__ = dict(
                shape=(1080, 1280),
                strides=(self.pitch, 1),
                typestr="|u1",
                data=(self.pointer, False),
                version=3,
                stream=self.stream.cuda_stream,
            )

        with torch.cuda.stream(self.stream):
            tensor = torch.as_tensor(Surface(), device="cuda")
        assert tensor.data_ptr() == self.pointer
        return tensor


class WrongContext(RuntimeError):
    """Rejected before native work; caller can restore its context and retry."""


class NvdecDecoder:
    """Explicit-lifetime decoder; pass a built library path or ISS_NVDEC_LIBRARY."""

    def __init__(self, stream, polling=True, *, library_path=None):
        location = library_path or os.environ.get("ISS_NVDEC_LIBRARY")
        if not location:
            raise ValueError(
                "Provide library_path or ISS_NVDEC_LIBRARY for the native decoder"
            )
        self.stream = stream
        self.lib = c.CDLL(str(Path(location).resolve()))
        self.lib.nv_error.restype = c.c_char_p
        self.lib.nv_create.argtypes = [c.c_uint64, c.c_int]
        self.lib.nv_create.restype = c.c_void_p
        self.lib.nv_decode.argtypes = [
            c.c_void_p,
            c.c_uint64,
            c.c_uint64,
            c.c_int64,
            c.POINTER(Frame),
        ]
        self.lib.nv_decode.restype = c.c_int
        self.lib.nv_decode_graph.argtypes = [
            c.c_void_p,
            c.c_uint64,
            c.c_uint64,
            c.c_int64,
            c.POINTER(Frame),
            c.POINTER(GraphEntry),
            c.c_uint,
            c.c_uint,
            c.c_uint64,
        ]
        self.lib.nv_decode_graph.restype = c.c_int
        for name in (
            "nv_release",
            "nv_abort",
            "nv_check_context",
            "nv_set_complete_frames",
        ):
            getattr(self.lib, name).argtypes = [c.c_void_p]
            getattr(self.lib, name).restype = c.c_int
        self.lib.nv_destroy.argtypes = [c.c_void_p]
        self.lib.nv_destroy.restype = None
        self.handle = self.lib.nv_create(stream.cuda_stream, int(polling))
        if not self.handle:
            raise RuntimeError(self.lib.nv_error().decode())
        self.mapped = self.poisoned = False
        self.generation = 0
        self._reserved_ticket = None
        if self.lib.nv_set_complete_frames(self.handle):
            error = self.lib.nv_error().decode()
            self.lib.nv_destroy(self.handle)
            self.handle = None
            raise RuntimeError(error)

    def check_context(self):
        if not self.handle:
            raise RuntimeError("Decoder closed")
        code = self.lib.nv_check_context(self.handle)
        if code == -2:
            raise WrongContext(self.lib.nv_error().decode())
        if code:
            raise RuntimeError(self.lib.nv_error().decode())

    def validate_packet(self, packet):
        if not self.handle or self.poisoned or self.mapped:
            raise RuntimeError(
                "Decoder closed, aborted, or previous mapping not released"
            )
        if packet.discardable or packet.bDiscontinuity:
            raise ValueError("Contiguous complete-frame packets required")

    def _result(self, count, metadata):
        if count == -2:
            raise WrongContext(self.lib.nv_error().decode())
        if count < 0:
            raise RuntimeError(self.lib.nv_error().decode())
        if not count:
            return None
        self.mapped = True
        self.generation += 1
        return MappedFrame(self, metadata, self.generation)

    def decode_prepared(self, ticket, packet):
        if self._reserved_ticket is not ticket:
            raise RuntimeError("Decoder does not own this submission ticket")
        count = self.lib.nv_decode_graph(
            ticket.handle,
            packet.bsl_data,
            packet.bsl,
            packet.pts,
            ticket.metadata_ref,
            ticket.bindings,
            len(ticket.bindings),
            ticket.index,
            ticket.ready_handle,
        )
        return self._result(count, ticket.metadata)

    def decode(self, packet):
        if self._reserved_ticket is not None:
            raise RuntimeError("Cancel or submit the reserved ticket first")
        self.check_context()
        self.validate_packet(packet)
        metadata = Frame()
        count = self.lib.nv_decode(
            self.handle, packet.bsl_data, packet.bsl, packet.pts, c.byref(metadata)
        )
        return self._result(count, metadata)

    def release(self, completed):
        self.check_context()
        # An explicit completed event proves that all caller GPU reads ended.
        if not completed.query():
            raise RuntimeError("GPU readers still use the mapped surface")
        if not self.mapped:
            raise RuntimeError("No mapped surface")
        if self.lib.nv_release(self.handle):
            raise RuntimeError(self.lib.nv_error().decode())
        self.mapped = False

    def abort(self):
        if not self.handle:
            return
        self.poisoned = True
        if self.lib.nv_abort(self.handle):
            raise RuntimeError(self.lib.nv_error().decode())
        self.mapped = False
        self.generation += 1

    def close(self):
        if self._reserved_ticket is not None:
            raise RuntimeError("Cancel the reserved ticket before closing its decoder")
        if self.handle:
            self.check_context()
        if self.mapped:
            raise RuntimeError("Release the GPU surface before closing")
        if self.handle:
            self.lib.nv_destroy(self.handle)
            self.handle = None
