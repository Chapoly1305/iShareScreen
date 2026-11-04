"""GPU engine identity must be checked before loading optional GPU libraries."""

import builtins

import pytest

from isharescreen.enhancement.cuda_surface import CudaSurfaceEDSR
from isharescreen.enhancement.cuda_native import NativeCudaSurfaceEDSR


@pytest.mark.parametrize("runtime", [CudaSurfaceEDSR, NativeCudaSurfaceEDSR])
def test_unverified_engine_never_loads_gpu_libraries(tmp_path, monkeypatch, runtime):
    engine = tmp_path / "wrong.engine"
    engine.write_bytes(b"not the verified engine")
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in ("torch", "tensorrt", "triton"):
            raise AssertionError("GPU library imported before engine hash verification")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="hash mismatch"):
        runtime(engine, engine_sha256="0" * 64, input_format="nv12_bt709_full")


def test_missing_expected_engine_identity_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="SHA256"):
        CudaSurfaceEDSR(tmp_path / "missing.engine", engine_sha256="")


def test_unspecified_nv12_color_interpretation_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Input format"):
        CudaSurfaceEDSR(
            tmp_path / "missing.engine", engine_sha256="0" * 64, input_format="nv12"
        )


def test_native_library_must_be_explicit(monkeypatch):
    from isharescreen.enhancement.cuda_decoder import NvdecDecoder

    monkeypatch.delenv("ISS_NVDEC_LIBRARY", raising=False)
    with pytest.raises(ValueError, match="library_path"):
        NvdecDecoder(None)


def test_native_frame_abi_matches_cpp():
    import ctypes
    from isharescreen.enhancement.cuda_decoder import Frame, GraphEntry

    assert ctypes.sizeof(Frame) == 56
    assert Frame.pts.offset == 24
    assert Frame.wait_ms.offset == 40
    assert ctypes.sizeof(GraphEntry) == 32
    assert GraphEntry.executable.offset == 16
