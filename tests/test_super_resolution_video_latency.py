"""Frame sampling and timing contracts for the optional GPU recording probe."""
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.super_resolution.video_latency import adapter_probe, resident, sample_indices, summary


def test_video_sampling_is_unique_ordered_and_covers_clip():
    indices = sample_indices(360, 120)
    assert len(indices) == len(set(indices)) == 120
    assert indices[0] == 0 and indices[-1] == 359
    assert indices == sorted(indices)
    assert sample_indices(3, 10) == [0, 1, 2]
    with pytest.raises(ValueError):
        sample_indices(0, 10)


def test_latency_summary_keeps_raw_samples():
    values = [1.0, 2.0, 3.0]
    result = summary(values)
    assert result["p50_ms"] == 2 and result["mean_ms"] == 2
    assert result["p95_ms"] == pytest.approx(2.9)
    assert result["samples_ms"] is values


def test_adapter_probe_excludes_warmup_calls():
    calls = []

    def adapter(rgb):
        calls.append(rgb)
        return rgb, {"total_ms": float(len(calls))}

    result = adapter_probe(adapter, [0, 1, 2], 2)
    assert calls == [0, 1, 0, 1, 2]
    assert result["total_ms"]["samples_ms"] == [3.0, 4.0, 5.0]


def test_host_timed_resident_path_and_output_shape():
    torch = pytest.importorskip("torch")
    adapter = SimpleNamespace(device="cpu", dtype=torch.float32, provenance={"precision": "fp32"},
        key="plainusr", sync=lambda: None,
        model=lambda value: torch.nn.functional.interpolate(value, scale_factor=2, mode="nearest"))
    result = resident(adapter, [np.zeros((540,960,3), np.uint8)], 1)
    assert result["timer"] == "synchronized-host-wall"
    assert result["output_shape"] == [1,3,1080,1920]
    assert len(result["samples_ms"]) == 1


def test_strict_probe_rejects_cpu_capable_defaults_before_loading_models(monkeypatch, tmp_path, capsys):
    import sys
    from benchmarks.super_resolution.video_latency import main
    monkeypatch.setenv("ISS_SR_GPU_ONLY", "1")
    monkeypatch.setattr(sys, "argv", ["video_latency", "--video", str(tmp_path/"video.mkv"),
                                     "--output", str(tmp_path/"output")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "excludes Core ML and CPU baselines" in capsys.readouterr().err
    assert not (tmp_path/"output").exists()
