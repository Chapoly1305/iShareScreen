"""Partitioning, immutable resume inputs and explicit cache-binding regressions."""
from copy import deepcopy
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("skimage")
pytest.importorskip("PIL")
from benchmarks.super_resolution.resume_parallel import (
    assemble, derived_cache, partition, read_json, read_rows, snapshot,
    subset_manifest, validate_runtime, write_json, write_rows,
)
from isharescreen.enhancement.models import sha256


def fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("reference", "low"):
        (tmp_path / name).write_bytes(name.encode())
    clip = dict(id="clip", frames=61, fps="60/1", reference="reference", low="low",
                reference_sha256=sha256(tmp_path / "reference"), low_sha256=sha256(tmp_path / "low"))
    path = tmp_path / "manifest.json"
    write_json(path, dict(clips=[clip]))
    (source / "video.mkv").write_bytes(b"verified output fixture")
    env = {key: key for key in ("python", "platform", "torch", "av", "cuda", "runner_sha256", "adapter_sha256", "precision", "gpu")}
    env.update(manifest_sha256=sha256(path), reuse_identical=True, metrics_every=60)
    result = dict(clip="clip", model="bicubic", output="video.mkv", frames=61, inference_calls=61,
                  reused_frames=0, lossless_roundtrip_verified=True, sampled_quality_frames=2,
                  sha256=sha256(source / "video.mkv"), psnr=20., ssim=.8)
    rows = [dict(clip="clip", model="bicubic", frame=i, time_s=i/60, psnr=20., ssim=.8) for i in (0, 60)]
    write_json(source / "environment.json", env)
    write_json(source / "clips.json", [result])
    write_rows(source / "quality.csv", rows)
    return source, path, env, result, rows


def test_partition_has_exact_disjoint_coverage_and_balances():
    clips = [dict(id=str(i), frames=n) for i, n in enumerate([100, 90, 80, 30, 20, 10])]
    groups = partition(clips, 3)
    assert sorted(c["id"] for group in groups for c in group) == sorted(c["id"] for c in clips)
    assert [sum(c["frames"] for c in group) for group in groups] == [110, 110, 110]
    assert partition([], 3) == []
    assert len(partition(clips[:1], 3)) == 1


def test_snapshot_ignores_only_unpublished_partial_rows(tmp_path):
    source, path, env, result, rows = fixture(tmp_path)
    partial = dict(rows[0], clip="not-published")
    (source / "quality.csv").unlink()
    write_rows(source / "quality.csv", rows + [partial])
    _, _, records, accepted = snapshot(source, path)
    assert records == [result] and len(accepted) == 2
    (source / "video.mkv").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="video changed"):
        snapshot(source, path)


def test_snapshot_rejects_missing_or_duplicate_samples(tmp_path):
    source, path, env, result, rows = fixture(tmp_path)
    for invalid in (rows[:1], rows + [rows[0]]):
        (source / "quality.csv").unlink()
        write_rows(source / "quality.csv", invalid)
        with pytest.raises(ValueError, match="coverage"):
            snapshot(source, path)


def test_assembly_is_fresh_disjoint_complete_and_audited(tmp_path):
    source, path, env, result, rows = fixture(tmp_path)
    status = {"bicubic": dict(state="completed")}
    target = tmp_path / "assembled"
    assemble(target, path, env, status, [(source, env, [result], rows)])
    assert (target / "COMPLETED.json").exists()
    assert sha256(target / "bicubic/clip.mkv") == result["sha256"]
    assert read_json(target / "environment.json")["execution"] == "resumed_clip_shards"
    with pytest.raises(FileExistsError):
        assemble(target, path, env, status, [])
    with pytest.raises(ValueError, match="Overlapping"):
        assemble(tmp_path / "overlap", path, env, status, [(source, env, [result, result], rows)])
    with pytest.raises(ValueError, match="Incomplete"):
        assemble(tmp_path / "missing", path, env, status, [])


def test_assembly_rejects_runtime_drift_and_removes_positive_marker(tmp_path):
    source, path, env, result, rows = fixture(tmp_path)
    with pytest.raises(ValueError, match="precision"):
        validate_runtime(env, dict(env, precision="changed"))
    bad = deepcopy(rows)
    bad[0]["psnr"] = 0
    target = tmp_path / "stale"
    with pytest.raises(AssertionError, match="Stale quality"):
        assemble(target, path, env, {"bicubic": dict(state="completed")}, [(source, env, [result], bad)])
    assert not (target / "COMPLETED.json").exists()
    assert (target / "AUDIT_REJECTED.json").exists()


def test_derived_cache_is_explicit_and_requires_identical_published_outputs(tmp_path):
    source, path, env, result, rows = fixture(tmp_path)
    manifest = read_json(path)
    view_manifest = tmp_path / "subset.json"
    write_json(view_manifest, subset_manifest(manifest, manifest["clips"], path.parent))
    view = tmp_path / "view"
    assemble(view, view_manifest, env, {"bicubic": dict(state="completed")}, [(source, env, [result], rows)])
    cache = tmp_path / "cache"
    cache.mkdir()
    original_env = dict(producer_environment_sha256=sha256(source / "environment.json"),
                        manifest_sha256=sha256(path), run=str(source.resolve()), state_sha256="state")
    write_json(cache / "environment.json", original_env)
    cached = [dict(r, lpips=.1, reference_rgb_sha256="a"*64, output_rgb_sha256="b"*64) for r in rows]
    write_rows(cache / "quality.csv", cached)
    derived = tmp_path / "derived"
    derived_cache(derived, cache, source, path, view, view_manifest)
    bound = read_json(derived / "environment.json")
    assert bound["derived_cache"]["origin_environment"] == original_env
    assert bound["producer_environment_sha256"] == sha256(view / "environment.json")
    assert read_rows(derived / "quality.csv") == read_rows(cache / "quality.csv")
    assert read_json(cache / "environment.json") == original_env
    changed = read_json(view / "clips.json")
    changed[0]["sha256"] = "changed"
    write_json(view / "clips.json", changed)
    with pytest.raises(ValueError, match="not identical"):
        derived_cache(tmp_path / "rejected", cache, source, path, view, view_manifest)
