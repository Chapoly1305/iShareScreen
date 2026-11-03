"""Geometry and split protections for the new codec training path."""

import cv2
import numpy as np
import torch
import os

from benchmarks.super_resolution.codescan_campaign.data import (
    frame_at,
    render,
    frozen_recipe,
)
from isharescreen.enhancement.models import sha256
from PIL import Image
import json
import pytest
from benchmarks.super_resolution.codescan_campaign.runner import assigned_models
from benchmarks.super_resolution.codescan_campaign.train import (
    area_4_to_3,
    crop_coordinates,
    trainable_parameters,
    aggregate_crop_quality,
)
from benchmarks.super_resolution.codescan_campaign.monitor import heartbeat_age


def test_exact_area_resize_matches_display_and_has_gradients():
    image = np.random.default_rng(7).random((32, 48, 3), dtype=np.float32)
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].requires_grad_()
    result = area_4_to_3(tensor)
    expected = cv2.resize(image, (36, 24), interpolation=cv2.INTER_AREA)
    np.testing.assert_allclose(
        result[0].permute(1, 2, 0).detach().numpy(), expected, atol=2e-7
    )
    result.mean().backward()
    assert torch.isfinite(tensor.grad).all() and tensor.grad.min() > 0


def test_crop_is_aligned_and_reproducible_for_portrait_content():
    row = {
        "transform": {
            "content_box": [600, 0, 1320, 1080],
            "line_boxes": [[610, 400, 1310, 422]],
        }
    }
    for seed in range(50):
        x, y = crop_coordinates(row, seed)
        assert (x, y) == crop_coordinates(row, seed)
        assert x % 2 == y % 2 == 0 and 0 <= x <= 1152 and 0 <= y <= 592
        assert (x + 128) * 3 // 2 - x * 3 // 2 == 192
        assert (x + 128) * 1.5 > 600 and x * 1.5 < 1320


def test_motion_reference_uses_exact_encoded_frame_coordinates():
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[100:110] = 180
    bg = np.array([1, 2, 3], np.uint8)
    np.testing.assert_array_equal(frame_at(image, bg, 0), image)
    shift = -round(30 * (45 - 30) / 29)
    moved = frame_at(image, bg, 45)
    np.testing.assert_array_equal(moved[100 + shift : 110 + shift], image[100:110])
    assert (moved[shift:] == bg).all()


def test_fixed_rgb_normalization_is_not_optimized():
    class MeanShift(torch.nn.Conv2d):
        pass

    model = torch.nn.Sequential(MeanShift(3, 3, 1), torch.nn.Conv2d(3, 3, 3))
    model.requires_grad_(False)
    parameters = trainable_parameters(model)
    assert not any(p.requires_grad for p in model[0].parameters())
    assert all(p.requires_grad for p in model[1].parameters())
    assert {id(p) for p in parameters} == {id(p) for p in model[1].parameters()}


def test_perfect_validation_crop_does_not_break_aggregate():
    result = aggregate_crop_quality([0, 2], [1, 0.8])
    assert result["exact_crops"] == 1 and result["views"] == 2 and result["ssim"] == 0.9
    assert abs(result["psnr"] - 10 * np.log10(65025)) < 1e-10
    assert aggregate_crop_quality([0], [1])["psnr"] is None


def test_monitor_uses_worker_progress_and_ignores_completed_runs(tmp_path):
    status = tmp_path / "STATUS.json"
    status.write_text("{}")
    os.utime(status, (10, 10))
    worker = tmp_path / "models/edsr/STATUS.json"
    worker.parent.mkdir(parents=True)
    worker.write_text("{}")
    os.utime(worker, (90, 90))
    assert heartbeat_age(tmp_path, {"stage": "training", "model": "edsr"}, 100) == 10
    assert heartbeat_age(tmp_path, {"stage": "completed"}, 100000) is None


def test_null_text_annotation_is_skipped_without_changing_pixels(tmp_path):
    folder = tmp_path / "sample"
    folder.mkdir()
    Image.fromarray(np.full((32, 48, 3), 127, np.uint8)).save(folder / "img.png")
    annotations = folder / "annotations.json"

    def draw(lines):
        annotations.write_text(json.dumps({"annotations": [{"lines_data": lines}]}))
        row = {
            "sample": "sample",
            "files": {
                "img.png": sha256(folder / "img.png"),
                "annotations.json": sha256(annotations),
            },
        }
        return render(tmp_path, row)

    with_null = draw([{"text": None}])
    empty = draw([])
    for i in (0, 1):
        np.testing.assert_array_equal(with_null[i], empty[i])
    assert with_null[2] == empty[2]


def test_recipe_repair_requires_exact_registered_hashes(tmp_path):
    original = {"bitrate": 600000, "runner_sha256": "original"}
    assert frozen_recipe(tmp_path, original) == original
    changed = dict(original, runner_sha256="repaired")
    with pytest.raises(FileNotFoundError):
        frozen_recipe(tmp_path, changed)
    (tmp_path / "RUNNER-REPAIR.json").write_text(
        json.dumps(
            {"original_runner_sha256": "original", "repaired_runner_sha256": "repaired"}
        )
    )
    assert frozen_recipe(tmp_path, changed) == original
    with pytest.raises(AssertionError):
        frozen_recipe(tmp_path, dict(changed, bitrate=1000000))


def test_parallel_allocation_has_one_owner_per_model():
    plan = {"queues": {"lab": ["edsr", "plainusr"], "h100": ["imdn"]}}
    assert assigned_models(["edsr", "imdn", "plainusr"], plan, "h100") == ["imdn"]
    with pytest.raises(AssertionError):
        assigned_models(
            ["edsr", "imdn"],
            {"queues": {"lab": ["edsr"], "h100": ["edsr", "imdn"]}},
            "lab",
        )
    with pytest.raises(AssertionError):
        assigned_models(
            ["edsr", "imdn"], {"queues": {"lab": ["edsr"], "h100": []}}, "lab"
        )
