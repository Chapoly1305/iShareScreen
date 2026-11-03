"""Prevent fold leakage, checkpoint substitution and best-fold reporting bias."""

import json

import pytest

from benchmarks.super_resolution.finetuned_text_poc import (
    aggregate_group,
    checkpoint_identity,
    choose,
)
from benchmarks.super_resolution.training_state import fingerprint
from isharescreen.enhancement.models import sha256


def completed(tmp_path):
    policy = dict(
        schema="sr_candidate_true_epoch_protocol_v1",
        schedule=dict(model="quicksrnet", fold=3),
        folds_sha256="folds",
        pairs_sha256="pairs",
        official_checkpoint_sha256="official",
        sources={},
    )
    (tmp_path / "best.pt").write_bytes(b"selected weights")
    done = dict(
        model="quicksrnet",
        fold=3,
        epochs=1500,
        best_epoch=1490,
        best_step=100,
        protocol_binding=fingerprint(policy),
        checkpoint_sha256=sha256(tmp_path / "best.pt"),
    )
    (tmp_path / "protocol.json").write_text(json.dumps(policy))
    (tmp_path / "COMPLETED.json").write_text(json.dumps(done))
    return done


def test_checkpoint_selection_rejects_wrong_held_out_fold(tmp_path):
    completed(tmp_path)
    identity = checkpoint_identity(tmp_path, "quicksrnet", 3, "folds", "pairs")
    assert identity["fold"] == 3 and identity["epoch"] == 1490
    with pytest.raises(ValueError, match="model/fold"):
        checkpoint_identity(tmp_path, "quicksrnet", 0, "folds", "pairs")


def test_changed_selected_weights_and_unbound_protocol_are_rejected(tmp_path):
    done = completed(tmp_path)
    (tmp_path / "best.pt").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="Changed selected"):
        checkpoint_identity(tmp_path, "quicksrnet", 3, "folds", "pairs")
    done["checkpoint_sha256"] = sha256(tmp_path / "best.pt")
    done["protocol_binding"] = "unbound"
    (tmp_path / "COMPLETED.json").write_text(json.dumps(done))
    with pytest.raises(ValueError, match="Unbound"):
        checkpoint_identity(tmp_path, "quicksrnet", 3, "folds", "pairs")


def test_scores_average_folds_and_scenes_without_cherry_picking():
    rows = [
        dict(scene="a", fold=0, region="text", edit_distance=0, characters=100),
        dict(scene="a", fold=1, region="text", edit_distance=100, characters=100),
        dict(scene="b", fold=0, region="text", edit_distance=0, characters=900),
        dict(scene="b", fold=1, region="text", edit_distance=0, characters=900),
    ]
    result = aggregate_group(rows)
    assert result["cer"] == 0.25  # Not best-fold 0 or globally character-pooled .05.
    assert result["worst_scene_fold_cer"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_group(rows + [rows[0]])


def test_selection_accounts_for_keyframe_and_keeps_baseline_eligible():
    rows = []
    for model, variant, values, passed in [
        ("bicubic", "baseline", [0.10, 0.08], True),
        ("quicksrnet", "finetuned", [0.30, 0.0], True),
        ("plainusr", "finetuned", [0.0, 0.0], False),
    ]:
        for t, cer in zip([2, 3], values):
            rows.append(
                dict(
                    dataset="pdf",
                    config="540p-600k-420-30",
                    model=model,
                    variant=variant,
                    time_s=t,
                    cer=cer,
                    passed=passed,
                    parameters=dict(bps=600000),
                )
            )
    selected = choose(rows, "pdf", 600000)
    assert selected["variant"] == "baseline" and selected["max_sampled_cer"] == 0.10
