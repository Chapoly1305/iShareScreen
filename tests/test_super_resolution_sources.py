"""Completed epoch provenance remains readable without retaining retired runners."""
from copy import deepcopy

import pytest

from benchmarks.super_resolution import train_epochs, training_state


ARCHIVED_SOURCES = {
    "train_epochs.py": "4488d22a9d03f410c43be7282d4c56d45a89df17c0b143bde69fd5221b175d33",
    "epochs.py": "9be03923811fb7274f18c43cde4066374e546af7e2a0fddb62bfb2e9dd13f6fd",
    "train.py": "8c6b7364d91c90251986160e075eb5dab31cfb5f9face2453676076baf1d4d36",
    "training_state.py": "64eb9e3b3ce2631a4c806ca16320f0c81345ff69838b6a312609346db3556ef5",
    "training_data.py": "ca0c4e8b8a793b53bece3636a85929150bfe33a424c49a37f46a6913c46d7e0f",
    "train_long.py": "0402676b33963c61467c2b789f3f6a0f79bc8a48788bb52154d0144e95dd34e2",
    "folds.py": "9ce50829f0a5a9a4dc9db393d25324e408d68738046298896c51ee2e62c8ca46",
    "setup_models.py": "3d202bfe0342a4c62f257dffdbddd78d1eb3cc135019bfeba68c120853cf790b",
    "models.lock.json": "22d3161b69174c7ac6313e7f13dc73c6c0352aa4eb96418623fb3e98ff71b041",
    "benchmark.py": "5deb528ae673f66ccc9986f8c582cf0120df4ddf479f474dcf532a16dffbed69",
    "video_perceptual.py": "bdbf75ddf79f0f56e76a9eb64b927edb6a75e4aafc156a75d26ba988ab07bda3",
    "adapters": "a6c10ba8fd7674bbe8e67e053f101f8113e2722db34b7f5fb9f04a7d0ec74067"
}


def test_current_training_sources_are_complete_and_match():
    sources = train_epochs.source_hashes()
    assert train_epochs.recorded_sources_match(sources)
    assert "metrics.py" in sources
    assert not {"train_long.py", "benchmark.py", "extend_training.py"} & sources.keys()
    changed = dict(sources, **{"train.py": "changed"})
    assert not train_epochs.recorded_sources_match(changed)


def test_completed_epoch_sources_match_the_pinned_historical_revision():
    assert training_state.fingerprint(ARCHIVED_SOURCES) == (
        "a0ba08a71834be7c483a9107f6bf1a1dc92219a53696c66d0d908aaff331bb9a"
    )
    assert train_epochs.recorded_sources_match(ARCHIVED_SOURCES)
    for key in ARCHIVED_SOURCES:
        changed = dict(ARCHIVED_SOURCES, **{key: "changed"})
        assert not train_epochs.recorded_sources_match(changed)


def test_archived_acceptance_is_read_only_and_requires_unchanged_adapter(monkeypatch):
    original = deepcopy(ARCHIVED_SOURCES)
    current = dict(train_epochs.source_hashes(), adapters="changed")
    monkeypatch.setattr(train_epochs, "source_hashes", lambda: current)
    assert not train_epochs.recorded_sources_match(original)
    assert original == ARCHIVED_SOURCES


def test_archived_epoch_checkpoint_cannot_silently_extend_with_changed_sources(tmp_path):
    import json
    from benchmarks.super_resolution.extend_epochs import prepare
    pytest.importorskip("torch")
    source, output = tmp_path / "completed", tmp_path / "extension"
    source.mkdir()
    (source / "protocol.json").write_text(json.dumps(dict(
        schema="sr_true_epoch_protocol_v1", sources=ARCHIVED_SOURCES
    )))
    for name in ("config.json", "COMPLETED.json"):
        (source / name).write_text("{}")
    with pytest.raises(ValueError, match="Pinned training sources changed"):
        prepare(source, output, 2000, tmp_path / "pairs", tmp_path / "folds", tmp_path)
    assert not output.exists()
