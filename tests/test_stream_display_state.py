import pytest

from benchmarks.super_resolution.mac_readability.stream_display_state import (
    DisplayState,
)


def test_codec_noise_does_not_undo_current_source_sr():
    state = DisplayState("source")
    state.begin(1)
    assert state.base(1)
    assert state.sr(1, 10, 12)  # New codec reconstruction, same source identity.
    assert not state.base(1) and state.phase == "sr"
    assert state.sr(1, 12, 15)
    assert not state.sr(1, 11, 15) and state.sr_revision == 12


def test_new_source_and_exact_source_preempt_all_older_results():
    state = DisplayState("source")
    state.begin(1)
    state.sr(1, 10, 12)
    state.begin(2)
    assert state.phase == "blank"
    assert not state.sr(1, 100, 101) and not state.base(1) and not state.detail(1)
    assert state.base(2) and state.detail(2)
    assert not state.sr(2, 101, 105) and not state.base(2)
    state.begin(3)
    assert state.base(3)
    with pytest.raises(ValueError):
        state.begin(3)


def test_controls_separate_hold_from_revision_acceptance():
    pixel = DisplayState("pixel")
    hold = DisplayState("hold")
    for state in [pixel, hold]:
        state.begin(1)
        assert not state.sr(1, 5, 6)
        assert state.sr(1, 6, 6)
    assert pixel.base(1)
    assert not hold.base(1)


def test_source_backed_refresh_preserves_detail_and_rejects_stale_work():
    from benchmarks.super_resolution.mac_readability.stream_display_state import (
        DisplayState,
    )

    state = DisplayState("hold")
    state.begin(1)
    assert not state.sr(1, 1, 1, source_backed=True)
    assert state.sr(1, 1, 1)
    assert state.detail(1)
    assert not state.sr(1, 2, 2)
    assert not state.sr(1, 2, 3, source_backed=True)
    assert state.sr(1, 3, 3, source_backed=True)
    assert state.phase == "detail"
    assert not state.base(1)
    assert not state.sr(1, 3, 3, source_backed=True)
    state.begin(2)
    assert not state.sr(1, 4, 4, source_backed=True)
    assert not state.sr(2, 4, 4, source_backed=True)
