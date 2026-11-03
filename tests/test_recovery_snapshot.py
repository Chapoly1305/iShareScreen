import json
from types import SimpleNamespace

from benchmarks.super_resolution.mac_readability.recovery_snapshot import snapshot


def test_snapshot_preserves_recovery_counters_without_payloads(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.super_resolution.mac_readability.recovery_snapshot.time.monotonic",
        lambda: 10.0,
    )
    gate = SimpleNamespace(
        _keyframe_required={2},
        _idr_observed={0},
        _fir_attempts=[1, 0, 3],
        _idr_observed_at=[9.0, 0.0, 8.0],
    )
    decoder = SimpleNamespace(
        _gate=gate, _feed_total=400, _decode_errors=37, _last_pub_t=9.5
    )
    session = SimpleNamespace(
        _decoder=decoder,
        _lost_pkts_per_tile=[1, 2, 3],
        _nack_pending={123: {4, 5}},
        _last_fir_per_tile={2: 9.8},
        private_payload=b"secret",
    )
    value = snapshot(session)
    assert value["decoder_feed_total"] == 400 and value["decoder_error_total"] == 37
    assert value["decoder_last_publish_age_ms"] == 500
    assert value["last_idr_age_ms"] == [1000, None, 2000]
    assert value["nack_pending_count"] == 2 and value["keyframe_required_tiles"] == [2]
    assert "secret" not in json.dumps(value)
    assert session._nack_pending == {123: {4, 5}} and gate._keyframe_required == {2}


def test_snapshot_handles_decoder_not_initialized():
    value = snapshot(SimpleNamespace())
    assert value["decoder_feed_total"] is None and value["decoder_error_total"] is None
    assert value["keyframe_required_tiles"] == [] and value["nack_pending_count"] == 0
