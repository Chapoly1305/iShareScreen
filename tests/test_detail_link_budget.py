from benchmarks.super_resolution.mac_readability.detail_replay import shared_link
from benchmarks.super_resolution.mac_readability.detail_transport import wire_bytes


def test_shared_serializer_counts_both_streams_and_cancels_obsolete_detail():
    # An oversized first-page refinement cannot fit before the page changes.
    # It must not block new base frames or leak into the next generation.
    run = dict(
        group="synthetic",
        probe=dict(
            packets=[dict(pts_time=str(i / 10), size="2000") for i in range(60)]
        ),
    )
    tile = dict(
        payload_bytes=250000, wire_bytes=wire_bytes(250000), width=256, height=256
    )
    images = [dict(state_id=i + 1, encode_ms=1, tiles=[tile]) for i in range(2)]
    result = shared_link(run, images)
    events = result["events"]
    assert result["cancelled_queued_detail_wire_bytes"] > 0
    assert result["wire_bytes"]["base"] > 0 and result["wire_bytes"]["detail"] > 0
    assert all(a["finish_s"] <= b["start_s"] for a, b in zip(events, events[1:]))
    assert sum(e["wire"] for e in events) * 8 <= events[-1]["finish_s"] * 600000 + 1e-5
    assert not any(
        e["accepted"]
        and e["kind"] == "detail"
        and e["state_index"] == 0
        and e["finish_s"] >= 3
        for e in events
    )
    assert all(s["complete_age_s"] is None for s in result["states"])
    # Base priority allows at most one in-flight 1244-byte fragment to obstruct
    # an arriving base packet, followed by its own two fragment transmissions.
    assert result["base_frame_delay_p95_ms"] < (1244 + 2088) * 8 / 600
