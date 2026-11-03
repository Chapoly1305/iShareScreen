"""Numeric-only recovery counters for controlled captures; no packet contents."""

import time


def snapshot(session):
    decoder = getattr(session, "_decoder", None)
    gate = getattr(decoder, "_gate", None)
    now = time.monotonic()
    return dict(
        decoder_feed_total=getattr(decoder, "_feed_total", None),
        decoder_error_total=getattr(decoder, "_decode_errors", None),
        decoder_last_publish_age_ms=(now - decoder._last_pub_t) * 1000
        if getattr(decoder, "_last_pub_t", 0) > 0
        else None,
        lost_packets_per_tile=list(getattr(session, "_lost_pkts_per_tile", [])),
        nack_pending_count=sum(
            len(v) for v in list(getattr(session, "_nack_pending", {}).values())
        ),
        keyframe_required_tiles=sorted(getattr(gate, "_keyframe_required", set())),
        idr_observed_tiles=sorted(getattr(gate, "_idr_observed", set())),
        fir_attempts=list(getattr(gate, "_fir_attempts", [])),
        last_idr_age_ms=[
            (now - t) * 1000 if t > 0 else None
            for t in list(getattr(gate, "_idr_observed_at", []))
        ],
        last_fir_age_ms={
            str(i): (now - t) * 1000
            for i, t in list(getattr(session, "_last_fir_per_tile", {}).items())
        },
    )
