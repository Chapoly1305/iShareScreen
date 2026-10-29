"""Concurrent lifecycle operations must not interleave teardown and rebuild."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from isharescreen.proxy.media.hevc import HevcDecoder
from isharescreen.proxy.session import Session


@pytest.mark.parametrize("second", ["restart", "close", "fallback"])
def test_hevc_lifecycle_is_serialized(monkeypatch, second):
    decoder = HevcDecoder(1)
    decoder._vps, decoder._sps, decoder._all_pps = b"vps", b"sps", {0: b"pps"}
    entered, release, second_attempted = (threading.Event() for _ in range(3))
    order = []

    def teardown():
        order.append("teardown")
        if not entered.is_set():
            entered.set()
            assert release.wait(3)

    monkeypatch.setattr(decoder, "_teardown", teardown)
    monkeypatch.setattr(decoder, "_create_codec", lambda **kw: order.append("build"))

    def other_operation():
        second_attempted.set()
        if second == "fallback":
            decoder._fallback_to_software(None)
        else:
            getattr(decoder, second)()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(decoder.restart)
        try:
            assert entered.wait(3)
            other = pool.submit(other_operation)
            assert second_attempted.wait(3)
            acquired = decoder._lifecycle_lock.acquire(blocking=False)
            if acquired:
                decoder._lifecycle_lock.release()
            assert not acquired
            assert order == ["teardown"]
        finally:
            release.set()
        first.result(timeout=3)
        other.result(timeout=3)
    expected = ["teardown", "build", "teardown"]
    if second != "close":
        expected.append("build")
    assert order == expected


def test_restart_claim_has_one_winner_across_threads():
    session = Session.__new__(Session)
    session._restart_guard = threading.Lock()
    session._last_decoder_restart_t = 0.0
    barrier = threading.Barrier(8)

    def claim(_):
        barrier.wait(timeout=3)
        return session._claim_restart(100.0, 3.0)

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(claim, range(8))) == 1
    assert session._claim_restart(102.0, 3.0) is False
    assert session._claim_restart(103.0, 3.0) is True
    # A new AVC generation must reset even within the HEVC debounce interval.
    assert session._claim_restart(103.1, 0.0) is True
