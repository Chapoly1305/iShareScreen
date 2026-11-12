"""A second iss session on this machine must not share the media ports."""
from __future__ import annotations

import pytest

from isharescreen.proxy.session import Session


def test_second_bind_on_same_media_port_fails_clearly():
    first = Session._bind_udp("127.0.0.1", 0)
    port = first.getsockname()[1]
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            Session._bind_udp("127.0.0.1", port)
    finally:
        first.close()


def test_port_rebinds_after_previous_session_closes():
    first = Session._bind_udp("127.0.0.1", 0)
    port = first.getsockname()[1]
    first.close()

    second = Session._bind_udp("127.0.0.1", port)
    second.close()
