"""Retired profiling output must not flood the diagnostics log."""
from collections import deque
import queue

from isharescreen.gui import connect as C


def test_profiling_logs_are_not_buffered_or_broadcast(monkeypatch):
    logs, subscriber = deque(), queue.Queue()
    monkeypatch.setattr(C, '_LOG', logs)
    monkeypatch.setattr(C, '_SUBSCRIBERS', [subscriber])
    for message in (
        'INFO app | pipeline_timing: {"stages":{}}',
        'INFO app | presentation: draw_calls=170/s tile_uploads/s=[60]',
    ):
        C._emit('log', message)
    assert not logs and subscriber.empty()
    error = 'ERROR app | decoding failed'
    C._emit('log', error)
    assert list(logs) == [error]
    assert subscriber.get_nowait() == ('log', error)


def test_clear_removes_replay_history_and_notifies_open_tabs(monkeypatch):
    logs, subscriber = deque(['old log']), queue.Queue()
    monkeypatch.setattr(C, '_LOG', logs)
    monkeypatch.setattr(C, '_SUBSCRIBERS', [subscriber])
    C._clear_logs()
    assert not logs
    assert subscriber.get_nowait() == ('log_reset', {})
