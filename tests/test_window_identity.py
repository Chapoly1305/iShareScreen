"""Compositor identity hints must tolerate older GLFW installations."""
from __future__ import annotations

import pytest

from isharescreen.frontend.desktop import app


@pytest.mark.parametrize("version, wayland", [((3, 3, 0), False), ((3, 4, 0), True)])
def test_identity_hints_match_native_library(monkeypatch, version, wayland):
    monkeypatch.setattr(app.sys, "platform", "linux")
    monkeypatch.setattr(app.glfw, "init", lambda: True)
    monkeypatch.setattr(app.glfw, "get_version", lambda: version)
    calls = []
    monkeypatch.setattr(app.glfw, "window_hint_string", lambda *args: calls.append(args))
    app._set_window_identity()
    assert (app.glfw.X11_CLASS_NAME, "isharescreen") in calls
    assert (app.glfw.X11_INSTANCE_NAME, "isharescreen") in calls
    assert ((app.glfw.WAYLAND_APP_ID, "isharescreen") in calls) is wayland


def test_missing_hint_api_is_optional(monkeypatch):
    monkeypatch.setattr(app.sys, "platform", "linux")
    monkeypatch.setattr(app.glfw, "init", lambda: True)
    monkeypatch.delattr(app.glfw, "window_hint_string")
    app._set_window_identity()


def test_identity_does_not_initialize_glfw_on_windows(monkeypatch):
    monkeypatch.setattr(app.sys, "platform", "win32")
    monkeypatch.setattr(app.glfw, "init", lambda: pytest.fail("Linux hints on Windows"))
    app._set_window_identity()
