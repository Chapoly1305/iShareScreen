"""Desktop frontend: glfw window + wgpu render loop + input forwarding.

The loop pumps glfw events, drains fresh tiles from `Session`, uploads
them to the GPU, and presents. The decoder runs on its own thread
inside `Session`; audio decode + playback also live off-thread (see
`audio_sink.AudioSink`), so neither competes with the input forwarder
on the render thread.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import glfw
import wgpu
from rendercanvas.glfw import RenderCanvas

from ...proxy.session import Session, SessionConfig
from .audio_sink import make_audio_sink
from .gpu import Renderer
from .keymap import GLFW_KEY_TO_X11, glfw_button_to_rfb_bit


# X11 keysyms for the Cmd-equivalent (Super_L / Super_R). When Ctrl→Cmd
# remap is on we substitute these for KEY_LEFT_CONTROL / KEY_RIGHT_CONTROL
# so the user's Linux Ctrl+C / Ctrl+A / Ctrl+V etc. land as Cmd+C/A/V
# on the macOS host. The Linux WM (GNOME) typically grabs the actual
# Super key for the activities overview, so without this remap there's
# no usable way to trigger Cmd-shortcuts on the Mac.
_KEYSYM_SUPER_L = 0xffeb
_KEYSYM_SUPER_R = 0xffec


log = logging.getLogger(__name__)


# How long the loop blocks waiting for the next tile before pumping
# events again. Short enough to stay responsive, long enough that we
# don't busy-spin and starve the decoder thread.
_FRESH_TILE_WAIT_S = 0.005


def run(
    config: SessionConfig,
    *,
    title: str = "iShareScreen",
    auto_quit_secs: int = 0,
    **_unused: object,
) -> int:
    """Open the window streaming `config`. Blocks until close (or
    `auto_quit_secs` elapses; 0 = forever)."""
    log.info("opening desktop frontend → %s", config.host)
    session = Session(config)
    session.connect()

    # Audio playback. `make_audio_sink()` returns None when the OS
    # has no usable output device or sounddevice is missing — we
    # silently continue with video-only in that case. When the sink
    # is up, the proxy's audio rx thread feeds it directly; the
    # render thread never touches it.
    audio_sink = make_audio_sink() if config.audio else None
    if audio_sink is not None:
        session.set_audio_callback(audio_sink.feed)

    # Clipboard sync (both directions) lives in Session — every frontend
    # gets it for free.

    # Cursor pseudo-encoding: by default the daemon stops baking the
    # system cursor into the encoded frames (saves ~0.5-1 Mbps of
    # cursor-motion bandwidth) and instead sends cursor pixmaps over
    # the RFB control channel. We hand each one to GLFW so the local
    # OS cursor adopts the host's shape — I-beam over text, resize
    # cursors over window edges, busy spinner, etc. Set
    # ISS_LEGACY_CURSOR=1 to revert to the cursor-in-framebuffer path.
    # Captured-by-reference so we can flip cursors from the proxy rx
    # thread; GLFW calls below all run on the render thread.
    # Sentinel that distinguishes "no pending change" from "pending =
    # None (revert to OS default)" — both pass `is None` so we use a
    # unique object instead.
    _NO_PENDING = object()
    _pending_cursor: dict[str, "Optional[object]"] = {"img": _NO_PENDING}
    _cursor_lock = threading.Lock()

    # GLFW set_cursor thread-safety:
    #   macOS: Cocoa requires NSCursor calls on main thread
    #   Linux X11 / Windows: GLFW handles cross-thread set_cursor
    # On Linux/Windows we apply directly from the proxy thread so we
    # don't lose intermediate cursor frames between video-frame-paced
    # render iterations. On macOS we defer to the render loop.
    # `_set_cursor_now` is defined later (after the GLFW window setup);
    # Python closure resolution happens at call time, by which point
    # the callback will only fire post-connect when everything's set up.
    import sys as _sys
    _cursor_direct_apply = _sys.platform in ("linux", "win32")

    def _on_cursor(img):
        if _cursor_direct_apply:
            _set_cursor_now(img)
        else:
            with _cursor_lock:
                _pending_cursor["img"] = img

    session.set_cursor_callback(_on_cursor)

    canvas_w, canvas_h = session.canvas_dims
    server_w, server_h = session.server_dims
    num_tiles = session.num_tiles
    # Apple's HEVC encodes each tile padded up to a CTU boundary —
    # typically 304 rows for a 1920×1200/4-tile canvas, not the 300
    # you'd get from canvas_h//num_tiles. Refined to tile.height on the
    # first tile that arrives.
    slot_h = canvas_h // num_tiles
    log.info("session ready: canvas=%dx%d server=%dx%d tiles=%d hw=%s",
             canvas_w, canvas_h, server_w, server_h, num_tiles,
             session.hw_accel)

    # ── window + wgpu surface ──────────────────────────────────────────
    window = RenderCanvas(title=title, size=(canvas_w, canvas_h), max_fps=120)
    glfw_window = window._window  # for raw glfw input callbacks only
    # Window resizes freely. The viewport always fills the window, so
    # the decoded video stretches to whatever aspect the user picks.
    # That's deliberate: when the advertised resolution doesn't match
    # the host's panel aspect, Apple's encoder bakes black bars into
    # the video; stretching to fill hides those bars at the cost of
    # mild aspect distortion when the user resizes off-aspect. No GPU
    # cost — it's the same shader pass either way.

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    surface_ctx = window.get_context("wgpu")
    # Pick the linear variant so we don't double-encode sRGB.
    preferred = surface_ctx.get_preferred_format(adapter)
    surface_format = (
        preferred[: -len("-srgb")] if preferred.endswith("-srgb") else preferred
    )
    surface_ctx.configure(
        device=device, format=surface_format, alpha_mode="opaque",
    )

    renderer = Renderer(device, surface_format, canvas_w, canvas_h)

    # ── input forwarding ───────────────────────────────────────────────
    button_mask = 0
    cursor: Optional[tuple[int, int]] = None

    def to_canvas(wx: float, wy: float) -> Optional[tuple[int, int]]:
        """Map glfw cursor (in glfw window coords) → canvas coords for
        `InputController.pointer_event`.

        glfw's cursor callback delivers coords in the same coordinate
        space as `glfw.get_window_size()`. With the aspect-lock above,
        the window dimensions are always the canvas scaled by some
        positive factor on both axes, so a single proportional rescale
        on each axis maps cursor → canvas exactly.

        Maps to canvas dims, NOT server-init dims: the daemon's
        composite ServerInit (e.g. 2940×1912 when a SkyLight virtual
        display is active alongside the panel) doesn't correspond to
        what's rendered in the iss window — only the canvas does.
        """
        win_w, win_h = glfw.get_window_size(glfw_window)
        if win_w == 0 or win_h == 0:
            return None
        sx = int(wx * canvas_w / win_w)
        sy = int(wy * canvas_h / win_h)
        return (max(0, min(canvas_w - 1, sx)),
                max(0, min(canvas_h - 1, sy)))

    def on_cursor_pos(_w, x, y):
        nonlocal cursor
        cursor = to_canvas(x, y)
        if cursor is not None:
            session.input.pointer_event(button_mask, cursor[0], cursor[1])

    def on_mouse_button(_w, button, action, _mods):
        nonlocal button_mask
        bit = glfw_button_to_rfb_bit(button)
        if bit == 0:
            return
        if action == glfw.PRESS:
            button_mask |= bit
        elif action == glfw.RELEASE:
            button_mask &= ~bit
        if cursor is not None:
            session.input.pointer_event(button_mask, cursor[0], cursor[1])

    # Wheel velocity acceleration. macOS's native scroll amplifies
    # consecutive fast events exponentially; RFB doesn't carry that
    # across the wire, so we recreate the curve here. Without it,
    # RFB scroll feels glacial vs the OS-native scroll the user is
    # used to.
    _wheel_accum = [0.0]
    _wheel_last_t = [0.0]

    def on_scroll(_w, _x, dy):
        if cursor is None or dy == 0:
            return
        now = time.monotonic() * 1000.0  # ms
        dt = now - _wheel_last_t[0]
        _wheel_last_t[0] = now
        if dt < 25:    mult = 10.0
        elif dt < 50:  mult = 6.0
        elif dt < 100: mult = 3.5
        elif dt < 200: mult = 2.0
        elif dt < 350: mult = 1.4
        else:          mult = 1.0
        # glfw's dy is in wheel ticks already (1.0 per notch on a
        # discrete wheel; fractional on trackpads). `scroll_event`
        # interprets dy<0 as scroll-up.
        _wheel_accum[0] += -dy * mult
        ticks = int(_wheel_accum[0])
        if ticks == 0:
            ticks = -1 if dy > 0 else 1
            _wheel_accum[0] = 0.0
        else:
            _wheel_accum[0] -= ticks
        ticks = max(-50, min(50, ticks))
        session.input.scroll_event(cursor[0], cursor[1], 0, ticks)

    # Ctrl→Cmd remap: rewrite local Ctrl into the Mac's Cmd modifier
    # so Ctrl+C / Ctrl+V "feel right". Off by default now — the OS
    # keyboard grab below routes the actual Win/Super key to iss
    # without WM interception, so users get the Mac's native Cmd by
    # pressing Win/Super directly. The remap conflicted with bare-Ctrl
    # shortcuts (Ctrl+W expecting word-delete became Cmd+W close-tab,
    # etc.). Opt back in with ISS_CTRL_AS_CMD=1 if you prefer the
    # rewrite (e.g. on a Mac without keyboard-grab support).
    ctrl_as_cmd = os.environ.get("ISS_CTRL_AS_CMD", "0") == "1"
    if ctrl_as_cmd:
        log.info("input: Ctrl→Cmd remap on (set ISS_CTRL_AS_CMD=0 to disable)")

    def on_key(_w, key, _scancode, action, mods):
        if action not in (glfw.PRESS, glfw.RELEASE, glfw.REPEAT):
            return
        # Force-IDR (formerly local F12) now lives in the TUI's session
        # screen as the `f` keybinding -- it sends an `{"action":"fir"}`
        # command over the control socket which the session's command
        # handler translates into the same `request_fir(None)` call.
        # The desktop window stays a pure viewer; all keys forward.
        # Ctrl→Cmd: substitute Super for the Control modifier-key
        # events so when iss tells the Mac "Control_L is down" it
        # actually says "Super_L is down" → macOS treats it as Cmd.
        if ctrl_as_cmd:
            if key == glfw.KEY_LEFT_CONTROL:
                session.input.key_event(action != glfw.RELEASE, _KEYSYM_SUPER_L)
                return
            if key == glfw.KEY_RIGHT_CONTROL:
                session.input.key_event(action != glfw.RELEASE, _KEYSYM_SUPER_R)
                return
        keysym = GLFW_KEY_TO_X11.get(key, 0)
        if keysym == 0:
            # Printable key. Normally `on_char` handles these so the
            # user's keyboard layout (Dvorak / AZERTY / Shift) is
            # respected. But `on_char` does NOT fire when a modifier
            # is held (e.g. Ctrl+C, Cmd+V), so we'd drop the letter
            # and only forward the modifier press — host sees Ctrl
            # held + Ctrl release, no C in between. Catch the
            # modifier-held case here and synthesize a keysym for the
            # raw ASCII letter / digit. X11 keysyms for A–Z and 0–9
            # are just their ASCII codepoints.
            held = mods & (glfw.MOD_CONTROL | glfw.MOD_SUPER | glfw.MOD_ALT)
            if held and glfw.KEY_A <= key <= glfw.KEY_Z:
                keysym = ord("a") + (key - glfw.KEY_A)
            elif held and glfw.KEY_0 <= key <= glfw.KEY_9:
                keysym = ord("0") + (key - glfw.KEY_0)
            elif held and key == glfw.KEY_SPACE:
                keysym = ord(" ")
            else:
                return
        session.input.key_event(action != glfw.RELEASE, keysym)

    def on_char(_w, codepoint):
        session.input.key_event(True, codepoint)
        session.input.key_event(False, codepoint)

    glfw.set_cursor_pos_callback(glfw_window, on_cursor_pos)
    glfw.set_mouse_button_callback(glfw_window, on_mouse_button)
    glfw.set_scroll_callback(glfw_window, on_scroll)
    glfw.set_key_callback(glfw_window, on_key)
    glfw.set_char_callback(glfw_window, on_char)

    # Grab the keyboard at the OS level so Win/Super combos and
    # Alt+Tab go to the remote Mac instead of the local desktop. On by
    # default; opt out with ISS_GRAB_KEYBOARD=0 (e.g. when debugging
    # local shortcuts while iss is foreground).
    if os.environ.get("ISS_GRAB_KEYBOARD", "1") != "0":
        from .keyboard_grab import make_grab
        # The Win32 backend needs to deliver intercepted keys directly
        # to iss's input pipeline (it bypasses GLFW for hooked keys —
        # see keyboard_grab.py for why). Hand it a sender bound to the
        # session's InputController so Win+C/V/etc. reach the Mac.
        kbd_grab = make_grab(
            glfw_window,
            key_sender=lambda is_down, ks: session.input.key_event(is_down, ks),
        )
        # Initial enable if window already focused at startup.
        if glfw.get_window_attrib(glfw_window, glfw.FOCUSED):
            kbd_grab.enable()

        def on_window_focus(_w, focused):
            if focused:
                kbd_grab.enable()
            else:
                kbd_grab.disable()

        glfw.set_window_focus_callback(glfw_window, on_window_focus)
    else:
        kbd_grab = None

    # ── render loop ────────────────────────────────────────────────────
    first_seen: list[bool] = [False] * num_tiles
    slot_h_resolved = False
    deadline = time.monotonic() + auto_quit_secs if auto_quit_secs > 0 else float("inf")

    # rendercanvas presents only from inside the draw callback (it owns
    # the swap chain). We do the GPU work here and trigger via
    # `force_draw()` from the main loop so render is paced to tile
    # arrivals, not the rendercanvas internal animation timer.
    # Set by the draw callback when wgpu reports the GPU device is
    # lost (Windows DXGI TDR, driver reset, display mode change). Main
    # loop checks this and shuts down cleanly instead of letting the
    # underlying Rust panic kill the process with an unhelpful trace.
    _device_lost: dict[str, bool] = {"v": False}

    def draw_callback():
        try:
            target = surface_ctx.get_current_texture()
            renderer.draw(
                target.create_view(),
                (0.0, 0.0, float(target.width), float(target.height)),
            )
        except Exception as e:
            msg = str(e)
            if "device is lost" in msg.lower() or "Validation Error" in msg:
                if not _device_lost["v"]:
                    log.error(
                        "GPU device lost (likely Windows DXGI TDR / driver "
                        "reset / display change) — closing viewer cleanly. "
                        "Restart iss to recover. wgpu: %s", msg[:200],
                    )
                    _device_lost["v"] = True
                return
            raise

    window.request_draw(draw_callback)

    # Currently-applied cursor + GLFW handle. We hold the GLFW cursor
    # alive in a ref because glfw.set_cursor doesn't take ownership;
    # destroying the cursor while it's set crashes the X server.
    _cursor_state: dict[str, "Optional[object]"] = {
        "img": None, "handle": None,
    }

    import ctypes

    def _scale_cursor_rgba(rgba: bytes, w: int, h: int,
                           hx: int, hy: int, scale: int,
                           ) -> tuple[bytes, int, int, int, int]:
        """Integer-scale RGBA cursor pixmap (nearest-neighbor). Integer
        scale keeps edges crisp; non-integer would blur small bitmap
        cursors. Inner loop uses row-replication + bytes slicing
        instead of a per-pixel Python loop."""
        if scale <= 1:
            return rgba, w, h, hx, hy
        nw, nh = w * scale, h * scale
        out = bytearray(nw * nh * 4)
        row_bytes = nw * 4
        for sy in range(h):
            scaled_row = bytearray(row_bytes)
            for sx in range(w):
                src_off = (sy * w + sx) * 4
                pix = rgba[src_off:src_off + 4]
                for k in range(scale):
                    o = (sx * scale + k) * 4
                    scaled_row[o:o + 4] = pix
            for k in range(scale):
                base = (sy * scale + k) * row_bytes
                out[base:base + row_bytes] = scaled_row
        return bytes(out), nw, nh, hx * scale, hy * scale

    # Minimum on-screen cursor size in device pixels (Linux/Windows
    # only). Some host cursors are very small in their native pixmap
    # (the I-beam ships at 9x18 from Apple's daemon) and on a
    # non-HiDPI Linux desktop that would render as 9x18 device pixels
    # — practically invisible. Bump small cursors up to this floor
    # via integer scaling so every shape is recognisable.
    _MIN_CURSOR_DEVICE_PX = 24

    def _build_glfw_cursor(img):
        """Translate our `_CursorImage` into a GLFW cursor handle sized
        to match the local OS cursor.

        GLFW's cursor image dimensions are interpreted differently per
        platform:
          - macOS: dimensions are LOGICAL POINTS (NSCursor expects
            point-sized images and the OS upscales to device pixels).
            The daemon already ships at host-pixel size which matches
            point-level expectations on Retina-to-Retina, so we pass
            through unmodified.
          - Linux X11 / Windows: dimensions are DEVICE PIXELS. We
            integer-scale by max(content_scale, ceil(MIN/longer_side))
            so HiDPI displays match the local cursor's footprint AND
            tiny cursors (like the 9×18 I-beam) get bumped up enough
            to be visible. Without the floor, the I-beam was effectively
            invisible on non-HiDPI Linux."""
        if _sys.platform == "darwin":
            scale = 1
        else:
            try:
                sx, _sy = glfw.get_window_content_scale(glfw_window)
                dpi_scale = max(1, int(round(sx)))
            except Exception:
                dpi_scale = 1
            long_side = max(img.width, img.height)
            min_scale = -(-_MIN_CURSOR_DEVICE_PX // long_side)  # ceil div
            scale = max(dpi_scale, min_scale)
        rgba, w, h, hx, hy = _scale_cursor_rgba(
            img.rgba, img.width, img.height,
            img.hotspot_x, img.hotspot_y, scale,
        )
        n_bytes = len(rgba)
        arr_type = ctypes.c_ubyte * n_bytes
        buf = arr_type.from_buffer_copy(rgba)
        gimg = glfw._GLFWimage()
        gimg.width = w
        gimg.height = h
        gimg.pixels = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))
        gimg.pixels_array = buf
        handle = glfw._glfw.glfwCreateCursor(
            ctypes.pointer(gimg), hx, hy,
        )
        return handle, gimg

    # GLFW cursor handle cache, keyed by id(_CursorImage). The host's
    # daemon sends rapid alternating cache hits at UI element
    # boundaries (20-50 ms swap intervals). On X11 each set_cursor
    # with a fresh handle does 4 X-server round trips (create + load
    # + define + free old), producing visible tearing under rapid
    # alternation. Caching the handle collapses set_cursor to a
    # single XDefineCursor — fast enough that rapid alternations
    # don't visibly tear.
    #
    # Each entry stores (img, handle, gimg). The img reference
    # both keeps the _CursorImage alive (preventing id() reuse) AND
    # acts as an identity guard: if the proxy-side cursor cache evicts
    # an entry and Python recycles its id() for a new _CursorImage, we
    # detect that mismatch on lookup and rebuild the handle.
    _glfw_cursor_handle_cache: dict[int, "tuple[object, object, object]"] = {}

    def _set_cursor_now(img):
        """Apply a cursor immediately — direct path used on platforms
        where GLFW is thread-safe enough for set_cursor off the main
        thread (Windows + Linux/X11). Avoids the per-frame delay of
        the render-loop deferral, which was losing intermediate
        cursor frames when shape changed faster than video FPS.

        `img is None` reverts to the local OS default cursor."""
        if img is _cursor_state["img"]:
            return
        try:
            if img is None:
                glfw.set_cursor(glfw_window, None)
                handle = None
            else:
                cached = _glfw_cursor_handle_cache.get(id(img))
                if cached is None or cached[0] is not img:
                    handle, gimg = _build_glfw_cursor(img)
                    _glfw_cursor_handle_cache[id(img)] = (img, handle, gimg)
                else:
                    handle = cached[1]
                glfw.set_cursor(glfw_window, handle)
            _cursor_state["img"] = img
            _cursor_state["handle"] = handle
            # Don't destroy old handles — the cache owns them and the
            # daemon may reference any of them again at any moment via
            # cache hit. Total memory is bounded by the proxy-side
            # `_cursor_cache` cap (64 entries) since handles are keyed
            # by id() of those entries; once they get GC'd, the next
            # _set_cursor_now never finds them by id() and we'd build
            # fresh anyway. Worst case = ~64 small X cursors.
        except Exception as e:
            log.debug("cursor apply failed: %s", e)

    def _apply_pending_cursor():
        """macOS-only deferred path: pull from the pending slot on the
        render thread. On Linux/Windows the proxy thread calls
        `_set_cursor_now` directly so we skip this slot entirely.

        `_pending_cursor["img"]` may legitimately be None (revert to
        OS default) — distinguished from "no pending update" by a
        sentinel object."""
        with _cursor_lock:
            img = _pending_cursor["img"]
            _pending_cursor["img"] = _NO_PENDING
        if img is _NO_PENDING:
            return
        _set_cursor_now(img)

    while time.monotonic() < deadline:
        glfw.poll_events()
        _apply_pending_cursor()
        if window.get_closed() or glfw.window_should_close(glfw_window):
            break
        if _device_lost["v"]:
            break
        if not session.is_connected:
            log.error("connection lost — closing viewer")
            break
        session.wait_for_fresh_tile(timeout=_FRESH_TILE_WAIT_S)

        # Drain fresh decoded frames + upload.
        any_fresh = False
        for ti in range(num_tiles):
            tf = session.get_frame(ti)
            if tf is None:
                continue
            first_seen[ti] = True
            # Refine slot_h on the first tile — encoder's CTU-padded
            # picture height (typically 4 rows taller than canvas_h//
            # num_tiles). See comment at session-ready above.
            if not slot_h_resolved:
                slot_h = tf.height
                slot_h_resolved = True
            renderer.upload_tile(ti, tf, slot_h)
            any_fresh = True

        # Present after a fresh upload. No fresh content = skip the
        # draw, swap chain holds the previous good frame.
        if any_fresh and any(first_seen):
            window.force_draw()

    log.info("desktop frontend closing")
    if kbd_grab is not None:
        kbd_grab.disable()
    if audio_sink is not None:
        audio_sink.stop()
    session.close()
    # Explicit window close — triggers orderly wgpu cleanup. Without
    # this we'd rely on __del__ when locals fall out of scope, and the
    # interpreter's GC ordering can deadlock the wgpu native poller
    # (observed: main thread stuck in wgpu _release proxy_func at exit).
    try:
        window.close()
    except Exception as e:
        log.debug("window.close() failed: %s", e)
    # Fail-safe: bypass Python's interpreter-shutdown phase. We've
    # already released the network sockets, audio device, and X11
    # grab; anything left to clean up belongs to the kernel and gets
    # reclaimed by the process exit. Avoids the rare wgpu-__del__
    # deadlock at scope unwind that leaves the process hung.
    os._exit(0)


__all__ = ["run"]
