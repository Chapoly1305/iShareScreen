"""Both trackpad axes must survive the viewers and the encrypted input writer."""
import ast
from pathlib import Path
import shutil
import struct
import subprocess
from types import SimpleNamespace

from Crypto.Cipher import AES
import pytest

from isharescreen.frontend.desktop import app
from isharescreen.proxy.input import AsyncInputController, InputController
from isharescreen.proxy.protocol.enc1103 import StreamCipher


@pytest.mark.parametrize("controller_type", [InputController, AsyncInputController])
@pytest.mark.parametrize("encrypted", [False, True])
@pytest.mark.parametrize("dx,dy,presses", [
    (-2, 0, [0x20, 0x20]),
    (2, 0, [0x40, 0x40]),
    (0, -2, [0x08, 0x08]),
    (0, 2, [0x10, 0x10]),
    (-1, -1, [0x08, 0x20]),
    (1, -1, [0x08, 0x40]),
    (-1, 1, [0x10, 0x20]),
    (1, 1, [0x10, 0x40]),
    (0, 0, []),
])
def test_scroll_wire_events(controller_type, encrypted, dx, dy, presses):
    chunks = []
    cipher = StreamCipher(bytes(36), ecb_key=bytes(16)) if encrypted else None
    controller = controller_type(
        SimpleNamespace(sendall=chunks.append), cipher,
        server_width=1920, server_height=1080,
    )
    try:
        controller.scroll_event(100, 200, dx, dy)
    finally:
        controller.close()  # Also drains the asynchronous writer.
    wire = b"".join(chunks)
    expected_masks = [mask for press in presses for mask in (press, 0)]
    if encrypted:
        decoder = StreamCipher(bytes(36), ecb_key=bytes(16))
        records, consumed = decoder.decrypt_stream(wire)
        assert consumed == len(wire)
        events = []
        aes = AES.new(decoder.cbc_key, AES.MODE_ECB)
        for record in records:
            assert record[:2] == b"\x10\x00"
            payload = aes.decrypt(record[2:])
            assert payload[10] == 0xff
            events.append(struct.unpack(">BHH", payload[11:]))
    else:
        events = []
        for kind, buttons, x, y in struct.iter_unpack(">BBHH", wire):
            assert kind == 5
            events.append((buttons, x, y))
    assert events == [(mask, 100, 200) for mask in expected_masks]
    assert controller.tx_pkts == len(expected_masks)


@pytest.mark.parametrize("callback", ["on_scroll", "on_scroll2"])
@pytest.mark.parametrize("dx,dy,expected", [
    (1, 0, (-12, 0)), (-1, 0, (12, 0)),
    (0, 1, (0, -12)), (0, -1, (0, 12)),
    (1, -1, (-12, 12)), (0, 0, None),
])
def test_desktop_callbacks_forward_both_axes(callback, dx, dy, expected):
    # Run the actual registered closures without creating GPU/GLFW windows.
    tree = ast.parse(Path(app.__file__).read_text())
    definition = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == callback)
    sent = []
    state = SimpleNamespace(cursor=(100, 200), wheel_accum=[0.0, 0.0], wheel_last=[0.0])
    scope = dict(
        cursor=state.cursor, _wheel_accum=state.wheel_accum,
        _wheel_last_t=state.wheel_last, st=state,
        time=SimpleNamespace(monotonic=lambda: 1.0),
        _scroll_ticks=app._scroll_ticks,
        session=SimpleNamespace(input=SimpleNamespace(
            scroll_event=lambda *args: sent.append(args))),
    )
    exec(compile(ast.Module(body=[definition], type_ignores=[]), app.__file__, "exec"), scope)
    scope[callback](None, dx, dy)
    assert sent == ([] if expected is None else [(100, 200, *expected)])
    state.cursor = scope["cursor"] = None
    scope[callback](None, 1, 1)
    assert len(sent) == (expected is not None)


def test_desktop_preserves_small_movements_independently():
    accum = [0.0, 0.0]
    assert app._scroll_ticks(accum, 0.04, 0, 500) == (0, 0)
    assert app._scroll_ticks(accum, 0, -0.04, 500) == (0, 0)
    assert app._scroll_ticks(accum, 0.04, -0.04, 500) == (0, 0)
    assert app._scroll_ticks(accum, 0.04, -0.04, 500) == (-1, 1)
    assert accum == pytest.approx([-0.44, 0.44])


def test_desktop_caps_each_axis_without_draining_the_idle_axis():
    accum = [0.0, 0.0]
    assert app._scroll_ticks(accum, 10, -10, 10) == (-50, 50)
    assert accum == [-150, 150]
    assert app._scroll_ticks(accum, 0, -0.1, 500) == (0, 50)
    assert accum == pytest.approx([-150, 101.2])


def test_browser_wheel_events():
    node = shutil.which("node")
    if not node:
        pytest.skip("node unavailable")
    viewer = (Path(app.__file__).parents[1] / "wt/static/viewer.js").read_text()
    handler = viewer[viewer.index("  const wheelAccum ="):
                     viewer.index('  document.addEventListener("keydown"')]
    script = """
const assert = require('node:assert/strict');
function makeWheel() {
  let now = 1000, callback;
  const sent = [];
  const performance = {now: () => now};
  const canvas = {addEventListener: (type, fn) => {
    assert.equal(type, 'wheel'); callback = fn;
  }};
  const inWindowToCanvas = e => e.outside ? null : [100, 200];
  const send = e => sent.push(e);
""" + handler + """
  return {sent, scroll(dx, dy, mode = 0, outside = false, elapsed = 400) {
    now += elapsed;
    let prevented = false;
    callback({deltaX: dx, deltaY: dy, deltaMode: mode, outside,
              preventDefault: () => {prevented = true;}});
    assert.ok(prevented);
  }};
}
for (const [dx, dy, mode, tx, ty] of [
  [-96, 0, 0, -12, 0], [96, 0, 0, 12, 0],
  [0, -96, 0, 0, -12], [0, 96, 0, 0, 12],
  [96, -96, 0, 12, -12], [-96, 96, 0, -12, 12],
  [1, -1, 1, 2, -2], [-1, 1, 2, -50, 50],
  [0.1, 0, 0, 1, 0], [0, -0.1, 0, 0, -1],
  [10000, -10000, 0, 50, -50],
]) {
  const wheel = makeWheel();
  wheel.scroll(dx, dy, mode);
  assert.deepEqual(wheel.sent, [{type: 'scroll', x: 100, y: 200, dx: tx, dy: ty}]);
}
const wheel = makeWheel();
wheel.scroll(0, 0);
wheel.scroll(96, 0, 0, true);
assert.deepEqual(wheel.sent, []);
// Fractional remainders belong to their original axis.
wheel.scroll(12, 0);
wheel.scroll(0, 12);
wheel.scroll(12, 0);
wheel.scroll(0, 12);
assert.deepEqual(wheel.sent.map(e => [e.dx, e.dy]), [[1, 0], [0, 1], [2, 0], [0, 2]]);
// Preserve the existing velocity acceleration, on either axis.
wheel.scroll(8, -8, 0, false, 10);
assert.deepEqual(wheel.sent.at(-1), {type: 'scroll', x: 100, y: 200, dx: 10, dy: -10});
"""
    result = subprocess.run([node, "-"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
