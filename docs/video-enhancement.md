# Video enhancement

The native desktop viewer supports optional ×2 upscaling after decoding. The
default is **Original (native scaling)**. After connecting, open Diagnostics →
Video enhancement → Display model to select an installed model or return to
Original. Browser playback is unchanged.

## Installation

For a source checkout, use Python 3.10 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[enhancement]'
.venv/bin/python -m benchmarks.super_resolution.setup_models
```

On NVIDIA hosts, install the appropriate PyTorch CUDA build for the host before
installing the enhancement dependencies. The pinned benchmark environment is
separate; see [the evaluation tools](../benchmarks/super_resolution/README.md).

Setup downloads pinned official source checkouts and pretrained checkpoints. It
checks source revisions, rejects tracked vendor edits, and verifies checkpoint
SHA256 hashes. Inference requires exact tensor keys/shapes. There is no automatic
download during playback and no random-weight fallback. Missing models remain
unavailable in the dropdown. EDSR-baseline, IMDN and SAFMN have pinned ×2 weights;
ECBSR remains unavailable until an eligible original checkpoint is supplied.

Model data defaults to `~/.cache/isharescreen/super_resolution/`, independently of
the installation directory. Set `ISS_SR_HOME` before both setup and playback to
use another location. Setup also accepts `--home`, but playback must use the
matching `ISS_SR_HOME`. Sources are stored under `vendor/` and weights under
`checkpoints/`; neither is bundled in the viewer wheel. Refer to each provider's
terms before redistributing model assets.

`ISS_SR_DEVICE=auto` chooses CUDA, then MPS, then CPU. Set it to `cuda`, `mps` or
`cpu` to choose explicitly. CPU inference may be too slow for interactive use.

## Runtime boundaries

- `src/isharescreen/enhancement/models.py` owns the shared inference adapter.
- `live.py` assembles decoded tiles and runs inference outside the control/render
  thread, with at most one pending canvas and generation-bound results.
- The native renderer uploads RGB reconstructions while keeping cursor composition
  separate. Switching models, resizing or returning to Original invalidates stale
  results. A model/upload failure restores the original rendering path.
- Offline tools import the same adapter; they do not duplicate model definitions.

Enhancement is opt-in and does not change transport negotiation or source stream
resolution. Original mode needs neither PyTorch nor OpenCV. Offline inference
timings are not input-to-display latency or sustained viewer FPS.

## Maintenance and verification

Keep runtime behavior in `src/`, reproducible evaluation tools in `benchmarks/`,
and user-facing documentation here. Do not add datasets, checkpoints, machine
paths or campaign-specific schedulers to the runtime package.

```sh
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
ISS_TEST_GPU=1 .venv/bin/python -m pytest tests/test_enhancement_gpu.py
.venv/bin/python -m pip wheel --no-deps .
```

Tests cover optional-dependency bypass, bounded work queues, stale-result rejection,
control dispatch and geometry. GPU readback tests are opt-in. Benchmark tests skip
when their optional dependencies are absent; CUDA parity tests need a CUDA device.
Recorded-video checks do not establish live Screen Sharing or physical-device
performance.
