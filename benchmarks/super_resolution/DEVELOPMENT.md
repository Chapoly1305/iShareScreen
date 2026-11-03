# Training and readability experiment code

## Entry points

| Directory | Responsibility |
| --- | --- |
| `text_corpus/` | Source collection, rendering, grouped splits and earlier text training |
| `codescan_campaign/` | Frozen CodeSCAN codec pairs, seven-model training, host guards and monitoring |
| `low_bandwidth_e2e/` | Protocol freezing, source overlap audits, link and quality evaluation |
| `mac_readability/` | Native Mac capture/encoding, source-detail transport, SR and independent confirmation |
| `experiments/` (ignored) | Immutable protocols, data indexes, checkpoints, measurements and reports |

See [the Mac experiment runbook](mac_readability/README.md) for the completed
independent confirmation and its limitations. These scripts are research tools;
the source-detail transport is not integrated into the normal Screen Sharing
client. Training services are historical campaign configurations, not services
to enable indiscriminately on a new machine. Inspect current GPU ownership and
the applicable `AGENTS.md` before any new run.

`datasets/split_codescan.py` is source code and belongs in Git. Dataset contents,
original DVR recordings, model assets, generated videos and captured credentials
do not. Local exclude rules may hide the splitter even though the repository
ignore rules explicitly allow it.

## Regression checks

Activate the project environment with `source dev-environment.sh`. The CPU test
suite must be allowed to create loopback TCP/UDP sockets:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest --disable-warnings -ra
python -m compileall -q src benchmarks/super_resolution tests datasets/split_codescan.py
git diff --check
```

The November 3 cleanup used Ruff 0.17.3 (`ruff format` and the error checks
`E9,F63,F7,F82`) for the changed Python files. The optional browser, GPU, Core ML,
macOS, Windows and official-model tests report explicit skips when their
environment is unavailable. A passing CPU suite does not validate those paths
on real hardware.

On 2026-11-03 the cleaned tree passed 664 tests, with 57 explicit skips and two
warnings. Compilation, whitespace checks and the selected Ruff error checks
passed. Detailed local receipts are in
`experiments/code-review-20261103/VALIDATION.json`, `pytest.log` and `ruff.log`.

Core regression coverage includes immediate exposure of new decoded content,
discarding stale SR results even across upload boundaries, identical-frame
reuse, native 420/444 plane geometry, opt-in bitrate offers, component payload
integrity, bounded framing and cancellation of obsolete source generations.

## Frozen experiments versus cleaned source

Completed experiments record source hashes, not just model and input hashes.
Before formatting, source copies and SHA256 receipts were preserved locally in
`experiments/code-review-20261103/original-source/`. Frozen primary stream
versions also remain in the experiment's `evidence/stream-v*/` directories.
The independent confirmation's 232 bound evidence files remain unchanged.

Formatting changes the source hash even when the Python syntax tree is
identical. Use the recorded original source to reproduce a frozen run; use a
new output directory and protocol binding for runs of cleaned code. Do not
rewrite old fingerprints to make a resume check pass. Four training runners
now release captured model/optimizer references by assigning `None` after the
last checkpoint, avoiding deleted closure variables; their historical runs and
checkpoints are preserved. These changes do not start or resume training.

## Visual review

Generate a separate preview from the actual receiver's saved screenshots:

```bash
python -m benchmarks.super_resolution.mac_readability.visual_review \
  --root benchmarks/super_resolution/experiments/mac-readability-v2-20261102/confirmation-v1 \
  --output benchmarks/super_resolution/experiments/your-new-visual-review
```

The output contains `index.html`, native-pixel PNG crops, a 32-second MP4 and a
hash-bound source manifest. It shows fixed code/table/English/Chinese examples
at receiver ages 0.5, 1, 2 and 2.8 seconds. Each saved screenshot is held for two
seconds for inspection, so this is **not continuous recording or real-time
playback**. It cannot measure flicker, transition smoothness or end-to-end delay.
The left side is decoded video without SR, not the trained EDSR comparator.
Use the confirmation report for trained/official/source-only comparisons.

Inspect the original PNGs at 100% to avoid player scaling and video compression.
For human testing, record reading errors and preference separately from OCR;
include cases with no improvement and regressions. A new continuous recording
must capture every displayed transition and report recording overhead before it
can support temporal-quality conclusions.
