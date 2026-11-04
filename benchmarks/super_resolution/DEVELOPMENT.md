# Research code and result maintenance

## Entry points

| Directory | Responsibility |
| --- | --- |
| `text_corpus/` | Source collection, rendering, grouped splits and earlier text training |
| `codescan_campaign/` | Frozen CodeSCAN codec pairs, seven-model training, host guards and monitoring |
| `low_bandwidth_e2e/` | Protocol freezing, source overlap audits, link and quality evaluation |
| `mac_readability/` | Native Mac capture/encoding, source-detail transport, SR and independent confirmation |
| `experiments/` (ignored) | Immutable protocols, data indexes, checkpoints, measurements and reports |

The final result uses `codescan_campaign/` for training and `codescan_test.py`
for independent testing. See [the current protocol](TRAINING.md) and
[the canonical report](../../results/README.md). The Mac source-detail runbook
is historical; supplement transport is excluded from the final project.
Training services are completed-campaign configurations, not services to enable
on a new machine. Inspect GPU ownership and `AGENTS.md` before new runs.

`datasets/split_codescan.py` is source code and belongs in Git. Dataset contents,
original DVR recordings, model assets, generated videos and captured credentials
do not. Reviewed static presentation assets and measurement evidence in `results/`
are the intentional exception for published research outputs. Local exclude rules may hide the splitter even though the repository
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

The recorded pre-presentation cleanup on 2026-11-03 passed 664 tests, with 57 explicit skips and two
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
The historical source-detail confirmation's 232 bound evidence files remain unchanged.

Formatting changes the source hash even when the Python syntax tree is
identical. Use the recorded original source to reproduce a frozen run; use a
new output directory and protocol binding for runs of cleaned code. Do not
rewrite old fingerprints to make a resume check pass. Four training runners
now release captured model/optimizer references by assigning `None` after the
last checkpoint, avoiding deleted closure variables; their historical runs and
checkpoints are preserved. These changes do not start or resume training.

## Published presentation

[`../../results/`](../../results/README.md) is the only maintained results
presentation. It is independent of the local experiment archive and can be
served unchanged by a static file host. The primary table covers 1,200 held-out CodeSCAN test samples. The supplementary
Mac validation gallery preserves seven scenes,
49 pretrained/fine-tuned pairs and the no-SR controls. Keep limitations and
failure cases visible when editing it.

Historical experiment outputs have moved outside the repository. The ignored
`experiments/` compatibility link preserves paths used by frozen manifests and
local training tooling. Old source-detail experiments are research history;
their gains are not included in the published receiver-only SR results.
