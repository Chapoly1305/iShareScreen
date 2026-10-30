# iShareScreen development and GPU resources

## Primary environment

- Develop, download datasets, process data, train, and run Linux benchmarks in
  `/home/jchen73/iShareScreen` on H100 (`jchen73@10.193.124.20`).
- Use the project-local `.venv`. Run `source dev-environment.sh` to activate it
  and configure project tool/cache paths. Keep Python dependencies in this venv.
- Preserve original datasets, frozen experiments, and unrelated working changes.
  Record new experiments separately with their configuration and evidence.
- macOS client, VideoToolbox, and Core ML playback checks require a Mac;
  H100 inference performance does not establish client-device performance.

## User-authorized GPU allocation

The user explicitly supplied these rules on 2026-10-21. They apply to future
project Goals, training, evaluation, and remote GPU work.

| Machine | Connection | GPU permission |
| --- | --- | --- |
| A5000 | `junming@10.193.124.99:15360` | Use physical GPU indices **1 and 3 only**. |
| H100 | `jchen73@10.193.124.20` | Inspect current usage first. Do not compete with another user's active GPU workload. |
| Hopper | `jchen73@hopper.orc.gmu.edu` | May request **up to 8 A100 GPUs**, subject to the scheduler's actual allocation. |
| lab | `chen@10.193.124.90` | The user permits unrestricted use of its RTX 4090 for project GPU work. |
| Other RTX 4090 | `alex@10.193.124.46:2022` | The user permits unrestricted use of this GPU for project GPU work. |

- Both the Mac and H100 SSH configurations provide `lab`, `a5000`, `hopper`
  (also its full domain), and `rtx4090` (also `other4090`). `lab` is the
  `chen@10.193.124.90` machine; `rtx4090` is `alex@10.193.124.46:2022`.
  All eight source/target combinations passed public-key-only, noninteractive
  `BatchMode=yes` SSH checks on 2026-10-21.
- A5000 device indices refer to `nvidia-smi` physical indices. Set
  `CUDA_VISIBLE_DEVICES=1,3` for jobs using both allowed GPUs; inside that process
  CUDA exposes them as logical devices 0 and 1. Do not use physical GPUs 0 or 2.
- Before using H100, inspect GPU memory, utilization, compute processes, and
  process owners. Zero instantaneous utilization alone does not mean a GPU is
  available. Do not stop or preempt another user's task to make room. If GPUs
  are occupied, use another authorized machine or wait while doing useful CPU
  work. Recheck immediately before launching.
- Request Hopper GPUs through Slurm and use only the assigned resources. An
  authorization for 8 A100s is not evidence that an allocation is currently held.
- No new GPU-usage confirmation is required within these permissions. Keep
  commands, host/GPU identities, jobs, logs, and checkpoints traceable.
- The other 4090 address and port were explicitly supplied by the user. Verify
  the actual hostname/GPU when connecting; do not count two addresses for the
  same physical machine as two GPUs.
- Keep each machine's private SSH key on that machine. Configure key-based
  access by authorizing public keys; preserve host-key verification.

## Other 4090 driver validation

- On 2026-10-21, a user-performed reboot resolved the old loaded NVIDIA kernel
  module versus updated userspace-library mismatch on `rtx4090`.
- The verified state was kernel `7.0.0-38-generic`, NVIDIA driver `595.99.02`,
  PyTorch `2.6.0+cu124`, CUDA runtime `12.4`, and an RTX 4090 with 24 GiB VRAM.
  Actual CUDA matrix multiplication, finite backward gradients, and an Adam
  parameter update passed using system driver libraries, without an
  `LD_LIBRARY_PATH` override.
- Treat these versions as a dated validation record. Recheck the current state
  when needed; do not blindly reuse the historical `595.91.07` library override
  from earlier benchmarks against the updated driver.

## Evaluation chroma convention

- On 2026-10-29 the user selected **4:2:0** for future video evaluations for
  broader applicability. Default to `yuv420p`; do not add 4:4:4 to new sweeps
  unless the user explicitly requests a chroma comparison.
- Keep completed 4:4:4 observations and frozen protocols as historical evidence.
  Present 4:2:0 as the primary comparison for the current baseline campaign.
