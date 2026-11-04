# Local EDSR research archive

The dated experiment directories contain frozen measurements, engine plans,
Nsight traces, compiler outputs and downloaded dependencies. They remain local
and are excluded from Git; do not delete or rewrite their manifests.

The maintained implementation is
[`src/isharescreen/enhancement/cuda_native.py`](../src/isharescreen/enhancement/cuda_native.py).
Use the independent [build and validation entry points](../benchmarks/super_resolution/edsr_cuda/README.md).
The [curated report](../docs/analysis/edsr-cuda/README.md) records the accepted
configuration, measurements and limitations without bundling binary captures.

[`edsr-source-archive-20261104/`](edsr-source-archive-20261104/README.md)
preserves the original exploratory benchmark sources byte for byte. These
include unsuccessful variants and are not imported by the maintained runtime.
Historical commands that refer to their former paths need those sources
restored into a separate checkout before execution.
