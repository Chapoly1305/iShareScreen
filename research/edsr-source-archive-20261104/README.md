# Frozen exploratory sources

These 57 files were moved from `benchmarks/super_resolution/` without changing
their bytes. `MANIFEST.json` maps each original path to its archived source,
size and SHA256. They preserve the profiling and engine-building investigation,
including rejected variants; their presence is not an endorsement of a result.

The maintained runtime has no imports from this directory. Its build and
validation commands are in
[`benchmarks/super_resolution/edsr_cuda/`](../../benchmarks/super_resolution/edsr_cuda/README.md).
Raw engines, datasets and traces remain in their existing local archives.

Verify the source archive:

```bash
python research/edsr-source-archive-20261104/restore.py --verify-only
```

To replay historical commands, restore the original source paths into a
**separate project checkout** that also has the recorded inputs and optional
dependencies. Existing files with different content are rejected:

```bash
python research/edsr-source-archive-20261104/restore.py --destination /path/to/separate-checkout
```

Historical scripts may contain dated host paths, versions, hardware guards and
experiment-specific assumptions. Read their recorded configuration before use;
follow the project's GPU allocation rules.
