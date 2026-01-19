# Nsight Systems (low overhead)

The `dist-kgeArc/*/config.yaml` and `dist-kgeArc/*/worker-*/config.yaml` files are *outputs* (snapshots) created inside a run folder. Editing them after the run starts does not change the configuration used by that run.

## Recommended config

Use the dedicated experiment config:

`examples/experiments/wikidata5m/dim128/complex/complex-wikidata5m-parallel-stratification-CARL-2@1-nsys-lowoverhead.yaml`

It enables NVTX ranges while disabling high-overhead timing options:

- `train.profile_nvtx: true`
- `train.profile_interval_batches: 0`
- `train.profile_cuda_events: false`
- `train.sync_cuda_timing: false`

## CLI overrides to avoid

If you pass these on the command line, they will override the YAML:

- `--train.profile_interval_batches ...`
- `--train.profile_cuda_events true`
- `--train.sync_cuda_timing true`

For NVTX-only tracing, omit them (or set them to `0/false`).
