#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-/home/smansou2/dist-kge/examples/experiments/fb15k/dim128/complex/complex-fb15k-parallel-random-R-2@1.yaml}"
OUT_ROOT="${2:-/home/smansou2/dist-kge/local/experiments/fb15k-opt-bench}"
ONLY="${ONLY:-}"

SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
VALID_EVERY="${VALID_EVERY:-}"

COMMON_ARGS=(
  "--random_seed.default" "${SEED}"
  "--random_seed.torch" "${SEED}"
  "--random_seed.numpy" "${SEED}"
  "--random_seed.python" "${SEED}"
)
if [[ -n "${MAX_EPOCHS}" ]]; then
  COMMON_ARGS+=("--train.max_epochs" "${MAX_EPOCHS}")
fi
if [[ -n "${VALID_EVERY}" ]]; then
  COMMON_ARGS+=("--valid.every" "${VALID_EVERY}")
fi

mkdir -p "${OUT_ROOT}"

declare -a EXPERIMENTS=(
  "baseline"
  "fused"
  "sample_on_gpu"
  "materialize"
  "stage_local_ids"
  "all"
)

declare -A EXP_ENV
declare -A EXP_ARGS

EXP_ENV["baseline"]="KGE_COMPLEX_FUSED=0"
EXP_ARGS["baseline"]=""

EXP_ENV["fused"]="KGE_COMPLEX_FUSED=1"
EXP_ARGS["fused"]=""

EXP_ENV["sample_on_gpu"]="KGE_COMPLEX_FUSED=0"
EXP_ARGS["sample_on_gpu"]="--job.distributed.sample_on_gpu true"

EXP_ENV["materialize"]="KGE_COMPLEX_FUSED=0"
EXP_ARGS["materialize"]="--job.distributed.materialize_partition_batches true"

EXP_ENV["stage_local_ids"]="KGE_COMPLEX_FUSED=0"
EXP_ARGS["stage_local_ids"]="--job.distributed.stage_local_ids true --job.distributed.map_ids_on_gpu true"

EXP_ENV["all"]="KGE_COMPLEX_FUSED=1"
EXP_ARGS["all"]="--job.distributed.sample_on_gpu true --job.distributed.materialize_partition_batches true --job.distributed.stage_local_ids true --job.distributed.map_ids_on_gpu true"

run_experiment() {
  local name="$1"
  local env_prefix="${EXP_ENV[$name]}"
  local extra_args="${EXP_ARGS[$name]}"
  local folder="${OUT_ROOT}/${name}"

  echo "==> Running ${name}"
  echo "    folder: ${folder}"
  echo "    env: ${env_prefix}"
  echo "    args: ${extra_args}"

  # shellcheck disable=SC2086
  eval ${env_prefix} python -m kge start \
    "${CONFIG_PATH}" \
    --folder "${folder}" \
    "${COMMON_ARGS[@]}" \
    ${extra_args}

  python /home/smansou2/dist-kge/scripts/summarize_kge_run.py "${folder}"
  echo ""
}

for exp in "${EXPERIMENTS[@]}"; do
  if [[ -n "${ONLY}" && ",${ONLY}," != *",${exp},"* ]]; then
    continue
  fi
  run_experiment "${exp}"
done
