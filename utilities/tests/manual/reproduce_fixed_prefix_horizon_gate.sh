#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output_dir="${1:-${project_root}/opi/preanswer/fixed_prefix_horizon_20260729T082827Z}"

export HF_HOME="${project_root}/shared/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="/tmp/mpl-fixed-prefix-horizon-gate"

mkdir -p "${MPLCONFIGDIR}"
cd "${project_root}"

"${project_root}/venv/bin/python" \
  shared/utilities/tests/manual/fixed_prefix_horizon_inventory.py \
  --output-dir "${output_dir}"
"${project_root}/venv/bin/python" \
  shared/utilities/tests/manual/fixed_prefix_horizon_finalize.py \
  --output-dir "${output_dir}"
