#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output_dir="${1:-${project_root}/opi/preanswer/visible_prefix_hidden_increment_20260729T075247Z}"

export HF_HOME="${project_root}/shared/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "${project_root}"
"${project_root}/venv/bin/python" \
  shared/utilities/tests/manual/visible_prefix_hidden_increment_inventory.py \
  --output-dir "${output_dir}"
"${project_root}/venv/bin/python" \
  shared/utilities/tests/manual/visible_prefix_hidden_increment_experiment.py \
  --stage audit \
  --output-dir "${output_dir}"
