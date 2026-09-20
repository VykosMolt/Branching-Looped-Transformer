#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output_dir="${1:-${project_root}/opi/preanswer/visible_prefix_hidden_increment_20260729T075247Z}"

export HF_HOME="${project_root}/shared/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="/tmp/mpl-visible-prefix-hidden-increment"

mkdir -p "${MPLCONFIGDIR}"
cd "${project_root}"

test -f "${output_dir}/inventory_report.json"
test -f "${output_dir}/preregistration.sha256"
test -f "${output_dir}/sealed_protocol_manifest.json"

"${project_root}/venv/bin/python" \
  shared/utilities/tests/manual/visible_prefix_hidden_increment_experiment.py \
  --stage audit \
  --output-dir "${output_dir}"
"${project_root}/venv/bin/python" \
  shared/utilities/tests/manual/visible_prefix_hidden_increment_finalize.py \
  --output-dir "${output_dir}"
