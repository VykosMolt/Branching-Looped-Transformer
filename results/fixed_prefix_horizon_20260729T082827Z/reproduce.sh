#!/usr/bin/env bash
set -euo pipefail
artifact_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${artifact_dir}/../../.." && pwd)"
exec bash "${project_root}/utilities/tests/manual/reproduce_fixed_prefix_horizon_gate.sh" "${artifact_dir}"
