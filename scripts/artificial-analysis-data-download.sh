#!/usr/bin/env bash
# Load artificial analysis models data from the API and save to a file.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
env_file="${repo_root}/.env"
output_dir="${repo_root}/typst/data"
output_file="${output_dir}/artificial-analysis-models.json"

if [[ -f "${env_file}" ]]; then
  set -a
  # Export variables defined in the repo root .env file.
  source "${env_file}"
  set +a
fi

if [[ -z "${ARTIFICIAL_ANALYSIS_API_KEY:-}" ]]; then
  echo "ARTIFICIAL_ANALYSIS_API_KEY is not set in ${env_file}" >&2
  exit 1
fi

mkdir -p "${output_dir}"

curl -X GET "https://artificialanalysis.ai/api/v2/data/llms/models" \
  -H "x-api-key: ${ARTIFICIAL_ANALYSIS_API_KEY}" \
  -o "${output_file}"

echo "Saved Artificial Analysis data to ${output_file}"