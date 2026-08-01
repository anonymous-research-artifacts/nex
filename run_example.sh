#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [OUTPUT_DIR]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/example_output}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_nex.py" score \
  --cache-dir "${SCRIPT_DIR}/example_cache" \
  --weights "${SCRIPT_DIR}/example_weights.npz" \
  --output-dir "${OUTPUT_DIR}" \
  --log-gap-output
