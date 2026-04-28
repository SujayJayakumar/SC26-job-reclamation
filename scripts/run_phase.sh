#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_phase.sh <phase_id> [additional args]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONHASHSEED=42
if [[ -n "${PYTHON_CMD:-}" ]]; then
  PYTHON_BIN="${PYTHON_CMD}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No Python interpreter found. Set PYTHON_CMD or install python3." >&2
  exit 1
fi

PHASE_ID="$1"
shift

# Previous Linux/Swapnil-machine invocation kept for future switching:
# python3 -m src.hpc_sim.pipeline --config "${ROOT_DIR}/config/default.json" --phase "${PHASE_ID}" "$@"
"${PYTHON_BIN}" -m src.hpc_sim.pipeline --config "${ROOT_DIR}/config/default.json" --phase "${PHASE_ID}" "$@"
