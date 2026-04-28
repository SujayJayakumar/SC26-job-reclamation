#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUTS_DIR="${ROOT_DIR}/outputs"
LOG_DIR="${OUTPUTS_DIR}/logs"
export PYTHONHASHSEED=42
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
if [[ -n "${PYTHON_CMD:-}" ]]; then
  PYTHON_BIN="${PYTHON_CMD}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No Python interpreter found. Set PYTHON_CMD or install python3." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Run log: ${RUN_LOG}"
echo "Starting full reproducibility run"

# Previous Linux/Swapnil-machine invocation kept for future switching:
# python3 main.py --phase preprocess
# python3 main.py --phase simulate
# python3 main.py --phase metrics
# python3 main.py --phase plots
"${PYTHON_BIN}" "${ROOT_DIR}/main.py" --phase preprocess "$@"
"${PYTHON_BIN}" "${ROOT_DIR}/main.py" --phase simulate "$@"
"${PYTHON_BIN}" "${ROOT_DIR}/main.py" --phase metrics "$@"
"${PYTHON_BIN}" "${ROOT_DIR}/main.py" --phase plots "$@"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/validate_pipeline.py" --config "${ROOT_DIR}/config/default.json"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/export_outputs.py" --config "${ROOT_DIR}/config/default.json"

echo "Completed full reproducibility run"
