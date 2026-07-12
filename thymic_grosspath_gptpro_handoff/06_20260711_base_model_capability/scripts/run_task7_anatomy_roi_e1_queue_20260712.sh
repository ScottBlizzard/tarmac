#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/workspace/thymic_project"
RUN_DIR="${PROJECT_ROOT}/experiments/base_model_capability_20260711/e1_anatomy_roi_20260712"
SCRIPT_DIR="/root"
STATUS_PATH="${RUN_DIR}/E1_QUEUE_STATUS.json"
ANALYZER_LOG="${RUN_DIR}/final_analysis/analyzer.log"
PYTHON_BIN="/root/miniconda3/envs/thymic_baseline/bin/python"

mkdir -p "${RUN_DIR}"
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export MALLOC_ARENA_MAX=2

write_status() {
  local state="$1"
  local detail="$2"
  "${PYTHON_BIN}" - "${STATUS_PATH}" "${state}" "${detail}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "state": sys.argv[2],
    "detail": sys.argv[3],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}
if path.exists():
    path.unlink()
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

trap 'write_status failed "runner or analyzer exited nonzero"' ERR
write_status running "fivefold and source-LODO direct-model evaluation"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_task7_anatomy_roi_e1_20260712.py" \
  --output-dir "${RUN_DIR}" \
  --split-modes fivefold,source_lodo

write_status analyzing "paired bootstrap and locked-gate analysis"
mkdir -p "$(dirname "${ANALYZER_LOG}")"
"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_task7_anatomy_roi_e1_20260712.py" \
  --run-dir "${RUN_DIR}" \
  > "${ANALYZER_LOG}" 2>&1

write_status complete "E1 results and advancement decision are ready"
