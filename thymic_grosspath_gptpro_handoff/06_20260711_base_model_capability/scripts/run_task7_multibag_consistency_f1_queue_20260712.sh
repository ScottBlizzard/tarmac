#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/workspace/thymic_project"
RUN_DIR="${PROJECT_ROOT}/experiments/base_model_capability_20260711/f1_multibag_consistency_20260712"
PYTHON_BIN="/root/miniconda3/envs/thymic_baseline/bin/python"
RUNNER="/root/run_task7_multibag_consistency_f1_20260712.py"
ANALYZER="/root/analyze_task7_multibag_consistency_f1_20260712.py"
STATUS_PATH="${RUN_DIR}/F1_QUEUE_STATUS.json"
RUNNER_LOG="${RUN_DIR}/runner.log"
ANALYZER_LOG="${RUN_DIR}/final_analysis/analyzer.log"

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
write_status running "F1 extraction and direct-model evaluation"

"${PYTHON_BIN}" "${RUNNER}" \
  --output-dir "${RUN_DIR}" \
  --split-modes fivefold,source_lodo \
  > "${RUNNER_LOG}" 2>&1

write_status analyzing "paired bootstrap and locked-gate analysis"
mkdir -p "$(dirname "${ANALYZER_LOG}")"
"${PYTHON_BIN}" "${ANALYZER}" \
  --run-dir "${RUN_DIR}" \
  > "${ANALYZER_LOG}" 2>&1

write_status complete "F1 results and advancement decision are ready"
