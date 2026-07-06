# Phase 3 Sidecar Original Project Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use test-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a no-modification sidecar adapter that reads project output tables, validates or maps them into the risk-control runtime feature contract, and runs the existing hard-gate deployable wrapper.

**Architecture:** The sidecar lives entirely under `experiments/risk_control_rejection_20260621/sidecar_original_project_adapter/`. It does not modify root project scripts. It accepts a CSV, applies deterministic alias mapping where possible, validates the final table against `final_policy_contract.json`, calls the staged hard-gate wrapper, and writes sidecar-only outputs.

**Tech Stack:** Python, pandas, existing experiment integration wrapper, existing policy contract validator, unittest.

---

### Task 1: Sidecar Adapter Core

**Files:**
- Create: `experiments/risk_control_rejection_20260621/sidecar_original_project_adapter/original_project_sidecar_adapter.py`
- Modify: `experiments/risk_control_rejection_20260621/tests/test_phase1_helpers.py`

- [x] **Step 1: Write failing tests**

Add tests that verify a stable-feature input runs through the wrapper, v185-style aliases are mapped, and forbidden relaxed columns are rejected.

- [x] **Step 2: Verify tests fail**

Run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
python -m unittest /workspace/thymic_project/experiments/risk_control_rejection_20260621/tests/test_phase1_helpers.py -v
```

Expected: failure because `original_project_sidecar_adapter` does not exist.

- [x] **Step 3: Implement sidecar**

Create a focused adapter with:

- `build_runtime_features_from_project_output(frame)`
- `run_sidecar_adapter(input_csv, out_dir, contract_path)`
- CLI arguments for input CSV, output directory, and contract.

- [x] **Step 4: Verify tests pass**

Run the same unittest command and confirm all tests pass.

### Task 2: Sidecar Reproducibility

**Files:**
- Modify: `experiments/risk_control_rejection_20260621/scripts/phase2_reproducibility_runner.py`
- Create sidecar output files under `experiments/risk_control_rejection_20260621/sidecar_original_project_adapter/outputs/`

- [x] **Step 1: Add sidecar to runner**

Append the sidecar CLI command after the Phase 3 staging package step.

- [x] **Step 2: Run full verification**

Run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
python experiments/risk_control_rejection_20260621/scripts/phase2_reproducibility_runner.py
```

Expected: all runner steps pass.

### Task 3: Documentation

**Files:**
- Create: `experiments/risk_control_rejection_20260621/sidecar_original_project_adapter/README.md`
- Update: `experiments/risk_control_rejection_20260621/reports/phase2_final_experiment_report.md`

- [x] **Step 1: Write sidecar usage notes**

Document the no-modification boundary, accepted inputs, outputs, and blocked relaxed logic.

- [x] **Step 2: Update final report**

Add the sidecar path and verification result.

- [x] **Step 3: Final verification**

Run the full reproducibility runner again before reporting status.
