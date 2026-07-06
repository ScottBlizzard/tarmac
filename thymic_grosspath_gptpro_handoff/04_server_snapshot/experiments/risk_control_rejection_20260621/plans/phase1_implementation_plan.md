# Risk-Controlled Rejection Layer Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated Phase 1 experiment that generates stable case risk features and frozen audit summaries without modifying existing project code or outputs.

**Architecture:** All new files live under `experiments/risk_control_rejection_20260621/`. The experiment reads existing CSV artifacts as immutable inputs, writes derived tables under the experiment `outputs/`, and writes human-readable summaries under `reports/`.

**Tech Stack:** Python standard library plus pandas/numpy/sklearn available in the `thymic_baseline` conda environment.

---

## Files

- Create: `experiments/risk_control_rejection_20260621/scripts/build_case_risk_features.py`
  - Reads v201/v77 source artifacts and writes a stable `case_risk_features.csv`.
- Create: `experiments/risk_control_rejection_20260621/scripts/audit_rejection_layer.py`
  - Reads `case_risk_features.csv`, reconstructs v195 baseline behavior, applies conservative risk rankings, and writes audit summaries.
- Create: `experiments/risk_control_rejection_20260621/reports/phase1_summary.md`
  - Generated report with baseline checks and audit interpretation.
- Create: `experiments/risk_control_rejection_20260621/outputs/`
  - Contains generated CSV/JSON outputs only for this experiment.
- Create: `experiments/risk_control_rejection_20260621/logs/`
  - Contains command logs if long-running commands are needed.

## Task 1: Build Stable Case Risk Features

**Files:**
- Create: `experiments/risk_control_rejection_20260621/scripts/build_case_risk_features.py`
- Output: `experiments/risk_control_rejection_20260621/outputs/case_risk_features.csv`
- Output: `experiments/risk_control_rejection_20260621/outputs/batch_shift_features.csv`

- [ ] **Step 1: Implement source validation**

The script must verify that these read-only inputs exist:

```text
outputs/grosspath_rc_v201_stable_supported_domain_flip_20260527/v201_stable_supported_domain_flip_cases.csv
outputs/grosspath_rc_v77_batch_shift_audit_policy_switch_20260527/v77_unlabeled_batch_shift_audit.csv
```

- [ ] **Step 2: Derive stable case features**

The script must output one row per `case_id`, using stable features:

```text
domain, case_id, original_case_id, task_l6_label, label_idx, final_pred,
prob_mean_core, fold_id, base_wrong, base_confidence, base_uncertainty,
base_entropy, model_disagreement_rate, corrector_confidence_mean,
corrector_prob_high_mean, router_risk_mean, router_risk_max,
batch_shift_index, domain_auc_cv, mean_outside_ref_05_95_rate,
quality_proxy_mean
```

Version-specific action columns may be copied only as baseline labels:

```text
adaptive_review, adaptive_auto_decision, remaining_review, flip_trigger,
flip_error
```

- [ ] **Step 3: Run feature generation**

Run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
cd /workspace/thymic_project
python experiments/risk_control_rejection_20260621/scripts/build_case_risk_features.py
```

Expected:

```text
[features] wrote experiments/risk_control_rejection_20260621/outputs/case_risk_features.csv
```

## Task 2: Reconstruct v195 Baseline and Risk Curves

**Files:**
- Create: `experiments/risk_control_rejection_20260621/scripts/audit_rejection_layer.py`
- Output: `experiments/risk_control_rejection_20260621/outputs/v195_baseline_reconstruction.csv`
- Output: `experiments/risk_control_rejection_20260621/outputs/coverage_risk_curves.csv`
- Output: `experiments/risk_control_rejection_20260621/outputs/phase1_audit_summary.csv`
- Output: `experiments/risk_control_rejection_20260621/reports/phase1_summary.md`

- [ ] **Step 1: Implement baseline reconstruction**

The script must reproduce v195 behavior from:

```text
outputs/grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527/v185_unlabeled_shift_adaptive_cases.csv
outputs/grosspath_rc_v195_adaptive_autocorrect_gate_scan_20260527/v195_selected_candidate_action_cases.csv
```

It must compute per-domain and all-domain:

```text
n, auto_decision_n, auto_decision_rate, review_or_reject_n,
review_or_reject_rate, auto_error_n, fn, fp, accuracy, balanced_accuracy
```

- [ ] **Step 2: Implement stable risk score baselines**

Compute simple, auditable risk ranking scores from stable features:

```text
uncertainty_score = base_uncertainty
disagreement_score = model_disagreement_rate
router_score = router_risk_mean
hybrid_score = mean rank of uncertainty_score, disagreement_score, router_score
```

These are ranking baselines, not deployable models.

- [ ] **Step 3: Generate coverage-risk curves**

For each score and domain, evaluate automatic decision coverage at target review rates:

```text
0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80
```

Higher risk score means reviewed first. Remaining cases are automatic decisions.

- [ ] **Step 4: Apply conservative hard-gate audit label**

Use v77 shift metrics only for descriptive audit:

```text
strict_external = severe_unknown_shift
old_data = normal_or_known_shift
third_batch = normal_or_known_shift
```

Do not tune this from strict external labels.

- [ ] **Step 5: Run audit**

Run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline
cd /workspace/thymic_project
python experiments/risk_control_rejection_20260621/scripts/audit_rejection_layer.py
```

Expected:

```text
[audit] wrote experiments/risk_control_rejection_20260621/outputs/phase1_audit_summary.csv
```

## Task 3: Verify Outputs

**Files:**
- Read: `experiments/risk_control_rejection_20260621/outputs/case_risk_features.csv`
- Read: `experiments/risk_control_rejection_20260621/outputs/v195_baseline_reconstruction.csv`
- Read: `experiments/risk_control_rejection_20260621/outputs/coverage_risk_curves.csv`
- Read: `experiments/risk_control_rejection_20260621/reports/phase1_summary.md`

- [ ] **Step 1: Confirm feature row counts**

Run:

```bash
python - <<'PY'
import pandas as pd
p = "experiments/risk_control_rejection_20260621/outputs/case_risk_features.csv"
df = pd.read_csv(p)
print(df.shape)
print(df["domain"].value_counts().to_dict())
PY
```

Expected:

```text
(699, ...)
{'third_batch': 306, 'old_data': 285, 'strict_external': 108}
```

- [ ] **Step 2: Confirm v195 reconstruction**

The reconstructed all-domain v195 row should match:

```text
n=699
auto_decision_n=447
review_or_reject_n=252
auto_error_n=1
fn=1
fp=0
balanced_accuracy approximately 0.998148
```

- [ ] **Step 3: Confirm strict external freeze**

The report must explicitly say strict external is a frozen audit set and that coverage
improvements are exploratory only.

