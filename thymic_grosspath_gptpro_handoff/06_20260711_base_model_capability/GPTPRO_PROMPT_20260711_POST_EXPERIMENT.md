# GPTPro Prompt: Post-Experiment Scientific Audit and Next Decision

You are a senior research lead in medical AI, computational pathology, domain generalization, and clinical study design. Inspect this repository carefully:

Repository: https://github.com/ScottBlizzard/tarmac

Package: `thymic_grosspath_gptpro_handoff/`

Read these files first:

1. `06_20260711_base_model_capability/README_20260711_BASE_MODEL_CAPABILITY.md`
2. `06_20260711_base_model_capability/reports/Task7_Base_Model_Capability_Experiments_20260711.md`
3. `06_20260711_base_model_capability/scripts/phase2_fresh_external_candidate_lock_20260711.csv`
4. `06_20260711_base_model_capability/reports/FRESH_EXTERNAL_BLIND_TEST_PROTOCOL_20260711.md`
5. `06_20260711_base_model_capability/reports/AI Pathology Model Improvement.md`
6. Relevant implementations under `06_20260711_base_model_capability/scripts/`

Use the older folders only to verify history and avoid proposing repeats.

## Scientific Objective

Task7 classifies thymic gross pathology images at 100% coverage:

- Low risk: A / AB / B1.
- High risk: B2 / B3 / thymic carcinoma.

The primary objective is a stronger image-only forced classifier with honest cross-hospital generalization. Rejection, physician review, and selective release are downstream safety layers, not substitutes for base-model ability.

## Fixed Data Boundary

- Internal development: 591 cases, comprising batch1 117, batch2 168, and third_batch 306.
- Historical strict external: 108 cases.
- Newer external: 162 cases.

Both external cohorts have already been inspected. They are consumed audit sets and cannot confirm the 2026-07-11 Phase-2 candidates. Any new external claim requires a genuinely fresh, label-blinded cohort.

## Current Locked Candidates

`C1`: SigLIP-L at 512 px, six deterministic views from one primary image, all dense patch tokens, gated pooling.

- OOF BAcc/AUC: 0.7477/0.8240.
- Source-LODO BAcc/AUC: 0.7397/0.8072.

`C2`: equal average of AIMv2 MixStyle run 253 and C1.

- OOF BAcc/AUC: 0.7514/0.8377.
- Source-LODO BAcc/AUC: 0.7441/0.8108.
- LODO B1/B2 risk accuracy: 0.5000/0.6629.

The six-view representation produced a statistically supported OOF improvement over the previous two-view SigLIP-512 model, but not a significant LODO BAcc improvement. C2 improved LODO AUC over the old locked fusion, while its BAcc difference remained inconclusive. Do not describe either candidate as externally validated.

## Completed Negative Evidence

The project has already tested broad backbone changes, dense versus pooled tokens, many pooling heads, LoRA, VICReg, supervised contrastive learning, cross-source pairs, subtype and B1/B2 hard negatives, DANN, GroupDRO, REx, class-conditional alignment, MixStyle, SAM optimization, multiple ROI and quality preprocessing strategies, six-class and ordinal heads, concepts, boundary experts, sentinels, MoE, and large fusion searches. The report contains exact results.

Do not recommend a low-level repeat such as another ordinary backbone probe, another minor loss weight, or another internal fusion sweep unless you identify a genuinely changed assumption and explain why existing negative evidence does not apply.

## Your Task

Answer in English as a critical research lead, not as a supportive summarizer.

1. Audit the evidence chain.
   - Check data roles, split logic, OOF versus source-LODO use, threshold discipline, fusion-selection bias, bootstrap interpretation, and external-label leakage.
   - Identify any claim in the 2026-07-11 report that is too strong, incomplete, or incorrectly framed.

2. Judge the candidate lock.
   - Is locking C1 and C2 defensible?
   - Should one be primary and the other secondary?
   - Is the old `215+253+254` fusion the correct frozen comparator?
   - Specify exactly what result would justify replacing the historical model.

3. Redesign the fresh external study if needed.
   - Propose cohort size, center count, class/subtype coverage, primary-image rules, metadata, blinding, prediction commitment, and statistical analysis.
   - Give an approximate sample-size or precision argument for sensitivity, specificity, and paired model comparison.
   - Address multi-image cases without changing the primary endpoint after seeing outcomes.

4. Diagnose the remaining B1/B2 bottleneck.
   - Separate irreducible gross-visual ambiguity, label noise/borderline histology, view inadequacy, acquisition shift, and model failure.
   - Design a physician review protocol that can distinguish these explanations.
   - Interpret the corrected `texture_soft` and `nodular_lobulated` associations cautiously.

5. Define the next technical branch only if fresh external performance remains insufficient.
   - Provide 8-12 structurally new, high-upside directions that are not repetitions of completed runs.
   - For each, state the changed assumption, required new data, implementation, internal falsification test, and stopping rule.
   - Prioritize data/acquisition changes when they are more defensible than further tuning on the same 591 cases.

6. Give a manuscript decision tree.
   - Strong fresh-external result.
   - Modest or statistically inconclusive result.
   - External failure despite internal improvement.
   - For each, specify defensible contribution, target evidence, required figures/tables, and forbidden claims.

## Required Output Structure

1. `Evidence-Chain Audit`
2. `Candidate-Lock Decision`
3. `Fresh External Study Design and Sample-Size Logic`
4. `B1/B2 Failure Adjudication Plan`
5. `Only Genuinely New Technical Directions`
6. `Manuscript Decision Tree`
7. `Immediate Action List for the Modeling Team and Physicians`

Be explicit about uncertainty. Distinguish internally supported facts, source-based inference, and recommendations.
