# Request for a Structurally New Base-Capability Plan After Sequential Expert Failure

Repository: https://github.com/ScottBlizzard/tarmac

Please read first:

1. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/H1_SEQUENTIAL_AB_TC_FALLBACK_RESULTS_20260713.md`
2. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/Task7_WaveC_D1_E1_F1_F2_Results_20260712.md`
3. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/Task7_Base_Model_Capability_Experiments_20260711.md`
4. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/reports/AI Pathology Model Improvement.md`
5. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/scripts/run_task7_sequential_ab_tc_fallback_20260713.py`

## Scientific Objective

We need a stronger 100%-coverage, image-only low-risk versus high-risk classifier for thymic gross pathology. Low risk is A/AB/B1; high risk is B2/B3/thymic carcinoma. Rejection, manual review, confidence-only correction, source-aware calibration, and behavior-level output stacking are not substitutes for base visual capability.

Multi-model systems are acceptable only when their members read images and learn complementary morphology. Existing physician concept annotations may be used for retrospective error analysis and hypothesis generation, but not as deployment inputs. Do not require a new 120-case physician annotation exercise; the current physician table already documents visible characteristics for 589 of 591 internal cases.

## Data Constraint

Internal development contains 591 cases:

- A 44, AB 262, B1 62, B2 89, B3 24, TC 110.
- Low risk 368, high risk 223.
- Canonical evaluation is five-fold OOF plus source-LODO over batch1, batch2, and third batch.
- Historical external 108 and newer external 162 were already inspected and are consumed audit sets. They cannot be reused for training, selection, calibration, or a new generalization claim.

## Locked Baselines

- C1: SigLIP-L@512, whole/foreground/four-quadrant six views, all dense patch tokens, low-capacity gated pooling.
  - Five-fold BAcc/AUC 0.7477/0.8240.
  - Source-LODO BAcc/AUC 0.7397/0.8072.
- C2: equal image-model fusion of C1 and AIMv2 MixStyle.
  - Five-fold BAcc/AUC 0.7514/0.8377.
  - Source-LODO BAcc/AUC 0.7441/0.8108.
  - LODO B1/B2 risk accuracy 0.5000/0.6629.

## New Sequential Expert Experiment

We preregistered and completed a sequential hierarchy:

1. AB-versus-rest image expert.
2. TC-versus-A/B1/B2/B3 expert, augmented with AB cases missed by training-side cross-fitted AB experts.
3. Six-subtype-covered binary fallback using all A/B1/B2/B3 and 40 dynamically sampled AB plus 40 TC per epoch, half enriched for training-side expert misses.
4. AB-only routes emitted low risk, TC-only routes emitted high risk, and conflicts/non-routes used the fallback.
5. All heads used the same frozen C1 six-view dense image tokens with independent gated pooling heads.

Exact result:

- Five-fold BAcc/AUC 0.7195/0.8118.
- Source-LODO BAcc/AUC 0.7278/0.7895.
- LODO sensitivity/specificity 0.6457/0.8098.
- Versus C2, LODO sensitivity changed by -0.0882, 95% CI [-0.1435, -0.0359], while specificity changed by +0.0574 [0.0217, 0.0951].
- LODO subtype accuracy was A 0.7273, AB 0.8626, B1 0.6452, B2 0.5281, B3 0.6250, TC 0.7455.
- The fallback core A/B1-versus-B2/B3 BAcc was only 0.5765 five-fold and 0.5998 LODO, below C2's 0.6072/0.6193.
- AB expert BAcc/AUC fell from 0.7811/0.8505 five-fold to 0.6178/0.6280 LODO.
- TC expert retained 0.7473/0.8450 LODO, but no LODO fold met the fixed high-purity route gate, so TC routed zero cases.
- Routing itself was slightly net positive. The dominant failure was that balanced residual training did not learn a stronger visual boundary; it improved B1 while harming B2/B3/TC.

The formal decision was NO-GO. We will not search 30/40/50 anchor counts, thresholds, seeds, or fusion weights.

## Already Exhausted Families

Do not repeat ordinary backbone swaps, LoRA capacity changes, class weighting, focal loss, SAM, GroupDRO/REx/DANN, MixStyle sweeps, supervised contrastive variants, B1/B2 hard-negative losses, six-class auxiliary heads, prototype/concept heads, boundary experts, MoE gates, standard preprocessing, fixed spatial pyramids, native image grids, label-trained attention ROI, deterministic anatomy ROI, random bags, bag-consistency weights, confidence routing, or probability-fusion searches. These have already been executed under OOF/LODO and mainly move errors between sources or B1/B2.

## Questions

1. Does the new evidence close subtype-specialist hierarchies on the existing single-photo dataset, or only this hard-routing and balanced-fallback realization? Give a precise causal interpretation.
2. Is there any remaining experiment on the existing 591 images that changes the visual information or learning assumption enough to be scientifically defensible? Do not propose another threshold, sampler-count, loss-weight, boundary-head, backbone, or output-stack search.
3. If one experiment remains, specify exact inputs, model, nested training protocol, controls, expected failure mode, and a hard GO/NO-GO gate. Limit the answer to at most two primary experiments.
4. If the correct conclusion is that existing single photographs are information-limited, state this directly and propose a minimum new-data acquisition design: which standardized views, which subtypes, how many independent cases per source/center, and how to preserve a genuinely untouched external test.
5. Explain how the next step would improve actual image-grounded capability rather than merely move the operating point between sensitivity and specificity.

Be explicit about verified facts, inferences, and recommendations. The priority is genuine base-model capability and cross-source generalization, not release coverage or physician review workload.
