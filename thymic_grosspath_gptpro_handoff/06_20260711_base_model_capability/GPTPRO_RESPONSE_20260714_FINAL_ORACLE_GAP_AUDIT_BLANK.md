# GPT Pro Response: Final Audit of the Cross-Representation Oracle Gap

Paste the complete GPT Pro response below this line.
# Final Audit of the Cross-Representation Oracle Gap

Evidence lock: `ScottBlizzard/tarmac@ff81fe4c98fd5f1b9d2bcfd53f4379e5741e38e8`  
Audit date: 2026-07-14  
Task: one selected thymic gross photograph per case; low risk A/AB/B1 versus high risk B2/B3/TC; 100% coverage; threshold 0.5.

## 1. **Executive verdict**

**Decision: Branch A — run exactly one terminal experiment.**

The only scientifically defensible remaining experiment is:

> **H8_C1_H3_DIRECT_CASE_EMBEDDING_FUSION_20260714** — a fully nested source-LODO Task7 classifier with a single fixed 6,226-parameter nonlinear disease head trained directly on the concatenation of the frozen, fold-specific **C1 SigLIP-L case embedding** and **H3 PE-Spatial case embedding**.

This decision is based on three distinct levels of evidence:

- **Direct repository evidence:** the apparent 0.9006 oracle is label-aware and non-deployable; ordinary averaging and confidence selection fail. The best locked source-LODO direct model remains H3 at 0.7539 BAcc, with a B1 gain but a B2/sensitivity loss. fileciteturn69file0L13-L25 fileciteturn69file0L42-L60
- **Audit inference after code inspection:** the repository contains output-level averaging, prediction/logit meta-modeling, same-representation MoE, same-backbone cascades, and first/second-order fusion inside PE; it does **not** contain a model that gives aligned C1 and H3 case tensors to one low-capacity Task7 head under nested source-LODO. This is an absence conclusion from the required reports plus the relevant implementations, not a claim inferred from experiment names.
- **Prospective hypothesis:** same-case cross-family embeddings may contain disease evidence that output probabilities discard. That possibility is plausible but unproven. H8 is permitted only as a one-shot terminal mechanism test, not as the start of another fusion search.

The advancement rule is deliberately asymmetric: **all preregistered gates must pass**. Failure of any integrity, source-LODO, boundary, source-control, five-fold, or confirmation-seed gate triggers the prominent final decision:

> **STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT**

No subsequent hidden-width, loss, seed, threshold, fusion, router, or architecture search is allowed.

A positive H8 result would support only a narrowly worded claim: direct, same-case, cross-representation image features improved internal acquisition-shift performance under a fixed protocol. It would not establish multicenter generalization, clinical validity, a 90% classifier, or a biological Bayes ceiling.

## 2. **Repository audit of cross-representation fusion coverage**

### Audit classification

- **Direct evidence** means the behavior is explicit in code or a locked result file.
- **Audit inference** means the conclusion follows from checking all required reports and the relevant implementation paths at the evidence commit.
- **Speculation/hypothesis** is reserved for what H8 might show and is not represented as an established fact.

| Required audit question | Finding | Evidence level and basis |
| --- | --- | --- |
| 1. Were locked C1 and H3 case embeddings ever concatenated or jointly attended by a low-capacity Task7 disease head? | **No implementation was found.** | **Audit inference.** The C1 model forms one shared embedding and its risk, boundary, subtype, sentinel, and MoE heads all consume that same embedding. fileciteturn72file0L11-L80 fileciteturn73file0L24-L64 H3 separately forms a PE-only masked-gated embedding. fileciteturn51file0L112-L159 H5 concatenates first- and second-order summaries derived from the **same PE tensor**, not C1 and H3. fileciteturn52file0L119-L206 |
| 2. Was such a direct C1–H3 model evaluated in fully nested outer source-LODO? | **No.** | **Audit inference.** C1 and H3 each have source-LODO results, but the inspected fusion implementations consume predictions or a single representation family. The locked split code excludes the held source and uses an internal validation fold, but no two-family head invokes both branches. fileciteturn68file0L39-L62 |
| 3. What did earlier “fusion” actually combine? | Predominantly probabilities, logits, prediction summaries, or hand-engineered ROI values. | **Direct evidence.** The large internal search loads prediction files and averages probability vectors. fileciteturn74file0L30-L49 fileciteturn74file0L98-L138 The qkvb/ROI meta-model builds probability, logit, range, and ROI numeric features from OOF predictions. fileciteturn75file0L59-L111 fileciteturn75file0L137-L146 F2 is an exact 0.5/0.5 probability average. fileciteturn59file0L81-L92 |
| 4. Did MoE and cascades use genuinely different image representations? | Generally **no** for MoE; the main cascade’s second reader used different crops but the learned fusion was over stage probabilities and remained within the SigLIP family. | **Direct evidence.** C1 MoE experts and gate share one pooled embedding. fileciteturn73file0L48-L64 H1 explicitly states all experts consumed the frozen C1 six-view representation. fileciteturn76file0L13-L22 In the B1 cascade, M3 fits logistic fusion to C1 and stage-2 logits/probabilities; the second reader is another SigLIP native-tile reader, not PE-Spatial. fileciteturn77file0L42-L56 fileciteturn57file0L15-L23 |
| 5. Is the oracle mainly a threshold phenomenon? | **No.** Thresholding cannot explain most of the gap, but the oracle still does not provide a transferable selector. | **Direct evidence.** Even label-informed post-hoc global thresholding peaks at 0.7730 for H3, and cross-source threshold transfer remains around 0.76. fileciteturn71file0L4-L24 The six-model oracle is 0.9006, while averaging is 0.7602 and maximum-confidence selection is 0.7462. fileciteturn69file0L64-L86 |
| 6. Are the required assets available or safely regenerable? | **Conditionally yes, with a mandatory offline preflight.** | **Direct evidence plus current-state uncertainty.** The PE-Spatial checkpoint and official source are local with a locked SHA-256; its large dense bank was removed but can be streamed case by case. fileciteturn65file0L26-L42 C1’s extraction queue used the cached timm SigLIP-L model with offline flags, generated run 348, and deleted the regenerable dense bank after saving small manifests. fileciteturn47file0L20-L23 fileciteturn47file0L53-L75 fileciteturn47file0L120-L130 GitHub cannot verify that every server checkpoint/cache file still exists today; H8 therefore begins with a no-download hash-and-presence lock, and absence is an immediate stop. |

### What has and has not been covered

The repository has already covered four superficially similar but scientifically different mechanisms:

1. **Output fusion:** fixed or searched means of case probabilities. The fusion search explicitly loads `probability` columns and applies `np.mean`; it never reads C1/H3 feature arrays. fileciteturn74file0L30-L49 fileciteturn74file0L119-L138
2. **Behavior/meta fusion:** classifiers over OOF probabilities, logits, distribution summaries, and optional ROI statistics. fileciteturn75file0L66-L111
3. **Shared-representation specialists:** MoE, boundary, subtype, or sequential heads attached to one C1 embedding. fileciteturn72file0L41-L79 fileciteturn76file0L13-L22
4. **Within-PE feature fusion:** H5 joins first-order and covariance summaries, both generated from PE-Spatial tokens. fileciteturn52file0L119-L206

The untested path is narrower: **two separately trained, fold-clean image representation families; their pre-classifier case embeddings; one new disease head; no probabilities, source labels, confidence, text, or correctness targets as inputs.**

## 3. **What the 0.9006 oracle does and does not prove**

### What it proves

The oracle calculation proves that the six classifiers do not make identical errors. When the true label is allowed to decide retrospectively which model to trust, the aligned prediction set reaches 0.9006 BAcc; excluding C1 still gives 0.8925, while the PE-only H3/H5/H6/H7 oracle falls to 0.8462. That pattern is evidence that some error diversity lies **between** representation families, not only among nearby PE heads. fileciteturn69file0L64-L84

It also proves that the plateau is not reducible to one global threshold. H3’s best same-cohort post-hoc threshold reaches only 0.7730 BAcc, far below 0.9006. fileciteturn71file0L4-L22

### What it does not prove

The oracle is not a trainable target because its decision rule contains the true answer: “select any model that was correct.” It gives no observable deployment-time variable identifying that model. The repository’s maximum-confidence selector performs below H3, and ordinary averaging changes BAcc by only about 0.0063 relative to H3 with a paired interval crossing zero; B2 worsens. fileciteturn69file0L68-L86

The oracle therefore does **not** prove any of the following:

- that 90% is attainable from the current photograph;
- that a gate can identify the right representation for each case;
- that error diversity is pathological rather than acquisition-related;
- that the remaining 53 cases missed by all six models can be solved without new image evidence;
- that source-LODO performance will transfer to independent hospitals.

The common-error audit is particularly important: 53/591 cases are wrong for all six models and 83/591 are wrong for at least five; B1/B2 and third-batch B2 are overrepresented. fileciteturn70file0L37-L56

### Actionable interpretation

The oracle creates **one testable opportunity**, not a performance promise. Output probabilities compress each representation into a one-dimensional decision score. A direct head over C1 and H3 pre-classifier embeddings could, in principle, use complementary image features that were discarded before probability averaging. That is a hypothesis. H8 must show that the gain:

1. exceeds both single-branch heads;
2. survives outer source-LODO;
3. preserves H3’s B1 benefit and C2’s B2/sensitivity benefit;
4. remains when judged separately in held sources;
5. exceeds a source-preserving, same-case-pair-destroyed negative control.

Without all five, the oracle remains retrospective error diversity rather than usable visual capability.

## 4. **Source-confounding and identifiability assessment**

### Direct project evidence

The cohort is not a balanced domain-generalization design. Source is associated with risk and much more strongly with six-class subtype; two source-by-subtype cells are empty and batch2-B3 contains only four cases. fileciteturn69file0L88-L120 Fixed frequency features predict acquisition source better than risk after the reciprocal adjustment, and all 156 audited frequency features have a larger source partial effect than risk partial effect. fileciteturn70file0L3-L11

The source-by-subtype behavior is not a minor calibration effect. For B2, C1/C2 are nearly perfect in batch1 but all methods deteriorate in third batch; H3 answers only 13/29 third-batch B2 correctly, while C2 answers 18/29. B1 difficulty moves in the opposite direction across sources. fileciteturn70file0L17-L35

Input completeness is also fixed: 574/591 cases have one photograph, only 17 old-domain cases have a second image, and all “six views” are deterministic derivatives of the same photograph. fileciteturn70file0L74-L97 This limits what any fusion can recover: it can combine representations of visible evidence, but it cannot reconstruct an unphotographed cut surface, capsule interface, or internal heterogeneous region.

### Primary literature context

Zech et al. showed that medical-image models can identify acquisition institution almost perfectly and can obtain inflated internal performance when site and disease prevalence are coupled; the apparent advantage failed to transfer externally. citeturn986992view0 Badgeley et al. showed that scanner, patient, and hospital-process variables were predictable from radiographs and that matching those confounders could reduce a fracture model to near-random performance. citeturn223971view0 These studies do not prove the same mechanism in thymic gross photography, but they establish why source-preserving controls are necessary rather than optional.

The project’s medical background sources also matter: the WHO endpoint is histological; the cited ITMIG consensus and reproducibility work describe classification criteria and interobserver limitations, including difficulty around neighboring thymoma categories. The repository correctly frames this as an evidence-level mismatch, not proof of an immutable ceiling. fileciteturn63file0L20-L25 fileciteturn70file0L58-L72

### Identifiability judgment

H8 is **identifiable enough for one terminal internal test**, but not for a broad generalization claim.

- Outer source-LODO removes the held batch from all supervised fitting and model selection.
- Source×risk-balanced sampling on the remaining sources reduces the direct incentive to use source prevalence.
- Per-source advancement gates prevent a global gain caused by one dominant batch.
- A deterministic **within-source case derangement** preserves each branch’s source distribution while destroying C1–H3 same-case alignment. If exact fusion does not beat this control, the result is compatible with source recognition or independent marginal behavior rather than conditional same-case evidence.
- The control still cannot exclude every case-specific acquisition nuisance shared by both encoders. Therefore even a pass supports “cross-representation image evidence under internal batch shift,” not “source-invariant pathology.”

Repeated reuse of the 591-case cohort remains a limitation. The H8 mechanism is new, but its nomination followed many prior results. Domain-generalization model selection is itself nontrivial, and an algorithm without a prespecified selection rule is methodologically incomplete. citeturn223971view1 Accordingly, H8 confidence intervals are used as **stability gates**, not as pristine confirmatory inference, and the consumed 108- and 162-case cohorts are not opened during selection.

## 5. **Decision: one locked experiment or stop**

### Decision

Run **one** experiment: `H8_C1_H3_DIRECT_CASE_EMBEDDING_FUSION_20260714`.

### A1. Changed assumption

H8 changes exactly one assumption: instead of combining final probabilities or placing several heads on one representation, it tests whether the **aligned pre-classifier case embeddings from two genuinely different image encoders** contain jointly useful Task7 evidence. C1/C2, F2, and the internal fusion searches combined outputs; H1 and MoE reused C1; the B1 cascade fused stage behavior from SigLIP-family readers; H3-H7 and H5’s first/second-order fusion remained within PE. A positive H8 result would be direct image capability only if a fold-clean C1+H3 head beats same-capacity branch heads, locked H3, and a within-source pair-destroyed control while improving at least two held sources and jointly preserving B1 and B2. The hypothesis is closed if **any** preregistered gate fails; the exact closure statement is `STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT`.

### Why immediate stopping is not yet warranted

Stopping now would leave a specific repository gap untested: no inspected code path sends C1 and H3 image-derived case tensors to the same disease head. The required assets are locally available or regenerable without a download, and the experiment can be implemented with a tiny head and streamed embeddings rather than multi-gigabyte banks. fileciteturn65file0L26-L42 fileciteturn47file0L53-L75

### Why this is terminal

The experiment uses one architecture, one hidden width, one loss, one learning rate, one primary seed, fixed threshold 0.5, and fixed gates. There is no backup fusion family. A negative result eliminates the remaining direct C1–H3 case-embedding hypothesis under the fixed data boundary; more search would be adaptive reuse rather than a distinct scientific test.

## 6. **Complete branch-A preregistration**

### A2. Immutable model specification

#### Cohort and views

- Cases: exactly 591 unique registry rows.
- Task: A/AB/B1 = 0; B2/B3/TC = 1.
- Coverage: 591/591.
- Decision threshold: exactly 0.5.
- C1 views: `whole,crop,crop_q0,crop_q1,crop_q2,crop_q3`, 512×512, deterministic existing preprocessing.
- H3 views: the same six semantic views, 448×448, deterministic existing H3 preprocessing.
- No second-image bag, text, gross-concept table, source label, confidence, probability, logit, prediction, margin, or correctness feature is provided to the candidate head.

#### Frozen tensors and embeddings

For outer fold `f` and case `i`:

- C1 dense tensor `T^C_{if} ∈ R^(6×1024×1024)` from frozen `vit_large_patch16_siglip_512.v2_webli`; frozen fold-specific C1 gated pooler produces pre-classifier `c_if ∈ R^256`.
- H3 dense tensor `T^H_{if} ∈ R^(6×1024×1024)` with valid-token mask `M^H_i ∈ {0,1}^(6×1024)` from frozen PE-Spatial-L14-448; frozen fold-specific H3 masked-gated pooler produces pre-classifier `h_if ∈ R^128`.
- The expected H3 bank shape and hashes are already locked in the H7 implementation. fileciteturn60file0L40-L47
- The candidate never stores either dense tensor. It streams one encoder at a time, computes all fold-specific pooled embeddings for the case, writes the small embedding shard atomically, and discards tokens.
- Both pretrained encoders and all existing C1/H3 pooling-head parameters remain frozen.

Before training, the extractor must reproduce each locked branch probability from its extracted embedding plus the locked classifier with maximum absolute error `≤1e-5`. Failure for any case/fold is an integrity failure, not a tuning opportunity.

#### Candidate equation

For each case:

```text
c̄ = c / max(||c||₂, 1e-6)
h̄ = h / max(||h||₂, 1e-6)
x = concat(c̄, h̄) ∈ R^384
u = GELU(W₁x + b₁),              W₁ ∈ R^(16×384)
z = LayerNorm₁₆(u)
logits = W₂ Dropout₀.₁₀(z) + b₂, W₂ ∈ R^(2×16)
p_high = softmax(logits)[1]
```

There is no branch gate, attention block, residual MLP, calibration layer, temperature, or learned threshold.

#### Trainable parameters

| Layer | Parameters |
| --- | ---: |
| `Linear(384,16)` | 6,160 |
| affine `LayerNorm(16)` | 32 |
| `Linear(16,2)` | 34 |
| **Total** | **6,226** |

Initialization is fixed:

- `W1`, `W2`: Xavier uniform, gain 1.0;
- all linear biases: zero;
- LayerNorm scale: one; bias: zero;
- deterministic seed: `20260714`.

#### Optimization

| Item | Locked value |
| --- | --- |
| Loss | two-class cross-entropy |
| Sampler | inverse-frequency `source_dataset × binary-risk` sampler over **outer-training cases only**, replacement, `num_samples=len(train)` |
| Additional class weights | none |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Batch size | `32` |
| Epoch ceiling | `80` |
| Early-stopping patience | `12` |
| Validation criterion | balanced accuracy at threshold 0.5 |
| Tie rule | earliest epoch |
| Scheduler | cosine annealing, `T_max=80` |
| Gradient clipping | L2 norm `5.0` |
| Precision | FP32 head; frozen extraction may use the existing safe autocast mode |
| Primary seed | `20260714` |
| Conditional confirmation seed | `20260715`, allowed only after all primary and secondary gates pass |
| Determinism | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN benchmark off, deterministic algorithms on |

### A3. Leakage-safe source-LODO

Source-LODO is run first and is the primary endpoint. The existing fold convention is retained exactly: held source is indexed in the order `batch1,batch2,third_batch`; validation uses `val_fold=(fold_id mod 5)+1` among non-held cases. fileciteturn66file0L23-L25 fileciteturn68file0L39-L62

| Outer fold | Held test source | Validation master fold within remaining sources |
| --- | --- | ---: |
| 1 | batch1 | 2 |
| 2 | batch2 | 3 |
| 3 | third_batch | 4 |

For each fold:

1. Use only the pre-existing fold-specific C1 and H3 source-LODO checkpoints whose supervised fitting excluded the held source.
2. Extract frozen embeddings for train, validation, and held-test cases without any target-source normalization or adaptation.
3. Fit the H8 head only on outer-training cases.
4. Calculate source×risk sampler weights only from outer-training cases.
5. Early-stop only on the predefined validation subset from the remaining sources.
6. Apply the chosen state once to the held source.
7. Emit exactly one prediction per case at threshold 0.5.
8. Merge the three held-source predictions to a 591-case OOF table.

No target-source statistic is allowed to alter the decision function. Samplewise L2 normalization and LayerNorm are permitted because they do not estimate cohort statistics.

### A4. Required controls

All learned controls use the same 6,226-parameter code path, optimizer, sampler, seed, folds, and early stopping.

| Tag | Input to the identical 384→16→2 head | Purpose |
| --- | --- | --- |
| `C1_ONLY_PADDED` | `concat(c̄, zeros(128))` | Tests C1 with the same new head capacity. |
| `H3_ONLY_PADDED` | `concat(zeros(256), h̄)` | Tests H3 with the same new head capacity. |
| `C1_H3_EXACT` | `concat(c̄_i, h̄_i)` | **Only advancing configuration.** |
| `C1_H3_SAME_SOURCE_DERANGED` | `concat(c̄_i, h̄_π(i))` | Source-preserving negative control. |

The derangement is immutable and label-free:

- separately within each `split × source_dataset` group;
- sort cases by `SHA256("H8|fold|split|case_id")`;
- cyclically shift the H3 order by one;
- require group size at least two;
- never use risk or subtype labels to form the permutation;
- create separate train, validation, and held-test permutations;
- store the mapping server-only.

This control preserves the source distribution and marginal H3 feature distribution but destroys same-case alignment. Exact fusion must beat it; otherwise the putative gain is not attributable to same-case cross-family evidence.

Additional fixed, non-advancing comparators:

- locked C1 predictions on the exact cases;
- locked C2 predictions on the exact cases;
- locked H3 predictions on the exact cases;
- fixed diagnostic `0.5 × locked C1 + 0.5 × locked H3` probability average, with no weight search.

The any-model-correct oracle is descriptive only and is never an advancement comparator.

### A5. Required metrics

The source-LODO and, conditionally, five-fold reports must include:

- accuracy, BAcc, AUC, sensitivity, specificity, TN, FP, FN, TP;
- all six subtype counts, correct counts, accuracy, and mean `p_high`;
- B1 and B2 exact counts;
- each held source’s BAcc, sensitivity, specificity, and confusion matrix;
- every source×subtype count and accuracy, with a dedicated third-batch-B2 row;
- same-case rescue, harm, net rescue, and McNemar discordant counts versus H3 and C2;
- exact-fusion versus each new-head control;
- exact-fusion versus same-source derangement;
- 20,000-replicate paired bootstrap deltas, sampling within `source_dataset × binary-risk` strata;
- minimum-source BAcc;
- parameter count, best epoch, wall time, peak GPU allocation, peak resident memory, and peak new disk usage.

All confidence intervals are percentile 95% intervals and are explicitly labeled as repeated-cohort stability diagnostics rather than independent confirmatory intervals.

### A6. Advancement gates

#### Gate 0: integrity and feasibility

All conditions are mandatory:

- 591 unique aligned case IDs; expected 368/223 risk counts and six subtype totals;
- all image paths accessible;
- all three C1 and H3 source-LODO checkpoints present;
- C1 pretrained cache resolves offline to exactly one immutable weight snapshot and is SHA-256 locked;
- PE checkpoint SHA-256 equals `47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1`; fileciteturn65file0L26-L36
- extracted branch probabilities reproduce locked C1/H3 files within `1e-5`;
- no missing, duplicate, nonfinite, or misaligned embedding;
- fixed batch size 1 extraction fits the RTX 4090 without changing resolution or architecture.

Any failure stops the experiment before classifier training.

#### Primary source-LODO gates

All gates below must pass simultaneously.

| Gate | Exact requirement | Rationale |
| --- | --- | --- |
| P1 Coverage | `591/591`, threshold `0.5` | Preserves the required endpoint. |
| P2 Overall BAcc | `≥0.7739` | At least +0.0200 over H3 0.7539. |
| P3 Sensitivity | `TP ≥164/223 = 0.7354` | No point loss from C2; requires at least 12 additional high-risk correct cases versus H3’s 152/223. |
| P4 Specificity | `TN ≥299/368 = 0.8125` | Together with P3 guarantees BAcc ≥0.77396 while allowing at most five low-risk losses versus H3’s 304 TN. |
| P5 B1 | `correct ≥40/62 = 0.6452` | Preserve H3’s B1 result. |
| P6 B2 | `correct ≥59/89 = 0.6629` | Recover at least C2’s B2 result; +7 correct versus H3. |
| P7 Third-batch B2 | `correct ≥18/29 = 0.6207` | +5 correct versus H3’s 13/29 and at least C2’s locked count. |
| P8 Held-source direction | ΔBAcc versus H3 `>0` in at least 2/3 sources | Prevent one-batch dominance. |
| P9 Held-source safety | no source ΔBAcc `<−0.0200`; minimum-source BAcc `≥0.7381` | No new worst-source regression. |
| P10 Branch necessity | exact fusion BAcc `≥ max(C1_ONLY_PADDED,H3_ONLY_PADDED)+0.0100` | Requires joint evidence beyond either branch with the same head. |
| P11 Source-preserving control | exact fusion minus deranged BAcc `≥0.0100`, positive in ≥2/3 held sources, paired 95% CI lower bound `>0` | Requires same-case evidence rather than source-level feature marginals. |
| P12 H3 paired stability | exact fusion minus locked H3 point ΔBAcc `≥0.0200` and paired 95% CI lower bound `>0` | Excludes a trivial or unstable nominal gain. |
| P13 Boundary net rescue | within B1+B2, net correct gain versus H3 `≥7`, with neither P5 nor P6 failing | Prevents another B1-up/B2-down trade. |

No gate can compensate for another. AUC improvement does not rescue a BAcc, sensitivity, B2, source, or control failure.

#### Secondary five-fold gates

Five-fold is not started unless every primary source-LODO gate passes. The architecture and all optimization settings remain identical; only the existing fold-specific upstream C1/H3 five-fold checkpoints replace the source-LODO checkpoints.

All must pass:

- coverage `591/591`, threshold 0.5;
- BAcc `≥0.7903` (no decline greater than 0.0100 from H3’s 0.8003);
- sensitivity `≥176/223 = 0.7892`;
- specificity `≥285/368 = 0.7745`;
- B1 `≥40/62`; B2 `≥60/89`;
- no individual test fold BAcc `<0.7000`;
- exact fusion exceeds each same-capacity branch by `≥0.0100`;
- exact fusion exceeds same-source derangement by `≥0.0100`, with paired 95% CI lower bound `>0`.

#### Conditional confirmation seed

Seed `20260715` is run on **source-LODO only** after all primary and secondary gates pass. It reuses the immutable embeddings and changes no other value.

Confirmation requires:

- all P1–P13 gates to pass again;
- the mean source-LODO ΔBAcc versus H3 across the two seeds to be `≥0.0200`;
- no seed ensembling and no selection of the better seed.

The primary seed remains the designated result.

### A7. Execution plan

#### Repository files to add

No existing result or implementation file is modified. Add exactly:

```text
thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/
  reports/H8_C1_H3_DIRECT_CASE_FUSION_PREREGISTRATION_20260714.md
  scripts/lock_task7_h8_assets_20260714.py
  scripts/extract_task7_h8_fold_embeddings_20260714.py
  scripts/run_task7_h8_direct_case_fusion_20260714.py
  scripts/analyze_task7_h8_direct_case_fusion_20260714.py
  scripts/run_task7_h8_direct_case_fusion_queue_20260714.sh
```

After execution, add only one aggregate result report:

```text
reports/H8_C1_H3_DIRECT_CASE_FUSION_RESULTS_20260714.md
```

#### Immutable server inputs

```text
REGISTRY=/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv
SPLIT=/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv

C1_LODO_ROOT=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711
C1_LODO_PRED=$C1_LODO_ROOT/oof_predictions.csv
C2_LODO_PRED=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion/lodo_predictions.csv
H3_LODO_ROOT=/workspace/thymic_project/experiments/h3_representation_renewal_20260713/h3b_runs/pe_spatial_l14_448/source_lodo
H3_LODO_PRED=$H3_LODO_ROOT/oof_predictions.csv

C1_5F_ROOT=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/347_siglipl512_localpyramid6_gated_fivefold_cw_20260711
C1_5F_PRED=$C1_5F_ROOT/oof_predictions.csv
C2_5F_PRED=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion/oof_predictions.csv
H3_5F_ROOT=/workspace/thymic_project/experiments/h3_representation_renewal_20260713/h3b_runs/pe_spatial_l14_448/fivefold
H3_5F_PRED=$H3_5F_ROOT/oof_predictions.csv

PE_CKPT=/root/model_weights/modelscope/facebook/PE-Spatial-L14-448/PE-Spatial-L14-448.pt
PE_SOURCE=/root/third_party/perception_models_3e352cca
C1_CACHE_ROOT=/root/.cache/huggingface/hub/models--timm--vit_large_patch16_siglip_512.v2_webli
```

Expected fold checkpoints are:

```text
$C1_LODO_ROOT/fold_{1,2,3}/best_model.pt
$H3_LODO_ROOT/fold_{1,2,3}/best_head.pt
$C1_5F_ROOT/fold_{1,2,3,4,5}/best_model.pt
$H3_5F_ROOT/fold_{1,2,3,4,5}/best_head.pt
```

The C1 snapshot file beneath `C1_CACHE_ROOT` is not assumed from GitHub. The asset-lock script follows the cache’s `refs/main`, requires exactly one resolved snapshot weight, and records its absolute realpath, byte count, and SHA-256. Zero, ambiguous, or broken resolution exits with failure and forbids a download.

#### Exact primary commands

```bash
set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PY=/root/miniconda3/envs/thymic_baseline/bin/python
CODE=/root/task7_h8_20260714
OUT=/workspace/thymic_project/experiments/h8_c1_h3_direct_case_fusion_20260714
mkdir -p "$OUT"/{locks,source_lodo,logs}

REGISTRY=/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv
SPLIT=/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv
C1R=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711
C2P=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion/lodo_predictions.csv
H3R=/workspace/thymic_project/experiments/h3_representation_renewal_20260713/h3b_runs/pe_spatial_l14_448/source_lodo
PE=/root/model_weights/modelscope/facebook/PE-Spatial-L14-448/PE-Spatial-L14-448.pt

"$PY" "$CODE/lock_task7_h8_assets_20260714.py" \
  --evidence-commit ff81fe4c98fd5f1b9d2bcfd53f4379e5741e38e8 \
  --registry-csv "$REGISTRY" \
  --split-csv "$SPLIT" \
  --c1-root "$C1R" --c1-predictions "$C1R/oof_predictions.csv" \
  --c2-predictions "$C2P" \
  --h3-root "$H3R" --h3-predictions "$H3R/oof_predictions.csv" \
  --c1-model-name vit_large_patch16_siglip_512.v2_webli \
  --c1-cache-root /root/.cache/huggingface/hub/models--timm--vit_large_patch16_siglip_512.v2_webli \
  --pe-checkpoint "$PE" \
  --expected-pe-sha256 47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1 \
  --pe-source-root /root/third_party/perception_models_3e352cca \
  --expected-pe-source-revision 3e352cca660658d4b5c90f42a7808b11469e4c66 \
  --output-manifest "$OUT/locks/source_lodo_assets.json"

"$PY" "$CODE/extract_task7_h8_fold_embeddings_20260714.py" \
  --asset-manifest "$OUT/locks/source_lodo_assets.json" \
  --split-mode source_lodo \
  --views whole,crop,crop_q0,crop_q1,crop_q2,crop_q3 \
  --c1-image-size 512 --h3-image-size 448 \
  --batch-size 1 --num-workers 0 --device cuda --seed 20260714 \
  --output-dir "$OUT/source_lodo/embeddings"

"$PY" "$CODE/run_task7_h8_direct_case_fusion_20260714.py" \
  --embedding-manifest "$OUT/source_lodo/embeddings/embedding_manifest.json" \
  --split-csv "$SPLIT" --split-mode source_lodo \
  --configuration H8_C1_H3_CONCAT_MLP16 \
  --hidden-dim 16 --dropout 0.10 \
  --epochs 80 --patience 12 --batch-size 32 \
  --lr 0.0003 --weight-decay 0.0001 --grad-clip 5.0 \
  --seed 20260714 --device cuda \
  --output-dir "$OUT/source_lodo/primary_seed20260714"

"$PY" "$CODE/analyze_task7_h8_direct_case_fusion_20260714.py" \
  --stage source_lodo \
  --run-dir "$OUT/source_lodo/primary_seed20260714" \
  --c1-predictions "$C1R/oof_predictions.csv" \
  --c2-predictions "$C2P" \
  --h3-predictions "$H3R/oof_predictions.csv" \
  --bootstrap-replicates 20000 --bootstrap-seed 20260714 \
  --enforce-gates \
  --output-dir "$OUT/source_lodo/aggregate"
```

The analyzer must exit nonzero and write `FINAL_DECISION.txt` containing `STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT` on any failed gate. The queue script proceeds to five-fold only when `gate_decision.json` contains an exact `"all_primary_gates_pass": true` and its own SHA-256 matches the sidecar.

The five-fold command is the same locked sequence with `C1_5F_ROOT`, `H3_5F_ROOT`, `split-mode fivefold`, and output under `$OUT/fivefold/`. The confirmation command reuses the source-LODO embedding manifest, changes only `--seed 20260715`, and is forbidden unless the five-fold gate file passes.

#### Runtime, GPU, and storage budget

Planning estimate, not a measured result:

- asset lock and two-case smoke: 15–30 minutes;
- primary dual-backbone streaming extraction: 4–8 hours on one RTX 4090;
- all source-LODO heads and controls: under 30 minutes;
- bootstrap/reporting: under 15 minutes;
- conditional five-fold extraction and heads: another 4–8 hours;
- conditional confirmation seed: under 30 minutes because embeddings are reused;
- maximum total if every stage advances: approximately 9–17 wall-clock hours.

Constraints:

- C1 and H3 encoders are never resident on GPU simultaneously;
- extraction batch size is fixed at one;
- target peak allocated GPU memory is `<22 GiB` and must be recorded;
- new disk usage ceiling is `1 GiB`;
- expected embedding data are under 20 MiB; all checkpoints, predictions, logs, and aggregate outputs should remain under 250 MiB;
- no dense multi-gigabyte feature bank is reconstructed.

If fixed batch-one sequential extraction still exceeds memory or the 1-GiB new-storage ceiling, the experiment stops. Resolution, token count, or architecture may not be changed as a workaround.

#### Interruption and recovery

- Each case is written to `case_shards/<opaque_hash>.npz.tmp`, fsynced, atomically renamed, and accompanied by a SHA-256 record.
- Restart skips a shard only after validating its hash, case/fold cardinality, finite values, and expected dimensions.
- Raw tokens are never checkpointed.
- A partially trained fold/control is deleted and restarted from epoch 1 with the same seed; optimizer-state resume is forbidden.
- Aggregate CSV/JSON files are written to temporary files and atomically renamed.
- Every stage writes `running`, `complete`, or `failed rc=<code>` status and a manifest hash.

#### GitHub versus server-only outputs

Allowed in GitHub:

- preregistration and source code;
- package/code/asset hashes with symbolic asset names;
- aggregate metrics, subtype/source counts, bootstrap summaries, gate table;
- parameter/runtime/storage summary;
- final interpretation and stop/go decision.

Server-only:

- images and image paths;
- registry rows, case IDs, original case IDs, and fold membership;
- C1/H3 embeddings and masks;
- per-case predictions, rescue/harm rows, and derangement maps;
- model weights/checkpoints and raw training histories containing identifiers.

## 7. **Evidence and literature table**

| Evidence | Type | Material finding | Decision implication |
| --- | --- | --- | --- |
| Plateau root-cause report | Repository, direct | H3 source-LODO BAcc 0.7539; six-model average 0.7602; confidence selector 0.7462; oracle 0.9006. fileciteturn69file0L13-L25 | Oracle diversity exists, but output combination is not the missing capability. |
| Oracle decomposition | Repository, direct | PE-only oracle 0.8462; error overlap remains high; averaging gain over H3 is small and unstable. fileciteturn69file0L77-L86 | Cross-family diversity is the only plausible remaining fusion target. |
| Equal-fusion implementation | Repository, direct | Reads OOF/LODO probability columns and uses means. fileciteturn74file0L98-L138 | Does not cover direct C1–H3 feature fusion. |
| qkvb/ROI meta-fusion | Repository, direct | Builds probability/logit statistics and ROI numeric columns. fileciteturn75file0L66-L111 fileciteturn75file0L137-L146 | Behavior/meta correction is already covered and excluded. |
| C1 MoE implementation | Repository, direct | Experts and gate all receive one C1 pooled embedding. fileciteturn73file0L48-L64 | Not cross-representation fusion. |
| H1 sequential experts | Repository, direct | Every expert consumes frozen C1 six-view dense features. fileciteturn76file0L13-L22 | Specialist failure does not answer the C1–H3 joint-feature question. |
| B1 cascade | Repository, direct | Stage-2 is a SigLIP native-tile reader; M3 fuses stage logits/probabilities and fails source-LODO. fileciteturn77file0L42-L78 | Same-family cascade and output fusion are not the proposed mechanism. |
| H3 | Repository, direct | PE dense tokens reach 0.8003 five-fold but 0.7539 source-LODO; B1 improves while B2 and sensitivity fall. fileciteturn64file0L56-L85 | H3 supplies a distinct candidate representation but is not independently robust enough. |
| H5 | Repository, direct | First- and second-order features are both computed from PE tokens before classification. fileciteturn52file0L119-L206 | True feature fusion exists only within PE, not across C1 and H3. |
| Source-frequency audit | Repository, direct | Source effect exceeds risk effect across all fixed frequency features. fileciteturn70file0L3-L11 | Source-preserving negative control is mandatory. |
| Multi-image audit | Repository, direct | 574/591 cases are single-image; only 17 have a second image. fileciteturn70file0L74-L82 | Fusion cannot recover absent views; scope remains one-photo evidence. |
| Local asset audit | Repository, direct | PE checkpoint/source are local and hash-locked; dense bank may be streamed rather than rebuilt. fileciteturn65file0L26-L42 | H8 is technically feasible subject to preflight. |
| WHO/ITMIG/Zucali sources | Primary medical literature cited by project | Histological categories and their reproducibility limitations define an endpoint not directly equivalent to a gross-photo object class. fileciteturn63file0L20-L25 | Positive results must be framed as risk prediction from gross appearance, not automated histologic truth. |
| Zech et al., PLOS Medicine 2018 | Primary medical-imaging study | Hospital identity and site-specific prevalence can inflate internal performance and fail externally. citeturn986992view0 | Per-source LODO and source-preserving controls are required. |
| Badgeley et al. | Primary medical-imaging study | Scanner/process variables were highly predictable; matching confounders sharply reduced diagnostic performance. citeturn223971view0 | Image-derived does not automatically mean disease-derived. |
| Gulrajani & Lopez-Paz, DomainBed | Primary methods paper | Domain-generalization model selection is nontrivial and must be prespecified. citeturn223971view1 | No adaptive fusion or seed search after H8. |

## 8. **Exact next actions and hard stopping rule**

1. Commit the H8 preregistration and five implementation scripts **before** reading any H8 metric.
2. Upload only those small source files to `/root/task7_h8_20260714`; do not download any package or checkpoint.
3. Run the offline asset lock. If any checkpoint, cache, hash, path, count, free-space, or reproduction check fails, write the stop decision and end the project’s current-cohort classifier track.
4. Run primary source-LODO extraction, the one locked candidate, and its fixed controls.
5. Analyze all P1–P13 gates. Do not inspect external cohorts.
6. Only after all P gates pass, run the locked five-fold stage and its secondary gates.
7. Only after all secondary gates pass, run source-LODO seed `20260715` and apply the confirmation rule.
8. Do not ensemble seeds. Do not tune threshold. Do not change hidden width, dropout, sampler, loss, resolution, view set, pooling, or pair control.
9. If all stages pass, report H8 as an internally supported cross-representation mechanism with explicit source and repeated-cohort limitations. The 108- and 162-case cohorts remain consumed retrospective stress tests and cannot upgrade the claim to fresh external confirmation.
10. If any stage fails, retain C1, C2, and H3 as the honest locked comparators, with H3 as the best direct single-model BAcc and C2 as the higher-sensitivity/B2 comparator. No further classifier search follows.

### Hard stopping rule

> **At the first failed integrity check or failed preregistered gate, write `STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT`, do not run the next stage, do not consult the consumed external cohorts for rescue, and do not initiate any additional seed, threshold, fusion, routing, pooling, loss, augmentation, or architecture experiment on these 591 cases.**

### Scope of this completed task

This audit and preregistration are complete. The H8 server experiment itself was not executed here because the case images, checkpoints, and GPU environment remain on the project server; no performance result has been invented.

# H8 C1–H3 Direct Case-Embedding Fusion — Immutable Preregistration

Evidence lock: `ScottBlizzard/tarmac@ff81fe4c98fd5f1b9d2bcfd53f4379e5741e38e8`  
Preregistration date: 2026-07-14  
Primary configuration: `H8_C1_H3_DIRECT_CASE_EMBEDDING_FUSION_20260714`

## 5. **Decision: one locked experiment or stop**

### Decision

Run **one** experiment: `H8_C1_H3_DIRECT_CASE_EMBEDDING_FUSION_20260714`.

### A1. Changed assumption

H8 changes exactly one assumption: instead of combining final probabilities or placing several heads on one representation, it tests whether the **aligned pre-classifier case embeddings from two genuinely different image encoders** contain jointly useful Task7 evidence. C1/C2, F2, and the internal fusion searches combined outputs; H1 and MoE reused C1; the B1 cascade fused stage behavior from SigLIP-family readers; H3-H7 and H5’s first/second-order fusion remained within PE. A positive H8 result would be direct image capability only if a fold-clean C1+H3 head beats same-capacity branch heads, locked H3, and a within-source pair-destroyed control while improving at least two held sources and jointly preserving B1 and B2. The hypothesis is closed if **any** preregistered gate fails; the exact closure statement is `STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT`.

### Why immediate stopping is not yet warranted

Stopping now would leave a specific repository gap untested: no inspected code path sends C1 and H3 image-derived case tensors to the same disease head. The required assets are locally available or regenerable without a download, and the experiment can be implemented with a tiny head and streamed embeddings rather than multi-gigabyte banks. fileciteturn65file0L26-L42 fileciteturn47file0L53-L75

### Why this is terminal

The experiment uses one architecture, one hidden width, one loss, one learning rate, one primary seed, fixed threshold 0.5, and fixed gates. There is no backup fusion family. A negative result eliminates the remaining direct C1–H3 case-embedding hypothesis under the fixed data boundary; more search would be adaptive reuse rather than a distinct scientific test.

## 6. **Complete branch-A preregistration**

### A2. Immutable model specification

#### Cohort and views

- Cases: exactly 591 unique registry rows.
- Task: A/AB/B1 = 0; B2/B3/TC = 1.
- Coverage: 591/591.
- Decision threshold: exactly 0.5.
- C1 views: `whole,crop,crop_q0,crop_q1,crop_q2,crop_q3`, 512×512, deterministic existing preprocessing.
- H3 views: the same six semantic views, 448×448, deterministic existing H3 preprocessing.
- No second-image bag, text, gross-concept table, source label, confidence, probability, logit, prediction, margin, or correctness feature is provided to the candidate head.

#### Frozen tensors and embeddings

For outer fold `f` and case `i`:

- C1 dense tensor `T^C_{if} ∈ R^(6×1024×1024)` from frozen `vit_large_patch16_siglip_512.v2_webli`; frozen fold-specific C1 gated pooler produces pre-classifier `c_if ∈ R^256`.
- H3 dense tensor `T^H_{if} ∈ R^(6×1024×1024)` with valid-token mask `M^H_i ∈ {0,1}^(6×1024)` from frozen PE-Spatial-L14-448; frozen fold-specific H3 masked-gated pooler produces pre-classifier `h_if ∈ R^128`.
- The expected H3 bank shape and hashes are already locked in the H7 implementation. fileciteturn60file0L40-L47
- The candidate never stores either dense tensor. It streams one encoder at a time, computes all fold-specific pooled embeddings for the case, writes the small embedding shard atomically, and discards tokens.
- Both pretrained encoders and all existing C1/H3 pooling-head parameters remain frozen.

Before training, the extractor must reproduce each locked branch probability from its extracted embedding plus the locked classifier with maximum absolute error `≤1e-5`. Failure for any case/fold is an integrity failure, not a tuning opportunity.

#### Candidate equation

For each case:

```text
c̄ = c / max(||c||₂, 1e-6)
h̄ = h / max(||h||₂, 1e-6)
x = concat(c̄, h̄) ∈ R^384
u = GELU(W₁x + b₁),              W₁ ∈ R^(16×384)
z = LayerNorm₁₆(u)
logits = W₂ Dropout₀.₁₀(z) + b₂, W₂ ∈ R^(2×16)
p_high = softmax(logits)[1]
```

There is no branch gate, attention block, residual MLP, calibration layer, temperature, or learned threshold.

#### Trainable parameters

| Layer | Parameters |
| --- | ---: |
| `Linear(384,16)` | 6,160 |
| affine `LayerNorm(16)` | 32 |
| `Linear(16,2)` | 34 |
| **Total** | **6,226** |

Initialization is fixed:

- `W1`, `W2`: Xavier uniform, gain 1.0;
- all linear biases: zero;
- LayerNorm scale: one; bias: zero;
- deterministic seed: `20260714`.

#### Optimization

| Item | Locked value |
| --- | --- |
| Loss | two-class cross-entropy |
| Sampler | inverse-frequency `source_dataset × binary-risk` sampler over **outer-training cases only**, replacement, `num_samples=len(train)` |
| Additional class weights | none |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Batch size | `32` |
| Epoch ceiling | `80` |
| Early-stopping patience | `12` |
| Validation criterion | balanced accuracy at threshold 0.5 |
| Tie rule | earliest epoch |
| Scheduler | cosine annealing, `T_max=80` |
| Gradient clipping | L2 norm `5.0` |
| Precision | FP32 head; frozen extraction may use the existing safe autocast mode |
| Primary seed | `20260714` |
| Conditional confirmation seed | `20260715`, allowed only after all primary and secondary gates pass |
| Determinism | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN benchmark off, deterministic algorithms on |

### A3. Leakage-safe source-LODO

Source-LODO is run first and is the primary endpoint. The existing fold convention is retained exactly: held source is indexed in the order `batch1,batch2,third_batch`; validation uses `val_fold=(fold_id mod 5)+1` among non-held cases. fileciteturn66file0L23-L25 fileciteturn68file0L39-L62

| Outer fold | Held test source | Validation master fold within remaining sources |
| --- | --- | ---: |
| 1 | batch1 | 2 |
| 2 | batch2 | 3 |
| 3 | third_batch | 4 |

For each fold:

1. Use only the pre-existing fold-specific C1 and H3 source-LODO checkpoints whose supervised fitting excluded the held source.
2. Extract frozen embeddings for train, validation, and held-test cases without any target-source normalization or adaptation.
3. Fit the H8 head only on outer-training cases.
4. Calculate source×risk sampler weights only from outer-training cases.
5. Early-stop only on the predefined validation subset from the remaining sources.
6. Apply the chosen state once to the held source.
7. Emit exactly one prediction per case at threshold 0.5.
8. Merge the three held-source predictions to a 591-case OOF table.

No target-source statistic is allowed to alter the decision function. Samplewise L2 normalization and LayerNorm are permitted because they do not estimate cohort statistics.

### A4. Required controls

All learned controls use the same 6,226-parameter code path, optimizer, sampler, seed, folds, and early stopping.

| Tag | Input to the identical 384→16→2 head | Purpose |
| --- | --- | --- |
| `C1_ONLY_PADDED` | `concat(c̄, zeros(128))` | Tests C1 with the same new head capacity. |
| `H3_ONLY_PADDED` | `concat(zeros(256), h̄)` | Tests H3 with the same new head capacity. |
| `C1_H3_EXACT` | `concat(c̄_i, h̄_i)` | **Only advancing configuration.** |
| `C1_H3_SAME_SOURCE_DERANGED` | `concat(c̄_i, h̄_π(i))` | Source-preserving negative control. |

The derangement is immutable and label-free:

- separately within each `split × source_dataset` group;
- sort cases by `SHA256("H8|fold|split|case_id")`;
- cyclically shift the H3 order by one;
- require group size at least two;
- never use risk or subtype labels to form the permutation;
- create separate train, validation, and held-test permutations;
- store the mapping server-only.

This control preserves the source distribution and marginal H3 feature distribution but destroys same-case alignment. Exact fusion must beat it; otherwise the putative gain is not attributable to same-case cross-family evidence.

Additional fixed, non-advancing comparators:

- locked C1 predictions on the exact cases;
- locked C2 predictions on the exact cases;
- locked H3 predictions on the exact cases;
- fixed diagnostic `0.5 × locked C1 + 0.5 × locked H3` probability average, with no weight search.

The any-model-correct oracle is descriptive only and is never an advancement comparator.

### A5. Required metrics

The source-LODO and, conditionally, five-fold reports must include:

- accuracy, BAcc, AUC, sensitivity, specificity, TN, FP, FN, TP;
- all six subtype counts, correct counts, accuracy, and mean `p_high`;
- B1 and B2 exact counts;
- each held source’s BAcc, sensitivity, specificity, and confusion matrix;
- every source×subtype count and accuracy, with a dedicated third-batch-B2 row;
- same-case rescue, harm, net rescue, and McNemar discordant counts versus H3 and C2;
- exact-fusion versus each new-head control;
- exact-fusion versus same-source derangement;
- 20,000-replicate paired bootstrap deltas, sampling within `source_dataset × binary-risk` strata;
- minimum-source BAcc;
- parameter count, best epoch, wall time, peak GPU allocation, peak resident memory, and peak new disk usage.

All confidence intervals are percentile 95% intervals and are explicitly labeled as repeated-cohort stability diagnostics rather than independent confirmatory intervals.

### A6. Advancement gates

#### Gate 0: integrity and feasibility

All conditions are mandatory:

- 591 unique aligned case IDs; expected 368/223 risk counts and six subtype totals;
- all image paths accessible;
- all three C1 and H3 source-LODO checkpoints present;
- C1 pretrained cache resolves offline to exactly one immutable weight snapshot and is SHA-256 locked;
- PE checkpoint SHA-256 equals `47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1`; fileciteturn65file0L26-L36
- extracted branch probabilities reproduce locked C1/H3 files within `1e-5`;
- no missing, duplicate, nonfinite, or misaligned embedding;
- fixed batch size 1 extraction fits the RTX 4090 without changing resolution or architecture.

Any failure stops the experiment before classifier training.

#### Primary source-LODO gates

All gates below must pass simultaneously.

| Gate | Exact requirement | Rationale |
| --- | --- | --- |
| P1 Coverage | `591/591`, threshold `0.5` | Preserves the required endpoint. |
| P2 Overall BAcc | `≥0.7739` | At least +0.0200 over H3 0.7539. |
| P3 Sensitivity | `TP ≥164/223 = 0.7354` | No point loss from C2; requires at least 12 additional high-risk correct cases versus H3’s 152/223. |
| P4 Specificity | `TN ≥299/368 = 0.8125` | Together with P3 guarantees BAcc ≥0.77396 while allowing at most five low-risk losses versus H3’s 304 TN. |
| P5 B1 | `correct ≥40/62 = 0.6452` | Preserve H3’s B1 result. |
| P6 B2 | `correct ≥59/89 = 0.6629` | Recover at least C2’s B2 result; +7 correct versus H3. |
| P7 Third-batch B2 | `correct ≥18/29 = 0.6207` | +5 correct versus H3’s 13/29 and at least C2’s locked count. |
| P8 Held-source direction | ΔBAcc versus H3 `>0` in at least 2/3 sources | Prevent one-batch dominance. |
| P9 Held-source safety | no source ΔBAcc `<−0.0200`; minimum-source BAcc `≥0.7381` | No new worst-source regression. |
| P10 Branch necessity | exact fusion BAcc `≥ max(C1_ONLY_PADDED,H3_ONLY_PADDED)+0.0100` | Requires joint evidence beyond either branch with the same head. |
| P11 Source-preserving control | exact fusion minus deranged BAcc `≥0.0100`, positive in ≥2/3 held sources, paired 95% CI lower bound `>0` | Requires same-case evidence rather than source-level feature marginals. |
| P12 H3 paired stability | exact fusion minus locked H3 point ΔBAcc `≥0.0200` and paired 95% CI lower bound `>0` | Excludes a trivial or unstable nominal gain. |
| P13 Boundary net rescue | within B1+B2, net correct gain versus H3 `≥7`, with neither P5 nor P6 failing | Prevents another B1-up/B2-down trade. |

No gate can compensate for another. AUC improvement does not rescue a BAcc, sensitivity, B2, source, or control failure.

#### Secondary five-fold gates

Five-fold is not started unless every primary source-LODO gate passes. The architecture and all optimization settings remain identical; only the existing fold-specific upstream C1/H3 five-fold checkpoints replace the source-LODO checkpoints.

All must pass:

- coverage `591/591`, threshold 0.5;
- BAcc `≥0.7903` (no decline greater than 0.0100 from H3’s 0.8003);
- sensitivity `≥176/223 = 0.7892`;
- specificity `≥285/368 = 0.7745`;
- B1 `≥40/62`; B2 `≥60/89`;
- no individual test fold BAcc `<0.7000`;
- exact fusion exceeds each same-capacity branch by `≥0.0100`;
- exact fusion exceeds same-source derangement by `≥0.0100`, with paired 95% CI lower bound `>0`.

#### Conditional confirmation seed

Seed `20260715` is run on **source-LODO only** after all primary and secondary gates pass. It reuses the immutable embeddings and changes no other value.

Confirmation requires:

- all P1–P13 gates to pass again;
- the mean source-LODO ΔBAcc versus H3 across the two seeds to be `≥0.0200`;
- no seed ensembling and no selection of the better seed.

The primary seed remains the designated result.

### A7. Execution plan

#### Repository files to add

No existing result or implementation file is modified. Add exactly:

```text
thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/
  reports/H8_C1_H3_DIRECT_CASE_FUSION_PREREGISTRATION_20260714.md
  scripts/lock_task7_h8_assets_20260714.py
  scripts/extract_task7_h8_fold_embeddings_20260714.py
  scripts/run_task7_h8_direct_case_fusion_20260714.py
  scripts/analyze_task7_h8_direct_case_fusion_20260714.py
  scripts/run_task7_h8_direct_case_fusion_queue_20260714.sh
```

After execution, add only one aggregate result report:

```text
reports/H8_C1_H3_DIRECT_CASE_FUSION_RESULTS_20260714.md
```

#### Immutable server inputs

```text
REGISTRY=/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv
SPLIT=/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv

C1_LODO_ROOT=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711
C1_LODO_PRED=$C1_LODO_ROOT/oof_predictions.csv
C2_LODO_PRED=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion/lodo_predictions.csv
H3_LODO_ROOT=/workspace/thymic_project/experiments/h3_representation_renewal_20260713/h3b_runs/pe_spatial_l14_448/source_lodo
H3_LODO_PRED=$H3_LODO_ROOT/oof_predictions.csv

C1_5F_ROOT=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/347_siglipl512_localpyramid6_gated_fivefold_cw_20260711
C1_5F_PRED=$C1_5F_ROOT/oof_predictions.csv
C2_5F_PRED=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion/oof_predictions.csv
H3_5F_ROOT=/workspace/thymic_project/experiments/h3_representation_renewal_20260713/h3b_runs/pe_spatial_l14_448/fivefold
H3_5F_PRED=$H3_5F_ROOT/oof_predictions.csv

PE_CKPT=/root/model_weights/modelscope/facebook/PE-Spatial-L14-448/PE-Spatial-L14-448.pt
PE_SOURCE=/root/third_party/perception_models_3e352cca
C1_CACHE_ROOT=/root/.cache/huggingface/hub/models--timm--vit_large_patch16_siglip_512.v2_webli
```

Expected fold checkpoints are:

```text
$C1_LODO_ROOT/fold_{1,2,3}/best_model.pt
$H3_LODO_ROOT/fold_{1,2,3}/best_head.pt
$C1_5F_ROOT/fold_{1,2,3,4,5}/best_model.pt
$H3_5F_ROOT/fold_{1,2,3,4,5}/best_head.pt
```

The C1 snapshot file beneath `C1_CACHE_ROOT` is not assumed from GitHub. The asset-lock script follows the cache’s `refs/main`, requires exactly one resolved snapshot weight, and records its absolute realpath, byte count, and SHA-256. Zero, ambiguous, or broken resolution exits with failure and forbids a download.

#### Exact primary commands

```bash
set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PY=/root/miniconda3/envs/thymic_baseline/bin/python
CODE=/root/task7_h8_20260714
OUT=/workspace/thymic_project/experiments/h8_c1_h3_direct_case_fusion_20260714
mkdir -p "$OUT"/{locks,source_lodo,logs}

REGISTRY=/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv
SPLIT=/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv
C1R=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711
C2P=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion/lodo_predictions.csv
H3R=/workspace/thymic_project/experiments/h3_representation_renewal_20260713/h3b_runs/pe_spatial_l14_448/source_lodo
PE=/root/model_weights/modelscope/facebook/PE-Spatial-L14-448/PE-Spatial-L14-448.pt

"$PY" "$CODE/lock_task7_h8_assets_20260714.py" \
  --evidence-commit ff81fe4c98fd5f1b9d2bcfd53f4379e5741e38e8 \
  --registry-csv "$REGISTRY" \
  --split-csv "$SPLIT" \
  --c1-root "$C1R" --c1-predictions "$C1R/oof_predictions.csv" \
  --c2-predictions "$C2P" \
  --h3-root "$H3R" --h3-predictions "$H3R/oof_predictions.csv" \
  --c1-model-name vit_large_patch16_siglip_512.v2_webli \
  --c1-cache-root /root/.cache/huggingface/hub/models--timm--vit_large_patch16_siglip_512.v2_webli \
  --pe-checkpoint "$PE" \
  --expected-pe-sha256 47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1 \
  --pe-source-root /root/third_party/perception_models_3e352cca \
  --expected-pe-source-revision 3e352cca660658d4b5c90f42a7808b11469e4c66 \
  --output-manifest "$OUT/locks/source_lodo_assets.json"

"$PY" "$CODE/extract_task7_h8_fold_embeddings_20260714.py" \
  --asset-manifest "$OUT/locks/source_lodo_assets.json" \
  --split-mode source_lodo \
  --views whole,crop,crop_q0,crop_q1,crop_q2,crop_q3 \
  --c1-image-size 512 --h3-image-size 448 \
  --batch-size 1 --num-workers 0 --device cuda --seed 20260714 \
  --output-dir "$OUT/source_lodo/embeddings"

"$PY" "$CODE/run_task7_h8_direct_case_fusion_20260714.py" \
  --embedding-manifest "$OUT/source_lodo/embeddings/embedding_manifest.json" \
  --split-csv "$SPLIT" --split-mode source_lodo \
  --configuration H8_C1_H3_CONCAT_MLP16 \
  --hidden-dim 16 --dropout 0.10 \
  --epochs 80 --patience 12 --batch-size 32 \
  --lr 0.0003 --weight-decay 0.0001 --grad-clip 5.0 \
  --seed 20260714 --device cuda \
  --output-dir "$OUT/source_lodo/primary_seed20260714"

"$PY" "$CODE/analyze_task7_h8_direct_case_fusion_20260714.py" \
  --stage source_lodo \
  --run-dir "$OUT/source_lodo/primary_seed20260714" \
  --c1-predictions "$C1R/oof_predictions.csv" \
  --c2-predictions "$C2P" \
  --h3-predictions "$H3R/oof_predictions.csv" \
  --bootstrap-replicates 20000 --bootstrap-seed 20260714 \
  --enforce-gates \
  --output-dir "$OUT/source_lodo/aggregate"
```

The analyzer must exit nonzero and write `FINAL_DECISION.txt` containing `STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT` on any failed gate. The queue script proceeds to five-fold only when `gate_decision.json` contains an exact `"all_primary_gates_pass": true` and its own SHA-256 matches the sidecar.

The five-fold command is the same locked sequence with `C1_5F_ROOT`, `H3_5F_ROOT`, `split-mode fivefold`, and output under `$OUT/fivefold/`. The confirmation command reuses the source-LODO embedding manifest, changes only `--seed 20260715`, and is forbidden unless the five-fold gate file passes.

#### Runtime, GPU, and storage budget

Planning estimate, not a measured result:

- asset lock and two-case smoke: 15–30 minutes;
- primary dual-backbone streaming extraction: 4–8 hours on one RTX 4090;
- all source-LODO heads and controls: under 30 minutes;
- bootstrap/reporting: under 15 minutes;
- conditional five-fold extraction and heads: another 4–8 hours;
- conditional confirmation seed: under 30 minutes because embeddings are reused;
- maximum total if every stage advances: approximately 9–17 wall-clock hours.

Constraints:

- C1 and H3 encoders are never resident on GPU simultaneously;
- extraction batch size is fixed at one;
- target peak allocated GPU memory is `<22 GiB` and must be recorded;
- new disk usage ceiling is `1 GiB`;
- expected embedding data are under 20 MiB; all checkpoints, predictions, logs, and aggregate outputs should remain under 250 MiB;
- no dense multi-gigabyte feature bank is reconstructed.

If fixed batch-one sequential extraction still exceeds memory or the 1-GiB new-storage ceiling, the experiment stops. Resolution, token count, or architecture may not be changed as a workaround.

#### Interruption and recovery

- Each case is written to `case_shards/<opaque_hash>.npz.tmp`, fsynced, atomically renamed, and accompanied by a SHA-256 record.
- Restart skips a shard only after validating its hash, case/fold cardinality, finite values, and expected dimensions.
- Raw tokens are never checkpointed.
- A partially trained fold/control is deleted and restarted from epoch 1 with the same seed; optimizer-state resume is forbidden.
- Aggregate CSV/JSON files are written to temporary files and atomically renamed.
- Every stage writes `running`, `complete`, or `failed rc=<code>` status and a manifest hash.

#### GitHub versus server-only outputs

Allowed in GitHub:

- preregistration and source code;
- package/code/asset hashes with symbolic asset names;
- aggregate metrics, subtype/source counts, bootstrap summaries, gate table;
- parameter/runtime/storage summary;
- final interpretation and stop/go decision.

Server-only:

- images and image paths;
- registry rows, case IDs, original case IDs, and fold membership;
- C1/H3 embeddings and masks;
- per-case predictions, rescue/harm rows, and derangement maps;
- model weights/checkpoints and raw training histories containing identifiers.


## 8. **Exact next actions and hard stopping rule**

1. Commit the H8 preregistration and five implementation scripts **before** reading any H8 metric.
2. Upload only those small source files to `/root/task7_h8_20260714`; do not download any package or checkpoint.
3. Run the offline asset lock. If any checkpoint, cache, hash, path, count, free-space, or reproduction check fails, write the stop decision and end the project’s current-cohort classifier track.
4. Run primary source-LODO extraction, the one locked candidate, and its fixed controls.
5. Analyze all P1–P13 gates. Do not inspect external cohorts.
6. Only after all P gates pass, run the locked five-fold stage and its secondary gates.
7. Only after all secondary gates pass, run source-LODO seed `20260715` and apply the confirmation rule.
8. Do not ensemble seeds. Do not tune threshold. Do not change hidden width, dropout, sampler, loss, resolution, view set, pooling, or pair control.
9. If all stages pass, report H8 as an internally supported cross-representation mechanism with explicit source and repeated-cohort limitations. The 108- and 162-case cohorts remain consumed retrospective stress tests and cannot upgrade the claim to fresh external confirmation.
10. If any stage fails, retain C1, C2, and H3 as the honest locked comparators, with H3 as the best direct single-model BAcc and C2 as the higher-sensitivity/B2 comparator. No further classifier search follows.

### Hard stopping rule

> **At the first failed integrity check or failed preregistered gate, write `STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT`, do not run the next stage, do not consult the consumed external cohorts for rescue, and do not initiate any additional seed, threshold, fusion, routing, pooling, loss, augmentation, or architecture experiment on these 591 cases.**

### Scope of this completed task

This audit and preregistration are complete. The H8 server experiment itself was not executed here because the case images, checkpoints, and GPU environment remain on the project server; no performance result has been invented.

已完成逐文件、逐实现脚本的终局审计，并形成两份可直接归档的 Markdown 文档：

* [完整终局审计报告](sandbox:/mnt/data/FINAL_ORACLE_GAP_AUDIT_COMPLETED_20260714.md)
* [H8 独立不可变预注册](sandbox:/mnt/data/H8_C1_H3_DIRECT_CASE_FUSION_PREREGISTRATION_20260714.md)

## 1. **Executive verdict**

**决策：选择 Branch A，只运行一个终局实验。**

唯一仍具有科学可辩护性的实验是：

> **`H8_C1_H3_DIRECT_CASE_EMBEDDING_FUSION_20260714`**
> 冻结每个外层折对应的 C1 SigLIP-L 与 H3 PE-Spatial 图像模型，提取各自的病例级、分类器前嵌入，将二者直接拼接，并用唯一固定的 **6,226 参数低容量 Task7 疾病头**训练低危/高危分类器。

作出这一判断的依据是：

1. **现有 0.9006 是标签知情的 retrospective oracle，不是可部署性能。**六模型平均只有 0.7602 BAcc，选择最高置信度模型只有 0.7462；当前最佳单模型 H3 的 source-LODO BAcc 为 0.7539。
2. **跨表征错误互补确实存在。**去掉 C1 后 oracle 仍为 0.8925，而只保留共享 PE 表征的 H3/H5/H6/H7 时降至 0.8462，说明机会主要存在于不同表征家族之间，而不是继续优化 PE 家族内部的近邻头部。
3. **所要求的直接 C1–H3 特征级疾病头尚未被做过。**历史“融合”主要是概率平均、logit/预测元模型、同一表征上的 MoE，或 PE 内部一阶/二阶特征融合；没有发现把 C1 与 H3 病例张量交给同一个 Task7 头并进行完整 outer source-LODO 的实现。
4. **资产原则上可用且无需下载。**PE-Spatial 权重与官方源码在本地；大型 PE bank 虽已删除，但可逐病例流式提取。C1 过去也在 offline 模式下成功生成 run 348，但其当前缓存与折级 checkpoint 必须在训练前重新做存在性和 SHA-256 锁定。

这是**最后一次当前队列分类器实验**。任何预注册门槛失败，立即输出：

> **STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT**

此后不得继续搜索 hidden width、loss、seed、threshold、fusion、router、pooling、augmentation 或 architecture。

---

## 2. **Repository audit of cross-representation fusion coverage**

| 审计问题                                            | 结论           | 证据                                                                                                                                                     |
| ----------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| C1 与 H3 的病例嵌入是否曾被拼接或联合注意并交给同一低容量疾病头？            | **没有发现。**    | C1 的 risk、boundary、subtype、sentinel 与 MoE 都共享同一个 C1 pooled embedding。  H3 单独形成 PE-only masked-gated embedding。H5 虽然是特征级融合，但两条分支都来自同一个 PE token tensor。 |
| 这种 C1–H3 模型是否做过 fully nested outer source-LODO？ | **没有。**      | C1 与 H3 分别有 source-LODO，但没有共同疾病头。现有 split 实现本身是可复用的：留出一个来源，验证只在其余来源内部完成。                                                                               |
| 历史 fusion 融合的是图像张量还是输出行为？                       | **主要是输出行为。** | 大规模内部搜索读取 prediction CSV 的 probability 列并取均值。  qkvb/ROI meta-fusion 使用概率、logit、range、统计量及 ROI 数值。                                                      |
| MoE 是否有不同图像表征？                                  | **没有。**      | 三个 MoE expert 与 gate 都读取同一个 C1 embedding。                                                                                                              |
| Cascade 是否等于 C1–H3 特征融合？                        | **不等于。**     | B1 cascade 的第二 reader 是同一 SigLIP 家族的 native-tile reader；M3 通过两个阶段的概率/logit 做 logistic fusion。                                                          |
| Oracle 是否主要是阈值问题？                               | **不是。**      | 即使在同一 591 例上事后看标签选全局阈值，H3 也只有 0.7730 BAcc；跨来源迁移阈值约为 0.76。                                                                                              |
| 资产是否可执行？                                        | **有条件可执行。**  | PE checkpoint 已锁定；dense bank 可流式重建。C1 需在严格 offline preflight 中定位唯一缓存快照并锁 hash，缺失时不得下载，直接停止。                                                            |

因此，下面四类机制虽然都曾被称为 fusion 或 specialist，但不能视为已覆盖 H8：

* 概率平均或多数投票；
* OOF prediction/logit 元模型；
* 在一个共同 embedding 上增加多个专家头；
* 在 PE 内部融合一阶与二阶纹理特征。

真正未覆盖的路径是：

> 两个独立图像表征家族、两个 fold-clean 病例嵌入、一个直接受 Task7 标签监督的低容量疾病头；模型输入中没有 probability、logit、margin、confidence、source、text 或 correctness。

---

## 3. **What the 0.9006 oracle does and does not prove**

### 它证明了什么

0.9006 证明六个模型的错误集合并不相同。尤其是，PE-only oracle 明显低于包含 C1/C2 与 PE 分支的 oracle，支持“跨表征家族存在一些互补视觉信息”的判断。

它也证明平台期不是单纯的阈值校准问题。H3 的同队列事后最优阈值只有 0.7730 BAcc，远低于 0.9006。

### 它没有证明什么

Oracle 的规则实际是“知道真实标签后，选择一个答对的模型”。因此它没有提供部署时可观察的选择变量，也没有证明：

* 当前单张照片足以达到 90%；
* 置信度能识别正确专家；
* 错误互补是病理形态而不是采集差异；
* 一个 gate 可以跨来源学习到正确路由；
* 被所有模型共同判错的病例能够由融合解决；
* source-LODO 结果等同于独立医院外部验证。

53/591 病例被六个模型全部判错，83/591 被至少五个模型判错；B1、B2，尤其 third-batch B2，是共同错误中心。

因此，0.9006 只能被解释为：

> 存在一次直接跨表征病例特征融合的可检验机会；它不是预期性能，也不是可训练标签。

要把这种机会认定为真实能力，H8 必须同时证明：

1. 超过 C1-only 与 H3-only 同容量头；
2. 超过锁定 H3 至少 0.02 BAcc；
3. 在至少两个 held source 同方向改善；
4. 保留 H3 的 B1 能力并恢复 C2 的 B2/敏感度；
5. 超过保留来源信息但破坏同病例配对的 negative control。

---

## 4. **Source-confounding and identifiability assessment**

### 项目内部直接证据

来源、风险和亚型并非平衡设计。来源与风险的 Cramer’s V 为 0.234，与六亚型的 Cramer’s V 为 0.455；有两个 source×subtype 单元为空，batch2-B3 只有四例。

固定 Haar 频率审计发现：

* 控制风险后，频率特征仍可预测来源；
* 控制来源后，风险预测弱得多；
* 156/156 个频率特征的来源偏效应大于风险偏效应。

同一亚型在不同来源中的模型行为也显著不同：

* batch1-B2 对 C1/C2 几乎是简单类；
* third-batch B2 是全部模型的主要高危漏诊来源；
* B1 的来源困难方向与 B2 相反。

此外，574/591 例只有一张照片，只有 17 例有第二张；所谓六视图均来自同一照片，无法补回未拍摄的切面、包膜界面或内部异质区。

### 文献支持

Zech 等发现，医学影像模型可以近乎完美地识别采集医院；当站点与疾病患病率耦合时，内部表现会被抬高，而优势不能外部迁移。([PLOS][1])

Badgeley 等进一步表明，扫描仪型号、品牌和医院流程变量可以从影像中被预测；在匹配患者及采集混杂后，疾病模型可降至接近随机。([arxiv.org][2])

这些研究不能直接证明本项目就是相同机制，但说明“输入来自图像”并不自动等于“输入代表疾病形态”。

### 可识别性判断

H8 足以作为**一次内部终局机制检验**，但不足以建立广义的“来源不变病理表征”结论。

其可识别性来自：

* held source 不参与任何监督拟合；
* sampler 只在其余来源内做 source×risk 平衡；
* 至少两个 held source 必须改善；
* 最差来源不能下降；
* B1 与 B2 必须共同满足病例数门槛；
* 同来源病例 derangement 保留来源分布，却破坏 C1/H3 同病例对齐。

derangement control 仍不能排除所有同病例采集伪影。因此即使 H8 通过，也只能声称：

> 在当前内部 acquisition-shift 代理协议下，两个图像表征的同病例特征提供了超过单分支及来源边际信息的预测证据。

不能声称它已经证明病理因果性或独立医院泛化。

此外，591 例已经反复参与方法开发。DomainBed 的核心方法学结论之一正是 domain generalization 的 model selection 本身必须被预先定义。([arxiv.org][3]) 因而 H8 的 bootstrap 区间只能作为稳定性 gate，不能包装成全新的独立确认性推断。

---

## 5. **Decision: one locked experiment or stop**

### 决策

运行且只运行：

> `H8_C1_H3_DIRECT_CASE_EMBEDDING_FUSION_20260714`

### Changed assumption

H8 改变的唯一假设是：**最终概率可能丢弃了跨表征联合判别所需的图像信息，而 C1 与 H3 的分类器前病例嵌入可能保留了这些信息。**

C1/C2、F2 和内部 fusion search 融合的是输出；H1 和 MoE 复用 C1；B1 cascade 使用 SigLIP-family 第二 reader 并在输出层融合；H3–H7 与 H5 的一阶/二阶融合均局限于 PE 家族。因此，它们没有回答“C1 与 H3 直接病例特征是否能形成一个更好的疾病边界”。

正结果只有在以下条件同时成立时才代表直接图像能力：

* exact C1+H3 超过同容量 C1-only 和 H3-only；
* 超过 H3 至少 0.02 source-LODO BAcc；
* 至少两个来源改善；
* 保留 B1 并恢复 B2；
* 超过同来源 deranged-pair control。

任一条件失败即关闭该假设，并停止当前队列分类器开发。

---

## 6. **Complete branch-A preregistration**

### A2. Immutable model specification

#### 输入

* 591 个唯一病例；
* A/AB/B1 = 0，B2/B3/TC = 1；
* 100% coverage；
* threshold = 0.5；
* 不使用第二张图、文本概念、source、probability、logit、margin、confidence 或 correctness。

#### 冻结分支

**C1**

* encoder：`vit_large_patch16_siglip_512.v2_webli`
* resolution：512×512
* views：`whole,crop,crop_q0,crop_q1,crop_q2,crop_q3`
* dense tensor：`6 × 1024 tokens × 1024 dimensions`
* fold-specific frozen gated pooler
* 输出分类器前 embedding：`c ∈ R^256`

**H3**

* encoder：PE-Spatial-L14-448
* resolution：448×448
* 相同六个语义视图
* dense tensor：最多 `6 × 1024 × 1024`
* valid-token mask
* fold-specific frozen masked-gated pooler
* 输出分类器前 embedding：`h ∈ R^128`

H3 的预期 shape 与哈希已在既有实现中锁定。

两个 encoder 与两个已有 pooler 均完全冻结。原始 dense token 不落盘；逐病例流式计算 embedding 后立即释放。

#### 候选模型

```text
c̄ = c / max(||c||₂, 1e-6)
h̄ = h / max(||h||₂, 1e-6)
x = concat(c̄, h̄) ∈ R384

u = GELU(W1 x + b1)          W1: 384 → 16
z = LayerNorm16(u)
logits = W2 Dropout0.10(z)+b2 W2: 16 → 2
p_high = softmax(logits)[1]
```

无 gate、无 attention、无 residual block、无 temperature、无 calibration。

| 层             |        参数 |
| ------------- | --------: |
| Linear 384→16 |     6,160 |
| LayerNorm 16  |        32 |
| Linear 16→2   |        34 |
| **总计**        | **6,226** |

初始化：

* 线性层 Xavier uniform，gain 1.0；
* bias 全零；
* LayerNorm scale=1，bias=0；
* primary seed=`20260714`。

#### 训练设置

| 项目                | 锁定值                                                      |
| ----------------- | -------------------------------------------------------- |
| Loss              | two-class cross-entropy                                  |
| Sampler           | outer-training 内 `source×risk` inverse-frequency sampler |
| Class weight      | 无                                                        |
| Optimizer         | AdamW                                                    |
| LR                | `3e-4`                                                   |
| Weight decay      | `1e-4`                                                   |
| Batch size        | `32`                                                     |
| Epoch ceiling     | `80`                                                     |
| Patience          | `12`                                                     |
| Validation metric | threshold 0.5 的 BAcc                                     |
| Tie               | 最早 epoch                                                 |
| Scheduler         | cosine annealing，T_max=80                                |
| Gradient clip     | 5.0                                                      |
| Head precision    | FP32                                                     |
| Primary seed      | `20260714`                                               |
| Conditional seed  | `20260715`                                               |
| Determinism       | deterministic algorithms；cuDNN benchmark off             |

### A3. Leakage-safe source-LODO

继续沿用已锁定 split 规则：held source 依次为 batch1、batch2、third_batch；validation 使用其余来源中的下一 master fold。

| Fold | Held source | Validation fold |
| ---- | ----------- | --------------: |
| 1    | batch1      |               2 |
| 2    | batch2      |               3 |
| 3    | third_batch |               4 |

每一折必须：

1. 使用已经排除 held source 的 C1/H3 fold checkpoint；
2. sampler、loss weighting、early stopping 全部只使用非 held source；
3. 不使用 target-source normalization、calibration 或 adaptation；
4. 每例只输出一次；
5. threshold 固定 0.5；
6. 合并后必须为 591/591。

提取完成后，必须用 embedding 加锁定原分类器重建 C1/H3 概率；任一 case/fold 最大绝对误差大于 `1e-5`，立即停止。

### A4. Required controls

所有 learned controls 使用相同的 384→16→2 代码路径、6,226 参数、optimizer、fold、sampler 和 seed。

| Control                      | 输入                 |
| ---------------------------- | ------------------ |
| `C1_ONLY_PADDED`             | `[c̄; zeros(128)]` |
| `H3_ONLY_PADDED`             | `[zeros(256); h̄]` |
| `C1_H3_EXACT`                | `[c̄_i; h̄_i]`     |
| `C1_H3_SAME_SOURCE_DERANGED` | `[c̄_i; h̄_π(i)]`  |

Derangement 固定为：

* 在每个 `split × source` 内单独生成；
* 按 `SHA256("H8|fold|split|case_id")` 排序；
* H3 序列循环平移一位；
* 不使用 risk/subtype 标签；
* train、validation、held test 各自独立 derange；
* 映射仅保存在服务器。

它保留来源与 H3 边际分布，但破坏同病例对齐。若 exact fusion 不能显著超过它，不能把结果解释为同病例跨表征疾病证据。

另报告以下非 advancing comparators：

* locked C1；
* locked C2；
* locked H3；
* 固定 `0.5×C1 + 0.5×H3` 概率平均，不搜索权重。

### A5. Required metrics

必须报告：

* accuracy、BAcc、AUC、sensitivity、specificity、TN、FP、FN、TP；
* 六个亚型的 n、correct、accuracy；
* B1/B2 精确病例数；
* 每个 held source 的 BAcc、sensitivity、specificity；
* 全部 source×subtype，单列 third-batch B2；
* 相对 H3/C2 的 same-case rescue、harm、net rescue；
* exact fusion 相对两个 branch controls 和 derangement control；
* 20,000 次 `source×risk` 分层病例配对 bootstrap；
* minimum-source BAcc；
* 参数、best epoch、wall time、GPU peak、RAM peak、新增磁盘峰值。

Bootstrap 95% 区间必须标注为 repeated-cohort stability interval，而不是独立确认性置信区间。

### A6. Advancement gates

#### Gate 0：完整性

以下任一失败即停止：

* 591 个唯一、正确对齐的 case；
* 368/223 risk 数和六亚型总数正确；
* 所有图像可访问；
* 三个 C1 和三个 H3 source-LODO checkpoint 存在；
* C1 cache 能在 offline 模式下唯一解析并锁 SHA-256；
* PE SHA-256 必须为
  `47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1`；
* 重建 C1/H3 概率误差不超过 `1e-5`；
* embedding 无 missing、duplicate、NaN、Inf；
* batch size 1 可在 RTX 4090 上运行。

#### Primary source-LODO gates

所有门槛必须同时通过：

| Gate | 要求                                                                       |
| ---- | ------------------------------------------------------------------------ |
| P1   | coverage `591/591`，threshold 0.5                                         |
| P2   | BAcc `≥0.7739`                                                           |
| P3   | sensitivity：`TP≥164/223=0.7354`                                          |
| P4   | specificity：`TN≥299/368=0.8125`                                          |
| P5   | B1：`≥40/62=0.6452`                                                       |
| P6   | B2：`≥59/89=0.6629`                                                       |
| P7   | third-batch B2：`≥18/29=0.6207`                                           |
| P8   | 相对 H3，至少 2/3 held source 的 ΔBAcc `>0`                                    |
| P9   | 无 held source ΔBAcc `<−0.0200`；minimum-source BAcc `≥0.7381`             |
| P10  | exact fusion ≥ max(C1-only,H3-only) `+0.0100`                            |
| P11  | exact − deranged BAcc `≥0.0100`；至少 2/3 source 正向；bootstrap CI lower `>0` |
| P12  | exact − locked H3 BAcc `≥0.0200`；bootstrap CI lower `>0`                 |
| P13  | B1+B2 相对 H3 net correct gain `≥7`，且 P5/P6 均通过                            |

P3 意味着至少恢复 H3 漏掉的 12 个高危病例；P4 最多允许相对 H3 损失五个低危病例。两者共同保证 BAcc 至少为 0.77396。

P6 要求 B2 比 H3 多答对至少七例，并达到 C2 水平；P7 要求 third-batch B2 比 H3 多答对至少五例。

AUC 不得补偿 BAcc、敏感度、B2、来源或 control gate 的失败。

#### Secondary five-fold gates

只有 P1–P13 全部通过后才能运行：

* coverage 591/591；
* BAcc `≥0.7903`；
* sensitivity `≥176/223=0.7892`；
* specificity `≥285/368=0.7745`；
* B1 `≥40/62`；
* B2 `≥60/89`；
* 无单个 test fold BAcc `<0.7000`；
* exact 超过每个 branch control `≥0.0100`；
* exact 超过 derangement `≥0.0100`，且 CI lower `>0`。

#### Confirmation seed

只有 primary 和 secondary 全部通过，才运行 seed `20260715` 的 source-LODO。

要求：

* P1–P13 再次全部通过；
* 两 seed 平均 ΔBAcc versus H3 `≥0.0200`；
* 不 ensemble；
* 不选择表现更好的 seed；
* primary seed 仍为指定结果。

### A7. Execution plan

#### 需要新增的仓库文件

```text
reports/H8_C1_H3_DIRECT_CASE_FUSION_PREREGISTRATION_20260714.md
scripts/lock_task7_h8_assets_20260714.py
scripts/extract_task7_h8_fold_embeddings_20260714.py
scripts/run_task7_h8_direct_case_fusion_20260714.py
scripts/analyze_task7_h8_direct_case_fusion_20260714.py
scripts/run_task7_h8_direct_case_fusion_queue_20260714.sh
```

运行后只新增：

```text
reports/H8_C1_H3_DIRECT_CASE_FUSION_RESULTS_20260714.md
```

#### 核心不可变路径

```text
REGISTRY=/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv

SPLIT=/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv

C1_LODO_ROOT=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711

C2_LODO_PRED=/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion/lodo_predictions.csv

H3_LODO_ROOT=/workspace/thymic_project/experiments/h3_representation_renewal_20260713/h3b_runs/pe_spatial_l14_448/source_lodo

PE_CKPT=/root/model_weights/modelscope/facebook/PE-Spatial-L14-448/PE-Spatial-L14-448.pt

PE_SOURCE=/root/third_party/perception_models_3e352cca

C1_CACHE_ROOT=/root/.cache/huggingface/hub/models--timm--vit_large_patch16_siglip_512.v2_webli

OUT=/workspace/thymic_project/experiments/h8_c1_h3_direct_case_fusion_20260714
```

#### Primary command sequence

```bash
set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PY=/root/miniconda3/envs/thymic_baseline/bin/python
CODE=/root/task7_h8_20260714
OUT=/workspace/thymic_project/experiments/h8_c1_h3_direct_case_fusion_20260714

mkdir -p "$OUT"/{locks,source_lodo,logs}

"$PY" "$CODE/lock_task7_h8_assets_20260714.py" \
  --evidence-commit ff81fe4c98fd5f1b9d2bcfd53f4379e5741e38e8 \
  --registry-csv /root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv \
  --split-csv /workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv \
  --c1-root /workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711 \
  --c2-predictions /workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion/lodo_predictions.csv \
  --h3-root /workspace/thymic_project/experiments/h3_representation_renewal_20260713/h3b_runs/pe_spatial_l14_448/source_lodo \
  --c1-model-name vit_large_patch16_siglip_512.v2_webli \
  --c1-cache-root /root/.cache/huggingface/hub/models--timm--vit_large_patch16_siglip_512.v2_webli \
  --pe-checkpoint /root/model_weights/modelscope/facebook/PE-Spatial-L14-448/PE-Spatial-L14-448.pt \
  --expected-pe-sha256 47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1 \
  --pe-source-root /root/third_party/perception_models_3e352cca \
  --expected-pe-source-revision 3e352cca660658d4b5c90f42a7808b11469e4c66 \
  --output-manifest "$OUT/locks/source_lodo_assets.json"

"$PY" "$CODE/extract_task7_h8_fold_embeddings_20260714.py" \
  --asset-manifest "$OUT/locks/source_lodo_assets.json" \
  --split-mode source_lodo \
  --views whole,crop,crop_q0,crop_q1,crop_q2,crop_q3 \
  --c1-image-size 512 \
  --h3-image-size 448 \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda \
  --seed 20260714 \
  --output-dir "$OUT/source_lodo/embeddings"

"$PY" "$CODE/run_task7_h8_direct_case_fusion_20260714.py" \
  --embedding-manifest "$OUT/source_lodo/embeddings/embedding_manifest.json" \
  --configuration H8_C1_H3_CONCAT_MLP16 \
  --split-mode source_lodo \
  --hidden-dim 16 \
  --dropout 0.10 \
  --epochs 80 \
  --patience 12 \
  --batch-size 32 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --grad-clip 5.0 \
  --seed 20260714 \
  --device cuda \
  --output-dir "$OUT/source_lodo/primary_seed20260714"

"$PY" "$CODE/analyze_task7_h8_direct_case_fusion_20260714.py" \
  --stage source_lodo \
  --run-dir "$OUT/source_lodo/primary_seed20260714" \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 20260714 \
  --enforce-gates \
  --output-dir "$OUT/source_lodo/aggregate"
```

Analyzer 必须在任何 gate 失败时：

* 返回 nonzero；
* 写入 `FINAL_DECISION.txt`；
* 内容精确为 `STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT`；
* queue 不得继续 five-fold。

#### 资源预算

规划值而非已测量结果：

* asset lock 和 smoke：15–30 分钟；
* primary 双 backbone 流式提取：4–8 小时；
* source-LODO candidate 与 controls：少于 30 分钟；
* bootstrap/report：少于 15 分钟；
* 条件性 five-fold：另 4–8 小时；
* confirmation seed：少于 30 分钟；
* 全部阶段均通过时总计约 9–17 小时。

硬约束：

* C1 与 H3 encoder 不同时驻留 GPU；
* extraction batch size 固定为 1；
* GPU allocated memory 目标 `<22 GiB`；
* 新增磁盘上限 `1 GiB`；
* embedding 预计不足 20 MiB；
* 不重建多 GB dense bank。

固定 batch-one 仍 OOM 或超过 1 GiB 时，直接停止；不得改变 resolution、token count 或 architecture。

#### 中断恢复

* 每个 case shard 先写 `.tmp`，fsync 后 atomic rename；
* restart 只有在 SHA-256、shape、finite 和 case/fold cardinality 均通过时才跳过；
* 不保存 raw token；
* 未完成的训练 fold/control 删除后以相同 seed 从 epoch 1 重跑；
* 不恢复部分 optimizer state；
* aggregate 文件均 atomic write。

#### GitHub 与服务器边界

允许进入 GitHub：

* preregistration、代码；
* symbolic asset hashes；
* aggregate metrics；
* source/subtype counts；
* bootstrap summaries；
* gate decision；
* runtime/storage 报告。

必须留在服务器：

* 图像和路径；
* case ID 与 registry；
* embedding、mask；
* per-case prediction；
* rescue/harm 明细；
* derangement map；
* checkpoint 和权重。

---

## 7. **Evidence and literature table**

| 证据                   | 关键发现                                                      | 对决策的意义                         |
| -------------------- | --------------------------------------------------------- | ------------------------------ |
| Plateau report       | H3 LODO 0.7539；平均 0.7602；confidence 0.7462；oracle 0.9006。 | 存在错误互补，但输出融合不是解决方案。            |
| Oracle decomposition | PE-only oracle 0.8462，跨家族 oracle 更高。                      | 唯一剩余合理方向是跨家族直接特征。              |
| Equal fusion code    | 读取概率并 `np.mean`。                                          | 未覆盖病例特征级融合。                    |
| qkvb/ROI meta-fusion | 使用 probability/logit/ROI numeric。                         | 行为纠错已覆盖且不合格。                   |
| C1 MoE               | 所有 expert/gate 共享一个 embedding。                            | 不是跨表征专家。                       |
| H1 sequential        | 全部读取 frozen C1 representation。                            | 不能关闭 C1–H3 假设。                 |
| B1 cascade           | 第二 reader 为 SigLIP native tiles，fusion 在输出层。              | 不是所提机制。                        |
| H3                   | 五折 0.8003，LODO 0.7539；B1 上升、B2/敏感度下降。                     | H3 是不同表征，但单分支不够稳。              |
| H5                   | 一阶与二阶均来自 PE tensor。                                       | 只证明 PE 内部 feature fusion 做过。   |
| Source audit         | 来源效应显著大于风险效应。                                             | 必须有 source-preserving control。 |
| Local assets         | PE checkpoint/source 本地可用，可流式提取。                          | H8 可在 no-download 边界内执行。       |
| WHO/ITMIG/Zucali     | 终点是组织学分类，B1/B2 边界存在形态连续和一致性限制。                            | 不能把 gross-photo 风险预测表述为显微分型真值。 |
| Zech et al.          | 医院与患病率耦合可抬高内部表现。([PLOS][1])                               | source-LODO 是必要但非充分条件。         |
| Badgeley et al.      | 采集与流程混杂可成为主要预测来源。([arxiv.org][2])                         | 图像输入不自动等于疾病证据。                 |
| DomainBed            | DG 的 model-selection rule 必须预设。([arxiv.org][3])           | H8 后禁止 adaptive search。        |

---

## 8. **Exact next actions and hard stopping rule**

1. 在查看任何 H8 指标前，先提交预注册和五个实现脚本。
2. 只上传小型源码；不下载任何模型、包或数据。
3. 运行 offline asset lock。
4. 资产、hash、路径、病例数、checkpoint 或概率重建任一失败，立即停止。
5. 运行 primary source-LODO candidate 与固定 controls。
6. 只有 P1–P13 全部通过才运行 five-fold。
7. 只有 five-fold 全部通过才运行 seed `20260715`。
8. 不调 threshold，不 ensemble seeds，不改变模型配置。
9. 两个已消费的 108 和 162 例队列不参与选择，也不能用于挽救失败结果。
10. 即便 H8 通过，它们最多在模型完全锁定后被标注为 **retrospective consumed stress tests**，不能称为 fresh external confirmation。
11. 任一阶段失败后，保留 C1、C2、H3 作为诚实锁定基线：H3 是最佳 direct single-model BAcc，C2 是较高 sensitivity/B2 的固定视觉 ensemble comparator。
12. 不再进行任何当前 591 例分类器搜索。

### Hard stopping rule

> **在第一个完整性失败或预注册 gate 失败时，写入 `STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT`；不运行下一阶段，不读取外部队列进行挽救，也不再进行任何 seed、threshold、fusion、routing、pooling、loss、augmentation 或 architecture 实验。**

本次已完成的是完整仓库审计、终局决策与不可变实验预注册。由于病例图像、checkpoint 与 RTX 4090 环境只存在于项目服务器，H8 数值实验本身未在此会话中执行；以上没有虚构任何新性能结果。

[1]: https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002683 "https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002683"
[2]: https://arxiv.org/abs/1811.03695 "https://arxiv.org/abs/1811.03695"
[3]: https://arxiv.org/abs/2007.01434 "https://arxiv.org/abs/2007.01434"
