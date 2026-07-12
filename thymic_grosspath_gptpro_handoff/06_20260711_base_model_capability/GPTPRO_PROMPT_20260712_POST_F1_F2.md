# GPTPro Follow-up: Task7 Direct Visual Capability After Wave C, D1, E1, F1, and F2

Repository: <https://github.com/ScottBlizzard/tarmac>

Read first:

1. `thymic_grosspath_gptpro_handoff/06_20260711_base_model_capability/README_20260711_BASE_MODEL_CAPABILITY.md`
2. `.../reports/Task7_WaveC_D1_E1_F1_F2_Results_20260712.md`
3. `.../reports/PHYSICIAN_BLINDED_ROI_ORACLE_PROTOCOL_20260712.md`
4. `.../reports/PHYSICIAN_ROI_ANNOTATION_LOCK_WORKFLOW_20260712.md`
5. The corresponding scripts in `.../scripts/`.

## Objective and non-negotiable boundary

The objective is a stronger 100%-coverage image classifier for thymic gross pathology low-risk versus high-risk classification. Low risk is A/AB/B1; high risk is B2/B3/thymic carcinoma. Rejection, manual review, probability-only correction, source-aware calibration, and confidence routing are not substitutes for base visual capability.

A multi-model method is acceptable only when every diagnostic member reads images and learns complementary visual evidence. A confidence or error router may be analyzed, but it cannot be credited as the source of diagnostic knowledge.

Both previously inspected external cohorts (108 and 162 cases) are consumed audit sets. They cannot be used for further selection, unsupervised adaptation, thresholding, fusion, or confirmation.

## Locked references

| Model | Five-fold BAcc/AUC | Source-LODO BAcc/AUC |
| --- | ---: | ---: |
| C1: SigLIP-L@512 six-view dense gated | 0.7477/0.8240 | 0.7397/0.8072 |
| C2: AIMv2 MixStyle + C1 fixed equal image fusion | 0.7514/0.8377 | 0.7441/0.8108 |

## Newly completed changed-information experiments

| Family | Five-fold BAcc/AUC | Source-LODO BAcc/AUC | Decision |
| --- | ---: | ---: | --- |
| SwinV2-L six-view | 0.6909/0.7533 | 0.6098/0.6594 | No-go |
| ConvNeXtV2-L six-view | 0.7384/0.8184 | 0.6831/0.7520 | No-go |
| SigLIP-SO400M six-view | 0.7433/0.8227 | 0.7070/0.7593 | No-go |
| D1 label-trained attention ROI | 0.7655/0.8428 | 0.6788/0.7679 | No-go; OOF gain reverses under LODO |
| E1 fixed label-free anatomy ROI | 0.7302/0.7800 | 0.7051/0.7759 | No-go; loses to matched random ROI |
| E1 matched single random ROI bag | 0.7407/0.8027 | 0.7284/0.8004 | Control only |
| F1 three random bags + consistency | 0.7450/0.8173 | 0.7357/0.8098 | No-go; B1 collapse |
| F2 fixed 0.5 C1 + 0.5 F1 | 0.7401/0.8369 | 0.7443/0.8326 | No-go; source/subtype instability |

F1 is the most informative partial signal. Versus its matched base model, source-LODO BAcc improved by 0.0232 and AUC by 0.0460, with source BAcc changes `+0.0425 / 0 / +0.0100`. However five-fold BAcc decreased by 0.0069 and LODO B1 accuracy decreased by 0.1452. F2 significantly improved ranking versus C1 but not forced classification; it reduced third-batch BAcc by 0.0480 and harmed B2/B3.

Historical experiments on the same strong C1 representation already tested B1/B2 auxiliary shaping, a conservative boundary expert, and six-class auxiliary supervision. They moved errors between B1 and B2 without improving both. Internal VICReg, LoRA, DANN, GroupDRO, REx, MixStyle variants, SAM, preprocessing, structured pooling, native tiles, and confidence-routed visual cascades also failed their gates.

## Physician ROI oracle is ready

A 120-case, six-subtype, three-source, two-reader packet exists on the server. Readers are blinded to labels, subtype, C1 output, and C1 correctness. The package now has:

- separate 120-row reader forms;
- normalized-coordinate and conditional-field validation;
- cryptographic annotation locking before secure-key access;
- top-ROI IoU agreement;
- manual ROI at exact and 1.5x context scales;
- matched random ROI controls;
- five-fold and source-LODO direct-model evaluation;
- B1/B2 net-rescue and predeclared PASS/NO-GO analysis.

No patient images, secure keys, or case-level predictions are in GitHub.

## Your task

Act as a skeptical research lead and answer these questions.

1. Does the evidence now justify stopping architecture, crop, loss, threshold, and fusion searches on the same 591 single photographs until the physician ROI oracle or genuinely new views arrive?
2. Is there any remaining experiment on existing data that changes the visual information or causal assumption enough to be defensible, rather than creating another meta-selection layer? Do not propose another backbone, loss-weight, boundary-head, bag-count, confidence-router, threshold, or fusion-weight sweep.
3. Audit the G1 physician ROI oracle protocol and code. Identify any unfair comparison, leakage path, insufficient nesting, ROI-agreement flaw, or random-control flaw that must be fixed before real annotation lock.
4. Interpret the repeated pattern of improved AUC but unstable BAcc and B1/B2 tradeoffs. Distinguish calibration instability, class-boundary overlap, limited image information, and source shift. State what additional evidence would separate these explanations.
5. If G1 is positive, give the exact fully nested automatic anatomical ROI detector experiment. If G1 is negative, give the exact standardized multi-view acquisition protocol and minimum new multicenter cohort composition.
6. Define what can honestly be claimed in a medical-engineering paper now, what requires fresh external confirmation, and what must be reported as negative evidence.

Required output:

1. `Stop-or-Continue Verdict on Existing Single Images`
2. `Audit of the Physician ROI Oracle`
3. `Only Remaining Defensible Existing-Data Experiment, If Any`
4. `Decision Tree After Positive or Negative Manual ROI Oracle`
5. `Fresh Multi-View and Multicenter Data Requirements`
6. `Publication Claims Now Versus Claims Requiring New Evidence`

Be explicit about uncertainty and distinguish verified results, code-based inference, and recommendations.
