# GPT Pro Response: Fixed-Data Literature Search and Executable Capability Plan

Complete GPT Pro response pasted on 2026-07-14.

## 1. Fixed-data decision

One bounded experiment is scientifically justified: **H6 Nuisance-Anchored Common–Specific Decomposition (NA-CSD)** on the frozen H3 PE-Spatial dense representation. It directly targets the observed failure mechanism: all 156 fixed Haar features carry more source than risk signal, so forcing the whole representation to become source-invariant may erase genuine morphology; CSD instead lets training environments absorb unstable source-dependent label directions and discards those directions at inference.  ([arXiv][1]) Training-only nuisance subdomains are added because ordinary source-LODO leaves only two training sources, which is weakly identified for common/specific decomposition. Fishr is the only conditional backup; LISA is the third-ranked reserve but is not queued. Even a successful result would be an exploratory engineering improvement and internal cross-batch robustness result, not independent external validation.

---

## 2. Literature evidence table

The search was restricted to primary papers and official implementations available by **July 14, 2026**.

| Primary source                                                                                                                                                    | Core mechanism                                                                                                                                                | Failed assumption it could change here                                                                                                                                                | Overlap with completed work                                                                                                                                                                         | Required assets                                                                                                       | Executable without download?                   |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Efficient Domain Generalization via Common-Specific Low-Rank Decomposition**, 2020, `arXiv:2003.12815` ([arXiv][1])                                             | Decomposes only the final classifier as (W_d=W_c+W_s\gamma_d); orthogonalizes common and domain-specific directions; discards (W_s,\gamma_d) at inference.    | Does **not** assume that useful features lack source information. It separates unstable source–label associations at the predictor level rather than erasing source-bearing features. | No equivalent project experiment. DANN, CORAL, REx and GroupDRO alter features or risks; they do not learn and then discard low-rank source-specific classifiers.                                   | Standard PyTorch only.                                                                                                | **Yes. Primary basis.**                        |
| **Prompt-driven Latent Domain Generalization for Medical Image Classification**, 2024, `arXiv:2401.03002` ([arXiv][2])                                            | Discovers latent domains from bias/style-associated features, then trains shared and domain prompts with domain mixup.                                        | The three batch labels may be too coarse to describe variation in color, compression, focus and device era.                                                                           | Per-image PE part clustering was tested, but not cross-case nuisance-environment discovery. Full prompt learning is not needed.                                                                     | Full method would require new prompt code; PCA-based latent-domain discovery needs only sklearn.                      | **Latent-domain component yes; full PLDG no.** |
| **Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization**, 2022, `arXiv:2109.02934`; official `alexrame/fishr` ([arXiv][3])                  | Matches domain-wise variances of per-example loss gradients, approximating alignment of Fisher information and local loss landscapes.                         | Source nuisance may appear as different local optimization geometry even when feature means or covariances should not be aligned.                                                     | Distinct from REx, which matches scalar risks, and CORAL, which matches feature covariance.                                                                                                         | Official code uses BackPACK, which is unavailable; final-classifier gradients can be derived analytically in PyTorch. | **Yes, as the single backup.**                 |
| **Improving Out-of-Distribution Robustness via Selective Augmentation (LISA)**, 2022, `arXiv:2201.00299` ([arXiv][4])                                             | Mixes same-label cases across domains and different-label cases within domains to learn an invariant predictor without requiring invariant internal features. | Cross-source interpolation could preserve risk while smoothing source-specific decision directions.                                                                                   | Different from MixStyle and H4 photometric consistency; nevertheless, project contrastive and augmentation results lower its prior probability.                                                     | Standard PyTorch; no new images or checkpoint.                                                                        | **Yes. Third-ranked only.**                    |
| **Learning to Balance Specificity and Invariance for In- and Out-of-Domain Generalization**, 2020, `arXiv:2008.12839` ([arXiv][5])                                | Learns domain-specific masks while retaining a shared model.                                                                                                  | Source-specific features may contain useful class information and should be isolated rather than globally suppressed.                                                                 | Similar intuition to CSD but considerably more parameters; overlaps prior attention, MoE and specialist behavior.                                                                                   | New mask modules trained from scratch.                                                                                | Technically yes; statistically unattractive.   |
| **Aggregation of Disentanglement: Reconsidering Domain Variations in Domain Generalization**, 2023, `arXiv:2302.02350` ([arXiv][6])                               | Learns domain-expert features, removes noise and aggregates expert information for a new domain.                                                              | Domain-varying features can contain classification information rather than being pure nuisance.                                                                                       | Substantial overlap with tested MoE, expert, contrastive and prototype systems.                                                                                                                     | Multiple expert branches and contrastive machinery.                                                                   | Technically yes; not bounded enough.           |
| **DISPEL: Domain Generalization via Domain-Specific Liberating**, 2023, `arXiv:2307.07181` ([arXiv][7])                                                           | Per-sample embedding mask filters predicted domain-specific dimensions after model fitting.                                                                   | Nuisance may differ case by case, not merely by batch.                                                                                                                                | Closely adjacent to attention masks, part selection, quality gates and instance routing already tested.                                                                                             | New mask generator; no external checkpoint.                                                                           | Yes, but high adaptive-overfitting risk.       |
| **Gradient Matching for Domain Generalization (Fish)**, 2021, `arXiv:2104.09937` ([arXiv][8])                                                                     | First-order meta-learning encourages gradients from different domains to point in similar directions.                                                         | Directly targets conflicting source updates rather than matching losses or features.                                                                                                  | No exact duplicate, but each LODO fit has only two source gradients and relatively few high-risk examples.                                                                                          | Standard autograd; repeated inner updates.                                                                            | Yes, but statistically weak here.              |
| **SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances**, 2021, `arXiv:2106.02266` ([arXiv][9])                                      | Continuously masks parameter updates according to agreement in gradient sign and magnitude across domains.                                                    | Could prevent updates driven by one source only.                                                                                                                                      | Directionally related to prior invariance objectives; much noisier with only two training sources per LODO fold.                                                                                    | Per-source gradients; standard PyTorch.                                                                               | Yes, but unsuitable as a primary test.         |
| **Domain Generalization by Mutual-Information Regularization with Pre-trained Models (MIRO)**, 2022, `arXiv:2203.10789`; official `kakaobrain/miro` ([arXiv][10]) | Uses a frozen pretrained model as an oracle and constrains a fine-tuned student to retain its generalizable information.                                      | PE may contain transferable information that ordinary task adaptation destroys.                                                                                                       | LoRA, LP-style starts and partial fine-tuning were already insufficient; using PE as both oracle and student also supplies no independent representation.                                           | Frozen teacher activations plus an end-to-end student.                                                                | Technically possible, not compelling now.      |
| **SoRA: Singular Value Decomposed Low-Rank Adaptation for Domain-Generalizable Representation Learning**, 2024, `arXiv:2412.04077` ([arXiv][11])                  | Tunes minor singular components while freezing dominant components thought to contain generalizable pretrained knowledge.                                     | More selective than ordinary LoRA and could reduce catastrophic loss of PE generalization.                                                                                            | Meaningful modification of failed LoRA, but still an encoder-adaptation hypothesis; the paper’s main evidence is detection and segmentation rather than this small binary classifier.               | PE checkpoint plus custom SVD adapters; no official code is locally cached.                                           | Possible from scratch, but not first-line.     |
| **SWAD: Domain Generalization by Seeking Flat Minima**, 2021, `arXiv:2102.08604`; official `khanrc/swad` ([arXiv][12])                                            | Dense trajectory weight averaging with an overfit-aware window to seek flatter minima.                                                                        | A flat solution might transfer better than an early-stopped head.                                                                                                                     | The project tested SAM and several strong regularization/optimizer variants; the current head is also very small. It changes optimization, not available visual information or nuisance structure.  | Checkpoint averaging only.                                                                                            | Yes, but not a primary capability mechanism.   |
| **ERM++: An Improved Baseline for Domain Generalization**, 2023, `arXiv:2304.01973`; official `piotr-teterwak/erm_plusplus` ([arXiv][13])                         | Improves ERM through better initialization, data use, training duration, validation handling and regularization.                                              | Warns against crediting a complex DG loss when training discipline explains the gain.                                                                                                 | Useful as a control principle. Most components already exist in H3’s locked fitting protocol.                                                                                                       | No special assets.                                                                                                    | Yes; baseline discipline only.                 |
| **In Search of Lost Domain Generalization / DomainBed**, 2021, `arXiv:2007.01434` ([arXiv][14])                                                                   | Standardizes architectures, splits and model-selection criteria and shows that DG methods without a selection rule are incomplete.                            | Directly addresses the project’s extensive adaptive reuse of the same internal sources.                                                                                               | Supports the existing outer-fold and source-LODO protocol rather than nominating a new model.                                                                                                       | Evaluation code only.                                                                                                 | Yes; methodological requirement.               |
| **A Principled Approach to Model Validation in Domain Generalization**, 2023, `arXiv:2304.00629` ([arXiv][15])                                                    | Selects models using both validation risk and domain discrepancy.                                                                                             | A high validation BAcc checkpoint can still be highly source-dependent.                                                                                                               | With only two training sources in LODO, discrepancy estimates are noisy; adding a new selection score would introduce another adaptive degree of freedom.                                           | No checkpoint; discrepancy implementation.                                                                            | Yes, but not advisable for this locked run.    |
| **Self-Challenging Improves Cross-Domain Generalization**, 2020, `arXiv:2007.02454` ([arXiv][16])                                                                 | Suppresses currently dominant task features so that the model must use alternative cues.                                                                      | Could reduce dependence on source-correlated dominant PE dimensions.                                                                                                                  | Dropout, attention, multiple-view and specialist experiments already show that alternate cues frequently move B1 against B2.                                                                        | Standard autograd.                                                                                                    | Yes, but low expected value.                   |
| **Semantic Data Augmentation Enhanced Invariant Risk Minimization for Medical Image Domain Generalization**, 2025, `arXiv:2502.05593` ([arXiv][17])               | Uses inter-domain covariance to select augmentation directions rather than applying random perturbations.                                                     | More targeted than generic color or blur augmentation in small medical data.                                                                                                          | H4, DANN, REx, source-balanced augmentation and covariance alignment were already negative; the measured covariance is predominantly acquisition-related.                                           | Custom augmentation; no new checkpoint required.                                                                      | Technically yes; already tested in substance.  |
| **Magnification-Invariant Image Classification via Domain Generalization and Stable Sparse Embedding Signatures**, 2026, `arXiv:2604.25817` ([arXiv][18])         | Uses gradient reversal and sparse stable embeddings under leave-one-magnification-out histopathology evaluation.                                              | Recent evidence that acquisition invariance can improve a medical image classifier.                                                                                                   | Its operative mechanism is DANN/gradient reversal, which the project has already tested; magnification-defined microscopy domains are also cleaner than the current mixed gross-photograph batches. | Standard DANN and sparsity machinery.                                                                                 | Yes, but a substantive repeat.                 |

### Literature synthesis

Three conclusions survive the project-specific elimination:

1. **Predictor decomposition is more defensible than representation erasure.** CSD explicitly covers settings in which every useful feature also contains domain information—the pattern now observed in the fixed frequency and PE audits. ([arXiv][1])

2. **The known batch label is probably too coarse.** PLDG supports deriving training environments from bias/style features, but the full prompt architecture would be too adaptive for 591 cases. Only the latent-environment idea should be borrowed. ([arXiv][2])

3. **Model selection must remain simple and outer-fold clean.** DomainBed and later validation work show that adding complex DG objectives and then choosing them on the same domains can erase the apparent advantage. ([arXiv][14])

---

## 3. Method-family elimination matrix

| Method family                                                                         | Classification                                                              | Decision and reason                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Training-only nuisance environments + final-layer common/specific decomposition**   | **Genuinely new mechanism**                                                 | Neither the nuisance audit nor any prior classifier used source-dominant, risk-weak features solely to define training environments. CSD isolates source-specific label directions rather than removing source information from all PE features. **Advance as H6.**                                                         |
| **Source-only CSD**                                                                   | **Genuinely new, but control only**                                         | Final-layer decomposition is new, but two training sources in each LODO fold place rank-1 CSD at the (D-1) extreme where finite-sample noise is a concern. Use only as the required ablation.                                                                                                                               |
| **Fishr gradient-variance matching**                                                  | **Genuinely new mechanism**                                                 | REx matches scalar risks; Fishr matches per-parameter gradient variances. It is executable analytically on the final classifier, but two source groups make the estimate noisy. **Conditional backup only.**                                                                                                                |
| **LISA selective cross-source interpolation**                                         | **Meaningful modification of a failed mechanism**                           | H4 augmented pixels and matched clean/augmented predictions; LISA instead changes which class/source pairs are interpolated and targets invariant prediction. It remains risky because B1/B2 hard-pair and contrastive interventions already moved the boundary rather than strengthening it. **Ranked third, not queued.** |
| **Domain-specific masks, DDN, DISPEL**                                                | **Statistically unsuitable for 591 cases**                                  | These add per-domain experts, per-input masks or disentanglement networks. They are substantially higher-capacity than CSD and overlap prior MoE, attention, part and specialist failures.                                                                                                                                  |
| **Fish, SAND-mask and other gradient-agreement methods**                              | **Statistically unsuitable with two training sources**                      | A single noisy gradient comparison can dominate a LODO update. B3 has 24 total cases and B1 only 62, making domain-wise gradient agreement especially unstable.                                                                                                                                                             |
| **MIRO and oracle-preserving adaptation**                                             | **Meaningful modification of failed adaptation, but no independent oracle** | PE could be both frozen oracle and student initialization, but this does not add independent knowledge. Multiple LoRA/partial-fine-tuning branches already improved in-domain fitting without restoring LODO.                                                                                                               |
| **SoRA or another PEFT design**                                                       | **Meaningful modification of failed LoRA**                                  | SoRA changes which pretrained singular directions are adapted, but it remains an encoder-adaptation experiment with more implementation choices than CSD and no direct evidence on this classification regime. Do not run before H6.                                                                                        |
| **SWAD, ERM++, large-LR flat-minimum methods**                                        | **Already tested in substance / no changed visual assumption**              | SAM and strong regularization were negative; H3 already uses frozen features, a low-capacity head and early stopping. These may alter a fraction of a point but do not address the documented source-dependent B2 decision direction.                                                                                       |
| **RSC/self-challenging**                                                              | **Meaningful but weak modification**                                        | Dominant-feature suppression could remove a shortcut, but PE’s source-bearing and risk-bearing dimensions need not be separable. Existing local-view and expert results show that forcing alternate evidence often protects B1 at B2’s expense.                                                                             |
| **Full PLDG prompt learning**                                                         | **Statistically unsuitable and not locally implemented**                    | Latent style discovery is useful, but prompts, prompt generators and domain mixup would create several new trainable modules and hyperparameters. Borrow the environment-discovery concept only.                                                                                                                            |
| **Covariance-guided augmentation, DANN/IRM, 2025–2026 medical DG variants**           | **Already tested in substance**                                             | H4, DANN, REx, class-conditional alignment, CORAL-like losses, domain randomization and source balancing directly cover these assumptions. H4 retained OOF strength but reduced LODO BAcc to 0.7254.                                                                                                                        |
| **Texture, frequency or wavelet risk heads**                                          | **Already contradicted**                                                    | H5 lowered LODO BAcc to 0.7422 and B2 accuracy to 0.4719; all fixed frequency features were more source-associated than risk-associated. Frequency is appropriate as a nuisance descriptor, not a prediction input.                                                                                                         |
| **Parts, spatial transformers, relational grids, ROI, tiles, extra same-photo crops** | **Already tested in substance**                                             | The true-grid relational model, H2, native tiles, attention/anatomy ROI and random bags all failed; the PE part audit found stable parts but source-sensitive stability and no classifier justification.                                                                                                                    |
| **Broad multi-image MIL**                                                             | **Impossible as a main mechanism**                                          | Only 17 cases have a second image and all are from old data; 574/591 have one image.                                                                                                                                                                                                                                        |
| **SAM2/Hiera-SLCA**                                                                   | **Impossible under local assets**                                           | Architecture names are installed, but no compatible package or pretrained weights exist locally, and downloading them is prohibited.                                                                                                                                                                                        |
| **New backbones or foundation checkpoints**                                           | **Impossible or already tested**                                            | SigLIP2, RADIO, MedSigLIP and PE were already screened; no additional download is allowed. PE is the only locally cached representation with a clear mixed-source capability gain.                                                                                                                                          |
| **Current-cohort SSL using all 591 images**                                           | **Evaluation-invalid unless nested; underpowered if nested**                | Full-cohort SSL would make OOF transductive. Fold-specific SSL would use only a few hundred highly correlated images and repeats earlier VICReg-style adaptation.                                                                                                                                                           |
| **Test-time adaptation, source calibration, source thresholds**                       | **Prohibited**                                                              | These use target-source information at deployment and do not establish a stronger fixed image classifier.                                                                                                                                                                                                                   |
| **Probability fusion, confidence routing and threshold searches**                     | **Already exhausted and out of scope**                                      | They manipulate existing outputs and cannot be credited as visual-capability gains.                                                                                                                                                                                                                                         |

---

## 4. Ranked top three executable experiments

### Rank 1 — H6 Nuisance-Anchored Common–Specific Decomposition

**Changed assumption.** Useful PE dimensions may carry both risk and acquisition information; the problem is not the presence of source information itself, but source-dependent correlations between those dimensions and the risk label.

**Why prior negative evidence does not close it.** DANN, CORAL, REx, GroupDRO, H4 and H5 either suppressed, aligned, reweighted or augmented representations. None trained a common risk classifier alongside low-rank environment-specific residual classifiers and then discarded those residuals. CSD was designed precisely for settings where every feature can be domain-predictive. ([arXiv][1])

**Input.** Exact H3 PE-Spatial six-view dense bank: up to (6\times1024) valid 1,024-dimensional tokens per case, with the locked valid-token mask.

**Model.** Exact H3 masked gated pooling through its 128-dimensional case embedding; replace only the final linear layer with rank-1 CSD. Training environments are acquisition source × training-only nuisance-PC bin. Inference uses only the common classifier.

**Expected effect.**

* Source nuisance: lower dependence of the decision vector on acquisition-specific texture/color directions.
* Sensitivity: recover approximately 12 high-risk cases needed to move from H3’s 0.6816 to C2’s 0.7354.
* B2: recover approximately seven cases needed to move from 0.5843 to 0.6629.
* B1: retain at least 38/62 cases, close to H3’s 40/62.
* Specificity: expected to decline moderately from H3’s 0.8261, but remain well above C2 if the decomposition works.

**Cost.** Approximately 5–12 GPU-hours total; peak new disk about 8 GB; no new package or checkpoint.

**Most informative negative result.** If source-specific residuals become nontrivial and improve training-source validation, but the common head still cannot recover sensitivity/B2, then unstable decision directions cannot be adequately isolated as a low-rank final-layer nuisance.

**Main risk.** The four training environments in a source-LODO fold may still be too few, and nuisance-PC bins may represent continuous acquisition variation poorly.

---

### Rank 2 — Final-classifier Fishr on the H3 PE embedding

**Changed assumption.** Source dependence may not be representable by one low-rank classifier direction; it may instead appear as source-specific gradient variance and local curvature.

**Why not already closed.** REx matches source risks, while Fishr matches variances of per-example gradients and thereby approximates local Fisher/Hessian alignment. ([arXiv][3])

**Input and model.** Exact H3 PE tokens and masked gated head. The Fishr term is computed analytically for per-example gradients of the final (2\times128) classifier, avoiding BackPACK.

**Expected effect.** Reduced source-to-source instability without forcing feature means to align; potentially better sensitivity and B2 than H3, though B1 retention is uncertain.

**Cost.** Approximately 6–12 GPU-hours using the same regenerated bank; negligible additional storage.

**Most informative negative result.** Failure would indicate that neither predictor-direction decomposition nor final-layer loss-landscape alignment can stabilize the PE boundary.

**Main risk.** Gradient variances estimated from only two training sources and small source×risk cells may be too noisy.

---

### Rank 3 — PE-embedding LISA

**Changed assumption.** An invariant decision boundary may be learnable through selective interpolation even when invariant representations are not.

**Input and model.** H3’s 128-dimensional case embedding. Half of mixed examples pair the same risk across different training sources; half pair opposite risks within the same source. The classifier is the ordinary H3 common head.

**Expected effect.** Same-risk cross-source mixes may suppress source-specific directions; within-source cross-risk mixes may smooth the B1/B2 boundary. The risk is excessive smoothing of genuinely adjacent morphology.

**Cost.** Approximately 4–8 GPU-hours; no additional feature bank beyond H6.

**Most informative negative result.** It would close the remaining distinction between simple style augmentation and source/label-selective interpolation.

**Main risk.** Linear interpolation in PE embedding space may not correspond to a medically valid visual path, and prior contrastive/hard-negative results suggest that boundary manipulation often trades B1 against B2.

**Decision:** rank 3 is not scheduled. It is retained only as an explicit elimination result if H6 and the conditional backup fail.

---

## 5. Exact preregistration for the primary experiment

### 5.1 Locked name and hypothesis

**Experiment:** `H6_NUISANCE_ANCHORED_CSD_20260714`

**Primary hypothesis:** PE contains a common visual risk direction plus acquisition-dependent risk directions. Training low-rank environment-specific residual classifiers will absorb the latter, allowing the common classifier to retain H3’s B1 and mixed-source capability while restoring C2-level high-risk sensitivity and B2 accuracy.

**Primary protocol:** source-LODO.

**Secondary protocol:** canonical five-fold OOF.

**Threshold and coverage:** fixed 0.5, 100%.

### 5.2 Immutable inputs

Use:

* PE checkpoint
  `/root/model_weights/modelscope/facebook/PE-Spatial-L14-448/PE-Spatial-L14-448.pt`
* checkpoint SHA-256
  `47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1`
* local PE source
  `/root/third_party/perception_models_3e352cca`
* source revision
  `3e352cca660658d4b5c90f42a7808b11469e4c66`
* fixed Haar features
  `/workspace/thymic_project/experiments/frequency_source_vs_risk_audit_20260713/frequency_features.float32.npy`
* frequency metadata
  `/workspace/thymic_project/experiments/frequency_source_vs_risk_audit_20260713/frequency_feature_metadata.csv`

These assets are already present and require no transfer.

Rebuild the PE dense bank once under the locked H3 extractor. Expected immutable output hashes are:

* dense features:
  `e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34`
* valid mask:
  `af92eb26a78f5b563c50ede322287d5f0342bcefb022ca6f301b9447ab07a48c`
* spatial shapes:
  `14ed638d5194353da15b52d16a65f8363869259cb4431680510ee888a3406e9f`

A mismatch stops execution and triggers an engineering audit; it does not authorize training on the mismatched bank.

### 5.3 Training-only nuisance environment construction

For each outer fold independently:

1. Select only the outer-training cases.
2. Let (X\in\mathbb{R}^{n\times156}) be their fixed Haar feature matrix.
3. Construct a design matrix containing:
   [
   D=[1,\ y,\ I(\text{batch2}),\ I(\text{third_batch})],
   ]
   omitting source columns absent from the training fold.
4. Fit multivariate least squares on the outer-training set:
   [
   \hat B=(D^\top D)^+D^\top X.
   ]
5. Compute residual nuisance features:
   [
   R=X-D\hat B.
   ]
   This removes the training-set mean effects of risk and acquisition source; no validation or test case contributes to (\hat B).
6. Standardize each column of (R) with training-only mean and standard deviation. Columns with standard deviation below (10^{-8}) are set to zero.
7. Fit `PCA(n_components=1, svd_solver="full")` on the standardized training residuals.
8. Orient the PC deterministically: if the sum of its 156 loadings is negative, multiply the component and scores by (-1).
9. Within every **source × binary-risk** training stratum, sort cases by `(PC1 score, case_id)`. Assign the lower half to nuisance bin 0 and the upper half to nuisance bin 1; for odd counts, the median case belongs to bin 1.
10. Define:
    [
    e_i=(\text{source}_i,\text{nuisance-bin}_i).
    ]
    Source-LODO therefore has four training environments; five-fold training has up to six.
11. Preflight requirement: every environment must contain at least four low-risk and four high-risk cases. A fold failing this requirement aborts; no alternate cluster count, split rule or PCA dimension is allowed.

The Haar features and environment labels never enter the image network and are never required at deployment.

### 5.4 Architecture and tensor flow

For case (i), the bank supplies:

[
X_i\in\mathbb{R}^{6\times1024\times1024},
\qquad
M_i\in{0,1}^{6\times1024}.
]

The shared visual head is exactly H3:

1. Feature LayerNorm over dimension 1,024.
2. Linear projection (1024\rightarrow128).
3. GELU.
4. Dropout 0.10.
5. Learned 128-dimensional embedding for each of the six views.
6. Mask invalid padded tokens before the attention softmax.
7. Gated attention:
   [
   a_j=v^\top\left[\tanh(W_tz_j)\odot\sigma(W_sz_j)\right],
   \quad
   \alpha_j=\operatorname{softmax}(a_j).
   ]
8. Pool:
   [
   h_i=\operatorname{LayerNorm}\left(\sum_j\alpha_j z_j\right)
   \in\mathbb{R}^{128}.
   ]

This is the same low-capacity H3 head that produced 0.8003 OOF BAcc; only its final classifier changes.

#### Rank-1 CSD classifier

Trainable parameters:

[
W_c\in\mathbb{R}^{2\times128}, \quad b_c\in\mathbb{R}^{2},
]
[
W_s\in\mathbb{R}^{2\times128\times1}, \quad b_s\in\mathbb{R}^{2\times1},
]
[
\gamma_e\in\mathbb{R}\quad\text{for each training environment}.
]

Common logits:

[
\ell_i^{(c)}=W_ch_i+b_c.
]

Training-environment logits:

[
\ell_i^{(e)}
============

\left(W_c+\gamma_{e_i}W_s[:,:,0]\right)h_i
+b_c+\gamma_{e_i}b_s[:,0].
]

At validation, source-LODO test, five-fold test, retrospective external stress testing and deployment:

[
p_i=\operatorname{softmax}(\ell_i^{(c)})_1.
]

`W_s`, `b_s`, nuisance labels and every (\gamma_e) are discarded.

### 5.5 Parameter count

Exact shared H3 parameters:

* input LayerNorm: 2,048;
* projection: 131,200;
* view embeddings: 768;
* gated attention: 16,577;
* output LayerNorm: 256;
* shared subtotal: **150,849**.

CSD classifier:

* common weights and bias: 258;
* rank-1 specific weights and bias: 258;
* environment coefficients: 4 in LODO or at most 6 in five-fold.

Total:

* source-LODO: **151,369 trainable parameters**;
* five-fold: at most **151,371**.

H3 had 151,107 parameters, so H6 adds at most **264 parameters**.

PE-Spatial itself remains fully frozen.

### 5.6 Losses

For a balanced training batch (B):

[
L_{\text{common}}
=\frac{1}{|B|}\sum_{i\in B}
\operatorname{CE}(\ell_i^{(c)},y_i),
]

[
L_{\text{specific}}
=\frac{1}{|B|}\sum_{i\in B}
\operatorname{CE}(\ell_i^{(e)},y_i).
]

For class (k\in{0,1}), form:

[
Q_k=
\begin{bmatrix}
W_c[k,:]^\top & W_s[k,:,0]^\top
\end{bmatrix}
\in\mathbb{R}^{128\times2}.
]

Use the normalized orthonormality penalty:

[
L_{\text{orth}}
===============

\frac{1}{4C}
\sum_{k=1}^{C}
\left|Q_k^\top Q_k-I_2\right|_F^2,
\qquad C=2.
]

Locked total:

[
\boxed{
L
=

0.5L_{\text{common}}
+0.5L_{\text{specific}}
+0.05L_{\text{orth}}
}
]

No class weights, focal term, subtype term, source adversary, contrastive term, calibration loss or sensitivity surrogate is permitted.

### 5.7 Initialization

* Shared H3 head: identical PyTorch initialization used in the existing H3 script.
* View embeddings: normal distribution, standard deviation 0.02.
* (W_c,W_s): Xavier uniform.
* (b_c,b_s): zeros.
* NA-CSD nuisance-bin coefficients: initialize all bin-0 environments to (-1) and bin-1 environments to (+1); coefficients remain trainable.
* Primary seed: `20260714`.
* A second seed, `20260715`, is prohibited unless every primary advancement gate passes.

### 5.8 Sampling and optimization

* Sampler: exact source×risk inverse-frequency `WeightedRandomSampler` used by H3.
* Samples per epoch: number of outer-training cases, with replacement.
* Batch size: 8.
* Workers: 0.
* Optimizer: AdamW.
* Learning rate: (3\times10^{-4}), constant.
* Weight decay: (10^{-4}) on the shared head, (W_c) and (W_s); zero on biases and (\gamma_e).
* Scheduler: none.
* Maximum epochs: 80.
* Early-stopping patience: 12 epochs.
* Gradient clipping: global norm 5.0.
* AMP: FP16 autocast and GradScaler; logits, softmax, cross-entropy and orthogonality penalty evaluated in FP32.
* CUDA deterministic flags and all Python/NumPy/PyTorch seeds locked.

### 5.9 Outer splits and checkpoint selection

#### Source-LODO primary

For held source (s):

* test: all cases from (s);
* validation: the locked next master fold among the other sources;
* training: all remaining cases from the other sources.

All nuisance residualization, PCA, binning, sampling and fitting use only outer-training cases.

#### Five-fold secondary

For test fold (k):

* test: fold (k);
* validation: fold (k+1), cycling after fold 5;
* training: remaining three folds.

#### Checkpoint selection

Use only **common-head validation BAcc at threshold 0.5**.

Lexicographic tie rule:

1. greater common validation BAcc;
2. if equal within (10^{-8}), greater common validation sensitivity;
3. if still equal, earlier epoch.

Environment-specific validation logits, external cohorts and held-out test predictions cannot select a checkpoint.

### 5.10 Required identifying control

Run exactly one new control:

**`SOURCE_ONLY_CSD`**

* same PE bank;
* same shared head;
* same rank, loss, initialization, optimizer, sampler, seed and splits;
* environment ID is acquisition source only;
* no frequency residualization, PCA or nuisance bins.

Existing H3 is the matched ordinary-ERM comparator. No additional dropout, rank, orthogonality weight or clustering ablations are allowed.

Interpretation:

* NA-CSD > H3 and source-only CSD: supports nuisance subdomains plus decomposition.
* NA-CSD ≈ source-only CSD > H3: supports decomposition, not nuisance subdomains; preregistered primary still fails its mechanism-specific gate.
* NA-CSD ≤ H3: NO-GO.

### 5.11 Server command plan

```bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

ROOT=/workspace/thymic_project
EXP=${ROOT}/experiments/h6_nuisance_anchored_csd_20260714
REGISTRY=${ROOT}/experiments/base_model_expansion_20260706/outputs/registry/task7_four_domain_master_registry.csv
SPLIT=${ROOT}/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv
FREQ=${ROOT}/experiments/frequency_source_vs_risk_audit_20260713
PE_BANK=${EXP}/pe_dense_bank

mkdir -p "${EXP}"/{logs,manifest,aggregate}
```

#### Build and lock the bank

```bash
python ${ROOT}/scripts/extract_task7_h3_dense_bank_20260713.py \
  --backend pe \
  --registry-csv "${REGISTRY}" \
  --model-id /root/model_weights/modelscope/facebook/PE-Spatial-L14-448/PE-Spatial-L14-448.pt \
  --canonical-model-id facebook/PE-Spatial-L14-448 \
  --weight-sha256 47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1 \
  --model-code-dir /root/third_party/perception_models_3e352cca \
  --code-revision 3e352cca660658d4b5c90f42a7808b11469e4c66 \
  --output-dir "${PE_BANK}" \
  --domains old_data,third_batch \
  --views whole,crop,crop_q0,crop_q1,crop_q2,crop_q3 \
  --max-num-patches 1024 \
  --device cuda \
  --seed 20260714 \
  --local-files-only \
  2>&1 | tee "${EXP}/logs/extract_pe_bank.log"
```

Create a manifest containing SHA-256, byte size and semantic configuration for:

* checkpoint;
* PE source revision;
* dense bank, mask and spatial shapes;
* registry;
* split;
* Haar features and metadata;
* H6 trainer and analyzer;
* C2/H3 prediction files.

#### Run source-LODO first

```bash
for VARIANT in nuisance_csd source_csd; do
  python ${ROOT}/scripts/run_task7_h6_nuisance_csd_20260714.py \
    --feature-bank-dir "${PE_BANK}" \
    --frequency-features "${FREQ}/frequency_features.float32.npy" \
    --frequency-metadata "${FREQ}/frequency_feature_metadata.csv" \
    --split-csv "${SPLIT}" \
    --integrity-manifest "${EXP}/manifest/integrity.json" \
    --variant "${VARIANT}" \
    --split-mode source_lodo \
    --output-dir "${EXP}/source_lodo/${VARIANT}" \
    --seed 20260714 \
    --device cuda \
    2>&1 | tee "${EXP}/logs/source_lodo_${VARIANT}.log"
done
```

#### Run five-fold without changing configuration

```bash
for VARIANT in nuisance_csd source_csd; do
  python ${ROOT}/scripts/run_task7_h6_nuisance_csd_20260714.py \
    --feature-bank-dir "${PE_BANK}" \
    --frequency-features "${FREQ}/frequency_features.float32.npy" \
    --frequency-metadata "${FREQ}/frequency_feature_metadata.csv" \
    --split-csv "${SPLIT}" \
    --integrity-manifest "${EXP}/manifest/integrity.json" \
    --variant "${VARIANT}" \
    --split-mode fivefold \
    --output-dir "${EXP}/fivefold/${VARIANT}" \
    --seed 20260714 \
    --device cuda \
    2>&1 | tee "${EXP}/logs/fivefold_${VARIANT}.log"
done
```

#### Aggregate

```bash
python ${ROOT}/scripts/analyze_task7_h6_nuisance_csd_20260714.py \
  --h6-root "${EXP}" \
  --h3-oof "${H3_OOF_PREDICTIONS}" \
  --h3-lodo "${H3_LODO_PREDICTIONS}" \
  --c2-oof "${C2_OOF_PREDICTIONS}" \
  --c2-lodo "${C2_LODO_PREDICTIONS}" \
  --output-dir "${EXP}/aggregate" \
  --bootstrap-repetitions 20000 \
  --seed 20260714
```

The four prediction-path variables must be resolved from the existing server manifests before execution.

### 5.12 Output tree and recovery

```text
h6_nuisance_anchored_csd_20260714/
├── manifest/
│   ├── integrity.json
│   └── integrity.sha256
├── pe_dense_bank/
│   ├── dense_features.float16.npy
│   ├── valid_token_mask.uint8.npy
│   ├── spatial_shapes.int16.npy
│   ├── processed.uint8.npy
│   ├── metadata.csv
│   └── dense_bank_config.json
├── source_lodo/
│   ├── nuisance_csd/fold_{1,2,3}/
│   └── source_csd/fold_{1,2,3}/
├── fivefold/
│   ├── nuisance_csd/fold_{1..5}/
│   └── source_csd/fold_{1..5}/
├── aggregate/
│   ├── model_comparison.csv
│   ├── source_comparison.csv
│   ├── subtype_comparison.csv
│   ├── bootstrap_vs_h3.csv
│   ├── bootstrap_vs_c2.csv
│   ├── mechanism_diagnostics.csv
│   ├── decision.json
│   └── RESULTS.md
└── logs/
```

Recovery rules:

* PE extraction resumes from `processed.uint8.npy`; it flushes every five cases, as already implemented.
* A completed fold is skipped only when its prediction file, summary, checkpoint and configuration hashes all agree.
* A crash during a head epoch restarts that fold from epoch 1 with the locked seed; no partial optimizer state is reused.
* Checkpoints are written to a temporary file and atomically renamed.
* A hash mismatch never triggers automatic overwrite.
* After the final aggregate and hash lock, the 6.93-GiB regenerable dense bank may be deleted.

### 5.13 Compute and storage estimate

* PE dense bank: **6.93 GiB** float16.
* Mask: approximately 3.5 MiB.
* Sixteen head fits: two variants × eight outer folds.
* Head checkpoints, predictions and logs: well below 1 GiB.
* Expected peak new disk: **8–9 GiB**.
* Expected GPU time: **5–12 hours**, depending mainly on PE extraction speed and early stopping.
* Expected VRAM: comfortably below the available 48 GB because PE extraction is one case/view at a time and training uses frozen bank features.
* No package, checkpoint or dataset download.

### 5.14 Required outputs and privacy

At threshold 0.5, report:

* Acc, BAcc, AUC;
* sensitivity and specificity;
* TN, FP, FN, TP;
* each held-source BAcc, sensitivity and specificity;
* minimum held-source BAcc;
* six subtype risk accuracies;
* B1 and B2;
* fold metrics;
* paired source×risk-stratified bootstrap against H3 and C2;
* NA-CSD versus source-only CSD;
* common-versus-environment-specific validation performance;
* (|\gamma_eW_s|_F/|W_c|_F) by fold;
* coverage, which must equal 591/591.

Images, Haar rows, PE arrays, case predictions and weights remain server-only. GitHub receives only code, hashes, aggregate tables and the decision report.

---

## 6. One conditional backup experiment

### Final-classifier Fishr

Run Fishr **only** when all of the following mechanistic trigger conditions hold:

1. In every LODO fold,
   [
   \operatorname{median}_e
   \frac{|\gamma_eW_s|_F}{|W_c|_F}
   \ge 0.10,
   ]
   showing that the residual branch did not collapse.
2. Environment-specific validation logits exceed common validation BAcc by at least 0.03 in at least two of the three LODO folds.
3. NA-CSD nevertheless fails because common-head LODO sensitivity or B2 remains below its advancement floor.

That pattern would mean source-specific decision structure exists, but is not well described by one low-rank classifier direction.

The backup uses:

* exact H3 PE bank and head;
* actual acquisition sources only;
* ordinary common classifier;
* per-example final-layer gradient
  [
  g_i=\operatorname{vec}\left[(p_i-\operatorname{onehot}(y_i))h_i^\top\right];
  ]
* centered gradient variance (v_d) within each source;
* EMA decay 0.95;
* Fishr penalty
  [
  L_{\text{Fishr}}
  ================

  \frac{1}{D}
  \sum_d
  \left|
  \operatorname{EMA}(v_d)
  -----------------------

  \frac1D\sum_{d'}\operatorname{EMA}(v_{d'})
  \right|_2^2;
  ]
* total loss:
  [
  L=\operatorname{CE}+\lambda(t)L_{\text{Fishr}},
  ]
  where (\lambda(t)=0) for the first 1,500 optimizer updates and 1,000 thereafter;
* all other H3 optimizer, sampler, epoch, seed, threshold and split settings unchanged.

The fixed penalty, annealing point and EMA match the official Fishr defaults. ([GitHub][19]) Because the final-layer gradients have a closed form, BackPACK is not required.

If the trigger is absent, no backup runs. If Fishr fails any advancement gate, current-data visual development stops without LISA, SoRA or another coefficient search.

---

## 7. Advancement gates and stopping rule

### 7.1 Why the gates cannot rely only on superiority confidence intervals

The primary source-LODO sample contains:

* 223 high-risk cases;
* 368 low-risk cases;
* 62 B1 cases;
* 89 B2 cases.

One changed case equals:

* 0.45 percentage points of sensitivity;
* 0.27 points of specificity;
* 1.61 points of B1 accuracy;
* 1.12 points of B2 accuracy.

A useful 1–2 point BAcc improvement is therefore unlikely to produce a 95% superiority interval excluding zero. Requiring that would be underpowered; ignoring uncertainty would be overly permissive. The locked rule consequently combines integer-case clinical floors, directional consistency and bootstrap noninferiority.

### 7.2 Mandatory source-LODO gates

Every gate must pass:

| Gate                         |                                                                                             Requirement | Interpretation                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------------------: | ---------------------------------------------------------------------------------- |
| **LODO BAcc**                |                                                                                            **≥ 0.7641** | At least +0.0102 over H3 and +0.0200 over C2.                                      |
| **Sensitivity**              |                                                                                            **≥ 0.7354** | At least 164/223 high-risk cases correct; recovers C2.                             |
| **Specificity**              |                                                                                            **≥ 0.7800** | At least 288/368 low-risk cases correct; retains substantially more of H3 than C2. |
| **B1 accuracy**              |                                                                                            **≥ 0.6129** | At least 38/62; no more than two cases below H3.                                   |
| **B2 accuracy**              |                                                                                            **≥ 0.6629** | At least 59/89; recovers C2.                                                       |
| **Minimum held-source BAcc** |                                                                                            **≥ 0.7381** | No source below H3’s current minimum.                                              |
| **Source direction**         |                                                                  At least 2/3 sources improve versus H3 | Excludes a one-source average gain.                                                |
| **Maximum source harm**      |                                                        No source decreases by more than 0.015 versus H3 | Tighter than the previous 0.02 collapse rule.                                      |
| **Bootstrap versus H3**      |                                       Mean ΔBAcc > 0; (P(\Delta>0)\ge0.80); 95% CI lower bound > −0.010 | Power-aware superiority direction plus 1-point noninferiority.                     |
| **Sensitivity versus C2**    |                                                            95% CI lower bound for ΔSensitivity > −0.020 | Excludes clinically important sensitivity loss.                                    |
| **Mechanism control**        | NA-CSD BAcc ≥ source-only CSD +0.005, and NA-CSD has higher sensitivity or B2 without losing >1 B1 case | Required to credit nuisance anchoring rather than decomposition alone.             |
| **Coverage/threshold**       |                                                                                          591/591 at 0.5 | Non-negotiable.                                                                    |

### 7.3 Secondary five-fold retention gates

All must pass:

* BAcc ≥ **0.7903**, no more than one point below H3’s 0.8003;
* sensitivity ≥ **0.7800**;
* specificity ≥ **0.7800**;
* B1 ≥ **0.6452**, at least 40/62;
* B2 ≥ **0.6742**, at least 60/89;
* no fold BAcc below 0.70.

These prevent a model that restores LODO merely by destroying the strongest mixed-source direct capability.

### 7.4 Confirmation seed

Only after every primary and secondary gate passes:

* run seed `20260715`;
* no architecture, rank, environment rule, coefficient, epoch or optimizer change;
* the confirmation seed must independently pass all point-estimate gates;
* the two-seed mean must also pass them.

A failure on the confirmation seed is NO-GO.

### 7.5 Hard stopping rule

* **All gates pass:** engineering GO; freeze H6 before any retrospective external run.
* **Gate failure plus the exact Fishr trigger:** run the single Fishr backup.
* **Gate failure without the trigger:** stop.
* **Fishr failure:** stop.
* No rank 2, PCA dimension, nuisance-bin count, orthogonality coefficient, loss weight, seed series, threshold, fusion or LISA search follows.

---

## 8. Evidence ceiling and manuscript-safe wording

### Engineering improvement on fixed data

A passing H6 result would support:

> A low-capacity common/specific readout improved the locked PE-Spatial classifier on the fixed 591-case development cohort while preserving 100% coverage and threshold 0.5.

That is a valid engineering statement.

### Internal cross-batch robustness

Source-LODO would support:

> The improvement was retained when each of the three available acquisition batches was withheld from fitting in turn.

It must still be called **internal acquisition-batch robustness**, not multicenter or independent external generalization. The H3 report already makes this distinction explicit.

### Consumed external stress tests

After H6 is frozen by internal gates, the 108- and 162-case cohorts may be run once as transparent retrospective stress tests. The report must state:

* both cohorts were previously inspected;
* they were not used for H6 training, checkpoint selection, environment construction, thresholding or advancement;
* results are descriptive;
* no model modification follows them within this study.

Safe wording:

> The locked model was additionally evaluated on two previously inspected historical cohorts as retrospective stress tests. These results are not independent confirmatory validation.

### Claim that cannot be made

Without a fresh cohort, the paper cannot claim:

* prospective external validation;
* independent multicenter confirmation;
* generalization to unseen hospitals;
* a clinically validated sensitivity improvement;
* readiness for autonomous diagnosis;
* proof that source nuisance has been causally removed.

### Manuscript-safe positive wording if H6 passes

> In a fixed 591-case development cohort spanning three acquisition batches, a preregistered nuisance-anchored common/specific classifier improved fixed-threshold source-held balanced accuracy while recovering high-risk sensitivity and B2 performance relative to the strongest PE-Spatial and balanced visual references. The result constitutes exploratory internal cross-batch robustness evidence. Because the available external cohorts had been inspected previously, no independent external-validation claim is made.

### Manuscript-safe wording if H6 fails

> Explicit separation of training-environment-specific classifier directions did not improve the source-held low/high-risk boundary. Together with the completed representation, augmentation, alignment, texture, expert and spatial-model experiments, this result shows that no tested method established a reproducible improvement in cross-batch visual capability on the fixed cohort.

---

## 9. Missing repository fact that materially blocks implementation

The architecture, local checkpoint, Haar features, bank hashes, data totals and split logic are sufficiently specified. The only material operational information absent from GitHub is the **exact server path and immutable hash for the case-level C2 and H3 OOF/LODO prediction files**, plus confirmation that the authoritative registry and split paths still match the prior H3 run manifests. Those case-level files are required for paired bootstrap and exact one-to-one comparison, although they are not required to train H6.

Before coding, the server agent must resolve and lock:

```text
H3_OOF_PREDICTIONS
H3_LODO_PREDICTIONS
C2_OOF_PREDICTIONS
C2_LODO_PREDICTIONS
authoritative registry path + SHA-256
authoritative split path + SHA-256
```

If any reference prediction file is missing, the experiment must not substitute aggregate metrics or reconstruct case predictions from published tables. It must either recover the original locked server artifact or regenerate the corresponding locked reference under its original immutable configuration before H6 begins.

[1]: https://arxiv.org/abs/2003.12815 "https://arxiv.org/abs/2003.12815"
[2]: https://arxiv.org/abs/2401.03002 "https://arxiv.org/abs/2401.03002"
[3]: https://arxiv.org/abs/2109.02934 "https://arxiv.org/abs/2109.02934"
[4]: https://arxiv.org/abs/2201.00299 "https://arxiv.org/abs/2201.00299"
[5]: https://arxiv.org/abs/2008.12839 "https://arxiv.org/abs/2008.12839"
[6]: https://arxiv.org/abs/2302.02350 "https://arxiv.org/abs/2302.02350"
[7]: https://arxiv.org/abs/2307.07181 "https://arxiv.org/abs/2307.07181"
[8]: https://arxiv.org/abs/2104.09937 "https://arxiv.org/abs/2104.09937"
[9]: https://arxiv.org/abs/2106.02266 "https://arxiv.org/abs/2106.02266"
[10]: https://arxiv.org/abs/2203.10789 "https://arxiv.org/abs/2203.10789"
[11]: https://arxiv.org/abs/2412.04077 "https://arxiv.org/abs/2412.04077"
[12]: https://arxiv.org/abs/2102.08604 "https://arxiv.org/abs/2102.08604"
[13]: https://arxiv.org/abs/2304.01973 "https://arxiv.org/abs/2304.01973"
[14]: https://arxiv.org/abs/2007.01434 "https://arxiv.org/abs/2007.01434"
[15]: https://arxiv.org/abs/2304.00629 "https://arxiv.org/abs/2304.00629"
[16]: https://arxiv.org/abs/2007.02454?utm_source=chatgpt.com "Self-Challenging Improves Cross-Domain Generalization"
[17]: https://arxiv.org/abs/2502.05593?utm_source=chatgpt.com "Semantic Data Augmentation Enhanced Invariant Risk Minimization for Medical Image Domain Generalization"
[18]: https://arxiv.org/abs/2604.25817?utm_source=chatgpt.com "Magnification-Invariant Image Classification via Domain Generalization and Stable Sparse Embedding Signatures"
[19]: https://github.com/alexrame/fishr/blob/main/domainbed/hparams_registry.py "fishr/domainbed/hparams_registry.py at main · alexrame/fishr · GitHub"
