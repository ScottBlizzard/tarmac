# GPT Pro Response After H3-H5 No-Go

Complete GPT Pro response pasted on 2026-07-13.

# Ranked decision

## 1. **No current-data classifier experiment remains scientifically defensible**

**Primary next action: freeze model development on the 591-case cohort and launch a standardized, multicenter, multi-view acquisition protocol with a sealed external test cohort.**

This is not because every conceivable image transform has been tried. It is because none of the untried proposals now has the combination of:

1. a sufficiently distinct and independently motivated visual mechanism;
2. a control that would make either a positive or negative result broadly interpretable;
3. enough independence from the repeatedly inspected OOF/source-LODO system to justify another model-advancement claim.

H3 demonstrated that a substantially stronger representation can increase mixed-source BAcc from 0.7514 to 0.8003, but its held-source sensitivity and B2 accuracy declined and its LODO improvement was statistically unresolved. H4 then made the quality-robustness intervention worse under LODO, including a statistically negative change relative to H3. H5 showed that its texture branch was functional—it improved mixed-source B1/B2—but the information was acquisition-dependent and B2 collapsed under source shift. These are not three generic architecture failures; they triangulate the same problem: the model can repeatedly find additional predictive structure in the current photographs, but that structure does not remain risk-discriminative when an acquisition batch is excluded.

Earlier evidence also closes the main routes by which another same-photo model could plausibly claim to recover missed information. Explicit true-grid spatial relations were worse than their controls under LODO; native-detail tiles, hierarchical MIL, and cross-attention failed across sources; label-driven attention ROI and label-free anatomy ROI did not transfer; and random local bags mainly improved ranking while moving errors between B1, B2, and sources.

A further problem is project-level adaptive reuse. A new preregistration could prevent tuning *within* the next run, but it cannot make the 591 cases or the three repeatedly inspected acquisition batches fresh again. The reported bootstrap intervals are conditional on each chosen model; they do not account for the long sequence of model families proposed after observing earlier OOF and LODO failure modes. Source-LODO also has only three batches from one hospital, not independent center-level replication. Another internally positive model would therefore remain an exploratory nomination requiring genuinely new confirmation, while another negative model would usually reject only one arbitrary implementation.

The governing H3 protocol reached the same eligibility boundary: intermediate-layer fusion, second-order texture, spatial re-embedding, and SAFT were allowed only after an H3B candidate passed; otherwise representation optimization on the single-photo cohort was to close. Neither candidate passed.

## Repository correction on image availability

The inaccessible-folder statement in the message has been superseded by the current repository. The corrected root-aware audit says all three roots are mounted, all 591 selected originals and all 608 registered case-bag image paths resolve, and the earlier failure resulted from treating relative paths as absolute. However, 574 cases still have exactly one image and only 17 have a second image, all in the old-data domain; third-batch cases are all single-image. No remount is needed, but the information-content limitation remains.

Thus a broad all-image MIL experiment is **technically possible but scientifically uninformative as a multi-view training test**. Image count is almost perfectly “one,” and the only additional-view cases are confined to one historical domain.

# Audit of the seven remaining ideas

## Unsupervised part discovery

**Ruling: useful engineering or hypothesis-generation audit; not a valid current-data efficacy experiment. Conditionally valid after new data or blinded part annotations.**

Unsupervised token clustering can discover salient foreground structure, but generic methods such as normalized-cut discovery identify self-similar foreground objects; they do not establish that the resulting regions correspond to diagnostic cut surface, capsule, interface, viable tumor, or heterogeneity. ([arXiv][1])

On this cohort, a classification implementation would reopen the already negative localization family with a different proposal generator. Label-driven attention ROI, label-free anatomy ROI, fixed high-resolution regions, random bags, and true-grid relational modeling have already shown that rearranging or selecting regions from the same photograph readily produces domain-specific rather than transferable evidence. A label-free part-map audit could still answer questions such as whether discovered regions are stable across augmentations and batches or merely separate specimen from background. It must not nominate a classifier.

**Download dependency:** none if implemented with the cached PE tokens. A valid model experiment would need new multi-view images, independent ROI/part annotations, or both.

## Selective high-resolution subtoken refinement

**Ruling: already invalidated as a current-photo model direction.**

A1 supplied the classifier with a whole image, specimen crop, four medium-scale regions, eight fine-scale regions, and dense tokens, using both hierarchical MIL and global-to-local cross-attention. Both high-resolution variants deteriorated under source-LODO and were positive in zero of three held-source comparisons. Attention ROI and label-free anatomy ROI also failed to provide transferable region selection.

A coarse PE selector followed by a higher-resolution reread combines two failed assumptions—automatic region selection and same-photo detail refinement. It would become materially different only if the high-resolution image were a **newly acquired standardized close-up** or the region source came from independent, blinded annotations.

**Download dependency:** no new checkpoint is intrinsically required. The scientific family, rather than the asset, is closed.

## RGB plus frequency/wavelet auxiliary tokens

**Ruling: not logically disproven, but methodologically invalid as the next classifier experiment. A source-versus-risk frequency audit would be legitimate.**

This is the one idea that is not literally identical to a completed implementation. Raw multiscale frequency coefficients are different from covariance of PE semantic tokens, and fixed wavelet-scattering representations can preserve high-frequency information while gaining some translation and deformation stability. ([arXiv][2])

Nevertheless, it fails the present value-of-information test:

* H4 directly perturbed downsampling, blur, JPEG compression, photometry, and white balance—operations that alter spectral content—and the resulting consistency objective significantly worsened transfer relative to H3.
* H5 showed that additional texture information can be learned in mixed-source folds yet be strongly source-dependent, producing a large B2 and sensitivity loss under LODO.
* “Wavelet tokens” still leaves many scientifically consequential choices: color space, wavelet family, number of scales, decimation, phase or energy encoding, local tokenization, normalization, and fusion. A failure of one arbitrary locked choice would not close the frequency hypothesis; an internal success would remain vulnerable to the accumulated model-selection history.
* Frequency features are at least as exposed to camera, focus, compression, distance, and resizing differences as the representations already implicated in batch-dependent behavior.

A legitimate non-model audit could quantify whether wavelet-band energy is more separable by acquisition batch than by risk after source stratification. That could guide the capture protocol. It should not be used to produce another candidate classifier.

**Download dependency:** none. That low implementation cost does not compensate for the weak inferential value.

## PE intermediate-layer extraction

**Ruling: methodologically invalid continuation of closed H3.**

The PE paper’s central observation is that useful embeddings can reside in intermediate layers, and its spatial alignment was designed to expose dense spatial information. ([arXiv][3]) The repository implementation already uses the `PE-Spatial-L14-448` checkpoint and records its feature source as the **aligned final dense layer**, calling `forward_features(..., norm=True, strip_cls_token=True)`.

Most importantly, H3 explicitly allowed early/middle/final fusion only after H3B passed. It did not. Running layer combinations now would be precisely the prohibited post-gate H3 sweep, regardless of the PE paper’s general motivation.

**Download dependency:** none; the existing checkpoint and official source are sufficient. The block is methodological, not technical.

## SAM2/Hiera-SLCA

**Ruling: blocked by an undocumented checkpoint dependency and, even if cached, a low-information continuation of the backbone/spatial-attention search.**

There is also a nomenclature issue. The published SLCA classification method uses a frozen **SAM ViT-H** encoder, extracts patch, early, middle, late, and convolutional features, and injects their spatially localized channel attention into a separate classifier. It is not an evaluated SAM2/Hiera-SLCA recipe. ([arXiv][4])

A SAM2/Hiera implementation would therefore be an extrapolation. Official SAM2 requires its code and a checkpoint; the Hiera-L model is approximately 224 million parameters. The repository documents SAM2+SLCA as deferred custom architecture but does not establish that a compatible checkpoint is cached. Unless a server-side asset audit proves otherwise, this proposal requires a new model/code download and conflicts with the billable-transfer constraint.  ([GitHub][5])

Even with a cached checkpoint, its claimed mechanism is another combination of new backbone, intermediate features, and spatial attention after broad backbone, ROI, attention, relational, and multi-layer families have failed. It would become more defensible with new multicenter data and independent segmentation or ROI targets.

## LP-FT followed by SAFT

**Ruling: methodologically invalid continuation of the closed adaptation branch.**

LP-FT is motivated by evidence that direct fine-tuning can distort strong pretrained features under distribution shift and that fitting a linear head first can reduce that distortion. SAFT then updates only a small gradient-selected subset of parameters—reported around 0.1% in its original work—to preserve pretrained generalization. ([arXiv][6]) ([arXiv][7])

That general rationale is not enough here. H3 preregistered LP-to-SAFT only if a frozen H3B representation passed. It did not. The project also has direct negative evidence from conservative LoRA, broader LoRA, source-balanced LoRA, and other adaptation variants. On 591 batch-confounded cases, selecting weights by training-fold gradient magnitude could just select parameters most responsive to the acquisition-specific correlations that already produce the OOF/LODO reversal.

SAFT changes optimization; it does not introduce independent visual evidence. Running it now would be a closed-family search.

**Download dependency:** no new checkpoint if the existing PE assets are used.

## Gross-pathology self-supervised pretraining

**Ruling: real visual-capability hypothesis, but blocked by insufficient new data. This is the strongest of the seven ideas for a future cohort.**

Domain-specific self-supervision could plausibly change what the encoder represents rather than merely changing its readout. But the current asset is only 608 images from 591 cases, overwhelmingly one image per case and one hospital. The prior fold-wise VICReg initialization did not improve the final classifier.

A stronger SSL run on current data faces two unacceptable choices:

* Pretrain separately within every outer training fold or LODO training partition, leaving only a few hundred unique cases per run—too little diversity for a credible gross-pathology representation renewal.
* Pretrain on all 591 cases, including the nominally held-out fold or batch. That would make OOF/LODO transductive and invalidate comparison with the locked inductive protocol, even without using labels.

This direction becomes valid with a large, center-diverse unlabeled development pool that excludes every external-test center and patient.

**Download dependency:** new image data are required. A new large model checkpoint is not necessary if SSL starts from the cached PE or another existing local encoder.

# Why there is no current-data preregistration

I am deliberately not providing architecture, storage, runtime, and success gates for another current-data model. A detailed preregistration cannot restore independence or scientific eligibility to a proposal whose relevant mechanism has already failed, whose prerequisite gate was not met, or whose negative result would reject only one arbitrary implementation.

The stopping rule should now be:

> No new model may be nominated from the existing 591 cases. Label-free engineering audits may inform the acquisition protocol, but their results cannot reopen model selection. Model development reopens only after the new-data package below passes its completeness and center-diversity gates.

This is consistent with H5’s interpretation that another covariance, head-capacity, or loss search is not justified and that the next valid model experiment needs additional image information or a genuinely new cohort.

# Minimum new-data package to reopen development

These are **minimum credible design floors**, not clinical-deployment sufficiency.

## Image protocol

Collect four mandatory, fixed-slot, full-resolution photographs per case:

1. **Whole-specimen overview**, with fixed background, scale reference, and color reference.
2. **Dominant complete cut-surface overview**.
3. **Standardized close-up of the largest viable solid cut-surface region**, selected by a geometric rule rather than perceived malignancy.
4. **Capsule/tumor–adjacent-tissue interface close-up**; use a predefined second-cut-surface fallback when no interface is visible.

One slot—preferably the complete cut-surface overview—must be designated as the **primary single image before pathology is known**. This supports the mandatory mechanism control: the same encoder and training pipeline using that one image versus all four images.

Additional photographs can be retained, but the primary experiment should not depend on variable photographer-selected image counts.

## Per-case completeness

Target 100% four-view completion. Do not reopen model training until:

* at least **95% of enrolled cases have all four views**;
* every development center has at least **90% completeness**;
* all eligible cases remain in the manifest, including acquisition failures;
* missing-view reasons are recorded before label access;
* the model has a fixed masked-slot fallback so it still produces a prediction for every enrolled case.

All raw files, repeats, and rejected captures must be retained. A retrospective “best photograph” selection should not define the model input.

## Centers

Use at least:

* **four development hospitals**, each following the same capture protocol;
* **two completely untouched external-test hospitals**, preferably three.

The original hospital may be one development center only for newly acquired standardized cases. The historical 591 cases can remain a legacy auxiliary stratum, but they do not count as multicenter multi-view evidence.

No development center should dominate the sample, and B1, B2, and both risk classes must occur at multiple centers. Otherwise subtype and center remain inseparable.

## Approximate labeled case counts and subtype balance

A credible reopening development set is approximately **800 new labeled cases** across the four development centers. A practical target is:

| Subtype   | Approximate development target |
| --------- | -----------------------------: |
| A         |                             60 |
| AB        |                            200 |
| B1        |                            140 |
| B2        |                            170 |
| B3        |                             80 |
| TC        |                            150 |
| **Total** |                        **800** |

This deliberately enriches B1, B2, and B3 and gives approximately balanced low/high risk. It is a modeling design, not a prevalence estimate. Accrual should be consecutive where possible, with any rare-subtype enrichment rule declared before labels are released to the modeling team.

For gross-pathology SSL from scratch or substantial continued pretraining, 800 labeled cases are still not enough. That branch should require at least approximately **5,000 unlabeled cases or 20,000 images from five or more development centers**, with patient grouping and no images from the external-test institutions.

## Acquisition metadata

Record, but do not feed to the primary classifier:

* hospital and acquisition date or era;
* operator role;
* camera/phone body, lens, focal length, and native resolution;
* file format and compression history;
* flash, illumination setup, white-balance mode, and exposure;
* camera-to-specimen distance;
* background, ruler, and color-card presence;
* fresh versus fixed state;
* specimen orientation and view-slot identifier;
* number of captures and repeat reason;
* crop, resize, enhancement, export, and de-identification history;
* missing-view reason.

Final histopathologic subtype should have a fixed adjudication protocol, with central review or dual review focused particularly on mixed cases and the B1/B2 boundary. Those review fields remain labels or audit variables, not image-model inputs.

## Untouched external test

Reserve approximately **600 fresh cases** from two or three hospitals not represented in supervised training, self-supervision, normalization fitting, architecture selection, or acquisition-protocol tuning. A reasonable subtype target is approximately:

* A 50;
* AB 150;
* B1 100;
* B2 130;
* B3 60;
* TC 110.

That yields roughly 300 low-risk and 300 high-risk cases. With about 300 observations per risk class, a class recall near 0.75 has an approximate 95% interval half-width of five percentage points. A 600-case paired test is also near the lower end of what may detect a five-point improvement when comparator/candidate discordance is moderate; the final power calculation should use discordance from the locked internal comparator predictions.

The external procedure should be:

1. Freeze model family, preprocessing, weights, code hash, view handling, and threshold 0.5 before receiving test images.
2. Use a separate non-test technical packet to validate file transfer and decoding.
3. Prohibit SSL, normalization fitting, test-time adaptation, thresholding, or model selection on external images.
4. Deposit 100%-coverage predictions and hashes with an independent custodian.
5. Merge predictions with labels only after the deposit.
6. Report each center separately, pooled BAcc/AUC/sensitivity/specificity, every subtype, B1/B2, and paired center-by-risk-stratified bootstrap comparisons.
7. Compare the multi-view model against the **matched designated-primary-view model**. That paired comparison—not a probability stack—is the mandatory control isolating the added visual information.

# Final classification of the work

**Real visual-capability hypotheses:** standardized multi-view learning on newly acquired images; gross-pathology SSL on a large multicenter development pool.

**Useful engineering audits:** corrected image-manifest integrity; label-free part-map stability; source-versus-risk frequency analysis; prospective capture-completeness monitoring. These cannot nominate a model.

**Methodologically invalid continuations:** selective high-resolution rereading, PE layer selection, LP-FT/SAFT, current-data wavelet classifiers, and another current-photo part-based classifier.

**Blocked work:** broad multi-view learning and domain-specific SSL are blocked by insufficient view multiplicity and center diversity—not by inaccessible original folders. SAM2/Hiera-SLCA is additionally blocked unless a compatible local checkpoint is verified; otherwise it requires a new checkpoint/code download.

The only ranked primary action is therefore to **lock the current experiment ledger and begin the four-view, multicenter acquisition with the external centers sealed before any new model is fit**.

[1]: https://arxiv.org/abs/2202.11539 "Self-Supervised Transformers for Unsupervised Object Discovery using Normalized Cut"
[2]: https://arxiv.org/abs/1203.1513 "Invariant Scattering Convolution Networks"
[3]: https://arxiv.org/abs/2504.13181 "https://arxiv.org/abs/2504.13181"
[4]: https://arxiv.org/abs/2505.06217 "https://arxiv.org/abs/2505.06217"
[5]: https://github.com/facebookresearch/sam2 "https://github.com/facebookresearch/sam2"
[6]: https://arxiv.org/abs/2202.10054 "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution"
[7]: https://arxiv.org/abs/2407.03036 "https://arxiv.org/abs/2407.03036"
