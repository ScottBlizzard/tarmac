# Local Asset Audit After GPT Pro Decision (2026-07-13)

## Constraint

No new model, code, image, feature, or checkpoint download is authorized. This
audit inspected filenames, installed modules, cached model directories, and
existing experiment manifests only.

## SAM2 / Hiera-SLCA

- The `sam2` Python package is not installed.
- The `segment_anything` Python package is not installed.
- No SAM2 or Hiera checkpoint was found under the project, root model cache, or
  root download cache.
- The installed `timm` package exposes Hiera and SAM2-Hiera architecture names,
  including `sam2_hiera_{tiny,small,base_plus,large}`. These are architecture
  definitions, not proof that pretrained weights are local.
- Instantiating one of these models with pretrained weights would require a new
  transfer unless a compatible checkpoint is supplied separately.

Decision: SAM2/Hiera-SLCA remains blocked. Do not call a pretrained model
constructor or package installer for this family.

## PE-Spatial

- Local checkpoint:
  `/root/model_weights/modelscope/facebook/PE-Spatial-L14-448/PE-Spatial-L14-448.pt`
- Checkpoint size: 1,215,952,299 bytes.
- Locked SHA-256:
  `47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1`
- Official local source:
  `/root/third_party/perception_models_3e352cca`
- Source revision:
  `3e352cca660658d4b5c90f42a7808b11469e4c66`

The full H3 PE dense array was removed during regenerable-bank cleanup. Its
configuration and hashes remain, along with a two-case smoke bank. A diagnostic
audit can stream PE tokens from the local checkpoint and discard them per case;
it must not rebuild a multi-gigabyte bank unless a later locked protocol
requires one.

## Current permitted diagnostics

- Fixed Haar frequency source-versus-risk audit: no checkpoint dependency.
- PE label-free part-map stability audit: uses the cached PE checkpoint and
  streams tokens without retaining a bank.

Neither audit can nominate a current-cohort classifier.
