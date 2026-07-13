# Task7 Multi-Image Availability Audit (2026-07-13)

## Question

Can the next direct-classifier experiment replace one selected photograph per
case with a case-level bag of all available gross photographs?

## Method

The server-side registry was audited without emitting case IDs or paths. The
script checked:

- all 591 selected cached image paths;
- every registered `source_case_folder`;
- direct and recursive image-file counts for common raster extensions;
- the registry's existing `original_image_count` field.

No image, path mapping, or case-level row was copied from the server. The
aggregate output remains at:

`/workspace/thymic_project/experiments/h5_second_order_texture_20260713/data_availability_audit/case_image_availability.json`

## Result

- All 591 selected cached images exist.
- None of the 591 registered `source_case_folder` paths is currently accessible
  on the server.
- `original_image_count` is present for 285 cases:
  - 268 cases have one original image;
  - 17 cases have two original images.
- `original_image_count` is missing for all 306 third-batch cases.

## Decision

A complete case-level all-image bag experiment cannot be launched from the
currently mounted server data. Treating the shared selected-image cache as a
multi-image source would be incorrect, and training a bag model only on the 17
known two-image cases would not test the intended hypothesis.

Before reopening this direction, recover or remount the original case folders,
rebuild a de-identified case-to-image manifest, and verify image-count
distributions separately by acquisition source. The existing single selected
photograph remains the only uniformly accessible input for all 591 cases.
