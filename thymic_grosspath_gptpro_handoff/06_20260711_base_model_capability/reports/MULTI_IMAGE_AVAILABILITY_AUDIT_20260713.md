# Task7 Multi-Image Availability Audit (2026-07-13)

## Question

Can the next direct-classifier experiment replace one selected photograph per
case with a case-level bag of all available gross photographs?

## Method

The server-side registries were audited without emitting case IDs or paths. The
corrected script checked:

- all 591 selected cached image paths;
- every source-directory reference after joining its dataset-specific root;
- every image path in the existing all-image case-bag registry;
- direct and recursive resolution of each selected original filename;
- the registry's existing `original_image_count` field.

An earlier version of this audit incorrectly treated `source_case_folder` as
an absolute path. It is a relative source-directory field. Third-batch rows use
`source_folder` instead. The earlier conclusion that all source folders were
unavailable was therefore a path-resolution error, not a data-loss event.

No image, path mapping, or case-level row was copied from the server. The
aggregate output remains at:

`/workspace/thymic_project/experiments/h5_second_order_texture_20260713/data_availability_audit/case_image_availability.json`

## Result

- All three dataset roots are mounted and readable.
- All 591 source-directory references resolve: 117/117 batch-1, 168/168
  batch-2, and 306/306 third-batch rows.
- All 591 cached selected images and all 591 corresponding selected originals
  resolve. Eleven batch-1 originals require recursive lookup within the
  recorded source directory; the other 580 resolve directly.
- The existing case-bag registry contains 608 currently accessible images for
  the 591 internal cases:
  - old data: 285 cases, 302 images, 17 two-image cases;
  - third batch: 306 cases, 306 images, no multi-image cases.
- The raw dataset roots contain 168, 174, and 306 image files, respectively.
  These root totals include files excluded from the 591-case Task7 cohort.
- The source-directory fields have only 22 unique values and are shared
  class/source directories, not one unique physical folder per case.

## Decision

No recovery or remount is needed. The de-identified all-image case-bag manifest
already exists and its 608 internal image paths are currently valid.

This correction changes feasibility, but not the information-content caveat:
only 17/591 cases have a second view, all in the old-data domain. A conventional
MIL model would therefore receive exactly one image for 574 cases and cannot be
claimed as a broadly trained multi-view solution. The 17 paired cases can still
support a tightly scoped paired-view sensitivity or consistency analysis.
