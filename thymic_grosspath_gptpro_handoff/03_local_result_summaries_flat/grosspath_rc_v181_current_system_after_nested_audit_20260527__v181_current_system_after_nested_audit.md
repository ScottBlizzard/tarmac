# v181 Current System After Nested Audit

## Revised Recommendation

- Main safety baseline: v118, BAcc 99.81%, review 79.97%, FN=1, FP=0.
- Recommended efficiency candidate: v161, BAcc 99.81%, review 57.51%, released errors 0.0.
- Downgraded candidate: v178 full-fit looks good, but v180 nested audit gives released errors 3, BAcc 99.40%, review 48.21%.

## Paper Boundary

Do not promote v178 as the recommended operating point. Use it to show that we actively tested image-agreement release and rejected it after nested audit. This strengthens the rigor of the final framework.