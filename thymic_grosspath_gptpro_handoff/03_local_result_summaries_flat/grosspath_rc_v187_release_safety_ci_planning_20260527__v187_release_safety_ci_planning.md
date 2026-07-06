# v187 Release Safety CI and Prospective Planning

## Current Wilson Boundary

- v185 adaptive all-domain auto decisions: n=287, errors=1, observed error rate 0.35%, Wilson95 upper 1.95%.
- v185 adaptive strict external auto decisions: n=14, errors=0, Wilson95 upper 21.53%.
- fixed v182 all-domain auto decisions: n=307, errors=1, Wilson95 upper 1.82%.

## Prospective Planning

- If future prospective auto decisions have zero errors, 35 auto-decided cases are needed for a Wilson95 upper bound <=10%, and 73 are needed for <=5%.
- If one error is allowed, 53 auto-decided cases are needed for <=10%, and 110 are needed for <=5%.

## Writing Boundary

Observed zero-error release should be reported with Wilson intervals and framed as a current-split safety estimate, not a guaranteed clinical error rate.