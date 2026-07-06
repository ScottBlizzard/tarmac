# Candidate Margin Audit

## Purpose

This audit inspects cases added when increasing the model-disagreement quota beyond the conservative 15% candidate.

## Margin Bands

- old_data 0.10->0.15: band n=4, errors=0, high-risk FN=0, score range=0.000-0.000.
- old_data 0.15->0.20: band n=4, errors=0, high-risk FN=0, score range=0.000-0.000.
- old_data 0.20->0.25: band n=3, errors=0, high-risk FN=0, score range=0.000-0.083.
- old_data 0.25->0.30: band n=4, errors=0, high-risk FN=0, score range=0.083-0.167.
- third_batch 0.10->0.15: band n=6, errors=0, high-risk FN=0, score range=0.000-0.000.
- third_batch 0.15->0.20: band n=6, errors=0, high-risk FN=0, score range=0.000-0.000.
- third_batch 0.20->0.25: band n=6, errors=0, high-risk FN=0, score range=0.000-0.000.
- third_batch 0.25->0.30: band n=6, errors=2, high-risk FN=2, score range=0.000-0.000.
- strict_external 0.10->0.15: band n=3, errors=0, high-risk FN=0, score range=0.000-0.000.
- strict_external 0.15->0.20: band n=3, errors=0, high-risk FN=0, score range=0.000-0.000.
- strict_external 0.20->0.25: band n=3, errors=0, high-risk FN=0, score range=0.083-0.083.
- strict_external 0.25->0.30: band n=2, errors=0, high-risk FN=0, score range=0.083-0.083.

## Fold Margin Bands

- old_data 0.10->0.15: fold-band n=4, errors sum/max=0/0, high-risk FN sum/max=0/0.
- old_data 0.15->0.20: fold-band n=4, errors sum/max=0/0, high-risk FN sum/max=0/0.
- old_data 0.20->0.25: fold-band n=3, errors sum/max=0/0, high-risk FN sum/max=0/0.
- old_data 0.25->0.30: fold-band n=4, errors sum/max=0/0, high-risk FN sum/max=0/0.
- third_batch 0.10->0.15: fold-band n=6, errors sum/max=0/0, high-risk FN sum/max=0/0.
- third_batch 0.15->0.20: fold-band n=7, errors sum/max=1/1, high-risk FN sum/max=1/1.
- third_batch 0.20->0.25: fold-band n=7, errors sum/max=0/0, high-risk FN sum/max=0/0.
- third_batch 0.25->0.30: fold-band n=5, errors sum/max=1/1, high-risk FN sum/max=1/1.