# Input Order Invariance Audit

## Purpose

This audit shuffles the feature table repeatedly and confirms that final runtime decisions
do not change. It protects against accidental dependence on CSV row order in large tie groups.

## Result

- baseline decision rows: 1398
- shuffle seeds: 0, 1, 2, 7, 13, 29, 101
- mismatch rows: 0