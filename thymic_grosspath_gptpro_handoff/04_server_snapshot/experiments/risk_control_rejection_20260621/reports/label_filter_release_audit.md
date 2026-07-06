# Label Filter Release Audit

## Purpose

This audit tests whether the 15% quota candidate mainly works by releasing AB cases.

- ab_only / old_data: released=4, released errors=0, review=71, auto errors=0.
- ab_only / strict_external: released=8, released errors=0, review=48, auto errors=0.
- ab_only / third_batch: released=18, released errors=0, review=103, auto errors=1.
- all_labels / old_data: released=11, released errors=0, review=64, auto errors=0.
- all_labels / strict_external: released=8, released errors=0, review=48, auto errors=0.
- all_labels / third_batch: released=18, released errors=0, review=103, auto errors=1.
- non_ab_only / old_data: released=7, released errors=0, review=68, auto errors=0.
- non_ab_only / strict_external: released=0, released errors=0, review=56, auto errors=0.
- non_ab_only / third_batch: released=0, released errors=0, review=121, auto errors=1.