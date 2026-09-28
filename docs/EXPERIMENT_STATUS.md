# Experimental status at 29 September 2026

## Inputs and reference set

All 206 manifest-listed benchmark files and all 39 required LAZ tiles have
been checked. The tiles contain 66,265,220 returns. Site point-density and CHM
chip calculations are reproducible.

The original reference contains 492 polygons, including 219 individual trees.
The current author pre-freeze export contains 523 polygons: 249 individual
trees, 229 groups, 41 selected non-tree regions and four uncertain regions.
The current split counts are 82/76/91 individual trees for training,
validation and test. It remains unfrozen. The original 219-tree export remains
the correct first input for comparison with the pending S016/S020 independent
annotations.

## Classical baseline

The received 30 July frozen SQLite database passed read-only integrity and
table checks. Its 441 site-level detection records match the delivered caches.
The associated frozen crown-polygon geometry is absent, and the supervisor is
still checking which code commit and parameters produced the snapshot. Exact
reproduction and training-only tuning therefore remain open; M3 is not
complete.

## SegmentAnyTree pretrained development

The pinned pretrained model ran in a V100 container with a recorded image
digest, source commit and checkpoint hash. The original five-level S017 nested
density run and its coordinate/CRS audits are complete. In addition, all 14
source tiles needed by all eight training sites now have full-density,
coordinate-reconciled and audited pretrained predictions. The persistent
nine-tile queue completed with exit code 0 after 10:34:06 and processed
21,212,880 points. The earlier interrupted attempt is preserved and excluded.

The point outputs were assembled with a common 10 m site context. All eight
site merges completed; the three multi-tile sites removed only five exact
duplicate XYZ records in total and had no semantic conflicts. Site
header-envelope coverage ranges from 0.9216 to 1.0000 and remains a documented
limitation rather than a reason to alter a denominator.

Seven point-instance-to-crown configurations were compared across all eight
training sites, giving 56 training-only evaluations against 82 current
pre-freeze individual references. The provisional ranking selects a 1.0 m
buffered convex hull: TP 36, FP 203, FN 46, precision 0.1506, recall 0.4390,
micro F1 0.2243 and pooled median matched IoU 0.4174. These weak training
diagnostics are not held-out benchmark accuracy and do not establish M4.

Object-level audits identify different causes at the two zero-TP sites. S008
has positive instance points near all five references, but candidate centroids
fall outside the fixed 2 m rule; S024 has no nearby positive instance for its
single reference. Matched and false-positive candidates have nearly identical
median point counts and areas, so no post hoc size filter was added. The 2 m
tolerance was not changed after viewing results.

The run IDs, aggregate table, known failed attempts and evidence-archive hash
are recorded in the [queue receipt](cluster_runs/20260920T142229Z_gpu_v100_4319/TRAINING_MULTITILE_QUEUE_20260928.md)
and [conversion receipt](cluster_runs/20260920T142229Z_gpu_v100_4319/TRAINING_CONVERSION_PREFREEZE249_20260928.md).

## Current gates

The following work cannot be completed from the current inputs:

- inter-rater F1, matched IoU and class agreement; adjudication; formal
  reference freeze; and once-only validation/test scoring require the
  supervisor's S016/S020 blind export;
- exact classical reproduction and training-only tuning require confirmation
  of the producing code version, parameters and crown-polygon output or exact
  regeneration path;
- supervised local SegmentAnyTree fine-tuning requires checked point-level
  `semantic_seg` and per-tree `treeID` labels, or a supplied 2D-to-point
  assignment workflow followed by manual point-level QA.

No validation or test prediction has been used for parameter selection. No
formal acceptance threshold has been changed. M2, M3, M4 and M5 remain open.
