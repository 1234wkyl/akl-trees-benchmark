# Annotation and evaluation protocol

## Reference definition

The canonical author-confirmed export is `data/labels/crown_labels_XZ_confirmed_2026-10-10.json`. Coordinates use NZTM2000 (EPSG:2193). Its SHA-256 is `a8766662d51c16cf5cf87df3cbd8046512e1f493f2055f58374da3f878dbeb76`. The final scoring GeoPackage is a derived geometry product; repairs are recorded in its provenance and do not alter the raw JSON.

Delineate visible crown outlines using RGB and CHM context. Assign `tree` to a defensibly separable individual crown, `tree_group` to unresolved continuous canopy, `not_tree` to an explicitly annotated non-tree confuser and `uncertain` where interpretation remains unresolved. Decompose groups only where crown evidence supports individual boundaries; preserve remaining ambiguity. Group components are not biological tree counts. Alto is auxiliary context and may share provenance with annotations; it is not independent evaluation truth.

The reference contains 645 objects: 458 trees, 142 groups, 41 non-tree regions and four uncertain regions. Each stratum has one site per split. Single-tree counts are 147/141/170 for training/validation/test. Full split-by-stratum composition is in `results/reference_composition/`.

## Independent labels and subsequent revisions

Preserve the original XZ and TW JSON exports. Original blind comparison uses one-to-one greedy IoU matching at IoU at least 0.50. Original individual-tree F1 is 0.5882 and matched median IoU is 0.7412. TW uncertainty is excluded from tree scoring, with its coverage recorded.

After author revisions, correspondence is F1 0.8031 at IoU 0.50 and F1 0.8661 at IoU 0.40. These are post-review measures. IoU 0.40 is a sensitivity analysis. The operational F1 0.80/median IoU 0.60 review convention is post-hoc and is not a universally validated rater threshold. Revised correspondence must not replace the original independent reliability result. A degenerate matched-class kappa is not a whole-site quality verdict.

## Model scoring

Use the fixed operational policy in `config/evaluation_draft.json`. Eligible predictions and references are clipped to site support and matched one-to-one using the recorded 2 m crown-centroid tolerance and IoU/distance/ID tie order. Matched crown IoU is reported separately from detection proximity. Unmatched predictions whose centroids lie in group or uncertain regions are ignored and counted. Predictions in explicit `not_tree` regions remain eligible false positives. Non-tree coverage is selective, not an exhaustive background survey.

Precision is TP/(TP+FP), recall TP/(TP+FN), and F1 their harmonic mean. The operational false-positive precision uses the same scored TP/(TP+FP), with the predeclared stricter 0.95 threshold. Report per-stratum results, ignored counts, over/under-segmentation and paired whole-site bootstrap intervals (2,000 resamples, seed 42). Preserve the predeclared thresholds: precision/recall 0.85, matched median IoU 0.60, false-positive precision 0.95, minimum evaluable stratum recall 0.70, and a 95% paired F1-difference interval excluding zero for a winner. Stem/apex error is excluded because no independent location layer exists.

## Weak point labels

The original LAZ supplies actual XYZ returns. Training crowns guide candidate `semantic_seg` and `treeID` assignments; unresolved, overlapping or unsupported points remain unknown and are masked in the weak objective. Alto-assisted group/height evidence supplements these candidates. The final revised experiment has 158 candidate IDs and withdraws 134 assignments. These are partially assigned weak targets, not verified independent point ownership. Training, conversion selection and classical tuning use training sites only. Neither checkpoint is selected using test performance.
