# Method and scope decisions

| Date | Decision | Consequence |
|---|---|---|
| Before model runs | Preserve the eight strata, spatial split, and original acceptance thresholds in `config/project.json`. | Thresholds are not adjusted after seeing model results. |
| 2026-09-17 | Run directly on a standalone Nectar GPU VM. | SLURM examples are historical; every run has a separate output directory and receipt. |
| 2026-09-17 | Compare with the frozen 30 July classical baseline. | The received snapshot is kept read-only; exact code and parameters are still being verified. |
| 2026-09-17 | Remove stem/apex location error from core evaluation because no independent location references were collected. | Other acceptance thresholds remain unchanged. |
| 2026-09-18 | Treat tree groups as annotation work but report them separately from individual-tree metrics. | No forced partition is made solely to increase the individual-tree count. |
| 2026-09-18 | Preserve 492 original polygons and create a separate accepted pre-freeze refinement. | The working layer has 234 individual trees; the reference is not frozen. |
| 2026-09-18 | Retain four unresolved `uncertain` polygons but exclude them from scored reference objects. | Ambiguity is recorded and predictions ignored there are counted explicitly. |
| 2026-09-18 | Use one-to-one IoU ≥ 0.50 for independent-rater agreement. | This threshold does not replace the model's median-IoU acceptance threshold. |
| 2026-09-18 | Use a fixed 2 m clipped-crown-centroid candidate tolerance for model detection, with a stable IoU-first one-to-one matching order. | A prediction inside a group/uncertain region is ignored only if still unmatched; selected `not_tree` examples do not trigger ignore. |
| 2026-09-18 | Use 2,000 paired whole-site bootstrap resamples with seed 42 for model comparisons. | Resample the same sites for each model and recalculate pooled metrics; one site per test stratum does not support stratum-specific site-bootstrap claims. |
| 2026-09-18 | Keep the selective `not_tree` annotations as hard-negative examples. | They do not define an exhaustive background denominator; ordinary precision uses scored TP and FP. |
| 2026-09-20 | Treat pretrained training-tile inference and density runs as technical development. | They do not establish validation/test accuracy or achievement of M3/M4. |

The 0.95 false-positive-precision wording is interpreted conservatively as the same scored precision TP/(TP+FP), without changing the threshold. The 18 September supervisor reply supported the reference-review plan but did not constitute a formal freeze. Final adjudication and any targeted top-up decision are still open.
