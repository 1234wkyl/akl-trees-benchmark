# Major research decisions

This log records decisions affecting research design, evaluation, model adaptation and delivery. Operational retries, document edits and interim annotation versions are recorded in run receipts instead. Dates identify the decision or its final consolidation.

## 2026-07-16 — Acceptance criteria and benchmark scope

**Decision.** Retain the M1 criteria: precision and recall at least 0.85, matched median crown IoU at least 0.60, false-positive precision at least 0.95, no evaluable stratum recall below 0.70, and the predeclared paired-bootstrap rule for a winner. Preserve the recorded 7 August M1 threshold date in the evaluation policy.

**Why.** Fixed criteria make the comparison interpretable across methods and density conditions.

**Rejected.** Relaxing acceptance thresholds after inspecting predictions or treating a small point estimate advantage as a confirmed winner.

**Consequence.** Completed experiments can constitute a valid negative finding even when model acceptance is not achieved.

## 2026-07-27 — Stratified and spatially separated reference sites

**Decision.** Retain eight strata and 24 sites, with one training, validation and test site per stratum. Preserve the shipped selection, spatial blocks, buffers and separation rules.

**Why.** The design covers common canopy and difficult cases while limiting spatial leakage. The original cache reproduces the balanced 8/8/8 split.

**Rejected.** Simple random selection, selecting only dense canopy, and changing sites after seeing model errors.

**Consequence.** Strata remain database-derived proxies, and selection from existing detections is disclosed as a possible source of model dependence.

## 2026-09-17 — Standalone GPU execution and fixed model provenance

**Decision.** Execute directly on the standalone Nectar V100 VM, following the supervisor's clarification. Record commands, run IDs, timing, separate output paths, environment and hashes. Use SmartForest-no/SegmentAnyTree at commit a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9 and the recorded PointGroup-PAPER.pt checkpoint.

**Why.** The allocated VM has no required SLURM scheduler. Source and checkpoint identities must remain explicit.

**Rejected.** Treating SLURM as mandatory on this VM, silently updating upstream code, or overwriting previous successful runs.

## 2026-09-17 — Remove unsupported stem/apex error

**Decision.** Remove stem/apex location error from core evaluation, with the supervisor's agreement; retain the other acceptance thresholds.

**Why.** The reference contains crown polygons and no independent stem or apex location layer.

**Rejected.** Presenting polygon centroids or model-derived apex positions as independently measured stems.

## 2026-09-18 — Separate single trees, ambiguity and selected hard negatives

**Decision.** Count tree groups as annotation work and report them separately. Match eligible single trees first; ignore unmatched predictions whose centroids fall in unresolved group or uncertain regions. Retain not_tree regions as explicit negative coverage and report ignored counts.

**Why.** Unresolved tree groups do not supply known individual identities. Selected non-tree regions are useful error examples but do not exhaustively label background.

**Rejected.** Counting every group as one reference tree, forcing a fixed number of instances, treating unknown interiors as background, or substituting a selective hard-negative audit for whole-site precision.

**Consequence.** The operational false-positive precision is the scored TP/(TP+FP); coverage and ambiguity limitations remain explicit.

## 2026-09-18 — Matching and uncertainty estimation

**Decision.** Use the recorded 2 m centroid-distance rule for model detection matching, with one-to-one assignment, and report matched crown IoU separately. Preserve IoU at least 0.50 for original object-level rater comparison. Use paired whole-site bootstrap with 2,000 resamples and seed 42.

**Why.** Detection proximity, crown boundary agreement and rater correspondence answer different questions. Resampling complete sites retains within-site dependence and pairing between models.

**Rejected.** Applying the model tolerance to raters without disclosure, treating every crown as an independent bootstrap unit, or changing primary matching rules to obtain a pass.

## 2026-09-28 — Training-only crown conversion and cross-tile assembly

**Decision.** Compare seven fixed conversion configurations on training sites and retain the selected convex hull with a 1.0 m buffer. Restore world coordinates, audit CRS and point coverage, and reconcile instances across source-tile boundaries before site evaluation.

**Why.** Point-instance predictions need a consistent two-dimensional representation. Training-only selection prevents choosing conversion rules from held-out performance.

**Rejected.** Tuning polygon conversion on validation/test results, scoring unmerged tile fragments as separate trees, or assigning reference contours to model predictions.

## 2026-10-06 — Frozen baseline and bounded historical reconstruction

**Decision.** Preserve the 30 July SQLite snapshot read-only. Reconstruct crown generation using frozen seeds and independently reconstruct detection using supplied historical code and recovered inputs. Report these as functional reconstruction evidence with explicit limits.

**Why.** The original polygon export and verified producer commit/input identity were not supplied. Frozen-seed generation recovers 126 candidates, with 125 matching four stored attributes; independent detection recovers 108 retained inferred detections, with 115 of 126 crown candidates matching those attributes.

**Rejected.** Treating matching summary attributes as proof of identical polygon vertices, unique historical parameters or exact source provenance.

**Consequence.** Historical reconstruction and the training-selected classical comparator remain distinct. Later authoritative provenance can update the reconstruction claim without altering preserved evidence.

## 2026-10-07 — Fixed density-transfer experiment

**Decision.** Use nested point-record retention at 100%, 75%, 50%, 25% and 12.5%, with the recorded seed and point-selection implementation. Keep model weights and conversion rules fixed across density levels.

**Why.** Nested sampling isolates sensitivity to fewer recorded LiDAR returns.

**Rejected.** Calling recorded-return density pulse density, interpreting thinning as a new sensor survey, or treating unchanged CHM inputs as a classical density experiment.

## 2026-10-09 — Preserve independent agreement and distinguish revised correspondence

**Decision.** Keep the original blind XZ and TW exports immutable. Report the original IoU 0.50 agreement separately from later author revisions. Treat IoU 0.40 as supplementary sensitivity, and F1 0.80/median IoU 0.60 as an operational post-review quality convention, not a universal literature threshold.

**Why.** Original agreement measures independent delineation; revisions informed by the returned annotations measure subsequent correspondence. Boundary sensitivity helps explain granularity differences.

**Rejected.** Relabelling revised agreement as independent reliability, omitting original results, copying returned coordinates to inflate agreement, or claiming a post-hoc threshold was predeclared.

**Consequence.** Original F1 is 0.5882 at IoU 0.50. Final revised correspondence is 0.8031 at IoU 0.50 and 0.8661 at IoU 0.40; these are reported with their different provenance.

## 2026-10-10 — Author-confirmed reference and consistent rescoring

**Decision.** Use crown_labels_XZ_confirmed_2026-10-10.json, SHA-256 a8766662d51c16cf5cf87df3cbd8046512e1f493f2055f58374da3f878dbeb76, as the author-confirmed analysis source. Derive scoring geometry reproducibly and rescore fixed predictions consistently.

**Why.** This version contains 645 objects: 458 trees (147/141/170 training/validation/test), 142 groups, 41 non-tree regions and four uncertain regions. Geometry repair is recorded separately from raw labels.

**Rejected.** Mixing reference versions between models, silently replacing original independent annotations, forcing 500 single trees, or claiming author confirmation retrospectively makes previously viewed held-out results blind.

**Consequence.** Reference locking and prior test exposure are both disclosed; supervisor ratification is not inferred from the author lock.

## 2026-10-10 — Training-selected classical setting and weak adaptation

**Decision.** Select the classical h3 setting on training sites. Adapt SAT with partially assigned point targets, unknown-target masking, fixed 80 updates and the recorded optimisation schedule. Retain the original pretrained comparator.

**Why.** Two-dimensional crowns guide candidate point ownership but do not provide complete independent three-dimensional truth. A masked objective supports partial supervision without fabricating labels for unresolved groups.

**Rejected.** Training on validation/test labels, labelling unknown group returns as background, describing candidate point IDs as checked field truth, or choosing the adapted checkpoint from test scores.

**Consequence.** Weak adaptation is a separate experiment with explicit annotation uncertainty. The classical h3 comparator is not identified as the original historical producer configuration.

## 2026-10-11 — Alto-assisted weak-label revision as a separate experiment

**Decision.** Use the cached Alto release metro-2026.10.10-pointcloud-v5.2 for auxiliary crown/height checks. Preserve original LAZ coordinates and point order; withdraw unsupported assignments to unknown. Preserve both adapted checkpoints and their predictions.

**Why.** Alto provides contextual structure, not a verified independent point-to-tree truth layer for these LAZ returns. The revised weak set retains 158 candidate IDs and withdraws 134 assignments. The separate repeat permits a direct comparison with the original weak pilot.

**Rejected.** Training on mesh vertices as if they were original returns, treating Alto-derived labels as independent evaluation truth, or replacing a weaker result with an earlier preferred experiment.

**Consequence.** Test F1 is 0.1818 after revision versus 0.1852 before; the paired difference interval includes zero. Improvement is not demonstrated. Fusion with newer Alto methods remains discussion/future work.

## 2026-10-12 — Research closeout and final delivery scope

**Decision.** Close out completed inference, density, classical and weak-adaptation experiments with their immutable receipts. Deliver a complete local research package and a separate minimal English GitHub release containing code, environments, reference annotations, protocols, manifests and compact results.

**Why.** All queued prediction experiments have completed. The study supports a negative/inconclusive performance finding; none of the models meets all acceptance criteria.

**Rejected.** Additional unplanned GPU runs and relaxing acceptance criteria.

**Consequence.** Document review, final word-count compliance and submission remain separate from completed experimental execution. Future model changes require a new experiment.
