# Decision log

Each methodological choice records its date, decision, reason, and rejected alternative. A consequence is included where it changes later work. Historical proposals are distinguished from the researcher's operational rules and from decisions explicitly confirmed by the supervisor. A recorded rule does not imply that the reference set is frozen or that the supervisor approved an unasked detail.

## 2026-10-06 — Provisional historical crown-stage reproduction

**Decision.** Attempt the historical crown stage on the eight training sites using the hash-checked supplied source, its unchanged `segment_chunk` function and default segmentation constants. Use the original metropolitan grid established by the CHM manifest and an immutable read-only frozen seed database. Acquire current OSM masks explicitly and retain their query responses; record unavailable historical greenness and mask provenance. Keep any mask-free smoke result separate.

**Why.** A bounded training-only attempt can produce candidate geometry and quantify agreement with the frozen crown summary attributes while the producing revision and original crown geometry remain unconfirmed. The original CHM manifest establishes the metropolitan extent, 1 m pixels and 1,024 m chunks; the current default pilot configuration alone would choose a different extent.

**Rejected.** Writing to the frozen database, downloading replacement CHM, silently discarding masks, claiming current OSM responses are historical inputs, fabricating a freeze record, and treating scalar equality as proof of crown-geometry identity were rejected.

**Consequence.** The attempt is provisional and does not close M3. It reconstructs the crown stage from retained frozen inventory seeds, not the full detection/canonicalisation history. Reference evaluation remains training-only and all formal acceptance thresholds remain unchanged.

## 2026-10-06 — Explicit incomplete-mask approximation

**Decision.** After repeated whole-cell, subdivided-query, mirror and POST acquisition failures, permit a separate training-only attempt with all nine building responses and eight of nine nonvegetation responses. Record the missing query cell and response filename explicitly; keep strict cache completeness as the default.

**Why.** The missing current OSM response is an external acquisition failure. Available masks and frozen seeds still support a useful bounded crown-stage diagnostic while exact historical inputs remain unavailable.

**Rejected.** Fabricating an empty successful response, silently omitting the mask, deleting failed run records, or describing this approximation as the frozen default baseline was rejected.

**Consequence.** The candidate carries incomplete-mask status and cannot establish exact reproduction or close M3. If the response becomes available, the complete-mask attempt must use a new output directory and retain the earlier evidence.

## 2026-07-16 — Initial acceptance thresholds

**Decision.** Use detection precision and recall of at least 0.85, median matched-crown IoU of at least 0.60, false-positive precision of at least 0.95, and recall of at least 0.70 in every stratum. Declare an F1 improvement over the classical baseline only when its bootstrap 95% confidence interval excludes zero. The original plan also specified a 2 m match tolerance and seed 42; `config/project.json` records the thresholds as predeclared at M1 on 7 August.

**Why.** These project-specific thresholds were fixed before benchmark model results so that the outcome could not determine the pass criteria.

**Rejected.** Choosing or relaxing thresholds after inspecting validation or test performance was rejected. These values are not asserted to be universal standards.

**Consequence.** Operational definitions may be made explicit before held-out scoring, but the numerical thresholds remain unchanged.

## 2026-07-16 — Initial reference-set design

**Decision.** Start with 24 sites of 100 × 100 m, eight strata, roughly 20 crown annotations per site, 400 m minimum site separation, a 4 km spatial-block split and a 1 km buffer between different splits. Allocate approximately equal thirds to training, validation and test.

**Why.** Each stratum needs a held-out site for the predeclared stratum-recall criterion, while spatial separation reduces leakage and the site size keeps manual annotation feasible.

**Rejected.** Random point sampling was rejected because it could over-represent common suburban canopy and omit palms, conifers or hard-negative built settings.

**Consequence.** The design was an initial proposal subject to a site-rule review; the approximate 500-crown planning target was not a requirement to manufacture 500 independent tree instances.

## 2026-07-27 — Review of reference-site strata and sampling rules

**Decision.** Retain the eight-stratum, 24-site design for the pilot and M1 scope without changing the selection code. It includes uncommon and difficult cases, places one train, one validation and one test site in every stratum, and reproduced the balanced 8/8/8 split from the shipped aggregate cache.

**Why.** Stratification avoids over-sampling common suburban canopy. The site size, 4 km spatial blocks, 1 km split buffer and 400 m minimum separation balance feasibility with reduced spatial leakage.

**Rejected.** Changing the rules before testing a complete site annotation was rejected because it would require regenerating chips without evidence that the design failed. Simple random sampling and choosing only the densest cells were rejected because they would reduce coverage of known failure modes or make labelling infeasible.

**Consequence.** Use the selected pilot sites, while reporting that strata are proxies from an imperfect inventory and that crown-count-based selection may inherit model dependence.

## 2026-09-17 — Standalone Nectar execution

**Decision.** Run benchmark jobs directly on the standalone Nectar GPU VM, with separate output directories, logs, timings and run IDs.

**Why.** The supervisor confirmed that this VM is the intended compute environment; it has no default SLURM scheduler.

**Rejected.** Submitting the historical `.sbatch` example to this VM or implying that a scheduler was used was rejected.

**Consequence.** The VM wrapper and run receipts provide the execution record. The confirmed V100 training-tile runs are documented under `docs/cluster_runs/`.

## 2026-09-17 — Frozen classical comparison

**Decision.** Compare against the 30 July frozen classical baseline, keeping its SQLite snapshot read-only. Exact reproduction and tuning wait for the confirmed producing code, parameters and corresponding crown geometry.

**Why.** The live database and historical scripts may not correspond to the frozen outputs. The delivered source checkout is explicitly labelled historical pending provenance verification.

**Rejected.** Comparing with a changing live database, treating the historical commit as the verified producer, or running a writing script against the frozen snapshot was rejected.

**Consequence.** Existing detection caches and database records can be audited now; M3 is not complete until the frozen result is reproduced and tuning uses training sites only.

## 2026-09-17 — Stem and apex metric scope

**Decision.** Remove stem/apex location error from the core evaluation while retaining all other predeclared acceptance thresholds.

**Why.** The reference contains manually delineated crowns but no independently observed stem or apex positions. The supervisor explicitly approved this scope change.

**Rejected.** Treating crown centroids or model-detected peaks as independent stem/apex ground truth was rejected.

**Consequence.** The methods and results report crown-based detection and delineation; stem/apex error is not used to declare a winner.

## 2026-09-18 — Tree groups and the annotation target

**Decision.** Count `tree_group` polygons as annotation work and report them separately from individual-tree metrics. Consider splitting only where an individual crown boundary can be defended from independent imagery and height evidence.

**Why.** The supervisor accepted 492 original polygons as annotation work, identified only 219 original individual references, and requested split-by-stratum counts and quality review before deciding whether a small top-up is needed.

**Rejected.** Counting a group as one verified tree, forcing equal or peak-only partitions to approach 500, or assuming the top-up decision had already been made was rejected.

**Consequence.** The accepted pre-freeze revision contains 234 individual trees and 216 groups; it remains separate from the 219-tree original and is not a frozen reference.

## 2026-09-18 — Preserve original blind labels and version refinements

**Decision.** Keep all 492 original XZ polygons unchanged, and store the 14 accepted group refinements as a distinct pre-freeze derivative. Compare the supervisor's S016/S020 blind export with the original XZ version before adjudicating either site.

**Why.** Replacing original labels before the independent comparison would erase the observation needed to measure inter-rater agreement. Separate versions also preserve parent-child lineage and the author's acceptance record.

**Rejected.** Overwriting the original JSON, silently using the later 243-tree proposal, or comparing the supervisor against a post-discussion refinement was rejected.

**Consequence.** The working derivative has 495 objects, including 234 single trees; eight second-pass proposals remain unaccepted and the formal reference freeze remains pending.

## 2026-09-18 — Large groups and uncertain regions

**Decision.** Retain the eleven original groups above the 900 m² review flag as groups after RGB/CHM inspection. Keep all four unresolved `uncertain` polygons and exclude them from scored reference objects while recording their ambiguity regions.

**Why.** A large, connected canopy does not reveal a defensible number of individual trees. The four uncertain locations lack enough visual evidence to establish either a tree or background object.

**Rejected.** Promoting a large group or an uncertain polygon to a single tree solely to increase the denominator, deleting the uncertain geometry, or treating it as a confirmed hard negative was rejected.

**Consequence.** The review and counts are in `docs/REFERENCE_SET_FREEZE_REVIEW.md`; predictions ignored in ambiguity regions are counted explicitly.

## 2026-09-18 — Independent-rater matching

**Decision.** Compute S016/S020 original-label agreement by one-to-one crown matching at IoU ≥ 0.50, then report F1, matched IoU and class agreement where defined.

**Why.** The agreement calculation compares two independently drawn polygon sets and must remain separate from the model's 2 m detection rule.

**Rejected.** Replacing independent matching with detector-centroid matching, using synthetic labels as real agreement, or comparing with already adjudicated XZ polygons was rejected.

**Consequence.** Agreement cannot be populated until the supervisor's original blind JSON arrives; undefined denominators or kappa remain undefined rather than being replaced by zero.

## 2026-09-18 — Operational model matching and ambiguity rule

**Decision.** Before held-out model scoring, apply the predeclared 2 m tolerance to centroids of crowns clipped to the same site. Make one-to-one candidates in descending IoU, then ascending distance and stable ID order. Match eligible single-tree references first; only an otherwise unmatched prediction whose centroid lies in a group or uncertain region is ignored.

**Why.** The fixed ordering makes ties reproducible, and matching before ambiguity exclusion prevents a prediction overlapping a valid single tree from disappearing merely because a group is nearby. This is a researcher-fixed operational interpretation, not an individually confirmed supervisor rule.

**Rejected.** Changing tolerance after viewing test results, counting group interiors as independent trees, or ignoring predictions in selected `not_tree` polygons was rejected.

**Consequence.** Report scored TP/FP/FN and ignored prediction IDs separately; `config/evaluation_draft.json` records the implementation status.

## 2026-09-18 — Paired site bootstrap

**Decision.** Use 2,000 paired whole-site resamples with PCG64 seed 42 and a percentile 95% confidence interval for model differences.

**Why.** Sites are the spatial sampling units. Resampling the same site draw for both models preserves pairing and makes the F1 improvement rule executable.

**Rejected.** Resampling individual crowns as though spatially independent, using different site draws for the models, or claiming a site-bootstrap interval for a stratum with one test site was rejected.

**Consequence.** Recalculate pooled metrics within each draw and report small-stratum denominators without unsupported stratum-specific intervals.

## 2026-09-18 — Selective hard negatives and precision

**Decision.** Retain the 41 `not_tree` annotations as selected hard-negative examples, not an exhaustive background map. Interpret the predeclared “false-positive precision ≥ 0.95” using the same scored detection precision TP/(TP+FP), while preserving the separate 0.85 precision threshold and reporting commission errors.

**Why.** A selective set of negative polygons cannot define the denominator for all possible false detections. The research-side interpretation was recorded before held-out outputs; it was not separately confirmed by the supervisor.

**Rejected.** Treating every unlabelled background area as a verified negative, ignoring a prediction inside `not_tree`, inventing a second precision denominator from 41 examples, or changing the 0.95 threshold was rejected.

**Consequence.** Hard-negative cases support error analysis; ordinary scored FP across the eligible domain determine precision.

## 2026-09-20 — Training-site GPU runs are technical development

**Decision.** Use the pretrained V100 training-tile and density runs to audit point restoration, crown conversion, failure modes and timing. Keep provisional training-site metrics out of the locked benchmark table.

**Why.** The reference was not frozen, the crown conversion was a draft, and only part of the full site/tile workflow had run. Four fully tile-covered training sites supply just 26 working individual references.

**Rejected.** Calling one-tile diagnostics held-out predictions, selecting a density condition from test results, or declaring M3/M4 from a successful GPU exit code was rejected.

**Consequence.** Preserve run IDs and hashes for reproducibility; formal validation/test predictions and acceptance claims wait for the required locks.

## 2026-09-22 — Checked 3D labels for local fine-tuning

**Decision.** Prepare SegmentAnyTree fine-tuning labels only from manually checked training-site point-cloud regions. Assign each verified tree its own point-level `treeID` and semantic class; keep unresolved group points out of stock supervised training until the relevant loss paths have been tested.

**Why.** The 39 LAZ tiles have XYZ but no `treeID` or `semantic_seg`. A 2D crown polygon or CHM height cannot establish which vertically overlapping points belong to one tree. The upstream trainer expects point-level semantic and instance labels.

**Rejected.** Extruding crowns through all Z values, assigning one ID to a whole unresolved group, mapping ASPRS class directly to the upstream tree taxonomy, or treating pretrained predictions as independent ground truth was rejected.

**Consequence.** Pretrained inference can proceed without these labels. Fine-tuning remains pending a checked training PLY, its QA record, format validation and a one-batch loss-path test; reference freeze is a separate evaluation decision. See `docs/SAT_POINT_LABEL_STRATEGY.md`.

## 2026-09-22 — Second-round tree-group review

**Decision.** Accept six additional single-crown revisions: `S001-ms2lqiho`, `S004-ms2ppqgf`, `S007-mspevauw`, `S008-mspfzo9n`, `S009-mspgeven`, and `S010-mspgnaw8`. Keep them with the previously accepted 14 parent-object changes in a new pre-freeze working export. Preserve the original XZ annotations for the blind S016/S020 comparison.

**Why.** The author reviewed the second-round comparison gallery and explicitly selected these six objects. The export retains exact source decisions, parent-child lineage and geometry checks, so the accepted working reference can be identified without replacing the original labels.

**Rejected.** The proposed revisions to `S009-mspge9kf` and `S018-mss1xwhr` were declined. Their original `tree_group` labels remain active; the declined before/after panels remain in the review gallery for audit. Treating all eight second-round suggestions as accepted was rejected.

**Consequence.** The working export now has 495 objects: 240 individual crowns, 210 groups, 41 selected non-tree regions and four uncertain regions (train/validation/test individual counts 77/74/89). It is not the frozen reference, and the supervisor's blind annotations have not yet been compared.

## 2026-09-24 — Current author pre-freeze export and derived scoring geometry

**Decision.** Use `crown_labels_XZ_2026-09-22-2.json` as the current author version for the freeze discussion while retaining the original 219-tree export for the first XZ/TW blind comparison. Build a separate software-compatible candidate by applying `buffer(0)` only to invalid polygons and then clipping every object to its fixed site boundary.

**Why.** The new manual export contains 523 objects: 249 trees, 229 groups, 41 selected non-tree regions and four uncertain regions. Three raw polygons self-intersect and one group extends 0.0393 m² beyond its site, while the evaluator requires valid site-bounded geometry. A derived layer permits reproducible training-only QA without altering the manual source or claiming that topology repair resolves biological identity.

**Rejected.** Continuing to use the historical 240-tree evaluation layer as the current denominator, overwriting the browser JSON, silently dropping invalid objects, treating repaired geometry as independent adjudication, or using the candidate for locked validation/test scoring was rejected.

**Consequence.** The current counts are 82/76/91 trees in training/validation/test. Eight current groups exceed 900 m². `scripts/45_prepare_prefreeze_scoring_candidate.py` records all repairs, clipping, hashes and class counts; the resulting candidate remains explicitly unfrozen until independent comparison, adjudication and supervisor confirmation are complete.

## 2026-09-24 — Training-site multi-tile instance assembly

**Decision.** Treat instance numbers as tile-local. During training-site assembly, assign global instance numbers after exact XYZ deduplication and join positive instances across tiles only when the same XYZ record occurs in both tiles with one agreed semantic label. Select the duplicate record furthest from its source tile edge, use stable tile and row order for ties, and retain header-footprint coverage as a diagnostic layer.

**Why.** SegmentAnyTree can reuse the same numeric instance ID independently in different tiles, so concatenating IDs would merge unrelated crowns. Exact repeated points provide direct identity evidence at overlapping tile boundaries, while proximity or polygon overlap alone can join neighbouring crowns incorrectly. A deterministic winner and a recorded coverage domain make the conversion reproducible.

**Rejected.** Concatenating raw local IDs, joining instances by ID equality, proximity or crown overlap, using a header footprint as proof that point support is complete, applying the pre-freeze tool to validation or test sites, and using the deduplicated output for return-density accounting were rejected.

**Consequence.** `scripts/46_merge_training_site_tiles.py` now produces a hash-bound merged LAZ, coverage layers and a merge receipt for training-only crown-conversion diagnostics. It rejects incomplete tile lists, coordinate-audit failures, held-out tile conflicts and non-training sites. Two synthetic cases verify exact-overlap joining, ID-collision isolation and semantic-conflict handling; real multi-tile runs still require the corresponding reconciled tile predictions.

## 2026-09-28 — Provisional training-only crown conversion

**Decision.** Retain the 1.0 m buffered convex hull (`convex_b100`) as the provisional point-instance-to-crown conversion candidate. Select it only by the recorded all-training-site ranking rule: maximum micro F1, then pooled median matched IoU, fewer false positives and configuration ID. Recompute this selection if the frozen training reference differs from the current candidate, and lock the conversion before held-out scoring.

**Why.** Across all eight training sites and all 82 current pre-freeze training crowns, `convex_b100` had the highest micro F1 among the seven fixed candidates: 36 TP, 203 FP, 46 FN, F1 0.2243 and pooled median matched IoU 0.4174. Every input is bound to an audited merge receipt and no validation or test label was used.

**Rejected.** Selecting a method from validation or test results, calling these values benchmark accuracy, increasing the fixed 2 m matching tolerance after observing S008, and locking the conversion against an unfrozen reference were rejected.

**Consequence.** The candidate is suitable for continued training-only workflow development, but its low precision and two zero-TP sites remain explicit risks. It does not complete M4 or populate the dissertation's locked Results table.

## 2026-09-28 — No post hoc size filter for pretrained crowns

**Decision.** Do not add a minimum point-count or polygon-area filter to the provisional pretrained crown conversion.

**Why.** Matched and false-positive candidates have nearly identical median point counts (65.5 and 64) and polygon areas (31.95 and 31.72 m²). The observed distributions do not provide a defensible threshold that removes false positives without also removing true positives.

**Rejected.** Searching a threshold grid against the same 82 pre-freeze training crowns solely to improve the reported score, or applying an undocumented visual cutoff, was rejected.

**Consequence.** The weak false-positive performance remains visible. Any later filter must have an independent scientific basis, be selected on training data only and be locked before held-out evaluation.
