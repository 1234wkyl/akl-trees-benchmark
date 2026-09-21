# Auckland Airborne LiDAR Tree-Crown Benchmark

This private research repository contains the code, site definitions, original annotations, and versioned evidence for comparing a frozen Auckland classical detector with SegmentAnyTree on sparse airborne LiDAR. The benchmark uses 24 spatially separated 100 m × 100 m sites across eight landscape strata. Training, validation, and test sites are assigned before model evaluation.

## Current status — 22 September 2026

- The original annotation export contains 492 polygons: 219 individual crowns, 228 tree groups, 41 selected non-tree confusers, and four uncertain regions. A separate accepted **pre-freeze** refinement contains 234 individual crowns and 216 groups. Neither layer is the final frozen reference.
- Two independent blind annotations for S016 and S020 are pending. No inter-rater statistic or locked validation/test result is reported yet.
- The 30 July frozen classical database has been received and checked read-only. Its 441 site-level detection records match the site caches. The exact generating code/parameters and associated crown geometry are still being verified, so classical reproduction and training-only tuning remain open.
- The pinned SegmentAnyTree pretrained model has run on the standalone Nectar V100. Training-tile density and crown-conversion outputs are **development diagnostics**, not benchmark results. Fine-tuning requires checked point-level labels.

See [experimental status](docs/EXPERIMENT_STATUS.md), [reference freeze review](docs/REFERENCE_SET_FREEZE_REVIEW.md), and [decisions](docs/DECISIONS.md) for the evidence and scope of these statements.

## Repository layout

| Path | Purpose |
|---|---|
| `config/` | Fixed project thresholds and draft scoring/model metadata |
| `data/labels/` | Original author annotations; preserve unchanged for blind agreement |
| `data/sites/` | Site definitions, source-tile list, and RGB/CHM chips |
| `data/exports/` | Original derived reference geometry |
| `docs/group_refinement_2026-09-18/workflow_accepted234/` | Separate accepted pre-freeze working reference |
| `scripts/` | Annotation, geometry QA, evaluation, baseline intake, density, and model-output workflows |
| `scripts/hpc/` | Standalone GPU VM execution helpers |
| `tests/` | Synthetic and workflow checks |
| `labeller/` | Browser annotation and visual-review tools |

Large LAZ tiles, the frozen 11.54 GB baseline database, the upstream model source and checkpoint, personal communications, the dissertation, and VM output archives are held outside this repository. File hashes and run receipts are maintained with their associated external data. The site chips in this private repository are research inputs, not a public redistribution license.

## Quick start

```bash
conda env create -f environment.yml
conda activate akl-benchmark
python -m pytest -q
python scripts/00_check_environment.py
```

The browser labeller can be started with `make label`. Original annotations must be exported to a new file; do not overwrite `data/labels/crown_labels_XZ_2026-08-13.json`. The evaluation command and its required inputs are described in [RUNBOOK.md](RUNBOOK.md). The source project uses a standalone Nectar GPU VM directly; no SLURM scheduler is assumed.

## Evaluation boundaries

Only individual crowns enter the primary tree denominator. Unresolved groups and uncertain regions have explicit exclusion handling, while selected `not_tree` polygons support a supplementary confuser audit rather than an exhaustive background map. The 2 m model-matching tolerance and IoU ≥ 0.50 inter-rater matching rule serve different analyses. Predeclared thresholds in `config/project.json` are unchanged. No validation or test prediction may be used to revise labels, tune parameters, or choose the crown-conversion method.

This repository records an unfinished study. Empty dissertation result cells and missing benchmark metrics are intentional until reference adjudication, verified baseline reproduction, configuration lock, and once-only held-out runs are complete.
