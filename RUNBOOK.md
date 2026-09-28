# Reproducible workflow

This runbook identifies the active workflow and its required gates. Run commands from the repository root. Every model run needs a unique run ID, command, input hashes, stdout/stderr, exit code, timing, configuration, and output hashes. Use separate output directories. Run long GPU commands in a foreground terminal on the standalone Nectar VM.

## 1. Verify local inputs and tests

```bash
python scripts/00_check_environment.py
python -m pytest -q
python scripts/24_reproduce_manuscript_numbers.py --help
```

Check `data/sites/required_tiles.txt` against the 39 external LAZ tiles and `data/benchmark_input_sha256.txt` against the shared input bundle. The local site chips and original labels are versioned here. The frozen database and model weights are external inputs, not Git assets.

## 2. Reference review and independent agreement

The original JSON is `data/labels/crown_labels_XZ_2026-08-13.json`. The current author pre-freeze export is `docs/group_refinement_2026-09-18/export_20260922T060150894212Z/crown_labels_XZ_2026-09-22-2.json`; it contains 249 individual trees but is not frozen. Earlier 234/240/243 layers are historical drafts. Read the illustrated [freeze review](docs/REFERENCE_SET_FREEZE_REVIEW.md) before changing any label. The original 219-tree export remains the first XZ comparison input for the pending blind S016/S020 JSON.

When the independent JSON arrives, archive it unchanged, record SHA-256, and follow `scripts/31_receive_independent_labels.py --help`. Inspect every disagreement before assembling an adjudicated reference. Keep the original, proposed, and final layers distinct. No reference is frozen at present.

## 3. Frozen classical baseline

Read the 30 July SQLite database read-only. `scripts/10_inspect_frozen_baseline.py` and `scripts/22_export_received_baseline.py` inspect the snapshot and site-level detection records. The frozen generating commit, parameters, and crown-polygon output or exact regeneration path are still pending verification. Do not label the supplied historical script as the verified frozen pipeline. Once provenance is established, reproduce the frozen output first, then tune on training sites only. Preserve the frozen output and every trial separately.

## 4. SegmentAnyTree on the standalone GPU VM

Use `scripts/hpc/build_sat_image.py` and `scripts/hpc/run_sat_container.py` with the pinned upstream checkout and checkpoint recorded in `config/segmentanytree_source.json`. Record `nvidia-smi`, container image digest, checkpoint hash, code commit, driver, CUDA, and PyTorch versions. Prepare and validate one training tile, then use `scripts/13_prepare_density.py` and `scripts/hpc/run_density_training.py` for nested return retention. Do not infer crown polygons directly from point-instance scores without a fixed conversion. The current conversion grid is a training-only diagnostic and must be resolved before held-out inference.

Stage any additional explicitly named training tile with `scripts/47_prepare_training_tile.py`. The script requires the verified coverage plan, rejects tiles that touch held-out sites, preserves the raw LAZ, applies the already recorded ASPRS 7/18 exclusion by default, and writes input/provenance hashes. Run it under `scripts/hpc/run_vm.sh` so preparation also receives a run ID. Never use this entry point to prepare a validation or test tile.

After staging, `scripts/48_run_training_tile_queue.py` serially performs pretrained inference, point-coordinate reconciliation and the independent coordinate audit. Pass the immutable image digest and repeat `--tile` in the intended order. The queue stops on partial or failed existing output and updates `queue_receipt.json` after every successful state transition. Run it in a foreground VM terminal with `--execute`; omitting `--execute` performs staging validation and writes a plan-state receipt without calling Docker.

Full-site inference must cover every relevant source tile, deduplicate overlap, restore coordinates/CRS, clip to the site consistently, and report incomplete coverage. Fine-tuning can start only after checked semantic and instance point labels exist.

For a training site whose reconciled tile predictions and coordinate audits already exist, assemble the point-instance input for crown conversion with:

```bash
python scripts/46_merge_training_site_tiles.py \
  --site-id S001 --context-m 10 \
  --tile pc_BA31_1000_3848.laz PATH/3848_reconciled.laz PATH/3848_coordinate_audit.json \
  --tile pc_BA31_1000_3849.laz PATH/3849_reconciled.laz PATH/3849_coordinate_audit.json \
  --tile pc_BA31_1000_3948.laz PATH/3948_reconciled.laz PATH/3948_coordinate_audit.json \
  --tile pc_BA31_1000_3949.laz PATH/3949_reconciled.laz PATH/3949_coordinate_audit.json \
  --output outputs/training_site_merges/RUN_ID_S001
```

Supply every tile listed for the site in the frozen coverage plan exactly once and use a fresh output directory. The merger rejects non-training sites and training tiles that intersect held-out sites. Tile-local instance IDs remain separate unless an exact duplicate XYZ record with the same semantic label links them. Proximity alone never joins instances across a seam. The output is deduplicated for crown conversion and must not be used to calculate return density. Preserve `merge_receipt.json`, `SHA256_manifest.json`, the coverage GPKG and the merged LAZ with the model run receipts.

`scripts/20_sat_crown_candidates.py` and
`scripts/35_compare_crown_conversion_candidates.py` accept the completed
`merge_receipt.json` as the point-input verification record. After generating
the same fixed configuration grid for every training site, aggregate it with:

```bash
python scripts/49_summarise_training_conversion_grid.py \
  --coverage-plan PATH/training_multitile_coverage.json \
  --grid PATH/S001/summary.json \
  --grid PATH/S006/summary.json \
  --grid PATH/S008/summary.json \
  --grid PATH/S010/summary.json \
  --grid PATH/S015/summary.json \
  --grid PATH/S017/summary.json \
  --grid PATH/S019/summary.json \
  --grid PATH/S024/summary.json \
  --output PATH/fresh_aggregate_directory
```

The command requires all eight training sites exactly once, verifies hashes
and source-tile coverage, and refuses non-training results. Its selection is
provisional until the reference is frozen. If the frozen training reference
changes, repeat this training-only aggregation before locking the conversion.
Do not revise the matching rule after inspecting these results.

Create a compact audit archive after the queue, merges and conversion grids
are complete:

```bash
python scripts/50_archive_training_development.py \
  --work-root PATH/outputs/nectar/RUN_ID \
  --reference-dir PATH/current_reference_candidate \
  --output PATH/fresh_training_development_evidence.tar.gz
```

Record and independently verify the resulting archive SHA-256. The compact
archive deliberately retains hashes instead of duplicating full-tile LAZ
predictions; keep those large files on the VM until the study is complete.

## 5. Scoring and reporting

`python scripts/12_evaluate.py --help` describes scored inputs. Verify reference and prediction CRS, site coverage, version hashes, matching policy, and run provenance before scoring. Synthetic/workflow tests must pass. Perform model and parameter selection only on training data. After reference and configurations are locked, run the validation/test protocol once. Report all strata, paired whole-site bootstrap confidence intervals, ignored predictions, geometry QA, and compute cost. Keep Chapter 6 result cells blank until valid held-out numbers exist.

The current code and methodological limits are summarized in `docs/EXPERIMENT_STATUS.md` and `docs/DECISIONS.md`.
