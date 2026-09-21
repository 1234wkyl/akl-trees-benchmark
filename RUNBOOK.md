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

The original JSON is `data/labels/crown_labels_XZ_2026-08-13.json`. The accepted 234-tree working geometry is in `docs/group_refinement_2026-09-18/workflow_accepted234/`. Read the illustrated [freeze review](docs/REFERENCE_SET_FREEZE_REVIEW.md) before changing any label. The original export remains the first comparison input for the pending blind S016/S020 JSON.

When the independent JSON arrives, archive it unchanged, record SHA-256, and follow `scripts/31_receive_independent_labels.py --help`. Inspect every disagreement before assembling an adjudicated reference. Keep the original, proposed, and final layers distinct. No reference is frozen at present.

## 3. Frozen classical baseline

Read the 30 July SQLite database read-only. `scripts/10_inspect_frozen_baseline.py` and `scripts/22_export_received_baseline.py` inspect the snapshot and site-level detection records. The frozen generating commit, parameters, and crown-polygon output or exact regeneration path are still pending verification. Do not label the supplied historical script as the verified frozen pipeline. Once provenance is established, reproduce the frozen output first, then tune on training sites only. Preserve the frozen output and every trial separately.

## 4. SegmentAnyTree on the standalone GPU VM

Use `scripts/hpc/build_sat_image.py` and `scripts/hpc/run_sat_container.py` with the pinned upstream checkout and checkpoint recorded in `config/segmentanytree_source.json`. Record `nvidia-smi`, container image digest, checkpoint hash, code commit, driver, CUDA, and PyTorch versions. Prepare and validate one training tile, then use `scripts/13_prepare_density.py` and `scripts/hpc/run_density_training.py` for nested return retention. Do not infer crown polygons directly from point-instance scores without a fixed conversion. The current conversion grid is a training-only diagnostic and must be resolved before held-out inference.

Full-site inference must cover every relevant source tile, deduplicate overlap, restore coordinates/CRS, clip to the site consistently, and report incomplete coverage. Fine-tuning can start only after checked semantic and instance point labels exist.

## 5. Scoring and reporting

`python scripts/12_evaluate.py --help` describes scored inputs. Verify reference and prediction CRS, site coverage, version hashes, matching policy, and run provenance before scoring. Synthetic/workflow tests must pass. Perform model and parameter selection only on training data. After reference and configurations are locked, run the validation/test protocol once. Report all strata, paired whole-site bootstrap confidence intervals, ignored predictions, geometry QA, and compute cost. Keep Chapter 6 result cells blank until valid held-out numbers exist.

The current code and methodological limits are summarized in `docs/EXPERIMENT_STATUS.md` and `docs/DECISIONS.md`.
