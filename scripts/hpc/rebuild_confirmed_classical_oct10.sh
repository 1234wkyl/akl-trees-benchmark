#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/akl-trees-benchmark
export PATH=/home/ubuntu/miniforge3/envs/akl-benchmark-prep/bin:$PATH
export OMP_NUM_THREADS=2
AKL_CURRENT_CLASSICAL=outputs/nectar/confirmed_classical_h3_20261010
AKL_CURRENT_CHOICE=outputs/preparation/confirmed_classical_choice_20261010
bash scripts/hpc/run_vm.sh confirmed_classical_config -- \
  python scripts/91_prepare_confirmed_classical.py \
  --training-rescore outputs/preparation/confirmed_classical_training_rescore_20261010 \
  --reference-lock outputs/preparation/confirmed_reference_20261010/AUTHOR_DERIVED_REFERENCE_LOCK.json \
  --original-inputs outputs/nectar/20261006T073500Z_classical_heldout_inputs_busy_backoff \
  --output "$AKL_CURRENT_CHOICE"
bash scripts/hpc/run_vm.sh confirmed_classical_h3_predictions -- \
  python scripts/60_run_fixed_classical_predictions.py \
  --source /home/ubuntu/Xi_Zhang_Benchmark/source_reference \
  --chm /home/ubuntu/Xi_Zhang_Benchmark/chm/reference_sites.vrt \
  --database /home/ubuntu/baselines/xi-capstone-20260730/akl_trees.sqlite \
  --intake outputs/nectar/20260920T142229Z_gpu_v100_4319/baseline_intake/intake_report.json \
  --inputs "$AKL_CURRENT_CHOICE/reused_inputs" \
  --classical-selection "$AKL_CURRENT_CHOICE/selection.json" \
  --author-model-lock "$AKL_CURRENT_CHOICE/model_lock.json" \
  --output "$AKL_CURRENT_CLASSICAL"
for AKL_CURRENT_SPLIT in val test; do
  bash scripts/hpc/run_vm.sh "confirmed_classical_standard_$AKL_CURRENT_SPLIT" -- \
    python scripts/16_standardize_predictions.py \
    --input "$AKL_CURRENT_CLASSICAL/fixed_classical_crowns.gpkg" --layer crowns \
    --id-column tree_id --split "$AKL_CURRENT_SPLIT" \
    --output "$AKL_CURRENT_CLASSICAL/${AKL_CURRENT_SPLIT}_standardized"
done
printf 'Author-confirmed training-only classical continuation complete.\n'
