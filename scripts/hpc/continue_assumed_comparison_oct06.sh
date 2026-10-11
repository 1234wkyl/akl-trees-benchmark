#!/usr/bin/env bash
# Automatic continuation of the fixed assumption-only queue; not a formal freeze.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH=/home/ubuntu/miniforge3/envs/akl-benchmark-prep/bin:$PATH
AKL_HELD=outputs/nectar/20261006T065811Z_assumed_reference_pretrained_heldout
AKL_CLASSIC=outputs/nectar/20261006T081000Z_fixed_classical_heldout
bash scripts/hpc/run_vm.sh assemble_assumed_heldout_sites -- \
  python scripts/58_assemble_assumed_heldout_predictions.py \
  --base "$AKL_HELD" --output "$AKL_HELD/assembled_sites_v1"
# Missing/invalid producers stop this continuation, rather than substitute outputs.
bash scripts/hpc/run_vm.sh evaluate_hypothetical_oct06_reference -- \
  python scripts/61_evaluate_assumed_reference.py \
  --inference-scenario "$AKL_HELD/inference_scenario.json" \
  --reference outputs/reference_20261006_assumed_adjudicated/prefreeze_evaluation_reference.gpkg \
  --sat-assembly "$AKL_HELD/assembled_sites_v1" \
  --classical-predictions "$AKL_CLASSIC/test_standardized/predictions.gpkg" \
  --classical-receipt "$AKL_CLASSIC/prediction_receipt.json" \
  --output "$AKL_HELD/hypothetical_comparison_v1"
bash scripts/hpc/run_vm.sh seal_assumed_comparison_receipts -- \
  python scripts/62_archive_assumed_run_receipts.py --phase comparison \
  --output "$AKL_HELD/completed_comparison_receipts.tar.gz"
