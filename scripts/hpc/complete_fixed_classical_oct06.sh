#!/usr/bin/env bash
# Foreground dependency continuation; never restarts acquisition or inference.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH=/home/ubuntu/miniforge3/envs/akl-benchmark-prep/bin:$PATH
AKL_INPUT=outputs/nectar/20261006T073500Z_classical_heldout_inputs_busy_backoff
AKL_CLASSIC=outputs/nectar/20261006T081000Z_fixed_classical_heldout
AKL_HELD=outputs/nectar/20261006T065811Z_assumed_reference_pretrained_heldout
AKL_PARENT=610673
printf 'Waiting for the already-running input producer %s; no repeat download.\n' "$AKL_PARENT"
while kill -0 "$AKL_PARENT" 2>/dev/null; do sleep 5; done
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/benchmarks/20261006T073440Z_prepare_fixed_classical_heldout_inputs_busy_backoff_610673/timing.json')
if not p.exists() or json.loads(p.read_text())['exit_code'] != 0:
    raise SystemExit('Input acquisition failed or is incomplete; preserve receipts and stop.')
if not Path('outputs/nectar/20261006T073500Z_classical_heldout_inputs_busy_backoff/input_receipt.json').exists():
    raise SystemExit('Complete input receipt missing; stop.')
PY
bash scripts/hpc/run_vm.sh fixed_classical_heldout_predictions -- \
  python scripts/60_run_fixed_classical_predictions.py \
  --source /home/ubuntu/Xi_Zhang_Benchmark/source_reference \
  --chm /home/ubuntu/Xi_Zhang_Benchmark/chm/reference_sites.vrt \
  --database /home/ubuntu/baselines/xi-capstone-20260730/akl_trees.sqlite \
  --intake outputs/nectar/20260920T142229Z_gpu_v100_4319/baseline_intake/intake_report.json \
  --inputs "$AKL_INPUT" \
  --classical-selection outputs/nectar/20261006T064424Z_classical_training_trials_oct06/selection_v2/selection.json \
  --inference-scenario "$AKL_HELD/inference_scenario.json" \
  --output "$AKL_CLASSIC"
for AKL_SPLIT in val test; do
  bash scripts/hpc/run_vm.sh "standardize_fixed_classical_$AKL_SPLIT" -- \
    python scripts/16_standardize_predictions.py \
    --input "$AKL_CLASSIC/fixed_classical_crowns.gpkg" --layer crowns \
    --id-column tree_id --split "$AKL_SPLIT" --output "$AKL_CLASSIC/${AKL_SPLIT}_standardized"
done
printf 'Fixed classical predictions are complete. No held-out labels used by the producer.\n'
bash scripts/hpc/run_vm.sh seal_classical_oct06_receipts -- \
  python scripts/62_archive_assumed_run_receipts.py --phase classical \
  --output "$AKL_CLASSIC/completed_classical_receipts.tar.gz"
