#!/usr/bin/env bash
# Foreground tmux continuation of the predeclared four-condition transfer study.
set -euo pipefail
cd /home/ubuntu/akl-trees-benchmark
export PATH=/home/ubuntu/miniforge3/envs/akl-benchmark-prep/bin:$PATH
AKL_DENSITY=outputs/nectar/20261007T024000Z_fixed_density_transfer
AKL_CLASSICAL=outputs/nectar/20261006T081000Z_fixed_classical_heldout
bash scripts/hpc/run_vm.sh fixed_density_sweep_oct07 -- \
  python scripts/67_run_fixed_density_sweep.py \
  --base "$AKL_DENSITY" --classical-base "$AKL_CLASSICAL" \
  --source /home/ubuntu/SegmentAnyTree
if [[ ! -f "$AKL_DENSITY/density_report/report_receipt.json" ]]; then
  bash scripts/hpc/run_vm.sh report_fixed_density_sweep_oct07 -- \
    python scripts/68_report_fixed_density_sweep.py \
    --base "$AKL_DENSITY" --output "$AKL_DENSITY/density_report"
fi
printf 'Fixed density sweep and conditional report completed. Preserve outputs before stopping the VM.\n'
