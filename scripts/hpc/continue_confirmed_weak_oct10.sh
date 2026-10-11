#!/usr/bin/env bash
# Foreground in a persistent attached tmux session; no daemon/nohup.
set -euo pipefail
cd /home/ubuntu/akl-trees-benchmark
export PATH=/home/ubuntu/miniforge3/envs/akl-benchmark-prep/bin:$PATH
export OMP_NUM_THREADS=4
BENCH_WEAK_BASE=outputs/nectar/confirmed_weak_full_20261010
python scripts/89_run_weak_checkpoint_queue.py \
  --producer-queue outputs/nectar/20261006T065811Z_assumed_reference_pretrained_heldout/queue_receipt.json \
  --checkpoint outputs/nectar/confirmed_weak_pilot10_20261010/PointGroup-WEAK-final.pt \
  --training-receipt outputs/nectar/confirmed_weak_pilot10_20261010/receipt.json \
  --author-lock outputs/preparation/confirmed_screened_weak_20261010/AUTHOR_REFERENCE_LOCK.json \
  --source /home/ubuntu/SegmentAnyTree \
  --image sha256:fc814e8b230060ea5475b29e2ec7ad01fa1b0e1308116585a7f2ef9f24c82d91 \
  --output "$BENCH_WEAK_BASE"
python scripts/90_complete_weak_checkpoint_results.py \
  --base "$BENCH_WEAK_BASE" \
  --reference outputs/preparation/confirmed_reference_20261010/prefreeze_evaluation_reference.gpkg \
  --reference-lock outputs/preparation/confirmed_reference_20261010/AUTHOR_DERIVED_REFERENCE_LOCK.json \
  --pretrained-assembly outputs/nectar/20261006T065811Z_assumed_reference_pretrained_heldout/assembled_sites_v1 \
  --classical-base outputs/nectar/20261006T081000Z_fixed_classical_heldout \
  --output "$BENCH_WEAK_BASE/completed_sites"
printf 'Completed fixed weak-model inference, assembly and scoring.\n'
