#!/usr/bin/env bash
# Foreground in a persistent tmux session; all stages have separate receipts.
set -euo pipefail
cd /home/ubuntu/akl-trees-benchmark
export PATH=/home/ubuntu/miniforge3/envs/akl-benchmark-prep/bin:$PATH
export OMP_NUM_THREADS=4
export TERM=xterm-256color
BENCH_ALTO_PREP=outputs/preparation/alto_3d_weak_revision_20261011
BENCH_ALTO_TRAIN=outputs/nectar/alto_3d_weak_pilot10_20261011
BENCH_ALTO_BASE=outputs/nectar/alto_3d_weak_full_20261011
BENCH_ALTO_IMAGE=sha256:fc814e8b230060ea5475b29e2ec7ad01fa1b0e1308116585a7f2ef9f24c82d91
BENCH_ALTO_STAGE=preflight
mkdir -p outputs/nectar/alto_3d_weak_control_20261011
BENCH_ALTO_CONTROL=outputs/nectar/alto_3d_weak_control_20261011
exec > >(tee -a "$BENCH_ALTO_CONTROL/pipeline.log") 2>&1
trap 'printf "FAILED stage=%s exit=%s UTC=%s\n" "$BENCH_ALTO_STAGE" "$?" "$(date -u +%FT%TZ)" > "$BENCH_ALTO_CONTROL/FAILED.txt"' ERR
set -x
python - <<'PY'
import hashlib,json,shutil
from pathlib import Path
p=Path('outputs/preparation/alto_3d_weak_revision_20261011')
for rel,digest in json.loads((p/'labels/SHA256_manifest.json').read_text()).items():
 assert hashlib.sha256((p/'labels'/rel).read_bytes()).hexdigest()==digest,rel
qa=json.loads((p/'assistant_review.json').read_text())
assert qa['heldout_used_for_label_revision'] is False and qa['point_labels_manually_checked'] is False
assert hashlib.sha256((p/'labels/SHA256_manifest.json').read_bytes()).hexdigest()==qa['training_manifest_sha256']
assert shutil.disk_usage('.').free>6*1024**3
assert not any(Path(x).exists() for x in ['outputs/nectar/alto_3d_weak_smoke_20261011','outputs/nectar/alto_3d_weak_pilot10_20261011','outputs/nectar/alto_3d_weak_full_20261011'])
print('Verified fresh outputs, assistant-review receipt, revised PLY manifest and disk headroom.')
PY
BENCH_ALTO_STAGE=smoke
bash scripts/hpc/run_vm.sh alto_3d_weak_smoke -- \
 docker run --rm --gpus all --user "$(id -u):$(id -g)" \
 --env HOME=/tmp --env PYTHONPATH=/sat:/kit/scripts --env OMP_NUM_THREADS=4 \
 --tmpfs /home/datascience:rw,mode=1777 \
 --mount type=bind,src=/home/ubuntu/SegmentAnyTree,dst=/sat,readonly \
 --mount type=bind,src=/home/ubuntu/akl-trees-benchmark,dst=/kit \
 --workdir /sat --entrypoint python3 "$BENCH_ALTO_IMAGE" /kit/scripts/hpc/86_weak_sat_cuda.py \
 --screened "/kit/$BENCH_ALTO_PREP/labels" --site-definitions /kit/data/sites/reference_sites.json \
 --checkpoint /sat/model_file/PointGroup-PAPER.pt --source-config /kit/config/segmentanytree_source.json \
 --epochs 0 --smoke-site S010 --output /kit/outputs/nectar/alto_3d_weak_smoke_20261011
BENCH_ALTO_STAGE=training
bash scripts/hpc/run_vm.sh alto_3d_weak_pilot10 -- \
 docker run --rm --gpus all --user "$(id -u):$(id -g)" \
 --env HOME=/tmp --env PYTHONPATH=/sat:/kit/scripts --env OMP_NUM_THREADS=4 \
 --tmpfs /home/datascience:rw,mode=1777 \
 --mount type=bind,src=/home/ubuntu/SegmentAnyTree,dst=/sat,readonly \
 --mount type=bind,src=/home/ubuntu/akl-trees-benchmark,dst=/kit \
 --workdir /sat --entrypoint python3 "$BENCH_ALTO_IMAGE" /kit/scripts/hpc/86_weak_sat_cuda.py \
 --screened "/kit/$BENCH_ALTO_PREP/labels" --site-definitions /kit/data/sites/reference_sites.json \
 --checkpoint /sat/model_file/PointGroup-PAPER.pt --source-config /kit/config/segmentanytree_source.json \
 --epochs 10 --learning-rate 0.00001 --output "/kit/$BENCH_ALTO_TRAIN"
BENCH_ALTO_STAGE=inference
python scripts/104_run_compact_weak_checkpoint_queue.py \
 --producer-queue outputs/nectar/20261006T065811Z_assumed_reference_pretrained_heldout/queue_receipt.json \
 --checkpoint "$BENCH_ALTO_TRAIN/PointGroup-WEAK-final.pt" \
 --training-receipt "$BENCH_ALTO_TRAIN/receipt.json" \
 --author-lock outputs/preparation/confirmed_screened_weak_20261010/AUTHOR_REFERENCE_LOCK.json \
 --source /home/ubuntu/SegmentAnyTree --image "$BENCH_ALTO_IMAGE" --output "$BENCH_ALTO_BASE"
BENCH_ALTO_STAGE=assembly_scoring
bash scripts/hpc/run_vm.sh alto_3d_weak_assembly_score -- \
 python scripts/90_complete_weak_checkpoint_results.py --base "$BENCH_ALTO_BASE" \
 --reference outputs/preparation/confirmed_reference_20261010/prefreeze_evaluation_reference.gpkg \
 --reference-lock outputs/preparation/confirmed_reference_20261010/AUTHOR_DERIVED_REFERENCE_LOCK.json \
 --pretrained-assembly outputs/nectar/20261006T065811Z_assumed_reference_pretrained_heldout/assembled_sites_v1 \
 --classical-base outputs/nectar/20261006T081000Z_fixed_classical_heldout \
 --current-classical-base outputs/nectar/confirmed_classical_h3_20261010 \
 --output "$BENCH_ALTO_BASE/completed_sites"
# Script 90 generates the comparison report as part of assembly. Running
# script 93 again would reject the already complete report directory.
BENCH_ALTO_STAGE=archive
bash scripts/hpc/run_vm.sh alto_3d_weak_archive -- \
 python scripts/95_archive_final_weak_results.py --base "$BENCH_ALTO_BASE" \
 --output outputs/nectar_receipts/alto_3d_weak_full_20261011.tar.gz
printf 'COMPLETED UTC=%s\n' "$(date -u +%FT%TZ)" > "$BENCH_ALTO_CONTROL/COMPLETED.txt"
