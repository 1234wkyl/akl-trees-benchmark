#!/usr/bin/env python3
"""CPU real-data target/loss/backward probe; does NOT instantiate or train SAT.

Uses the pinned upstream reader and target builder, then the separate weak loss.
A real sparse-convolution model smoke test on the VM is still required.
"""
import argparse
import ast
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from benchmark.common import ROOT, sha256, site_definitions, write_json
from benchmark.weak_pointgroup_loss import compute_weak_pointgroup_loss


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screened', type=Path, required=True)
    ap.add_argument('--site-id', default='S017')
    ap.add_argument('--sat-source', type=Path, default=ROOT.parent / 'SegmentAnyTree')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() or site_definitions()[args.site_id]['split'] != 'train':
        ap.error('fresh output and training site required')
    start = time.monotonic()
    config = json.loads((ROOT / 'config/segmentanytree_source.json').read_text())
    commit = subprocess.check_output(['git', '-C', str(args.sat_source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != config['commit']:
        raise ValueError('unexpected upstream commit')
    reader_path = args.sat_source / 'torch_points3d/datasets/segmentation/treeins.py'
    ply_path = args.sat_source / 'torch_points3d/modules/KPConv/plyutils.py'
    targets_path = args.sat_source / 'torch_points3d/datasets/panoptic/utils.py'
    for path in [reader_path, ply_path, targets_path]:
        rel = str(path.relative_to(args.sat_source))
        pristine = subprocess.check_output(['git', '-C', str(args.sat_source), 'show', 'HEAD:' + rel])
        if path.read_bytes() != pristine:
            raise ValueError('upstream reader/target source modified')
    ply = load(ply_path, 'ply73')
    # Execute the exact reader function without importing its CUDA-heavy dataset package.
    tree = ast.parse(reader_path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'read_treeins_format')
    namespace = {'np': np, 'torch': torch, 'read_ply': ply.read_ply}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(reader_path), 'exec'), namespace)
    target_builder = load(targets_path, 'targets73').set_extra_labels
    record = json.loads((args.screened / args.site_id / 'screening_record.json').read_text())
    source = args.screened / args.site_id / (args.site_id + '_screened_weak.ply')
    if record['screened_ply_sha256'] != sha256(source):
        raise ValueError('screened PLY changed')
    pos, y, ids = namespace['read_treeins_format'](str(source))
    data = SimpleNamespace(pos=pos, y=y, instance_labels=ids)
    data = target_builder(data, instance_classes=[1], num_max_objects=256)
    data.batch = torch.zeros(len(pos), dtype=torch.long)
    if not (y == 1).any() or not (y == -1).any() or not (y == 0).any():
        raise ValueError('probe needs tree, background and unknown real points')
    unknown = torch.where(y == -1)[0]
    clusters = [torch.cat([torch.where(data.instance_labels == ident)[0], unknown[:5]])
                for ident in torch.unique(data.instance_labels) if ident > 0]
    clusters += [torch.where(y == 0)[0][:10], unknown[:10]]
    # Leaf tensors exercise the loss contract; they are not network predictions.
    semantic_leaf = torch.zeros(len(pos), 2, requires_grad=True)
    offset_leaf = (data.vote_label.clone() * .5).requires_grad_()
    score_leaf = torch.full((len(clusters),), .5, requires_grad=True)
    mask_leaf = torch.zeros(sum(len(c) for c in clusters), 1, requires_grad=True)
    model = SimpleNamespace(input=data, labels=SimpleNamespace(y=y),
        output=SimpleNamespace(semantic_logits=F.log_softmax(semantic_leaf, dim=1),
            offset_logits=offset_leaf, cluster_scores=score_leaf, mask_scores=mask_leaf, clusters=clusters),
        opt=SimpleNamespace(loss_weights={k: 1. for k in ['semantic', 'offset_norm_loss',
                            'offset_dir_loss', 'score_loss', 'mask_loss']},
                            min_iou_threshold=.25, max_iou_threshold=.75),
        _scorer_type='MLP', mask_supervise=True, cal_iou_based_on_mask=False,
        cal_iou_based_on_mask_start_epoch=0)
    compute_weak_pointgroup_loss(model, 1)
    model.loss.backward()
    tensors = [semantic_leaf, offset_leaf, score_leaf, mask_leaf]
    finite = bool(torch.isfinite(model.loss) and all(torch.isfinite(t.grad).all() for t in tensors))
    sem_unknown_zero = bool((semantic_leaf.grad[y == -1] == 0).all())
    offset_unknown_zero = bool((offset_leaf.grad[y == -1] == 0).all())
    flat_unknown = torch.cat([y[c] == -1 for c in clusters])
    mask_unknown_zero = bool((mask_leaf.grad.reshape(-1)[flat_unknown] == 0).all())
    unknown_proposal_zero = bool(score_leaf.grad[-1] == 0)
    if not all([finite, sem_unknown_zero, offset_unknown_zero, mask_unknown_zero, unknown_proposal_zero]):
        raise ValueError('unknown-target gradient or finite-loss contract failed')
    result = {'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_weak_loss_target_probe',
        'argv': sys.argv, 'status': 'CPU REAL-DATA TARGET/LOSS PROBE PASSED; FULL SAT SMOKE PENDING',
        'site_id': args.site_id, 'split': 'train', 'points': len(pos),
        'candidate_instances': int(data.num_instances.item()),
        'unknown_points': int((y == -1).sum()), 'candidate_background_points': int((y == 0).sum()),
        'candidate_tree_points': int((y == 1).sum()),
        'losses': {k: float(getattr(model, k).detach()) for k in ['loss', 'semantic_loss',
            'offset_norm_loss', 'offset_dir_loss', 'score_loss', 'mask_loss']},
        'finite_loss_and_gradients': finite, 'unknown_semantic_target_gradient_zero': sem_unknown_zero,
        'unknown_offset_target_gradient_zero': offset_unknown_zero,
        'unknown_mask_target_gradient_zero': mask_unknown_zero,
        'all_unknown_proposal_score_gradient_zero': unknown_proposal_zero,
        'upstream_commit': commit, 'torch_version': torch.__version__,
        'elapsed_seconds': time.monotonic()-start, 'exit_code': 0,
        'neural_backbone_loaded': False, 'model_weights_updated': False,
        'gpu_used': False, 'point_labels_manually_checked': False, 'fine_tuning_permitted': False,
        'sources': {str(p): sha256(p) for p in [source, reader_path, targets_path, ply_path,
                   ROOT / 'scripts/benchmark/weak_pointgroup_loss.py', Path(__file__)]},
        'limitations': ['Leaf output tensors and constructed proposals, not a SAT forward pass',
                        'Unknown context can still affect learned features and clustering',
                        'Known-only IoU and offset centers use incomplete weak tree membership',
                        'Accuracy benefit and full CUDA model compatibility have not been measured']}
    args.output.mkdir(parents=True)
    write_json(args.output / 'result.json', result)
    for source_path in [Path(__file__), ROOT / 'scripts/benchmark/weak_pointgroup_loss.py']:
        (args.output / source_path.name).write_bytes(source_path.read_bytes())
    write_json(args.output / 'SHA256_manifest.json', {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
