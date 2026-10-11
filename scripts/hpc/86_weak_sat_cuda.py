#!/usr/bin/env python3
"""Real pinned SAT CUDA smoke or a fixed, training-only weak-supervision pilot.

Run inside the immutable inference image with upstream source/checkpoint read
only. Unknown points remain inputs and are ignored by the separate weak loss.
This never certifies manual point QA or substitutes for a blind benchmark.
"""
import argparse
import hashlib
import json
import random
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_points3d.core.data_transform import instantiate_transforms
from torch_points3d.datasets.segmentation.treeins import read_treeins_format
from torch_points3d.datasets.panoptic.utils import set_extra_labels
from torch_points3d.models.model_factory import instantiate_model
from benchmark.weak_pointgroup_loss import compute_weak_pointgroup_loss


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screened', type=Path, required=True)
    ap.add_argument('--site-definitions', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--source-config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=0, help='0: one full-site smoke; >0: fixed weak pilot')
    ap.add_argument('--smoke-site', default='S017')
    ap.add_argument('--learning-rate', type=float, default=1e-5)
    args = ap.parse_args()
    if args.output.exists() or args.epochs < 0 or not 0 < args.learning_rate <= 1e-3:
        ap.error('fresh output, nonnegative epochs, and conservative positive learning rate required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the real backbone smoke')
    config = json.loads(args.source_config.read_text())
    if sha(args.checkpoint) != config['checkpoint_sha256']:
        raise ValueError('official checkpoint hash mismatch')
    summary = json.loads((args.screened/'summary.json').read_text())
    sites = {s['site_id']: s for s in json.loads(args.site_definitions.read_text())['sites']}
    ids = [r['site_id'] for r in summary['sites']]
    if any(sites[s]['split'] != 'train' for s in ids) or len(ids) != len(set(ids)):
        raise ValueError('unique training inputs only')
    if not args.epochs and args.smoke_site not in ids:
        raise ValueError('smoke site missing from screened training data')
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    run = OmegaConf.create(checkpoint['run_config'])
    props = SimpleNamespace(**checkpoint['dataset_properties'])
    # The stored properties omit this derived panoptic dataset property.
    props.thing_classes = [i for i in range(props.num_classes) if i not in props.stuff_classes]
    model = instantiate_model(run, props)
    # Upstream inference config uses latest. Keep the same explicit weight set.
    model.load_state_dict(checkpoint['models']['latest'], strict=True)
    model._compute_loss = types.MethodType(compute_weak_pointgroup_loss, model)
    model.to('cuda'); model.train()
    # Single sparse crops can have one activation at a deep level. Preserve
    # pretrained BN statistics while training affine and all other parameters.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    transform = instantiate_transforms(run.data.train_transforms if args.epochs else run.data.test_transform)
    raw = {}
    for sid in ids if args.epochs else [args.smoke_site]:
        path = args.screened/sid/(sid+'_screened_weak.ply')
        rec = json.loads((args.screened/sid/'screening_record.json').read_text())
        if rec['split'] != 'train' or rec['screened_ply_sha256'] != sha(path):
            raise ValueError('screened training file changed')
        raw[sid] = read_treeins_format(str(path))
    args.output.mkdir(parents=True)
    started = datetime.now(timezone.utc); tick = time.monotonic()
    receipt = {'run_id': started.strftime('%Y%m%dT%H%M%SZ')+'_weak_sat_cuda',
               'started_utc': started.isoformat(), 'status': 'RUNNING',
               'mode': 'fixed exploratory weak pilot' if args.epochs else 'disposable one-batch CUDA smoke',
               'epochs': args.epochs, 'learning_rate': args.learning_rate, 'seed': seed,
               'training_sites': list(raw), 'validation_test_used_for_gradients': False,
               'checkpoint_sha256': sha(args.checkpoint), 'checkpoint_weight_key': 'latest',
               'screened_manifest_sha256': sha(args.screened/'SHA256_manifest.json'),
               'reference_json_sha256': summary['reference_sha256'],
               'source_code_sha256': sha(Path(__file__)),
               'loss_adapter_sha256': sha(Path(__import__('benchmark.weak_pointgroup_loss',fromlist=['x']).__file__)),
               'point_labels_manually_checked': False, 'stock_supervised_training_claimed': False,
               'batch_norm_running_statistics_frozen': True, 'radius_m': 8. if args.epochs else None,
               'checkpoint_selection': 'fixed final epoch; no validation/test selection',
               'heldout_results_previously_exposed': True,
               'limitations': ['Weak incomplete XY-guided point memberships are not independent truth',
                              'ALTO proposals share annotation provenance',
                              'Known-only offset centers can be incomplete or crop-truncated',
                              'Unknown inputs can affect features and clustering',
                              'Sparse kernel execution may not be bitwise deterministic'], 'steps': []}
    def save():
        (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    try:
        for epoch in range(max(args.epochs,1)):
            for sid,(pos,y,labels) in raw.items():
                if not (y>=0).any():
                    raise ValueError('site lacks any known weak targets: '+sid)
                keep = torch.ones(len(pos),dtype=torch.bool)
                if args.epochs:
                    tree_indices = torch.where(y==1)[0] if (y==1).any() else torch.where(y>=0)[0]
                    center = pos[tree_indices[torch.randint(len(tree_indices),(1,)).item()]]
                    keep = ((pos[:,:2]-center[:2]).square().sum(1) <= 8.**2)
                data = Data(pos=pos[keep].clone(),y=y[keep].clone(),instance_labels=labels[keep].clone())
                data = transform(data)
                data = set_extra_labels(data,instance_classes=[1],num_max_objects=256)
                data.batch = torch.zeros(len(data.pos),dtype=torch.long)
                if not (data.y>=0).any():
                    raise ValueError('no usable targets after crop/voxel transform')
                data = data.to('cuda'); optimizer.zero_grad()
                model.set_input(data,'cuda')
                active_epoch = max(int(model.opt.prepare_epoch)+1,1)
                model.forward(epoch=active_epoch)
                model._compute_loss(active_epoch)
                if not torch.isfinite(model.loss):
                    raise ValueError('nonfinite real model loss')
                model.loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):
                    raise ValueError('absent or nonfinite real model gradients')
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):
                    raise ValueError('nonfinite gradient norm')
                optimizer.step(); torch.cuda.synchronize()
                row = {'epoch':epoch+1,'site_id':sid,'points':len(data.pos),
                       'known_tree_points':int((data.y==1).sum()),'unknown_points':int((data.y==-1).sum()),
                       'proposal_count':len(model.output.clusters or []),
                       'finite_loss_and_gradients':True,'gradient_norm':float(norm),
                       'losses':{k:float(getattr(model,k).detach()) for k in model.loss_names},
                       'elapsed_seconds':time.monotonic()-tick}
                receipt['steps'].append(row); save(); print(json.dumps(row),flush=True)
                del data
        if args.epochs:
            # Separate file; the official model is mounted read only.
            checkpoint['models'] = {'latest':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
            checkpoint['benchmark_weak_training_receipt'] = {k:v for k,v in receipt.items() if k!='steps'}
            target = args.output/'PointGroup-WEAK-final.pt'
            torch.save(checkpoint,target)
            receipt['output_checkpoint_sha256'] = sha(target)
        receipt.update(status='COMPLETE',exit_code=0,finished_utc=datetime.now(timezone.utc).isoformat(),
                       wall_seconds=time.monotonic()-tick,
                       peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                       torch_version=torch.__version__,cuda_version=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0),
                       official_checkpoint_unchanged=sha(args.checkpoint)==config['checkpoint_sha256'],
                       smoke_weights_discarded=not bool(args.epochs))
    except Exception as exc:
        receipt.update(status='FAILED',exit_code=1,error=str(exc),finished_utc=datetime.now(timezone.utc).isoformat())
        save(); raise
    save(); print('Completed',receipt['run_id'],flush=True)


if __name__ == '__main__':
    main()
