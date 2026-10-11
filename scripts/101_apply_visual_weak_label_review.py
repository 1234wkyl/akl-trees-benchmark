#!/usr/bin/env python3
"""Apply explicit, visually inspected training-only abstentions to a new version.

This is partial weak-label review, not independently checked point truth.
Local-Z cuts isolate inspected disconnected layers; they are not HAG rules.
Original labels, reference polygons, checkpoints and scored results are immutable.
"""
import argparse
import csv
import html
import importlib.util
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from benchmark.common import ROOT, sha256, site_definitions, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screened', type=Path, required=True)
    ap.add_argument('--candidates', type=Path, required=True)
    ap.add_argument('--decisions', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    for key in ('screened', 'candidates', 'decisions', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    if args.output.exists():
        ap.error('fresh output directory required')
    decisions = json.loads(args.decisions.read_text())
    defs = site_definitions()
    for d in decisions['changes']:
        if defs[d['site_id']]['split'] != 'train':
            raise ValueError('training sites only')
    reader = ROOT.parent / 'SegmentAnyTree/torch_points3d/modules/KPConv/plyutils.py'
    spec = importlib.util.spec_from_file_location('ply101', reader)
    ply = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ply)
    rows = list(csv.DictReader((args.screened / 'instance_screening.csv').open()))
    args.output.mkdir(parents=True)
    labels = args.output / 'labels'
    labels.mkdir()
    shutil.copy2(args.decisions, args.output / 'review_decisions.json')
    shutil.copy2(Path(__file__), args.output / 'executed_source.py')
    shutil.copy2(args.screened / 'instance_screening.csv', labels / 'instance_screening.csv')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_partial_visual_weak_label_review'
    point_edits, site_records, change_records, panels = [], [], [], []
    inputs = {}
    for rp in sorted(args.screened.glob('*/screening_record.json')):
        rec = json.loads(rp.read_text())
        sid = rec['site_id']
        if defs[sid]['split'] != 'train':
            raise ValueError('training-only source required')
        src = rp.parent / (sid + '_screened_weak.ply')
        src_hash = sha256(src)
        if src_hash != rec['screened_ply_sha256']:
            raise ValueError('source PLY hash mismatch')
        inputs[str(src.relative_to(ROOT))] = src_hash
        raw = ply.read_ply(str(src))
        xyz = np.column_stack([raw[k] for k in ('X', 'Y', 'Z')])
        sem = raw['semantic_seg'].astype('i4')
        ids = raw['treeID'].astype('i4')
        new_sem, new_ids = sem.copy(), ids.copy()
        viewer_path = args.candidates / sid / 'viewer_points.json'
        original = json.loads(viewer_path.read_text())
        inputs[str(viewer_path.relative_to(ROOT))] = sha256(viewer_path)
        old = np.asarray(original['points'])
        if len(old) != len(xyz) or np.abs(old[:, :3] - xyz).max() >= 1e-4:
            raise ValueError('candidate point-row alignment mismatch')
        chosen = [r for r in rows if r['site_id'] == sid and r['retained'] == 'True']
        mapping = {r['treeID']: r['label_id'] for r in chosen}
        changed = np.zeros(len(xyz), dtype=bool)
        for d in [c for c in decisions['changes'] if c['site_id'] == sid]:
            if src_hash != d['source_ply_sha256'] or mapping[str(d['treeID'])] != d['label_id']:
                raise ValueError('decision does not identify this source instance')
            instance = (sem == 2) & (ids == d['treeID'])
            low = instance & (xyz[:, 2] < d['local_z_cut_m'])
            upper = instance & ~low
            if low.sum() != d['expected_changed_points'] or upper.sum() < 20 or (low & changed).any():
                raise ValueError('decision point count or ownership mismatch')
            gap = float(xyz[upper, 2].min() - xyz[low, 2].max())
            if gap < d['minimum_observed_vertical_gap_m']:
                raise ValueError('inspected layer separation no longer holds')
            changed |= low
            new_sem[low], new_ids[low] = 0, -1
            for index in np.flatnonzero(low):
                point_edits.append({'site_id': sid, 'point_index_zero_based': int(index),
                    'X': float(xyz[index, 0]), 'Y': float(xyz[index, 1]), 'Z': float(xyz[index, 2]),
                    'old_semantic_seg': 2, 'old_treeID': d['treeID'],
                    'new_semantic_seg': 0, 'new_treeID': -1, 'reason': d['reason']})
            change_records.append({**d, 'observed_vertical_gap_m': gap,
                'retained_points_in_instance': int(upper.sum()),
                'changed_point_indices_zero_based': np.flatnonzero(low).tolist()})
            pts = xyz[instance]
            lo, hi = pts.min(0), pts.max(0)
            ctx = (xyz[:, 0] >= lo[0]-3) & (xyz[:, 0] <= hi[0]+3) & (xyz[:, 1] >= lo[1]-3) & (xyz[:, 1] <= hi[1]+3)
            origin = np.asarray(rec['coordinate_origin_xyz_m'])
            bb = defs[sid]['bbox_epsg2193']
            extent = (bb[0]-origin[0], bb[2]-origin[0], bb[1]-origin[1], bb[3]-origin[1])
            chip = ROOT / 'data/sites/chips' / sid
            with rasterio.open(chip / 'chm.tif') as raster:
                chm = raster.read(1, masked=True)
            inputs[str((chip/'rgb.png').relative_to(ROOT))] = sha256(chip/'rgb.png')
            inputs[str((chip/'chm.tif').relative_to(ROOT))] = sha256(chip/'chm.tif')
            fig, axs = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
            for ax, backdrop, title in [(axs[0,0], plt.imread(chip/'rgb.png'), 'RGB: inspected footprint'), (axs[1,0], chm, 'CHM: canopy height (m)')]:
                im = ax.imshow(backdrop, extent=extent, **({'cmap':'viridis'} if backdrop.ndim == 2 else {}))
                ax.scatter(xyz[upper,0], xyz[upper,1], s=6, c='#139cbb', alpha=.7)
                ax.scatter(xyz[low,0], xyz[low,1], s=12, c='#e45b35')
                ax.set_xlim(lo[0]-3, hi[0]+3); ax.set_ylim(lo[1]-3, hi[1]+3)
                ax.set_aspect('equal'); ax.set_title(title)
                ax.set_xlabel('Local easting (m)'); ax.set_ylabel('Local northing (m)')
                if backdrop.ndim == 2: fig.colorbar(im, ax=ax, shrink=.8)
            for col, axis in [(1,0), (2,1)]:
                for row in [0,1]:
                    ax = axs[row,col]
                    ax.scatter(xyz[ctx,axis], xyz[ctx,2], s=2, c='#bac0c5', alpha=.3)
                    ax.scatter(xyz[upper,axis], xyz[upper,2], s=8, c='#139cbb')
                    ax.scatter(xyz[low,axis], xyz[low,2], s=13, c='#e45b35' if row == 0 else '#747d85')
                    ax.axhline(d['local_z_cut_m'], c='#666', ls='--', lw=1)
                    ax.set_title(('Before: orange = disputed assigned points' if row == 0 else 'After: grey = unknown ownership'))
                    ax.set_xlabel('Local '+('easting' if axis == 0 else 'northing')+' (m)')
                    ax.set_ylabel('Local Z (m; not HAG)'); ax.grid(alpha=.2)
            name = f'{sid}_ID{d["treeID"]}_before_after.png'
            fig.suptitle(f'{sid} / treeID {d["treeID"]}: {int(low.sum())} assignments withdrawn; {int(upper.sum())} retained\nVertical gap {gap:.2f} m; lower-layer ownership remains unknown', fontsize=14)
            fig.savefig(args.output/name, dpi=180); plt.close(fig)
            panels.append(f'<section><h2>{sid} / ID {d["treeID"]}</h2><p>{html.escape(d["reason"])} {int(low.sum())} returns changed from tree candidate to unknown.</p><img src="{name}"></section>')
        dst = labels / sid
        dst.mkdir()
        target = dst / src.name
        if changed.any():
            if not ply.write_ply(str(target), [xyz,new_sem,new_ids], ['X','Y','Z','semantic_seg','treeID']):
                raise ValueError('PLY write failed')
        else: shutil.copy2(src, target)
        decoded = ply.read_ply(str(target))
        if not np.array_equal(xyz, np.column_stack([decoded[k] for k in ('X','Y','Z')])):
            raise ValueError('XYZ or row order changed')
        if not np.array_equal(new_sem, decoded['semantic_seg']) or not np.array_equal(new_ids, decoded['treeID']):
            raise ValueError('label round trip failed')
        if not np.array_equal(sem[~changed],new_sem[~changed]) or not np.array_equal(ids[~changed],new_ids[~changed]):
            raise ValueError('unselected labels changed')
        if set(np.unique(new_ids[new_sem == 2])) != set(np.unique(ids[sem == 2])):
            raise ValueError('instance removed or added')
        new_rec = {**rec, 'screened_ply_sha256':sha256(target), 'review_source_ply_sha256':src_hash,
            'run_id':run_id, 'retained_tree_points':int((new_sem==2).sum()),
            'unknown_points':int((new_sem==0).sum()), 'visual_review_demotions':int(changed.sum()),
            'labels_demoted_to_unknown':rec['labels_demoted_to_unknown']+int(changed.sum()),
            'label_type':'partial visual abstention revision of weak candidates',
            'point_labels_manually_checked':False, 'partial_visual_review':True,
            'review_method':'assistant inspection of training RGB, CHM and two side views; no field verification',
            'used_in_completed_experiment':False, 'fine_tuning_permitted':False}
        write_json(dst/'screening_record.json',new_rec)
        site_records.append(new_rec)
        viewer = args.output/'viewer'/sid
        viewer.mkdir(parents=True)
        shutil.copy2(target,viewer/target.name)
        counts = {i:int(((new_sem==2)&(new_ids==int(i))).sum()) for i in mapping}
        write_json(viewer/'viewer_points.json',{'site_id':sid,'origin_xyz_m':original['origin_xyz_m'],
            'mapping':mapping,'points':np.column_stack([xyz,new_sem,new_ids,old[:,5]]).tolist()})
        write_json(viewer/'candidate_record.json',{'site_id':sid,'screened_weak':True,
            'review_note':'Partially visually revised weak candidates. This version has not been trained. Detached ambiguous lower-layer assignments are now unknown; biological point ownership remains uncertified.',
            'display_status':'Revised weak-label version; no new training run',
            'points':len(xyz),'candidate_instances_with_points':len(counts),'candidate_instance_point_counts':counts,
            'unknown_points':int((new_sem==0).sum()),'ply_file':target.name,
            'point_labels_manually_checked':False,'used_in_completed_experiment':False})
    fields = ['site_id','point_index_zero_based','X','Y','Z','old_semantic_seg','old_treeID','new_semantic_seg','new_treeID','reason']
    with (args.output/'point_edits.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(point_edits)
    receipt = {'run_id':run_id,'status':'PARTIAL VISUAL WEAK-LABEL REVISION; NOT TRAINED',
        'points':sum(s['points'] for s in site_records),'retained_instance_ids':sum(s['retained_original_ids']+s['retained_alto_ids'] for s in site_records),
        'assignments_demoted_to_unknown':len(point_edits),'retained_tree_points':sum(s['retained_tree_points'] for s in site_records),
        'coordinate_and_row_order_preservation_verified':True,'unselected_labels_preservation_verified':True,
        'background_labels_changed':False,'reference_polygons_changed':False,'model_weights_changed':False,
        'completed_experiment_changed':False,'supervised_point_truth_certified':False,
        'review_decisions_sha256':sha256(args.decisions),'script_sha256':sha256(Path(__file__)),
        'upstream_ply_reader_sha256':sha256(reader),'inputs':inputs,'changes':change_records,'sites':site_records}
    write_json(args.output/'receipt.json',receipt)
    write_json(labels/'summary.json',receipt)
    write_json(labels/'SHA256_manifest.json',{str(p.relative_to(labels)):sha256(p) for p in sorted(labels.rglob('*')) if p.is_file()})
    table=''.join(f'<tr><td>{s["site_id"]}</td><td>{s["visual_review_demotions"]}</td><td>{s["retained_tree_points"]}</td><td>{s["unknown_points"]}</td><td><a href="../../labeller/point_review.html?data=../outputs/weak_label_review_20261011/revision/viewer/&amp;site={s["site_id"]}">Rotate revised cloud</a></td></tr>' for s in site_records)
    text = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Partial visual weak-label revision</title><style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px;color:#193445}img{width:100%}p{line-height:1.6}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}section{margin:40px 0}</style><h1>Partial visual weak-label revision</h1>'
    text += f'<p>{len(point_edits)} ambiguous point assignments withdrawn in three training instances. All {receipt["points"]:,} acquisition returns and all {receipt["retained_instance_ids"]} candidate IDs are retained. Unknown points remain input geometry without known semantic or instance targets.</p>'
    text += '<p>This is a separate, partially reviewed weak-label version. It has not been used for training, and no accuracy improvement is claimed. Archived training inputs, checkpoints, scores and the reference polygons remain unchanged. The local-Z cuts apply only to the three inspected instances; they are not a site-wide height rule.</p><p><a href="review_decisions.json">Decisions</a> · <a href="point_edits.csv">Every edited row</a> · <a href="receipt.json">Provenance and checks</a></p><table><tr><th>Site</th><th>Withdrawn assignments</th><th>Remaining tree points</th><th>Unknown points</th><th>Viewer</th></tr>'+table+'</table>'+''.join(panels)+'</html>'
    text=text.replace('href="../../labeller/','href="../../../labeller/')
    (args.output/'index.html').write_text(text)
    print(json.dumps({k:receipt[k] for k in ['run_id','points','retained_instance_ids','assignments_demoted_to_unknown','retained_tree_points']},indent=2))


if __name__ == '__main__':
    main()
