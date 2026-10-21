#!/usr/bin/env python3
"""Automate training weak-label screening and retain all real points, including unknowns.

This exports candidates, not checked labels or permission for stock training.
No reference polygons, model weights or validation/test inputs are changed.
"""
import argparse
import csv
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from shapely import make_valid, intersects_xy
from shapely.geometry import Polygon, box, shape
from benchmark.common import ROOT, sha256, site_definitions, write_json


def screen_reason(points, geometry_inside_site, proposal=None, minimum_points=20):
    reasons = []
    if points < minimum_points:
        reasons.append('insufficient_point_support')
    if not geometry_inside_site:
        reasons.append('source_crown_crosses_site')
    if proposal is not None:
        if proposal['source_geometry_extends_outside_group_union']:
            reasons.append('source_crown_only_partly_labelled_by_group')
        if len(proposal['parent_group_ids']) != 1:
            reasons.append('multiple_parent_groups')
    return reasons


def retain_labels(sem, ids, keep_ids):
    new_sem, new_ids = sem.copy(), ids.copy()
    rejected = (sem == 2) & ~np.isin(ids, list(keep_ids))
    new_sem[rejected] = 0
    new_ids[rejected] = -1
    return new_sem, new_ids, rejected


def crop_checks(xy, sem, ids, region, instance_geometries):
    selected = intersects_xy(region, xy[:, 0], xy[:, 1])
    present = sorted(map(int, np.unique(ids[selected & (sem == 2)])))
    truncated = [k for k in present if not region.covers(instance_geometries[k])]
    unknown = int(((sem == 0) & selected).sum())
    return selected, present, truncated, unknown


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidates', type=Path, required=True)
    ap.add_argument('--reference-json', type=Path, required=True)
    ap.add_argument('--alto-crowns', type=Path, required=True)
    ap.add_argument('--sat-source', type=Path, default=ROOT.parent / 'SegmentAnyTree')
    ap.add_argument('--minimum-points', type=int, default=20)
    ap.add_argument('--crop-context-m', type=float, default=2.)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() or args.minimum_points < 1 or not np.isfinite(args.crop_context_m) or args.crop_context_m < 0:
        ap.error('fresh output, positive support and nonnegative finite context required')
    start = time.monotonic()
    source_manifest = json.loads((args.candidates / 'SHA256_manifest.json').read_text())
    summary = json.loads((args.candidates / 'summary.json').read_text())
    if sha256(args.reference_json) != summary['reference_sha256']:
        raise ValueError('reference changed after candidate generation')
    sites = site_definitions()
    site_ids = summary['site_ids']
    if len(set(site_ids)) != len(site_ids) or any(sites[s]['split'] != 'train' for s in site_ids):
        raise ValueError('unique training sites only')
    labels = json.loads(args.reference_json.read_text())['sites']
    reader = args.sat_source / 'torch_points3d/modules/KPConv/plyutils.py'
    spec = importlib.util.spec_from_file_location('ply72', reader)
    ply = importlib.util.module_from_spec(spec); spec.loader.exec_module(ply)
    args.output.mkdir(parents=True)
    instances, crops, site_results = [], [], []
    for sid in site_ids:
        input_dir = args.candidates / sid
        path = input_dir / (sid + '_point_candidates.ply')
        for p in [path, input_dir / 'candidate_record.json']:
            if sha256(p) != source_manifest[str(p.relative_to(args.candidates))]:
                raise ValueError('candidate hash changed: ' + str(p))
        record = json.loads((input_dir / 'candidate_record.json').read_text())
        if record['site_id'] != sid or record['split'] != 'train' or sha256(path) != record['ply_sha256']:
            raise ValueError('candidate provenance mismatch')
        export = args.alto_crowns / (sid + '_public_crowns.geojson')
        if sha256(export) != record['alto_export_sha256']:
            raise ValueError('ALTO input hash changed')
        geo = json.loads(export.read_text())
        if geo.get('crs', {}).get('properties', {}).get('name') != 'EPSG:2193':
            raise ValueError('explicit EPSG:2193 required')
        alto = {str(f['properties']['tree_id']): make_valid(shape(f['geometry'])) for f in geo['features']}
        raw = ply.read_ply(str(path))
        xyz = np.column_stack([raw[k] for k in ['X', 'Y', 'Z']])
        world = xyz.astype('f8') + np.asarray(record['coordinate_origin_xyz_m'])
        sem, ids = raw['semantic_seg'], raw['treeID']
        region = box(*sites[sid]['bbox_epsg2193'])
        singles = {r['id']: make_valid(Polygon(r['world'])) for r in labels[sid]['labels']
                   if r['cls'] == 'tree' and not r.get('deleted')}
        proposals = {p['candidate_treeID']: p for p in record['new_group_candidates']}
        mapping = record['candidate_id_to_polygon_id']
        geometry, keep = {}, set()
        for key, label_id in mapping.items():
            ident = int(key)
            is_alto = label_id.startswith('ALTO:')
            g = alto[label_id[5:]] if is_alto else singles[label_id]
            geometry[ident] = g
            count = int(((sem == 2) & (ids == ident)).sum())
            reasons = screen_reason(count, region.covers(g), proposals.get(ident), args.minimum_points)
            if not reasons:
                keep.add(ident)
            instances.append({'site_id': sid, 'treeID': ident, 'label_id': label_id,
                'label_source': 'ALTO' if is_alto else 'manual_crown_guidance',
                'assigned_points_before': count, 'retained': not reasons,
                'rejection_reasons': '|'.join(reasons)})
        new_sem, new_ids, rejected = retain_labels(sem, ids, keep)
        if np.any(new_ids[new_sem != 2] != -1) or np.any(new_ids[new_sem == 2] < 0):
            raise ValueError('raw label convention violated')
        out = args.output / sid; out.mkdir()
        target = out / (sid + '_screened_weak.ply')
        if not ply.write_ply(str(target), [xyz, new_sem, new_ids], ['X', 'Y', 'Z', 'semantic_seg', 'treeID']):
            raise ValueError('PLY write failed')
        check = ply.read_ply(str(target))
        if (not np.array_equal(xyz, np.column_stack([check[k] for k in ['X', 'Y', 'Z']]))
                or not np.array_equal(new_sem, check['semantic_seg'])
                or not np.array_equal(new_ids, check['treeID'])):
            raise ValueError('coordinate/label round-trip failure')
        site_crops = []
        for ident in sorted(keep):
            x0, y0, x1, y1 = geometry[ident].bounds
            c = args.crop_context_m
            crop = box(x0-c, y0-c, x1+c, y1+c).intersection(region)
            selected, present, truncated, unknown = crop_checks(world[:, :2], new_sem, new_ids, crop, geometry)
            complete = bool(present) and not truncated and unknown == 0
            row = {'site_id': sid, 'seed_treeID': ident, 'points': int(selected.sum()),
                   'tree_ids': '|'.join(map(str, present)), 'unknown_points': unknown,
                   'truncated_tree_ids': '|'.join(map(str, truncated)),
                   'complete_under_candidate_labels': complete,
                   'crop_bounds_epsg2193': json.dumps(crop.bounds), 'exported_ply': ''}
            if complete:
                dest = out / ('crop_' + str(ident) + '_weak.ply')
                if not ply.write_ply(str(dest), [xyz[selected], new_sem[selected], new_ids[selected]],
                                     ['X', 'Y', 'Z', 'semantic_seg', 'treeID']):
                    raise ValueError('crop write failed')
                row['exported_ply'] = str(dest.relative_to(args.output))
            crops.append(row); site_crops.append(row)
        report = {'site_id': sid, 'split': 'train', 'points': len(xyz),
            'coordinate_origin_xyz_m': record['coordinate_origin_xyz_m'],
            'source_ply_sha256': sha256(path), 'screened_ply_sha256': sha256(target),
            'retained_original_ids': sum(not mapping[str(k)].startswith('ALTO:') for k in keep),
            'retained_alto_ids': sum(mapping[str(k)].startswith('ALTO:') for k in keep),
            'retained_tree_points': int((new_sem == 2).sum()),
            'labels_demoted_to_unknown': int(rejected.sum()), 'unknown_points': int((new_sem == 0).sum()),
            'crop_candidates': len(site_crops),
            'complete_crops_under_candidate_labels': sum(r['complete_under_candidate_labels'] for r in site_crops),
            'point_labels_manually_checked': False, 'fine_tuning_permitted': False,
            'label_type': 'automatically screened weak candidates with shared annotation provenance',
            'unknown_retained_as_input_points': True, 'original_point_order_preserved': True}
        write_json(out / 'screening_record.json', report)
        site_results.append(report)
        print(sid, report['retained_original_ids'], 'original IDs;', report['retained_alto_ids'],
              'ALTO IDs;', report['complete_crops_under_candidate_labels'], 'complete candidate crops', flush=True)
    for name, rows in [('instance_screening.csv', instances), ('crop_screening.csv', crops)]:
        with (args.output / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['site_id'])
            writer.writeheader(); writer.writerows(rows)
    report = {'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_screen_weak_labels',
        'status': 'AUTOMATED TRAINING WEAK-LABEL SCREEN; NOT CHECKED DATA',
        'sites': site_results, 'minimum_points': args.minimum_points, 'crop_context_m': args.crop_context_m,
        'support_threshold_purpose': 'engineering feasibility screen, not measured membership accuracy',
        'total_retained_original_ids': sum(r['retained_original_ids'] for r in site_results),
        'total_retained_alto_ids': sum(r['retained_alto_ids'] for r in site_results),
        'total_retained_tree_points': sum(r['retained_tree_points'] for r in site_results),
        'total_complete_crops_under_candidate_labels': sum(r['complete_crops_under_candidate_labels'] for r in site_results),
        'elapsed_seconds': time.monotonic() - start, 'heldout_used': False,
        'model_training_executed': False, 'fine_tuning_permitted': False,
        'source_manifest_sha256': sha256(args.candidates / 'SHA256_manifest.json'),
        'reference_sha256': sha256(args.reference_json), 'reader_sha256': sha256(reader),
        'source_script_sha256': sha256(Path(__file__)),
        'limitations': ['XY geometry and support do not establish 3D ownership',
                        'Unknown points remain real inputs, not negative background',
                        'Complete under weak labels is not independent/manual verification',
                        'Shared ALTO annotation provenance',
                        'A stock model forward/loss/backward check is still pending']}
    write_json(args.output / 'summary.json', report)
    (args.output / 'executed_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output / 'SHA256_manifest.json', {str(p.relative_to(args.output)): sha256(p)
        for p in args.output.rglob('*') if p.is_file()})
    print(json.dumps({k: report[k] for k in ['run_id', 'total_retained_original_ids',
        'total_retained_alto_ids', 'total_retained_tree_points',
        'total_complete_crops_under_candidate_labels', 'elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
