#!/usr/bin/env python3
"""Use cached ALTO map crowns to propose point membership inside training groups.

Review-only outputs: no reference edits, checked-label claim or model training.
The original single-tree point candidates are preserved. Public simplified
crowns supply XY guidance, never original per-return labels or surveyed Z.
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from shapely import intersects_xy, make_valid
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union
from benchmark.common import ROOT, sha256, site_definitions, write_json


def point_mask(geometries, x, y):
    result = np.zeros(len(x), dtype=bool)
    for geometry in geometries:
        result |= intersects_xy(geometry, x, y)
    return result


def assign_group_proposals(x, y, classification, original_sem, original_ids,
                           groups, exclusions, crowns, start_id, margin_m=0.5):
    """Unique vegetation in one group and one ALTO crown core; abstain otherwise.

    IDs refer to ALTO model hypotheses, not verified biological individuals.
    Full-crown overlaps are rejected even if only one shrunken core contains
    the point. All original known labels and all unassigned non-group points
    are preserved exactly.
    """
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError('finite nonnegative review margin required')
    group_hits = np.zeros(len(x), dtype=np.int32)
    for geometry in groups:
        group_hits += intersects_xy(geometry, x, y)
    blocked = point_mask(exclusions, x, y)
    eligible = ((group_hits == 1) & ~blocked & (original_sem == 0)
                & (original_ids == -1) & np.isin(classification, [4, 5]))
    hits = np.zeros(len(x), dtype=np.int32)
    core_owner = np.full(len(x), -1, dtype=np.int32)
    for index, (_, geometry) in enumerate(crowns):
        hits += intersects_xy(geometry, x, y)
        core = geometry.buffer(-margin_m) if margin_m else geometry
        core_owner[intersects_xy(core, x, y)] = start_id + index
    selected = eligible & (hits == 1) & (core_owner >= start_id)
    semantics = original_sem.copy()
    instances = original_ids.copy()
    semantics[selected] = 2
    instances[selected] = core_owner[selected]
    diagnostics = {
        'new_group_proposal_points': int(selected.sum()),
        'eligible_group_vegetation_points': int(eligible.sum()),
        'eligible_overlapping_alto_crown_points': int((eligible & (hits > 1)).sum()),
        'eligible_boundary_or_no_core_points': int((eligible & (hits == 1) & (core_owner < start_id)).sum()),
        'eligible_points_without_alto_crown': int((eligible & (hits == 0)).sum()),
    }
    return semantics, instances, selected, diagnostics


def training_sites(requested, definitions):
    ids = requested or sorted(k for k, v in definitions.items() if v['split'] == 'train')
    if len(set(ids)) != len(ids) or any(k not in definitions or definitions[k]['split'] != 'train' for k in ids):
        raise ValueError('unique training sites only; validation/test are forbidden')
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-json', type=Path, required=True)
    ap.add_argument('--point-candidates', type=Path, required=True)
    ap.add_argument('--alto-crowns', type=Path, required=True)
    ap.add_argument('--alto-receipt', type=Path, required=True)
    ap.add_argument('--sat-source', type=Path, default=ROOT.parent / 'SegmentAnyTree')
    ap.add_argument('--review-margin-m', type=float, default=0.5)
    ap.add_argument('--site-ids', nargs='+')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('a fresh output directory is required')
    if not np.isfinite(args.review_margin_m) or args.review_margin_m < 0:
        ap.error('finite nonnegative review margin required')
    definitions = site_definitions()
    sites = training_sites(args.site_ids, definitions)
    reference = json.loads(args.reference_json.read_text())['sites']
    source_manifest = json.loads((args.point_candidates / 'SHA256_manifest.json').read_text())
    alto_receipt = json.loads(args.alto_receipt.read_text())
    if alto_receipt.get('public_map_zoom') != 18:
        raise ValueError('expected the recorded zoom-18 map export')
    expected_exports = alto_receipt['artifact_manifest']
    reader_path = args.sat_source / 'torch_points3d/modules/KPConv/plyutils.py'
    spec = importlib.util.spec_from_file_location('sat_ply_reader', reader_path)
    ply = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ply)
    args.output.mkdir(parents=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_alto_group_point_proposals'
    (args.output / 'executed_source.py').write_bytes(Path(__file__).read_bytes())
    records = []
    for sid in sites:
        source = args.point_candidates / sid
        for name in ['candidate_record.json', 'viewer_points.json', sid + '_point_candidates.ply']:
            if sha256(source / name) != source_manifest[sid + '/' + name]:
                raise ValueError('point input differs from its original recorded hash: ' + name)
        old_record = json.loads((source / 'candidate_record.json').read_text())
        old_ply = source / (sid + '_point_candidates.ply')
        viewer = json.loads((source / 'viewer_points.json').read_text())
        if (old_record['site_id'] != sid or old_record['split'] != 'train'
                or old_record['crs'] != 'EPSG:2193' or sha256(old_ply) != old_record['ply_sha256']
                or viewer['site_id'] != sid
                or viewer['origin_xyz_m'] != old_record['coordinate_origin_xyz_m']):
            raise ValueError('point candidate provenance mismatch: ' + sid)
        decoded = ply.read_ply(str(old_ply))
        xyz = np.column_stack([decoded[k] for k in ['X', 'Y', 'Z']])
        sem = decoded['semantic_seg'].astype('i4')
        ids = decoded['treeID'].astype('i4')
        points = np.asarray(viewer['points'], dtype='f8')
        if (len(points) != len(xyz) or points.shape[1] != 6
                or not np.isfinite(points).all()
                or np.abs(points[:, :3] - xyz).max() > 1e-4
                or not np.array_equal(points[:, 3], sem)
                or not np.array_equal(points[:, 4], ids)):
            raise ValueError('point rows/classification cannot be aligned to source PLY')
        world = xyz.astype('f8') + np.asarray(viewer['origin_xyz_m'])
        site_box = box(*definitions[sid]['bbox_epsg2193'])
        if not intersects_xy(site_box, world[:, 0], world[:, 1]).all():
            raise ValueError('point coordinates outside training site')
        labels = [r for r in reference[sid]['labels'] if not r.get('deleted')]
        original_ids = old_record['candidate_id_to_polygon_id']
        if set(original_ids.values()) != {r['id'] for r in labels if r['cls'] == 'tree'}:
            raise ValueError('current single-tree IDs differ from the original candidate inputs')
        groups = [(r['id'], make_valid(Polygon(r['world'])).intersection(site_box))
                  for r in labels if r['cls'] == 'tree_group']
        exclusions = [make_valid(Polygon(r['world'])) for r in labels
                      if r['cls'] in ['tree', 'not_tree', 'uncertain']]
        export = args.alto_crowns / (sid + '_public_crowns.geojson')
        if sha256(export) != expected_exports['comparison/' + export.name]:
            raise ValueError('cached ALTO crown hash differs from intake receipt')
        geo = json.loads(export.read_text())
        if geo.get('crs', {}).get('properties', {}).get('name') != 'EPSG:2193':
            raise ValueError('explicit EPSG:2193 ALTO geometry required')
        crowns = []
        for feature in geo['features']:
            prop = feature['properties']
            geometry = make_valid(shape(feature['geometry']))
            if prop['site_id'] != sid:
                raise ValueError('ALTO site mismatch')
            if any(geometry.intersection(g).area > 0 for _, g in groups):
                crowns.append((str(prop['tree_id']), geometry))
        crowns.sort(key=lambda r: r[0])
        if len({k for k, _ in crowns}) != len(crowns):
            raise ValueError('duplicate ALTO instance key')
        offset = max(map(int, original_ids), default=-1) + 1
        new_sem, new_ids, added, diagnostic = assign_group_proposals(
            world[:, 0], world[:, 1], points[:, 5], sem, ids,
            [g for _, g in groups], exclusions, crowns, offset, args.review_margin_m)
        mapping = dict(original_ids)
        mapping.update({str(offset + i): 'ALTO:' + ident for i, (ident, _) in enumerate(crowns)})
        out = args.output / sid
        out.mkdir()
        target = out / (sid + '_point_candidates.ply')
        if not ply.write_ply(str(target), [xyz, new_sem, new_ids], ['X', 'Y', 'Z', 'semantic_seg', 'treeID']):
            raise ValueError('PLY export failed')
        check = ply.read_ply(str(target))
        if (not np.array_equal(xyz, np.column_stack([check[k] for k in ['X', 'Y', 'Z']]))
                or not np.array_equal(new_ids, check['treeID'])
                or not np.array_equal(new_sem, check['semantic_seg'])
                or not np.array_equal(ids[~added], new_ids[~added])):
            raise ValueError('coordinate/label round trip or preservation failed')
        counts = {str(k): int((new_ids[new_sem == 2] == int(k)).sum()) for k in mapping}
        proposals = []
        group_union = unary_union([g for _, g in groups])
        for i, (ident, geometry) in enumerate(crowns):
            proposals.append({'candidate_treeID': offset + i, 'alto_tree_id': ident,
                              'assigned_points': counts[str(offset + i)],
                              'parent_group_ids': [k for k, g in groups if geometry.intersection(g).area > 0],
                              'source_geometry_extends_outside_site': not site_box.covers(geometry),
                              'source_geometry_extends_outside_group_union': not group_union.covers(geometry)})
            # The original geometry is retained; assignment is explicitly limited
            # to group points. No clipped fragment is claimed as a complete tree.
        record = {**old_record, **diagnostic,
                  'run_id': run_id, 'status': 'ALTO-ASSISTED TRAINING GROUP PROPOSALS; MANUAL QA PENDING',
                  'label_source': 'ALTO-assisted training-group point proposals',
                  'reference_json_sha256': sha256(args.reference_json),
                  'source_candidate_ply_sha256': sha256(old_ply),
                  'source_viewer_sha256': sha256(source / 'viewer_points.json'),
                  'source_candidate_record_sha256': sha256(source / 'candidate_record.json'),
                  'alto_export_sha256': sha256(export), 'ply_sha256': sha256(target),
                  'alto_release': alto_receipt['source_release_metadata'],
                  'alto_map_geometry': 'zoom 18; simplified, quantized map geometry',
                  'same_acquisition_verified': False, 'review_margin_m': args.review_margin_m,
                  'candidate_id_to_polygon_id': mapping, 'candidate_instance_point_counts': counts,
                  'candidate_instances_with_points': sum(v > 0 for v in counts.values()),
                  'new_group_ids_with_points': sum(p['assigned_points'] > 0 for p in proposals),
                  'tree_candidate_points': int((new_sem == 2).sum()),
                  'unknown_points': int((new_sem == 0).sum()),
                  'new_group_candidates': proposals, 'point_labels_manually_checked': False,
                  'fine_tuning_permitted': False, 'reference_labels_modified': False,
                  'limitations': ['ALTO model boundaries are hypotheses, not independent truth',
                                  'Source acquisition identity has not been verified',
                                  'Group-only point subsets may omit part of a tree; inspect complete crops',
                                  'Vegetation class and XY location do not establish 3D ownership',
                                  'No validation/test labels are used; no model weights are updated']}
        write_json(out / 'candidate_record.json', record)
        new_points = np.column_stack([xyz, new_sem, new_ids, points[:, 5]]).tolist()
        write_json(out / 'viewer_points.json', {'site_id': sid, 'origin_xyz_m': viewer['origin_xyz_m'],
                                               'mapping': mapping, 'points': new_points})
        write_json(out / 'qa_template.json', {'site_id': sid, 'ply_sha256': sha256(target),
            'coordinate_origin_xyz_m': viewer['origin_xyz_m'], 'reviewer': None,
            'point_labels_manually_checked': False, 'fine_tuning_permitted': False,
            'tasks': ['Compare ALTO proposals with original RGB/CHM and real 3D returns',
                      'Correct complete tree membership and background, including understory',
                      'Resolve or remove ambiguous regions; never mark these files checked automatically']})
        records.append({k: record[k] for k in ['site_id', 'points', 'new_group_ids_with_points',
                                             'new_group_proposal_points', 'unknown_points']})
        print(sid, diagnostic['new_group_proposal_points'], 'new proposal points;',
              record['new_group_ids_with_points'], 'ALTO IDs with support', flush=True)
    summary = {'run_id': run_id, 'argv': sys.argv, 'site_ids': sites, 'sites': records,
               'reference_sha256': sha256(args.reference_json),
               'site_definitions_sha256': sha256(ROOT / 'data/sites/reference_sites.json'),
               'alto_receipt_sha256': sha256(args.alto_receipt),
               'source_reader_sha256': sha256(reader_path),
               'source_script_sha256': sha256(Path(__file__)),
               'review_margin_m': args.review_margin_m,
               'total_new_group_proposal_points': sum(r['new_group_proposal_points'] for r in records),
               'total_new_group_ids_with_points': sum(r['new_group_ids_with_points'] for r in records),
               'same_acquisition_verified': False, 'reference_labels_modified': False,
               'fine_tuning_permitted': False, 'model_run_executed': False}
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'SHA256_manifest.json', {str(p.relative_to(args.output)): sha256(p)
               for p in sorted(args.output.rglob('*')) if p.is_file()})


if __name__ == '__main__':
    main()
