#!/usr/bin/env python
"""Count restored SAT semantic/instance support inside training reference crowns.

These are point-level error diagnostics, not biological stem counts or an
alternative detection match. The reference remains pre-freeze.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
from shapely import contains_xy

from benchmark.common import sha256, site_definitions, write_json


def support(geometry, x, y, semantic, instance, tree_values):
    mask = contains_xy(geometry, x, y)
    positive = instance[mask & (instance > 0)]
    return {
        'points': int(mask.sum()),
        'tree_semantic_points': int((mask & np.isin(semantic, tree_values)).sum()),
        'positive_instance_points': int((mask & (instance > 0)).sum()),
        'positive_instance_ids': ';'.join(map(str, np.unique(positive))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site-id', required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--prediction', type=Path, required=True)
    ap.add_argument('--coordinate-audit', type=Path, required=True)
    ap.add_argument('--tree-semantic-values', type=int, nargs='+', required=True)
    ap.add_argument('--buffer-m', type=float, default=2.0)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sites = site_definitions()
    if args.site_id not in sites or sites[args.site_id]['split'] != 'train':
        ap.error('point-support audit is restricted to training sites')
    if args.output.exists():
        ap.error('use a fresh output folder')
    if not math.isfinite(args.buffer_m) or args.buffer_m < 0:
        ap.error('finite nonnegative buffer required')
    tree_values = tuple(args.tree_semantic_values)
    if not tree_values or len(set(tree_values)) != len(tree_values) or any(v < 0 or v > 255 for v in tree_values):
        ap.error('explicit distinct uint8 tree semantic values required')
    audit = json.loads(args.coordinate_audit.read_text())
    if (audit.get('prediction_sha256') != sha256(args.prediction)
            or audit.get('epsg2193_coordinate_consistency') is not True
            or audit.get('uint16_boundary_warning')):
        raise ValueError('require matching successful restored-coordinate audit')

    refs = gpd.read_file(args.reference, layer='single_trees')
    if refs.crs is None or refs.crs.to_epsg() != 2193:
        raise ValueError('reference CRS must be EPSG:2193')
    refs = refs[refs.site_id == args.site_id].sort_values('label_id')
    if refs.empty or refs.label_id.astype(str).duplicated().any() or not refs.geometry.is_valid.all():
        raise ValueError('require unique valid eligible single-tree references')
    las = laspy.read(args.prediction)
    crs = las.header.parse_crs()
    if crs is None or crs.to_epsg() != 2193:
        raise ValueError('restored prediction CRS must be EPSG:2193')
    x, y = np.asarray(las.x), np.asarray(las.y)
    semantic = np.asarray(las['PredSemantic'])
    instance = np.asarray(las['PredInstance'])
    rows = []
    for ref in refs.itertuples():
        inside = support(ref.geometry, x, y, semantic, instance, tree_values)
        nearby = support(ref.geometry.buffer(args.buffer_m), x, y, semantic, instance, tree_values)
        rows.append({
            'site_id': args.site_id, 'label_id': str(ref.label_id),
            'reference_area_m2': float(ref.geometry.area), 'buffer_m': args.buffer_m,
            **{f'inside_{key}': value for key, value in inside.items()},
            **{f'buffered_{key}': value for key, value in nearby.items()},
        })
    args.output.mkdir(parents=True)
    with (args.output/'reference_point_support.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output/'summary.json', {
        'status': 'DEVELOPMENT ONLY; PRE-FREEZE TRAINING REFERENCE POINT AUDIT',
        'site_id': args.site_id, 'reference_singles': len(rows),
        'tree_semantic_values': list(tree_values), 'buffer_m': args.buffer_m,
        'reference_sha256': sha256(args.reference),
        'prediction_sha256': sha256(args.prediction),
        'coordinate_audit_sha256': sha256(args.coordinate_audit),
        'csv_sha256': sha256(args.output/'reference_point_support.csv'),
        'note': ('Inside and buffered point counts locate a possible semantic/instance failure; '
                 'they do not change references, detection matching or held-out scores.'),
    })
    print(f'Audited {len(rows)} training reference crowns on {args.site_id}')


if __name__ == '__main__':
    main()
