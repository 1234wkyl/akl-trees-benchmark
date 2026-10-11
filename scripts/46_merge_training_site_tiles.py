#!/usr/bin/env python3
"""Merge reconciled SegmentAnyTree tiles for one training site.

Local instance IDs are made globally unique. Exact duplicate XYZ records link
positive instances only when their semantic predictions agree. A stable
tile-interior rule selects one record per duplicate XYZ. Header footprints are
exported as an observed-domain diagnostic; they do not change any denominator.
"""
import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
from pyproj import CRS
from shapely.geometry import box
from shapely.ops import unary_union

from benchmark.common import ROOT, sha256, site_definitions, write_json
from benchmark.inference_scenario import validate_inference_scenario

DEFAULT_PLAN = ROOT / 'docs/project_status_2026-09-18/tile_coverage_preflight/coverage_plan.json'


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a == b:
            return False
        if b < a:
            a, b = b, a
        self.parent[b] = a
        return True


def _load_tile(tile_name, prediction_path, audit_path, footprint, context, verified_input=None):
    audit = json.loads(audit_path.read_text())
    if audit.get('prediction_sha256') != sha256(prediction_path):
        raise ValueError(f'{tile_name}: coordinate audit does not bind prediction')
    if audit.get('epsg2193_coordinate_consistency') is not True or audit.get('uint16_boundary_warning'):
        raise ValueError(f'{tile_name}: successful coordinate audit required')
    cloud = laspy.read(prediction_path)
    crs = cloud.header.parse_crs()
    if crs is None or crs.to_epsg() != 2193:
        raise ValueError(f'{tile_name}: explicit EPSG:2193 required')
    dimensions = set(cloud.point_format.dimension_names)
    if not {'PredInstance', 'PredSemantic'} <= dimensions:
        raise ValueError(f'{tile_name}: predicted fields missing')
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError(f'{tile_name}: empty or non-finite prediction')
    expected = np.asarray(footprint, dtype=float)
    actual = np.asarray([cloud.x.min(), cloud.y.min(), cloud.x.max(), cloud.y.max()])
    extent = {'raw_coverage_bbox_epsg2193': expected.tolist(),
              'prediction_bbox_epsg2193': actual.tolist(),
              'prediction_minus_raw_bbox_m': (actual - expected).tolist(),
              'validation_basis': 'raw coverage extent'}
    if verified_input is None:
        if np.max(np.abs(actual - expected)) > .02:
            raise ValueError(f'{tile_name}: prediction extent differs from coverage plan')
    else:
        verified_input = Path(verified_input)
        input_hash = sha256(verified_input)
        if audit.get('input_sha256') != input_hash:
            raise ValueError(f'{tile_name}: audit does not bind verified filtered input')
        if audit.get('point_count_equal') is not True or audit.get('bidirectional_coordinate_coverage_ok') is not True:
            raise ValueError(f'{tile_name}: complete coordinate coverage audit required')
        staged = laspy.read(verified_input)
        if staged.header.parse_crs() is None or staged.header.parse_crs().to_epsg() != 2193:
            raise ValueError(f'{tile_name}: verified input CRS mismatch')
        if len(staged.points) != len(cloud.points) or audit.get('input_points') != len(staged.points):
            raise ValueError(f'{tile_name}: verified input point count mismatch')
        filtered_bbox = np.asarray([staged.x.min(), staged.y.min(), staged.x.max(), staged.y.max()])
        if not np.isfinite(filtered_bbox).all() or np.max(np.abs(actual - filtered_bbox)) > .002:
            raise ValueError(f'{tile_name}: prediction extent differs from audited filtered input')
        if (np.any(filtered_bbox[:2] < expected[:2] - .02) or
                np.any(filtered_bbox[2:] > expected[2:] + .02)):
            raise ValueError(f'{tile_name}: filtered input extends outside raw coverage')
        extent.update(validation_basis='hash-bound filtered inference input and bidirectional coordinate audit',
                      verified_input=str(verified_input.resolve()), verified_input_sha256=input_hash,
                      filtered_bbox_epsg2193=filtered_bbox.tolist())
    keep = ((xyz[:, 0] >= context[0]) & (xyz[:, 0] <= context[2]) &
            (xyz[:, 1] >= context[1]) & (xyz[:, 1] <= context[3]))
    margin = np.minimum.reduce((xyz[:, 0] - expected[0], expected[2] - xyz[:, 0],
                                xyz[:, 1] - expected[1], expected[3] - xyz[:, 1]))
    instance = np.asarray(cloud['PredInstance']).astype(np.int64)
    semantic = np.asarray(cloud['PredSemantic']).astype(np.int64)
    if np.any(instance < 0):
        raise ValueError(f'{tile_name}: negative instance IDs are not supported')
    if np.any((semantic < 0) | (semantic > 255)):
        raise ValueError(f'{tile_name}: semantic labels must fit uint8')
    return {'xyz': xyz[keep], 'instance': instance[keep],
            'semantic': semantic[keep],
            'margin': margin[keep], 'prediction_sha256': sha256(prediction_path),
            'audit_sha256': sha256(audit_path), 'points_full_tile': len(cloud.points),
            'points_context': int(keep.sum()), 'extent_validation': extent}


def merge_tiles(tiles):
    xyz = np.concatenate([tile['xyz'] for tile in tiles])
    instance = np.concatenate([tile['instance'] for tile in tiles])
    semantic = np.concatenate([tile['semantic'] for tile in tiles])
    margin = np.concatenate([tile['margin'] for tile in tiles])
    tile_index = np.concatenate([np.full(len(tile['xyz']), index, dtype=np.int32)
                                 for index, tile in enumerate(tiles)])
    row_index = np.concatenate([np.arange(len(tile['xyz']), dtype=np.int64) for tile in tiles])
    if not len(xyz):
        raise ValueError('no points within requested site context')
    nodes = sorted({(int(t), int(i)) for t, i in zip(tile_index, instance) if i > 0})
    union = UnionFind(nodes)
    _, inverse, counts = np.unique(xyz, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    selected, duplicate_groups, semantic_conflicts, positive_zero_conflicts, union_links = [], 0, 0, 0, 0
    for start, count in zip(starts, counts):
        indexes = order[start:start + count]
        if count > 1:
            duplicate_groups += 1
            values = set(map(int, semantic[indexes]))
            if len(values) > 1:
                semantic_conflicts += 1
            positive = [(int(tile_index[i]), int(instance[i])) for i in indexes if instance[i] > 0]
            if positive and len(positive) != len(indexes):
                positive_zero_conflicts += 1
            if len(values) == 1 and len(set(positive)) > 1:
                first = positive[0]
                for node in positive[1:]:
                    union_links += int(union.union(first, node))
        # Prefer the point furthest from its source tile edge, then stable tile/row order.
        best = sorted(indexes, key=lambda i: (-margin[i], tile_index[i], row_index[i]))[0]
        selected.append(best)
    represented_roots = sorted({union.find((int(tile_index[i]), int(instance[i])))
                                for i in selected if instance[i] > 0})
    roots_without_selected_points = len({union.find(node) for node in nodes}) - len(represented_roots)
    root_to_global = {root: index + 1 for index, root in enumerate(represented_roots)}
    global_instance = np.zeros(len(selected), dtype=np.uint32)
    for output_index, source_index in enumerate(selected):
        local = int(instance[source_index])
        if local > 0:
            global_instance[output_index] = root_to_global[union.find((int(tile_index[source_index]), local))]
    selected = np.asarray(selected, dtype=np.int64)
    return {'xyz': xyz[selected], 'instance': global_instance,
            'semantic': semantic[selected].astype(np.uint8),
            'source_tile': tile_index[selected].astype(np.uint16),
            'input_context_records': len(xyz), 'output_unique_xyz': len(selected),
            'duplicate_xyz_groups': duplicate_groups,
            'duplicate_records_removed': len(xyz) - len(selected),
            'semantic_conflict_groups': semantic_conflicts,
            'positive_zero_conflict_groups': positive_zero_conflicts,
            'local_positive_instance_nodes': len(nodes),
            'cross_tile_union_links': union_links,
            'positive_instance_roots_without_selected_points': roots_without_selected_points,
            'global_positive_instances': len(represented_roots)}


def _write_cloud(path, merged):
    xyz = merged['xyz']
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.add_crs(CRS.from_epsg(2193))
    header.scales = np.array([.001, .001, .001])
    header.offsets = np.floor(xyz.min(axis=0))
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cloud.add_extra_dim(laspy.ExtraBytesParams(name='PredInstance', type='uint32'))
    cloud.add_extra_dim(laspy.ExtraBytesParams(name='PredSemantic', type='uint8'))
    cloud.add_extra_dim(laspy.ExtraBytesParams(name='SourceTile', type='uint16'))
    cloud['PredInstance'] = merged['instance']
    cloud['PredSemantic'] = merged['semantic']
    cloud['SourceTile'] = merged['source_tile']
    cloud.write(path, do_compress=path.suffix.lower() == '.laz')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site-id', required=True)
    parser.add_argument('--context-m', type=float, required=True)
    parser.add_argument('--tile', nargs=3, action='append', required=True,
                        metavar=('SOURCE_TILE_NAME', 'RECONCILED_PREDICTION', 'COORDINATE_AUDIT'))
    parser.add_argument('--coverage-plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inference-scenario', type=Path)
    parser.add_argument('--verified-input', nargs=2, action='append',
                        metavar=('SOURCE_TILE_NAME', 'FILTERED_INPUT_LAZ'),
                        help='audit-bound actual model input; retains raw footprint diagnostics')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh output directory')
    if not math.isfinite(args.context_m) or args.context_m < 0:
        parser.error('finite nonnegative context required')
    sites = site_definitions()
    scenario = validate_inference_scenario(args.inference_scenario) if args.inference_scenario else None
    if args.site_id not in sites or (not scenario and sites[args.site_id]['split'] != 'train'):
        raise ValueError('this pre-freeze merger accepts training sites only')
    plan = json.loads(args.coverage_plan.read_text())
    plan_sites = {row['site_id']: row for row in plan['sites']}
    plan_tiles = {row['tile']: row for row in plan['tiles']}
    expected = set(plan_sites[args.site_id]['required_tiles'])
    provided = [row[0] for row in args.tile]
    if len(provided) != len(set(provided)) or set(provided) != expected:
        raise ValueError(f'provide each planned site tile exactly once; expected {sorted(expected)}')
    conflicts = {name: plan_tiles[name].get('held_out_intersecting_sites', [])
                 for name in expected if plan_tiles[name].get('held_out_intersecting_sites')}
    if conflicts and not scenario:
        raise ValueError(f'training merge rejected because required tiles touch held-out sites: {conflicts}')
    verified = dict(args.verified_input or [])
    if args.verified_input and (len(verified) != len(args.verified_input) or set(verified) != expected):
        raise ValueError('provide each planned filtered input exactly once')
    site = sites[args.site_id]
    site_box = box(*site['bbox_epsg2193'])
    context = site_box.buffer(args.context_m, cap_style=3).bounds
    loaded, tile_receipts = [], []
    for tile_name, prediction, audit in sorted(args.tile):
        prediction, audit = Path(prediction), Path(audit)
        if tile_name not in plan_tiles or not prediction.is_file() or not audit.is_file():
            raise ValueError(f'{tile_name}: missing plan, prediction or audit')
        record = _load_tile(tile_name, prediction, audit, plan_tiles[tile_name]['bbox_epsg2193'], context, verified.get(tile_name))
        loaded.append(record)
        tile_receipts.append({'tile': tile_name, 'prediction': str(prediction.resolve()),
                              'prediction_sha256': record['prediction_sha256'],
                              'coordinate_audit': str(audit.resolve()),
                              'coordinate_audit_sha256': record['audit_sha256'],
                              'points_full_tile': record['points_full_tile'],
                              'points_context': record['points_context'],
                              'extent_validation': record['extent_validation']})
    merged = merge_tiles(loaded)
    args.output.mkdir(parents=True)
    target = args.output / f'{args.site_id}_merged_points.laz'
    _write_cloud(target, merged)
    footprints = [box(*plan_tiles[name]['bbox_epsg2193']) for name in sorted(expected)]
    observed = unary_union(footprints).intersection(site_box)
    unobserved = site_box.difference(observed)
    coverage_path = args.output / f'{args.site_id}_coverage_domains.gpkg'
    gpd.GeoDataFrame({'site_id': [args.site_id], 'domain': ['site_boundary']},
                     geometry=[site_box], crs=2193).to_file(coverage_path, layer='site_boundary', driver='GPKG')
    gpd.GeoDataFrame({'site_id': [args.site_id], 'domain': ['header_bbox_observed']},
                     geometry=[observed], crs=2193).to_file(coverage_path, layer='observed_header_domain', driver='GPKG')
    if not unobserved.is_empty:
        gpd.GeoDataFrame({'site_id': [args.site_id], 'domain': ['header_bbox_unobserved']},
                         geometry=[unobserved], crs=2193).to_file(coverage_path, layer='unobserved_header_domain', driver='GPKG')
    report = {
        'status': ('ASSUMPTION-ONLY SITE MERGE COMPLETED; NO SCORING' if scenario else 'TRAINING-SITE MULTI-TILE MERGE COMPLETED'),
        'site_id': args.site_id, 'split': site['split'], 'stratum': site['stratum'],
        'context_m': args.context_m, 'expected_source_tiles': sorted(expected),
        'tiles': tile_receipts, 'merge': {key: value for key, value in merged.items() if key not in {'xyz', 'instance', 'semantic', 'source_tile'}},
        'duplicate_rule': 'exact XYZ; select greatest source-tile interior margin then stable tile/row order',
        'instance_rule': 'tile-local IDs are distinct; union positive nodes only through exact duplicate XYZ with one semantic value',
        'seam_rule': 'No proximity-only or polygon-overlap seam merge is applied; exact duplicate XYZ is the only cross-tile identity evidence.',
        'semantic_conflicts_are_not_unioned': True,
        'output_laz': str(target.resolve()), 'output_laz_sha256': sha256(target),
        'coverage_domains': str(coverage_path.resolve()), 'coverage_domains_sha256': sha256(coverage_path),
        'header_bbox_site_coverage_fraction': observed.area / site_box.area,
        'unobserved_header_bbox_area_m2': unobserved.area,
        'coverage_limit': 'Header footprints do not prove point support inside their bounds. The mask is diagnostic and does not remove references or change scoring denominators.',
        'density_limit': 'Exact XYZ deduplication makes this a crown-conversion cloud; do not use it for return-density accounting.',
        'coverage_plan_sha256': sha256(args.coverage_plan),
        'site_definitions_sha256': sha256(ROOT / 'data/sites/reference_sites.json'),
        'reference_frozen': False, 'benchmark_scores_created': False,
        'inference_scenario_sha256': sha256(args.inference_scenario) if scenario else None,
        'evaluation_permitted': False,
    }
    write_json(args.output / 'merge_receipt.json', report)
    write_json(args.output / 'SHA256_manifest.json',
               {path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()})
    print(json.dumps({'output': str(args.output), 'site_id': args.site_id,
                      'points': merged['output_unique_xyz'],
                      'global_instances': merged['global_positive_instances'],
                      'coverage_fraction': report['header_bbox_site_coverage_fraction']}, indent=2))


if __name__ == '__main__':
    main()
