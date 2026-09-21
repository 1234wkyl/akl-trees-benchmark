#!/usr/bin/env python
"""Summarise matching full-density crown candidates across training sites only.

This is a pre-freeze development comparison. It never reads validation/test
predictions and cannot establish a benchmark result or pooled matched IoU.
It checks the split, retained fraction and safe source-tile header containment
against the coverage plan. Header extrema cannot prove observed point coverage.
"""
import argparse
import csv
import json
from pathlib import Path

from benchmark.common import ROOT, sha256, site_definitions, write_json


def full_safe_tile_names(site_id, sites, coverage):
    site = sites[site_id]
    record = next((row for row in coverage['sites'] if row['site_id'] == site_id), None)
    if record is None or record['split'] != 'train':
        raise ValueError(f'{site_id}: absent from training coverage plan')
    tiles = {row['tile']: row for row in coverage['tiles']}
    x1, y1, x2, y2 = site['bbox_epsg2193']
    full = []
    for name in record['safe_full_development_tiles']:
        tile = tiles[name]
        a, b, c, d = tile['bbox_epsg2193']
        if tile['safe_full_tile_for_development'] and a <= x1 and b <= y1 and c >= x2 and d >= y2:
            full.append(name)
    if not full:
        raise ValueError(f'{site_id}: no safe single source tile covers the full site header box')
    return full


def load_grid(path, sites, coverage):
    grid = json.loads(path.read_text())
    site_id = grid.get('site_id')
    if (grid.get('status') != 'DEVELOPMENT_ONLY_TRAINING_SITE_CONVERSION_COMPARISON'
            or site_id not in sites or sites[site_id]['split'] != 'train'
            or grid.get('site_split') != 'train'):
        raise ValueError(f'{path}: require one training-only conversion grid')
    if not isinstance(grid.get('reference_sha256'), str) or len(grid['reference_sha256']) != 64:
        raise ValueError(f'{path}: missing reference hash')
    safe_tiles = full_safe_tile_names(site_id, sites, coverage)
    full_conditions = [row for row in grid.get('conditions', []) if float(row['fraction']) == 1.0]
    if len(full_conditions) != 1:
        raise ValueError(f'{path}: require one full-density prediction condition')
    prediction_name = Path(full_conditions[0]['prediction']).name
    if not any(prediction_name.startswith(Path(name).stem + '_') for name in safe_tiles):
        raise ValueError(f'{path}: predicted tile is not a full, safe training tile for {site_id}')
    rows = {}
    for row in grid.get('rows', []):
        if float(row['fraction']) != 1.0:
            continue
        config_id = row['config_id']
        if config_id in rows:
            raise ValueError(f'{path}: duplicate full-density {config_id}')
        if row['method'] != 'convex_hull' or row['concave_ratio'] is not None:
            continue
        counts = {name: int(row[name]) for name in
                  ('tp', 'fp', 'fn', 'site_predictions', 'ignored_predictions')}
        if any(value < 0 for value in counts.values()):
            raise ValueError(f'{path}: negative count')
        if counts['site_predictions'] != counts['tp'] + counts['fp'] + counts['ignored_predictions']:
            raise ValueError(f'{path}: predictions do not reconcile')
        rows[config_id] = {**counts, 'buffer_m': float(row['buffer_m'])}
    return site_id, grid['reference_sha256'], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=Path, action='append', required=True,
                    help='summary.json from a separately verified complete training site')
    ap.add_argument('--coverage-plan', type=Path, required=True,
                    help='header-box safe-tile plan; this is not a point-coverage proof')
    ap.add_argument('--config', action='append', required=True,
                    help='identical convex candidate ID present in every grid')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if len(args.grid) < 2:
        ap.error('at least two training-site grids are required')
    if len(set(args.config)) != len(args.config):
        ap.error('configuration IDs must be unique')
    if args.output.exists():
        ap.error('use a fresh output folder')
    sites = site_definitions()
    coverage = json.loads(args.coverage_plan.read_text())
    if coverage.get('site_definitions_sha256') != sha256(ROOT/'data/sites/reference_sites.json'):
        raise ValueError('coverage plan does not match current site definitions')
    loaded = [(path, *load_grid(path, sites, coverage)) for path in args.grid]
    site_ids = [item[1] for item in loaded]
    if len(set(site_ids)) != len(site_ids):
        raise ValueError('duplicate training site')
    reference_hashes = {item[2] for item in loaded}
    if len(reference_hashes) != 1:
        raise ValueError('grids use different reference versions')
    rows = []
    for config_id in args.config:
        missing = [site for _, site, _, grid_rows in loaded if config_id not in grid_rows]
        if missing:
            raise ValueError(f'{config_id} absent from full-density convex grid(s): {missing}')
        buffers = {grid_rows[config_id]['buffer_m'] for _, _, _, grid_rows in loaded}
        if len(buffers) != 1:
            raise ValueError(f'{config_id} changes buffer by site')
        tp = sum(grid_rows[config_id]['tp'] for _, _, _, grid_rows in loaded)
        fp = sum(grid_rows[config_id]['fp'] for _, _, _, grid_rows in loaded)
        fn = sum(grid_rows[config_id]['fn'] for _, _, _, grid_rows in loaded)
        ignored = sum(grid_rows[config_id]['ignored_predictions'] for _, _, _, grid_rows in loaded)
        denominator = 2*tp + fp + fn
        rows.append({'config_id': config_id, 'buffer_m': buffers.pop(),
                     'sites': len(loaded), 'reference_singles': tp+fn,
                     'tp': tp, 'fp': fp, 'fn': fn, 'ignored_predictions': ignored,
                     'micro_precision': tp/(tp+fp) if tp+fp else None,
                     'micro_recall': tp/(tp+fn) if tp+fn else None,
                     'micro_f1': 2*tp/denominator if denominator else None})
    args.output.mkdir(parents=True)
    with (args.output/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output/'summary.json', {
        'status': 'DEVELOPMENT ONLY; PRE-FREEZE TRAINING-SITE COMPARISON',
        'warning': ('Full-density training sites only; complete-site coverage must be independently verified. Reference is not frozen; '
                    'do not treat micro metrics as held-out accuracy or use them to change acceptance rules.'),
        'site_ids': site_ids, 'reference_sha256': reference_hashes.pop(),
        'coverage_plan_sha256': sha256(args.coverage_plan),
        'coverage_limit': 'Header-box containment and safe split are verified, but header extrema do not prove every canopy return is observed.',
        'sources': [{'site_id': site_id, 'path': str(path), 'sha256': sha256(path)}
                    for path, site_id, _, _ in loaded],
        'site_rows': {site_id: {key: values[key] for key in args.config}
                      for _, site_id, _, values in loaded},
        'rows': rows,
        'matched_iou_note': 'Per-site medians cannot be pooled; this script does not report pooled median IoU.',
    })
    print(f'Compared {len(args.config)} full-density convex candidates on {len(loaded)} training sites')


if __name__ == '__main__':
    main()
