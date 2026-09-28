#!/usr/bin/env python3
"""Aggregate full-density crown-conversion candidates over training sites.

Every input grid must be bound to a completed training-site merge receipt.
The reference may be pre-freeze, so this command selects only a provisional
conversion candidate and never creates a held-out benchmark result.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

from benchmark.common import ROOT, sha256, site_definitions, write_json


def resolve(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def load_grid(path, coverage, expected_reference_hash):
    grid = json.loads(path.read_text())
    site_id = grid.get('site_id')
    sites = site_definitions()
    if (grid.get('status') != 'DEVELOPMENT_ONLY_TRAINING_SITE_CONVERSION_COMPARISON'
            or site_id not in sites or sites[site_id]['split'] != 'train'
            or grid.get('site_split') != 'train'):
        raise ValueError(f'{path}: require a training-only conversion grid')
    if expected_reference_hash and grid.get('reference_sha256') != expected_reference_hash:
        raise ValueError(f'{path}: reference hash differs between grids')
    conditions = [row for row in grid.get('conditions', []) if float(row['fraction']) == 1.0]
    if len(conditions) != 1 or conditions[0].get('verification_flag') != '--merge-receipt':
        raise ValueError(f'{path}: require one full-density merge-receipt condition')
    condition = conditions[0]
    prediction, receipt_path = resolve(condition['prediction']), resolve(condition['audit'])
    if (not prediction.is_file() or not receipt_path.is_file()
            or sha256(prediction) != condition.get('prediction_sha256')
            or sha256(receipt_path) != condition.get('audit_sha256')):
        raise ValueError(f'{path}: prediction or merge-receipt hash mismatch')
    receipt = json.loads(receipt_path.read_text())
    plan_site = next((row for row in coverage['sites'] if row['site_id'] == site_id), None)
    if (receipt.get('status') != 'TRAINING-SITE MULTI-TILE MERGE COMPLETED'
            or receipt.get('site_id') != site_id or receipt.get('split') != 'train'
            or receipt.get('output_laz_sha256') != sha256(prediction)
            or receipt.get('benchmark_scores_created') is not False
            or plan_site is None
            or set(receipt.get('expected_source_tiles', [])) != set(plan_site['required_tiles'])):
        raise ValueError(f'{path}: invalid or incomplete training merge receipt')

    config_meta = {row['config_id']: row for row in grid.get('configs', [])}
    rows = {}
    for row in grid.get('rows', []):
        if float(row['fraction']) != 1.0:
            continue
        config_id = row['config_id']
        if config_id in rows or config_id not in config_meta:
            raise ValueError(f'{path}: duplicate or unknown configuration')
        evaluation = path.parent / 'fraction_1' / config_id / 'evaluation' / 'evaluation.json'
        if not evaluation.is_file() or sha256(evaluation) != row.get('evaluation_sha256'):
            raise ValueError(f'{path}: {config_id} evaluation hash mismatch')
        result = json.loads(evaluation.read_text())
        if result.get('status') != 'DEVELOPMENT ONLY' or result.get('split') != 'train':
            raise ValueError(f'{evaluation}: not a training development result')
        by_site = result.get('by_site', {})
        if set(by_site) != {site_id}:
            raise ValueError(f'{evaluation}: evaluation contains another site')
        detail = by_site[site_id]
        summary = result['summary']
        for key in ('tp', 'fp', 'fn', 'ignored_predictions'):
            if int(row[key]) != int(summary[key]):
                raise ValueError(f'{evaluation}: grid/evaluation count mismatch')
        rows[config_id] = {
            'site_id': site_id, 'config_id': config_id,
            'method': row['method'], 'concave_ratio': row['concave_ratio'],
            'buffer_m': float(row['buffer_m']), 'site_predictions': int(row['site_predictions']),
            'tp': int(row['tp']), 'fp': int(row['fp']), 'fn': int(row['fn']),
            'ignored_predictions': int(row['ignored_predictions']),
            'precision': row['precision'], 'recall': row['recall'], 'f1': row['f1'],
            'median_matched_iou': row['median_matched_iou'],
            'ious': [float(value) for value in detail.get('ious', [])],
            'over_segmented_references': int(summary['over_segmented_references']),
            'under_segmented_predictions': int(summary['under_segmented_predictions']),
            'evaluation_sha256': sha256(evaluation),
        }
    return {
        'site_id': site_id, 'reference_sha256': grid['reference_sha256'],
        'grid_path': path, 'grid_sha256': sha256(path), 'rows': rows,
        'coverage_fraction': float(receipt['header_bbox_site_coverage_fraction']),
        'merge_receipt': receipt_path, 'merge_receipt_sha256': sha256(receipt_path),
        'prediction_sha256': sha256(prediction),
    }


def safe_mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=Path, action='append', required=True)
    ap.add_argument('--coverage-plan', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if len(args.grid) < 2 or args.output.exists():
        ap.error('require at least two grids and a fresh output directory')
    coverage = json.loads(args.coverage_plan.read_text())
    if coverage.get('site_definitions_sha256') != sha256(ROOT/'data/sites/reference_sites.json'):
        raise ValueError('coverage plan does not match site definitions')
    loaded, reference_hash = [], None
    for path in args.grid:
        item = load_grid(path, coverage, reference_hash)
        reference_hash = item['reference_sha256']
        loaded.append(item)
    site_ids = [item['site_id'] for item in loaded]
    expected_sites = sorted(sid for sid, site in site_definitions().items() if site['split'] == 'train')
    if sorted(site_ids) != expected_sites or len(set(site_ids)) != len(site_ids):
        raise ValueError(f'require all training sites exactly once: {expected_sites}')
    config_sets = [set(item['rows']) for item in loaded]
    if any(values != config_sets[0] for values in config_sets[1:]):
        raise ValueError('configuration sets differ between sites')

    aggregates, site_rows = [], []
    for config_id in sorted(config_sets[0]):
        values = [item['rows'][config_id] for item in loaded]
        signatures = {(row['method'], row['concave_ratio'], row['buffer_m']) for row in values}
        if len(signatures) != 1:
            raise ValueError(f'{config_id}: configuration changes between sites')
        tp, fp, fn = (sum(row[key] for row in values) for key in ('tp','fp','fn'))
        denominator = 2*tp + fp + fn
        ious = [value for row in values for value in row['ious']]
        method, ratio, buffer_m = signatures.pop()
        aggregate = {
            'config_id': config_id, 'method': method, 'concave_ratio': ratio,
            'buffer_m': buffer_m, 'sites': len(values), 'reference_singles': tp+fn,
            'tp': tp, 'fp': fp, 'fn': fn,
            'ignored_predictions': sum(row['ignored_predictions'] for row in values),
            'micro_precision': tp/(tp+fp) if tp+fp else None,
            'micro_recall': tp/(tp+fn) if tp+fn else None,
            'micro_f1': 2*tp/denominator if denominator else None,
            'macro_f1': safe_mean([row['f1'] for row in values]),
            'worst_site_recall': min(row['recall'] for row in values if row['recall'] is not None),
            'zero_tp_sites': sum(row['tp'] == 0 for row in values),
            'pooled_median_matched_iou': statistics.median(ious) if ious else None,
            'over_segmented_references': sum(row['over_segmented_references'] for row in values),
            'under_segmented_predictions': sum(row['under_segmented_predictions'] for row in values),
        }
        aggregates.append(aggregate)
        site_rows.extend({key: value for key, value in row.items() if key != 'ious'} for row in values)

    ranked = sorted(aggregates, key=lambda row: (
        -(row['micro_f1'] if row['micro_f1'] is not None else -1),
        -(row['pooled_median_matched_iou'] if row['pooled_median_matched_iou'] is not None else -1),
        row['fp'], row['config_id']))
    selected = ranked[0]
    args.output.mkdir(parents=True)
    for name, rows in [('aggregate.csv', aggregates), ('by_site.csv', site_rows)]:
        with (args.output/name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    write_json(args.output/'summary.json', {
        'status': 'PROVISIONAL TRAINING-ONLY PRE-FREEZE CONVERSION SELECTION',
        'warning': ('The reference is not frozen and header coverage is incomplete for some sites. '
                    'This is not validation/test accuracy and does not lock a benchmark model.'),
        'selection_rule': ('maximum training micro-F1; ties by pooled median matched IoU, '
                           'then fewer false positives, then configuration ID'),
        'selected_config_id': selected['config_id'], 'selected_row': selected,
        'site_ids': site_ids, 'reference_sha256': reference_hash,
        'coverage_plan_sha256': sha256(args.coverage_plan),
        'minimum_header_bbox_coverage_fraction': min(item['coverage_fraction'] for item in loaded),
        'sources': [{'site_id': item['site_id'], 'grid': str(item['grid_path']),
                     'grid_sha256': item['grid_sha256'], 'merge_receipt': str(item['merge_receipt']),
                     'merge_receipt_sha256': item['merge_receipt_sha256'],
                     'prediction_sha256': item['prediction_sha256'],
                     'header_bbox_coverage_fraction': item['coverage_fraction']} for item in loaded],
        'rows': aggregates,
    })
    print(json.dumps({'selected_config_id': selected['config_id'],
                      'selected_row': selected, 'sites': len(site_ids)}, indent=2))


if __name__ == '__main__':
    main()
