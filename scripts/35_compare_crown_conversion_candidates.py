#!/usr/bin/env python
"""Compare explicit SAT crown conversions on one training site only."""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

from benchmark.common import ROOT, sha256, site_definitions, write_json


def parse_config(values):
    config_id, method, ratio_text, buffer_text = values
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', config_id):
        raise ValueError('config ID must contain only letters, digits, underscores or hyphens')
    if method not in {'convex_hull', 'concave_hull'}:
        raise ValueError(f'{config_id}: unsupported hull method')
    ratio = None if ratio_text == 'none' else float(ratio_text)
    buffer_m = float(buffer_text)
    if method == 'concave_hull' and (ratio is None or not np.isfinite(ratio) or not 0 <= ratio <= 1):
        raise ValueError(f'{config_id}: concave_hull requires a ratio in [0,1]')
    if method == 'convex_hull' and ratio is not None:
        raise ValueError(f'{config_id}: convex_hull ratio must be none')
    if not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError(f'{config_id}: buffer must be finite and nonnegative')
    return {'config_id': config_id, 'method': method, 'ratio': ratio, 'buffer_m': buffer_m}


def run(command, commands):
    command = [str(value) for value in command]
    commands.append(command)
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', nargs=3, action='append', required=True,
                    metavar=('FRACTION', 'PREDICTION', 'VERIFICATION_RECORD'))
    ap.add_argument('--config', nargs=4, action='append', required=True,
                    metavar=('ID', 'METHOD', 'RATIO_OR_NONE', 'BUFFER_M'))
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--site-id', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('use a fresh output folder')

    sites = site_definitions()
    if args.site_id not in sites or sites[args.site_id]['split'] != 'train':
        ap.error('--site-id must name one training site')
    if not args.reference.is_file():
        ap.error('--reference must exist')

    configs = [parse_config(values) for values in args.config]
    if len({item['config_id'] for item in configs}) != len(configs):
        ap.error('config IDs must be unique')
    conditions = []
    for fraction_text, prediction_text, audit_text in args.condition:
        fraction = float(fraction_text)
        prediction, audit = Path(prediction_text), Path(audit_text)
        if not np.isfinite(fraction) or not 0 < fraction <= 1 or not prediction.is_file() or not audit.is_file():
            ap.error(f'invalid condition: {fraction_text} {prediction} {audit}')
        verification_record = json.loads(audit.read_text())
        verification_flag = ('--merge-receipt' if verification_record.get('status') ==
                             'TRAINING-SITE MULTI-TILE MERGE COMPLETED' else '--coordinate-audit')
        conditions.append({'fraction': fraction, 'prediction': prediction, 'audit': audit,
                           'verification_flag': verification_flag})
    if len({item['fraction'] for item in conditions}) != len(conditions):
        ap.error('condition fractions must be unique')

    args.output.mkdir(parents=True)
    commands, rows = [], []
    python = sys.executable
    for condition in sorted(conditions, key=lambda item: item['fraction'], reverse=True):
        fraction_name = f"fraction_{condition['fraction']:g}"
        for config in configs:
            root = args.output / fraction_name / config['config_id']
            crowns, standard, evaluation = root / 'crowns', root / 'standardized', root / 'evaluation'
            convert = [python, ROOT/'scripts/20_sat_crown_candidates.py',
                       '--prediction', condition['prediction'],
                       condition['verification_flag'], condition['audit'],
                       '--tree-semantic-values', '1', '--hull-method', config['method'],
                       '--buffer-m', config['buffer_m'], '--output', crowns]
            if config['method'] == 'concave_hull':
                convert += ['--concave-ratio', config['ratio']]
            run(convert, commands)
            run([python, ROOT/'scripts/16_standardize_predictions.py',
                 '--input', crowns/'candidate_crowns.gpkg', '--layer', 'candidate_crowns',
                 '--id-column', 'prediction_id', '--split', 'train', '--site-ids', args.site_id,
                 '--output', standard], commands)
            run([python, ROOT/'scripts/12_evaluate.py', '--predictions', standard/'predictions.gpkg',
                 '--reference', args.reference, '--stage', 'development', '--site-ids', args.site_id,
                 '--output', evaluation], commands)

            all_crowns = gpd.read_file(crowns/'candidate_crowns.gpkg', layer='candidate_crowns')
            site_crowns = gpd.read_file(standard/'predictions.gpkg', layer='predictions')
            result = json.loads((evaluation/'evaluation.json').read_text())['summary']
            rows.append({
                'fraction': condition['fraction'], 'config_id': config['config_id'],
                'method': config['method'], 'concave_ratio': config['ratio'],
                'buffer_m': config['buffer_m'], 'tile_crowns': len(all_crowns),
                'tile_median_area_m2': float(all_crowns.geometry.area.median()) if len(all_crowns) else None,
                'site_predictions': len(site_crowns),
                'site_median_area_m2': float(site_crowns.geometry.area.median()) if len(site_crowns) else None,
                **{key: result[key] for key in ('tp','fp','fn','precision','recall','f1','median_matched_iou','ignored_predictions')},
                'evaluation_sha256': sha256(evaluation/'evaluation.json'),
            })

    fields = list(rows[0])
    with (args.output/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        'status': 'DEVELOPMENT_ONLY_TRAINING_SITE_CONVERSION_COMPARISON',
        'site_id': args.site_id,
        'site_split': 'train',
        'warning': 'Do not use this one-site pre-freeze diagnostic as a held-out result or density-selection result.',
        'reference': str(args.reference), 'reference_sha256': sha256(args.reference),
        'conditions': [{**item, 'prediction': str(item['prediction']), 'audit': str(item['audit']),
                        'prediction_sha256': sha256(item['prediction']), 'audit_sha256': sha256(item['audit'])}
                       for item in conditions],
        'configs': configs, 'commands': commands, 'rows': rows,
    }
    write_json(args.output/'summary.json', manifest)
    print(f'Completed {len(rows)} development-only conversion rows for {args.site_id}')


if __name__ == '__main__':
    main()
