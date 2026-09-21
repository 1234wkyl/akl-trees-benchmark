#!/usr/bin/env python3
"""Describe unmatched training crowns without changing labels or scoring rules."""

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from benchmark.common import ROOT, sha256, site_definitions, write_json


def layer(path, name, site_id):
    frame = gpd.read_file(path, layer=name)
    if frame.crs is None or frame.crs.to_epsg() != 2193:
        raise ValueError(f'{name} must use EPSG:2193')
    return frame[frame.site_id == site_id].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site-id', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--point-support', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    site = site_definitions().get(args.site_id)
    if site is None or site['split'] != 'train':
        parser.error('only training sites are allowed')
    if args.output.exists():
        parser.error('choose a new output directory')
    evaluation = json.loads(args.evaluation.read_text())
    if evaluation.get('status') != 'DEVELOPMENT ONLY' or set(evaluation['by_site']) != {args.site_id}:
        raise ValueError('expected a single-site development evaluation')
    for key, path in [('reference', args.reference), ('predictions', args.predictions)]:
        if evaluation['sources'][key]['sha256'] != sha256(path):
            raise ValueError(f'{key} hash does not match the evaluation')
    reference = layer(args.reference, 'single_trees', args.site_id)
    predictions = layer(args.predictions, 'predictions', args.site_id)
    if reference.label_id.duplicated().any() or predictions.prediction_id.duplicated().any():
        raise ValueError('duplicate object identifiers')
    unmatched = set(evaluation['by_site'][args.site_id]['unmatched_reference_ids'])
    if unmatched != set(reference.label_id) - {pair['reference_id'] for pair in evaluation['by_site'][args.site_id]['matches']}:
        raise ValueError('unmatched reference IDs do not reconcile')
    with args.point_support.open(newline='') as handle:
        support = {row['label_id']: row for row in csv.DictReader(handle)}
    if not unmatched <= set(support):
        raise ValueError('point-support audit is missing unmatched references')

    rows = []
    for item in reference.itertuples():
        if item.label_id not in unmatched:
            continue
        centre = item.geometry.centroid
        ranked = sorted(
            ((centre.distance(pred.geometry.centroid), str(pred.prediction_id), pred.geometry)
             for pred in predictions.itertuples()),
            key=lambda value: (value[0], value[1]))
        near = [candidate for candidate in ranked if candidate[0] <= 2.0]
        nearest = ranked[0] if ranked else None
        geometry = nearest[2] if nearest else None
        intersection = item.geometry.intersection(geometry).area if geometry is not None else 0.0
        union = item.geometry.union(geometry).area if geometry is not None else 0.0
        record = support[item.label_id]
        rows.append({
            'site_id': args.site_id,
            'reference_id': item.label_id,
            'reference_area_m2': item.geometry.area,
            'inside_positive_instance_points': int(record['inside_positive_instance_points']),
            'buffered_positive_instance_points': int(record['buffered_positive_instance_points']),
            'nearest_prediction_id': nearest[1] if nearest else '',
            'nearest_centroid_distance_m': nearest[0] if nearest else '',
            'nearest_crown_iou': intersection / union if union else '',
            'candidate_centroids_within_2m': len(near),
            'diagnostic_category': ('no_prediction_polygon' if nearest is None else
                                    'no_centroid_within_2m' if not near else
                                    'within_2m_but_unmatched'),
        })
    rows.sort(key=lambda item: item['reference_id'])
    if len(rows) != evaluation['by_site'][args.site_id]['fn']:
        raise ValueError('false-negative count does not reconcile')
    args.output.mkdir(parents=True)
    with (args.output / 'misses.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [], lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)

    x0, y0, x1, y1 = site['bbox_epsg2193']
    chip_dir = ROOT / 'data/sites/chips' / args.site_id
    fig, axes = plt.subplots(max(len(rows), 1), 2, figsize=(10, 3.2 * max(len(rows), 1)), squeeze=False)
    by_id = reference.set_index('label_id')
    pred_by_id = predictions.set_index('prediction_id')
    for index, row in enumerate(rows):
        item = by_id.loc[row['reference_id']]
        centre = item.geometry.centroid
        radius = max(12.0, np.sqrt(item.geometry.area) * 1.5)
        for column, kind in enumerate(('rgb', 'chm')):
            ax = axes[index, column]
            ax.imshow(np.asarray(Image.open(chip_dir / f'{kind}.png')), extent=(x0, x1, y0, y1))
            gpd.GeoSeries([item.geometry], crs=reference.crs).boundary.plot(ax=ax, color='lime', linewidth=2)
            if row['nearest_prediction_id']:
                nearest_geom = pred_by_id.loc[row['nearest_prediction_id']].geometry
                gpd.GeoSeries([nearest_geom], crs=predictions.crs).boundary.plot(ax=ax, color='cyan', linewidth=1.7)
                radius = max(radius, centre.distance(nearest_geom.centroid) + 7.0)
            ax.set_xlim(max(x0, centre.x-radius), min(x1, centre.x+radius))
            ax.set_ylim(max(y0, centre.y-radius), min(y1, centre.y+radius))
            ax.set_aspect('equal'); ax.tick_params(labelsize=6)
            ax.set_title(f"{row['reference_id']} | {kind.upper()} | nearest {row['nearest_centroid_distance_m']:.2f} m",
                         fontsize=8)
    fig.suptitle(f'{args.site_id} training-only misses: green=reference, cyan=nearest candidate', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(args.output / 'misses_rgb_chm.png', dpi=180)
    plt.close(fig)
    write_json(args.output / 'provenance.json', {
        'status': 'TRAINING_ONLY_DEVELOPMENT_DIAGNOSTIC',
        'site_id': args.site_id,
        'reference_sha256': sha256(args.reference),
        'predictions_sha256': sha256(args.predictions),
        'evaluation_sha256': sha256(args.evaluation),
        'point_support_sha256': sha256(args.point_support),
        'rgb_sha256': sha256(chip_dir / 'rgb.png'),
        'chm_sha256': sha256(chip_dir / 'chm.png'),
        'csv_sha256': sha256(args.output / 'misses.csv'),
        'figure_sha256': sha256(args.output / 'misses_rgb_chm.png'),
        'unmatched_count': len(rows),
        'rule': 'nearest centroid is descriptive; the fixed evaluation was not changed',
    })
    print(f"{args.site_id}: {len(rows)} missed references; {sum(r['diagnostic_category']=='no_centroid_within_2m' for r in rows)} without a centroid within 2 m")


if __name__ == '__main__':
    main()
