#!/usr/bin/env python
"""Inspect scored false positives against selective hard negatives on a training site."""
import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd

from benchmark.common import sha256, site_definitions, write_json


def site_layer(path, layer, site_id):
    frame = gpd.read_file(path, layer=layer)
    if frame.crs is None or frame.crs.to_epsg() != 2193:
        raise ValueError(f'{layer} CRS must be EPSG:2193')
    return frame[frame.site_id == site_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site-id', required=True)
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--evaluation', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sites = site_definitions()
    if args.site_id not in sites or sites[args.site_id]['split'] != 'train':
        ap.error('false-positive diagnosis is restricted to training sites')
    if args.output.exists():
        ap.error('use a fresh output folder')
    result = json.loads(args.evaluation.read_text())
    if result.get('status') != 'DEVELOPMENT ONLY' or set(result['by_site']) != {args.site_id}:
        raise ValueError('require a single training-site development evaluation')
    if result['sources']['reference']['sha256'] != sha256(args.reference) or result['sources']['predictions']['sha256'] != sha256(args.predictions):
        raise ValueError('evaluation does not match supplied geometry hashes')
    pred = site_layer(args.predictions,'predictions',args.site_id)
    single = site_layer(args.reference,'single_trees',args.site_id)
    groups = site_layer(args.reference,'tree_groups',args.site_id)
    negatives = site_layer(args.reference,'hard_negatives',args.site_id)
    false_ids = set(result['by_site'][args.site_id]['false_positive_ids'])
    frame = pred[pred.prediction_id.astype(str).isin(false_ids)]
    if len(frame) != len(false_ids):
        raise ValueError('scored false-positive IDs missing or duplicated')
    rows = []
    for record in frame.itertuples():
        crown = record.geometry
        hard_hits = negatives[negatives.intersects(crown)]
        rows.append({
            'site_id': args.site_id,
            'prediction_id': str(record.prediction_id),
            'area_m2': crown.area,
            'hard_negative_overlap_count':len(hard_hits),
            'hard_negative_ids':';'.join(sorted(map(str, hard_hits.label_id))),
            'hard_negative_overlap_m2':sum(crown.intersection(g).area for g in hard_hits.geometry),
            'nearest_single_centroid_m':min((crown.centroid.distance(g.centroid) for g in single.geometry),default=None),
            'nearest_group_boundary_m':min((crown.centroid.distance(g) for g in groups.geometry),default=None),
        })
    rows.sort(key=lambda row:row['prediction_id'])
    args.output.mkdir(parents=True)
    with (args.output/'scored_false_positives.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else [
            'site_id','prediction_id','area_m2','hard_negative_overlap_count','hard_negative_ids',
            'hard_negative_overlap_m2','nearest_single_centroid_m','nearest_group_boundary_m'],
            lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    write_json(args.output/'summary.json',{
        'status':'DEVELOPMENT ONLY; no exhaustive negative audit',
        'site_id':args.site_id,'false_positive_count':len(rows),
        'false_positives_overlapping_explicit_hard_negatives':sum(bool(r['hard_negative_overlap_count']) for r in rows),
        'explicit_hard_negative_count':len(negatives),
        'reference_sha256':sha256(args.reference),
        'predictions_sha256':sha256(args.predictions),
        'evaluation_sha256':sha256(args.evaluation),
        'csv_sha256':sha256(args.output/'scored_false_positives.csv'),
        'warning':'not_tree polygons were selectively annotated and cannot establish exhaustive TN, specificity, or a separate 0.95 audit.'})
    print(f'Analysed {len(rows)} scored FPs; {sum(bool(r["hard_negative_overlap_count"]) for r in rows)} overlap explicit hard negatives')


if __name__=='__main__':
    main()
