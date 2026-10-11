#!/usr/bin/env python3
"""Separate exploratory training-only centroid-building filter; no GPU or held-out scoring."""
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from benchmark.common import ROOT, sha256, site_definitions, write_json
from benchmark.evaluation import aggregate, paired_site_bootstrap, score_site


def removal_ids(rows):
    """Apply one label-independent spatial rule, without inspecting scoring status."""
    return {str(r['prediction_id']) for r in rows
            if r['centroid_in_building'] in (True, 'True')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--policy', type=Path, default=ROOT / 'config/evaluation_draft.json')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('fresh output directory required')
    start = time.monotonic()
    audit = json.loads((args.audit / 'audit_receipt.json').read_text())
    manifest = json.loads((args.audit / 'SHA256_manifest.json').read_text())
    for name in ['audit_receipt.json', 'building_overlap.csv']:
        if sha256(args.audit / name) != manifest[name]:
            raise ValueError('audit artifact hash changed: ' + name)
    sites = {sid: s for sid, s in site_definitions().items() if s['split'] == 'train'}
    if audit['split'] != 'train' or set(audit['sites']) != set(sites):
        raise ValueError('all eight training sites required')
    if any(r['evaluation_reference_sha256'] != sha256(args.reference)
           for r in audit['sites'].values()):
        raise ValueError('use the exact original training reference')
    policy = json.loads(args.policy.read_text())
    if (policy['match_tolerance_m'] != 2 or policy['minimum_match_iou'] != 0
            or policy['prediction_ignore_rule'] != 'unmatched_centroid_in_group_or_uncertain'
            or policy['bootstrap_iterations'] != 2000 or policy['random_seed'] != 42):
        raise ValueError('fixed scoring policy required')
    refs = {}
    for layer in ['single_trees', 'tree_groups', 'uncertainty_regions']:
        frame = gpd.read_file(args.reference, layer=layer)
        if frame.crs.to_epsg() != 2193 or not frame.geometry.is_valid.all():
            raise ValueError('valid EPSG:2193 reference required')
        refs[layer] = frame[frame.site_id.isin(sites)]
    with (args.audit / 'building_overlap.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    raw_scores, filtered_scores, kept_frames, removed = {}, {}, [], []
    for sid, site in sites.items():
        path = args.audit / 'inputs' / (sid + '.gpkg')
        if sha256(path) != audit['sites'][sid]['prediction_sha256']:
            raise ValueError('original prediction hash changed')
        predictions = gpd.read_file(path)
        if set(predictions.site_id) != {sid} or predictions.crs.to_epsg() != 2193:
            raise ValueError('training site/CRS mismatch')
        local_rows = [r for r in rows if r['site_id'] == sid]
        if set(predictions.prediction_id.astype(str)) != {r['prediction_id'] for r in local_rows}:
            raise ValueError('audit prediction IDs differ')
        drop = removal_ids(local_rows)
        kept = predictions[~predictions.prediction_id.astype(str).isin(drop)].copy()
        kept_frames.append(kept)
        removed.extend(r for r in local_rows if r['prediction_id'] in drop)
        region = box(*site['bbox_epsg2193'])
        local = {layer: [(str(r.label_id), r.geometry.intersection(region))
                        for r in frame[frame.site_id == sid].itertuples()]
                 for layer, frame in refs.items()}
        def score(frame):
            return score_site(local['single_trees'],
                [(str(r.prediction_id), r.geometry.intersection(region)) for r in frame.itertuples()],
                [g for _, g in local['tree_groups']], [g for _, g in local['uncertainty_regions']],
                tolerance=2., minimum_iou=0.,
                ignore_rule='unmatched_centroid_in_group_or_uncertain')
        raw_scores[sid], filtered_scores[sid] = score(predictions), score(kept)
    original = aggregate(list(raw_scores.values()))
    if [original[k] for k in ['tp', 'fp', 'fn']] != [36, 203, 46]:
        raise ValueError('original training scoring was not reproduced')
    import pandas as pd
    args.output.mkdir(parents=True)
    target = args.output / 'training_predictions_building_centroid_filtered.gpkg'
    gpd.GeoDataFrame(pd.concat(kept_frames, ignore_index=True), crs='EPSG:2193').to_file(
        target, layer='predictions', driver='GPKG')
    with (args.output / 'removed_predictions.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(removed)
    result = {'status': 'EXPLORATORY TRAINING-ONLY BUILDING CONTEXT VARIANT',
        'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_training_building_filter',
        'raw_summary': original, 'filtered_summary': aggregate(list(filtered_scores.values())),
        'raw_by_site': raw_scores, 'filtered_by_site': filtered_scores,
        'paired_training_bootstrap': paired_site_bootstrap(filtered_scores, raw_scores, 2000, 42),
        'removed_by_original_status': {k: sum(r['status'] == k for r in removed)
                                       for k in ['matched', 'false_positive', 'ignored']},
        'rule': 'remove crown iff its centroid is covered by cached building footprint',
        'prediction_geometry_changed': False, 'heldout_used': False,
        'gpu_inference_executed': False, 'reference_frozen': False,
        'original_benchmark_replaced': False, 'elapsed_seconds': time.monotonic() - start,
        'sources': {'audit_receipt': sha256(args.audit / 'audit_receipt.json'),
                    'reference': sha256(args.reference), 'policy': sha256(args.policy),
                    'script': sha256(Path(__file__)), 'filtered_predictions': sha256(target)},
        'limitations': ['Current building footprints are not certified historical inputs',
                        'A centroid flag does not prove that a return is roof rather than canopy',
                        'Training resubstitution does not demonstrate generalisation',
                        'Test results were viewed before this exploratory variant was proposed']}
    write_json(args.output / 'result.json', result)
    (args.output / 'executed_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output / 'SHA256_manifest.json', {str(p.relative_to(args.output)): sha256(p)
        for p in args.output.rglob('*') if p.is_file()})
    print(json.dumps({k: result[k] for k in ['run_id', 'raw_summary', 'filtered_summary',
                                          'removed_by_original_status', 'elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
