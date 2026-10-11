#!/usr/bin/env python3
"""Read-only training diagnosis of SAT crown overlap with cached OSM buildings.

No prediction filtering, new inference, held-out scoring or parameter selection.
Overlap is a spatial flag, not proof that the returns belong to a building.
"""
import argparse
import csv
import importlib.util
import json
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from benchmark.common import ROOT, sha256, site_definitions, write_json


def overlap_metrics(crown, building_union):
    area = crown.intersection(building_union).area
    return {'building_overlap_m2': area,
            'building_overlap_fraction': area / crown.area if crown.area else 0.,
            'centroid_in_building': bool(building_union.covers(crown.centroid))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--training-archive', type=Path, required=True)
    ap.add_argument('--building-cache', type=Path, required=True)
    ap.add_argument('--source', type=Path, default=ROOT.parent / 'Xi_Zhang_Benchmark/source_reference')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('fresh output directory required')
    start = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_training_building_overlap_audit'
    expected = 'bf73e299d46f9f240f16633d070c7a765877d3c1a826bf3450e65825557180ed'
    if sha256(args.training_archive) != expected:
        raise ValueError('archive differs from the recorded September training evidence')
    spec = importlib.util.spec_from_file_location('historical_adapter', ROOT / 'scripts/51_attempt_historical_crowns.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    source, provenance, source_hashes = adapter.source_module(args.source)
    files = sorted(args.building_cache.glob('osm_buildings_*.json'))
    if len(files) != 9:
        raise ValueError('all nine cached building responses are required')
    buildings = source.osm_payloads_to_polygons([json.loads(p.read_text()) for p in files])
    tree = STRtree(buildings)
    sites = {k: v for k, v in site_definitions().items() if v['split'] == 'train'}
    args.output.mkdir(parents=True)
    (args.output / 'inputs').mkdir()
    (args.output / 'executed_source.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    bindings = {}
    with tarfile.open(args.training_archive) as archive:
        members = archive.getmembers()
        for sid, site in sites.items():
            prefix = '/training_conversion_prefreeze249_v1/' + sid + '/fraction_1/convex_b100/'
            selected = {}
            for suffix in ['evaluation/evaluation.json', 'standardized/predictions.gpkg']:
                found = [m for m in members if m.isfile() and m.name.endswith(prefix + suffix)]
                if len(found) != 1:
                    raise ValueError('unique authoritative training artifact required: ' + sid)
                selected[suffix] = found[0]
            evaluation = json.load(archive.extractfile(selected['evaluation/evaluation.json']))
            if evaluation['split'] != 'train' or set(evaluation['by_site']) != {sid}:
                raise ValueError('only single training-site evaluation artifacts allowed')
            prediction_path = args.output / 'inputs' / (sid + '.gpkg')
            prediction_path.write_bytes(archive.extractfile(selected['standardized/predictions.gpkg']).read())
            if sha256(prediction_path) != evaluation['sources']['predictions']['sha256']:
                raise ValueError('prediction hash does not match the original evaluation')
            frame = gpd.read_file(prediction_path)
            if frame.crs is None or frame.crs.to_epsg() != 2193:
                raise ValueError('explicit EPSG:2193 required')
            if set(frame.site_id) != {sid} or not frame.geometry.is_valid.all():
                raise ValueError('valid site-specific prediction geometries required')
            region = box(*site['bbox_epsg2193'])
            local = [buildings[int(i)] for i in tree.query(region, predicate='intersects')]
            union = unary_union(local)
            score = evaluation['by_site'][sid]
            fp = set(score['false_positive_ids'])
            ignored = set(score['ignored_prediction_ids'])
            matched = {r['prediction_id'] for r in score['matches']}
            for row in frame.itertuples():
                ident = str(row.prediction_id)
                status = 'matched' if ident in matched else 'false_positive' if ident in fp else 'ignored' if ident in ignored else None
                if status is None:
                    raise ValueError('prediction absent from original scoring statuses')
                rows.append({'site_id': sid, 'prediction_id': ident, 'status': status,
                             'crown_area_m2': row.geometry.area, **overlap_metrics(row.geometry, union)})
            bindings[sid] = {'prediction_sha256': sha256(prediction_path),
                             'archive_prediction_member': selected['standardized/predictions.gpkg'].name,
                             'evaluation_reference_sha256': evaluation['sources']['reference']['sha256'],
                             'local_building_polygons': len(local)}
    counts = {}
    for status in ['matched', 'false_positive', 'ignored']:
        subset = [r for r in rows if r['status'] == status]
        counts[status] = {'total': len(subset),
                         'any_building_overlap': sum(r['building_overlap_m2'] > 0 for r in subset),
                         'at_least_half_area_over_building': sum(r['building_overlap_fraction'] >= .5 for r in subset),
                         'centroid_in_building': sum(r['centroid_in_building'] for r in subset)}
    if [counts[k]['total'] for k in ['matched', 'false_positive', 'ignored']] != [36, 203, 464]:
        raise ValueError('training artifacts do not reproduce the authoritative status counts')
    with (args.output / 'building_overlap.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {'run_id': run_id, 'argv': sys.argv, 'exit_code': 0,
               'elapsed_seconds': time.monotonic() - start, 'split': 'train',
               'archive_sha256': expected, 'source_commit': provenance['git_commit'],
               'source_hashes': source_hashes,
               'building_cache_hashes': {p.name: sha256(p) for p in files},
               'building_polygons': len(buildings), 'sites': bindings, 'counts': counts,
               'prediction_filter_applied': False, 'heldout_used': False,
               'inference_executed': False, 'building_epoch_verified': False,
               'note': 'Current OSM footprints; overlap may reflect overhanging crowns or XY hull buffer, not building returns. No improvement measured.'}
    write_json(args.output / 'audit_receipt.json', receipt)
    write_json(args.output / 'SHA256_manifest.json', {str(p.relative_to(args.output)): sha256(p)
               for p in sorted(args.output.rglob('*')) if p.is_file()})
    print(json.dumps({'run_id': run_id, 'counts': counts, 'elapsed_seconds': receipt['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
