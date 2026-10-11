#!/usr/bin/env python3
"""Build a hash-bound derived scoring candidate from the current browser JSON.

This repairs invalid polygon topology with buffer(0), clips every object to its
fixed 100 m site, and records every change. It never edits the source JSON and
does not grant reference freeze.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon, box

from benchmark.common import ROOT, sha256, site_definitions, write_json

CLASSES = ('tree', 'tree_group', 'not_tree', 'uncertain')
LAYERS = {'tree': 'single_trees', 'tree_group': 'tree_groups',
          'not_tree': 'hard_negatives', 'uncertain': 'uncertainty_regions'}
DEFAULT_LABELS = (ROOT / 'docs/group_refinement_2026-09-18/'
                  'export_20260922T060150894212Z/'
                  'crown_labels_XZ_2026-09-22-2.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-json', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh output directory')
    source_hash = sha256(args.labels_json)
    payload = json.loads(args.labels_json.read_text())
    if payload.get('schema_version') != 'akl-trees-crown-labels/v1' or payload.get('crs') != 'EPSG:2193':
        raise ValueError('expected akl-trees-crown-labels/v1 in EPSG:2193')
    sites = site_definitions()
    records, geometries, audit, seen = [], [], [], set()
    for site_id, site_record in payload.get('sites', {}).items():
        labels = site_record.get('labels', [])
        if site_id not in sites:
            if site_id == 'undefined' and not labels:
                continue
            raise ValueError(f'unknown site: {site_id}')
        site = sites[site_id]
        site_box = box(*site['bbox_epsg2193'])
        for label in labels:
            label_id, label_class = label.get('id'), label.get('cls')
            if not label_id or label_id in seen:
                raise ValueError(f'missing or duplicate label ID: {label_id}')
            if label_class not in CLASSES:
                raise ValueError(f'unknown class: {label_class}')
            seen.add(label_id)
            # Partial group extraction can leave interior rings. The coordinate
            # arrays remain authoritative when a browser export is edited.
            raw = Polygon(label['world'], holes=label.get('holes_world', []))
            repaired = raw if raw.is_valid else raw.buffer(0)
            if repaired.is_empty or not repaired.is_valid or repaired.geom_type not in {'Polygon', 'MultiPolygon'}:
                raise ValueError(f'geometry cannot be repaired: {label_id}')
            clipped = repaired.intersection(site_box)
            if clipped.is_empty or not clipped.is_valid or clipped.area <= 0 or clipped.geom_type not in {'Polygon', 'MultiPolygon'}:
                raise ValueError(f'geometry invalid/empty after site clipping: {label_id}')
            records.append({'label_id': label_id, 'site_id': site_id, 'class': label_class,
                            'labeller': label.get('labeller') or payload.get('labeller'),
                            'split': site['split'], 'stratum': site['stratum']})
            geometries.append(clipped)
            audit.append({'label_id': label_id, 'site_id': site_id, 'class': label_class,
                          'raw_valid': raw.is_valid, 'topology_repaired': not raw.is_valid,
                          'raw_area_m2': raw.area, 'repaired_area_m2': repaired.area,
                          'repair_area_difference_m2': repaired.area - raw.area,
                          'outside_site_area_m2': repaired.difference(site_box).area,
                          'scoring_area_m2': clipped.area})
    frame = gpd.GeoDataFrame(records, geometry=geometries, crs='EPSG:2193')
    if len(frame) != len(seen) or frame.empty:
        raise ValueError('reference object count mismatch')
    args.output.mkdir(parents=True)
    target = args.output / 'prefreeze_evaluation_reference.gpkg'
    frame.to_file(target, layer='all_objects', driver='GPKG')
    for label_class, layer in LAYERS.items():
        frame[frame['class'] == label_class].to_file(target, layer=layer, driver='GPKG')
    bounds = gpd.GeoDataFrame(
        [{'site_id': site_id, 'split': site['split'], 'stratum': site['stratum']}
         for site_id, site in sites.items()],
        geometry=[box(*site['bbox_epsg2193']) for site in sites.values()], crs='EPSG:2193')
    bounds.to_file(target, layer='site_bounds', driver='GPKG')
    with (args.output / 'geometry_audit.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=audit[0].keys())
        writer.writeheader(); writer.writerows(audit)
    count_rows = []
    for label_class in CLASSES:
        for stratum in dict.fromkeys(site['stratum'] for site in sites.values()):
            for split in ('train', 'val', 'test'):
                count_rows.append({'class': label_class, 'stratum': stratum, 'split': split,
                                   'count': int(((frame['class'] == label_class) &
                                                 (frame.stratum == stratum) &
                                                 (frame.split == split)).sum())})
    with (args.output / 'counts_by_class_split_stratum.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=count_rows[0].keys())
        writer.writeheader(); writer.writerows(count_rows)
    repaired = [row for row in audit if row['topology_repaired']]
    outside = [row for row in audit if row['outside_site_area_m2'] > .01]
    provenance = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PRE-FREEZE DERIVED CANDIDATE; NOT A FROZEN REFERENCE',
        'source_json': str(args.labels_json.resolve()),
        'source_json_sha256': source_hash,
        'source_unchanged': sha256(args.labels_json) == source_hash,
        'site_definitions_sha256': sha256(ROOT / 'data/sites/reference_sites.json'),
        'output_gpkg_sha256': sha256(target),
        'crs': 'EPSG:2193', 'objects': len(frame),
        'class_counts': {label_class: int((frame['class'] == label_class).sum()) for label_class in CLASSES},
        'topology_repaired_count': len(repaired),
        'topology_repaired_ids': [row['label_id'] for row in repaired],
        'outside_site_over_0_01m2_count': len(outside),
        'outside_site_over_0_01m2_ids': [row['label_id'] for row in outside],
        'max_abs_repair_area_difference_m2': max(abs(row['repair_area_difference_m2']) for row in audit),
        'max_outside_site_area_m2': max(row['outside_site_area_m2'] for row in audit),
        'geometry_rule': 'buffer(0) only for invalid topology, then intersection with fixed site box',
        'reference_frozen': False,
        'independent_S016_S020_adjudicated': False,
        'permitted_use': 'training-only diagnostics and pre-freeze QA; no held-out benchmark scoring',
    }
    write_json(args.output / 'provenance.json', provenance)
    (args.output / 'README.md').write_text(
        '# Pre-freeze derived scoring candidate\n\n'
        'This directory is derived from the source JSON named in `provenance.json`. '
        'It repairs topology and clips geometry for software compatibility without changing biological labels. '
        'It is not frozen and must not be used for locked validation/test scoring.\n')
    manifest = {str(path.relative_to(args.output)): sha256(path)
                for path in sorted(args.output.iterdir()) if path.is_file()}
    write_json(args.output / 'SHA256_manifest.json', manifest)
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
