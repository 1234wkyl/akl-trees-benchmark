#!/usr/bin/env python3
"""Compare original and current reference counts by class, split and stratum."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('tree', 'tree_group', 'not_tree', 'uncertain')
SPLITS = ('train', 'val', 'test')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_counts(path, sites):
    data = json.loads(path.read_text())
    if data.get('schema_version') != 'akl-trees-crown-labels/v1' or data.get('crs') != 'EPSG:2193':
        raise ValueError(f'unsupported schema or CRS: {path}')
    counts, seen = Counter(), set()
    for site_id, record in data.get('sites', {}).items():
        if site_id not in sites:
            if site_id == 'undefined' and not record.get('labels'):
                continue
            raise ValueError(f'unknown site: {site_id}')
        for label in record.get('labels', []):
            label_id, label_class = label.get('id'), label.get('cls')
            if not label_id or label_id in seen:
                raise ValueError(f'missing or duplicate label ID: {label_id}')
            if label_class not in CLASSES:
                raise ValueError(f'unknown class: {label_class}')
            seen.add(label_id)
            site = sites[site_id]
            counts[(label_class, site['stratum'], site['split'])] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, default=ROOT / 'data/labels/crown_labels_XZ_2026-08-13.json')
    parser.add_argument('--current', type=Path, default=ROOT / 'docs/group_refinement_2026-09-18/export_20260922T060150894212Z/crown_labels_XZ_2026-09-22-2.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh output directory')
    definitions_path = ROOT / 'data/sites/reference_sites.json'
    definitions = json.loads(definitions_path.read_text())
    sites = {site['site_id']: site for site in definitions['sites']}
    strata = list(dict.fromkeys(site['stratum'] for site in definitions['sites']))
    original = load_counts(args.original, sites)
    current = load_counts(args.current, sites)
    rows = []
    for label_class in CLASSES:
        for stratum in strata:
            for split in SPLITS:
                rows.append({'class': label_class, 'stratum': stratum, 'split': split,
                             'original': original[label_class, stratum, split],
                             'current': current[label_class, stratum, split]})
    args.output.mkdir(parents=True)
    with (args.output / 'reference_composition_by_split_stratum.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    summary = {
        'status': 'pre-freeze composition comparison',
        'original': {'path': str(args.original), 'sha256': sha256(args.original)},
        'current': {'path': str(args.current), 'sha256': sha256(args.current)},
        'site_definitions': {'path': str(definitions_path), 'sha256': sha256(definitions_path)},
        'class_totals': {
            label_class: {
                version: sum(source[label_class, stratum, split] for stratum in strata for split in SPLITS)
                for version, source in (('original', original), ('current', current))}
            for label_class in CLASSES},
        'rows': rows,
    }
    (args.output / 'reference_composition_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'class_totals': summary['class_totals']}))


if __name__ == '__main__':
    main()
