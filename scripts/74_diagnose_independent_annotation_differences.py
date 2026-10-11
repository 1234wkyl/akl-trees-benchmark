#!/usr/bin/env python3
"""Explain original rater differences without modifying matching, labels or freeze.

Group links are review leads at the existing 10%-of-smaller overlap convention,
not verified splits or corrections. Original F1/IoU/kappa remain unchanged.
"""
import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
from benchmark.common import ROOT, sha256, write_json


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['site_id'])
        writer.writeheader(); writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comparison', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('fresh output directory required')
    receipt = json.loads((args.comparison / 'receipt.json').read_text())
    if receipt['synthetic_only'] or receipt['status'] != 'original_agreement_computed_pending_adjudication':
        raise ValueError('successful original real comparison required')
    manifest = json.loads((args.comparison / 'SHA256_manifest.json').read_text())
    for name in ['original_comparison.gpkg', 'metrics.json', 'overlap_links.csv', 'geometry_audit.csv']:
        if sha256(args.comparison / name) != manifest[name]:
            raise ValueError('comparison artifact changed: ' + name)
    metrics = json.loads((args.comparison / 'metrics.json').read_text())
    policy = json.loads((args.comparison / 'agreement_policy.json').read_text())
    spec = importlib.util.spec_from_file_location('agreement74', ROOT / 'scripts/05_interrater_agreement.py')
    agreement = importlib.util.module_from_spec(spec); spec.loader.exec_module(agreement)
    a = gpd.read_file(args.comparison / 'original_comparison.gpkg', layer='original_XZ')
    b = gpd.read_file(args.comparison / 'original_comparison.gpkg', layer='original_teacher')
    with (args.comparison / 'overlap_links.csv').open() as f:
        links = list(csv.DictReader(f))
    with (args.comparison / 'geometry_audit.csv').open() as f:
        geometry_audit = list(csv.DictReader(f))
    unmatched_rows, groups, uncertain = [], [], []
    matched_b = set()
    for sid in policy['sites']:
        singles_a = a[(a.site_id == sid) & (a['class'] == 'tree')].reset_index(drop=True)
        singles_b = b[(b.site_id == sid) & (b['class'] == 'tree')].reset_index(drop=True)
        matches, _, _ = agreement.greedy_match(list(singles_a.geometry), list(singles_b.geometry),
            policy['iou_threshold'], list(singles_a.label_id), list(singles_b.label_id))
        matched_b.update(str(singles_b.iloc[j].label_id) for _, j, _ in matches)
        for row in singles_b.itertuples():
            if row.label_id in matched_b:
                continue
            relevant = [r for r in links if r['site_id'] == sid and r['id_b'] == row.label_id]
            group_ids = [r['id_a'] for r in relevant if r['class_a'] == 'tree_group']
            single_ids = [r['id_a'] for r in relevant if r['class_a'] == 'tree']
            negative_ids = [r['id_a'] for r in relevant if r['class_a'] == 'not_tree']
            category = ('overlap_original_tree_group' if group_ids else
                        'overlap_original_single_tree' if single_ids else
                        'overlap_original_not_tree' if negative_ids else
                        'no_recorded_original_overlap_link')
            unmatched_rows.append({'site_id': sid, 'TW_tree_id': row.label_id,
                'review_category': category, 'linked_XZ_group_ids': '|'.join(group_ids),
                'linked_XZ_single_ids': '|'.join(single_ids),
                'linked_XZ_not_tree_ids': '|'.join(negative_ids),
                'TW_crown_area_m2': row.geometry.area,
                'max_IoU_with_original_single': max((agreement.iou(row.geometry, g) for g in singles_a.geometry), default=0.)})
        for row in a[(a.site_id == sid) & (a['class'] == 'tree_group')].itertuples():
            relevant = [r for r in links if r['site_id'] == sid and r['id_a'] == row.label_id]
            teacher_trees = sorted({r['id_b'] for r in relevant if r['class_b'] == 'tree'})
            teacher_uncertain = sorted({r['id_b'] for r in relevant if r['class_b'] == 'uncertain'})
            groups.append({'site_id': sid, 'XZ_group_id': row.label_id,
                           'linked_TW_tree_count': len(teacher_trees),
                           'linked_TW_tree_ids': '|'.join(teacher_trees),
                           'linked_TW_uncertain_ids': '|'.join(teacher_uncertain),
                           'group_area_m2': row.geometry.area, 'decision': 'pending_review'})
        for row in b[(b.site_id == sid) & (b['class'] == 'uncertain')].itertuples():
            relevant = [r for r in links if r['site_id'] == sid and r['id_b'] == row.label_id]
            uncertain.append({'site_id': sid, 'TW_uncertain_id': row.label_id,
                'area_m2': row.geometry.area,
                'linked_XZ_tree_ids': '|'.join(r['id_a'] for r in relevant if r['class_a'] == 'tree'),
                'linked_XZ_group_ids': '|'.join(r['id_a'] for r in relevant if r['class_a'] == 'tree_group'),
                'linked_XZ_not_tree_ids': '|'.join(r['id_a'] for r in relevant if r['class_a'] == 'not_tree'),
                'decision': 'pending_review'})
    tree_result = metrics['individual_tree_view']
    if len(matched_b) != tree_result['matched'] or len(unmatched_rows) != tree_result['n_b']-tree_result['matched']:
        raise ValueError('diagnostic pairing does not reproduce original single-tree counts')
    confusion = metrics['class_confusion']
    # Per-pair list is available separately; counts have one column for this export.
    with (args.comparison / 'agreement_pairs.csv').open() as f:
        pair_rows = [r for r in csv.DictReader(f) if r['status'].startswith('matched')]
    same = sum(r['class_a'] == r['class_b'] for r in pair_rows)
    result = {'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_diagnose_original_agreement',
        'argv': sys.argv, 'status': 'ORIGINAL AGREEMENT DIAGNOSTICS; ADJUDICATION PENDING',
        'single_tree_metrics_unchanged': tree_result,
        'unmatched_TW_tree_review_categories': dict(Counter(r['review_category'] for r in unmatched_rows)),
        'original_groups': len(groups),
        'groups_linked_to_multiple_TW_trees': sum(r['linked_TW_tree_count'] >= 2 for r in groups),
        'TW_uncertain_regions': len(uncertain),
        'matched_class_agreement_count': same, 'matched_class_pair_count': len(pair_rows),
        'matched_class_observed_agreement': same/len(pair_rows) if pair_rows else None,
        'matched_class_kappa': metrics['class_kappa'],
        'matched_TW_class_counts': dict(Counter(r['class_b'] for r in pair_rows)),
        'kappa_note': 'All spatially matched TW objects are tree. With this constant marginal, kappa is uninformative as an overall quality assessment; report its value and restricted scope, not a whole-site reliability claim.',
        'geometry': {'raw_invalid_polygons': sum(r['raw_valid'] == 'False' for r in geometry_audit),
                     'outside_site_over_0_01m2': sum(float(r['outside_site_area_m2']) > .01 for r in geometry_audit),
                     'max_absolute_repair_area_difference_m2': max(abs(float(r['repair_area_difference_m2'])) for r in geometry_audit)},
        'overlap_link_rule': policy['overlap_diagnostic_threshold'],
        'overlap_link_definition': 'intersection / smaller polygon area; review lead, not accepted matching or true tree identity',
        'source_receipt_sha256': sha256(args.comparison / 'receipt.json'),
        'source_metrics_sha256': sha256(args.comparison / 'metrics.json'),
        'script_sha256': sha256(Path(__file__)), 'labels_modified': False,
        'reference_frozen': False, 'automatic_pass_declared': False}
    args.output.mkdir(parents=True)
    write_csv(args.output / 'unmatched_TW_tree_review.csv', unmatched_rows)
    write_csv(args.output / 'group_correspondence_review.csv', groups)
    write_csv(args.output / 'TW_uncertainty_review.csv', uncertain)
    write_json(args.output / 'diagnosis.json', result)
    (args.output / 'executed_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output / 'SHA256_manifest.json', {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps({k: result[k] for k in ['run_id', 'unmatched_TW_tree_review_categories',
        'groups_linked_to_multiple_TW_trees', 'matched_class_observed_agreement', 'geometry']}, indent=2))


if __name__ == '__main__':
    main()
