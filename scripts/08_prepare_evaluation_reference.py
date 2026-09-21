#!/usr/bin/env python
"""Create draft evaluation layers without altering the canonical reference."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        ap.error('use a fresh output folder; no existing reference is overwritten')
    src = ROOT/'data/exports/reference_crowns.gpkg'
    definition = ROOT/'data/sites/reference_sites.json'
    policy = ROOT/'config/reference_review_policy.json'
    source_hash = sha(src)
    data = gpd.read_file(src, layer='reference_crowns')
    if data.crs is None or data.crs.to_epsg() != 2193:
        raise ValueError('reference must use EPSG:2193')
    if not data.geometry.is_valid.all():
        raise ValueError('repair exported geometry before preparing evaluation layers')
    if data.labeller.nunique() != 1 or set(data.labeller) != {'XZ'}:
        raise ValueError('canonical reference must contain XZ only, not independent-rater duplicates')
    sites = {s['site_id']: s for s in json.loads(definition.read_text())['sites']}
    if not set(data.site_id) <= set(sites):
        raise ValueError('unknown reference sites')
    if not set(data['class']) <= {'tree', 'tree_group', 'uncertain', 'not_tree'}:
        raise ValueError('unexpected label classes')
    clipped = data.copy()
    clipped['area_before_clip_m2'] = data.geometry.area
    clipped.geometry = [p.intersection(box(*sites[sid]['bbox_epsg2193'])) for sid, p in zip(data.site_id, data.geometry)]
    clipped['area_after_clip_m2'] = clipped.geometry.area
    clipped['removed_outside_area_m2'] = clipped.area_before_clip_m2-clipped.area_after_clip_m2
    if clipped.geometry.is_empty.any() or not clipped.geometry.is_valid.all():
        raise ValueError('empty/invalid geometry after clipping needs explicit review')
    out.mkdir(parents=True)
    target = out/'evaluation_reference_draft.gpkg'
    layers = {'single_trees':'tree', 'tree_groups':'tree_group', 'uncertainty_regions':'uncertain', 'hard_negatives':'not_tree'}
    counts = {}
    for layer, cls in layers.items():
        subset = clipped[clipped['class'] == cls].copy()
        subset.to_file(target, layer=layer, driver='GPKG')
        counts[layer] = len(subset)
    site_frame = gpd.GeoDataFrame(
        [{k:s[k] for k in ['site_id','stratum','split']} for s in sites.values()],
        geometry=[box(*s['bbox_epsg2193']) for s in sites.values()], crs=data.crs)
    site_frame.to_file(target, layer='site_bounds', driver='GPKG')
    clipped.drop(columns='geometry').to_csv(out/'object_clip_audit.csv', index=False)
    counts_split = clipped[clipped['class'] == 'tree'].groupby(['stratum','split']).size().unstack('split', fill_value=0)
    counts_split.to_csv(out/'single_tree_counts.csv')
    metadata = {
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'DRAFT PREPARATION; reference not frozen; no model scoring performed',
        'reference_sha256':source_hash, 'site_definitions_sha256':sha(definition),
        'review_policy_sha256':sha(policy), 'output_sha256':sha(target),
        'crs':'EPSG:2193', 'counts':counts,
        'geometry_rule':'intersection with original 100 m site box, no buffer',
        'clipped_objects_over_0_01m2':int((clipped.removed_outside_area_m2 > .01).sum()),
        'max_removed_outside_area_m2':float(clipped.removed_outside_area_m2.max()),
        'unresolved_rules':['group/uncertain unmatched-prediction ignore rule','false-positive precision operational definition','final detection matching rule'],
        'canonical_reference_unchanged':sha(src)==source_hash,
    }
    (out/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (out/'README.md').write_text('''# Draft evaluation reference layers

This is preparation, not a frozen reference or an evaluation result. The canonical XZ export is unchanged. Single trees, groups, uncertainty regions, hard negatives and site bounds are separate GeoPackage layers in EPSG:2193. All polygons are clipped to their original site box, with before/after areas retained in object_clip_audit.csv. Geometry has not been changed by visual adjudication.

The uncertainty and group layers describe regions requiring an explicit scoring policy; this script does not automatically ignore model predictions. Hard negatives are retained. Matching and false-positive precision require a fixed operational definition before model results. Recreate predictions with the same boundary convention when the evaluation harness is implemented.
''')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
