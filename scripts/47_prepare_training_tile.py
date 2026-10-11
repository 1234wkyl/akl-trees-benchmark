#!/usr/bin/env python3
"""Stage one named LAZ; held-out inputs require an explicit inference-only scenario."""
import argparse
import json
import shutil
from pathlib import Path

import laspy
import numpy as np

from benchmark.common import ROOT, sha256, write_json
from benchmark.inference_scenario import validate_inference_scenario


DEFAULT_PLAN = ROOT / 'docs/project_status_2026-09-18/tile_coverage_preflight/coverage_plan.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', required=True)
    parser.add_argument('--exclude-classes', nargs='*', type=int, default=[7, 18])
    parser.add_argument('--coverage-plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--source-dir', type=Path, default=ROOT.parent / 'Xi_Zhang_Benchmark/laz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inference-scenario', type=Path,
                        help='explicit assumption-only prediction protocol; does not permit scoring or tuning')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh output directory')
    plan = json.loads(args.coverage_plan.read_text())
    scenario = validate_inference_scenario(args.inference_scenario) if args.inference_scenario else None
    tiles = {row['tile']: row for row in plan['tiles']}
    if args.tile not in tiles:
        raise ValueError('tile is absent from the verified coverage plan')
    record = tiles[args.tile]
    held_out = record.get('held_out_intersecting_sites', [])
    training_sites = sorted(set(record.get('intersecting_sites', [])) - set(held_out))
    if not scenario and (held_out or not training_sites or record.get('safe_full_tile_for_development') is not True):
        raise ValueError(f'tile is not safe for training-only development: held_out={held_out}')
    source = args.source_dir / args.tile
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = sha256(source)
    if source_hash != record['sha256']:
        raise ValueError('source hash differs from the verified coverage plan')
    cloud = laspy.read(source)
    crs = cloud.header.parse_crs()
    if crs is None or crs.to_epsg() != 2193:
        raise ValueError('explicit EPSG:2193 input required')
    actual = np.asarray([cloud.x.min(), cloud.y.min(), cloud.x.max(), cloud.y.max()])
    expected = np.asarray(record['bbox_epsg2193'], dtype=float)
    if np.max(np.abs(actual - expected)) > .02:
        raise ValueError('input extent differs from the coverage plan')
    original_count = len(cloud.points)
    keep = ~np.isin(np.asarray(cloud.classification), args.exclude_classes)
    if not keep.any():
        raise ValueError('filter removed every point')
    cloud.points = cloud.points[keep]
    args.output.mkdir(parents=True)
    raw_dir = args.output / 'raw_source'
    input_dir = args.output / 'input_laz'
    raw_dir.mkdir(); input_dir.mkdir()
    raw_copy = raw_dir / source.name
    shutil.copy2(source, raw_copy)
    if sha256(source) != sha256(raw_copy):
        raise ValueError('raw copy hash mismatch')
    filtered = input_dir / f'{source.stem}_noise_excluded.laz'
    cloud.write(filtered)
    provenance = {
        'status': ('ASSUMPTION-ONLY INFERENCE TILE STAGED; NO SCORING' if scenario else 'TRAINING-ONLY TILE STAGED; MODEL NOT RUN'),
        'tile': args.tile,
        'training_sites': training_sites,
        'held_out_intersecting_sites': held_out,
        'safe_full_tile_for_development': not held_out,
        'intersecting_sites': record['intersecting_sites'],
        'source': str(source.resolve()),
        'source_sha256': source_hash,
        'raw_copy_sha256': sha256(raw_copy),
        'input_laz': str(filtered.resolve()),
        'input_laz_sha256': sha256(filtered),
        'original_points': original_count,
        'retained_points': len(cloud.points),
        'excluded_points': original_count - len(cloud.points),
        'excluded_asprs_classes': args.exclude_classes,
        'crs': 'EPSG:2193',
        'bbox_epsg2193': list(map(float, actual)),
        'coverage_plan_sha256': sha256(args.coverage_plan),
        'reference_labels_used': False,
        'held_out_inference_permitted': bool(scenario),
        'inference_scenario_sha256': sha256(args.inference_scenario) if scenario else None,
        'evaluation_permitted': False,
    }
    write_json(args.output / 'provenance.json', provenance)
    write_json(args.output / 'SHA256_manifest.json', {
        str(path.relative_to(args.output)): sha256(path)
        for path in sorted(args.output.rglob('*')) if path.is_file()
    })
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
