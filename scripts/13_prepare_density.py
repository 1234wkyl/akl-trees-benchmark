#!/usr/bin/env python
"""Exact-size nested random return thinning, preserving native LAS records."""
import argparse
import csv
import math
import shutil
from pathlib import Path

import laspy
import numpy as np
from shapely.geometry import box

from benchmark.common import sha256, site_definitions, write_json
from benchmark.inference_scenario import validate_inference_scenario


def nested_indices(n, fractions, seed):
    if n < 1 or not fractions or any(not math.isfinite(f) or not 0 < f <= 1 for f in fractions):
        raise ValueError('positive point count and fractions in (0,1] required')
    if len(set(fractions)) != len(fractions):
        raise ValueError('fractions must be distinct')
    rank = np.random.Generator(np.random.PCG64(seed)).permutation(n)
    return {fraction: np.sort(rank[:max(1, math.floor(n * fraction))])
            for fraction in fractions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--fractions', type=float, nargs='+', default=[1, .75, .5, .25, .125])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inference-scenario', type=Path, help='explicit fixed hypothetical protocol; default remains training-only')
    args = ap.parse_args()
    scenario = validate_inference_scenario(args.inference_scenario) if args.inference_scenario else None
    if args.output.exists():
        ap.error('use a fresh output folder')
    source_hash = sha256(args.input)
    data = laspy.read(args.input)
    crs = data.header.parse_crs()
    if crs is None or crs.to_epsg() != 2193:
        raise ValueError('input must use EPSG:2193')
    sites = site_definitions()
    bounds = box(*data.header.mins[:2], *data.header.maxs[:2])
    overlaps = [s for s in sites.values() if box(*s['bbox_epsg2193']).intersection(bounds).area > 0]
    if not overlaps or (not scenario and any(s['split'] != 'train' for s in overlaps)):
        raise ValueError('preparation requires a training-only tile, without held-out overlap')
    selections = nested_indices(len(data.points), args.fractions, args.seed)
    args.output.mkdir(parents=True)
    audit, runs = [], []
    for fraction in sorted(selections, reverse=True):
        selected = selections[fraction]
        name = f'fraction_{fraction:.6f}'.rstrip('0').rstrip('.')
        directory = args.output/name
        directory.mkdir()
        target = directory/args.input.name
        index_path = directory/'source_point_indices.npy'
        np.save(index_path, selected, allow_pickle=False)
        if fraction == 1:
            shutil.copy2(args.input, target)
        else:
            subset = laspy.LasData(data.header.copy())
            subset.points = data.points[selected].copy()
            subset.write(target)
        check = laspy.read(target)
        if not np.array_equal(check.points.array, data.points.array[selected]):
            raise ValueError('native record/ordering preservation failed')
        if check.header.parse_crs().to_epsg() != 2193:
            raise ValueError('CRS was lost during thinning')
        for site in overlaps:
            x0, y0, x1, y1 = site['bbox_epsg2193']
            mask = (check.x >= x0) & (check.x < x1) & (check.y >= y0) & (check.y < y1)
            xyz = np.column_stack((check.x[mask],check.y[mask],check.z[mask]))
            unique = len(np.unique(np.round(xyz,4),axis=0))
            area = (x1-x0)*(y1-y0)
            covered_area = box(x0,y0,x1,y1).intersection(bounds).area
            audit.append({'fraction':fraction,'site_id':site['site_id'],'split':site['split'],
                          'returns_in_site':int(mask.sum()),'unique_xyz_returns':unique,
                          'tile_header_overlap_area_m2':covered_area,
                          'tile_header_site_coverage_fraction':covered_area/area,
                          'tile_returns_per_full_site_m2':unique/area,
                          'returns_per_tile_header_overlap_m2':unique/covered_area})
        classes, counts = np.unique(check.classification,return_counts=True)
        runs.append({'fraction':fraction,'points':len(check.points),
                     'actual_retained_fraction':len(check.points)/len(data.points),
                     'input_file':str(target.relative_to(args.output)), 'sha256':sha256(target),
                     'point_indices_sha256':sha256(index_path),
                     'classes':dict(zip(map(str,classes),map(int,counts))),
                     'native_record_subset_verified':True})
    write_json(args.output/'manifest.json',{
        'status':'CPU preparation only; candidate density design, no model predictions',
        'source_file':str(args.input.resolve()), 'source_sha256':source_hash,
        'source_points':len(data.points), 'source_preprocessing':'inherited from provided input; no further classification filter',
        'crs':'EPSG:2193','seed':args.seed,'rng':'numpy.PCG64.permutation',
        'numpy_version':np.__version__,'thinning_unit':'individual return record, not laser pulse',
        'nesting':'lower fractions are subsets of higher fractions at the same seed/source',
        'count_rule':'max(1,floor(N*fraction)); original record order retained',
        'training_sites':[s['site_id'] for s in overlaps if s['split']=='train'],
        'held_out_sites':[s['site_id'] for s in overlaps if s['split']!='train'],
        'inference_scenario_sha256':sha256(args.inference_scenario) if scenario else None,
        'formal_reference_frozen':False,'runs':runs,
        'source_unchanged':sha256(args.input)==source_hash})
    with (args.output/'site_density.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(audit[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(audit)
    (args.output/'README.md').write_text('''# Training-tile density inputs

These are exact-size nested random subsets of individual LAS return records, not pulse thinning or simulated acquisition. Complete native point dimensions, CRS and order are retained and verified after writing. The 100% file is byte-identical to the supplied input. Its prior noise filter, if any, remains inherited and must be recorded with the original smoke manifest. Source indices link each result to that input. Unique XYZ is counted at four decimal places with half-open XY boundaries. The CSV reports this tile's contribution per full-site area and its header-overlap coverage; these are NOT whole-site densities when a site spans multiple tiles. A full benchmark must process all covering tiles and deduplicate their overlap consistently.

No density-performance curve or model result exists yet. Fractions and seed are candidate technical preparations; freeze the full experiment design before held-out inference. On VM, run each folder separately with the same fixed code, checkpoint, environment and segmentation settings. Do not regenerate a different random subset for each model. The historical classical CHM does not change when a LAZ is thinned; applying a density comparison to that baseline would require a separately verified DSM/DEM/CHM regeneration pipeline.
''')
    if scenario:
        (args.output/'README.md').write_text('''# Fixed conditional density inputs

I retained exact-size nested subsets of native LAS return records using PCG64 permutation, seed as recorded in the manifest. Source indices, complete point dimensions, ordering and EPSG:2193 were verified after writing. Input preprocessing is inherited, without any additional class filter. The held-out permission is explicit and bound to the unchanged assumption-only inference scenario; this does not approve a formal reference freeze.

The CSV gives this tile's contribution, not deduplicated whole-site density. These subsets simulate return thinning, not pulse thinning or a new acquisition. I reuse the existing full-density predictions and report all four predeclared retention conditions with fixed model and crown settings. The existing CHM stays unchanged and serves only as a fixed classical comparator.
''')
    print(f'Prepared {len(runs)} nested inputs; original points {len(data.points):,}')


if __name__ == '__main__':
    main()
