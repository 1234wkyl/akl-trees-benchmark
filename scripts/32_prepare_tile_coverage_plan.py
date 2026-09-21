#!/usr/bin/env python
"""Prepare complete-site LAZ staging lists from headers, not one-tile guesses.

Reads headers only; does not transfer, infer, decompress or train. Geometry
coverage is a bounding-box diagnostic, not proof of complete observations.
"""
import argparse
import csv
import json
from pathlib import Path

import laspy
from shapely.geometry import box
from shapely.ops import unary_union

from benchmark.common import ROOT,sha256,site_definitions,write_json


def union_coverage(roi,footprints):
    return unary_union([p.intersection(roi) for p in footprints]).area/roi.area if footprints else 0.0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--laz-root',type=Path,default=ROOT.parent/'Xi_Zhang_Benchmark/laz')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Use a fresh output directory')
    sites=site_definitions();required=[s.strip() for s in (ROOT/'data/sites/required_tiles.txt').read_text().splitlines() if s.strip() and not s.startswith('#')]
    if len(required)!=len(set(required)):raise ValueError('Duplicate required tile names')
    source_manifest=ROOT.parent/'Xi_Zhang_Benchmark/SHA256SUMS.txt'
    hashes={}
    for line in source_manifest.read_text().splitlines():
        digest,name=line.split(maxsplit=1);hashes[name.lstrip('*')]=digest
    tiles=[];links=[]
    for name in required:
        path=args.laz_root/name
        if not path.is_file():raise ValueError('Required tile missing: '+name)
        if 'laz/'+name not in hashes:raise ValueError('Required tile absent from delivered checksum manifest: '+name)
        with laspy.open(path) as f:
            h=f.header;crs=h.parse_crs()
            if crs is None or crs.to_epsg()!=2193:raise ValueError('Explicit EPSG:2193 required')
            footprint=box(h.mins[0],h.mins[1],h.maxs[0],h.maxs[1]);touch=[]
            for sid,s in sites.items():
                area=footprint.intersection(box(*s['bbox_epsg2193'])).area
                if area>0:
                    touch.append(sid);links.append({'site_id':sid,'split':s['split'],'stratum':s['stratum'],'tile':name,'intersection_area_m2':area})
            held=[sid for sid in touch if sites[sid]['split']!='train']
            tiles.append({'tile':name,'header_points':int(h.point_count),'bytes':path.stat().st_size,
                'bbox_epsg2193':list(footprint.bounds),'sha256':hashes['laz/'+name],
                'sha256_source':'previously verified delivered bundle manifest; not rehashed in header-only planner',
                'intersecting_sites':touch,'held_out_intersecting_sites':held,
                'safe_full_tile_for_development':bool(touch) and not held})
    by_name={r['tile']:r for r in tiles};site_rows=[]
    for sid,s in sites.items():
        names=[r['tile'] for r in links if r['site_id']==sid]
        shapes=[box(*by_name[n]['bbox_epsg2193']) for n in names]
        safe=[n for n in names if by_name[n]['safe_full_tile_for_development']]
        site_rows.append({'site_id':sid,'split':s['split'],'stratum':s['stratum'],'required_tiles':names,
            'header_bbox_union_coverage_fraction':union_coverage(box(*s['bbox_epsg2193']),shapes),
            'safe_full_development_tiles':safe if s['split']=='train' else [],
            'development_tile_conflicts':[n for n in names if by_name[n]['held_out_intersecting_sites']] if s['split']=='train' else []})
    args.output.mkdir(parents=True)
    with (args.output/'site_tile_links.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['site_id','split','stratum','tile','intersection_area_m2']);w.writeheader();w.writerows(links)
    for split in ['train','val','test']:
        names=sorted({r['tile'] for r in links if r['split']==split})
        (args.output/(split+'_tiles.txt')).write_text('\n'.join(names)+'\n')
    safe_names=sorted(r['tile'] for r in tiles if r['safe_full_tile_for_development'])
    (args.output/'safe_development_tiles.txt').write_text('\n'.join(safe_names)+'\n')
    (args.output/'LAZ_SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['tile']+'\n' for r in tiles))
    write_json(args.output/'coverage_plan.json',{'status':'staging/preflight plan, not permission for held-out inference',
        'tiles':tiles,'sites':site_rows,'source_manifest_sha256':sha256(source_manifest),
        'site_definitions_sha256':sha256(ROOT/'data/sites/reference_sites.json'),
        'required_tiles_sha256':sha256(ROOT/'data/sites/required_tiles.txt'),
        'header_only':True,'model_run_executed':False,'input_filtering_applied':False,
        'bbox_coverage_limit':'Header extrema describe point bounds, not authoritative acquisition footprints or complete pulses/returns. Multi-tile clouds still need recorded merge/dedup and context rules.',
        'development_conflict_limit':'A full training tile touching a val/test reference is not permitted by the current development runner; plan must resolve cropping/context before execution.'})
    print(json.dumps({'required_tiles':len(tiles),'sites':len(site_rows),'safe_development_tiles':len(safe_names),
        'minimum_header_bbox_coverage':min(r['header_bbox_union_coverage_fraction'] for r in site_rows),
        'training_sites_with_tile_conflicts':[r['site_id'] for r in site_rows if r['development_tile_conflicts']]}))


if __name__=='__main__':main()
