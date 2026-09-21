#!/usr/bin/env python
"""Recalculate complete training-site return density from prepared smoke inputs."""
import argparse
import csv
import json
from pathlib import Path

import laspy
import numpy as np
from shapely.geometry import box

from benchmark.common import sha256, site_definitions


def counts(path, bounds):
    data = laspy.read(path)
    crs = data.header.parse_crs()
    if crs is None or crs.to_epsg() != 2193:
        raise ValueError(f'{path}: expected EPSG:2193')
    if not box(*data.header.mins[:2], *data.header.maxs[:2]).covers(box(*bounds)):
        raise ValueError(f'{path}: header does not cover the complete site')
    x0,y0,x1,y1 = bounds
    x,y,z = map(np.asarray,(data.x,data.y,data.z))
    selected = (x>=x0)&(x<x1)&(y>=y0)&(y<y1)
    xyz = np.column_stack((x[selected],y[selected],z[selected]))
    return len(data.points),int(selected.sum()),int(len(np.unique(np.round(xyz,4),axis=0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site-preparation', action='append', required=True,
                    help='SITE_ID=folder produced by 09_prepare_sat_smoke.py')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('use a fresh output file')
    sites = site_definitions()
    rows, seen = [], set()
    for value in args.site_preparation:
        if '=' not in value:
            ap.error('--site-preparation expects SITE_ID=folder')
        site_id, folder = value.split('=',1)
        if site_id in seen or site_id not in sites or sites[site_id]['split'] != 'train':
            ap.error('site IDs must be unique training sites')
        seen.add(site_id)
        folder = Path(folder)
        meta = json.loads((folder/'provenance.json').read_text())
        if meta['training_site'] != site_id or meta['filtering']['excluded_asprs_classes'] != [7,18]:
            raise ValueError('preparation provenance/site/filter mismatch')
        source = folder/'raw_source'/meta['selected_tile']
        filtered = folder/'input_laz'/meta['input_laz_filename']
        if sha256(source) != meta['source_sha256'] or sha256(filtered) != meta['input_laz_sha256']:
            raise ValueError('input hash does not match preparation provenance')
        bounds = sites[site_id]['bbox_epsg2193']
        raw_total,raw_site,raw_unique = counts(source,bounds)
        filtered_total,site_returns,unique = counts(filtered,bounds)
        if raw_total != meta['original_points'] or filtered_total != meta['points'] or site_returns>raw_site or unique>raw_unique:
            raise ValueError('site counts or preparation provenance disagree')
        area=(bounds[2]-bounds[0])*(bounds[3]-bounds[1])
        rows.append(dict(site_id=site_id,stratum=sites[site_id]['stratum'],site_area_m2=area,
                         raw_tile_points=raw_total,raw_site_returns=raw_site,raw_site_unique_xyz=raw_unique,
                         filtered_tile_points=filtered_total,filtered_site_returns=site_returns,
                         filtered_site_unique_xyz=unique,filtered_returns_per_m2=site_returns/area,
                         filtered_unique_xyz_per_m2=unique/area,
                         source_sha256=sha256(source),filtered_input_sha256=sha256(filtered)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    print(f'Measured {len(rows)} complete training sites: {args.output}')


if __name__=='__main__':
    main()
