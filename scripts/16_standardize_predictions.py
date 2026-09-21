#!/usr/bin/env python
"""Reproject explicitly and clip crown outputs into stable per-site records."""
import argparse
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from benchmark.common import sha256, site_definitions, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--layer')
    ap.add_argument('--id-column',required=True)
    ap.add_argument('--split',choices=['train','val','test'],default='train')
    ap.add_argument('--site-ids',nargs='+',help='optional unique subset within --split')
    ap.add_argument('--allow-reprojection',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('use a fresh output folder')
    data=gpd.read_file(args.input,layer=args.layer)
    if data.crs is None:raise ValueError('missing CRS; cannot guess geographic origin')
    original_crs=data.crs.to_string()
    if data.crs.to_epsg()!=2193:
        if not args.allow_reprojection:raise ValueError('reprojection requires explicit --allow-reprojection')
        data=data.to_crs(2193)
    if args.id_column not in data or data[args.id_column].isna().any() or data[args.id_column].astype(str).duplicated().any():raise ValueError('unique stable object IDs required')
    if data.geometry.isna().any() or data.geometry.is_empty.any() or not data.geometry.is_valid.all() or not data.geometry.geom_type.isin(['Polygon','MultiPolygon']).all():raise ValueError('input must be valid crowns, not detection points')
    sites={sid:s for sid,s in site_definitions().items() if s['split']==args.split}
    if args.site_ids:
        requested=set(args.site_ids)
        if len(requested)!=len(args.site_ids) or not requested<=set(sites):
            raise ValueError('requested site IDs must be unique and belong to --split')
        sites={sid:sites[sid] for sid in args.site_ids}
    records=[];geometries=[]
    for _,r in data.iterrows():
        for sid,s in sites.items():
            geometry=r.geometry.intersection(box(*s['bbox_epsg2193']))
            if not geometry.is_empty and geometry.area>0:
                records.append({'site_id':sid,'prediction_id':str(r[args.id_column]),'split':s['split'],'stratum':s['stratum']})
                geometries.append(geometry)
    args.output.mkdir(parents=True)
    frame=gpd.GeoDataFrame(records,columns=['site_id','prediction_id','split','stratum'],geometry=geometries,crs=2193)
    target=args.output/'predictions.gpkg';frame.to_file(target,layer='predictions',driver='GPKG')
    write_json(args.output/'provenance.json',{'input_file':str(args.input),'input_sha256':sha256(args.input),
        'input_crs':original_crs,'output_crs':'EPSG:2193','split':args.split,'requested_site_ids':list(sites),'site_counts':{sid:int((frame.site_id==sid).sum()) for sid in sites},
        'output_sha256':sha256(target),'geometry_rule':'positive-area site intersection, no geometry repair or invented crowns',
        'note':'This transforms supplied crowns only. It neither certifies frozen provenance nor approves point-to-crown conversion. Outputs outside sites are excluded from site scoring.'})
    print(f'Standardised {len(frame)} crowns for {args.split}')


if __name__=='__main__':main()
