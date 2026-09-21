#!/usr/bin/env python
"""Draft XY convex-hull conversion; not an approved crown-IoU convention."""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
from shapely import concave_hull
from shapely.geometry import MultiPoint
from benchmark.common import sha256, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prediction',type=Path,required=True)
    ap.add_argument('--coordinate-audit',type=Path,required=True)
    ap.add_argument('--tree-semantic-values',type=int,nargs='+',required=True,
                    help='explicit values confirmed from the actual checkpoint taxonomy, never original ASPRS values')
    ap.add_argument('--hull-method',choices=['convex_hull','concave_hull'],default='convex_hull')
    ap.add_argument('--concave-ratio',type=float,default=.5,
                    help='GEOS concave-hull ratio in [0,1]; used only with --hull-method concave_hull')
    ap.add_argument('--buffer-m',type=float,default=0,
                    help='optional fixed outward buffer applied after hull construction')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('use a fresh output folder')
    if not np.isfinite(args.concave_ratio) or not 0<=args.concave_ratio<=1:
        ap.error('--concave-ratio must be in [0,1]')
    if not np.isfinite(args.buffer_m) or args.buffer_m<0:
        ap.error('--buffer-m must be finite and nonnegative')
    audit=json.loads(args.coordinate_audit.read_text())
    if audit.get('prediction_sha256')!=sha256(args.prediction) or audit.get('epsg2193_coordinate_consistency') is not True or audit.get('uint16_boundary_warning'):
        raise ValueError('require matching successful restored-coordinate audit, without ID-boundary warning')
    pred=laspy.read(args.prediction)
    ids=np.asarray(pred['PredInstance']);semantic=np.asarray(pred['PredSemantic'])
    values=set(map(int,np.unique(semantic)))
    if len(set(args.tree_semantic_values))!=len(args.tree_semantic_values) or any(v<0 or v>255 for v in args.tree_semantic_values):
        raise ValueError('explicit distinct uint8 semantic values required')
    xy=np.column_stack((pred.x,pred.y));mask=np.isin(semantic,args.tree_semantic_values)&(ids>0)
    rows=[];geometries=[];skipped=[]
    # Group once rather than scanning every point for every instance.
    order=np.argsort(ids[mask],kind='stable');sorted_ids=ids[mask][order];points=xy[mask][order]
    boundaries=np.r_[0,np.flatnonzero(np.diff(sorted_ids))+1,len(sorted_ids)]
    for start,stop in zip(boundaries[:-1],boundaries[1:]):
        if start==stop:continue
        iid=int(sorted_ids[start]);unique=np.unique(points[start:stop],axis=0)
        points_geometry=MultiPoint(unique)
        hull=(points_geometry.convex_hull if args.hull_method=='convex_hull'
              else concave_hull(points_geometry,ratio=args.concave_ratio,allow_holes=False))
        if hull.geom_type!='Polygon' or hull.area<=0:
            skipped.append({'instance_id':iid,'points':int(stop-start),'unique_xy':len(unique),'reason':'no nonzero-area polygon'});continue
        if args.buffer_m:
            hull=hull.buffer(args.buffer_m)
        if hull.geom_type!='Polygon' or hull.area<=0 or not hull.is_valid:
            skipped.append({'instance_id':iid,'points':int(stop-start),'unique_xy':len(unique),'reason':'invalid buffered polygon'});continue
        rows.append({'prediction_id':str(iid),'instance_id':iid,'points':int(stop-start),
                     'conversion':f'{args.hull_method}_candidate','area_m2':hull.area});geometries.append(hull)
    args.output.mkdir(parents=True)
    frame=gpd.GeoDataFrame(rows,columns=['prediction_id','instance_id','points','conversion','area_m2'],geometry=geometries,crs=2193)
    target=args.output/'candidate_crowns.gpkg';frame.to_file(target,layer='candidate_crowns',driver='GPKG')
    write_json(args.output/'provenance.json',{'status':'DRAFT CONVERSION ONLY; not approved benchmark crowns',
        'prediction_sha256':sha256(args.prediction),'coordinate_audit_sha256':sha256(args.coordinate_audit),
        'output_sha256':sha256(target),'crs':'EPSG:2193, derived from audited coordinates',
        'tree_semantic_values':args.tree_semantic_values,'instance_zero_excluded':True,
        'requested_semantic_values_absent_from_output':sorted(set(args.tree_semantic_values)-values),
        'crowns':len(frame),'skipped_instances':skipped,'hull_method':args.hull_method,
        'concave_ratio':args.concave_ratio if args.hull_method=='concave_hull' else None,
        'buffer_m':args.buffer_m,
        'method':f'{args.hull_method} of selected instance XY points followed by a fixed {args.buffer_m:g} m outward buffer',
        'limitations':'Candidate hull geometry and buffering depend on point support and may change with density; no anatomical boundary accuracy is asserted. uint16 upstream wrapping cannot be disproved from output alone. Fix conversion convention/settings on training data before held-out scoring.',
        'conversion_approved':False,'model_scores_created':False})
    print(f'Created {len(frame)} draft crown candidates; skipped {len(skipped)} degenerate instances')


if __name__=='__main__':main()
