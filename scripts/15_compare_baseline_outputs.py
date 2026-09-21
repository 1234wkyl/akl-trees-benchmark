#!/usr/bin/env python
"""Compare reproduced outputs against a frozen output, not against labels."""
import argparse
from pathlib import Path

import geopandas as gpd
from benchmark.common import sha256, write_json


def compare_frames(frozen,reproduced,id_column,area_tolerance=0.0,distance_tolerance=0.0):
    for frame in [frozen,reproduced]:
        if frame.crs is None or frame.crs.to_epsg()!=2193:raise ValueError('comparison requires EPSG:2193')
        if id_column not in frame or frame[id_column].isna().any() or frame[id_column].astype(str).duplicated().any():raise ValueError('stable unique nonmissing IDs required')
        if frame.geometry.isna().any() or frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():raise ValueError('invalid comparison geometry')
    a={str(r[id_column]):r.geometry for _,r in frozen.iterrows()}
    b={str(r[id_column]):r.geometry for _,r in reproduced.iterrows()}
    missing=sorted(set(a)-set(b));extra=sorted(set(b)-set(a));rows=[]
    for key in sorted(set(a)&set(b)):
        if a[key].geom_type!=b[key].geom_type:raise ValueError('point/crown geometry type differs')
        if a[key].geom_type in {'Polygon','MultiPolygon'}:
            difference=a[key].symmetric_difference(b[key]).area
            hausdorff=a[key].hausdorff_distance(b[key])
            match=difference<=area_tolerance and hausdorff<=distance_tolerance
            rows.append({'id':key,'symmetric_difference_m2':difference,'hausdorff_m':hausdorff,'matches':match})
        elif a[key].geom_type=='Point':
            distance=a[key].distance(b[key]);rows.append({'id':key,'distance_m':distance,'matches':distance<=distance_tolerance})
        else:raise ValueError('expected points or crown polygons')
    return {'frozen_objects':len(a),'reproduced_objects':len(b),'missing_ids':missing,'extra_ids':extra,
            'area_tolerance_m2':area_tolerance,'distance_tolerance_m':distance_tolerance,
            'geometry_matches':not missing and not extra and all(r['matches'] for r in rows),
            'matched_objects':rows,'note':'Geometry/ID check only. Algorithm, preprocessing, detection inventory and run provenance must also match; this is not label accuracy or full scalar-attribute equality.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--frozen',type=Path,required=True)
    ap.add_argument('--reproduced',type=Path,required=True)
    ap.add_argument('--layer')
    ap.add_argument('--id-column',default='tree_id')
    ap.add_argument('--area-tolerance-m2',type=float,default=0)
    ap.add_argument('--distance-tolerance-m',type=float,default=0)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    import math
    if args.output.exists():ap.error('use a fresh output file')
    if any(not math.isfinite(v) or v<0 for v in [args.area_tolerance_m2,args.distance_tolerance_m]):ap.error('finite nonnegative tolerances required')
    a=gpd.read_file(args.frozen,layer=args.layer);b=gpd.read_file(args.reproduced,layer=args.layer)
    report=compare_frames(a,b,args.id_column,args.area_tolerance_m2,args.distance_tolerance_m)
    report.update(frozen_sha256=sha256(args.frozen),reproduced_sha256=sha256(args.reproduced))
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,report)
    print(f"Geometry reproduction match: {report['geometry_matches']}")
    if not report['geometry_matches']:raise SystemExit(1)


if __name__=='__main__':main()
