#!/usr/bin/env python
"""Audit one restored predicted tile; no crown polygons or model scores invented."""
import argparse
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

from benchmark.common import sha256, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--prediction',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--coordinate-tolerance-m',type=float,default=.002)
    args=ap.parse_args()
    if args.output.exists():ap.error('use a fresh output file')
    if not np.isfinite(args.coordinate_tolerance_m) or args.coordinate_tolerance_m<=0:ap.error('positive tolerance required')
    source=laspy.read(args.input);pred=laspy.read(args.prediction)
    if source.header.parse_crs() is None or source.header.parse_crs().to_epsg()!=2193:
        raise ValueError('source CRS must be EPSG:2193')
    dimensions=set(pred.point_format.dimension_names)
    if not {'PredInstance','PredSemantic'}<=dimensions:raise ValueError('missing upstream predicted fields')
    xyz=np.column_stack((source.x,source.y,source.z));pxyz=np.column_stack((pred.x,pred.y,pred.z))
    if not len(xyz) or not len(pxyz) or not np.isfinite(pxyz).all():raise ValueError('empty/nonfinite output')
    distances,_=cKDTree(xyz).query(pxyz,k=1)
    reverse,_=cKDTree(pxyz).query(xyz,k=1)
    coordinates_ok=bool((distances<=args.coordinate_tolerance_m).all() and (reverse<=args.coordinate_tolerance_m).all())
    ids=np.asarray(pred['PredInstance'])
    values,counts=np.unique(ids,return_counts=True)
    crs=pred.header.parse_crs()
    report={'input_sha256':sha256(args.input),'prediction_sha256':sha256(args.prediction),
            'input_points':len(source.points),'prediction_points':len(pred.points),
            'point_count_equal':len(source.points)==len(pred.points),
            'coordinate_tolerance_m':args.coordinate_tolerance_m,'max_output_to_input_distance_m':float(distances.max()),
            'max_input_to_output_distance_m':float(reverse.max()),'bidirectional_coordinate_coverage_ok':coordinates_ok,
            'coordinate_identity_note':'Nearest-neighbour coverage and equal counts do not prove one-to-one identity when coordinates are duplicated; retain this limitation.',
            'output_crs_metadata':crs.to_string() if crs else None,
            'epsg2193_coordinate_consistency':coordinates_ok and len(source.points)==len(pred.points) and (crs is None or crs.to_epsg()==2193),
            'zero_id_points':int((ids==0).sum()),'positive_instances':int((values>0).sum()),
            'max_instance_id':int(ids.max()),'uint16_boundary_warning':bool(ids.max()>=65535),
            'instance_point_counts':dict(zip(map(str,values),map(int,counts))),
            'semantic_values':list(map(int,np.unique(pred['PredSemantic']))),
            'crown_conversion_approved':False,'benchmark_scores_created':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,report)
    print(f'Output audit saved: {args.output}')
    if not report['epsg2193_coordinate_consistency'] or report['uint16_boundary_warning']:
        raise SystemExit(1)


if __name__=='__main__':main()
