#!/usr/bin/env python
"""Audit retained groups and build scoring layers / training review regions.

No group is relabelled, no point instance is invented, no model is trained.
CHM peak counts and compactness are review aids, never tree-count truth.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from scipy import ndimage
from shapely.geometry import mapping

from benchmark.common import ROOT, sha256, site_definitions, write_json


def height_signals(geometry, path):
    with rasterio.open(path) as ds:
        height=ds.read(1,masked=True)
        inside=geometry_mask([mapping(geometry)],height.shape,ds.transform,invert=True)
        valid=inside & ~np.ma.getmaskarray(height) & np.isfinite(height.data)
        if not valid.any(): return {'valid_chm_pixels':0,'height_p95_m':None,'height_max_m':None,'candidate_height_peaks':0}
        # The 0.25 m chip is resampled from historical 1 m CHM: do not infer
        # new sub-metre structure or use these parameters for model selection.
        res=abs(ds.transform.a)
        values=height.data[valid]
        whole_valid=~np.ma.getmaskarray(height)&np.isfinite(height.data)
        weights=ndimage.gaussian_filter(whole_valid.astype(float),sigma=1/res)
        total=ndimage.gaussian_filter(np.where(whole_valid,height.data,0),sigma=1/res)
        smooth=np.divide(total,weights,out=np.zeros_like(total),where=weights>0)
        radius=max(1,round(3/res));size=2*radius+1
        peaks=(smooth==ndimage.maximum_filter(smooth,size=size)) & valid & (smooth>=3)
        _,n=ndimage.label(peaks)
        return {'valid_chm_pixels':int(valid.sum()),'height_p95_m':float(np.percentile(values,95)),
                'height_max_m':float(values.max()),'candidate_height_peaks':int(n)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',type=Path,required=True,help='Refinement GPKG with all_objects layer')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists(): ap.error('Use a fresh output directory')
    frame=gpd.read_file(args.reference,layer='all_objects')
    sites=site_definitions()
    if frame.crs is None or frame.crs.to_epsg()!=2193 or not set(frame.site_id)<=set(sites):
        raise ValueError('Explicit EPSG:2193 and known site IDs required')
    if not frame.geometry.is_valid.all() or frame.geometry.is_empty.any() or not frame.label_id.is_unique:
        raise ValueError('Valid nonempty geometry and unique IDs required')
    args.output.mkdir(parents=True)
    standardized=args.output/'evaluation_reference.gpkg'
    for layer,cls in [('single_trees','tree'),('tree_groups','tree_group'),('uncertainty_regions','uncertain'),('hard_negatives','not_tree')]:
        frame[frame['class']==cls].to_file(standardized,layer=layer,driver='GPKG')
    rows=[]
    for _,r in frame[frame['class']=='tree_group'].iterrows():
        p=r.geometry;chip=ROOT/'data/sites/chips'/r.site_id/'chm.tif'
        compact=4*math.pi*p.area/p.length**2
        signals=height_signals(p,chip)
        rows.append({'site_id':r.site_id,'label_id':r.label_id,'split':r['split'],'stratum':r.stratum,
                     'area_m2':p.area,'compactness':compact,**signals,
                     'review_priority':'compact_crown_candidate' if 28<p.area<220 and compact>.52 else 'inspect_separable_lobes_or_retain',
                     'automatic_single_tree_acceptance':False,'chm_sha256':sha256(chip)})
    with (args.output/'retained_group_review_queue.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # These are XY QA regions only. Interior/vertical checks are still required
    # even in single-tree regions; no semantic_seg/treeID values are emitted.
    training=frame[frame['split']=='train'].copy()
    training['qa_use']=training['class'].map({'tree':'inspect_3D_instance_before_label','tree_group':'unresolved_XY_region_no_instance_target','uncertain':'unresolved_XY_region_no_instance_target','not_tree':'inspect_3D_non_tree_points_before_label'})
    training.to_file(args.output/'training_point_QA_regions.gpkg',layer='qa_regions',driver='GPKG')
    counts=frame.groupby(['split','stratum','class']).size().unstack('class',fill_value=0)
    counts.to_csv(args.output/'counts_by_split_stratum.csv')
    write_json(args.output/'summary.json',{'source_reference_sha256':sha256(args.reference),
        'evaluation_reference_sha256':sha256(standardized),'classes':frame['class'].value_counts().to_dict(),
        'retained_groups':len(rows),'training_QA_regions':len(training),'training_only_QA':True,
        'point_labels_created':False,'reference_frozen':False,'model_outputs_used':False,
        'peak_method':'1 m Gaussian smoothing; 3 m square maximum-filter radius; >=3 m candidate; connected plateaus counted once',
        'peak_limit':'Heuristic review signal on resampled CHM, neither verified tree number nor splitting approval',
        'reference_layer_mapping':'tree->single_trees; tree_group->tree_groups; uncertain->uncertainty_regions; not_tree->hard_negatives'})
    print(json.dumps({'groups_audited':len(rows),'scoring_interface':str(standardized),'point_labels_created':False}))


if __name__=='__main__':main()
