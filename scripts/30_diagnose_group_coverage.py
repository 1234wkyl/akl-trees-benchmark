#!/usr/bin/env python
"""Secondary group canopy-area diagnostics, never individual-tree scores.

Default is training only. Held-out use requires the same hash-bound freeze
and completed prediction run record as the primary scorer.
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

from benchmark.common import ROOT, sha256, site_definitions, validate_freeze, write_json


def group_coverage(group,predictions):
    if group.is_empty or not group.is_valid or group.area<=0:
        raise ValueError('Valid nonempty positive-area group required')
    if any(p.is_empty or not p.is_valid or p.area<=0 for p in predictions):
        raise ValueError('Valid nonempty positive-area predictions required')
    linked=[p for p in predictions if p.intersection(group).area>0]
    union=unary_union(linked)
    covered=union.intersection(group).area
    union_area=union.union(group).area
    return {'group_area_m2':group.area,'covered_area_m2':covered,
            'uncovered_area_m2':group.area-covered,'coverage_fraction':covered/group.area,
            'linked_prediction_count':len(linked),'linked_union_iou':covered/union_area,
            'linked_union_outside_group_m2':union.difference(group).area}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--stage',choices=['development','benchmark'],default='development')
    ap.add_argument('--freeze-record',type=Path)
    ap.add_argument('--policy',type=Path,default=ROOT/'config/evaluation_draft.json')
    ap.add_argument('--prediction-run-record',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Use a fresh output file')
    sites={sid:s for sid,s in site_definitions().items() if s['split']==('train' if args.stage=='development' else 'test')}
    if args.stage=='benchmark':
        validate_freeze(args.freeze_record,sha256(args.reference),sha256(args.policy))
        if args.prediction_run_record is None:raise ValueError('Completed prediction run record required')
        run=json.loads(args.prediction_run_record.read_text())
        if run.get('status')!='completed' or set(run.get('completed_site_ids',[]))!=set(sites) or run.get('predictions_sha256')!=sha256(args.predictions) or not run.get('model_configuration_sha256'):
            raise ValueError('Run record must bind completed held-out coverage and frozen configuration')
    groups=gpd.read_file(args.reference,layer='tree_groups')
    predictions=gpd.read_file(args.predictions,layer='predictions')
    for frame in [groups,predictions]:
        if frame.crs is None or frame.crs.to_epsg()!=2193 or 'site_id' not in frame or frame.site_id.isna().any():raise ValueError('Explicit EPSG:2193 and site_id required')
        if frame.geometry.isna().any() or not frame.geometry.is_valid.all() or frame.geometry.is_empty.any() or not frame.geometry.geom_type.isin(['Polygon','MultiPolygon']).all():raise ValueError('Valid nonempty polygons required')
    if not set(groups.site_id)<=set(site_definitions()):raise ValueError('Unknown reference sites')
    if not {'site_id','prediction_id'}<=set(predictions) or predictions.prediction_id.isna().any() or predictions.duplicated(['site_id','prediction_id']).any():raise ValueError('Unique nonmissing site/prediction IDs required')
    if not set(predictions.site_id)<=set(sites):raise ValueError('Predictions outside requested split prohibited')
    rows=[]
    for _,r in groups[groups.site_id.isin(sites)].iterrows():
        roi=box(*sites[r.site_id]['bbox_epsg2193']);g=r.geometry.intersection(roi)
        ps=[p.intersection(roi) for p in predictions[predictions.site_id==r.site_id].geometry]
        ps=[p for p in ps if not p.is_empty and p.area>0]
        rows.append({'site_id':r.site_id,'label_id':r.label_id,**group_coverage(g,ps)})
    total=sum(r['group_area_m2'] for r in rows);covered=sum(r['covered_area_m2'] for r in rows)
    write_json(args.output,{'diagnostic_only':True,'individual_tree_count_inferred':False,
        'acceptance_thresholds_changed':False,'stage':args.stage,
        'reference_sha256':sha256(args.reference),'predictions_sha256':sha256(args.predictions),
        'group_count':len(rows),'area_weighted_coverage_fraction':covered/total if total else None,
        'groups':rows,'interpretation':'Union coverage measures how much annotated group area is covered, not whether individual trees are correctly split. Linked-union IoU includes adjoining crowns and boundary uncertainty; no pass threshold or winner rule. Overlapping reference groups can double-count area.'})
    print('Secondary group coverage written; individual-tree scores unchanged')


if __name__=='__main__':main()
