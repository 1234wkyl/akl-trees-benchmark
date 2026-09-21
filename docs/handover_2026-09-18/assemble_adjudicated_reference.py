"""Assemble manually curated full S016/S020 layers with the accepted base.

Never adjudicates, edits source files, generates labels or approves freeze.
Requires real original agreement receipt and a fully reviewed case queue.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark.common import sha256,site_definitions,write_json

CURATED_SITES={'S016','S020'}
CLASSES={'tree','tree_group','not_tree','uncertain'}


def prepare_frames(base,curated,sites):
    """Replace two entire curated sites, retain other sites, validate and clip."""
    for frame in [base,curated]:
        if frame.crs is None or frame.crs.to_epsg()!=2193:raise ValueError('Explicit EPSG:2193 required; do not guess CRS')
        if not {'label_id','site_id','class'}<=set(frame):raise ValueError('Require label_id, site_id and class')
        if frame[['label_id','site_id','class']].isna().any().any():raise ValueError('Missing IDs or classes')
        if not frame.label_id.map(lambda x:isinstance(x,str) and bool(x.strip())).all() or not frame.label_id.is_unique:raise ValueError('Unique nonempty string IDs required')
        if not set(frame['class'])<=CLASSES or not set(frame.site_id)<=set(sites):raise ValueError('Unknown class/site')
        if frame.geometry.isna().any() or frame.geometry.is_empty.any() or not frame.geometry.is_valid.all() or not frame.geometry.geom_type.isin(['Polygon','MultiPolygon']).all():raise ValueError('Manual geometry review required; invalid/empty/nonpolygon object')
    if set(base.site_id)!=set(sites):raise ValueError('Base must contain all declared sites')
    if not set(curated.site_id)<=CURATED_SITES:raise ValueError('Curated layer may only contain S016/S020')
    retained=base[~base.site_id.isin(CURATED_SITES)].copy()
    reviewed=curated.copy();reviewed['annotation_source']='manually_adjudicated_XZ_TW'
    frame=gpd.GeoDataFrame(pd.concat([retained,reviewed],ignore_index=True),geometry='geometry',crs=2193)
    if not frame.label_id.is_unique:raise ValueError('Curated IDs conflict with retained sites')
    audits=[];geoms=[]
    for _,r in frame.iterrows():
        s=sites[r.site_id];p=r.geometry;roi=box(*s['bbox_epsg2193'])
        # Preserve exact coordinate/ring ordering for already-clipped base
        # objects. An unnecessary intersection can reorder unchanged vertices.
        q=p if roi.covers(p) else p.intersection(roi)
        if q.is_empty or q.area<=0 or not q.is_valid or q.geom_type not in {'Polygon','MultiPolygon'}:raise ValueError('Object becomes empty/nonpolygon at site boundary: '+r.label_id)
        audits.append({'label_id':r.label_id,'site_id':r.site_id,'class':r['class'],'area_before_clip_m2':p.area,'area_after_clip_m2':q.area,'removed_outside_area_m2':p.area-q.area})
        geoms.append(q)
    frame.geometry=geoms
    frame['split']=frame.site_id.map(lambda sid:sites[sid]['split'])
    frame['stratum']=frame.site_id.map(lambda sid:sites[sid]['stratum'])
    frame['area_m2']=frame.geometry.area
    frame['centroid_x_2193']=frame.geometry.centroid.x
    frame['centroid_y_2193']=frame.geometry.centroid.y
    return frame,audits


def validate_records(args):
    agreement=json.loads(args.agreement_receipt.read_text())
    if agreement.get('synthetic_only') is not False or agreement.get('status')!='original_agreement_computed_pending_adjudication':raise ValueError('Real original independent agreement receipt required; synthetic/demo is prohibited')
    record=json.loads(args.curation_record.read_text())
    if record.get('status')!='reviewed' or not record.get('reviewer') or not record.get('reviewed_at') or set(record.get('completed_curated_site_ids',[]))!=CURATED_SITES:raise ValueError('Explicit complete manual curation of both sites required')
    checks={'base_reference_sha256':args.base,'curated_sites_sha256':args.curated_sites,
            'agreement_receipt_sha256':args.agreement_receipt,'adjudication_queue_sha256':args.adjudication_queue}
    if any(record.get(k)!=sha256(p) for k,p in checks.items()):raise ValueError('Curation record does not bind exact source files')
    with args.adjudication_queue.open() as f:cases=list(csv.DictReader(f))
    if not cases:raise ValueError('Adjudication queue must contain reviewed original cases')
    expected={f'case_{n:03}' for n in range(1,agreement['review_cases']+1)}
    if len(cases)!=len(expected) or {r.get('case_id') for r in cases}!=expected:raise ValueError('Do not drop, duplicate or renumber original adjudication cases')
    for r in cases:
        if r.get('decision','').strip().lower() in {'','pending','unresolved','deferred'} or not r.get('reviewer','').strip() or not r.get('reason','').strip() or not r.get('reviewed_at','').strip():raise ValueError('Every original case needs a decision, reviewer, date and reason')
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['base','curated-sites','agreement-receipt','adjudication-queue','curation-record','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Use a fresh output directory; existing references are preserved')
    try:
        validate_records(args)
        frame,audits=prepare_frames(gpd.read_file(args.base,layer='all_objects'),gpd.read_file(args.curated_sites,layer='all_objects'),site_definitions())
    except ValueError as e:ap.error(str(e))
    args.output.mkdir(parents=True)
    target=args.output/'adjudicated_all_objects.gpkg';frame.to_file(target,layer='all_objects',driver='GPKG')
    with (args.output/'geometry_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audits[0]));w.writeheader();w.writerows(audits)
    frame.groupby(['split','stratum','class']).size().unstack('class',fill_value=0).to_csv(args.output/'counts_by_split_stratum.csv')
    inputs={k:{'file':str(p),'sha256':sha256(p)} for k,p in vars(args).items() if k!='output'}
    for name in ['agreement_receipt','adjudication_queue','curation_record']:shutil.copyfile(getattr(args,name),args.output/(name+getattr(args,name).suffix))
    write_json(args.output/'assembly_receipt.json',{'status':'manually curated candidate assembled; NOT FROZEN','reference_frozen':False,
        'sources':inputs,'output_sha256':sha256(target),'classes':frame['class'].value_counts().to_dict(),
        'single_trees_by_split':frame[frame['class']=='tree'].groupby('split').size().to_dict(),
        'curated_sites':['S016','S020'],'curated_site_counts':{sid:int(sum(frame.site_id==sid)) for sid in CURATED_SITES},
        'geometry_rule':'valid polygons clipped to original site bounds; no automatic repair or splitting',
        'source_files_written':False,'code_sha256':sha256(Path(__file__))})
    print(str(target.resolve()))


if __name__=='__main__':main()
