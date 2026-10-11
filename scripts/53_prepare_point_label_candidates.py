#!/usr/bin/env python3
"""Prepare real-XYZ, crown-guided point-label candidates for manual 3D QA.

NOT checked ground truth and NOT permission to launch supervised fine-tuning.
Tree groups, boundary overlaps and unresolved points remain unknown. The
training-only raw PLY schema matches SAT; trainer ignore-loss behavior must
still be verified or reviewed complete subregions selected.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import geopandas as gpd
import laspy
import numpy as np
from shapely import intersects_xy
from benchmark.common import ROOT, sha256, site_definitions, write_json


def assign_candidates(x, y, classification, singles, ambiguous, negatives):
    hit_count=np.zeros(len(x),dtype=np.int16)
    owner=np.full(len(x),-1,dtype=np.int32)
    for index,geometry in enumerate(singles):
        mask=intersects_xy(geometry,x,y)
        hit_count[mask]+=1;owner[mask]=index
    uncertain=np.zeros(len(x),dtype=bool)
    for geometry in ambiguous:
        uncertain |= intersects_xy(geometry,x,y)
    hard_negative=np.zeros(len(x),dtype=bool)
    for geometry in negatives:
        hard_negative |= intersects_xy(geometry,x,y)
    conflict=(hit_count>1)|uncertain|((hit_count>0)&hard_negative)
    semantic=np.zeros(len(x),dtype=np.int32)
    instance=np.full(len(x),-1,dtype=np.int32)
    # Conservative weak candidates: medium/high vegetation only; low vegetation
    # and unclassified foliage remain unknown until point-level inspection.
    tree=(hit_count==1)&~conflict&np.isin(classification,[4,5])
    non_tree=(classification==2)|((hard_negative)&~np.isin(classification,[3,4,5]))
    non_tree &= ~conflict
    semantic[non_tree]=1
    semantic[tree]=2;instance[tree]=owner[tree]
    return semantic,instance,conflict


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--coverage-plan',type=Path,required=True)
    ap.add_argument('--laz-root',type=Path,required=True)
    ap.add_argument('--sat-source',type=Path,default=ROOT.parent/'SegmentAnyTree')
    ap.add_argument('--site-ids',nargs='+')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('fresh output directory required')
    defs=site_definitions();sites={k:v for k,v in defs.items() if v['split']=='train'}
    if args.site_ids:
        if len(args.site_ids)!=len(set(args.site_ids)) or not set(args.site_ids)<=set(sites):ap.error('only unique training site IDs')
        sites={k:sites[k] for k in args.site_ids}
    plan=json.loads(args.coverage_plan.read_text())
    if plan['site_definitions_sha256']!=sha256(ROOT/'data/sites/reference_sites.json'):raise ValueError('coverage plan mismatch')
    refs=gpd.read_file(args.reference,layer='all_objects')
    if refs.crs is None or refs.crs.to_epsg()!=2193 or not refs.geometry.is_valid.all():raise ValueError('valid EPSG:2193 reference required')
    reader_path=args.sat_source/'torch_points3d/modules/KPConv/plyutils.py'
    spec=importlib.util.spec_from_file_location('upstream_ply',reader_path)
    ply=importlib.util.module_from_spec(spec);spec.loader.exec_module(ply)
    args.output.mkdir(parents=True)
    (args.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    cache={};reports=[]
    for sid,site in sites.items():
        psite=next(row for row in plan['sites'] if row['site_id']==sid)
        a,b,c,d=site['bbox_epsg2193'];parts=[];fingerprints={}
        for tile in psite['required_tiles']:
            meta=next(row for row in plan['tiles'] if row['tile']==tile)
            if meta['safe_full_tile_for_development'] is not True or meta['held_out_intersecting_sites']:raise ValueError('held-out touching tile forbidden')
            path=args.laz_root/tile
            if tile not in cache:
                digest=sha256(path)
                if digest!=meta['sha256']:raise ValueError('raw LAZ hash mismatch: '+tile)
                cloud=laspy.read(path)
                if cloud.header.parse_crs().to_epsg()!=2193:raise ValueError('explicit EPSG:2193 LAZ required')
                cache[tile]=(cloud,digest)
            cloud,digest=cache[tile];fingerprints[tile]=digest
            x,y,z=map(np.asarray,(cloud.x,cloud.y,cloud.z));cls=np.asarray(cloud.classification)
            keep=(x>=a)&(x<c)&(y>=b)&(y<d)&~np.isin(cls,[7,18])
            # Return identifiers retain coincident but distinct returns.
            parts.append(np.column_stack([x[keep],y[keep],z[keep],cls[keep],
                np.asarray(cloud.return_number)[keep],np.asarray(cloud.number_of_returns)[keep],
                np.asarray(cloud.gps_time)[keep]]))
        data=np.vstack(parts)
        if not len(data) or not np.isfinite(data).all():raise ValueError('finite nonempty site points required')
        _,index=np.unique(data,axis=0,return_index=True);data=data[np.sort(index)]
        origin=np.array([a,b,math.floor(float(data[:,2].min()))]);xyz=(data[:,:3]-origin).astype('f4')
        own=refs[refs.site_id==sid];singles=own[own['class']=='tree'].sort_values('label_id')
        sem,ids,conflicts=assign_candidates(data[:,0],data[:,1],data[:,3],list(singles.geometry),
                list(own[own['class'].isin(['tree_group','uncertain'])].geometry),
                list(own[own['class']=='not_tree'].geometry))
        out=args.output/sid;out.mkdir()
        target=out/(sid+'_point_candidates.ply')
        if not ply.write_ply(str(target),[xyz,sem,ids],['X','Y','Z','semantic_seg','treeID']):raise ValueError('upstream PLY write failed')
        decoded=ply.read_ply(str(target))
        if len(decoded)!=len(data) or not np.array_equal(decoded['treeID'],ids) or not np.array_equal(decoded['semantic_seg'],sem):raise ValueError('PLY label round-trip failed')
        inverse=np.column_stack([decoded[k] for k in ['X','Y','Z']]).astype('f8')+origin
        if np.abs(inverse-data[:,:3]).max()>1e-4:raise ValueError('PLY coordinate round-trip failed')
        mapping={str(i):str(row.label_id) for i,row in enumerate(singles.itertuples())}
        values,counts=np.unique(ids[sem==2],return_counts=True)
        report={'status':'REAL-XYZ WEAK POINT-LABEL CANDIDATE; MANUAL 3D QA REQUIRED',
            'site_id':sid,'split':'train','crs':'EPSG:2193','coordinate_origin_xyz_m':origin.tolist(),
            'points':len(data),'source_records_in_roi':sum(len(p) for p in parts),
            'duplicate_full_return_records_removed':sum(len(p) for p in parts)-len(data),
            'raw_laz_hashes':fingerprints,'reference_sha256':sha256(args.reference),
            'ply_sha256':sha256(target),'upstream_reader_sha256':sha256(reader_path),
            'raw_semantics':'0 unknown, 1 non-tree candidate, 2 tree candidate',
            'raw_tree_ids':'-1 unknown/background; nonnegative individual candidate IDs',
            'candidate_id_to_polygon_id':mapping,'candidate_instance_point_counts':dict(zip(map(str,values),map(int,counts))),
            'reference_single_crowns':len(singles),'candidate_instances_with_points':len(values),
            'reference_singles_without_candidate_points':[mapping[str(i)] for i in range(len(singles)) if i not in values],
            'tree_candidate_points':int((sem==2).sum()),'non_tree_candidate_points':int((sem==1).sum()),
            'unknown_points':int((sem==0).sum()),'ambiguous_overlap_points':int(conflicts.sum()),
            'point_labels_manually_checked':False,'fine_tuning_permitted':False,
            'header_bbox_site_coverage_fraction':psite['header_bbox_union_coverage_fraction'],
            'limitations':['XY crown guidance does not prove 3D tree ownership; neighbouring/understory returns require review',
            'medium/high ASPRS vegetation labels are guidance, not independent per-tree annotation',
            'unknowns and unresolved groups must not become false negative background',
            'stock SAT proposal/mask losses are not certified to ignore incomplete labels',
            'use manually checked complete smaller crops before supervised training']}
        write_json(out/'candidate_record.json',report)
        write_json(out/'qa_template.json',{'site_id':sid,'ply_sha256':sha256(target),
            'coordinate_origin_xyz_m':origin.tolist(),'reviewer':None,'point_labels_manually_checked':False,
            'qa_tasks':['Review ground/structures/understory in XYZ','Correct each instance boundary and ID',
            'Resolve or crop all unknown/group regions; check the trainer loss paths','Export revised PLY; update hash only after real QA']})
        # Full ROI data, not an illustrative mesh; unknown points remain visible.
        points=np.column_stack([np.round(xyz,3),sem,ids,data[:,3].astype('i4')]).tolist()
        write_json(out/'viewer_points.json',{'site_id':sid,'origin_xyz_m':origin.tolist(),'mapping':mapping,'points':points})
        reports.append(report);print(sid,len(data),'real returns;',len(values),'candidate instances;',int((sem==0).sum()),'unknown',flush=True)
    write_json(args.output/'summary.json',{'status':'TRAINING-ONLY 3D POINT CANDIDATES; NOT CHECKED LABELS',
        'reference_frozen':False,'assumed_adjudicated_reference_only':True,
        'reference_sha256':sha256(args.reference),'site_ids':list(sites),'sites':reports,
        'total_candidate_instances_with_points':sum(x['candidate_instances_with_points'] for x in reports),
        'fine_tuning_permitted':False,'model_run_executed':False})
    write_json(args.output/'SHA256_manifest.json',{str(p.relative_to(args.output)):sha256(p) for p in sorted(args.output.rglob('*')) if p.is_file()})

if __name__=='__main__':main()
