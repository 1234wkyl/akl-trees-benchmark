#!/usr/bin/env python
"""Validate checked upstream-format PLY labels; never derive GT from crowns."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from benchmark.common import ROOT, sha256, site_definitions, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ply',type=Path,required=True)
    ap.add_argument('--qa-record',type=Path,required=True)
    ap.add_argument('--source',type=Path,default=ROOT.parent/'SegmentAnyTree')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('use a fresh output file')
    qa=json.loads(args.qa_record.read_text());digest=sha256(args.ply)
    if qa.get('point_labels_manually_checked') is not True or not qa.get('reviewer') or qa.get('ply_sha256')!=digest:
        raise ValueError('require recorded manual point-level checking of this exact PLY')
    site=site_definitions().get(qa.get('site_id'))
    if site is None or site['split']!='train':raise ValueError('fine-tuning inputs must come from a training site')
    reader_path=args.source/'torch_points3d/modules/KPConv/plyutils.py'
    spec=importlib.util.spec_from_file_location('sat_ply_reader',reader_path)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    data=reader.read_ply(str(args.ply));fields=set(data.dtype.names)
    coord=['X','Y','Z'] if {'X','Y','Z'}<=fields else ['x','y','z']
    if not set(coord+['semantic_seg','treeID'])<=fields:raise ValueError('require XYZ, semantic_seg and treeID, not ASPRS classification alone')
    xyz=np.column_stack([data[k] for k in coord]).astype('f8')
    origin=np.asarray(qa.get('coordinate_origin_xyz_m'),dtype='f8')
    if origin.shape!=(3,) or not np.isfinite(origin).all() or not len(xyz) or not np.isfinite(xyz).all():raise ValueError('finite XYZ and recorded local geographic origin required')
    world=xyz+origin;x0,y0,x1,y1=site['bbox_epsg2193']
    if not ((world[:,0]>=x0)&(world[:,0]<x1)&(world[:,1]>=y0)&(world[:,1]<y1)).all():raise ValueError('labelled PLY must be cropped to the declared training site')
    sem=data['semantic_seg'];ids=data['treeID']
    if not np.isfinite(sem).all() or not np.isfinite(ids).all() or not (sem==np.floor(sem)).all() or not (ids==np.floor(ids)).all():raise ValueError('labels must be finite integer values')
    if not set(map(int,np.unique(sem)))<={0,1,2}:raise ValueError('upstream raw semantic labels: 0 unknown, 1 non-tree, 2 tree; loader subtracts 1')
    if (ids[sem==2]<0).any() or (ids[sem!=2]!=-1).any():raise ValueError('tree IDs start at 0; non-tree/unknown raw treeID=-1 (loader adds 1)')
    if not (sem==2).any():raise ValueError('no checked tree points for fine-tuning')
    if ids.dtype.kind=='f' and ids.max()>2**24:raise ValueError('floating PLY ID risks float32 integer precision loss')
    values,counts=np.unique(ids[sem==2],return_counts=True)
    report={'status':'format and manual-QA-record checks passed, training launch not validated',
            'ply_sha256':digest,'qa_record_sha256':sha256(args.qa_record),'reader_sha256':sha256(reader_path),
            'site_id':site['site_id'],'split':'train','points':len(data),
            'tree_points':int((sem==2).sum()),'tree_instances':len(values),
            'instance_point_counts':dict(zip(map(str,values.astype('int64')),map(int,counts))),
            'raw_semantics':'0 unknown, 1 non-tree, 2 tree',
            'raw_instance_ids':'-1 background/unknown, nonnegative tree IDs',
            'reviewer':qa['reviewer'],'labels_generated_from_crowns':False,
            'note':'This checks consistency, not biological correctness or training quality. Original QA must be genuine; no fine-tuning is launched.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,report)
    print(f'Checked training format: {len(data)} points, {len(values)} instances')


if __name__=='__main__':main()
