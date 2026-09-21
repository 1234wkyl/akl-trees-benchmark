#!/usr/bin/env python
"""Archive an original teacher JSON, compute agreement, prepare adjudication.

Usage: python scripts/31_receive_independent_labels.py --tim-json RETURNED.json
Never writes canonical labels or grants reference freeze. Test mode is explicit.
"""
import argparse
import csv
import html
import importlib.util
import importlib.metadata
import json
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image,ImageDraw
from shapely.geometry import Polygon,box

from benchmark.common import ROOT,sha256,site_definitions,write_json

spec=importlib.util.spec_from_file_location('agreement',Path(__file__).with_name('05_interrater_agreement.py'))
agreement=importlib.util.module_from_spec(spec);spec.loader.exec_module(agreement)
POLICY=ROOT/'config/independent_agreement.json'


def audit_json(path,who,sites,required,margin=5):
    """Check provenance fields and XY mapping before any correspondence score."""
    data=json.loads(path.read_text())
    frame,complete=agreement.load_independent_json(path,who,sites)
    if not set(required)<=complete:raise ValueError('Both required sites must explicitly be complete: '+', '.join(sorted(set(required)-complete)))
    rows=[]
    for sid in required:
        x0,y0,x1,y1=sites[sid]['bbox_epsg2193'];roi=box(x0,y0,x1,y1)
        for lab in data['sites'][sid]['labels']:
            coords=np.asarray(lab['world'],dtype=float);raw=Polygon(coords)
            geom=raw if raw.is_valid else raw.buffer(0)
            if not roi.buffer(margin).covers(geom):raise ValueError(f'{sid}/{lab["id"]}: coordinates exceed site guard; check CRS or site ID')
            if geom.intersection(roi).area<=0:raise ValueError(f'{sid}/{lab["id"]}: polygon has no positive-area overlap with its site')
            px_error=None
            if 'px' in lab:
                px=np.asarray(lab['px'],dtype=float)
                if px.shape!=coords.shape or not np.isfinite(px).all():raise ValueError(f'{lab["id"]}: inconsistent px/world shape or nonfinite coordinates')
                expected=np.column_stack([x0+(px[:,0]+.5)*(x1-x0)/400,y1-(px[:,1]+.5)*(y1-y0)/400])
                px_error=float(np.max(np.abs(expected-coords)))
                if px_error>.001:raise ValueError(f'{lab["id"]}: px/world mapping differs by {px_error:.4f} m')
            outside=geom.difference(roi).area
            rows.append({'rater':who,'site_id':sid,'label_id':lab['id'],'class':lab['cls'],
                'raw_valid':raw.is_valid,'geometry_repaired':not raw.is_valid,
                'raw_area_m2':raw.area,'used_area_m2':geom.area,'repair_area_difference_m2':geom.area-raw.area,
                'outside_site_area_m2':outside,'px_world_max_error_m':px_error,
                'requires_geometry_review':not raw.is_valid or outside>.01})
    return frame[frame.site_id.isin(required)].copy(),rows,sorted(complete-set(required))


def write_csv(path,rows,columns):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def draw_geometry(draw,geom,bounds,offset,scale,color):
    x0,y0,x1,y1=bounds
    parts=[geom] if geom.geom_type=='Polygon' else geom.geoms
    for p in parts:
        for ring in [p.exterior,*p.interiors]:
            pts=[(offset[0]+((x-x0)*400/(x1-x0)-.5)*scale,offset[1]+((y1-y)*400/(y1-y0)-.5)*scale) for x,y in ring.coords]
            draw.line(pts,fill=color,width=2)


def build_gallery(out,a,b,pairs,sites,b_name,synthetic=False):
    gallery=out/'visual_review';gallery.mkdir()
    index_a={r.label_id:r for _,r in a.iterrows()};index_b={r.label_id:r for _,r in b.iterrows()}
    cards=[];banner='SYNTHETIC TEST ONLY | ' if synthetic else ''
    for sid in sites:
        canvas=Image.new('RGB',(840,870),'white');dr=ImageDraw.Draw(canvas)
        for k,(rater,frame) in enumerate([('XZ',a),(b_name,b)]):
            dr.text((8,8+435*k),banner+sid+' '+rater+' | RGB / CHM',fill='black')
            for j,name in enumerate(['rgb','chm']):
                im=Image.open(ROOT/'data/sites/chips'/sid/(name+'.png')).convert('RGB')
                di=ImageDraw.Draw(im)
                for _,r in frame[frame.site_id==sid].iterrows():
                    draw_geometry(di,r.geometry,sites[sid]['bbox_epsg2193'],(0,0),1,'#00ffff' if k==0 else '#ff00ff')
                canvas.paste(im,(8+420*j,28+435*k))
        canvas.save(gallery/(sid+'_overview.png'))
        cards.append(f'<article><h2>{sid} independent originals</h2><img src="{sid}_overview.png"></article>')
    for n,case in enumerate(pairs,1):
        sid=case['site_id'];geoms=[]
        for key,lookup,color in [('id_a',index_a,'#00ffff'),('id_b',index_b,'#ff00ff')]:
            if case[key]:geoms.append((lookup[case[key]].geometry,color))
        bounds=sites[sid]['bbox_epsg2193'];x0,y0,x1,y1=bounds
        xs=[];ys=[]
        for geom,_ in geoms:
            gx0,gy0,gx1,gy1=geom.bounds
            xs.extend([(gx0-x0)*4-.5,(gx1-x0)*4-.5]);ys.extend([(y1-gy1)*4-.5,(y1-gy0)*4-.5])
        crop=(max(0,int(min(xs))-10),max(0,int(min(ys))-10),min(400,int(max(xs))+11),min(400,int(max(ys))+11))
        canvas=Image.new('RGB',(840,500),'white');dr=ImageDraw.Draw(canvas)
        dr.text((8,8),banner+f'{n:03} {sid} {case["status"]} | cyan XZ, magenta {b_name}',fill='black')
        dr.text((8,465),'A '+case['id_a']+' | B '+case['id_b'],fill='black')
        for j,name in enumerate(['rgb','chm']):
            im=Image.open(ROOT/'data/sites/chips'/sid/(name+'.png')).convert('RGB');di=ImageDraw.Draw(im)
            for geom,color in geoms:draw_geometry(di,geom,bounds,(0,0),1,color)
            im=im.crop(crop);scale=min(400/im.width,400/im.height)
            im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST)
            canvas.paste(im,(8+420*j,35))
        name=f'case_{n:03}.png';canvas.save(gallery/name)
        cards.append(f'<article><h3>case_{n:03} {html.escape(case["status"])}</h3><img src="{name}"><p>{html.escape(case["id_a"])} / {html.escape(case["id_b"])}; IoU {case["iou"]}</p></article>')
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Independent annotation adjudication</title><style>body{font:16px system-ui;max-width:880px;margin:25px auto}img{max-width:100%}article{border-top:1px solid #aaa}</style><h1>Original independent comparison</h1><p>Cyan: original XZ. Magenta: original teacher. No predictions or refined XZ labels. All cases need manual review; images do not grant adjudication or freeze.</p>'+''.join(cards))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tim-json',type=Path,required=True)
    ap.add_argument('--tim-labeller',help='Defaults to labeller recorded in original returned JSON; never silently rename it')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--synthetic-test',action='store_true',help='Explicitly label all outputs synthetic; never paper results')
    args=ap.parse_args()
    if not args.tim_json.is_file():ap.error('Returned JSON does not exist')
    policy=json.loads(POLICY.read_text());xz=ROOT/policy['original_XZ_path'];sites=site_definitions();required=policy['sites']
    teacher_hash=sha256(args.tim_json)
    out=args.output or ROOT/'outputs/independent_agreement'/('TW_'+teacher_hash[:12]+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if out.exists():ap.error('Existing results preserved; use a fresh output directory')
    out.mkdir(parents=True);raw=out/'originals';raw.mkdir()
    source_a=raw/'XZ_original.json';source_b=raw/'teacher_original.json'
    shutil.copyfile(xz,source_a);shutil.copyfile(args.tim_json,source_b)
    source_a.chmod(0o444);source_b.chmod(0o444)
    shutil.copyfile(POLICY,out/'agreement_policy.json');shutil.copyfile(ROOT/'data/sites/reference_sites.json',out/'site_definitions.json')
    receipt={'status':'received_not_yet_validated','synthetic_only':args.synthetic_test,'received_at':datetime.now(timezone.utc).isoformat(),
        'teacher_input_name':args.tim_json.name,'XZ_sha256':sha256(source_a),'teacher_sha256':sha256(source_b),
        'policy_sha256':sha256(POLICY),'site_definitions_sha256':sha256(ROOT/'data/sites/reference_sites.json'),
        'sites':required,'source_reference_written':False,'reference_frozen':False,'manual_adjudication_required':True,
        'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'scripts/05_interrater_agreement.py',ROOT/'scripts/benchmark/common.py']},
        'python_version':sys.version,
        'library_versions':{name:importlib.metadata.version(name) for name in ['numpy','geopandas','shapely','pyogrio','Pillow']}}
    try:
        if receipt['XZ_sha256']!=policy['original_XZ_sha256']:raise ValueError('Original XZ differs from blind-comparison source; do not substitute the refinement revision')
        data=json.loads(source_b.read_text())
        if not isinstance(data,dict):raise ValueError('Teacher export must be a JSON object with a sites map')
        who=args.tim_labeller or data.get('labeller')
        if not isinstance(who,str) or not who or who=='XZ':raise ValueError('Teacher export must identify a distinct labeller; use its recorded labeller name')
        if not args.synthetic_test and ('synthetic' in args.tim_json.name.lower() or data.get('synthetic') or 'demo_labels' in args.tim_json.parts):raise ValueError('Synthetic/demo labels cannot be ingested as teacher results')
        a,audit_a,extra_a=audit_json(source_a,'XZ',sites,required,policy['coordinate_guard_margin_m'])
        b,audit_b,extra_b=audit_json(source_b,who,sites,required,policy['coordinate_guard_margin_m'])
        rows=audit_a+audit_b
        columns=['rater','site_id','label_id','class','raw_valid','geometry_repaired','raw_area_m2','used_area_m2','repair_area_difference_m2','outside_site_area_m2','px_world_max_error_m','requires_geometry_review']
        write_csv(out/'geometry_audit.csv',rows,columns)
        a.to_file(out/'original_comparison.gpkg',layer='original_XZ',driver='GPKG')
        b.to_file(out/'original_comparison.gpkg',layer='original_teacher',driver='GPKG')
        cmd=[sys.executable,str(ROOT/'scripts/05_interrater_agreement.py'),'--a','XZ','--b',who,'--json-a',str(source_a),'--json-b',str(source_b),'--sites',*required,'--iou',str(policy['iou_threshold']),'--output',str(out/'agreement.md')]
        process=subprocess.run(cmd,capture_output=True,text=True)
        (out/'calculation.log').write_text(process.stdout+process.stderr)
        if process.returncode:raise ValueError('Agreement calculation failed; see calculation.log')
        metrics=json.loads((out/'agreement_provenance.json').read_text())
        metrics.update(synthetic_only=args.synthetic_test,reference_frozen=False,
            individual_tree_view=metrics['per_class'].get('tree'),limitations=policy['limitations'])
        write_json(out/'agreement_provenance.json',metrics)
        write_json(out/'metrics.json',metrics)
        with (out/'agreement_pairs.csv').open() as f:pairs=list(csv.DictReader(f))
        pending=[]
        for n,c in enumerate(pairs,1):
            pending.append({'case_id':f'case_{n:03}',**c,'decision':'pending','final_class':'','chosen_geometry':'','reason':'','reviewer':'','reviewed_at':''})
        write_csv(out/'adjudication_queue.csv',pending,['case_id','site_id','status','id_a','id_b','class_a','class_b','iou','decision','final_class','chosen_geometry','reason','reviewer','reviewed_at'])
        links=[]
        for sid in required:
            for _,ra in a[a.site_id==sid].iterrows():
                for _,rb in b[b.site_id==sid].iterrows():
                    overlap=ra.geometry.intersection(rb.geometry).area/min(ra.geometry.area,rb.geometry.area)
                    if overlap>=policy['overlap_diagnostic_threshold']:links.append({'site_id':sid,'id_a':ra.label_id,'id_b':rb.label_id,'class_a':ra['class'],'class_b':rb['class'],'intersection_over_smaller_area':overlap})
        write_csv(out/'overlap_links.csv',links,['site_id','id_a','id_b','class_a','class_b','intersection_over_smaller_area'])
        build_gallery(out,a,b,pairs,{sid:sites[sid] for sid in required},who,args.synthetic_test)
        receipt.update(status='SYNTHETIC_TEST_ONLY' if args.synthetic_test else 'original_agreement_computed_pending_adjudication',
            teacher_labeller=who,ignored_completed_XZ_sites=extra_a,ignored_completed_teacher_sites=extra_b,
            n_XZ=len(a),n_teacher=len(b),review_cases=len(pairs),geometry_review_flags=sum(r['requires_geometry_review'] for r in rows),
            source_originals_byte_verified=sha256(source_a)==sha256(xz) and sha256(source_b)==teacher_hash,
            no_model_performance_computed=True)
        if args.synthetic_test:
            (out/'agreement.md').write_text('**SYNTHETIC TEST ONLY — DO NOT REPORT AS TEACHER AGREEMENT.**\n\n'+(out/'agreement.md').read_text())
            gallery=out/'visual_review/index.html'
            gallery.write_text(gallery.read_text().replace('<h1>Original independent comparison</h1>',"<h1>SYNTHETIC TEST ONLY — NOT TIM'S LABELS</h1>"))
        write_json(out/'receipt.json',receipt)
        files={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
        write_json(out/'SHA256_manifest.json',files)
        print(json.dumps({'output':str(out.resolve()),'status':receipt['status'],'tree_agreement':metrics['individual_tree_view'],'manual_review':str((out/'visual_review/index.html').resolve())},ensure_ascii=False))
        return 0
    except (ValueError,KeyError,TypeError) as e:
        receipt.update(status='validation_or_calculation_failed',error=str(e));write_json(out/'receipt.json',receipt)
        print(f'Original files retained in {raw}; no freeze or label edits. Error: {e}',file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
