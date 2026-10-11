#!/usr/bin/env python3
"""Conservative training-only ALTO-height screening on real benchmark returns.
Abstain on gross height conflicts; never turn synthetic meshes into point truth.
"""
import argparse,csv,html,importlib.util,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point,shape
from shapely import make_valid
from benchmark.common import ROOT,sha256,write_json,site_definitions

POLICY={'terrain_neighbours':8,'maximum_ground_support_distance_m':3.,'maximum_terrain_plane_rmse_m':.25,
        'maximum_terrain_design_condition':100.,'minimum_manual_match_iou':.5,
        'height_tolerance_m':3.,'height_tolerance_fraction':.3,'negative_hag_cutoff_m':-1.,
        'minimum_retained_instance_points':20,'point_labels_manually_checked':False,
        'unknown_promotions_allowed':False,'heldout_inputs_used':False,
        'interpretation':'Height checks withdraw uncertain targets; they do not certify biological membership.'}


def terrain_height(xyz,classification):
    """Local plane of nearby classified ground; reject sparse/unstable support."""
    ground=xyz[classification==2].astype('f8')
    if len(ground)<8:return np.full(len(xyz),np.nan),np.zeros(len(xyz),bool),np.full(len(xyz),np.nan)
    dist,index=cKDTree(ground[:,:2]).query(xyz[:,:2],k=8)
    near=ground[index];dx=near[:,:,:2]-xyz[:,None,:2]
    design=np.concatenate([dx,np.ones((*dx.shape[:2],1))],axis=2)
    u,s,vt=np.linalg.svd(design,full_matrices=False)
    inv=np.divide(1.,s,out=np.zeros_like(s),where=s>1e-9)
    beta=np.einsum('nij,nj,nkj,nk->ni',vt.transpose(0,2,1),inv,u,near[:,:,2])
    fitted=np.einsum('nki,ni->nk',design,beta)
    rmse=np.sqrt(np.mean((fitted-near[:,:,2])**2,axis=1))
    condition=np.divide(s[:,0],s[:,-1],out=np.full(len(s),np.inf),where=s[:,-1]>1e-9)
    # Interpolation only: the target must fall inside the XY bounding box.
    bracket=np.all((dx.min(1)<=0)&(dx.max(1)>=0),axis=1)
    supported=(dist[:,-1]<=POLICY['maximum_ground_support_distance_m'])&(rmse<=POLICY['maximum_terrain_plane_rmse_m'])&(condition<=100)&bracket
    return beta[:,2],supported,rmse


def height_conflict(hag,supported,height):
    if not np.isfinite(height) or height<=0:return np.zeros(len(hag),bool)
    margin=max(POLICY['height_tolerance_m'],POLICY['height_tolerance_fraction']*height)
    return supported&((hag>height+margin)|(hag<POLICY['negative_hag_cutoff_m']))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('screened','candidates','intake','output'):ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args()
    for k in ('screened','candidates','intake','output'):setattr(a,k,getattr(a,k).resolve())
    if a.output.exists():ap.error('fresh output required')
    reader=ROOT.parent/'SegmentAnyTree/torch_points3d/modules/KPConv/plyutils.py'
    spec=importlib.util.spec_from_file_location('ply103',reader);ply=importlib.util.module_from_spec(spec);spec.loader.exec_module(ply)
    matches=json.loads((a.intake/'instance_matches.json').read_text());details=json.loads((a.intake/'records.json').read_text())
    intake=json.loads((a.intake/'intake_receipt.json').read_text())
    if intake['heldout_inputs_used']:raise ValueError('training evidence only')
    original_summary=json.loads((ROOT/'outputs/confirmed_reference_20261010/screened_weak_labels/summary.json').read_text())
    defs=site_definitions();a.output.mkdir(parents=True);labels=a.output/'labels';labels.mkdir()
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_alto_3d_weak_revision'
    rows=list(csv.DictReader((a.screened/'instance_screening.csv').open()));shutil.copy2(a.screened/'instance_screening.csv',labels/'instance_screening.csv')
    write_json(a.output/'policy.json',POLICY);shutil.copy2(Path(__file__),a.output/'executed_source.py')
    edits=[];audits=[];site_records=[];pages=[]
    for path in sorted(a.screened.glob('*/screening_record.json')):
        rec=json.loads(path.read_text());sid=rec['site_id'];assert defs[sid]['split']=='train'
        src=path.parent/(sid+'_screened_weak.ply');assert sha256(src)==rec['screened_ply_sha256']
        raw=ply.read_ply(str(src));xyz=np.column_stack([raw[k] for k in ('X','Y','Z')]);sem=raw['semantic_seg'].astype('i4');ids=raw['treeID'].astype('i4')
        viewer=json.loads((a.candidates/sid/'viewer_points.json').read_text());old=np.asarray(viewer['points'])
        if len(old)!=len(xyz) or np.max(np.abs(old[:,:3]-xyz))>1e-4:raise ValueError('original return/classification alignment changed')
        terrain,supported,rmse=terrain_height(xyz,old[:,5]);hag=xyz[:,2]-terrain
        newsem=sem.copy();newids=ids.copy();selected=np.zeros(len(xyz),bool)
        site_matches=[m for m in matches if m['site_id']==sid];keep=np.unique(ids[sem==2])
        mapdata={str(f['properties']['tree_id']):make_valid(shape(f['geometry'])) for f in json.loads((ROOT/'outputs/alto_review_20261006/comparison'/f'{sid}_public_crowns.geojson').read_text())['features']}
        for m in site_matches:
            ident=m['treeID'];mask=(sem==2)&(ids==ident);record=details.get(m['alto_id'],{});ds=record.get('datasets',{})
            identity=ds.get('identity',{});pc=ds.get('pointcloud',{});crown=ds.get('crown',{})
            height=pc.get('canopy_top_m') or crown.get('crown_max_chm_m')
            height=float(height) if isinstance(height,(int,float)) else None
            previous=ds.get('previous_pointcloud',{});profile=previous.get('foliage_profile')
            if isinstance(profile,str):
                try:profile=json.loads(profile)
                except ValueError:profile=None
            profile_valid=isinstance(profile,list) and len(profile)>0 and all(isinstance(v,(int,float)) and np.isfinite(v) and v>=0 for v in profile)
            unique=sum(x['alto_id']==m['alto_id'] for x in site_matches)==1
            eligible=bool(record and height and unique and (m['label_source']=='ALTO' or (m['geometry_iou'] or 0)>=.5))
            seg=identity.get('seg_key') or '';parts=seg.split(':');anchor=False
            if len(parts)==3:
                try:
                    candidate=json.loads((a.candidates/sid/'candidate_record.json').read_text())
                    anchor=(parts[0]+'.laz' in candidate['raw_laz_hashes']) and mapdata.get(m['alto_id'],Point()).buffer(1.).covers(Point(float(parts[1]),float(parts[2])))
                except (TypeError,ValueError):anchor=False
            eligible=eligible and anchor
            conflict=mask&height_conflict(hag,supported,height) if eligible else np.zeros(len(xyz),bool)
            # Do not keep a tiny supervised instance after withdrawing its outliers.
            if conflict.any() and (mask&~conflict).sum()<20:conflict=mask.copy()
            newsem[conflict]=0;newids[conflict]=-1;selected|=conflict
            h= hag[mask&supported]
            audit={**m,'record_found':bool(record),'alto_canopy_top_m':height,'current_primary_tree_id':identity.get('current_primary_tree_id'),
                'seg_key':seg,'pointcloud_class':pc.get('pointcloud_class'),'one_to_one_auxiliary_match':unique,
                'matching_tile_and_top_anchor':anchor,'height_check_applied':eligible,'terrain_supported_points':int((mask&supported).sum()),
                'assigned_points_before':int(mask.sum()),'assigned_points_after':int((mask&~conflict).sum()),'withdrawn_points':int(conflict.sum()),
                'observed_hag_p99_m':float(np.quantile(h,.99)) if len(h) else None,
                'historical_foliage_profile_available':profile_valid,'historical_foliage_profile':profile if profile_valid else None,
                'historical_canopy_top_m':previous.get('canopy_top_m'),'historical_live_crown_height_m':previous.get('height_to_live_crown_m'),
                'profile_use':'auxiliary visual comparison only; historical spatial basis differs from current crown',
                'same_acquisition_verified':False,'biological_membership_certified':False}
            audits.append(audit)
            for n in np.flatnonzero(conflict):edits.append({'site_id':sid,'point_index_zero_based':int(n),'treeID':int(ident),'X':float(xyz[n,0]),'Y':float(xyz[n,1]),'Z':float(xyz[n,2]),'HAG_estimate_m':float(hag[n]) if supported[n] else None,'alto_canopy_top_m':height,'new_semantic_seg':0,'new_treeID':-1,'reason':'gross Alto-height/terrain conflict; unknown ownership'})
        out=labels/sid;out.mkdir();dst=out/src.name
        ply.write_ply(str(dst),[xyz,newsem,newids],['X','Y','Z','semantic_seg','treeID'])
        check=ply.read_ply(str(dst))
        assert np.array_equal(xyz,np.column_stack([check[k] for k in ('X','Y','Z')]))
        assert np.array_equal(newsem,check['semantic_seg']) and np.array_equal(newids,check['treeID'])
        assert np.array_equal(newsem[~selected],sem[~selected]) and np.array_equal(newids[~selected],ids[~selected])
        assert np.all(newids[newsem!=2]==-1)
        mapping={r['treeID']:r['label_id'] for r in rows if r['site_id']==sid and r['retained']=='True'}
        rec.update(screened_ply_sha256=sha256(dst),source_partial_review_ply_sha256=sha256(src),run_id=run_id,
            retained_tree_points=int((newsem==2).sum()),unknown_points=int((newsem==0).sum()),alto_height_abstentions=int(selected.sum()),
            retained_original_ids=sum(bool((newids==int(r['treeID'])).any()) for r in rows if r['site_id']==sid and r['retained']=='True' and r['label_source']!='ALTO'),
            retained_alto_ids=sum(bool((newids==int(r['treeID'])).any()) for r in rows if r['site_id']==sid and r['retained']=='True' and r['label_source']=='ALTO'),
            point_labels_manually_checked=False,assistant_visual_review_pending=True,stock_supervised_fine_tuning_permitted=False,
            label_type='Alto numeric 3D screened weak candidates; unknown-masked pilot only')
        write_json(out/'screening_record.json',rec);site_records.append(rec)
        vo=a.output/'viewer'/sid;vo.mkdir(parents=True);shutil.copy2(dst,vo/dst.name)
        write_json(vo/'viewer_points.json',{'site_id':sid,'origin_xyz_m':viewer['origin_xyz_m'],'mapping':mapping,'points':np.column_stack([xyz,newsem,newids,old[:,5]]).tolist()})
        write_json(vo/'candidate_record.json',{'site_id':sid,'screened_weak':True,'points':len(xyz),'candidate_instances_with_points':len(np.unique(newids[newsem==2])),
             'candidate_instance_point_counts':{str(i):int(((newsem==2)&(newids==i)).sum()) for i in keep},'unknown_points':int((newsem==0).sum()),'ply_file':dst.name,
             'review_note':'Alto numeric-height assisted weak labels; assistant review, no independent point certification.', 'display_status':'Revised training version; exploratory unknown-masked adaptation'})
        cols=np.full((len(xyz),4),[.65,.68,.7,.2]);cols[newsem==1]=[.5,.3,.15,.35]
        for i in keep:cols[(newsem==2)&(newids==i)]=[*matplotlib.colors.hsv_to_rgb(((i*.381966)%1,.7,.75)),.85]
        fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
        origin=np.asarray(viewer['origin_xyz_m']);bb=defs[sid]['bbox_epsg2193'];extent=(bb[0]-origin[0],bb[2]-origin[0],bb[1]-origin[1],bb[3]-origin[1])
        axes[0,0].imshow(plt.imread(ROOT/'data/sites/chips'/sid/'rgb.png'),extent=extent)
        axes[0,0].scatter(xyz[newsem==2,0],xyz[newsem==2,1],s=2,c=cols[newsem==2]);axes[0,0].set_title('RGB with retained ownership candidates')
        for ax,pair,title in [(axes[0,1],(0,1),'Top view'),(axes[1,0],(0,2),'East / local Z'),(axes[1,1],(1,2),'North / local Z')]:
            for target in (False,True):
                mask=(newsem==2) if target else newsem!=2;ax.scatter(xyz[mask,pair[0]],xyz[mask,pair[1]],s=1.5,c=cols[mask],rasterized=True)
            if selected.any():ax.scatter(xyz[selected,pair[0]],xyz[selected,pair[1]],s=12,c='red',marker='x')
            ax.set_title(title);ax.grid(alpha=.2)
        fig.suptitle(f'{sid}: real returns, colour = weak tree ID; grey = unknown; red = new abstentions\n{int(selected.sum())} additional assignments withdrawn; Alto height checks apply only to supported matches')
        fig.savefig(a.output/f'{sid}_overview.png',dpi=150);plt.close(fig)
        site_aud=[r for r in audits if r['site_id']==sid]
        table=''.join(f'<tr><td>{r["treeID"]}</td><td>{html.escape(str(r["alto_id"]))}</td><td>{r["alto_canopy_top_m"]}</td><td>{str(r["observed_hag_p99_m"])[:6]}</td><td>{r["height_check_applied"]}</td><td>{r["withdrawn_points"]}</td></tr>' for r in site_aud)
        pages.append(f'<section id="{sid}"><h2>{sid}</h2><img src="{sid}_overview.png"><p><a href="../../labeller/point_review.html?data=../outputs/alto_3d_weak_revision_20261011/revised/viewer/&amp;site={sid}">Rotate actual returns</a></p><details><summary>Per-instance Alto height audit</summary><table><tr><th>ID</th><th>Alto source</th><th>Alto top (m)</th><th>LAZ HAG p99</th><th>Eligible</th><th>Withdrawn</th></tr>{table}</table></details></section>')
        print(sid,int(selected.sum()),'new abstentions',flush=True)
    write_json(a.output/'instance_3d_audit.json',audits)
    with (a.output/'point_edits.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['site_id','point_index_zero_based','treeID','X','Y','Z','HAG_estimate_m','alto_canopy_top_m','new_semantic_seg','new_treeID','reason']);w.writeheader();w.writerows(edits)
    summary={**original_summary,'run_id':run_id,'sites':site_records,'total_retained_tree_points':sum(r['retained_tree_points'] for r in site_records),
       'total_retained_original_ids':sum(r['retained_original_ids'] for r in site_records),'total_retained_alto_ids':sum(r['retained_alto_ids'] for r in site_records),
       'status':'ALTO HEIGHT SCREENED WEAK CANDIDATES; EXPLORATORY PILOT ONLY','new_alto_height_abstentions':len(edits),
       'source_partial_review_receipt_sha256':sha256(a.screened.parent/'receipt.json'),'alto_intake_receipt_sha256':sha256(a.intake/'intake_receipt.json'),
       'unknown_masked_weak_pilot_authorized':True,'stock_supervised_training_authorized':False,'heldout_used':False}
    write_json(labels/'summary.json',summary)
    write_json(labels/'SHA256_manifest.json',{str(p.relative_to(labels)):sha256(p) for p in labels.rglob('*') if p.is_file()})
    receipt={'run_id':run_id,'additional_abstentions':len(edits),'prior_visual_abstentions_preserved':121,'total_tree_points':summary['total_retained_tree_points'],
       'total_instance_ids':summary['total_retained_original_ids']+summary['total_retained_alto_ids'],
       'instances_with_numeric_height_checks':sum(r['height_check_applied'] for r in audits),'terrain_supported_assigned_returns':sum(r['terrain_supported_points'] for r in audits),
       'coordinate_and_label_roundtrips_verified':True,'unselected_labels_preserved':True,'point_labels_manually_checked':False,
       'heldout_used_for_label_revision':False,'source_snapshots':{'release':intake['release'],'detail_schema':intake['detail_schema_snapshot']},
       'same_acquisition_verified':False,'profile_available':intake['schema_contains_foliage_profile'],
       'prior_experiment_unchanged':True,'script_sha256':sha256(Path(__file__))}
    write_json(a.output/'receipt.json',receipt)
    (a.output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Alto-assisted 3D weak-label review</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto}img{width:100%}td,th{padding:5px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>Alto-assisted 3D weak-label review</h1><p>Real benchmark XYZ are unchanged. Numeric Alto canopy heights provide auxiliary checks after local terrain estimation. No mesh vertices, synthetic heights or held-out labels are used. Grey points remain unknown and ignored by target losses. This is assistant-assisted review, not independent human certification. Current release is V5.2; the detail snapshot is dated 9 October. Historical foliage profiles retain their earlier spatial basis. Unmatched or ambiguous records cannot establish ownership.</p><p>'+html.escape(json.dumps(receipt))+'</p>'+''.join(pages)+'</html>')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
