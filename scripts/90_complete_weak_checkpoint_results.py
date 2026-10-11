#!/usr/bin/env python3
"""Merge audited fixed weak predictions and score against the author lock.

Reuse the exact duplicate-XYZ merge kernel and existing fixed hull/scorer.
This is a disclosed post-revision experiment, not a retrospective blind test.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from benchmark.common import ROOT,sha256,site_definitions,write_json
from benchmark.evaluation import aggregate,paired_site_bootstrap,score_site


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('base','reference','reference-lock','pretrained-assembly','classical-base','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--current-classical-base',type=Path,
        default=ROOT/'outputs/nectar/confirmed_classical_h3_20261010',
        help='current train-only 3.0m configuration; previous 3.5m is retained as a sensitivity comparator')
    a=p.parse_args()
    if a.output.exists():p.error('fresh output required')
    queue=json.loads((a.base/'queue_receipt.json').read_text())
    lock=json.loads(a.reference_lock.read_text())
    if (queue['status']!='COMPLETED' or len(queue['tiles'])!=19
        or any(r['status']!='AUDITED' for r in queue['tiles'])
        or lock['status']!='author_locked' or lock['reference_gpkg_sha256']!=sha256(a.reference)):
        raise ValueError('complete audited fixed-model queue and matching author-derived lock required')
    current=json.loads((a.current_classical_base/'prediction_receipt.json').read_text())
    current_lock=ROOT/'outputs/preparation/confirmed_classical_choice_20261010/model_lock.json'
    if (current['author_model_lock_sha256']!=sha256(current_lock)
        or current['parameters']['crown_minimum_height_m']!=3.
        or not current['prior_test_exposure_disclosed']):
        raise ValueError('completed current train-only classical configuration required')
    defs=site_definitions();plan=json.loads((ROOT/'docs/project_status_2026-09-18/tile_coverage_preflight/coverage_plan.json').read_text())
    if plan['site_definitions_sha256']!=sha256(ROOT/'data/sites/reference_sites.json'):raise ValueError('coverage plan changed')
    tiles={r['tile']:r for r in queue['tiles']};meta={r['tile']:r for r in plan['tiles']}
    merge=load('merge90',ROOT/'scripts/46_merge_training_site_tiles.py')
    ev=load('ev90',ROOT/'scripts/12_evaluate.py')
    a.output.mkdir(parents=True);parts=[];site_receipts=[]
    for row in plan['sites']:
        sid=row['site_id'];site=defs[sid]
        if site['split']=='train':continue
        region=box(*site['bbox_epsg2193']);context=region.buffer(10.,cap_style=3).bounds
        records=[]
        for tile in sorted(row['required_tiles']):
            r=tiles[tile]
            for key in ('reconciled_prediction','coordinate_audit','input'):
                if sha256(Path(r[key]))!=r[key+'_sha256']:raise ValueError('queue artifact changed: '+key)
            records.append(merge._load_tile(tile,Path(r['reconciled_prediction']),Path(r['coordinate_audit']),
                meta[tile]['bbox_epsg2193'],context,r['input']))
        merged=merge.merge_tiles(records);dest=a.output/sid;dest.mkdir()
        target=dest/(sid+'_merged_points.laz');merge._write_cloud(target,merged)
        audit=dest/'merged_coordinate_audit.json'
        write_json(audit,{'prediction_sha256':sha256(target),'epsg2193_coordinate_consistency':True,
            'uint16_boundary_warning':False,'basis':'Hash-verified bidirectional per-tile coordinate audits; exact existing duplicate-XYZ merge kernel',
            'queue_receipt_sha256':sha256(a.base/'queue_receipt.json'),'context_m':10.,'source_tiles':row['required_tiles'],
            'merge_diagnostics':{k:v for k,v in merged.items() if k not in {'xyz','instance','semantic','source_tile'}}})
        subprocess.run([sys.executable,str(ROOT/'scripts/20_sat_crown_candidates.py'),
            '--prediction',str(target),'--coordinate-audit',str(audit),'--tree-semantic-values','1',
            '--hull-method','convex_hull','--buffer-m','1.0','--output',str(dest/'crowns')],check=True,cwd=ROOT)
        subprocess.run([sys.executable,str(ROOT/'scripts/16_standardize_predictions.py'),
            '--input',str(dest/'crowns/candidate_crowns.gpkg'),'--layer','candidate_crowns',
            '--id-column','prediction_id','--split',site['split'],'--site-ids',sid,
            '--output',str(dest/'standardized')],check=True,cwd=ROOT)
        path=dest/'standardized/predictions.gpkg';parts.append(gpd.read_file(path,layer='predictions'))
        site_receipts.append({'site_id':sid,'split':site['split'],'prediction_sha256':sha256(path),'merged_points_sha256':sha256(target)})
        print('Assembled',sid,flush=True)
    joined=gpd.GeoDataFrame(pd.concat(parts,ignore_index=True),geometry='geometry',crs=2193)
    frames={k:gpd.read_file(a.reference,layer=k) for k in ('single_trees','tree_groups','uncertainty_regions')}
    policy=json.loads((ROOT/'config/evaluation_draft.json').read_text())
    if (policy['match_tolerance_m']!=2. or policy['minimum_match_iou']!=0.
        or policy['prediction_ignore_rule']!='unmatched_centroid_in_group_or_uncertain'):raise ValueError('fixed scorer policy changed')
    for split in ('val','test'):
        path=a.output/f'{split}_predictions.gpkg'
        joined[joined.split==split].to_file(path,layer='predictions',driver='GPKG')
        sites={s:v for s,v in defs.items() if v['split']==split}
        references={k:{s:[(str(r.label_id),r.geometry) for r in f[f.site_id==s].itertuples()] for s in sites} for k,f in frames.items()}
        def score(pred_path):
            predictions,_=ev.load_polygons(pred_path,'predictions',sites,'prediction_id')
            return {s:score_site(references['single_trees'][s],predictions[s],
                [g for _,g in references['tree_groups'][s]],[g for _,g in references['uncertainty_regions'][s]],
                2.,0.,policy['prediction_ignore_rule']) for s in sites}
        weak=score(path);pre=score(a.pretrained_assembly/f'{split}_predictions.gpkg')
        previous_path=a.classical_base/f'{split}_standardized/predictions.gpkg';previous=score(previous_path)
        classical_path=a.current_classical_base/f'{split}_standardized/predictions.gpkg'
        standard=json.loads((classical_path.parent/'provenance.json').read_text())
        if (standard['output_sha256']!=sha256(classical_path) or standard['input_sha256']!=current['crowns_sha256']):
            raise ValueError('current classical standardized source changed')
        classical=score(classical_path)
        write_json(a.output/f'{split}_comparison.json',{
            'scope':'POST-REVISION EXPLORATORY WEAK MODEL; PRIOR TEST EXPOSURE DISCLOSED',
            'author_reference_locked':True,'supervisor_freeze_claimed':False,
            'reference_sha256':sha256(a.reference),'checkpoint_sha256':queue['checkpoint_sha256'],
            'policy_sha256':sha256(ROOT/'config/evaluation_draft.json'),
            'weak_summary':aggregate(list(weak.values())),'pretrained_summary':aggregate(list(pre.values())),
            'classical_summary':aggregate(list(classical.values())),
            'previous_classical_3p5_summary':aggregate(list(previous.values())),
            'classical_crown_minimum_height_m':3.,
            'weak_by_site':weak,'pretrained_by_site':pre,'classical_by_site':classical,
            'weak_vs_pretrained_bootstrap':paired_site_bootstrap(weak,pre,2000,42),
            'weak_vs_classical_bootstrap':paired_site_bootstrap(weak,classical,2000,42),
            'weak_by_stratum':{stratum:aggregate([v for s,v in weak.items() if sites[s]['stratum']==stratum]) for stratum in dict.fromkeys(s['stratum'] for s in sites.values())},
            'prediction_sha256':{'weak':sha256(path),'pretrained':sha256(a.pretrained_assembly/f'{split}_predictions.gpkg'),'classical':sha256(classical_path)}})
    write_json(a.output/'receipt.json',{'run_id':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_weak_model_assembly_score',
        'status':'COMPLETE','sites':site_receipts,'queue_receipt_sha256':sha256(a.base/'queue_receipt.json'),
        'reference_lock_sha256':sha256(a.reference_lock),'reference_sha256':sha256(a.reference),
        'checkpoint_sha256':queue['checkpoint_sha256'],'source_code_sha256':sha256(Path(__file__)),
        'model_or_policy_retuned':False,'context_m':10.,'hull_method':'convex_hull','buffer_m':1.,
        'producer_kernel_sha256':{s:sha256(ROOT/'scripts'/s) for s in ['46_merge_training_site_tiles.py','20_sat_crown_candidates.py','16_standardize_predictions.py','12_evaluate.py']}})
    subprocess.run([sys.executable,str(ROOT/'scripts/93_report_weak_comparison.py'),
        '--comparisons',str(a.output),'--queue',str(a.base),
        '--classical',str(a.current_classical_base),'--output',str(a.output/'report')],check=True,cwd=ROOT)
    # The existing two-panel renderer accepts a named SAT model; retain the
    # statistical comparison hash in this display-only schema adapter.
    comparison=json.loads((a.output/'test_comparison.json').read_text())
    gallery_input=a.output/'test_gallery_input.json'
    write_json(gallery_input,{'reference_frozen':False,'actual_independent_agreement_available':True,
        'site_ids':list(comparison['weak_by_site']),
        'pretrained_by_site':comparison['weak_by_site'],
        'classical_by_site':comparison['classical_by_site'],
        'sources':{'reference':comparison['reference_sha256'],
                   'sat':comparison['prediction_sha256']['weak'],
                   'classical':comparison['prediction_sha256']['classical']},
        'display_model':'SegmentAnyTree weak pilot',
        'original_comparison_sha256':sha256(a.output/'test_comparison.json'),
        'purpose':'Display adapter only; no metric or model changes'})
    subprocess.run([sys.executable,str(ROOT/'scripts/64_build_conditional_error_gallery.py'),
        '--comparison',str(gallery_input),'--reference',str(a.reference),
        '--sat-predictions',str(a.output/'test_predictions.gpkg'),
        '--classical-predictions',str(a.current_classical_base/'test_standardized/predictions.gpkg'),
        '--author-lock',str(a.reference_lock),'--classical-label','Classical training-selected (3.0 m)',
        '--sat-label','SegmentAnyTree weak pilot','--output',str(a.output/'test_error_review')],check=True,cwd=ROOT)
    write_json(a.output/'SHA256_manifest.json',{str(f.relative_to(a.output)):sha256(f) for f in a.output.rglob('*') if f.is_file()})
    print('All 16 held-out sites assembled and conditional weak-model comparisons complete',flush=True)


if __name__=='__main__':main()
