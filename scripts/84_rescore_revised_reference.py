#!/usr/bin/env python3
"""Rescore immutable completed predictions against a new, unfrozen reference.

Producer receipts retain their original scenario binding. A distinct analysis
receipt binds the revised reference; it never rewrites producer provenance.
"""
import argparse,csv,importlib.util,json
from datetime import datetime,timezone
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
from benchmark.common import ROOT,sha256,site_definitions,write_json
from benchmark.evaluation import aggregate,paired_site_bootstrap,score_site

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('reference','density-root','comparator-root','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--current-classical',type=Path,help='separate current training-selected classical producer')
    p.add_argument('--current-classical-lock',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh analysis directory required')
    policy_path=ROOT/'config/evaluation_draft.json';policy=json.loads(policy_path.read_text())
    if (policy['match_tolerance_m']!=2 or policy['minimum_match_iou']!=0 or policy['prediction_ignore_rule']!='unmatched_centroid_in_group_or_uncertain' or policy['bootstrap_iterations']!=2000 or policy['random_seed']!=42):raise ValueError('Fixed benchmark policy required')
    prov=json.loads((a.reference.parent/'provenance.json').read_text())
    if sha256(a.reference)!=prov['output_gpkg_sha256']:raise ValueError('Reference receipt mismatch')
    scenario_path=ROOT/'docs/assumed_reference_inference_20261006/inference_scenario.json';scenario_hash=sha256(scenario_path)
    protocol=json.loads((a.density_root/'density_protocol.json').read_text());sweep=json.loads((a.density_root/'sweep_receipt.json').read_text())
    if sweep['status']!='COMPLETE' or sha256(a.density_root/'density_protocol.json')!=sweep['density_protocol_sha256'] or protocol['inference_scenario_sha256']!=scenario_hash:raise ValueError('Completed unchanged sweep required')
    full=a.comparator_root/'outputs/nectar/20261006T065811Z_assumed_reference_pretrained_heldout'
    classic=a.comparator_root/'outputs/nectar/20261006T081000Z_fixed_classical_heldout'
    cr=json.loads((classic/'prediction_receipt.json').read_text())
    if cr['scenario_sha256']!=scenario_hash or cr['reference_labels_used_for_prediction'] is not False:raise ValueError('Independent fixed classical producer required')
    roots={1.:full,**{f:a.density_root/f'fraction_{f:g}' for f in protocol['fractions']}}
    for f,root in roots.items():
        q=json.loads((root/'queue_receipt.json').read_text());assembly=json.loads((root/'assembled_sites_v1/assembly_receipt.json').read_text())
        if q['status']!='COMPLETED' or len(q['tiles'])!=19 or any(t['status']!='AUDITED' for t in q['tiles']):raise ValueError('Incomplete prediction queue')
        if assembly['inference_scenario_sha256']!=scenario_hash:raise ValueError('Changed producer scenario')
        for split in ('test','val'):
            if sha256(root/f'assembled_sites_v1/{split}_predictions.gpkg')!=assembly['output_sha256'][split]:raise ValueError('Changed assembled predictions')
    for split in ('test','val'):
        std=classic/f'{split}_standardized';r=json.loads((std/'provenance.json').read_text())
        if r['output_sha256']!=sha256(std/'predictions.gpkg') or r['input_sha256']!=cr['crowns_sha256']:raise ValueError('Classical provenance mismatch')
    current_lock_hash=None
    if a.current_classical:
        if a.current_classical_lock is None:p.error('current classical prediction needs its train-only lock')
        lock=json.loads(a.current_classical_lock.read_text())
        if (lock['status']!='AUTHOR-CONFIRMED TRAINING-ONLY CLASSICAL CONFIGURATION'
            or lock['heldout_used_for_selection'] is not False or not lock['prior_test_exposure_disclosed']):
            raise ValueError('disclosed train-only current classical configuration required')
        current_lock_hash=sha256(a.current_classical_lock)
        selection_path=a.current_classical_lock.parent/'selection.json'
        selection=json.loads(selection_path.read_text())
        if (sha256(selection_path)!=lock['selection_sha256']
            or selection['reference_json_sha256']!=prov['source_json_sha256']
            or selection['reference_gpkg_sha256']!=sha256(a.reference)):
            raise ValueError('classical training selection used a different reference')
        current=json.loads((a.current_classical/'prediction_receipt.json').read_text())
        if (current['author_model_lock_sha256']!=current_lock_hash or current['parameters']!=lock['parameters']
            or current['reference_labels_used_for_prediction'] is not False):
            raise ValueError('current classical producer/configuration mismatch')
        for split in ('test','val'):
            std=a.current_classical/f'{split}_standardized';standard=json.loads((std/'provenance.json').read_text())
            if standard['output_sha256']!=sha256(std/'predictions.gpkg') or standard['input_sha256']!=current['crowns_sha256']:
                raise ValueError('current classical standardized hash mismatch')
        classic=a.current_classical;cr=current
    spec=importlib.util.spec_from_file_location('ev',ROOT/'scripts/12_evaluate.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    defs=site_definitions();frames={k:gpd.read_file(a.reference,layer=k) for k in ('single_trees','tree_groups','uncertainty_regions')}
    for frame in frames.values():
        if frame.crs.to_epsg()!=2193 or not frame.geometry.is_valid.all():raise ValueError('Reference geometry invalid')
    a.output.mkdir(parents=True);all_sources={};rows=[];all_results={}
    started=datetime.now(timezone.utc)
    for split in ('val','test'):
        sites={s:v for s,v in defs.items() if v['split']==split}
        ref={k:{s:[(str(r.label_id),r.geometry.intersection(box(*v['bbox_epsg2193']))) for _,r in frame[frame.site_id==s].iterrows()] for s,v in sites.items()} for k,frame in frames.items()}
        def evaluate(path):
            predictions,_=ev.load_polygons(path,'predictions',sites,'prediction_id')
            return {s:score_site(ref['single_trees'][s],predictions[s],[g for _,g in ref['tree_groups'][s]],[g for _,g in ref['uncertainty_regions'][s]],2.,0.,policy['prediction_ignore_rule']) for s in sites}
        cp=classic/f'{split}_standardized/predictions.gpkg';baseline=evaluate(cp);scores={}
        for f,root in roots.items():
            sp=root/f'assembled_sites_v1/{split}_predictions.gpkg';model=evaluate(sp);scores[f]=model
            result={'status':'POST-REVISION CONDITIONAL COMPARISON; NOT FROZEN','reference_frozen':False,'actual_independent_agreement_available':True,'held_out_results_exposed':True,'model_settings_changed':False,'reference_changed_after_predictions':True,'site_ids':list(sites),'split':split,'fraction':f,'pretrained_summary':aggregate(list(model.values())),'classical_provisional_summary':aggregate(list(baseline.values())),'pretrained_by_site':model,'classical_by_site':baseline,'bootstrap':paired_site_bootstrap(model,baseline,2000,42),'sources':{'reference':sha256(a.reference),'policy':sha256(policy_path),'scenario':scenario_hash,'sat':sha256(sp),'classical':sha256(cp),'sat_assembly':sha256(root/'assembled_sites_v1/assembly_receipt.json'),'classical_receipt':sha256(classic/'prediction_receipt.json')},'acceptance_declared':False}
            name=f'{split}_fraction_{f:g}';write_json(a.output/f'{name}.json',result);all_results[name]=result;all_sources[name]=result['sources']
            summary=result['pretrained_summary'];rows.append({'split':split,'fraction':f,**summary,'f1_ci_lower':result['bootstrap']['intervals']['f1']['lower'],'f1_ci_upper':result['bootstrap']['intervals']['f1']['upper']})
            print(split,f,'reference',sum(v['n_reference'] for v in model.values()),'TP/FP/FN',summary['tp'],summary['fp'],summary['fn'],flush=True)
        write_json(a.output/f'{split}_density_vs_full_bootstrap.json',{str(f):paired_site_bootstrap(v,scores[1.],2000,42) for f,v in scores.items()})
    with (a.output/'density_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for ax,key in zip(axs,('precision','recall','f1')):
        for split in ('val','test'):
            sub=sorted((r for r in rows if r['split']==split),key=lambda r:r['fraction']);ax.plot([r['fraction']*100 for r in sub],[r[key] for r in sub],'-o',label=split)
        ax.set(xlabel='Retained returns (%)',ylabel=key,ylim=(0,1));ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Revised author reference: conditional fixed-model density response');fig.tight_layout();fig.savefig(a.output/'density_curve.png',dpi=160);plt.close(fig)
    receipt={'run_id':started.strftime('%Y%m%dT%H%M%SZ')+'_revised_reference_rescore','started_utc':started.isoformat(),'completed_utc':datetime.now(timezone.utc).isoformat(),'reference_frozen':False,'reference_json_sha256':prov['source_json_sha256'],'reference_gpkg_sha256':sha256(a.reference),'producer_scenario_unchanged_sha256':scenario_hash,'sources':all_sources,'source_code_sha256':sha256(Path(__file__)),'GPU_inference_rerun':False,'fine_tuning_run':False,'current_classical_lock_sha256':current_lock_hash,'classical_parameters':cr.get('parameters'),'exit_code':0}
    write_json(a.output/'receipt.json',receipt)
    (a.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(a.output/'sha256_manifest.json',{str(f.relative_to(a.output)):sha256(f) for f in a.output.iterdir() if f.is_file()})
if __name__=='__main__':main()
