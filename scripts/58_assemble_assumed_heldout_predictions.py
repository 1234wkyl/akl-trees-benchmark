#!/usr/bin/env python3
"""After all queued tiles finish, assemble fixed pretrained crowns without scoring."""
import argparse,json,subprocess,sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
from benchmark.common import ROOT,sha256,site_definitions,write_json
from benchmark.inference_scenario import validate_inference_scenario

def recorded(name,argv):
    subprocess.run(['bash',str(ROOT/'scripts/hpc/run_vm.sh'),name,'--',*map(str,argv)],cwd=ROOT,check=True)

def verify_completed_merge(path, sid, split, context, scenario, rows):
    r=json.loads((path/'merge_receipt.json').read_text())
    if (r['site_id']!=sid or r['split']!=split or r['context_m']!=context
            or r['inference_scenario_sha256']!=sha256(scenario)
            or r['status']!='ASSUMPTION-ONLY SITE MERGE COMPLETED; NO SCORING'
            or r['evaluation_permitted'] is not False
            or r['output_laz_sha256']!=sha256(path/f'{sid}_merged_points.laz')
            or r['coverage_domains_sha256']!=sha256(path/f'{sid}_coverage_domains.gpkg')
            or set(r['expected_source_tiles'])!=set(rows)):
        raise ValueError('existing merge does not bind the fixed site/protocol: '+sid)
    for t in r['tiles']:
        q=rows[t['tile']]
        if (t['prediction_sha256']!=q['reconciled_prediction_sha256']
                or t['coordinate_audit_sha256']!=q['coordinate_audit_sha256']):
            raise ValueError('existing merge does not bind the audited queue tile: '+sid)
    return r


def verify_completed_crowns(path, merge, scenario):
    r=json.loads((path/'provenance.json').read_text());cfg=json.loads(scenario.read_text())
    if (r['output_sha256']!=sha256(path/'candidate_crowns.gpkg')
            or r['prediction_sha256']!=sha256(merge/next(p.name for p in merge.glob('*_merged_points.laz')))
            or r['merge_receipt_sha256']!=sha256(merge/'merge_receipt.json')
            or r['hull_method']!=cfg['point_to_crown_conversion']['method']
            or r['buffer_m']!=cfg['point_to_crown_conversion']['buffer_m']
            or r['tree_semantic_values']!=[cfg['semantic_output_tree_value']]):
        raise ValueError('existing crown output differs from fixed producer/configuration')


def verify_completed_standard(path, crowns, sid, split):
    r=json.loads((path/'provenance.json').read_text())
    if (r['output_sha256']!=sha256(path/'predictions.gpkg')
            or r['input_sha256']!=sha256(crowns/'candidate_crowns.gpkg')
            or r['split']!=split or r['requested_site_ids']!=[sid]):
        raise ValueError('existing standardized crowns differ from fixed site/producer')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='reuse hash-verified completed child stages; never overwrite partial stages')
    a=p.parse_args();scenario=a.base/'inference_scenario.json';r=validate_inference_scenario(scenario)
    lock=json.loads((a.base/'postprocessing_lock.json').read_text())
    if lock['scenario_sha256']!=sha256(scenario) or lock['context_m']!=10.0 or lock['evaluation_permitted'] is not False:
        raise ValueError('require the recorded training-equivalent 10m assembly protocol')
    for name,digest in lock['script_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('postprocessing code changed after protocol pinning')
    if a.output.exists() and not a.resume:p.error('fresh postprocessing directory required, or explicit --resume')
    if (a.output/'assembly_receipt.json').exists():p.error('assembly already completed; do not repeat')
    q=json.loads((a.base/'queue_receipt.json').read_text());plan=json.loads((a.base/'queue_plan.json').read_text())
    rows={x['tile']:x for x in q['tiles']}
    if (q['status']!='COMPLETED' or set(rows)!=set(plan['tiles']) or any(x['status']!='AUDITED' for x in rows.values())
            or q['inference_scenario_sha256']!=sha256(scenario)):
        raise ValueError('all planned tiles must finish inference, coordinate reconciliation and audit first')
    coverage=ROOT/'docs/project_status_2026-09-18/tile_coverage_preflight/coverage_plan.json'
    c=json.loads(coverage.read_text());sites=site_definitions();a.output.mkdir(parents=True,exist_ok=a.resume);parts=[];receipts=[];resumed=[]
    for s in c['sites']:
        sid=s['site_id'];split=sites[sid]['split']
        if split=='train':continue
        dest=a.output/sid
        merge=[sys.executable,ROOT/'scripts/46_merge_training_site_tiles.py','--site-id',sid,'--context-m',str(lock['context_m']),
               '--coverage-plan',coverage,'--inference-scenario',scenario,'--output',dest/'merge']
        for name in s['required_tiles']:
            row=rows[name];merge+=['--tile',name,row['reconciled_prediction'],row['coordinate_audit'],
                                     '--verified-input',name,row['input']]
        existed=(dest/'merge/merge_receipt.json').is_file()
        if not existed:recorded('assemble_assumed_'+sid,merge)
        verify_completed_merge(dest/'merge',sid,split,lock['context_m'],scenario,{n:rows[n] for n in s['required_tiles']})
        if existed:resumed.append({'site_id':sid,'stage':'merge'})
        existed=(dest/'crowns/provenance.json').is_file()
        if not existed:recorded('fixed_crowns_assumed_'+sid,[sys.executable,ROOT/'scripts/20_sat_crown_candidates.py',
            '--prediction',dest/'merge'/f'{sid}_merged_points.laz','--merge-receipt',dest/'merge/merge_receipt.json',
            '--tree-semantic-values',str(r['semantic_output_tree_value']),'--hull-method',r['point_to_crown_conversion']['method'],
            '--buffer-m',str(r['point_to_crown_conversion']['buffer_m']),'--inference-scenario',scenario,'--output',dest/'crowns'])
        verify_completed_crowns(dest/'crowns',dest/'merge',scenario)
        if existed:resumed.append({'site_id':sid,'stage':'crowns'})
        existed=(dest/'standardized/provenance.json').is_file()
        if not existed:recorded('standardize_assumed_'+sid,[sys.executable,ROOT/'scripts/16_standardize_predictions.py',
            '--input',dest/'crowns/candidate_crowns.gpkg','--layer','candidate_crowns','--id-column','prediction_id',
            '--split',split,'--site-ids',sid,'--output',dest/'standardized'])
        verify_completed_standard(dest/'standardized',dest/'crowns',sid,split)
        if existed:resumed.append({'site_id':sid,'stage':'standardized'})
        f=gpd.read_file(dest/'standardized/predictions.gpkg',layer='predictions');parts.append(f)
        receipts.append({'site_id':sid,'split':split,'prediction_sha256':sha256(dest/'standardized/predictions.gpkg'),
            'merge_receipt_sha256':sha256(dest/'merge/merge_receipt.json'),'candidate_crowns_sha256':sha256(dest/'crowns/candidate_crowns.gpkg')})
    joined=gpd.GeoDataFrame(pd.concat(parts,ignore_index=True),geometry='geometry',crs=2193)
    for split in ('val','test'):
        f=joined[joined.split==split];target=a.output/f'{split}_predictions.gpkg'
        f.to_file(target,layer='predictions',driver='GPKG')
    write_json(a.output/'assembly_receipt.json',{'status':'ASSUMPTION-ONLY HELD-OUT PREDICTIONS ASSEMBLED; NO SCORING',
        'sites':receipts,'inference_scenario_sha256':sha256(scenario),'queue_receipt_sha256':sha256(a.base/'queue_receipt.json'),
        'output_sha256':{s:sha256(a.output/f'{s}_predictions.gpkg') for s in ('val','test')},
        'reference_frozen':False,'evaluation_permitted':False,'model_configuration_changed':False,
        'postprocessing_lock_sha256':sha256(a.base/'postprocessing_lock.json'),'reused_completed_stages':resumed})
    print('All 16 held-out site outputs assembled; no reference polygons or scoring metrics read.')
if __name__=='__main__':main()
