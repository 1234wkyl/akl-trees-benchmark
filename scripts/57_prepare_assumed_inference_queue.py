#!/usr/bin/env python3
"""Bind an explicit hypothetical reference to an immutable pretrained prediction queue.

This stages only previously unprocessed tiles intersecting validation/test sites.
It does not run evaluation, certify independent annotations, or permit fine-tuning.
"""
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from benchmark.common import ROOT,sha256,site_definitions,write_json
from benchmark.inference_scenario import training_signature,validate_inference_scenario

IMAGE='sha256:fc814e8b230060ea5475b29e2ec7ad01fa1b0e1308116585a7f2ef9f24c82d91'
PLAN=ROOT/'docs/project_status_2026-09-18/tile_coverage_preflight/coverage_plan.json'
OLD=ROOT/'docs/group_refinement_2026-09-18/export_20260922T060150894212Z/crown_labels_XZ_2026-09-22-2.json'
SELECTION=ROOT/'outputs/nectar/20260920T142229Z_gpu_v100_4319/training_conversion_prefreeze249_v1/aggregate/summary.json'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=ROOT/'data/labels/crown_labels_XZ_2026-10-06.json')
    p.add_argument('--original-reference',type=Path,default=OLD)
    p.add_argument('--selection',type=Path,default=SELECTION)
    p.add_argument('--source-dir',type=Path,default=ROOT.parent/'Xi_Zhang_Benchmark/laz')
    p.add_argument('--stage',action='store_true')
    a=p.parse_args()
    if a.base.exists():p.error('use a fresh scenario directory; staging is not automatically retried')
    current=a.reference.resolve();old=a.original_reference.resolve();selection=a.selection.resolve()
    chosen=json.loads(selection.read_text());sites=site_definitions()
    if training_signature(current)!=training_signature(old):raise ValueError('training reference changed: select conversion on training first')
    config=json.loads((ROOT/'config/segmentanytree_source.json').read_text())
    rows=json.loads(PLAN.read_text())['tiles']
    remaining=[t for t in rows if t['held_out_intersecting_sites']]
    if {s for t in remaining for s in t['held_out_intersecting_sites']}!={s for s,r in sites.items() if r['split']!='train'}:
        raise ValueError('remaining tile list does not cover all held-out sites')
    a.base.mkdir(parents=True)
    scenario=a.base/'inference_scenario.json'
    write_json(scenario,{'mode':'ASSUMED_REFERENCE_PRETRAINED_INFERENCE_ONLY',
        'created_utc':datetime.now(timezone.utc).isoformat(),'formal_reference_frozen':False,
        'independent_labels_received':False,'evaluation_permitted':False,
        'assumption':'Use the author 6 October export as the hypothetical adjudicated result for inference preparation only.',
        'model_configuration_locked_for_this_scenario':True,
        'reference_json':str(current.relative_to(ROOT)),'reference_json_sha256':sha256(current),
        'original_selection_reference_json':str(old.relative_to(ROOT)),
        'original_selection_reference_json_sha256':sha256(old),'training_geometry_signature':training_signature(current),
        'site_definitions_sha256':sha256(ROOT/'data/sites/reference_sites.json'),
        'sat_config_sha256':sha256(ROOT/'config/segmentanytree_source.json'),
        'training_conversion_selection':str(selection.relative_to(ROOT)),
        'training_conversion_selection_sha256':sha256(selection),'conversion_config_id':chosen['selected_config_id'],
        'point_to_crown_conversion':chosen['selected_row'],
        'image':IMAGE,'source_commit':config['commit'],'checkpoint_sha256':config['checkpoint_sha256'],
        'semantic_output_tree_value':1,'instance_output_rule':'positive prediction IDs only; tile-local IDs reconciled before assembly',
        'posthoc_point_or_area_filters':None,'fine_tuning_permitted':False,
        'prediction_policy':'Generate each remaining input once. Do not inspect held-out scores, retune models or claim a formal freeze.'})
    validate_inference_scenario(scenario)
    command=[sys.executable,str(ROOT/'scripts/48_run_training_tile_queue.py'),'--base',str(a.base),
        '--source',str(ROOT.parent/'SegmentAnyTree'),'--image',IMAGE,'--inference-scenario',str(scenario)]
    for t in remaining:command+=['--tile',t['tile']]
    write_json(a.base/'queue_plan.json',{'status':'PLANNED','tiles':[r['tile'] for r in remaining],
        'total_tiles':len(remaining),'raw_header_points':sum(r['header_points'] for r in remaining),
        'coverage_plan_sha256':sha256(PLAN),'scenario_sha256':sha256(scenario),
        'already_completed_training_tiles_reused':14,'nonintersecting_tiles_not_inferred':6,
        'command_argv':command,'evaluation_permitted':False})
    if a.stage:
        for row in remaining:
            stage=a.base/Path(row['tile']).stem/'staging'
            subprocess.run(['bash',str(ROOT/'scripts/hpc/run_vm.sh'),'stage_assumed_'+Path(row['tile']).stem,'--',
                sys.executable,str(ROOT/'scripts/47_prepare_training_tile.py'),'--tile',row['tile'],
                '--coverage-plan',str(PLAN),'--source-dir',str(a.source_dir),
                '--inference-scenario',str(scenario),'--output',str(stage)],cwd=ROOT,check=True)
        subprocess.run(command,cwd=ROOT,check=True)
        write_json(a.base/'staging_complete.json',{'status':'ALL INPUTS STAGED; MODEL NOT RUN',
            'tiles':len(remaining),'scenario_sha256':sha256(scenario),'queue_plan_sha256':sha256(a.base/'queue_plan.json')})
    print(json.dumps({'scenario':str(scenario),'tiles':len(remaining),'execution_argv':command+['--execute']},indent=2))
if __name__=='__main__':main()
