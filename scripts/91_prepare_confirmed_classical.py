#!/usr/bin/env python3
"""Fix a train-only classical choice and reuse unchanged raster/mask inputs."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from benchmark.common import ROOT,sha256,site_definitions,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('training-rescore','reference-lock','original-inputs','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh output directory required')
    lock=json.loads(a.reference_lock.read_text());report=json.loads((a.training_rescore/'summary.json').read_text())
    if lock['status']!='author_locked':raise ValueError('explicit author-derived reference lock required')
    train_ids={s for s,v in site_definitions().items() if v['split']=='train'}
    for row in report['rows']:
        result=json.loads((a.training_rescore/row['trial']/'evaluation.json').read_text())
        if (result['split']!='train' or set(result['by_site'])!=train_ids
            or result['sources']['reference']['sha256']!=lock['reference_gpkg_sha256']):
            raise ValueError('all and only current training sites required for selection')
    ranked=sorted(report['rows'],key=lambda r:(-r['f1'],-(r['median_matched_iou'] or 0),r['trial']))
    best=ranked[0];heights={'h25':2.5,'h3':3.,'h35':3.5}
    if set(r['trial'] for r in ranked)!=set(heights):raise ValueError('same three prespecified height trials required')
    parameters={'crown_minimum_height_m':heights[best['trial']],'detector_minimum_height_m':5.,
                'detector_gli_threshold':.06,'crown_gli_threshold':.03,'other_constants':'unchanged supplied source'}
    a.output.mkdir(parents=True)
    selection={'status':'Training-only selection using the author-confirmed reference; exposed-results continuation',
               'selection_rule':'largest micro F1, then median matched IoU, then stable trial ID',
               'selected':{'id':best['trial'],'parameters':parameters,'metrics':best},'ranked_trials':ranked,
               'reference_json_sha256':lock['source_json_sha256'],'reference_gpkg_sha256':lock['reference_gpkg_sha256'],
               'training_rescore_sha256':sha256(a.training_rescore/'summary.json'),'heldout_used_for_selection':False}
    write_json(a.output/'selection.json',selection)
    model_lock={'status':'AUTHOR-CONFIRMED TRAINING-ONLY CLASSICAL CONFIGURATION',
                'created_utc':datetime.now(timezone.utc).isoformat(),'selection_sha256':sha256(a.output/'selection.json'),
                'reference_lock_sha256':sha256(a.reference_lock),'training_site_ids':sorted(train_ids),
                'heldout_used_for_selection':False,'prior_test_exposure_disclosed':True,
                'historical_identity_verified':False,'parameters':parameters}
    write_json(a.output/'model_lock.json',model_lock)
    original=json.loads((a.original_inputs/'input_receipt.json').read_text())
    plan=json.loads((a.original_inputs/'plan.json').read_text())
    if original['plan_sha256']!=sha256(a.original_inputs/'plan.json'):raise ValueError('original input receipt changed')
    for rel,digest in original['source_files'].items():
        if sha256(a.original_inputs/rel)!=digest:raise ValueError('cached input changed: '+rel)
    target=a.output/'reused_inputs';target.mkdir()
    for child in a.original_inputs.iterdir():
        if child.name in {'plan.json','input_receipt.json'}:continue
        (target/child.name).symlink_to(child.resolve(),target_is_directory=child.is_dir())
    plan.update(classical_selection_sha256=sha256(a.output/'selection.json'),parameters=parameters,
                inference_scenario_sha256=None,author_model_lock_sha256=sha256(a.output/'model_lock.json'))
    write_json(target/'plan.json',plan)
    # The original intake manifest includes its plan as well as cached data.
    # Rebind that metadata entry explicitly; raster/mask hashes stay unchanged.
    original['source_files']['plan.json']=sha256(target/'plan.json')
    original.update(plan_sha256=sha256(target/'plan.json'),
        original_input_receipt_sha256=sha256(a.original_inputs/'input_receipt.json'),
        reused_without_redownload=True,author_model_lock_sha256=sha256(a.output/'model_lock.json'))
    write_json(target/'input_receipt.json',original)
    print('Fixed current train-only classical choice:',best['trial'],parameters,flush=True)


if __name__=='__main__':main()
