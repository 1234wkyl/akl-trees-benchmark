#!/usr/bin/env python
"""Rank supplied, completed classical trials using training sites only."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from benchmark.common import ROOT, sha256, site_definitions, write_json


def validate_reproduction_record(record, training_sites, provisional=False):
    if provisional:
        if (record.get('status') != 'TRAINING-ONLY FRESH DETECTION AND CROWN COMPARISON'
                or record.get('frozen_seed_coordinates_used_for_generation') is not False
                or record.get('historical_reproduction_verified') is not False
                or set(record.get('by_site', {})) != training_sites
                or not record.get('crown_input_sha256')):
            raise ValueError('require completed, explicitly provisional fresh-detection comparison on all training sites')
    elif record.get('geometry_matches') is not True or not record.get('confirmed_pipeline_provenance_record'):
        raise ValueError('first verify frozen output geometry AND historical pipeline provenance; geometry comparison alone is insufficient')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trials',type=Path,required=True,help='JSON trial list: id, predictions, parameters, completed_site_ids')
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--policy',type=Path,default=ROOT/'config/evaluation_draft.json')
    ap.add_argument('--reproduction-record',type=Path,required=True)
    ap.add_argument('--provisional-reconstruction',action='store_true',
                    help='explicit development mode for completed fresh-detection reconstruction; no historical certification')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('use a fresh output folder')
    reproduction=json.loads(args.reproduction_record.read_text())
    trials=json.loads(args.trials.read_text())['trials'];train={sid for sid,s in site_definitions().items() if s['split']=='train'}
    validate_reproduction_record(reproduction, train, args.provisional_reconstruction)
    seen=set()
    for trial in trials:
        tid=trial['id']
        if not re.fullmatch('[A-Za-z0-9_.-]+',tid) or tid in seen:raise ValueError('invalid/duplicate trial ID')
        seen.add(tid)
        if set(trial['completed_site_ids'])!=train:raise ValueError('each trial must cover all eight training sites; no held-out sites')
        path=(args.trials.parent/trial['predictions']).resolve()
        if not path.is_file():raise ValueError(f'trial output missing: {path}')
        trial['_path']=path
    if not trials:raise ValueError('no completed trials')
    args.output.mkdir(parents=True);ranked=[]
    for trial in trials:
        out=args.output/trial['id']
        subprocess.run([sys.executable,str(ROOT/'scripts/12_evaluate.py'),'--reference',str(args.reference),
                        '--predictions',str(trial['_path']),'--policy',str(args.policy),
                        '--stage','development','--output',str(out)],check=True)
        metric=json.loads((out/'evaluation.json').read_text())['summary']
        ranked.append({'id':trial['id'],'parameters':trial['parameters'],'metrics':metric,
                       'predictions_sha256':sha256(trial['_path'])})
    ranked.sort(key=lambda t:(-(t['metrics']['f1'] if t['metrics']['f1'] is not None else -1),
                              -(t['metrics']['median_matched_iou'] if t['metrics']['median_matched_iou'] is not None else -1),t['id']))
    if ranked[0]['metrics']['f1'] is None:raise ValueError('no defined training F1; cannot select a trial')
    write_json(args.output/'selection.json',{'status':'training-only selection, not held-out acceptance',
        'provisional_reconstruction':args.provisional_reconstruction,
        'historical_certification_implied':False,
        'reference_frozen':False,
        'selection_rule':'largest micro F1, then median matched IoU, then stable trial ID',
        'selected':ranked[0],'ranked_trials':ranked,'site_ids':sorted(train),
        'reference_sha256':sha256(args.reference),'policy_sha256':sha256(args.policy),
        'trials_sha256':sha256(args.trials),'reproduction_record_sha256':sha256(args.reproduction_record)})
    print(f"Training trial selected: {ranked[0]['id']}")


if __name__=='__main__':main()
