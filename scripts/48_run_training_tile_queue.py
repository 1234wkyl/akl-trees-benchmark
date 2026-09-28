#!/usr/bin/env python3
"""Run, reconcile and audit an explicit queue of staged training-only tiles."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from benchmark.common import ROOT, sha256, write_json


def successful_execution(path):
    plan_path = path / 'plan.json'
    if not plan_path.is_file():
        return False
    plan = json.loads(plan_path.read_text())
    return plan.get('status') == 'PROCESS COMPLETED' and plan.get('exit_code') == 0


def run_recorded(label, command):
    subprocess.run(['bash', str(ROOT / 'scripts/hpc/run_vm.sh'), label, '--', *map(str, command)],
                   cwd=ROOT, check=True)


def update_receipt(path, image, rows):
    write_json(path, {
        'status': 'RUNNING' if any(row['status'] != 'AUDITED' for row in rows) else 'COMPLETED',
        'scope': 'training-only pretrained tile inference, coordinate reconciliation and audit',
        'image': image,
        'tiles': rows,
        'held_out_inference_permitted': False,
        'reference_labels_used': False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--tile', action='append', required=True)
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'SegmentAnyTree')
    parser.add_argument('--image', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'(?:[^\s]+@)?sha256:[0-9a-f]{64}', args.image):
        parser.error('pin an immutable image digest')
    if len(args.tile) != len(set(args.tile)):
        parser.error('tile names must be unique')
    rows = []
    for tile in args.tile:
        stem = Path(tile).stem
        tile_root = args.base / stem
        provenance_path = tile_root / 'staging/provenance.json'
        if not provenance_path.is_file():
            raise ValueError(f'{tile}: staged provenance is missing')
        provenance = json.loads(provenance_path.read_text())
        if (provenance.get('tile') != tile or provenance.get('safe_full_tile_for_development') is not True
                or provenance.get('held_out_intersecting_sites')):
            raise ValueError(f'{tile}: training-only staging record failed')
        inputs = list((tile_root / 'staging/input_laz').glob('*.laz'))
        if len(inputs) != 1 or sha256(inputs[0]) != provenance['input_laz_sha256']:
            raise ValueError(f'{tile}: staged input does not match provenance')
        execution = tile_root / 'execution'
        reconciled_dir = tile_root / 'reconciled'
        reconciled = reconciled_dir / f'{inputs[0].stem}_reconciled.laz'
        reconciliation = reconciled_dir / 'coordinate_reconciliation.json'
        audit = reconciled_dir / 'output_audit.json'
        row = {'tile': tile, 'training_sites': provenance['training_sites'],
               'input': str(inputs[0].resolve()), 'input_sha256': provenance['input_laz_sha256'],
               'status': 'STAGED'}
        rows.append(row); update_receipt(args.base / 'queue_receipt.json', args.image, rows)
        if not args.execute:
            continue
        if execution.exists() and not successful_execution(execution):
            raise ValueError(f'{tile}: existing execution is not a recorded success; inspect before retrying')
        if not execution.exists():
            run_recorded(f'sat_pretrained_{stem}_training_tile', [
                sys.executable, ROOT / 'scripts/hpc/run_sat_container.py',
                '--source', args.source, '--input', inputs[0].parent,
                '--image', args.image, '--output', execution, '--execute'])
        row['status'] = 'INFERENCE_COMPLETED'
        outputs = sorted((execution / 'predictions/final_results').glob('*.laz'))
        if len(outputs) != 1:
            raise ValueError(f'{tile}: expected exactly one raw prediction, found {outputs}')
        row['execution_plan_sha256'] = sha256(execution / 'plan.json')
        row['raw_prediction'] = str(outputs[0].resolve())
        row['raw_prediction_sha256'] = sha256(outputs[0])
        update_receipt(args.base / 'queue_receipt.json', args.image, rows)
        if reconciled_dir.exists() and not (reconciled.is_file() and reconciliation.is_file()):
            raise ValueError(f'{tile}: partial reconciliation output exists')
        if not reconciled_dir.exists():
            reconciled_dir.mkdir()
            run_recorded(f'reconcile_{stem}_training_points', [
                sys.executable, ROOT / 'scripts/33_reconcile_sat_output.py',
                '--input', inputs[0], '--prediction', outputs[0], '--output', reconciled,
                '--report', reconciliation])
        report = json.loads(reconciliation.read_text())
        if (report.get('status') != 'RECONCILED' or report.get('output_sha256') != sha256(reconciled)
                or report.get('reciprocal_unique_mapping') is not True):
            raise ValueError(f'{tile}: reconciliation validation failed')
        row['status'] = 'RECONCILED'
        row['reconciled_prediction'] = str(reconciled.resolve())
        row['reconciled_prediction_sha256'] = sha256(reconciled)
        update_receipt(args.base / 'queue_receipt.json', args.image, rows)
        if not audit.exists():
            run_recorded(f'audit_{stem}_training_points', [
                sys.executable, ROOT / 'scripts/14_audit_sat_output.py',
                '--input', inputs[0], '--prediction', reconciled, '--output', audit])
        checked = json.loads(audit.read_text())
        required = (checked.get('point_count_equal') is True
                    and checked.get('bidirectional_coordinate_coverage_ok') is True
                    and checked.get('epsg2193_coordinate_consistency') is True
                    and checked.get('uint16_boundary_warning') is False
                    and checked.get('prediction_sha256') == sha256(reconciled))
        if not required:
            raise ValueError(f'{tile}: coordinate audit failed')
        row['status'] = 'AUDITED'
        row['coordinate_audit'] = str(audit.resolve())
        row['coordinate_audit_sha256'] = sha256(audit)
        row['points'] = checked['prediction_points']
        row['positive_instances'] = checked['positive_instances']
        update_receipt(args.base / 'queue_receipt.json', args.image, rows)
    print(json.dumps({'receipt': str((args.base / 'queue_receipt.json').resolve()),
                      'statuses': {row['tile']: row['status'] for row in rows}}, indent=2))


if __name__ == '__main__':
    main()
