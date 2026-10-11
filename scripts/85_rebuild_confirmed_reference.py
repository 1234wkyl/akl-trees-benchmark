#!/usr/bin/env python3
"""Rebuild derived analysis and weak labels from an explicit author reference lock.

Completed predictions remain immutable. Author locking does not retrospectively
make exposed test results blind or certify per-point biological ownership.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark.common import ROOT, sha256, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lock-record', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('fresh output directory required')
    lock = json.loads(args.lock_record.read_text())
    source = ROOT / lock['source_json']
    for key, path in [('source_json_sha256', source),
                      ('site_definitions_sha256', ROOT/'data/sites/reference_sites.json'),
                      ('evaluation_policy_sha256', ROOT/'config/evaluation_draft.json')]:
        if sha256(path) != lock[key]:
            raise ValueError('author lock hash mismatch: ' + key)
    if lock['status'] != 'author_locked' or not lock['held_out_results_previously_exposed']:
        raise ValueError('explicit author lock and exposure disclosure required')
    args.output.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    receipt = {'run_id': started.strftime('%Y%m%dT%H%M%SZ')+'_confirmed_reference_rebuild',
               'started_utc': started.isoformat(), 'status': 'RUNNING',
               'lock_record_sha256': sha256(args.lock_record),
               'reference_json_sha256': sha256(source), 'commands': [],
               'GPU_inference_rerun': False, 'model_weights_updated': False,
               'author_reference_locked': True, 'supervisor_freeze_claimed': False}

    def save():
        write_json(args.output/'rebuild_receipt.json', receipt)

    def run(name, script, *params):
        argv = [sys.executable, str(ROOT/'scripts'/script), *map(str, params)]
        step = {'step': name, 'argv': argv, 'started_utc': datetime.now(timezone.utc).isoformat()}
        receipt['commands'].append(step); save()
        print('+', ' '.join(argv), flush=True)
        tick = time.monotonic()
        with (args.output/(name+'.log')).open('w') as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            code = process.wait()
        step.update(exit_code=code, wall_seconds=time.monotonic()-tick,
                    finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        if code:
            raise subprocess.CalledProcessError(code, argv)

    try:
        run('reference', '45_prepare_prefreeze_scoring_candidate.py',
            '--labels-json', source, '--output', args.output/'reference')
        ref = args.output/'reference/prefreeze_evaluation_reference.gpkg'
        derived = {**lock, 'reference_gpkg': str(ref.resolve()),
                   'reference_gpkg_sha256': sha256(ref),
                   'derivation_receipt_sha256': sha256(ref.parent/'provenance.json')}
        write_json(args.output/'AUTHOR_DERIVED_REFERENCE_LOCK.json', derived)
        run('composition', '44_summarise_reference_composition.py',
            '--current', source, '--output', args.output/'composition')
        density = ROOT/'outputs/nectar_receipts/20261007T024000Z_fixed_density_transfer'
        run('fixed_model_scores', '84_rescore_revised_reference.py', '--reference', ref,
            '--density-root', density/'completed_results',
            '--comparator-root', density/'fixed_comparators', '--output', args.output/'scores')
        comparators = density/'fixed_comparators/outputs/nectar'
        run('test_error_gallery', '64_build_conditional_error_gallery.py',
            '--comparison', args.output/'scores/test_fraction_1.json', '--reference', ref,
            '--sat-predictions', comparators/'20261006T065811Z_assumed_reference_pretrained_heldout/assembled_sites_v1/test_predictions.gpkg',
            '--classical-predictions', comparators/'20261006T081000Z_fixed_classical_heldout/test_standardized/predictions.gpkg',
            '--output', args.output/'test_error_review')
        run('point_candidates', '53_prepare_point_label_candidates.py',
            '--reference', ref, '--coverage-plan', ROOT/'docs/project_status_2026-09-18/tile_coverage_preflight/coverage_plan.json',
            '--laz-root', ROOT.parent/'Xi_Zhang_Benchmark/laz', '--output', args.output/'point_candidates')
        run('alto_group_candidates', '69_prepare_alto_group_point_proposals.py',
            '--reference-json', source, '--point-candidates', args.output/'point_candidates',
            '--alto-crowns', ROOT/'outputs/alto_review_20261006/comparison',
            '--alto-receipt', ROOT/'outputs/alto_review_20261006/RUN_RECEIPT.json',
            '--output', args.output/'alto_group_candidates')
        run('weak_label_screen', '72_screen_weak_point_labels.py',
            '--reference-json', source, '--candidates', args.output/'alto_group_candidates',
            '--alto-crowns', ROOT/'outputs/alto_review_20261006/comparison',
            '--output', args.output/'screened_weak_labels')
        run('weak_loss_probe', '73_probe_weak_loss_targets.py',
            '--screened', args.output/'screened_weak_labels', '--site-id', 'S017',
            '--output', args.output/'weak_loss_probe')
        receipt.update(status='COMPLETE', exit_code=0,
                       completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        receipt.update(status='FAILED', error=str(exc), exit_code=1,
                       completed_utc=datetime.now(timezone.utc).isoformat())
        save(); raise
    save()
    (args.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output/'SHA256_manifest.json',
               {str(p.relative_to(args.output)): sha256(p)
                for p in sorted(args.output.rglob('*')) if p.is_file() and p.name!='SHA256_manifest.json'})
    print('Completed:', receipt['run_id'], flush=True)


if __name__ == '__main__':
    main()
