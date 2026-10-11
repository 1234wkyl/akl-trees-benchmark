#!/usr/bin/env python3
"""Export fixed weak-model comparisons and measured operational throughput."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
from shapely.geometry import box
from shapely.ops import unary_union
from benchmark.common import ROOT, sha256, site_definitions, write_json
from benchmark.evaluation import aggregate, paired_site_bootstrap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('comparisons', 'queue', 'classical', 'output'):
        p.add_argument('--'+n, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('fresh report directory required')
    queue = json.loads((a.queue/'queue_receipt.json').read_text())
    receipt = json.loads((a.comparisons/'receipt.json').read_text())
    if (queue['status'] != 'COMPLETED' or len(queue['tiles']) != 19
            or any(r['status'] != 'AUDITED' for r in queue['tiles'])
            or receipt['status'] != 'COMPLETE'
            or receipt['queue_receipt_sha256'] != sha256(a.queue/'queue_receipt.json')):
        raise ValueError('completed audited fixed-model results required')
    defs = site_definitions()
    rows, strata, sources = [], [], {}
    metrics = ('precision', 'recall', 'f1', 'median_matched_iou',
               'over_segmentation_rate', 'under_segmentation_rate')
    for split in ('val', 'test'):
        path = a.comparisons/f'{split}_comparison.json'
        r = json.loads(path.read_text())
        if (r['reference_sha256'] != receipt['reference_sha256']
                or r['checkpoint_sha256'] != queue['checkpoint_sha256']):
            raise ValueError('comparison reference/checkpoint mismatch')
        sources[split] = sha256(path)
        for name, key in (('Classical training-selected (3 m)', 'classical'),
                          ('SegmentAnyTree pretrained', 'pretrained'),
                          ('SegmentAnyTree weak pilot', 'weak')):
            records = r[key+'_by_site']
            boot = paired_site_bootstrap(records, iterations=2000, seed=42)
            rows.append({'split': split, 'model': name, **aggregate(list(records.values())),
                         **{m+'_'+b: boot['intervals'][m][b] for m in metrics for b in ('lower', 'upper')}})
            for stratum in dict.fromkeys(defs[s]['stratum'] for s in records):
                selected = [v for s,v in records.items() if defs[s]['stratum'] == stratum]
                strata.append({'split': split, 'model': name, 'stratum': stratum,
                               'sites': len(selected), **aggregate(selected)})
    a.output.mkdir(parents=True)
    for name, records in (('model_metrics.csv', rows), ('stratum_metrics.csv', strata)):
        with (a.output/name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    spec = importlib.util.spec_from_file_location('cost93', ROOT/'scripts/63_report_assumed_results.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    plan_path = ROOT/'docs/project_status_2026-09-18/tile_coverage_preflight/coverage_plan.json'
    plan = json.loads(plan_path.read_text())
    tiles = {r['tile']: r for r in plan['tiles']}
    timings, footprints = [], []
    for row in queue['tiles']:
        dest = Path(row['reconciled_prediction']).parent
        timing = helper.timing_for('*_weak_fixed_'+Path(row['input']).stem+'_*', str(dest/'prediction'))
        timings.append({'tile': row['tile'], **timing})
        footprints.append(box(*tiles[row['tile']]['bbox_epsg2193']))
    area = unary_union(footprints).area/1e6
    seconds = sum(r['wall_seconds'] for r in timings)
    classical_fragment = str(a.classical.resolve().relative_to(ROOT))
    classic = helper.timing_for('*_confirmed_classical_h3_predictions_rebound_*', classical_fragment)
    cp_path = ROOT/'outputs/preparation/confirmed_classical_choice_20261010/reused_inputs/plan.json'
    cp = json.loads(cp_path.read_text())
    classical_area = unary_union([box(*b) for b in cp['original_chunks']]).area/1e6
    write_json(a.output/'compute_receipt.json', {
        'scope': 'Measured wrapper throughput; not billing or an area-matched model cost ratio',
        'weak': {'model_wrapper_wall_seconds': seconds, 'allocated_device_wall_hours': seconds/3600,
                 'raw_header_envelope_union_km2': area,
                 'allocated_device_wall_hours_per_envelope_km2': seconds/3600/area,
                 'tile_timings': timings,
                 'includes': 'container, preprocessing, inference, postprocessing and I/O',
                 'excludes': 'training, coordinate audit, site assembly, staging, idle and environment setup'},
        'classical': {**classic, 'selected_grid_envelope_union_km2': classical_area,
                      'wall_hours_per_grid_envelope_km2': classic['wall_seconds']/3600/classical_area},
        'billing_rate': None, 'service_units': None, 'billed_GPU_hours': None,
        'limits': ['Raw tile-header rectangles are a technical coverage proxy.',
                   'Classical chunks contain nodata and have different support from SAT tiles.',
                   'GPU allocated wall time does not measure active CUDA kernel time.'],
        'queue_sha256': sha256(a.queue/'queue_receipt.json'),
        'coverage_plan_sha256': sha256(plan_path), 'classical_plan_sha256': sha256(cp_path)})
    def fmt(x):
        return 'NA' if x is None else f'{x:.4f}'
    lines = ['# Fixed weak-label pilot comparison', '',
             'I report the author-confirmed reference and all fixed model outputs. The weak pilot uses automatically screened polygon/ALTO candidates, not manually checked point truth. Previous test exposure and reference revision are disclosed. The final epoch, conversion and scoring policy were fixed before this queue; no result selects a new checkpoint.', '',
             '| Test model | Precision | Recall | F1 [95% site CI] | Median IoU | TP / FP / FN |',
             '|---|---:|---:|---|---:|---|']
    for r in rows:
        if r['split'] == 'test':
            lines.append(f"| {r['model']} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} [{fmt(r['f1_lower'])}, {fmt(r['f1_upper'])}] | {fmt(r['median_matched_iou'])} | {r['tp']} / {r['fp']} / {r['fn']} |")
    lines += ['', 'Whole-site paired bootstrap uses 2,000 draws and seed 42. Each stratum has only one site per split; stratum rows are point estimates. Weak-minus-pretrained and weak-minus-classical intervals remain in the source comparisons regardless of their sign.', '',
              f'The weak-model prediction wrappers took {seconds/3600:.3f} hours for {area:.4f} km² of raw header envelopes. This is {seconds/3600/area:.4f} allocated-device wall-hours per envelope km². Classical generation took {classic["wall_seconds"]} seconds for {classical_area:.4f} km² of selected grid envelopes. These different processing supports do not establish a fair cost ratio or a provider bill.']
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write_json(a.output/'receipt.json', {'sources': sources, 'model_rows': len(rows),
        'stratum_rows': len(strata), 'source_code_sha256': sha256(Path(__file__)),
        'model_retuned': False, 'supervisor_freeze_claimed': False,
        'output_sha256': {f.name: sha256(f) for f in a.output.iterdir() if f.is_file()}})
    print('Weak pilot tables, strata and measured throughput complete', flush=True)


if __name__ == '__main__':
    main()
