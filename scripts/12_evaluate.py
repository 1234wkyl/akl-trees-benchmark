#!/usr/bin/env python
"""Training-only development scoring or explicitly approved held-out scoring."""
import argparse
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from benchmark.common import ROOT, sha256, site_definitions, validate_freeze, write_json
from benchmark.evaluation import aggregate, paired_site_bootstrap, score_site


def load_polygons(path, layer, sites, id_column):
    frame = gpd.read_file(path, layer=layer)
    if frame.crs is None or frame.crs.to_epsg() != 2193:
        raise ValueError(f'{path}: expected explicit EPSG:2193')
    if not {'site_id', id_column} <= set(frame):
        raise ValueError(f'{path}: requires site_id and {id_column}')
    if frame.site_id.isna().any() or frame[id_column].isna().any():
        raise ValueError('missing site/object IDs')
    if not set(frame.site_id) <= set(sites):
        raise ValueError('unknown or out-of-split site in input; supply only requested split')
    if frame.duplicated(['site_id', id_column]).any():
        raise ValueError('duplicate site/object ID')
    records, outside = {sid: [] for sid in sites}, []
    for _, row in frame.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {'Polygon', 'MultiPolygon'}:
            raise ValueError('repair/review crown geometry before scoring')
        clipped = geometry.intersection(box(*sites[row.site_id]['bbox_epsg2193']))
        if clipped.is_empty or clipped.area == 0:
            outside.append({'site_id': row.site_id, 'object_id': str(row[id_column])})
            continue
        records[row.site_id].append((str(row[id_column]), clipped))
    return records, outside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--baseline-predictions', type=Path)
    ap.add_argument('--prediction-layer', default='predictions')
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--policy', type=Path, default=ROOT / 'config/evaluation_draft.json')
    ap.add_argument('--stage', choices=['development', 'benchmark'], default='development')
    ap.add_argument('--site-ids', nargs='+',
                    help='development-only subset of training site IDs; benchmark scoring always uses the full test split')
    ap.add_argument('--freeze-record', type=Path)
    ap.add_argument('--fp-audit', type=Path)
    ap.add_argument('--prediction-run-record', type=Path)
    ap.add_argument('--baseline-run-record', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.stage == 'benchmark' and args.site_ids:
        raise ValueError('site subsets are development-only; benchmark scoring requires the full test split')
    policy = json.loads(args.policy.read_text())
    project = json.loads((ROOT / 'config/project.json').read_text())
    thresholds = project['acceptance_thresholds']
    supported = {'matching': 'greedy_iou_then_distance_then_ids',
                 'distance_reference': 'centroids_after_site_clipping',
                 'bootstrap_unit': 'paired_site', 'bootstrap_stratification': 'none', 'crs': 'EPSG:2193'}
    if any(policy.get(k) != v for k, v in supported.items()):
        raise ValueError('policy requests an unimplemented method; do not silently substitute')
    if policy['match_tolerance_m'] != thresholds['match_tolerance_m'] or policy['bootstrap_iterations'] != thresholds['bootstrap_iterations'] or policy['random_seed'] != project['random_seed']:
        raise ValueError('policy changes predeclared tolerance, iterations or seed')
    if args.stage == 'benchmark':
        fixed_by_researcher = policy.get('status') == 'researcher_fixed' and bool(policy.get('researcher_decision_record'))
        confirmed_by_supervisor = policy.get('status') == 'approved' and bool(policy.get('supervisor_confirmation_record'))
        if not (fixed_by_researcher or confirmed_by_supervisor) or not policy.get('false_positive_precision_definition'):
            raise ValueError('held-out scoring requires a confirmed operational policy')
        validate_freeze(args.freeze_record, sha256(args.reference), sha256(args.policy))
    split = 'train' if args.stage == 'development' else 'test'
    sites = {sid: s for sid, s in site_definitions().items() if s['split'] == split}
    if args.site_ids:
        requested = set(args.site_ids)
        if len(requested) != len(args.site_ids) or not requested <= set(sites):
            raise ValueError('requested development sites must be unique training site IDs')
        sites = {sid: sites[sid] for sid in args.site_ids}
    if args.stage == 'benchmark':
        for record_path, predictions_path in [(args.prediction_run_record,args.predictions),
                                               (args.baseline_run_record,args.baseline_predictions)]:
            if predictions_path is None:
                continue
            if record_path is None:
                raise ValueError('held-out model coverage requires its completed run record')
            record=json.loads(record_path.read_text())
            if record.get('status')!='completed' or set(record.get('completed_site_ids',[]))!=set(sites) or record.get('predictions_sha256')!=sha256(predictions_path) or not record.get('model_configuration_sha256'):
                raise ValueError('run record must bind completed coverage and frozen model configuration to predictions')
    # Reference contains all splits: select explicitly before passing strict loader.
    ref_records = {}
    for layer in ['single_trees', 'tree_groups', 'uncertainty_regions']:
        frame = gpd.read_file(args.reference, layer=layer)
        if frame.crs is None or frame.crs.to_epsg() != 2193 or not set(frame.site_id) <= set(site_definitions()):
            raise ValueError('invalid reference CRS/site IDs')
        if frame.duplicated(['site_id','label_id']).any():
            raise ValueError('duplicate reference label IDs')
        if frame.geometry.isna().any() or frame.geometry.is_empty.any() or not frame.geometry.is_valid.all() or not frame.geometry.geom_type.isin(['Polygon','MultiPolygon']).all():
            raise ValueError('invalid reference geometry')
        selected = frame[frame.site_id.isin(sites)]
        ref_records[layer] = {sid: [(str(r.label_id), r.geometry.intersection(box(*sites[sid]['bbox_epsg2193']))) for _, r in selected[selected.site_id == sid].iterrows()] for sid in sites}
    predictions, outside = load_polygons(args.predictions, args.prediction_layer, sites, 'prediction_id')
    def score(pred):
        return {sid: score_site(ref_records['single_trees'][sid], pred[sid],
                               [g for _, g in ref_records['tree_groups'][sid]],
                               [g for _, g in ref_records['uncertainty_regions'][sid]],
                               policy['match_tolerance_m'], policy['minimum_match_iou'],
                               policy['prediction_ignore_rule']) for sid in sites}
    by_site = score(predictions)
    baseline, baseline_outside = None, []
    if args.baseline_predictions:
        bp, baseline_outside = load_polygons(args.baseline_predictions, args.prediction_layer, sites, 'prediction_id')
        baseline = score(bp)
    summary = aggregate(list(by_site.values()))
    by_stratum = {stratum: aggregate([by_site[sid] for sid, s in sites.items() if s['stratum'] == stratum]) for stratum in project['reference_set']['strata']}
    boot = paired_site_bootstrap(by_site, baseline, policy['bootstrap_iterations'], policy['random_seed'])
    same_precision = policy.get('false_positive_precision_definition') == 'same_scored_detection_precision'
    fp_precision = summary['precision'] if same_precision else None
    if args.fp_audit:
        if same_precision:
            raise ValueError('same detection precision uses all scored TP/FP; do not substitute a selective negative audit')
        audit = json.loads(args.fp_audit.read_text())
        if not policy.get('false_positive_precision_definition') or audit.get('definition') != policy['false_positive_precision_definition']:
            raise ValueError('FP audit must implement the approved operational definition')
        num, den = audit['numerator'], audit['denominator']
        if type(num) is not int or type(den) is not int or not 0 <= num <= den or den <= 0:
            raise ValueError('invalid FP audit counts')
        if audit.get('predictions_sha256') != sha256(args.predictions):
            raise ValueError('FP audit is not bound to these predictions')
        fp_precision = num / den
    def meets(value, threshold):
        return None if value is None else value >= threshold
    criteria = {'precision': meets(summary['precision'], thresholds['detection_precision_min']),
                'recall': meets(summary['recall'], thresholds['detection_recall_min']),
                'median_iou': meets(summary['median_matched_iou'], thresholds['median_matched_crown_iou_min']),
                'false_positive_precision': meets(fp_precision, thresholds['false_positive_precision_min']),
                'recall_each_stratum': {s: meets(r['recall'], thresholds['min_recall_any_stratum']) for s, r in by_stratum.items()},
                'positive_f1_improvement_ci': boot.get('positive_improvement_ci_excludes_zero')}
    all_checks = [criteria[k] for k in ['precision','recall','median_iou','false_positive_precision','positive_f1_improvement_ci']] + list(criteria['recall_each_stratum'].values())
    result = {'status': 'DEVELOPMENT ONLY' if args.stage == 'development' else 'HELD-OUT EVALUATION',
              'split': split, 'policy_status': policy['status'], 'summary': summary,
              'by_stratum': by_stratum, 'by_site': by_site, 'bootstrap': boot,
              'baseline_summary': aggregate(list(baseline.values())) if baseline else None,
              'baseline_by_site': baseline, 'fp_audit_precision': fp_precision,
              'false_positive_precision': fp_precision,
              'false_positive_precision_source': 'same scored TP/(TP+FP)' if same_precision else 'separate audit or undefined',
              'criteria': criteria,
              'core_acceptance_met': all(c is True for c in all_checks) if args.stage == 'benchmark' else None,
              'outside_site_predictions': outside, 'baseline_outside_site_predictions': baseline_outside,
              'sources': {k: {'file': str(p), 'sha256': sha256(p)} for k, p in [('reference',args.reference),('predictions',args.predictions),('policy',args.policy),('sites',ROOT/'data/sites/reference_sites.json'),('project',ROOT/'config/project.json')]},
              'baseline_predictions_source': {'file': str(args.baseline_predictions), 'sha256': sha256(args.baseline_predictions)} if args.baseline_predictions else None,
              'fp_audit_source_sha256': sha256(args.fp_audit) if args.fp_audit else None,
              'prediction_run_record_sha256': sha256(args.prediction_run_record) if args.prediction_run_record else None,
              'baseline_run_record_sha256': sha256(args.baseline_run_record) if args.baseline_run_record else None,
              'evaluation_code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [ROOT/'scripts/12_evaluate.py',ROOT/'scripts/benchmark/evaluation.py',ROOT/'scripts/benchmark/common.py']},
              'freeze_record_sha256': sha256(args.freeze_record) if args.freeze_record else None}
    if args.output.exists():
        raise ValueError('use a fresh output folder')
    args.output.mkdir(parents=True)
    write_json(args.output/'evaluation.json', result)
    rows = ['# Crown evaluation', '', f"Status: {result['status']}; split: {split}; policy: {policy['status']}.", '', '| Metric | Value |', '|---|---:|']
    rows += [f'| {k} | {v if v is not None else "undefined"} |' for k, v in summary.items()]
    rows += ['', 'All eight stratum recall requirements remain in force. Under same_scored_detection_precision, the 0.95 criterion uses the same TP/FP as detection precision; no selective hard-negative audit substitutes for it. Development results never constitute benchmark acceptance.', '', 'Detailed matches, ignored IDs, site/stratum counts, paired bootstrap and source hashes are in evaluation.json.']
    (args.output/'README.md').write_text('\n'.join(rows)+'\n')
    print(json.dumps({'status':result['status'],'split':split,'summary':summary},indent=2))


if __name__ == '__main__':
    main()
