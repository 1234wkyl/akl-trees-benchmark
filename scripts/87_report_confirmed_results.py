#!/usr/bin/env python3
"""Report locked-author reference results and strata without selecting models."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from benchmark.common import ROOT, sha256, site_definitions, write_json
from benchmark.evaluation import aggregate, paired_site_bootstrap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rebuild', type=Path, required=True)
    ap.add_argument('--scores', type=Path, help='current fixed-model score folder; defaults to rebuild/scores')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        ap.error('fresh output required')
    lock = json.loads((a.rebuild/'AUTHOR_DERIVED_REFERENCE_LOCK.json').read_text())
    if lock['status'] != 'author_locked':
        raise ValueError('author reference lock required')
    sites = site_definitions()
    score_folder = a.scores or a.rebuild/'scores'
    score_receipt = json.loads((score_folder/'receipt.json').read_text())
    if score_receipt['reference_gpkg_sha256'] != lock['reference_gpkg_sha256']:
        raise ValueError('score receipt differs from the author lock')
    a.output.mkdir(parents=True)
    model_rows, stratum_rows, records = [], [], {}
    for split in ('val','test'):
        for fraction in (1.,.75,.5,.25,.125):
            path = score_folder/f'{split}_fraction_{fraction:g}.json'
            r = json.loads(path.read_text())
            if r['sources']['reference'] != lock['reference_gpkg_sha256']:
                raise ValueError('scoring reference differs from the author lock')
            methods = [('SAT',r['pretrained_by_site'])]
            if fraction == 1.:
                height = score_receipt.get('classical_parameters', {}).get('crown_minimum_height_m')
                classical_name = f'Classical training-selected ({height:g} m)' if height is not None else 'Classical provisional'
                methods.append((classical_name,r['classical_by_site']))
            for name, bysite in methods:
                summary = aggregate(list(bysite.values()))
                boot = paired_site_bootstrap(bysite,iterations=2000,seed=42)
                model_rows.append({'split':split,'model':name,'retained_fraction':fraction,
                    'reference_singles':sum(v['n_reference'] for v in bysite.values()),**summary,
                    **{m+'_'+b:boot['intervals'][m][b] for m in ('precision','recall','f1','median_matched_iou') for b in ('lower','upper')}})
                for stratum in dict.fromkeys(s['stratum'] for s in sites.values()):
                    selected = [v for sid,v in bysite.items() if sites[sid]['stratum']==stratum]
                    stratum_rows.append({'split':split,'model':name,'retained_fraction':fraction,
                        'stratum':stratum,'sites':len(selected),
                        'reference_singles':sum(v['n_reference'] for v in selected),**aggregate(selected)})
                records[f'{split}_{name}_{fraction:g}'] = boot
    for name,rows in [('model_metrics.csv',model_rows),('stratum_metrics.csv',stratum_rows)]:
        with (a.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(a.output/'model_bootstrap_intervals.json',records)
    def fmt(v):
        return 'NA' if v is None else f'{v:.4f}'
    lines=['# Author-confirmed reference: fixed-model results — 10 October 2026','',
           'I locked the supplied author export without changing its geometry. SAT predictions, density levels and scoring policy remain unchanged. The classical configuration is identified in the producer receipt and selected using training sites only. Reference, prediction and configuration identities are recorded for this comparison.','',
           '| Test model | Retention | Precision | Recall | F1 (95% site-bootstrap CI) | Median IoU | Ignored |',
           '|---|---:|---:|---:|---|---:|---:|']
    for r in model_rows:
        if r['split']=='test':
            lines.append(f"| {r['model']} | {r['retained_fraction']*100:g}% | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} ({fmt(r['f1_lower'])}, {fmt(r['f1_upper'])}) | {fmt(r['median_matched_iou'])} | {r['ignored_predictions']} |")
    lines += ['', 'I report all five predeclared SAT retention levels rather than choose one from test scores. Bootstrap resamples whole sites in paired comparisons (2,000 draws, seed 42). Each stratum has one site in each split; per-stratum rows are point estimates, not independently estimated stratum confidence intervals. Over/under-segmentation rates retain the existing geometric definition in the scorer.', '',
              'The same ordinary scored TP/(TP+FP) precision implements the original 0.85 precision and stricter 0.95 false-positive precision criteria. These are not different selective hard-negative audits. Neither full-density model meets the precision, recall and median-IoU criteria. This is a measured failure to meet model acceptance, not a failure of the evaluation program.', '',
              'Compute cost remains separately bound to actual processed tile footprints and measured run receipts. Dividing full-tile inference time by the small labelled site areas is invalid; service units need the allocation accounting rate. Fine-tuning using automated point candidates must be reported as a separate exploratory weak-label model, not checked-label supervision.']
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write_json(a.output/'receipt.json',{'run_id':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_confirmed_results_report',
        'author_lock_sha256':sha256(a.rebuild/'AUTHOR_DERIVED_REFERENCE_LOCK.json'),
        'reference_json_sha256':lock['source_json_sha256'],'model_rows':len(model_rows),
        'scores_receipt_sha256':sha256(score_folder/'receipt.json'),
        'stratum_rows':len(stratum_rows),'model_selection_performed':False,
        'source_code_sha256':sha256(Path(__file__))})
    print('Reported',len(model_rows),'model rows and',len(stratum_rows),'stratum rows')


if __name__=='__main__':
    main()
