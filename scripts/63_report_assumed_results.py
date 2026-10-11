#!/usr/bin/env python3
"""Export a conditional comparison and measured throughput; no model changes."""
import argparse,csv,json,textwrap
from pathlib import Path
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from benchmark.common import ROOT,sha256,write_json,site_definitions
from benchmark.evaluation import aggregate,paired_site_bootstrap


def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def timing_for(pattern,output_fragment):
    found=[]
    for d in (ROOT/'outputs/benchmarks').glob(pattern):
        p=d/'timing.json';meta=d/'run_metadata.txt'
        if p.exists() and meta.exists() and output_fragment in meta.read_text():
            t=json.loads(p.read_text())
            if t['exit_code']==0:found.append((d,t))
    if len(found)!=1:raise ValueError('require one successful bound producer timing: '+pattern)
    d,t=found[0];return {'run_id':d.name,'wall_seconds':t['wall_seconds'],
        'timing_sha256':sha256(d/'timing.json'),'metadata_sha256':sha256(d/'run_metadata.txt')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['comparison','queue-root','classical-root','coverage-plan','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh report directory required')
    r=json.loads(a.comparison.read_text())
    if (r['status']!='HYPOTHETICAL TEST COMPARISON; NOT AN APPROVED BENCHMARK'
            or r['reference_frozen'] is not False or r['actual_independent_agreement_available'] is not False):
        raise ValueError('this reporter is explicitly assumption-only')
    q=json.loads((a.queue_root/'queue_receipt.json').read_text());plan=json.loads(a.coverage_plan.read_text())
    if q['status']!='COMPLETED' or len(q['tiles'])!=len(q['expected_tiles']) or any(x['status']!='AUDITED' for x in q['tiles']):
        raise ValueError('complete audited queue required')
    a.output.mkdir(parents=True);(a.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    baseline_ci=paired_site_bootstrap(r['classical_by_site'],iterations=2000,seed=42)
    models=[('Classical provisional (3.5 m)',r['classical_provisional_summary'],baseline_ci,r['classical_by_site']),
            ('SegmentAnyTree pretrained',r['pretrained_summary'],r['bootstrap'],r['pretrained_by_site'])]
    overall=[];sites=[];errors=[]
    metrics=['precision','recall','f1','median_matched_iou','over_segmentation_rate','under_segmentation_rate']
    definitions=site_definitions()
    for model,s,ci,records in models:
        row={'model':model,'reference_frozen':False,**{k:s[k] for k in ['tp','fp','fn','ignored_predictions']}}
        for k in metrics:row.update({k:s[k],k+'_ci_lower':ci['intervals'][k]['lower'],k+'_ci_upper':ci['intervals'][k]['upper']})
        overall.append(row)
        for sid,v in sorted(records.items()):
            sites.append({'model':model,'site_id':sid,'stratum':definitions[sid]['stratum'],
                'n_reference':v['n_reference'],'n_predictions':v['n_predictions'],'ignored':v['ignored'],
                **{k:v[k] for k in ['tp','fp','fn']},**{k:aggregate([v])[k] for k in metrics}})
            for kind,key in [('false_negative','unmatched_reference_ids'),('false_positive','false_positive_ids'),('ignored','ignored_prediction_ids')]:
                for oid in v[key]:errors.append({'model':model,'site_id':sid,'stratum':definitions[sid]['stratum'],'kind':kind,'object_id':oid})
    table(a.output/'overall_metrics.csv',overall);table(a.output/'per_site_metrics.csv',sites)
    if errors:table(a.output/'error_object_ids.csv',errors)
    write_json(a.output/'classical_bootstrap.json',baseline_ci)
    tiles={v['tile']:v for v in plan['tiles']};durations=[];footprints=[]
    for v in q['tiles']:
        stem=Path(v['tile']).stem
        durations.append({'tile':v['tile'],'points':v['points'],**timing_for('*_sat_pretrained_'+stem+'_training_tile_*',str(a.queue_root/stem/'execution'))})
        footprints.append(box(*tiles[v['tile']]['bbox_epsg2193']))
    sat_seconds=sum(d['wall_seconds'] for d in durations);area=unary_union(footprints).area/1e6
    classic=timing_for('*_fixed_classical_heldout_predictions_*',str(a.classical_root))
    input_plan=json.loads((ROOT/'outputs/nectar/20261006T073500Z_classical_heldout_inputs_busy_backoff/plan.json').read_text())
    classic_area=unary_union([box(*b) for b in input_plan['original_chunks']]).area/1e6
    cost={'status':'MEASURED OPERATIONAL THROUGHPUT; NOT PROVIDER BILLING OR AREA-MATCHED MODEL COST',
        'pretrained':{'tiles':len(durations),'model_wrapper_wall_seconds':sat_seconds,'allocated_device_wall_hours':sat_seconds/3600,
            'raw_header_envelope_union_km2':area,'allocated_device_wall_hours_per_header_envelope_km2':sat_seconds/3600/area,
            'reconciled_points':sum(d['points'] for d in durations),'tile_timings':durations,
            'includes':'container startup, checkpoint preflight, model preprocessing/inference/postprocessing and I/O; excludes staging, later coordinate reconciliation, site assembly, environment build and idle time',
            'area_limit':'Union of original raw point-header rectangles is a technical coverage proxy, not surveyed canopy area or complete ground coverage.'},
        'classical':{**classic,'selected_grid_envelope_union_km2':classic_area,'wall_hours_per_grid_envelope_km2':classic['wall_seconds']/3600/classic_area,
            'includes':'fixed detector, fresh height sampling, crown segmentation, integrity hashing and I/O; excludes prior mask/imagery acquisition and standardization',
            'area_limit':'Full selected chunk envelopes include nodata outside supplied reference CHM chips. Not the same processed support as SAT tile envelopes.'},
        'hourly_price':None,'billed_gpu_hours':None,'service_units':None,
        'comparison_limit':'No area-matched cost ratio or financial saving is asserted. CPU preparation overlapped some GPU inference; wall receipts are not isolated kernel timings.'}
    write_json(a.output/'compute_receipt.json',cost)
    def cell(s,ci,key):
        v=s[key];lo=ci['intervals'][key]['lower'];hi=ci['intervals'][key]['upper']
        return 'NA' if v is None else f'{v:.3f} [{lo:.3f}, {hi:.3f}]'
    lines=['# Conditional held-out comparison: 7 October 2026','',
        'I evaluated fixed predictions against the 6 October reference assumption. Actual independent annotations, adjudication and formal freeze remain pending. The classical model is a provisional reconstruction, not a certified historical baseline.','',
        f"The eight test sites contain **{sum(v['n_reference'] for v in r['pretrained_by_site'].values())} single-tree references**. Groups and unresolved regions follow the fixed ignore policy. Confidence intervals use 2,000 whole-site resamples, seed 42; there is one test site per stratum, so stratum-specific confidence intervals are not estimable from independent sites.",'',
        '| Model | Precision [95% CI] | Recall [95% CI] | F1 [95% CI] | Median matched IoU [95% CI] | TP / FP / FN | Ignored predictions |',
        '|---|---|---|---|---|---|---|']
    for name,s,ci,_ in models:lines.append('| '+name+' | '+' | '.join(cell(s,ci,k) for k in ['precision','recall','f1','median_matched_iou'])+f" | {s['tp']} / {s['fp']} / {s['fn']} | {s['ignored_predictions']} |")
    delta=r['bootstrap']['f1_improvement'];lines+=['',f"Paired F1 difference (pretrained minus provisional classical): **{r['pretrained_summary']['f1']-r['classical_provisional_summary']['f1']:.3f}**, 95% CI **[{delta['lower']:.3f}, {delta['upper']:.3f}]**. This is a conditional numerical contrast, not a formal winner declaration.",'',
        '## Stratum recall','', '| Stratum | Single-tree references | Classical provisional recall | Pretrained recall |','|---|---:|---:|---:|']
    strata=[]
    for sid in sorted(r['site_ids']):
        c=r['classical_by_site'][sid];m=r['pretrained_by_site'][sid];label=definitions[sid]['stratum'];strata.append((label,m['n_reference'],aggregate([c])['recall'],aggregate([m])['recall']))
        lines.append(f"| {label} ({sid}) | {m['n_reference']} | {aggregate([c])['recall']:.3f} | {aggregate([m])['recall']:.3f} |")
    lines+=['','## Measured runtime','',f"The 19-tile pretrained batch used **{sat_seconds/3600:.3f} hours** of model-wrapper wall time on the V100. Its original raw header-envelope union is **{area:.4f} km²**, giving **{sat_seconds/3600/area:.3f} allocated-device wall-hours per envelope km²**. This is a technical throughput proxy, not a provider bill or GPU active-kernel measurement.",'',
        f"The provisional classical producer took **{classic['wall_seconds']} seconds**. Its selected grid-envelope union is **{classic_area:.4f} km²**, including nodata/context outside the supplied CHM chips. The two area denominators and stage scopes differ, so no direct area-matched cost ratio is reported. Prices, billed GPU-hours and service units remain unavailable.",'',
        '## Error analysis and interpretation limits','',
        'The error table lists every scored false positive, false negative and ignored prediction, rather than selecting examples to improve the appearance of performance. Per-site tables include geometric over/under-segmentation diagnostics. Inspect corresponding RGB/CHM and model geometry before attributing a particular error to a building, height band, tree group or sensor density.','',
        'Matched IoU describes only matched crowns; it does not represent missed trees. Ignored counts quantify excluded ambiguity but do not estimate the number of true individual trees inside groups. Selective non-tree labels do not establish exhaustive hard-negative coverage.','',
        'These held-out results are now exposed. I will not change model, conversion, filtering or matching settings using this comparison. Any eventual reference corrections must be documented, and the already generated predictions retained. No actual inter-rater metric, historical polygon equivalence, fine-tuning result or formal acceptance is claimed.']
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(strata));width=.36
    ax.bar(x-width/2,[v[2] for v in strata],width,label='Classical provisional',color='#426a8c')
    ax.bar(x+width/2,[v[3] for v in strata],width,label='SegmentAnyTree pretrained',color='#bd7545')
    ax.set_xticks(x,['\n'.join(textwrap.wrap(v[0],15))+f'\n(n={v[1]})' for v in strata],fontsize=8)
    ax.set_ylim(0,1);ax.set_ylabel('Recall');ax.set_title('Conditional test recall: 6 October reference assumption');ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(a.output/'stratum_recall.png',dpi=200);plt.close(fig)
    write_json(a.output/'report_receipt.json',{'status':'CONDITIONAL RESULT TABLES AND OPERATIONAL THROUGHPUT; NO FORMAL ACCEPTANCE',
        'comparison_sha256':sha256(a.comparison),'queue_receipt_sha256':sha256(a.queue_root/'queue_receipt.json'),
        'coverage_plan_sha256':sha256(a.coverage_plan),'model_configuration_changed':False,'reference_frozen':False,
        'output_sha256':{f.name:sha256(f) for f in a.output.iterdir() if f.is_file()}})
    print('Conditional tables, error IDs, bootstrap intervals and measured throughput exported:',a.output)
if __name__=='__main__':main()
