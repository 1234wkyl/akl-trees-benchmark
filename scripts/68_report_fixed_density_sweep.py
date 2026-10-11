#!/usr/bin/env python3
"""Report every predeclared retention condition; never select a condition after scoring."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark.common import sha256,write_json
from benchmark.evaluation import paired_site_bootstrap

METRICS=('precision','recall','f1','median_matched_iou')

def assemble_rows(comparisons,counts):
    full=comparisons[1.0];rows=[];paired={}
    for fraction in (1.0,.75,.5,.25,.125):
        c=comparisons[fraction]
        if (c['reference_frozen'] is not False or c['actual_independent_agreement_available'] is not False
                or c['sources']['reference']!=full['sources']['reference']
                or c['sources']['policy']!=full['sources']['policy']
                or c['sources']['classical']!=full['sources']['classical']
                or set(c['site_ids'])!=set(full['site_ids'])):
            raise ValueError('identical conditional reference, scoring policy, sites and fixed comparator required')
        r={'retained_fraction':fraction,'model_input_points':counts[fraction],
           'tp':c['pretrained_summary']['tp'],'fp':c['pretrained_summary']['fp'],
           'fn':c['pretrained_summary']['fn'],'ignored_predictions':c['pretrained_summary']['ignored_predictions']}
        for metric in METRICS:
            r[metric]=c['pretrained_summary'][metric]
            interval=c['bootstrap']['intervals'][metric]
            r[metric+'_ci_lower']=interval['lower'];r[metric+'_ci_upper']=interval['upper']
        b=paired_site_bootstrap(c['pretrained_by_site'],full['pretrained_by_site'],2000,42)
        r['f1_difference_from_full']=None if r['f1'] is None or full['pretrained_summary']['f1'] is None else r['f1']-full['pretrained_summary']['f1']
        r['f1_difference_ci_lower']=b['f1_improvement']['lower']
        r['f1_difference_ci_upper']=b['f1_improvement']['upper']
        paired[str(fraction)]=b;rows.append(r)
    return rows,paired

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('fresh density result report required')
    protocol=json.loads((a.base/'density_protocol.json').read_text());sweep=json.loads((a.base/'sweep_receipt.json').read_text())
    if (sweep['status']!='COMPLETE' or len(sweep['completed'])!=4
            or sweep['density_protocol_sha256']!=sha256(a.base/'density_protocol.json')):raise ValueError('completed fixed sweep required')
    roots={1.0:Path(protocol['full_density_reused'])}
    roots.update({f:a.base/f'fraction_{f:g}' for f in protocol['fractions']})
    comparisons={};counts={};sources={}
    for f,root in roots.items():
        path=root/'hypothetical_comparison_v1/hypothetical_comparison.json'
        if f!=1.0 and sha256(path)!=next(r['comparison_sha256'] for r in sweep['completed'] if r['fraction']==f):raise ValueError('comparison changed')
        q=json.loads((root/'queue_receipt.json').read_text())
        if q['status']!='COMPLETED' or len(q['tiles'])!=19 or any(r['status']!='AUDITED' for r in q['tiles']):raise ValueError('complete audited input coverage required')
        comparisons[f]=json.loads(path.read_text());counts[f]=sum(r['points'] for r in q['tiles'])
        sources[str(f)]={'comparison_sha256':sha256(path),'queue_sha256':sha256(root/'queue_receipt.json')}
    rows,paired=assemble_rows(comparisons,counts);a.output.mkdir(parents=True)
    with (a.output/'density_metrics.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(a.output/'paired_against_full_density.json',paired)
    fig,axes=plt.subplots(1,3,figsize=(11,3.4))
    for ax,metric in zip(axes,('precision','recall','f1')):
        valid=[r for r in reversed(rows) if r[metric] is not None]
        x=[r['retained_fraction']*100 for r in valid];y=[r[metric] for r in valid]
        ax.plot(x,y,'o-',label='SegmentAnyTree')
        ax.fill_between(x,[r[metric+'_ci_lower'] for r in valid],[r[metric+'_ci_upper'] for r in valid],alpha=.2)
        ax.axhline(comparisons[1.0]['classical_provisional_summary'][metric],ls='--',color='gray',label='Fixed full-density CHM comparator')
        ax.set(xlabel='Retained returns (%)',ylabel=metric.capitalize(),ylim=(0,1));ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Conditional on the 6 October author reference; paired site bootstrap, 95% CI')
    fig.tight_layout();fig.savefig(a.output/'density_curve.png',dpi=180);plt.close(fig)
    (a.output/'RESULTS.md').write_text('''# Fixed density transfer experiment

I reused the completed full-density predictions and evaluated all four predeclared nested return-retention conditions with the same pretrained model, source, checkpoint, crown conversion and scoring policy. `density_metrics.csv` contains every condition and confidence interval; no condition is selected using these held-out results. Differences from full density use paired whole-site bootstrap resampling (2,000 draws, seed 42).

These remain conditional results, because the independent annotations and actual reference freeze are pending. Return thinning is not pulse thinning or a new sensor acquisition. The classical dashed line uses the existing full-density CHM and is a fixed comparator; it is not a measured classical response to thinning. Input point counts sum all 19 tile files and may count overlapping records; they are not whole-site unique densities. Confidence intervals reflect site sampling at one seed, not uncertainty across thinning seeds. Subsequent adjudicated scoring must disclose that these test outputs were already inspected.
''')
    write_json(a.output/'report_receipt.json',{'status':'ALL PREDECLARED CONDITIONS REPORTED; CONDITIONAL REFERENCE',
        'sources':sources,'protocol_sha256':sha256(a.base/'density_protocol.json'),'reference_frozen':False,
        'full_density_rerun':False,'classical_density_response_measured':False,
        'outputs':{p.name:sha256(p) for p in a.output.iterdir() if p.is_file()}})
    print('All five retention conditions reported:',a.output)
if __name__=='__main__':main()
