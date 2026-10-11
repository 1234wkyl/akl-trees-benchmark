#!/usr/bin/env python3
"""Compare the fixed Alto-height revision with the preserved weak pilot."""
import argparse
import csv
import json
from pathlib import Path
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark.common import sha256, write_json
from benchmark.evaluation import aggregate, paired_site_bootstrap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('revised', 'previous', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('fresh output required')
    receipt = json.loads((a.revised/'report/receipt.json').read_text())
    for name, digest in receipt['output_sha256'].items():
        if sha256(a.revised/'report'/name) != digest:
            raise ValueError('report hash mismatch: '+name)
    results = {}
    rows = []
    sources = {}
    for split in ('val', 'test'):
        newpath = a.revised/f'{split}_comparison.json'
        oldpath = a.previous/f'{split}_comparison.json'
        new = json.loads(newpath.read_text())
        old = json.loads(oldpath.read_text())
        for key in ('reference_sha256', 'policy_sha256'):
            if new[key] != old[key]:
                raise ValueError('revision comparison requires the same '+key)
        pred = a.revised/f'{split}_predictions.gpkg'
        if sha256(pred) != new['prediction_sha256']['weak']:
            raise ValueError('revised prediction changed')
        frame = gpd.read_file(pred, layer='predictions')
        if frame.crs.to_epsg() != 2193:
            raise ValueError('revised prediction CRS must be EPSG:2193')
        checks = {}
        for name, records in (
            ('classical', new['classical_by_site']),
            ('pretrained', new['pretrained_by_site']),
            ('previous_weak', old['weak_by_site']),
            ('revised_weak', new['weak_by_site']),
        ):
            checks[name] = {'summary': aggregate(list(records.values())),
                           'bootstrap': paired_site_bootstrap(records, iterations=2000, seed=42)}
        pairs = {name: paired_site_bootstrap(new['weak_by_site'], baseline, 2000, 42)
                 for name, baseline in (
                     ('previous_weak', old['weak_by_site']),
                     ('pretrained', new['pretrained_by_site']),
                     ('classical', new['classical_by_site']))}
        results[split] = {'models': checks, 'revision_minus_comparator': pairs}
        for site, record in new['weak_by_site'].items():
            before = aggregate([old['weak_by_site'][site]])
            after = aggregate([record])
            rows.append({'split': split, 'site_id': site,
                         **{k+'_before': before[k] for k in ('tp','fp','fn','ignored_predictions','f1')},
                         **{k+'_after': after[k] for k in ('tp','fp','fn','ignored_predictions','f1')}})
        sources[split] = {'revised_comparison_sha256': sha256(newpath),
                          'previous_comparison_sha256': sha256(oldpath),
                          'reference_sha256': new['reference_sha256'],
                          'policy_sha256': new['policy_sha256'],
                          'previous_checkpoint_sha256': old['checkpoint_sha256'],
                          'revised_checkpoint_sha256': new['checkpoint_sha256'],
                          'revised_predictions_epsg': 2193}
    a.output.mkdir(parents=True)
    write_json(a.output/'paired_revision_comparison.json', results)
    with (a.output/'site_revision_changes.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    labels = ['Classical', 'SAT pretrained', 'Weak pilot', 'Weak revision']
    names = ['classical', 'pretrained', 'previous_weak', 'revised_weak']
    colors = ['#3c6d4b', '#497da5', '#a78c60', '#805b9b']
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3), sharey=True)
    for ax, metric, title in zip(axes, ['precision','recall','f1'], ['Precision','Recall','F1']):
        for i, (name,color) in enumerate(zip(names, colors)):
            r = results['test']['models'][name]; v = r['summary'][metric]
            ci = r['bootstrap']['intervals'][metric]
            ax.errorbar(v, i, xerr=[[v-ci['lower']], [ci['upper']-v]], fmt='o',
                        color=color, capsize=4, markersize=6)
        ax.set_title(title); ax.set_xlim(0,0.7); ax.set_xticks([0,.2,.4,.6])
        ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(4), labels); axes[0].invert_yaxis()
    fig.suptitle('Fixed models on the same 170 test crowns',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.92])
    fig.savefig(a.output/'figure_7_6_alto_revision.png',dpi=240,bbox_inches='tight')
    fig.savefig(a.output/'figure_7_6_alto_revision.svg',bbox_inches='tight');plt.close(fig)
    write_json(a.output/'receipt.json', {'status':'COMPLETE','sources':sources,
               'model_retuned':False,'inference_restarted':False,
               'scope':'Post-revision exploratory comparison; previous test exposure retained',
               'source_code_sha256':sha256(Path(__file__)),
               'output_sha256':{f.name:sha256(f) for f in a.output.iterdir() if f.is_file()}})
    for split in results:
        print(split, 'revision versus previous weak:',
              results[split]['revision_minus_comparator']['previous_weak']['f1_improvement'])


if __name__ == '__main__':
    main()
