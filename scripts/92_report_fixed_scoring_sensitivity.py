#!/usr/bin/env python3
"""Report threshold sensitivity of fixed predictions without selecting a rule."""
import argparse,csv,importlib.util,json
from datetime import datetime,timezone
from pathlib import Path
import geopandas as gpd
from benchmark.common import ROOT,sha256,site_definitions,write_json
from benchmark.evaluation import aggregate,score_site


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('reference','lock','pretrained','classical','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh output required')
    lock=json.loads(a.lock.read_text())
    if lock['status']!='author_locked' or lock['reference_gpkg_sha256']!=sha256(a.reference):
        raise ValueError('matching author reference lock required')
    spec=importlib.util.spec_from_file_location('ev92',ROOT/'scripts/12_evaluate.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    frames={k:gpd.read_file(a.reference,layer=k) for k in ('single_trees','tree_groups','uncertainty_regions')}
    defs=site_definitions();rule='unmatched_centroid_in_group_or_uncertain'
    variants=[('primary',2.,0.,rule),('centroid_1m',1.,0.,rule),('centroid_3m',3.,0.,rule),
              ('detection_iou_0p25',2.,.25,rule),('detection_iou_0p50',2.,.5,rule),
              ('no_ambiguity_exclusion',2.,0.,'none')]
    rows=[];sources={}
    for split in ('val','test'):
        sites={s:v for s,v in defs.items() if v['split']==split}
        refs={k:{s:[(str(r.label_id),r.geometry) for r in f[f.site_id==s].itertuples()] for s in sites} for k,f in frames.items()}
        for model,path in [('SAT pretrained',a.pretrained/f'{split}_predictions.gpkg'),
                           ('Classical 3m',a.classical/f'{split}_standardized/predictions.gpkg')]:
            preds,_=ev.load_polygons(path,'predictions',sites,'prediction_id');sources[split+'_'+model]=sha256(path)
            for name,distance,iou,ignore in variants:
                records={s:score_site(refs['single_trees'][s],preds[s],
                    [g for _,g in refs['tree_groups'][s]],[g for _,g in refs['uncertainty_regions'][s]],
                    distance,iou,ignore) for s in sites}
                rows.append({'split':split,'model':model,'variant':name,'distance_m':distance,
                             'minimum_detection_iou':iou,'ignore_rule':ignore,**aggregate(list(records.values()))})
    a.output.mkdir(parents=True)
    with (a.output/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Fixed-prediction scoring sensitivity','',
           'I report six diagnostic variants with the same reference, source predictions and scored sites. These variants were specified after initial test exposure and are sensitivity analyses, not predeclared acceptance rules or model selection. The primary 2 m centroid rule with ambiguity exclusion remains unchanged. An additional IoU gate changes detection correspondence, not the matched-IoU formula. Removing exclusions counts unresolved-group predictions without asserting that they are biologically false positives.','',
           '| Test model | Variant | TP | FP | FN | F1 | Ignored |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['split']=='test':lines.append(f"| {r['model']} | {r['variant']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['f1']:.4f} | {r['ignored_predictions']} |")
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write_json(a.output/'receipt.json',{'run_id':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_fixed_scoring_sensitivity',
        'lock_sha256':sha256(a.lock),'reference_sha256':sha256(a.reference),'sources':sources,
        'variants':variants,'rows':len(rows),'primary_rule_changed':False,'model_selection_performed':False,
        'source_code_sha256':sha256(Path(__file__)),
        'outputs':{f.name:sha256(f) for f in a.output.iterdir() if f.is_file()}})
    print('Reported',len(rows),'fixed-prediction sensitivity rows')


if __name__=='__main__':main()
