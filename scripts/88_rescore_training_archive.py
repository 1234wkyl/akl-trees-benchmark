#!/usr/bin/env python3
"""Re-evaluate all seven existing crown conversions on the confirmed train set.

This is a sensitivity table, not permission to retune the already exposed test
comparison. No point inference or polygon conversion is repeated.
"""
import argparse
import csv
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
from benchmark.common import ROOT,sha256,site_definitions,write_json
from benchmark.evaluation import aggregate,score_site


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh output required')
    defs={s:v for s,v in site_definitions().items() if v['split']=='train'}
    ref=gpd.read_file(a.reference,layer='all_objects');a.output.mkdir(parents=True)
    results,by_config={},{}
    with tarfile.open(a.archive) as tar:
        manifest_name=next(n for n in tar.getnames() if n.endswith('/ARCHIVE_MANIFEST.json'))
        prefix=manifest_name.rsplit('/',1)[0]+'/'
        manifest=json.load(tar.extractfile(manifest_name))['included']
        for name in sorted(tar.getnames()):
            if not name.endswith('/standardized/predictions.gpkg'):continue
            parts=name.split('/')
            if 'training_conversion_prefreeze249_v1' not in parts:continue
            i=parts.index('training_conversion_prefreeze249_v1');sid=parts[i+1]
            if sid not in defs or parts[i+2]!='fraction_1':continue
            config=parts[i+3];contents=tar.extractfile(name).read()
            digest=hashlib.sha256(contents).hexdigest()
            if digest!=manifest[name[len(prefix):]]:raise ValueError('archived prediction hash mismatch')
            dest=a.output/'reused_predictions'/sid/config/'predictions.gpkg'
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(contents)
            frame=gpd.read_file(dest,layer='predictions')
            if frame.crs.to_epsg()!=2193 or not set(frame.site_id)<={sid}:raise ValueError('prediction site/CRS mismatch')
            sitebox=box(*defs[sid]['bbox_epsg2193']);objects=ref[ref.site_id==sid]
            own={c:[(str(r.label_id),r.geometry) for r in objects[objects['class']==c].itertuples()] for c in ('tree','tree_group','uncertain')}
            pred=[(str(r.prediction_id),r.geometry.intersection(sitebox)) for r in frame.itertuples()]
            score=score_site(own['tree'],pred,[g for _,g in own['tree_group']],[g for _,g in own['uncertain']],2.,0.,'unmatched_centroid_in_group_or_uncertain')
            if sid in by_config.setdefault(config,{}):raise ValueError('duplicate archive prediction')
            by_config[config][sid]=score
            results[sid+'/'+config]={'prediction_sha256':digest,'score':score}
    if len(by_config)!=7 or any(set(v)!=set(defs) for v in by_config.values()):raise ValueError('expected all seven configs and eight training sites')
    rows=[{'config_id':k,'reference_singles':sum(s['n_reference'] for s in v.values()),**aggregate(list(v.values()))} for k,v in sorted(by_config.items())]
    with (a.output/'training_conversion_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(a.output/'receipt.json',{'run_id':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_confirmed_training_rescore',
        'reference_sha256':sha256(a.reference),'archive_sha256':sha256(a.archive),
        'training_sites':list(defs),'conditions':rows,'by_site':results,
        'conversion_selection_changed':False,'heldout_used_for_tuning':False,
        'previously_fixed_config_id':'convex_b100','GPU_rerun':False,'source_code_sha256':sha256(Path(__file__))})
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
