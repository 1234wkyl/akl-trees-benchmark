#!/usr/bin/env python3
"""Cache public ALTO numeric 3D evidence for retained training weak instances.
No held-out inputs, mesh-derived points, reference edits or training occur.
"""
import argparse,csv,gzip,json,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import requests
from shapely.geometry import shape,Polygon
from shapely import make_valid
from benchmark.common import ROOT,sha256,write_json,site_definitions


def bucket_path(ident,algorithm):
    h=2166136261
    for b in ident.encode('utf8'): h=((h^b)*16777619)&0xffffffff
    if 'mod4096' in algorithm:return f'{h%4096:03x}.json.gz'
    if algorithm!='fnv1a_utf8_mod65536_hex4_split2':raise ValueError('unsupported schema bucket algorithm')
    n=f'{h%65536:04x}';return n[:2]+'/'+n[2:]+'.json.gz'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if (a.output/'intake_receipt.json').exists():ap.error('completed intake is immutable')
    a.output.mkdir(parents=True,exist_ok=True)
    def fetch(url,target):
        last=None
        for attempt in range(3):
            try:
                r=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=45);r.raise_for_status()
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(r.content)
                content=gzip.decompress(r.content) if r.content[:2]==b'\x1f\x8b' else r.content
                return json.loads(content),{'url':url,'sha256':sha256(target),'last_modified':r.headers.get('Last-Modified'),'http_status':r.status_code}
            except Exception as e:last=e;time.sleep(1+attempt)
        raise last
    release,relmeta=fetch('https://alto.tfwelch.com/release-metadata.json',a.output/'release-metadata.json')
    ver=release.get('release',release)['dataset_version']
    schema,schmeta=fetch('https://alto.tfwelch.com/data/tree_details/schema.json?v='+ver,a.output/'schema.json')
    original=ROOT/'outputs/confirmed_reference_20261010/screened_weak_labels'
    refs=json.loads((ROOT/'data/labels/crown_labels_XZ_confirmed_2026-10-10.json').read_text())['sites']
    retained=[r for r in csv.DictReader((original/'instance_screening.csv').open()) if r['retained']=='True']
    defs=site_definitions();matches=[];sources={}
    for row in retained:
        sid=row['site_id'];assert defs[sid]['split']=='train'
        export=ROOT/'outputs/alto_review_20261006/comparison'/f'{sid}_public_crowns.geojson'
        sources[str(export.relative_to(ROOT))]=sha256(export)
        crowns=[(str(f['properties']['tree_id']),make_valid(shape(f['geometry']))) for f in json.loads(export.read_text())['features']]
        d={**row,'treeID':int(row['treeID']),'alto_id':None,'geometry_iou':None,'match_basis':'unmatched'}
        if row['label_source']=='ALTO':
            ident=row['label_id'][5:];assert ident in dict(crowns);d.update(alto_id=ident,geometry_iou=1.,match_basis='direct source ID')
        else:
            lab=next(x for x in refs[sid]['labels'] if x['id']==row['label_id']);poly=make_valid(Polygon(lab['world']))
            pairs=[(poly.intersection(g).area/poly.union(g).area,k) for k,g in crowns if poly.intersects(g)]
            if pairs:
                v,k=max(pairs);d.update(alto_id=k,geometry_iou=v,match_basis='largest polygon IoU; auxiliary only')
        matches.append(d)
    ids=sorted({r['alto_id'] for r in matches if r['alto_id']})
    paths={bucket_path(k,schema['bucket_algorithm']) for k in ids}
    def get(path):
        try:return path,fetch('https://alto.tfwelch.com/data/tree_details/'+path+'?v='+ver,a.output/'buckets'/path),None
        except Exception as e:return path,None,str(e)
    buckets={};meta={};errors={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path,result,error in pool.map(get,sorted(paths)):
            if error:errors[path]=error
            else:buckets[path]=result[0];meta[path]=result[1]
    records={}
    for ident in ids:
        packed=buckets.get(bucket_path(ident,schema['bucket_algorithm']),{}).get(ident)
        if packed is None:continue
        datasets={}
        for ds in schema['datasets']:
            if ds['key'] not in packed:continue
            def unpack(values):return {k:values[i] if i<len(values) else None for i,k in enumerate(ds['fields'])}
            datasets[ds['key']]=[unpack(v) for v in packed[ds['key']]] if ds.get('multiple') else unpack(packed[ds['key']])
        records[ident]={'tree_id':ident,'datasets':datasets,'snapshot':schema['generated_utc']}
    write_json(a.output/'instance_matches.json',matches);write_json(a.output/'records.json',records)
    result={'run_id':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_alto_training_3d_intake',
      'release':release.get('release',release),'detail_schema_snapshot':schema['generated_utc'],
      'release_http':relmeta,'schema_http':schmeta,'buckets_http':meta,'download_errors':errors,
      'training_instances':len(matches),'direct_alto_instances':sum(r['label_source']=='ALTO' for r in matches),
      'matched_ids_requested':len(ids),'records_downloaded':len(records),'heldout_inputs_used':False,
      'raw_per_return_labels_available':False,'schema_contains_foliage_profile':any('foliage_profile' in d['fields'] for d in schema['datasets']),
      'source_hashes':sources,'source_script_sha256':sha256(Path(__file__))}
    write_json(a.output/'intake_receipt.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['buckets_http','source_hashes']},indent=2))

if __name__=='__main__':main()
