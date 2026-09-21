#!/usr/bin/env python
"""Recalculate preparation numbers in Chapters 3/4 from original inputs.

Run: python scripts/24_reproduce_manuscript_numbers.py --output NEW_FOLDER
Add --recount-laz to decompress all 39 files and reproduce site return density.
Without that flag the previously measured CSV is explicitly identified as the
source for density; it is not presented as a new point-cloud measurement.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from shapely.geometry import Polygon, box

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT.parent/'Xi_Zhang_Benchmark'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def csvout(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--recount-laz',action='store_true');args=ap.parse_args()
    if args.output.exists():ap.error('use a new output folder')
    args.output.mkdir(parents=True)
    source=ROOT/'data/labels/crown_labels_XZ_2026-08-13.json'
    labels=json.loads(source.read_text())
    sites={s['site_id']:s for s in json.loads((ROOT/'data/sites/reference_sites.json').read_text())['sites']}
    objects=[];repairs=[]
    for sid,rec in labels['sites'].items():
        for l in rec['labels']:
            p=Polygon(l['world']);q=p if p.is_valid else p.buffer(0)
            objects.append({'site_id':sid,'label_id':l['id'],'class':l['cls'],'split':sites[sid]['split'],
                'stratum':sites[sid]['stratum'],'area_m2':q.area,'raw_geometry_valid':p.is_valid,
                'outside_area_m2':q.difference(box(*sites[sid]['bbox_epsg2193'])).area})
            if not p.is_valid:repairs.append(abs(q.area-p.area))
    csvout(args.output/'object_numbers.csv',objects)
    matrix=[]
    for sid,s in sites.items():
        cc=Counter(r['class'] for r in objects if r['site_id']==sid)
        matrix.append(dict(site_id=sid,split=s['split'],stratum=s['stratum'],**{k:cc[k] for k in ['tree','tree_group','not_tree','uncertain']},total=sum(cc.values())))
    csvout(args.output/'counts_by_split_stratum.csv',matrix)
    assetchecks=[]
    for f in json.loads((BUNDLE/'manifest.json').read_text())['files']:
        p=BUNDLE/f['path'];ok=p.exists() and p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
        assetchecks.append({'path':f['path'],'size_and_sha256_pass':ok})
    csvout(args.output/'asset_checksums.csv',assetchecks)
    if not all(r['size_and_sha256_pass'] for r in assetchecks):raise ValueError('supplied inputs changed')
    total=0;tiles=[];points={sid:set() for sid in sites}
    for p in sorted((BUNDLE/'laz').glob('*.laz')):
        with laspy.open(p) as f:
            h=f.header;total+=h.point_count
            relevant=[sid for sid,s in sites.items() if box(*s['bbox_epsg2193']).intersects(box(h.mins[0],h.mins[1],h.maxs[0],h.maxs[1]))]
            read=0
            if args.recount_laz:
                for part in f.chunk_iterator(500000):
                    read+=len(part);x,y,z=np.asarray(part.x),np.asarray(part.y),np.asarray(part.z)
                    for sid in relevant:
                        x0,y0,x1,y1=sites[sid]['bbox_epsg2193'];m=(x>=x0)&(x<x1)&(y>=y0)&(y<y1)
                        points[sid].update(zip(np.round(x[m],4),np.round(y[m],4),np.round(z[m],4)))
                if read!=h.point_count:raise ValueError('decompression count mismatch')
            tiles.append({'file':p.name,'header_points':h.point_count,'points_decompressed':read if args.recount_laz else None})
        print(p.name,flush=True)
    csvout(args.output/'laz_counts.csv',tiles)
    density_source=ROOT/'docs/review_2026-09-17/site_point_density.csv'
    if args.recount_laz:
        density=[{'site_id':sid,'split':s['split'],'stratum':s['stratum'],
            'unique_xyz_returns':len(points[sid]),'area_m2':box(*s['bbox_epsg2193']).area,
            'density_all_returns':len(points[sid])/box(*s['bbox_epsg2193']).area} for sid,s in sites.items()]
        old={r['site_id']:int(r['unique_xyz_returns']) for r in csv.DictReader(density_source.open())}
        if any(old[r['site_id']]!=r['unique_xyz_returns'] for r in density):raise ValueError('new density count differs from manuscript source')
    else:
        density=list(csv.DictReader(density_source.open()))
    csvout(args.output/'site_point_density.csv',density)
    values=np.array([float(r['density_all_returns']) for r in density])
    chm=[]
    with rasterio.open(BUNDLE/'chm/reference_sites.vrt') as src:
        for sid,s in sites.items():
            win=from_bounds(*s['bbox_epsg2193'],src.transform);a=src.read(1,window=win,masked=True)
            with rasterio.open(ROOT/f'data/sites/chips/{sid}/chm.tif') as dst:
                b=dst.read(1);out=np.zeros_like(b)
                reproject(a.filled(src.nodata),out,src_transform=src.window_transform(win),src_crs=src.crs,
                    dst_transform=dst.transform,dst_crs=dst.crs,resampling=Resampling.bilinear)
            chm.append({'site_id':sid,'native_valid_pixels':int(a.count()),'native_total_pixels':a.size,
                'bilinear_chip_equal':bool(np.array_equal(b,out)),'max_abs_difference_m':float(np.max(np.abs(b-out)))})
    csvout(args.output/'chm_checks.csv',chm)
    areas=np.array([r['area_m2'] for r in objects if r['class']=='tree'])
    result={'annotation_classes':dict(Counter(r['class'] for r in objects)),
        'single_trees_by_split':{sp:sum(r['class']=='tree' and r['split']==sp for r in objects) for sp in ['train','val','test']},
        'objects_by_split':{sp:sum(r['split']==sp for r in objects) for sp in ['train','val','test']},
        'single_crown_area_m2':{'min':float(areas.min()),'median':float(np.median(areas)),'mean':float(areas.mean()),'max':float(areas.max())},
        'large_polygons_above_900m2':sum(r['area_m2']>900 for r in objects),
        'raw_invalid_geometries':len(repairs),'maximum_buffer0_area_change_m2':max(repairs,default=0),
        'objects_outside_site_by_more_than_0_01m2':sum(r['outside_area_m2']>.01 for r in objects),
        'maximum_outside_area_m2':max(r['outside_area_m2'] for r in objects),
        'bundle_manifest_files':len(assetchecks),'bundle_manifest_failures':sum(not r['size_and_sha256_pass'] for r in assetchecks),
        'laz_tiles':len(tiles),'laz_total_header_returns':int(total),'all_laz_fully_decompressed_this_run':args.recount_laz,
        'density':dict(zip(['min','q1','median','q3','max'],map(float,np.quantile(values,[0,.25,.5,.75,1])))),
        'density_definition':'All classes/all return numbers; half-open site XY; deduplicate XYZ rounded to 4 decimals across tile overlaps; unique count / site area. Observed unique return density, not pulse density.',
        'density_source': 'raw LAZ recount this run' if args.recount_laz else str(density_source.relative_to(ROOT)),
        'density_previous_csv_sha256':sha(density_source),'chm_sites_equal':sum(r['bilinear_chip_equal'] for r in chm),
        'source_label_sha256':sha(source),'site_definition_sha256':sha(ROOT/'data/sites/reference_sites.json'),
        'input_manifest_sha256':sha(BUNDLE/'manifest.json'),'program_sha256':sha(Path(__file__)),
        'model_results_or_independent_agreement_computed':False,'reference_frozen':False,
        'numpy_version':np.__version__,'laspy_version':laspy.__version__,'rasterio_version':rasterio.__version__}
    (args.output/'numbers.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
