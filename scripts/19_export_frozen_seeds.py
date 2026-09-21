#!/usr/bin/env python
"""Read-only export of the historical crown-input schema near training sites."""
import argparse
import sqlite3
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point
from benchmark.common import sha256, site_definitions, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--database',type=Path,required=True)
    ap.add_argument('--halo-m',type=float,default=25)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    import math
    if not math.isfinite(args.halo_m) or args.halo_m<14:ap.error('halo must cover the historical maximum 14 m crown radius')
    if args.output.exists():ap.error('use a fresh output folder')
    db=args.database.resolve()
    wal=Path(str(db)+'-wal')
    if not db.exists() or (wal.exists() and wal.stat().st_size):raise ValueError('need an immutable local frozen database without a nonempty WAL')
    digest=sha256(db)
    conn=sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True);conn.execute('PRAGMA query_only=ON')
    sites={sid:s for sid,s in site_definitions().items() if s['split']=='train'}
    records=[]
    try:
        tc={r[1] for r in conn.execute('PRAGMA table_info(trees)')}
        lc={r[1] for r in conn.execute('PRAGMA table_info(tree_lidar_pilot)')}
        if not {'tree_id','source_primary'}<=tc or not {'tree_id','x_2193','y_2193','chm_local_max_2m_m'}<=lc:
            raise ValueError('frozen schema differs from historical crown-input schema; inspect before adaptation')
        for sid,s in sites.items():
            x0,y0,x1,y1=s['bbox_epsg2193'];halo=args.halo_m
            query='''SELECT t.tree_id,t.source_primary,l.x_2193,l.y_2193,l.chm_local_max_2m_m
                     FROM trees t JOIN tree_lidar_pilot l ON t.tree_id=l.tree_id
                     WHERE l.chm_local_max_2m_m IS NOT NULL
                       AND l.x_2193>=? AND l.x_2193<? AND l.y_2193>=? AND l.y_2193<?'''
            for tid,source,x,y,height in conn.execute(query,(x0-halo,x1+halo,y0-halo,y1+halo)):
                if tid is None or not all(math.isfinite(v) for v in [x,y,height]):raise ValueError('invalid frozen seed attributes')
                records.append({'site_id':sid,'tree_id':str(tid),'source_primary':source,
                    'x_2193':x,'y_2193':y,'chm_local_max_2m_m':height,
                    'seed_inside_site':x0<=x<x1 and y0<=y<y1})
    finally:conn.close()
    args.output.mkdir(parents=True)
    frame=gpd.GeoDataFrame(records,columns=['site_id','tree_id','source_primary','x_2193','y_2193','chm_local_max_2m_m','seed_inside_site'],
                           geometry=[Point(r['x_2193'],r['y_2193']) for r in records],crs=2193)
    if frame.duplicated(['site_id','tree_id']).any():raise ValueError('multiple frozen inventory rows per tree/site; inspect schema before crown adaptation')
    target=args.output/'training_seeds.gpkg';frame.to_file(target,layer='training_seeds',driver='GPKG')
    write_json(args.output/'provenance.json',{'database_sha256':digest,'source_database_changed':False,
        'input_schema':'historical trees JOIN tree_lidar_pilot, not confirmed frozen pipeline',
        'training_site_ids':sorted(sites),'halo_m':args.halo_m,'seed_records':len(frame),
        'output_sha256':sha256(target),'query_mode':'SQLite mode=ro + query_only',
        'baseline_reproduction_verified':False,'note':'These are model-input inventory seeds, not independent stem ground truth. Nearby context seeds are retained to avoid boundary competition changes; no validation/test reference is read or scored.'})
    print(f'Exported {len(frame)} training/context seed records without modifying the database')


if __name__=='__main__':main()
