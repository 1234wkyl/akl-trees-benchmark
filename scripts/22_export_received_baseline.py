#!/usr/bin/env python
"""Export snapshot scalar detections and audit existing caches; no scoring.

Requires the completed read-only intake receipt for this actual database.
Never fabricates crown geometry from diameter/area. Does not run historic code.
"""
import argparse
import json
import math
import sqlite3
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
from benchmark.common import sha256, site_definitions, write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--database',required=True,type=Path)
    ap.add_argument('--intake',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    db=args.database.resolve();wal=Path(str(db)+'-wal')
    if args.output.exists():ap.error('Use a new output directory')
    if wal.exists() and wal.stat().st_size:ap.error('Nonempty WAL: obtain checkpointed snapshot')
    intake=json.loads(args.intake.read_text())
    observed=sha256(db)
    if observed!=intake['database_sha256_observed']:ap.error('Database changed since intake')
    sites=site_definitions();root=Path(__file__).resolve().parents[1]
    c=sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
    c.execute('PRAGMA query_only=ON')
    records=[];audits=[]
    try:
        schemas={r['name']:r['sql'] for r in c.execute("SELECT name,sql FROM sqlite_master WHERE type='table'")}
        metadata=[dict(r) for r in c.execute('SELECT * FROM run_metadata')]
        methods={}
        for name in ['tree_lidar_pilot','tree_crown_pilot']:
            methods[name]=[dict(r) for r in c.execute(f'SELECT method_id, created_at_utc, COUNT(*) AS rows FROM {name} GROUP BY method_id,created_at_utc')]
        for sid,s in sites.items():
            x0,y0,x1,y1=s['bbox_epsg2193']
            query='''SELECT t.tree_id,t.source_primary,l.x_2193 AS x,l.y_2193 AS y,
                COALESCE(p.canopy_top_m,l.chm_at_point_m) AS height_m,
                l.chm_local_max_2m_m AS seed_height_m,c.crown_area_m2,
                a.growth_form,a.growth_form_confidence,l.method_id AS lidar_method_id,
                c.method_id AS crown_method_id
                FROM trees t JOIN tree_lidar_pilot l ON t.tree_id=l.tree_id
                LEFT JOIN tree_crown_pilot c ON t.tree_id=c.tree_id
                LEFT JOIN tree_pointcloud_pilot p ON t.tree_id=p.tree_id
                LEFT JOIN tree_species_attributes a ON t.tree_id=a.tree_id
                WHERE l.x_2193>=? AND l.x_2193<? AND l.y_2193>=? AND l.y_2193<?
                ORDER BY t.tree_id'''
            rows=[dict(r) for r in c.execute(query,(x0,x1,y0,y1))]
            cached=json.loads((root/f'data/sites/chips/{sid}/existing_detections.json').read_text())
            cache={r['tree_id']:r for r in cached};actual={r['tree_id']:r for r in rows}
            diffs=[]
            for tid in sorted(cache.keys() & actual.keys()):
                for k in ['x','y','height_m','crown_area_m2','source_primary','growth_form','growth_form_confidence']:
                    a,b=cache[tid].get(k),actual[tid].get(k)
                    same=math.isclose(a,b,rel_tol=0,abs_tol=1e-8) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else a==b
                    if not same:diffs.append({'tree_id':tid,'field':k,'cache':a,'snapshot':b})
            audits.append({'site_id':sid,'split':s['split'],'stratum':s['stratum'],
                'snapshot_count':len(rows),'cache_count':len(cached),'cache_only_ids':sorted(cache.keys()-actual.keys()),
                'snapshot_only_ids':sorted(actual.keys()-cache.keys()),'attribute_differences':diffs,
                'cache_matches_snapshot':not diffs and cache.keys()==actual.keys()})
            records.extend(dict(site_id=sid,split=s['split'],stratum=s['stratum'],**r) for r in rows)
        geometry_fields={name:[r['name'] for r in c.execute('PRAGMA table_info("'+name.replace('"','""')+'")')
            if any(w in r['name'].lower() for w in ['geometry','geom','geojson','wkt','wkb'])] for name in schemas}
    finally:c.close()
    args.output.mkdir(parents=True)
    frame=gpd.GeoDataFrame(records,geometry=[Point(r['x'],r['y']) for r in records],crs=2193)
    frame.to_file(args.output/'snapshot_site_detections.gpkg',layer='snapshot_site_detections',driver='GPKG')
    frame.drop(columns='geometry').to_csv(args.output/'snapshot_site_detections.csv',index=False)
    write_json(args.output/'cache_comparison.json',audits)
    write_json(args.output/'database_metadata.json',{'schemas':schemas,'run_metadata':metadata,
        'detection_and_crown_method_dates':methods,'possible_geometry_columns':geometry_fields})
    write_json(args.output/'summary.json',{'database_sha256':observed,'intake_sha256':sha256(args.intake),
        'snapshot_received_from_supervisor':True,'sites':len(sites),'detection_records':len(records),
        'caches_matching_snapshot':sum(r['cache_matches_snapshot'] for r in audits),
        'crown_polygon_geometry_available':any(geometry_fields.values()),'model_scores_computed':False,
        'read_only_mode':'mode=ro&immutable=1; query_only=ON; absent/empty WAL',
        'code_and_parameters':'awaiting Tim confirmation; do not reproduce before reply',
        'geometry_note':'These points are frozen inventory/detection locations, not independent stems and not crown polygons. Do not create fake circular crowns from scalar area or diameter.'})
    print(f'Exported {len(records)} snapshot records; {sum(r["cache_matches_snapshot"] for r in audits)}/{len(sites)} caches match')


if __name__=='__main__':main()
