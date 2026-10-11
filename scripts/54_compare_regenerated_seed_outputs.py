#!/usr/bin/env python3
"""Compare completed regenerated seeds/crowns with frozen scalar records.

Comparison only; never generates seeds or modifies either database. Local IDs
are paired by coordinates, not relabelled as original metropolitan IDs.
"""
import argparse,csv,importlib.util,json,sqlite3
from pathlib import Path
import geopandas as gpd
from benchmark.common import ROOT,sha256,site_definitions,write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reconstruction',type=Path,required=True)
    ap.add_argument('--database',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():ap.error('fresh output required')
    original=json.loads((a.reconstruction/'summary.json').read_text())
    if original['frozen_inferred_coordinates_used_for_generation'] is not False:raise ValueError('fresh detection receipt required')
    frozen=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);frozen.row_factory=sqlite3.Row
    frozen.execute('PRAGMA query_only=ON')
    derived=sqlite3.connect((a.reconstruction/'reconstructed_inventory.sqlite').resolve().as_uri()+'?mode=ro&immutable=1',uri=True);derived.row_factory=sqlite3.Row
    spec=importlib.util.spec_from_file_location('fresh_detection',ROOT/'scripts/52_reconstruct_detection_and_crowns.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    frame=gpd.read_file(a.reconstruction/'reconstructed_crowns.gpkg',layer='crowns').set_index('tree_id')
    rows=[];by_site={};rejected=[];extra=[];fields=['crown_area_m2','crown_diameter_m','crown_mean_chm_m','crown_max_chm_m']
    for sid in original['training_site_ids']:
        site=site_definitions()[sid]
        if site['split']!='train':raise ValueError('training only')
        bounds=site['bbox_epsg2193']
        f=[dict(r) for r in frozen.execute('''SELECT l.*,c.crown_area_m2,c.crown_diameter_m,c.crown_mean_chm_m,c.crown_max_chm_m
            FROM tree_lidar_pilot l JOIN tree_crown_pilot c ON l.tree_id=c.tree_id
            WHERE l.x_2193>=? AND l.y_2193>=? AND l.x_2193<? AND l.y_2193<?''',bounds)]
        d=[dict(r) for r in derived.execute('''SELECT l.*,t.source_primary FROM tree_lidar_pilot l JOIN trees t ON l.tree_id=t.tree_id
            WHERE l.x_2193>=? AND l.y_2193>=? AND l.x_2193<? AND l.y_2193<?''',bounds)]
        pairs=m.coordinate_pairs([[r['x_2193'],r['y_2193']] for r in d],[[r['x_2193'],r['y_2193']] for r in f],1e-6)
        mapped={j:(i,distance) for i,j,distance in pairs};matched_d={i for i,j,distance in pairs}
        for j,fr in enumerate(f):
            row={'site_id':sid,'frozen_id':fr['tree_id'],'generated_id':None,'seed_coordinate_match':j in mapped,'crown_recovered':False,'all_four_match':False}
            for field in fields:row['frozen_'+field]=fr[field];row['generated_'+field]=None;row['delta_'+field]=None
            if j in mapped:
                i,distance=mapped[j];dr=d[i];row['generated_id']=dr['tree_id'];row['seed_distance_m']=distance
                row['fresh_local_max_chm_m']=dr['chm_local_max_2m_m'];row['frozen_local_max_chm_m']=fr['chm_local_max_2m_m']
                if dr['tree_id'] in frame.index:
                    cr=frame.loc[dr['tree_id']];row['crown_recovered']=True
                    for field in fields:row['generated_'+field]=float(cr[field]);row['delta_'+field]=float(cr[field])-fr[field]
                    row['all_four_match']=all(abs(row['delta_'+field])<=1e-6 for field in fields)
            rows.append(row)
        for i,dr in enumerate(d):
            if dr['source_primary']=='lidar_inferred_canopy' and dr['tree_id'] not in frame.index:
                rejected.append({'site_id':sid,'generated_id':dr['tree_id'],'x':dr['x_2193'],'y':dr['y_2193'],'reason':'no accepted crown intersecting this site; not retained as a published detection'})
            if i not in matched_d and dr['tree_id'] in frame.index:
                extra.append({'site_id':sid,'generated_id':dr['tree_id']})
        own=[r for r in rows if r['site_id']==sid]
        by_site[sid]={'frozen_crowns':len(f),'regenerated_seed_coordinates_match':sum(r['seed_coordinate_match'] for r in own),
                     'crowns_recovered':sum(r['crown_recovered'] for r in own),'four_attribute_matches':sum(r['all_four_match'] for r in own)}
    frozen.close();derived.close();a.output.mkdir(parents=True)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (a.output/'comparison_all126.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=columns);w.writeheader();w.writerows(rows)
    report={'status':'TRAINING-ONLY FRESH DETECTION AND CROWN COMPARISON',
        'frozen_seed_coordinates_used_for_generation':False,'original_ids_reassigned_to_generated_objects':False,
        'frozen_crowns':len(rows),'seed_coordinate_matches':sum(r['seed_coordinate_match'] for r in rows),
        'crown_matches':sum(r['crown_recovered'] for r in rows),'four_attribute_matches':sum(r['all_four_match'] for r in rows),
        'extra_accepted_seed_inside_crowns':extra,'rejected_inferred_peaks_inside_sites':rejected,'by_site':by_site,
        'raw_reconstruction_summary_sha256':sha256(a.reconstruction/'summary.json'),
        'crown_input_sha256':sha256(a.reconstruction/'reconstructed_crowns.gpkg'),
        'database_sha256_from_verified_reconstruction':original['source_database_sha256'],
        'reporting_correction':{'field':'crown_parameters.greenness_enabled','raw_summary_value':False,'actual_execution_value':True,
            'evidence':'executed_source.py passes gli_chunk from the fingerprinted greenness VRT; old no-GLI plan metadata was inherited incorrectly; numerical outputs unchanged'},
        'historical_reproduction_verified':False,'cleanup_scope':'only a post-crown retained detection view within training sites; no global cleanup/database deletion',
        'limitations':original['limitations']}
    write_json(a.output/'summary.json',report)
    write_json(a.output/'SHA256_manifest.json',{str(p.relative_to(a.output)):sha256(p) for p in a.output.iterdir() if p.is_file()})
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
