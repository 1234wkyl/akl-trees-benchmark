#!/usr/bin/env python3
"""Fresh detector/CHM/crown prediction on held-out sites; no reference scoring."""
import argparse,importlib.util,json,sqlite3,sys
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import box,Point
from shapely.ops import unary_union
from benchmark.common import ROOT,sha256,site_definitions,write_json
from benchmark.inference_scenario import validate_inference_scenario

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','chm','database','intake','inputs','classical-selection','output'):
        p.add_argument('--'+n,type=Path,required=True)
    scope=p.add_mutually_exclusive_group(required=True)
    scope.add_argument('--inference-scenario',type=Path)
    scope.add_argument('--author-model-lock',type=Path,
                       help='separate exposed-results classical continuation, bound to author-confirmed train selection')
    a=p.parse_args()
    if a.inference_scenario:validate_inference_scenario(a.inference_scenario)
    else:
        author_lock=json.loads(a.author_model_lock.read_text())
        if (author_lock.get('status')!='AUTHOR-CONFIRMED TRAINING-ONLY CLASSICAL CONFIGURATION'
            or author_lock.get('heldout_used_for_selection') is not False
            or author_lock.get('prior_test_exposure_disclosed') is not True
            or author_lock.get('selection_sha256')!=sha256(a.classical_selection)
            or set(author_lock.get('training_site_ids',[]))!={s for s,v in site_definitions().items() if v['split']=='train'}):
            raise ValueError('explicit author-reference train-only selection lock required')
    if a.output.exists():p.error('fresh fixed prediction directory required')
    plan=json.loads((a.inputs/'plan.json').read_text());completed=json.loads((a.inputs/'input_receipt.json').read_text())
    scope_matches=(plan['inference_scenario_sha256']==sha256(a.inference_scenario)) if a.inference_scenario else (
        plan.get('author_model_lock_sha256')==sha256(a.author_model_lock) and plan.get('inference_scenario_sha256') is None)
    if completed['plan_sha256']!=sha256(a.inputs/'plan.json') or plan['classical_selection_sha256']!=sha256(a.classical_selection) or not scope_matches:
        raise ValueError('prediction inputs/configuration differ from fixed preparation')
    for name,h in completed['source_files'].items():
        if sha256(a.inputs/name)!=h:raise ValueError('prepared input changed: '+name)
    sites={sid:s for sid,s in site_definitions().items() if s['split']!='train'}
    if set(sites)!=set(plan['selected_site_ids']):raise ValueError('all and only 16 held-out sites required')
    m=load('historical_adapter',ROOT/'scripts/51_attempt_historical_crowns.py');crown,_,hashes=m.source_module(a.source)
    if hashes!=plan['source_file_hashes']:raise ValueError('source changed after configuration lock')
    sys.modules['build_tree_crown_pilot']=crown
    detector=load('historical_detector',a.source/'scripts/detect_inferred_trees.py');lidar=load('historical_lidar',a.source/'scripts/build_lidar_pilot.py')
    choice=json.loads(a.classical_selection.read_text())['selected']['parameters']
    if choice!=plan['parameters'] or detector.DETECTION_MIN_HEIGHT_M!=choice['detector_minimum_height_m'] or detector.GLI_THRESHOLD!=choice['detector_gli_threshold']:
        raise ValueError('fixed detector parameters differ')
    crown.CANOPY_MIN_HEIGHT_M=choice['crown_minimum_height_m'];crown.CROWN_GLI_THRESHOLD=choice['crown_gli_threshold']
    original=a.database.resolve();wal=Path(str(original)+'-wal')
    if wal.exists() and wal.stat().st_size:raise ValueError('frozen database WAL must be absent/empty')
    before=sha256(original)
    if before!=json.loads(a.intake.read_text())['database_sha256_observed']:raise ValueError('frozen database differs from intake')
    chm=m.chm_fingerprints(a.chm);green=a.inputs/'current_greenness_complete/greenness.vrt';green_hashes=m.greenness_fingerprints(green)
    buildings,nonveg,mask_hashes=m.load_masks(crown,plan['osm_query_cells'],a.inputs/'current_osm_cache');exclusions=buildings+nonveg
    grid=plan['original_global_bbox'];chunks=plan['original_chunks'];index=crown.index_buildings_by_chunk(exclusions,detector.CHUNK_SIZE_M,grid)
    fields=['tree_id','source_primary','source_tree_id','species_common','species_latin','owner_class','is_protected_notable','lon','lat']
    original_conn=sqlite3.connect(original.as_uri()+'?mode=ro&immutable=1',uri=True);original_conn.row_factory=sqlite3.Row
    canonical=[dict(x) for x in original_conn.execute('SELECT '+','.join(fields)+' FROM trees WHERE source_primary != ?',(detector.INFERRED_SOURCE_TAG,))];original_conn.close()
    to_xy=Transformer.from_crs(4326,2193,always_xy=True);xy=np.asarray([to_xy.transform(r['lon'],r['lat']) for r in canonical]);kd=cKDTree(xy)
    peaks=[];a.output.mkdir(parents=True);(a.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    with rasterio.open(a.chm) as src,rasterio.open(green) as gli:
        for i,cell in enumerate(chunks):
            image,tr=crown.read_chm_window(src,cell);local=crown.buildings_for_chunk(cell,index,detector.CHUNK_SIZE_M,grid)
            peaks.extend(detector.detect_chunk(image,tr,cell,local,kd,gli_src=gli));print('Fixed held-out detection chunk',i+1,'peaks',len(peaks),flush=True)
    region=unary_union([box(*c).buffer(crown.CROWN_HALO_M) for c in chunks]);records=[r for r,pos in zip(canonical,xy) if region.covers(Point(*pos))]
    to_ll=Transformer.from_crs(2193,4326,always_xy=True)
    for i,peak in enumerate(peaks):
        lon,lat=to_ll.transform(peak['x'],peak['y']);tid=f'heldout_reconstructed_lid_{i+1:07d}'
        records.append(dict(tree_id=tid,source_primary=detector.INFERRED_SOURCE_TAG,source_tree_id=tid,species_common='Unknown',species_latin=None,owner_class='LiDAR-Inferred Canopy',is_protected_notable=0,lon=lon,lat=lat))
    derived=a.output/'derived_prediction_inventory.sqlite';conn=sqlite3.connect(derived)
    conn.execute('CREATE TABLE trees ('+','.join(n+' '+('REAL' if n in ('lon','lat') else 'INTEGER' if n=='is_protected_notable' else 'TEXT') for n in fields)+')')
    conn.executemany('INSERT INTO trees VALUES ('+','.join('?' for _ in fields)+')',[[r[k] for k in fields] for r in records]);conn.commit();conn.close()
    sampled=lidar.sample_tree_chm(a.chm,tuple(grid),derived)
    conn=sqlite3.connect(derived);conn.row_factory=sqlite3.Row;union=unary_union([box(*s['bbox_epsg2193']) for s in sites.values()]);crowns={}
    with rasterio.open(a.chm) as src,rasterio.open(green) as gli:
        for i,cell in enumerate(chunks):
            image,tr=crown.read_chm_window(src,cell);greenness=crown.read_gli_for_chunk(gli,cell,image.shape,tr)
            seeds=m.historical_chunk_seeds(conn,cell,crown,grid);local=[g for g in exclusions if g.intersects(box(*cell).buffer(crown.CROWN_HALO_M))]
            for site in sites.values():
                emission=box(*site['bbox_epsg2193']).buffer(16).envelope.intersection(box(*cell))
                if emission.is_empty or emission.area<=0:continue
                for r in crown.segment_chunk(image,tr,local,seeds,emission.bounds,gli_chunk=greenness):
                    if r['geometry_2193'].intersects(union):crowns[r['tree']['tree_id']]=r
            print('Fixed held-out crowns chunk',i+1,'accepted',len(crowns),flush=True)
    conn.close();values=list(crowns.values());rows=[];geoms=[]
    for r in values:
        rows.append({'tree_id':r['tree']['tree_id'],**{k:r[k] for k in ['crown_area_m2','crown_diameter_m','crown_mean_chm_m','crown_max_chm_m']}});geoms.append(r['geometry_2193'])
    f=gpd.GeoDataFrame(rows,geometry=geoms,crs=2193)
    if f.empty or not f.geometry.is_valid.all():raise ValueError('nonempty valid crown output required')
    f.to_file(a.output/'fixed_classical_crowns.gpkg',layer='crowns',driver='GPKG')
    if sha256(original)!=before:raise ValueError('immutable baseline changed')
    write_json(a.output/'prediction_receipt.json',{'status':'FIXED PROVISIONAL CLASSICAL HELD-OUT PREDICTIONS; NO SCORING',
        'site_ids':list(sites),'parameters':choice,'classical_selection_sha256':sha256(a.classical_selection),
        'scenario_sha256':sha256(a.inference_scenario) if a.inference_scenario else None,
        'author_model_lock_sha256':sha256(a.author_model_lock) if a.author_model_lock else None,
        'prior_test_exposure_disclosed':bool(a.author_model_lock),
        'input_receipt_sha256':sha256(a.inputs/'input_receipt.json'),
        'source_file_hashes':hashes,'database_sha256':before,'source_database_unchanged':True,
        'chm':chm,'greenness':green_hashes,'masks':mask_hashes,'sampled_heights':sampled,
        'generated_peaks':len(peaks),'site_intersecting_crowns':len(f),'crowns_sha256':sha256(a.output/'fixed_classical_crowns.gpkg'),
        'historical_identity_verified':False,'evaluation_permitted':False,'reference_frozen':False,
        'reference_labels_used_for_prediction':False,'frozen_inferred_coordinates_used_for_generation':False,
        'limits':['retained canonical inventory only','replacement imagery/masks','bounded supplied CHM','unverified historical cleanup','no frozen polygon geometry comparison']})
    print('Fixed provisional classical crowns generated; no held-out reference scores.')
if __name__=='__main__':main()
