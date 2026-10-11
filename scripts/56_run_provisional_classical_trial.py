#!/usr/bin/env python3
"""Training-only crown trial using freshly regenerated detector seeds.

The historical kernel and original CHM grid are retained. Emission is bounded
to a 16 m site margin intersected with the original owner chunk, while the
entire original raster/masks/competitor seed context is still supplied.
"""
import argparse,importlib.util,json,sqlite3
from pathlib import Path
import geopandas as gpd
import rasterio
from shapely.geometry import box
from shapely.ops import unary_union
from benchmark.common import ROOT,sha256,write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source','chm','recovery-root','reconstruction','output'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--minimum-height',type=float,required=True)
    a=ap.parse_args()
    if a.output.exists():ap.error('fresh output required')
    if not 0<a.minimum_height<=20:ap.error('finite positive minimum height <=20m')
    spec=importlib.util.spec_from_file_location('adapter',ROOT/'scripts/51_attempt_historical_crowns.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source,provenance,hashes=m.source_module(a.source);source.CANOPY_MIN_HEIGHT_M=a.minimum_height
    receipt=json.loads((a.reconstruction/'summary.json').read_text())
    if receipt['frozen_inferred_coordinates_used_for_generation'] is not False:raise ValueError('fresh detection required')
    sites=m.selected_training_sites(receipt['training_site_ids'])
    if len(sites)!=8:raise ValueError('all training sites required')
    plan=json.loads((a.recovery_root/'training_crowns_historical_order/plan.json').read_text())
    buildings,nonveg,masks=m.load_masks(source,plan['osm_query_cells'],a.recovery_root/'current_osm_cache')
    exclusions=buildings+nonveg
    db=a.reconstruction/'reconstructed_inventory.sqlite'
    c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
    union=unary_union([box(*s['bbox_epsg2193']) for s in sites.values()]);results={};chunks=[]
    a.output.mkdir(parents=True);(a.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    with rasterio.open(a.chm) as raster,rasterio.open(a.recovery_root/'current_greenness_complete/greenness.vrt') as green:
        for index,cell in enumerate(plan['original_chunks']):
            image,tr=source.read_chm_window(raster,cell);gli=source.read_gli_for_chunk(green,cell,image.shape,tr)
            local=[g for g in exclusions if g.intersects(box(*cell).buffer(source.CROWN_HALO_M))]
            seeds=m.historical_chunk_seeds(c,cell,source,plan['parameters']['original_global_bbox'])
            emit=[]
            for sid,s in sites.items():
                area=box(*s['bbox_epsg2193']).buffer(16).envelope.intersection(box(*cell))
                if area.is_empty or area.area<=0:continue
                emit.append(list(area.bounds))
                for result in source.segment_chunk(image,tr,local,seeds,area.bounds,gli_chunk=gli):
                    if result['geometry_2193'].intersects(union):results[result['tree']['tree_id']]=result
            chunks.append({'owner_chunk':cell,'emission_bounds':emit,'seeds':len(seeds)})
            print('Training crown trial',a.minimum_height,'chunk',index+1,'retained',len(results),flush=True)
    c.close()
    values=list(results.values());props=[{**{k:r[k] for k in ['crown_area_m2','crown_diameter_m','crown_mean_chm_m','crown_max_chm_m']},
        'tree_id':r['tree']['tree_id']} for r in values]
    f=gpd.GeoDataFrame(props,geometry=[r['geometry_2193'] for r in values],crs=2193)
    if f.empty or not f.geometry.is_valid.all():raise ValueError('valid nonempty crown output required')
    f.to_file(a.output/'crowns.gpkg',layer='crowns',driver='GPKG')
    equivalence=None
    if a.minimum_height==3.:
        old=gpd.read_file(a.reconstruction/'reconstructed_crowns.gpkg',layer='crowns').set_index('tree_id');new=f.set_index('tree_id')
        scalars=['crown_area_m2','crown_diameter_m','crown_mean_chm_m','crown_max_chm_m']
        equivalence=set(old.index)==set(new.index) and all(old.loc[k].geometry.equals(new.loc[k].geometry)
            and all(old.loc[k,col]==new.loc[k,col] for col in scalars) for k in old.index)
        if not equivalence:raise ValueError('bounded emission optimization differs from completed full default run')
    write_json(a.output/'trial.json',{'status':'PROVISIONAL TRAINING-ONLY RECONSTRUCTED CLASSICAL TRIAL',
        'completed_site_ids':list(sites),'reference_labels_read_during_generation':False,
        'reconstruction_receipt_sha256':sha256(a.reconstruction/'summary.json'),'regenerated_inventory_sha256':sha256(db),
        'source_file_hashes':hashes,'mask_files':masks,'minimum_height_m':a.minimum_height,
        'crown_gli_threshold':source.CROWN_GLI_THRESHOLD,'detector_gli_threshold':receipt['detector_parameters']['GLI_THRESHOLD'],
        'emission_optimization_default_equivalence':equivalence,'candidate_crowns':len(f),
        'crown_output_sha256':sha256(a.output/'crowns.gpkg'),'chunks':chunks,
        'historical_identity_verified':False,'reference_frozen':False})
    write_json(a.output/'SHA256_manifest.json',{str(p.relative_to(a.output)):sha256(p) for p in a.output.iterdir() if p.is_file()})
if __name__=='__main__':main()
