#!/usr/bin/env python3
"""Replay supplied CHM detection, fresh height sampling and crown generation.

Training sites only. Frozen inferred coordinates are read only AFTER detection,
for comparison, never as generation seeds. Non-inferred inventory is retained
from the delivered snapshot and explicitly is not a recovered raw inventory.
All writes go to a new compact derived database/output directory.
"""
import argparse
import csv
import importlib.util
import json
import math
import sqlite3
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import box, Point
from shapely.ops import unary_union
from benchmark.common import ROOT, sha256, write_json


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def coordinate_pairs(generated, frozen, tolerance):
    """One-to-one maximum-cardinality distance match, independent of numeric IDs."""
    from scipy.optimize import linear_sum_assignment
    if not len(generated) or not len(frozen):
        return []
    distance = np.linalg.norm(np.asarray(generated)[:, None, :] - np.asarray(frozen)[None, :, :], axis=2)
    n, m = distance.shape
    # A lost match costs more than every possible admissible distance sum.
    penalty = (min(n, m) + 1) * (tolerance + 1)
    cost = np.full((n, m + n), penalty, dtype=float)
    cost[:, :m] = np.where(distance <= tolerance, distance, 3 * penalty)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(distance[i, j])) for i, j in zip(rows, cols)
            if j < m and distance[i, j] <= tolerance]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'chm', 'database', 'intake', 'recovery-root', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--detector-gli-threshold', type=float)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('fresh output directory required')
    adapter = load_file('historical_adapter', ROOT/'scripts/51_attempt_historical_crowns.py')
    crown, provenance, source_hashes = adapter.source_module(args.source)
    # Source detector imports this module by its original name.
    sys.modules['build_tree_crown_pilot'] = crown
    detector = load_file('historical_detection_source', args.source/'scripts/detect_inferred_trees.py')
    lidar = load_file('historical_lidar_source', args.source/'scripts/build_lidar_pilot.py')
    supplied_threshold = detector.GLI_THRESHOLD
    if args.detector_gli_threshold is not None:
        if not math.isfinite(args.detector_gli_threshold) or not -1 <= args.detector_gli_threshold <= 1:
            ap.error('finite detector GLI threshold in [-1,1] required')
        detector.GLI_THRESHOLD = args.detector_gli_threshold
    plan_path = args.recovery_root/'training_crowns_historical_order/plan.json'
    plan = json.loads(plan_path.read_text())
    sites = adapter.selected_training_sites(plan['training_site_ids'])
    if set(sites) != set(plan['training_site_ids']) or len(sites) != 8:
        raise ValueError('all eight training sites required; held-out sites forbidden')
    database = args.database.resolve()
    wal = Path(str(database) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('frozen database requires absent/empty WAL')
    before = sha256(database)
    if before != json.loads(args.intake.read_text())['database_sha256_observed']:
        raise ValueError('frozen database differs from intake')
    chm_manifest = adapter.chm_fingerprints(args.chm)
    gli_path = args.recovery_root/'current_greenness_complete/greenness.vrt'
    gli_manifest = adapter.greenness_fingerprints(gli_path)
    buildings, nonveg, masks = adapter.load_masks(crown, plan['osm_query_cells'], args.recovery_root/'current_osm_cache')
    exclusions = buildings + nonveg
    grid = plan['parameters']['original_global_bbox']
    chunks = plan['original_chunks']
    mask_index = crown.index_buildings_by_chunk(exclusions, detector.CHUNK_SIZE_M, grid)
    union_sites = unary_union([box(*s['bbox_epsg2193']) for s in sites.values()])
    relevant = unary_union([box(*c).buffer(crown.CROWN_HALO_M) for c in chunks])
    conn = sqlite3.connect(database.as_uri()+'?mode=ro&immutable=1', uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA query_only=ON')
    inventory_fields = ['tree_id','source_primary','source_tree_id','species_common','species_latin',
                        'owner_class','is_protected_notable','lon','lat']
    noninfer = [dict(row) for row in conn.execute(
        'SELECT '+','.join(inventory_fields)+' FROM trees WHERE source_primary != ?',
        (detector.INFERRED_SOURCE_TAG,))]
    to_xy = Transformer.from_crs(4326, 2193, always_xy=True)
    canonical_xy = np.array([to_xy.transform(r['lon'], r['lat']) for r in noninfer])
    kd = cKDTree(canonical_xy) if len(canonical_xy) else None
    args.output.mkdir(parents=True)
    (args.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output/'input_manifest.json', {'source_provenance':provenance,
        'source_file_hashes':source_hashes,'database_sha256':before,
        'chm':chm_manifest,'greenness':gli_manifest,'mask_files':masks,
        'site_definitions_sha256':sha256(ROOT/'data/sites/reference_sites.json'),
        'training_plan_sha256':sha256(plan_path)})
    generated, receipts = [], []
    with rasterio.open(args.chm) as src, rasterio.open(gli_path) as gli:
        for index, cell in enumerate(chunks):
            image, transform = crown.read_chm_window(src, cell)
            local_masks = crown.buildings_for_chunk(cell, mask_index, detector.CHUNK_SIZE_M, grid)
            records = detector.detect_chunk(image, transform, cell, local_masks, kd, gli_src=gli)
            generated.extend(records)
            receipts.append({'bbox':cell,'generated_inferred':len(records),
                'finite_chm_fraction':float(np.isfinite(image).mean())})
            print(f'Detection {index+1}/{len(chunks)}: {len(records)} new peaks', flush=True)
    # Local IDs avoid pretending that metropolitan global enumeration was replayed.
    to_ll = Transformer.from_crs(2193, 4326, always_xy=True)
    derived_rows = []
    for index, row in enumerate(generated):
        lon, lat = to_ll.transform(row['x'], row['y'])
        row['tree_id'] = f'reconstructed_lid_{index+1:07d}'
        derived_rows.append({'tree_id':row['tree_id'],'source_primary':detector.INFERRED_SOURCE_TAG,
            'source_tree_id':row['tree_id'],'species_common':'Unknown','species_latin':None,
            'owner_class':'LiDAR-Inferred Canopy','is_protected_notable':0,'lon':lon,'lat':lat})
    selected_canonical = [r for r, xy in zip(noninfer, canonical_xy) if relevant.covers(Point(*xy))]
    derived_rows = selected_canonical + derived_rows
    derived_db = args.output/'reconstructed_inventory.sqlite'
    writable = sqlite3.connect(derived_db)
    column_sql = ','.join(f'{name} '+('INTEGER' if name=='is_protected_notable' else 'REAL' if name in ('lon','lat') else 'TEXT') for name in inventory_fields)
    writable.execute('CREATE TABLE trees ('+column_sql+')')
    writable.executemany('INSERT INTO trees VALUES ('+','.join('?' for _ in inventory_fields)+')',
                        [[r[k] for k in inventory_fields] for r in derived_rows])
    writable.commit(); writable.close()
    sampled = lidar.sample_tree_chm(args.chm, tuple(grid), derived_db)
    print('Fresh source height sampling:', json.dumps(sampled), flush=True)
    derived = sqlite3.connect(derived_db); derived.row_factory = sqlite3.Row
    crown_rows, crown_receipts = [], []
    with rasterio.open(args.chm) as src, rasterio.open(gli_path) as gli:
        for index, cell in enumerate(chunks):
            seeds = adapter.historical_chunk_seeds(derived, cell, crown, grid)
            image, transform = crown.read_chm_window(src, cell)
            local_masks = [g for g in exclusions if g.intersects(box(*cell).buffer(crown.CROWN_HALO_M))]
            gli_image = crown.read_gli_for_chunk(gli, cell, image.shape, transform)
            result = crown.segment_chunk(image, transform, local_masks, seeds, cell, gli_chunk=gli_image)
            crown_rows.extend(r for r in result if r['geometry_2193'].intersects(union_sites))
            crown_receipts.append({'bbox':cell,'seeds':len(seeds),'crowns':len(result)})
            print(f'Crown generation {index+1}/{len(chunks)}: {len(result)} crowns', flush=True)
    derived.close()
    columns = ['tree_id','source_primary','crown_area_m2','crown_diameter_m','crown_mean_chm_m','crown_max_chm_m']
    records = [{**{k:r[k] for k in columns[2:]},'tree_id':r['tree']['tree_id'],
                'source_primary':r['tree']['source_primary']} for r in crown_rows]
    frame = gpd.GeoDataFrame(records, columns=columns, geometry=[r['geometry_2193'] for r in crown_rows], crs=2193)
    frame.to_file(args.output/'reconstructed_crowns.gpkg',layer='crowns',driver='GPKG')
    point_frame = gpd.GeoDataFrame([{k:r[k] for k in ('tree_id','height_m')} for r in generated],
                geometry=[Point(r['x'],r['y']) for r in generated], crs=2193)
    point_frame.to_file(args.output/'reconstructed_peaks.gpkg',layer='peaks',driver='GPKG')
    # Comparison starts only after generated peak positions and crowns are fixed.
    by_site, comparison_rows = {}, []
    for sid, site in sites.items():
        a,b,c,d = site['bbox_epsg2193']
        gen = [r for r in generated if a<=r['x']<c and b<=r['y']<d]
        frozen = [dict(r) for r in conn.execute('''SELECT t.tree_id,l.* FROM tree_lidar_pilot l
            JOIN trees t ON t.tree_id=l.tree_id WHERE t.source_primary=?
            AND l.x_2193>=? AND l.x_2193<? AND l.y_2193>=? AND l.y_2193<?''',
            (detector.INFERRED_SOURCE_TAG,a,c,b,d))]
        gxy = [[r['x'],r['y']] for r in gen]; fxy = [[r['x_2193'],r['y_2193']] for r in frozen]
        exact = coordinate_pairs(gxy,fxy,1e-6); near = coordinate_pairs(gxy,fxy,2.)
        by_site[sid]={'generated_inferred':len(gen),'frozen_retained_inferred':len(frozen),
                     'exact_coordinate_matches_1e_6m':len(exact),'matches_within_2m':len(near)}
        for i,j,distance in near:
            comparison_rows.append({'site_id':sid,'generated_id':gen[i]['tree_id'],
                'frozen_id':frozen[j]['tree_id'],'distance_m':distance})
    conn.close()
    after = sha256(database)
    if before != after: raise ValueError('source frozen database changed')
    with (args.output/'coordinate_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['site_id','generated_id','frozen_id','distance_m'])
        writer.writeheader();writer.writerows(comparison_rows)
    write_json(args.output/'chunk_receipts.json',{'detection':receipts,'crowns':crown_receipts})
    report={'status':'PROVISIONAL DETECTION-TO-CROWN RECONSTRUCTION; TRAINING ONLY',
        'training_site_ids':list(sites),'frozen_inferred_coordinates_used_for_generation':False,
        'reference_labels_read':False,'source_database_unchanged':before==after,
        'canonical_inventory_total':len(noninfer),'canonical_inventory_in_processing_region':len(selected_canonical),
        'new_inferred_peaks_in_chunks':len(generated),'site_intersecting_crowns':len(frame),
        'by_site':by_site,'sampled_height_summary':sampled,
        'detector_parameters':{k:getattr(detector,k) for k in ['CHUNK_SIZE_M','DETECTION_MIN_HEIGHT_M','SMOOTH_SIGMA_PX','DEDUP_DISTANCE_M','GLI_THRESHOLD','GLI_WINDOW_PX','METHOD_ID']},
        'supplied_default_detector_GLI_THRESHOLD':supplied_threshold,
        'crown_parameters':{**plan['parameters'],'greenness_enabled':True},'source_database_sha256':before,
        'historical_reproduction_verified':False,'reference_frozen':False,
        'limitations':['retained non-inferred inventory is used; raw pre-canonical inventory and deleted entries are unavailable',
        'current masks and greenness are replacement inputs',
        'CHM is the supplied bounded VRT, not a replay of all metropolitan DSM/DEM acquisition',
        'local peak IDs do not recreate the original metropolitan enumeration',
        'source detector GLI reader uses core bounds with halo-array out_shape; retained unchanged for this replay',
        'original frozen crown geometry unavailable; cleanup recipe unconfirmed; this output is before unverified cleanup']}
    write_json(args.output/'summary.json',report)
    write_json(args.output/'SHA256_manifest.json',{str(p.relative_to(args.output)):sha256(p) for p in sorted(args.output.rglob('*')) if p.is_file()})
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
