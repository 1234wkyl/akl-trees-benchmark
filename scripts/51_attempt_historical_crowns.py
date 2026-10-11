#!/usr/bin/env python3
"""Attempt the historical crown stage on training sites with explicit provenance.

Imports the supplied, hash-checked segment_chunk unchanged. Never invokes its
production main(), downloads CHM, modifies a database, or claims frozen identity.
Current OSM acquisition is an explicit phase; crown execution is offline.
"""
import argparse
import csv
import importlib.util
import json
import math
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from benchmark.common import ROOT, sha256, site_definitions, write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_module(source):
    provenance = json.loads((source / 'PROVENANCE.json').read_text())
    hashes = {}
    for record in provenance['files']:
        path = source / record['path']
        observed = sha256(path)
        if observed != record['sha256']:
            raise ValueError(f'supplied source changed: {path}')
        hashes[record['path']] = observed
    sys.path.insert(0, str(source / 'scripts'))
    spec = importlib.util.spec_from_file_location(
        'historical_crown_source', source / 'scripts/build_tree_crown_pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, provenance, hashes


def selected_training_sites(requested=None):
    sites = {sid: s for sid, s in site_definitions().items() if s['split'] == 'train'}
    if requested:
        if len(set(requested)) != len(requested) or not set(requested) <= set(sites):
            raise ValueError('only unique training site IDs are allowed')
        sites = {sid: sites[sid] for sid in requested}
    return sites


def chm_fingerprints(vrt):
    """Bind actual source TIFF bytes, not only the small VRT wrapper."""
    bundle_root = vrt.resolve().parent.parent
    manifest_path = bundle_root / 'manifest.json'
    expected = {r['path']: r for r in json.loads(manifest_path.read_text())['files']}
    fingerprints = {}
    for node in ET.parse(vrt).getroot().iter('SourceFilename'):
        path = (vrt.parent / node.text).resolve() if node.get('relativeToVRT') == '1' else Path(node.text).resolve()
        relative = str(path.relative_to(bundle_root))
        record = expected.get(relative)
        if not record or not path.is_file():
            raise ValueError(f'CHM source is not present in the delivered manifest: {path}')
        digest = sha256(path)
        if digest != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError(f'CHM source changed: {path}')
        fingerprints[relative] = {'bytes': path.stat().st_size, 'sha256': digest}
    return {'benchmark_manifest_sha256': sha256(manifest_path), 'source_files': fingerprints}


def greenness_fingerprints(path):
    """Bind every TIFF behind a GLI VRT, including CRS and grid validation."""
    sources = [path.resolve()]
    if path.suffix.lower() == '.vrt':
        sources = []
        for node in ET.parse(path).getroot().iter('SourceFilename'):
            source = ((path.parent / node.text).resolve()
                      if node.get('relativeToVRT') == '1' else Path(node.text).resolve())
            if source.suffix.lower() == '.vrt':
                raise ValueError('nested greenness VRTs require a flattened, auditable manifest')
            sources.append(source)
        if not sources:
            raise ValueError('greenness VRT has no source rasters')
    fingerprints = {}
    for source in sorted(set(sources)):
        with rasterio.open(source) as raster:
            if raster.crs is None or raster.crs.to_epsg() != 2193 or raster.res != (1., 1.):
                raise ValueError('greenness must use the 1m EPSG:2193 grid')
        fingerprints[str(source)] = {'bytes': source.stat().st_size, 'sha256': sha256(source)}
    return {'wrapper_sha256': sha256(path), 'source_files': fingerprints,
            'historical_input_verified': False}


def select_grid_cells(global_bbox, cell_size, regions):
    """Select cells from the original global lattice, not a site-local lattice."""
    x0, y0, x1, y1 = global_bbox
    output = []
    for iy in range(math.ceil((y1 - y0) / cell_size)):
        for ix in range(math.ceil((x1 - x0) / cell_size)):
            b = (x0 + ix * cell_size, y0 + iy * cell_size,
                 min(x1, x0 + (ix + 1) * cell_size),
                 min(y1, y0 + (iy + 1) * cell_size))
            if any(box(*b).intersects(region) for region in regions):
                output.append(b)
    return output


def mask_names(module, cell):
    suffix = '{:.0f}_{:.0f}_{:.0f}_{:.0f}.json'.format(*cell)
    return ('buildings_tiles/osm_buildings_' + suffix,
            'nonveg_tiles/osm_nonveg_' + module.OSM_NONVEG_CACHE_VERSION + '_' + suffix)


def load_masks(module, cells, cache, allow_missing_nonvegetation=False):
    building_payloads, nonveg_payloads, files = [], [], {}
    for cell in cells:
        bname, nname = mask_names(module, cell)
        for name, payloads in [(bname, building_payloads), (nname, nonveg_payloads)]:
            p = cache / name
            if not p.is_file() or p.stat().st_size == 0:
                if allow_missing_nonvegetation and name == nname:
                    files[name] = {'status':'MISSING EXPLICITLY OMITTED',
                                   'original_query_cell_2193':list(cell)}
                    continue
                raise FileNotFoundError(f'mask cache missing; execute --phase masks first: {p}')
            data = json.loads(p.read_text())
            if 'elements' not in data or data.get('remark'):
                raise ValueError(f'incomplete/error Overpass response: {p}')
            payloads.append(data)
            files[name] = {'sha256': sha256(p), 'osm3s': data.get('osm3s')}
    return (module.osm_payloads_to_polygons(building_payloads),
            module.osm_payloads_to_exclusion_polygons(nonveg_payloads), files)


def subdivide_osm_query(query, subdivisions):
    """Partition the original geographic bbox; retain filters and full geometry."""
    if subdivisions == 1:
        return [query]
    pattern = r'\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)'
    bounds = re.findall(pattern, query)
    if not bounds or len(set(bounds)) != 1:
        raise ValueError('expected one repeated geographic bbox in historical query')
    south, west, north, east = map(float, bounds[0])
    queries = []
    for iy in range(subdivisions):
        for ix in range(subdivisions):
            part = (south + (north-south)*iy/subdivisions,
                    west + (east-west)*ix/subdivisions,
                    south + (north-south)*(iy+1)/subdivisions,
                    west + (east-west)*(ix+1)/subdivisions)
            queries.append(re.sub(pattern, '('+','.join(map(str, part))+')', query))
    return queries


def merge_osm_payloads(payloads):
    elements = {}
    for payload in payloads:
        for element in payload['elements']:
            key = (element['type'], element['id'])
            if key in elements and elements[key] != element:
                raise ValueError('OSM object changed between subrequests; retry from a consistent snapshot')
            elements[key] = element
    timestamps = sorted({p['osm3s']['timestamp_osm_base'] for p in payloads})
    return {'version': payloads[0].get('version', 0.6),
            'generator': 'historical-query-subrequest-merge',
            'osm3s': dict(payloads[0]['osm3s'], timestamp_osm_base=timestamps[0]),
            'elements': list(elements.values()),
            '_benchmark_request_partition': {'subrequests': len(payloads),
                                            'snapshot_timestamps': timestamps,
                                            'deduplication': 'type,id; conflicting objects rejected'}}


def fetch_current_masks(module, cells, cache, output, subdivisions=1, endpoints=None,
                        http_method='GET', query_timeout=None, maxsize_mib=None, server_busy_retries=0):
    if not 0 <= server_busy_retries <= 3:
        raise ValueError('bounded server-busy retries must be between 0 and 3')
    attempts = []

    def bounded_request(query):
        if query_timeout is not None:
            query = re.sub(r'\[timeout:\d+\]', f'[timeout:{query_timeout}]', query)
        if maxsize_mib is not None:
            if maxsize_mib not in {16,32,64,128,256,512}:
                raise ValueError('unsupported explicit request memory limit')
            # An admission hint only: incomplete/out-of-memory results are rejected below.
            limit=f'[maxsize:{maxsize_mib*1024*1024}]'
            query=(re.sub(r'\[maxsize:\d+\]',limit,query) if '[maxsize:' in query
                   else query.replace('[out:json]','[out:json]'+limit,1))
        last = None
        for endpoint in endpoints or module.OVERPASS_ENDPOINTS:
            for attempt in range(server_busy_retries + 1):
                retry = False
                started = time.monotonic()
                record = {'endpoint': endpoint, 'started_utc': utc(), 'query': query,
                          'http_method': http_method, 'busy_retry_number': attempt}
                try:
                    transport = {'params' if http_method == 'GET' else 'data': {'data':query}}
                    response = requests.request(http_method, endpoint, **transport,
                                            headers={'User-Agent': 'akl-trees-benchmark-research/1.0'},
                                            timeout=(15, 150))
                    record['http_status'] = response.status_code
                    if response.status_code != 200:
                        record['error_response_excerpt'] = response.text[:2000]
                    response.raise_for_status()
                    data = response.json()
                    if 'elements' not in data or data.get('remark'):
                        raise ValueError('Overpass response is incomplete or reports an error')
                    record['elements'] = len(data['elements'])
                    record['osm3s'] = data.get('osm3s')
                    return data
                except (requests.RequestException, ValueError) as exc:
                    last = exc
                    record['error'] = str(exc)
                    retry = record.get('http_status') in {429,504} and attempt < server_busy_retries
                    print(f'OSM endpoint failed: {endpoint}: {type(exc).__name__}; '
                          f'HTTP {record.get("http_status", "unavailable")}', flush=True)
                finally:
                    record['wall_seconds'] = time.monotonic() - started
                    attempts.append(record)
                    write_json(output / 'network_attempts.json', attempts)
                if not retry:
                    break
                print('Server busy; waiting 30 seconds before retrying the same subrequest', flush=True)
                time.sleep(30)
        raise RuntimeError(f'all historical Overpass endpoints failed: {last}')

    # Query texts and conversion functions remain those in the supplied source.
    # Only HTTP retry bounds and the cache/output directories are adapted.
    def request_partition(query):
        parts = subdivide_osm_query(query, subdivisions)
        payloads = []
        for i, part in enumerate(parts, 1):
            if len(parts) > 1:
                print(f'OSM subrequest {i}/{len(parts)}', flush=True)
            payloads.append(bounded_request(part))
            if i < len(parts):
                time.sleep(module.OVERPASS_POLITE_SLEEP_S)
        return payloads[0] if len(parts) == 1 else merge_osm_payloads(payloads)

    module._overpass_get = request_partition
    for i, cell in enumerate(cells, 1):
        print(f'Mask cell {i}/{len(cells)}: {cell}', flush=True)
        module.fetch_osm_buildings_tile(cell, cache / 'buildings_tiles')
        module.fetch_osm_non_vegetation_tile(cell, cache / 'nonveg_tiles')
    buildings, nonveg, files = load_masks(module, cells, cache)
    write_json(output / 'mask_cache_manifest.json', {
        'created_utc': utc(), 'source': 'current OpenStreetMap via historical queries',
        'historical_snapshot': False, 'linz_building_source_used': False,
        'uncached_query_subdivisions_per_axis': subdivisions,
        'requested_endpoints': list(endpoints or module.OVERPASS_ENDPOINTS),
        'uncached_query_http_method': http_method,
        'uncached_query_timeout_override_seconds': query_timeout,
        'uncached_query_maxsize_mib': maxsize_mib, 'server_busy_retries': server_busy_retries, 'server_busy_wait_seconds': 30,
        'cells': cells, 'files': files, 'building_polygons': len(buildings),
        'nonvegetation_polygons': len(nonveg),
    })
    print(f'Masks ready: {len(buildings)} buildings, {len(nonveg)} nonvegetation polygons', flush=True)


def chunk_seeds(conn, cell, halo, original_table_order=False):
    x0, y0, x1, y1 = cell
    query = '''SELECT t.tree_id, t.source_primary, t.source_tree_id,
               t.species_common, t.species_latin, t.owner_class,
               t.is_protected_notable, t.lon, t.lat,
               l.x_2193, l.y_2193, l.chm_local_max_2m_m
               FROM trees t JOIN tree_lidar_pilot l ON t.tree_id=l.tree_id
               WHERE l.chm_local_max_2m_m IS NOT NULL
                 AND l.x_2193>=? AND l.x_2193<=?
                 AND l.y_2193>=? AND l.y_2193<=?'''
    if original_table_order:
        # The supplied full-inventory query scans tree_lidar_pilot in rowid
        # order. Preserve it before calling the original chunk-index functions.
        query += ' ORDER BY l.rowid'
    records = [dict(r) for r in conn.execute(query, (x0-halo, x1+halo, y0-halo, y1+halo))]
    for r in records:
        if not all(math.isfinite(r[k]) for k in ['x_2193', 'y_2193', 'chm_local_max_2m_m']):
            raise ValueError('nonfinite frozen seed')
    return records


def historical_chunk_seeds(conn, cell, module, global_bbox):
    """Replay the supplied spatial-index traversal, including its seed order.

    Sorting by tree ID or distance would introduce a different tie convention.
    A bounded input retains every seed admitted by the original halo predicate.
    """
    records = chunk_seeds(conn, cell, module.CROWN_HALO_M, original_table_order=True)
    spatial_index = module.index_trees_by_chunk(records, module.CHUNK_SIZE_M, global_bbox)
    return module.trees_for_chunk(cell, spatial_index, module.CHUNK_SIZE_M, global_bbox)


def building_outline_polygons(path, module):
    """Load an explicitly supplied alternative; never infer an unknown CRS."""
    outlines = gpd.read_file(path)
    if outlines.crs is None or outlines.crs.to_epsg() != 2193:
        raise ValueError('alternative building outlines must declare EPSG:2193')
    polygons = [module._clean_polygon(geom) for geom in outlines.geometry]
    polygons = [geom for geom in polygons if geom is not None]
    if not polygons:
        raise ValueError('alternative building file has no usable polygons')
    return polygons


def configure_greenness_threshold(module, value):
    """Apply an explicit global candidate; do not change supplied file bytes."""
    default = module.CROWN_GLI_THRESHOLD
    if value is not None:
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('greenness threshold must be finite and within [-1,1]')
        module.CROWN_GLI_THRESHOLD = value
    return default


def frozen_crown_rows(conn, sites):
    rows = []
    for sid, site in sites.items():
        x0, y0, x1, y1 = site['bbox_epsg2193']
        query = '''SELECT c.*, l.x_2193, l.y_2193 FROM tree_crown_pilot c
                   JOIN tree_lidar_pilot l ON c.tree_id=l.tree_id
                   WHERE l.x_2193>=? AND l.x_2193<? AND l.y_2193>=? AND l.y_2193<?'''
        rows.extend(dict(site_id=sid, **dict(r)) for r in conn.execute(query, (x0, x1, y0, y1)))
    return rows


def scalar_comparison(records, frozen, sites):
    actual = {r['tree']['tree_id']: r for r in records}
    rows = []
    fields = ['crown_area_m2', 'crown_diameter_m', 'crown_mean_chm_m', 'crown_max_chm_m']
    for old in frozen:
        now = actual.get(old['tree_id'])
        row = {'site_id': old['site_id'], 'tree_id': old['tree_id'], 'reproduced': now is not None}
        for field in fields:
            row['frozen_' + field] = old[field]
            row['attempt_' + field] = now[field] if now else None
            row['delta_' + field] = (now[field] - old[field]) if now and old[field] is not None else None
        row['all_scalars_match_at_1e_6'] = bool(now and all(
            old[f] is not None and abs(now[f] - old[f]) <= 1e-6 for f in fields))
        rows.append(row)
    expected = {r['tree_id'] for r in frozen}
    extra = []
    for r in records:
        if r['tree']['tree_id'] in expected:
            continue
        t = r['tree']
        for sid, site in sites.items():
            x0, y0, x1, y1 = site['bbox_epsg2193']
            if x0 <= t['x_2193'] < x1 and y0 <= t['y_2193'] < y1:
                extra.append({'site_id': sid, 'tree_id': t['tree_id']})
    return rows, extra


def run_crowns(args, module, sites, cells, mask_cells, parameters):
    db = args.database.resolve()
    wal = Path(str(db) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('frozen database must be checkpointed with absent/empty WAL')
    intake = json.loads(args.intake.read_text())
    db_hash = sha256(db)
    if db_hash != intake['database_sha256_observed']:
        raise ValueError('database no longer matches its completed intake')
    raster_fingerprints = chm_fingerprints(args.chm)
    write_json(args.output / 'chm_source_fingerprints.json', raster_fingerprints)
    if args.mask_source == 'osm-current':
        buildings, nonveg, masks = load_masks(module, mask_cells, args.mask_cache,
                                             args.allow_missing_nonvegetation)
    else:
        buildings, nonveg, masks = [], [], {}
    building_receipt = None
    if args.building_outlines:
        buildings = building_outline_polygons(args.building_outlines, module)
        building_receipt = {
            'file':str(args.building_outlines), 'sha256':sha256(args.building_outlines),
            'provenance':json.loads(args.building_provenance.read_text()),
            'provenance_sha256':sha256(args.building_provenance),
            'polygons':len(buildings), 'osm_buildings_used':False,
            'historical_input_verified':False,
            'selection_rule':'supplied historical code prefers LINZ outlines over OSM when present',
        }
        write_json(args.output / 'alternative_building_input.json', building_receipt)
    exclusions = buildings + nonveg
    missing_masks = {k:v for k,v in masks.items() if v.get('status') == 'MISSING EXPLICITLY OMITTED'}
    write_json(args.output / 'used_mask_manifest.json', {
        'source': args.mask_source, 'files': masks,
        'complete_current_osm_coverage': bool(masks) and not missing_masks,
        'historical_snapshot':False, 'building_polygons':len(buildings),
        'nonvegetation_polygons':len(nonveg), 'missing_files':missing_masks,
        'alternative_building_input':building_receipt,
        'osm_building_payloads_used':building_receipt is None and bool(masks)})
    union_sites = unary_union([box(*s['bbox_epsg2193']) for s in sites.values()])
    inventory = {}
    records, chunks = [], []
    conn = sqlite3.connect(db.as_uri() + '?mode=ro&immutable=1', uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA query_only=ON')
    gli_src = rasterio.open(args.greenness) if args.greenness else None
    gli_fingerprints = greenness_fingerprints(args.greenness) if args.greenness else None
    if gli_fingerprints:
        write_json(args.output / 'greenness_input_manifest.json', gli_fingerprints)
    try:
        with rasterio.open(args.chm) as src:
            if src.crs is None or src.crs.to_epsg() != 2193 or src.res != (1.0, 1.0):
                raise ValueError('historical crown stage requires 1m EPSG:2193 CHM')
            grid = {'crs': src.crs.to_string(), 'resolution': list(src.res),
                    'transform': list(src.transform), 'nodata': src.nodata}
            for i, cell in enumerate(cells, 1):
                seeds = (historical_chunk_seeds(conn, cell, module, parameters['original_global_bbox'])
                         if args.seed_order == 'historical-chunk'
                         else chunk_seeds(conn, cell, module.CROWN_HALO_M))
                for seed in seeds:
                    inventory[seed['tree_id']] = seed
                chm, transform = module.read_chm_window(src, cell)
                region = box(*cell).buffer(module.CROWN_HALO_M)
                local_masks = [p for p in exclusions if p.intersects(region)]
                gli = module.read_gli_for_chunk(gli_src, cell, chm.shape, transform) if gli_src else None
                if gli_src and (gli is None or not np.isfinite(gli).any()):
                    raise ValueError('requested greenness is unavailable for an active chunk')
                start = time.monotonic()
                candidates = module.segment_chunk(chm, transform, local_masks, seeds, cell, gli_chunk=gli)
                retained = [r for r in candidates if r['geometry_2193'].intersects(union_sites)]
                if any(not r['geometry_2193'].is_valid for r in retained):
                    raise ValueError('historical function emitted invalid retained geometry')
                records.extend(retained)
                receipt = {'bbox': cell, 'seeds': len(seeds), 'mask_polygons': len(local_masks),
                           'valid_chm_fraction': float(np.isfinite(chm).mean()),
                           'generated_crowns': len(candidates), 'retained_site_intersections': len(retained),
                           'segmentation_wall_seconds': time.monotonic()-start}
                chunks.append(receipt)
                print(f'Chunk {i}/{len(cells)}: seeds={len(seeds)}, crowns={len(candidates)}, retained={len(retained)}', flush=True)
            frozen = frozen_crown_rows(conn, sites)
    finally:
        conn.close()
        if gli_src:
            gli_src.close()
    if len({r['tree']['tree_id'] for r in records}) != len(records):
        raise ValueError('duplicate crown IDs across original chunks')
    properties = []
    for r in records:
        properties.append({k: v for k, v in r.items() if k not in {'tree', 'geometry_2193'}} |
                          {'tree_id': r['tree']['tree_id'], 'source_primary': r['tree']['source_primary']})
    frame = gpd.GeoDataFrame(properties, columns=['crown_area_m2', 'crown_diameter_m',
                            'crown_mean_chm_m', 'crown_max_chm_m', 'tree_id', 'source_primary'],
                            geometry=[r['geometry_2193'] for r in records], crs=2193)
    frame.to_file(args.output / 'historical_default_crowns.gpkg', layer='crowns', driver='GPKG')
    # Same geographic geometry and stable ID convention as the original writer.
    geographic = frame.to_crs(4326)
    features = [{'type': 'Feature', 'id': row.tree_id,
                 'properties': {k: row[k] for k in properties[0]} if properties else {},
                 'geometry': mapping(row.geometry)} for _, row in geographic.iterrows()]
    write_json(args.output / 'tree_crowns_historical_attempt.geojson',
               {'type': 'FeatureCollection', 'features': features})
    write_json(args.output / 'chunk_receipts.json', chunks)
    write_json(args.output / 'frozen_seed_subset.json', list(inventory.values()))
    comparisons, extra = scalar_comparison(records, frozen, sites)
    with (args.output / 'frozen_scalar_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=comparisons[0].keys() if comparisons else ['site_id','tree_id'])
        writer.writeheader(); writer.writerows(comparisons)
    by_site = {sid: {'frozen_crowns_with_seed_inside': sum(r['site_id']==sid for r in comparisons),
                    'reproduced_frozen_ids': sum(r['site_id']==sid and r['reproduced'] for r in comparisons),
                    'all_scalars_match_at_1e_6': sum(r['site_id']==sid and r['all_scalars_match_at_1e_6'] for r in comparisons),
                    'extra_seed_inside_ids': sum(r['site_id']==sid for r in extra)} for sid in sites}
    # No source writes occur; verify the complete file once more after execution.
    db_after = sha256(db)
    if db_after != db_hash:
        raise ValueError('source database hash changed during the attempt')
    report = {'status': 'PROVISIONAL HISTORICAL CROWN-STAGE ATTEMPT',
              'created_utc': utc(), 'baseline_reproduction_verified': False,
              'reference_frozen': False, 'training_site_ids': list(sites),
              'crowns_intersecting_training_sites': len(frame), 'by_site': by_site,
              'frozen_seed_inside_crowns': len(frozen),
              'frozen_ids_reproduced': sum(r['reproduced'] for r in comparisons),
              'all_scalar_matches_at_1e_6': sum(r['all_scalars_match_at_1e_6'] for r in comparisons),
              'extra_seed_inside_ids': extra, 'database_sha256': db_hash,
              'source_database_unchanged': db_after == db_hash, 'chm_grid': grid,
              'chm_source_files_verified': len(raster_fingerprints['source_files']),
              'mask_source': args.mask_source, 'mask_files': masks,
              'current_mask_coverage_complete':bool(masks) and not missing_masks,
              'missing_nonvegetation_mask_files':missing_masks,
              'linz_building_source_used': bool(building_receipt),
              'alternative_building_input':building_receipt, 'historical_masks_available': False,
              'greenness_used': bool(args.greenness),
              'greenness_sha256': sha256(args.greenness) if args.greenness else None,
              'greenness_source_files_verified': len(gli_fingerprints['source_files']) if gli_fingerprints else 0,
              'missing_inputs': ['original historical masks and greenness; exact frozen-run use unknown'],
              'parameters': parameters,
              'limitations': ['producing commit/default settings not confirmed',
                              'current replacement masks cannot certify historical mask equality' if masks else 'building/nonvegetation masks explicitly disabled',
                              *(['one or more current nonvegetation query responses explicitly omitted; inspect used_mask_manifest.json'] if missing_masks else []),
                              'frozen postprocessing inventory is reused, not a reproduction of detection/canonicalisation',
                              'subset CHM may leave remote context nodata inside original chunks',
                              *(['bounded SQL order does not replay historical chunk-index traversal']
                                if args.seed_order == 'bounded-sql' else []),
                              'frozen crown polygon file unavailable; scalar equality cannot establish geometry equality']}
    write_json(args.output / 'summary.json', report)
    print(json.dumps({k: report[k] for k in ['status','crowns_intersecting_training_sites',
          'frozen_seed_inside_crowns','frozen_ids_reproduced','all_scalar_matches_at_1e_6','by_site']}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['masks','run'], required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--chm', type=Path, required=True)
    ap.add_argument('--chm-manifest', type=Path, required=True)
    ap.add_argument('--database', type=Path)
    ap.add_argument('--intake', type=Path)
    ap.add_argument('--mask-source', choices=['osm-current','none'], default='osm-current')
    ap.add_argument('--mask-cache', type=Path, required=True)
    ap.add_argument('--osm-request-subdivisions', type=int, choices=[1,2], default=1,
                    help='partition uncached geographic queries into N by N subrequests')
    ap.add_argument('--osm-endpoints', nargs='+',
                    help='explicit replacement read-only Overpass endpoints; recorded in provenance')
    ap.add_argument('--osm-http-method', choices=['GET','POST'], default='GET')
    ap.add_argument('--osm-query-timeout', type=int, choices=range(1,121),
                    help='optional lower server execution limit; partial/error responses remain rejected')
    ap.add_argument('--allow-missing-nonvegetation', action='store_true',
                    help='run-only approximation: record missing nonveg files explicitly; buildings still mandatory')
    ap.add_argument('--greenness', type=Path)
    ap.add_argument('--crown-gli-threshold', type=float,
                    help='explicit training-only inverse-calibration candidate; never historical confirmation')
    ap.add_argument('--building-outlines', type=Path,
                    help='explicit EPSG:2193 LINZ outline substitute; replaces OSM building polygons')
    ap.add_argument('--building-provenance', type=Path,
                    help='required provenance JSON for the alternative building file')
    ap.add_argument('--seed-order', choices=['historical-chunk','bounded-sql'],
                    default='historical-chunk',
                    help='replay original chunk-index traversal; bounded-sql reproduces the earlier adapter')
    ap.add_argument('--site-ids', nargs='+')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists(): ap.error('use a fresh output directory')
    if args.phase == 'run' and (not args.database or not args.intake):
        ap.error('--database and --intake are required for crown execution')
    if args.phase == 'masks' and args.mask_source != 'osm-current':
        ap.error('mask acquisition requires osm-current')
    if args.allow_missing_nonvegetation and (args.phase != 'run' or args.mask_source != 'osm-current'):
        ap.error('missing nonvegetation omission applies only to explicit current-OSM crown execution')
    if bool(args.building_outlines) != bool(args.building_provenance):
        ap.error('--building-outlines and --building-provenance must be supplied together')
    if args.building_outlines and (args.phase != 'run' or args.mask_source != 'osm-current'):
        ap.error('alternative building outlines require offline run with current nonvegetation caches')
    if args.crown_gli_threshold is not None:
        if args.phase != 'run' or not args.greenness:
            ap.error('--crown-gli-threshold requires an offline crown run with explicit greenness')
        if not math.isfinite(args.crown_gli_threshold) or not -1 <= args.crown_gli_threshold <= 1:
            ap.error('greenness threshold must be finite and within [-1,1]')
    sites = selected_training_sites(args.site_ids)
    module, source_provenance, source_hashes = source_module(args.source)
    original_gli_threshold = configure_greenness_threshold(module, args.crown_gli_threshold)
    manifest = json.loads(args.chm_manifest.read_text())
    global_bbox = manifest['pilot_bbox_2193']
    if manifest['resolution_m'] != 1.0 or manifest['chunk_size_m'] != module.CHUNK_SIZE_M:
        raise ValueError('CHM manifest and historical segmentation grid differ')
    regions = [box(*s['bbox_epsg2193']).buffer(16) for s in sites.values()]
    cells = select_grid_cells(global_bbox, module.CHUNK_SIZE_M, regions)
    mask_regions = [box(*cell).buffer(module.CROWN_HALO_M) for cell in cells]
    mask_cells = select_grid_cells(global_bbox, module.MAX_OSM_BBOX_M, mask_regions)
    parameters = {k: getattr(module,k) for k in ['CANOPY_MIN_HEIGHT_M','MIN_CROWN_AREA_M2',
                  'CHUNK_SIZE_M','CROWN_HALO_M','CROWN_GLI_THRESHOLD','CROWN_GLI_WINDOW_PX',
                  'MAX_CROWN_ASPECT_RATIO','MIN_CROWN_BBOX_FILL','OSM_NONVEG_CACHE_VERSION']}
    parameters.update(original_global_bbox=global_bbox, seed_order=args.seed_order,
                      supplied_default_CROWN_GLI_THRESHOLD=original_gli_threshold,
                      parameter_source=('inverse calibration against training-only frozen attributes'
                                        if args.crown_gli_threshold is not None else 'supplied defaults'),
                      radius_rule='clip(0.65*height,5,14); missing height 7m',
                      building_buffer_m=1.0, closing_m=0.65, simplify_m=0.45,
                      polygon_connectivity=4, candidate_connectivity=4,
                      seed_emission='original chunk strict interior', greenness_enabled=bool(args.greenness))
    args.output.mkdir(parents=True)
    (args.output / 'adapter_source.py').write_text(Path(__file__).read_text())
    write_json(args.output / 'plan.json', {'created_utc':utc(), 'phase':args.phase,
               'training_site_ids':list(sites), 'original_chunks':cells, 'osm_query_cells':mask_cells,
               'parameters':parameters, 'supplied_source_provenance':source_provenance,
               'source_file_hashes':source_hashes, 'adapter_sha256':sha256(Path(__file__)),
               'sites_sha256':sha256(ROOT/'data/sites/reference_sites.json'),
               'chm_vrt_sha256':sha256(args.chm), 'chm_manifest_sha256':sha256(args.chm_manifest),
               'reference_labels_read':False, 'source_database_write':False,
               'osm_request_subdivisions':args.osm_request_subdivisions,
               'osm_endpoints':list(args.osm_endpoints or module.OVERPASS_ENDPOINTS),
               'osm_http_method':args.osm_http_method,
               'osm_query_timeout':args.osm_query_timeout,
               'allow_missing_nonvegetation':args.allow_missing_nonvegetation,
               'producer_commit_verified':False})
    if args.phase == 'masks':
        fetch_current_masks(module, mask_cells, args.mask_cache, args.output,
                            args.osm_request_subdivisions, args.osm_endpoints,
                            args.osm_http_method, args.osm_query_timeout)
    else:
        run_crowns(args, module, sites, cells, mask_cells, parameters)
    files = {str(p.relative_to(args.output)):sha256(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
    write_json(args.output / 'SHA256_manifest.json', files)


if __name__ == '__main__':
    main()
