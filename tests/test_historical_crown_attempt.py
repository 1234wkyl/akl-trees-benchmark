import importlib.util
import json
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('historical_attempt', ROOT / 'scripts/51_attempt_historical_crowns.py')
ATTEMPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ATTEMPT)


class HistoricalCrownAttempt(unittest.TestCase):
    def test_busy_retry_retains_completed_subrequests_and_waits_before_same_query(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import requests
        query='[out:json][timeout:90];way["building"](-38,174,-36,176);out geom;'
        payload={'elements':[],'osm3s':{'timestamp_osm_base':'2026-10-06T00:00:00Z'}}
        good=SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:payload)
        def fail():raise requests.HTTPError('busy')
        busy=SimpleNamespace(status_code=429,text='busy',raise_for_status=fail)
        source=SimpleNamespace(OVERPASS_ENDPOINTS=['https://example.test'],OVERPASS_POLITE_SLEEP_S=0)
        source.fetch_osm_buildings_tile=lambda cell,cache:source._overpass_get(query)
        source.fetch_osm_non_vegetation_tile=lambda cell,cache:None
        with tempfile.TemporaryDirectory() as d,patch.object(ATTEMPT.requests,'request',side_effect=[good,good,busy,good,good]) as calls,patch.object(ATTEMPT,'load_masks',return_value=([],[],{})),patch.object(ATTEMPT.time,'sleep') as sleep:
            path=Path(d)
            ATTEMPT.fetch_current_masks(source,[(0,0,4000,4000)],path,path,subdivisions=2,server_busy_retries=1)
            self.assertEqual(calls.call_count,5)
            sent=[r.kwargs['params']['data'] for r in calls.call_args_list]
            self.assertEqual(sent[2],sent[3]);self.assertEqual(len(set(sent)),4)
            sleep.assert_any_call(30)
            self.assertEqual([r['http_status'] for r in json.loads((path/'network_attempts.json').read_text())],[200,200,429,200,200])

    def test_request_memory_hint_preserves_selectors_and_rejects_partial_results(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        query='[out:json][timeout:90];way["building"](-38,174,-36,176);out geom;'
        for partial in (False,True):
            source=SimpleNamespace(OVERPASS_ENDPOINTS=['https://example.test'],OVERPASS_POLITE_SLEEP_S=0)
            source.fetch_osm_buildings_tile=lambda cell,cache: source._overpass_get(query)
            source.fetch_osm_non_vegetation_tile=lambda cell,cache: None
            payload={'elements':[],'osm3s':{'timestamp_osm_base':'2026-10-06T00:00:00Z'}}
            if partial:payload['remark']='query ran out of memory'
            response=SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:payload)
            with tempfile.TemporaryDirectory() as folder,patch.object(ATTEMPT.requests,'request',return_value=response) as call,patch.object(ATTEMPT,'load_masks',return_value=([],[],{})):
                path=Path(folder)
                if partial:
                    with self.assertRaises(RuntimeError):ATTEMPT.fetch_current_masks(source,[(0,0,4000,4000)],path,path,query_timeout=25,maxsize_mib=64)
                    self.assertFalse((path/'mask_cache_manifest.json').exists())
                else:ATTEMPT.fetch_current_masks(source,[(0,0,4000,4000)],path,path,query_timeout=25,maxsize_mib=64)
                sent=call.call_args.kwargs['params']['data']
                self.assertIn('[maxsize:67108864]',sent)
                self.assertEqual(sent.split(';',1)[1],query.split(';',1)[1])
                self.assertEqual(json.loads((path/'network_attempts.json').read_text())[0]['query'],sent)

    def test_global_threshold_candidate_is_explicit_and_does_not_change_other_constants(self):
        from types import SimpleNamespace
        module=SimpleNamespace(CROWN_GLI_THRESHOLD=.03,CROWN_GLI_WINDOW_PX=5,
                               CANOPY_MIN_HEIGHT_M=3.,MIN_CROWN_AREA_M2=4.)
        self.assertEqual(ATTEMPT.configure_greenness_threshold(module,None),.03)
        self.assertEqual(module.CROWN_GLI_THRESHOLD,.03)
        self.assertEqual(ATTEMPT.configure_greenness_threshold(module,.02),.03)
        self.assertEqual(vars(module),{'CROWN_GLI_THRESHOLD':.02,'CROWN_GLI_WINDOW_PX':5,
                                     'CANOPY_MIN_HEIGHT_M':3.,'MIN_CROWN_AREA_M2':4.})

    def test_global_threshold_rejects_nonfinite_and_out_of_range_values(self):
        from types import SimpleNamespace
        for value in [float('nan'),float('inf'),-1.01,1.01]:
            module=SimpleNamespace(CROWN_GLI_THRESHOLD=.03)
            with self.assertRaisesRegex(ValueError,'finite'):
                ATTEMPT.configure_greenness_threshold(module,value)
            self.assertEqual(module.CROWN_GLI_THRESHOLD,.03)

    def test_greenness_fingerprint_binds_tiff_bytes_not_only_vrt_wrapper(self):
        with tempfile.TemporaryDirectory() as folder:
            raster=Path(folder)/'gli.tif'
            vrt=Path(folder)/'greenness.vrt'
            with rasterio.open(raster,'w',driver='GTiff',width=2,height=2,count=1,
                               dtype='float32',crs=2193,transform=from_origin(100,202,1,1)) as f:
                f.write(np.ones((2,2),dtype='float32'),1)
            vrt.write_text('<VRTDataset><VRTRasterBand><SimpleSource>'
                           '<SourceFilename relativeToVRT="1">gli.tif</SourceFilename>'
                           '</SimpleSource></VRTRasterBand></VRTDataset>')
            before=ATTEMPT.greenness_fingerprints(vrt)
            with rasterio.open(raster,'r+') as f:
                f.write(np.zeros((2,2),dtype='float32'),1)
            after=ATTEMPT.greenness_fingerprints(vrt)
            self.assertEqual(before['wrapper_sha256'],after['wrapper_sha256'])
            self.assertNotEqual(before['source_files'],after['source_files'])
            self.assertFalse(after['historical_input_verified'])

    def test_alternative_buildings_require_projected_crs_and_preserve_geometry(self):
        source = ROOT.parent / 'Xi_Zhang_Benchmark/source_reference'
        if not source.exists():
            self.skipTest('supplied historical source is an external dependency')
        module, _, _ = ATTEMPT.source_module(source)
        from shapely.geometry import Polygon
        polygon = Polygon([(0,0),(20,0),(20,20),(0,20)],
                          holes=[[(5,5),(5,15),(15,15),(15,5)]])
        with tempfile.TemporaryDirectory() as folder:
            valid=Path(folder)/'buildings.gpkg'
            gpd.GeoDataFrame({'geometry':[polygon]},crs='EPSG:2193').to_file(valid,driver='GPKG')
            loaded=ATTEMPT.building_outline_polygons(valid,module)
            self.assertEqual(len(loaded),1)
            self.assertTrue(loaded[0].equals(polygon))
            wrong=Path(folder)/'wrong_crs.gpkg'
            gpd.GeoDataFrame({'geometry':[polygon]},crs='EPSG:4326').to_file(wrong,driver='GPKG')
            with self.assertRaisesRegex(ValueError,'EPSG:2193'):
                ATTEMPT.building_outline_polygons(wrong,module)

    def test_seed_order_replays_original_chunk_index_not_sql_or_tree_id_order(self):
        source = ROOT.parent / 'Xi_Zhang_Benchmark/source_reference'
        if not source.exists():
            self.skipTest('supplied historical source is an external dependency')
        module, _, _ = ATTEMPT.source_module(source)
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE trees (tree_id TEXT, source_primary TEXT, source_tree_id TEXT, '
                     'species_common TEXT, species_latin TEXT, owner_class TEXT, '
                     'is_protected_notable INTEGER, lon REAL, lat REAL)')
        conn.execute('CREATE TABLE tree_lidar_pilot (tree_id TEXT, x_2193 REAL, y_2193 REAL, '
                     'chm_local_max_2m_m REAL)')
        rows = [('z_centre',1030.,1030.), ('a_west',1020.,1030.),
                ('b_south',1030.,1020.), ('far',5000.,5000.)]
        for ident,x,y in rows:
            conn.execute('INSERT INTO trees VALUES (?,?,?,?,?,?,?,?,?)',
                         (ident,'example',ident,None,None,None,0,0.,0.))
            conn.execute('INSERT INTO tree_lidar_pilot VALUES (?,?,?,?)',(ident,x,y,20.))
        cell=(1024,1024,2048,2048); bbox=(0,0,6144,6144)
        full=[{'tree_id':ident,'x_2193':x,'y_2193':y,'chm_local_max_2m_m':20.}
              for ident,x,y in rows]
        expected=module.trees_for_chunk(cell,module.index_trees_by_chunk(full,1024,bbox),1024,bbox)
        actual=ATTEMPT.historical_chunk_seeds(conn,cell,module,bbox)
        self.assertEqual([r['tree_id'] for r in actual],[r['tree_id'] for r in expected])
        self.assertEqual([r['tree_id'] for r in actual],['a_west','b_south','z_centre'])
        self.assertEqual(set(r['tree_id'] for r in actual),{'a_west','b_south','z_centre'})
        conn.close()

    def test_request_partition_covers_same_bbox_and_retains_selectors(self):
        query = '[out:json][timeout:90];(way["building"](-38,174,-36,176);relation["building"](-38,174,-36,176););out geom;'
        parts = ATTEMPT.subdivide_osm_query(query, 2)
        regions = []
        for part in parts:
            self.assertEqual(part.count('["building"]'), 2)
            bounds = re.findall(r'\((-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)\)', part)
            self.assertEqual(len(set(bounds)), 1)
            south, west, north, east = map(float, bounds[0])
            regions.append(box(west,south,east,north))
        from shapely.ops import unary_union
        self.assertEqual(len(parts), 4)
        self.assertTrue(unary_union(regions).equals(box(174,-38,176,-36)))
        self.assertEqual(sum(p.area for p in regions), 4)

    def test_subrequest_merge_deduplicates_and_rejects_inconsistent_geometry(self):
        first = {'osm3s': {'timestamp_osm_base':'2026-10-06T01:00:00Z'},
                 'elements':[{'type':'way','id':1,'geometry':[{'lat':1,'lon':2}]}]}
        self.assertEqual(len(ATTEMPT.merge_osm_payloads([first,first])['elements']), 1)
        changed = {'osm3s': first['osm3s'],
                   'elements':[{'type':'way','id':1,'geometry':[{'lat':2,'lon':2}]}]}
        with self.assertRaisesRegex(ValueError, 'changed'):
            ATTEMPT.merge_osm_payloads([first,changed])

    def test_original_grid_and_both_sides_of_chunk_boundary_are_retained(self):
        cells = ATTEMPT.select_grid_cells((100, 200, 3172, 3272), 1024,
                                         [box(1118, 500, 1130, 510)])
        self.assertEqual(cells, [(100, 200, 1124, 1224), (1124, 200, 2148, 1224)])

    def test_heldout_site_request_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'training'):
            ATTEMPT.selected_training_sites(['S020'])

    def test_offline_mode_does_not_silently_disable_missing_masks(self):
        class Source:
            OSM_NONVEG_CACHE_VERSION = 'v2'
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                ATTEMPT.load_masks(Source, [(0, 0, 4000, 4000)], Path(folder))

    def test_explicit_nonveg_omission_is_recorded_but_never_omits_buildings(self):
        class Source:
            OSM_NONVEG_CACHE_VERSION = 'v2'
            osm_payloads_to_polygons = staticmethod(lambda payloads: [])
            osm_payloads_to_exclusion_polygons = staticmethod(lambda payloads: [])
        cell = (0,0,4000,4000)
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            with self.assertRaises(FileNotFoundError):
                ATTEMPT.load_masks(Source, [cell], cache, True)
            bname,nname = ATTEMPT.mask_names(Source, cell)
            target = cache/bname
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps({'elements':[]}))
            with self.assertRaises(FileNotFoundError):
                ATTEMPT.load_masks(Source, [cell], cache)
            _,_,receipt = ATTEMPT.load_masks(Source, [cell], cache, True)
            self.assertEqual(receipt[nname]['status'], 'MISSING EXPLICITLY OMITTED')
            self.assertEqual(receipt[nname]['original_query_cell_2193'], list(cell))

    def test_real_historical_kernel_rejects_buildings_and_emits_only_seed_owner_chunk(self):
        source = ROOT.parent / 'Xi_Zhang_Benchmark/source_reference'
        if not source.exists():
            self.skipTest('supplied historical source is an external benchmark dependency')
        module, _, _ = ATTEMPT.source_module(source)
        heights = np.zeros((30, 30), dtype='float32')
        heights[9:21, 9:21] = 24
        transform = from_origin(0, 30, 1, 1)
        seeds = [{'tree_id':'example','x_2193':15.5,'y_2193':15.5,'chm_local_max_2m_m':24}]
        crown = module.segment_chunk(heights, transform, [], seeds, (0,0,30,30))
        self.assertEqual(len(crown), 1)
        self.assertEqual(crown[0]['crown_area_m2'], 144)
        self.assertEqual(module.segment_chunk(heights, transform, [box(8,8,22,22)], seeds, (0,0,30,30)), [])
        self.assertEqual(module.segment_chunk(heights, transform, [], seeds, (16,0,30,30)), [])


if __name__ == '__main__':
    unittest.main()
