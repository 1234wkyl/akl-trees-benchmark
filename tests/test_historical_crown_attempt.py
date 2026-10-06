import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('historical_attempt', ROOT / 'scripts/51_attempt_historical_crowns.py')
ATTEMPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ATTEMPT)


class HistoricalCrownAttempt(unittest.TestCase):
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
