import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import laspy
import numpy as np
from pyproj import CRS
from shapely.geometry import box

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
evaluate=load('evaluate_cli','12_evaluate.py')
compare=load('compare_baseline','15_compare_baseline_outputs.py')
compute=load('compute_cost','18_summarise_compute.py')
footprint=load('footprint','41_measure_laz_footprint.py')


class WorkflowGuards(unittest.TestCase):
    def setUp(self):
        base=ROOT/'outputs/preparation/synthetic_workflow_checks';base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.path=Path(self.tmp.name)
        self.reference=self.path/'synthetic_reference.gpkg'
        self.predictions=self.path/'synthetic_predictions.gpkg'
        self.site=next(s for s in evaluate.site_definitions().values() if s['split']=='train')
        x,y,_,_=self.site['bbox_epsg2193'];geom=box(x+10,y+10,x+12,y+12)
        self.geom=geom
        for layer in ['single_trees','tree_groups','uncertainty_regions']:
            frame=gpd.GeoDataFrame({'site_id':[self.site['site_id']] if layer=='single_trees' else [],
                                    'label_id':['synthetic_ref'] if layer=='single_trees' else []},
                                   geometry=[geom] if layer=='single_trees' else [],crs=2193)
            frame.to_file(self.reference,layer=layer,driver='GPKG')
        self.frame=gpd.GeoDataFrame({'site_id':[self.site['site_id']],'prediction_id':['synthetic_pred']},geometry=[geom],crs=2193)
        self.frame.to_file(self.predictions,layer='predictions',driver='GPKG')

    def tearDown(self):self.tmp.cleanup()

    def run_cli(self,extra=()):
        argv=['12_evaluate.py','--reference',str(self.reference),'--predictions',str(self.predictions),
              '--output',str(self.path/'result'),*extra]
        with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):evaluate.main()

    def test_end_to_end_training_fixture_not_benchmark(self):
        self.run_cli()
        result=json.loads((self.path/'result/evaluation.json').read_text())
        self.assertEqual(result['summary']['f1'],1)
        self.assertEqual(result['split'],'train')
        self.assertIsNone(result['core_acceptance_met'])
        self.assertEqual(len(result['by_site']),8)
        self.assertTrue(result['criteria']['false_positive_precision'])
        self.assertEqual(result['false_positive_precision_source'],'same scored TP/(TP+FP)')
        self.assertIsNone(result['by_stratum']['hard_negative_built']['recall'])

    def test_development_site_subset(self):
        self.run_cli(['--site-ids',self.site['site_id']])
        result=json.loads((self.path/'result/evaluation.json').read_text())
        self.assertEqual(list(result['by_site']),[self.site['site_id']])
        self.assertEqual(result['summary']['f1'],1)

    def test_development_site_subset_rejects_nontraining_site(self):
        with self.assertRaisesRegex(ValueError,'unique training site IDs'):
            self.run_cli(['--site-ids','S007'])

    def test_benchmark_rejects_site_subset(self):
        with self.assertRaisesRegex(ValueError,'development-only'):
            self.run_cli(['--stage','benchmark','--site-ids','S016'])

    def test_held_out_policy_guard_before_model_scoring(self):
        draft=json.loads((ROOT/'config/evaluation_draft.json').read_text());draft['status']='draft'
        path=self.path/'draft_policy.json';path.write_text(json.dumps(draft))
        with self.assertRaisesRegex(ValueError,'confirmed operational policy'):self.run_cli(['--stage','benchmark','--policy',str(path)])
        self.assertFalse((self.path/'result').exists())

    def test_researcher_fixed_policy_still_requires_reference_freeze(self):
        with self.assertRaisesRegex(ValueError,'reference freeze record'):self.run_cli(['--stage','benchmark'])
        self.assertFalse((self.path/'result').exists())

    def test_same_precision_rejects_selective_negative_audit(self):
        path=self.path/'synthetic_fp_audit.json';path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'do not substitute'):self.run_cli(['--fp-audit',str(path)])

    def test_training_input_rejects_validation_site(self):
        self.frame.loc[0,'site_id']='S007'
        self.frame.to_file(self.predictions,layer='predictions',driver='GPKG')
        with self.assertRaisesRegex(ValueError,'out-of-split'):self.run_cli()

    def test_reproduction_known_equal_shift_missing(self):
        a=gpd.GeoDataFrame({'tree_id':['synthetic_tree']},geometry=[self.geom],crs=2193)
        self.assertTrue(compare.compare_frames(a,a,'tree_id')['geometry_matches'])
        b=a.copy();b.geometry=[box(*(v+.1 for v in self.geom.bounds))]
        self.assertFalse(compare.compare_frames(a,b,'tree_id')['geometry_matches'])
        self.assertFalse(compare.compare_frames(a,a.iloc[:0],'tree_id')['geometry_matches'])

    def test_known_area_time_cost_and_unknown_rate(self):
        result=compute.compute_summary(3600,2,4,7200)
        self.assertEqual(result['wall_seconds_per_km2'],1800)
        self.assertEqual(result['wall_hours_per_km2'],.5)
        self.assertEqual(result['cost_per_km2'],4)
        self.assertEqual(result['km2_per_hour'],2)
        self.assertIsNone(compute.compute_summary(3600,2)['cost_per_km2'])
        with self.assertRaises(ValueError):compute.compute_summary(0,2)
        self.assertEqual(compute.wall_seconds_from_timing({'elapsed_seconds':3723,'exit_code':0}),3723)
        self.assertEqual(compute.wall_seconds_from_timing({'wall_seconds':3723}),3723)
        with self.assertRaisesRegex(ValueError,'conflicting'):
            compute.wall_seconds_from_timing({'elapsed_seconds':1,'wall_seconds':2})

    def test_laz_envelope_area_is_not_small_site_area(self):
        header=laspy.LasHeader(point_format=6,version='1.4')
        header.add_crs(CRS.from_epsg(2193))
        header.offsets=[1700000,5900000,0]
        header.scales=[.01,.01,.01]
        cloud=laspy.LasData(header)
        cloud.x=np.array([1700000,1701000])
        cloud.y=np.array([5900000,5902000])
        cloud.z=np.array([1,1])
        path=self.path/'synthetic_footprint.las'
        cloud.write(path)
        record=footprint.measure(path)
        self.assertAlmostEqual(record['processed_area_km2'],2)
        self.assertEqual(record['point_count'],2)
        self.assertIn('proxy',record['footprint_description'])


if __name__=='__main__':unittest.main()
