import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import geopandas as gpd
from shapely.geometry import box
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from benchmark.common import sha256,site_definitions
spec=importlib.util.spec_from_file_location('hypothetical_eval',ROOT/'scripts/61_evaluate_assumed_reference.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class AssumedEvaluation(unittest.TestCase):
    def test_known_perfect_models_have_zero_paired_difference_without_formal_claim(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sat=root/'sat';sat.mkdir();ref=root/'reference.gpkg';scenario=root/'scenario.json';scenario.write_text('{}')
            sites={sid:s for sid,s in site_definitions().items() if s['split']=='test'}
            ids=list(sites);geoms=[box(s['bbox_epsg2193'][0]+30,s['bbox_epsg2193'][1]+30,s['bbox_epsg2193'][0]+40,s['bbox_epsg2193'][1]+40) for s in sites.values()]
            frame=gpd.GeoDataFrame({'site_id':ids,'label_id':['a']*len(ids)},geometry=geoms,crs=2193)
            frame.to_file(ref,layer='single_trees',driver='GPKG')
            for layer in ['tree_groups','uncertainty_regions']:frame.iloc[:0].to_file(ref,layer=layer,driver='GPKG')
            source='synthetic-reference'
            (root/'provenance.json').write_text(json.dumps({'output_gpkg_sha256':sha256(ref),'source_json_sha256':source}))
            pred=gpd.GeoDataFrame({'site_id':ids,'prediction_id':['a']*len(ids)},geometry=geoms,crs=2193)
            pred.to_file(sat/'test_predictions.gpkg',layer='predictions',driver='GPKG')
            (sat/'assembly_receipt.json').write_text(json.dumps({'status':'ASSUMPTION-ONLY HELD-OUT PREDICTIONS ASSEMBLED; NO SCORING','inference_scenario_sha256':sha256(scenario),'output_sha256':{'test':sha256(sat/'test_predictions.gpkg')},'sites':[{'site_id':sid,'split':'test'} for sid in ids]}))
            classical=root/'classic';classical.mkdir();pred.to_file(classical/'predictions.gpkg',layer='predictions',driver='GPKG')
            (classical/'provenance.json').write_text(json.dumps({'output_sha256':sha256(classical/'predictions.gpkg'),'input_sha256':'synthetic-crowns','requested_site_ids':ids}))
            receipt=root/'classical.json';receipt.write_text(json.dumps({'status':'FIXED PROVISIONAL CLASSICAL HELD-OUT PREDICTIONS; NO SCORING','scenario_sha256':sha256(scenario),'site_ids':[sid for sid,s in site_definitions().items() if s['split']!='train'],'crowns_sha256':'synthetic-crowns'}))
            argv=['evaluate','--inference-scenario',str(scenario),'--reference',str(ref),'--sat-assembly',str(sat),'--classical-predictions',str(classical/'predictions.gpkg'),'--classical-receipt',str(receipt),'--output',str(root/'result')]
            with patch.object(sys,'argv',argv),patch.object(m,'validate_inference_scenario',return_value={'reference_json_sha256':source}):m.main()
            r=json.loads((root/'result/hypothetical_comparison.json').read_text())
            self.assertEqual(r['pretrained_summary']['f1'],1.0);self.assertEqual(r['pretrained_summary']['median_matched_iou'],1.0)
            self.assertEqual(r['pretrained_summary'],r['classical_provisional_summary'])
            self.assertFalse(r['reference_frozen']);self.assertFalse(r['acceptance_declared'])
            self.assertFalse(r['bootstrap']['positive_improvement_ci_excludes_zero'])
