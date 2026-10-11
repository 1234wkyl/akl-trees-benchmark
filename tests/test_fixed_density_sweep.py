"""Known-answer thinning guard and conditional density-report checks."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import laspy,numpy as np
from pyproj import CRS
from shapely.geometry import box
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark.evaluation import aggregate,paired_site_bootstrap,score_site

def load(name,file):
    s=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'scripts'/file)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
density=load('density13','13_prepare_density.py');report=load('density68','68_report_fixed_density_sweep.py')

class FixedDensity(unittest.TestCase):
    def test_heldout_requires_explicit_valid_protocol_and_preserves_records(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'tile.laz';scenario=root/'scenario.json';scenario.write_text('{}')
            header=laspy.LasHeader(point_format=3,version='1.2');header.add_crs(CRS.from_epsg(2193))
            data=laspy.LasData(header);data.x=np.arange(101)+100;data.y=np.arange(101)+200;data.z=np.arange(101)%7
            data.intensity=np.arange(101)*3;data.classification=np.arange(101)%5;data.write(source)
            sites={'S002':{'site_id':'S002','split':'test','bbox_epsg2193':[99,199,202,302]}}
            args=['density','--input',str(source),'--fractions','.75','.5','.25','.125','--output',str(root/'default')]
            with patch.object(density,'site_definitions',return_value=sites),patch.object(sys,'argv',args):
                with self.assertRaisesRegex(ValueError,'training-only'):density.main()
            args[args.index(str(root/'default'))]=str(root/'explicit');args+=['--inference-scenario',str(scenario)]
            with patch.object(density,'site_definitions',return_value=sites),patch.object(sys,'argv',args),patch.object(density,'validate_inference_scenario',return_value={'validated':True}):density.main()
            manifest=json.loads((root/'explicit/manifest.json').read_text())
            self.assertEqual(manifest['training_sites'],[]);self.assertEqual(manifest['held_out_sites'],['S002'])
            self.assertTrue(manifest['source_unchanged']);previous=set(range(101))
            for r in manifest['runs']:
                p=root/'explicit'/r['input_file'];indices=np.load(p.parent/'source_point_indices.npy')
                self.assertEqual(len(indices),int(101*r['fraction']));self.assertLessEqual(set(indices),previous)
                self.assertTrue(np.array_equal(laspy.read(p).points.array,laspy.read(source).points.array[indices]))
                previous=set(indices)
            with patch.object(density,'validate_inference_scenario',side_effect=ValueError('invalid protocol')),patch.object(sys,'argv',args):
                with self.assertRaisesRegex(ValueError,'invalid protocol'):density.main()

    def test_no_predictions_remain_undefined_and_reference_change_rejected(self):
        scores={'S002':score_site([('r',box(0,0,1,1))],[])}
        c={'reference_frozen':False,'actual_independent_agreement_available':False,
           'sources':{'reference':'ref','policy':'pol','classical':'classic'},'site_ids':['S002'],
           'pretrained_summary':aggregate(list(scores.values())),'pretrained_by_site':scores,
           'bootstrap':paired_site_bootstrap(scores,scores,2000,42)}
        comparisons={f:json.loads(json.dumps(c)) for f in (1.,.75,.5,.25,.125)}
        rows,_=report.assemble_rows(comparisons,{f:100 for f in comparisons})
        self.assertEqual(len(rows),5);self.assertIsNone(rows[-1]['precision']);self.assertIsNone(rows[-1]['median_matched_iou'])
        self.assertEqual(rows[-1]['recall'],0)
        comparisons[.5]['sources']['reference']='different'
        with self.assertRaisesRegex(ValueError,'identical conditional reference'):report.assemble_rows(comparisons,{f:100 for f in comparisons})
