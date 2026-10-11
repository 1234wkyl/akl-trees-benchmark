"""Known-answer regressions for removed extreme noise and safe stage reuse."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import laspy
from pyproj import CRS
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from benchmark.common import sha256

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
merge=module('filtered_merger','scripts/46_merge_training_site_tiles.py')
assembly=module('resumable_assembly','scripts/58_assemble_assumed_heldout_predictions.py')

class FilteredExtent(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        h=laspy.LasHeader(point_format=3,version='1.2');h.add_crs(CRS.from_epsg(2193))
        c=laspy.LasData(h);c.x=[1.,9.];c.y=[1.,9.];c.z=[3.,4.]
        self.input=self.root/'input.las';c.write(self.input)
        c.add_extra_dim(laspy.ExtraBytesParams(name='PredInstance',type='uint16'))
        c.add_extra_dim(laspy.ExtraBytesParams(name='PredSemantic',type='uint8'))
        c['PredInstance']=[1,1];c['PredSemantic']=[1,1]
        self.pred=self.root/'prediction.las';c.write(self.pred)
        self.audit=self.root/'audit.json';self.record={'input_sha256':sha256(self.input),
            'prediction_sha256':sha256(self.pred),'epsg2193_coordinate_consistency':True,
            'point_count_equal':True,'bidirectional_coordinate_coverage_ok':True,'input_points':2}
        self.audit.write_text(json.dumps(self.record))
    def tearDown(self):self.tmp.cleanup()
    def load(self,input_path):
        return merge._load_tile('synthetic',self.pred,self.audit,[0,0,10,10],[0,0,10,10],input_path)
    def test_removed_extreme_noise_is_not_lost_model_data(self):
        with self.assertRaisesRegex(ValueError,'coverage plan'):self.load(None)
        r=self.load(self.input)
        self.assertEqual(r['points_context'],2)
        self.assertEqual(r['extent_validation']['filtered_bbox_epsg2193'],[1,1,9,9])
        self.assertEqual(r['extent_validation']['raw_coverage_bbox_epsg2193'],[0,0,10,10])
    def test_wrong_input_hash_is_not_accepted(self):
        self.record['input_sha256']='wrong';self.audit.write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError,'does not bind'):self.load(self.input)
    def test_missing_coordinate_coverage_is_not_accepted(self):
        self.record['bidirectional_coordinate_coverage_ok']=False;self.audit.write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError,'coverage audit'):self.load(self.input)
    def test_point_count_mismatch_is_not_accepted(self):
        self.record['input_points']=3;self.audit.write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError,'point count'):self.load(self.input)

class ResumeBinding(unittest.TestCase):
    def test_matching_standard_output_reused_but_wrong_site_or_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'standard';p.mkdir();c=root/'crowns';c.mkdir()
            (p/'predictions.gpkg').write_bytes(b'known-standard-output');(c/'candidate_crowns.gpkg').write_bytes(b'known-crown-output')
            r={'output_sha256':sha256(p/'predictions.gpkg'),'input_sha256':sha256(c/'candidate_crowns.gpkg'),'split':'val','requested_site_ids':['S005']}
            (p/'provenance.json').write_text(json.dumps(r))
            assembly.verify_completed_standard(p,c,'S005','val')
            with self.assertRaises(ValueError):assembly.verify_completed_standard(p,c,'S007','val')
            (p/'predictions.gpkg').write_bytes(b'changed')
            with self.assertRaises(ValueError):assembly.verify_completed_standard(p,c,'S005','val')
if __name__=='__main__':unittest.main()
