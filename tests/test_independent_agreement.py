import copy
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('intake',ROOT/'scripts/31_receive_independent_labels.py')
intake=importlib.util.module_from_spec(spec);spec.loader.exec_module(intake)
a=intake.agreement


class IndependentAgreementCases(unittest.TestCase):
    def frame(self,rows):
        return gpd.GeoDataFrame(rows,columns=['label_id','class','geometry'],geometry='geometry',crs=2193)
    def test_class_disagreement_visible_and_not_a_single_match(self):
        fa=self.frame([{'label_id':'a','class':'tree','geometry':box(0,0,10,10)}])
        fb=self.frame([{'label_id':'b','class':'tree_group','geometry':box(0,0,10,10)}])
        pooled,_,_=a.greedy_match(list(fa.geometry),list(fb.geometry),.5)
        self.assertEqual(len(pooled),1)
        by_class=a.per_class_agreement({'S':(fa,fb)},.5,['tree','tree_group'])
        self.assertEqual(by_class['tree']['matched'],0)
        self.assertEqual(by_class['tree']['f1'],0)
        self.assertEqual(a.class_kappa([('tree','tree_group')]),0)
    def test_stable_ties_independent_of_input_order(self):
        g=box(0,0,10,10)
        first,_,_=a.greedy_match([g,g],[g],.5,['z','a'],['b'])
        second,_,_=a.greedy_match([g,g],[g],.5,['a','z'],['b'])
        self.assertEqual(first[0][0],1);self.assertEqual(second[0][0],0)
    def test_half_iou_is_inclusive(self):
        pairs,_,_=a.greedy_match([box(0,0,10,10)],[box(0,0,5,10)],.5)
        self.assertEqual(pairs,[(0,0,.5)])
    def test_no_objects_and_one_class_keep_undefined(self):
        empty=self.frame([])
        result=a.per_class_agreement({'S':(empty,empty)},.5,['tree'])
        self.assertIsNone(result['tree']['f1']);self.assertIsNone(result['tree']['median_iou'])
        self.assertIsNone(a.class_kappa([]));self.assertIsNone(a.class_kappa([('tree','tree')]))
    def test_kappa_known_two_class_table(self):
        self.assertAlmostEqual(a.class_kappa([('tree','tree')]*4+[('tree_group','tree_group')]*4+[('tree','tree_group'),('tree_group','tree')]),.6)
    def write_fixture(self,directory,complete=True,world_offset=0,duplicate=False):
        sites=intake.site_definitions();records={}
        for sid in ['S016','S020']:
            x0,y0,x1,y1=sites[sid]['bbox_epsg2193']
            world=[[x0+10+world_offset,y0+10],[x0+15+world_offset,y0+10],[x0+15+world_offset,y0+15],[x0+10+world_offset,y0+15]]
            lab={'id':sid+'-fixture','labeller':'TW','cls':'tree','world':world}
            records[sid]={'complete':complete,'labels':[lab,copy.deepcopy(lab)] if duplicate else [lab]}
        p=Path(directory)/'fixture.json';p.write_text(json.dumps({'schema_version':'akl-trees-crown-labels/v1','crs':'EPSG:2193','labeller':'TW','sites':records,'synthetic':True}))
        return p,sites
    def test_incomplete_site_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p,sites=self.write_fixture(td,complete=False)
            with self.assertRaisesRegex(ValueError,'complete'):intake.audit_json(p,'TW',sites,['S016','S020'])
    def test_loader_preserves_polygon_holes(self):
        with tempfile.TemporaryDirectory() as td:
            p,sites=self.write_fixture(td);d=json.loads(p.read_text())
            x0,y0,_,_=sites['S016']['bbox_epsg2193']
            d['sites']['S016']['labels'][0]['holes_world']=[[[x0+11,y0+11],[x0+13,y0+11],[x0+13,y0+13],[x0+11,y0+13]]]
            p.write_text(json.dumps(d))
            frame,_=a.load_independent_json(p,'TW',sites)
            geom=frame[frame.site_id=='S016'].geometry.iloc[0]
            self.assertAlmostEqual(geom.area,21);self.assertEqual(len(geom.interiors),1)
    def test_revision_attribution_is_explicit_and_author_only(self):
        with tempfile.TemporaryDirectory() as td:
            p,sites=self.write_fixture(td);d=json.loads(p.read_text());d['labeller']='XZ'
            for rec in d['sites'].values():rec['labels'][0]['labeller']='Codex_AI_refinement_of_XZ'
            p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'mixed labellers'):a.load_independent_json(p,'XZ',sites)
            frame,_=a.load_independent_json(p,'XZ',sites,allow_revision_attribution=True)
            self.assertEqual(len(frame),2)
            d['labeller']='TW';p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'mixed labellers'):a.load_independent_json(p,'TW',sites,allow_revision_attribution=True)
    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p,sites=self.write_fixture(td,duplicate=True)
            with self.assertRaisesRegex(ValueError,'duplicate'):intake.audit_json(p,'TW',sites,['S016','S020'])
    def test_wrong_site_coordinates_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p,sites=self.write_fixture(td,world_offset=1000)
            with self.assertRaisesRegex(ValueError,'guard'):intake.audit_json(p,'TW',sites,['S016','S020'])
    def test_nonfinite_coordinates_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p,sites=self.write_fixture(td);d=json.loads(p.read_text());d['sites']['S016']['labels'][0]['world'][0][0]=float('nan');p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'finite'):intake.audit_json(p,'TW',sites,['S016','S020'])
    def test_misaligned_pixel_world_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p,sites=self.write_fixture(td);d=json.loads(p.read_text());d['sites']['S016']['labels'][0]['px']=[[0,0],[1,0],[1,1],[0,1]];p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'mapping'):intake.audit_json(p,'TW',sites,['S016','S020'])
    def test_real_original_ignores_only_empty_undefined(self):
        frame,rows,extras=intake.audit_json(ROOT/'data/labels/crown_labels_XZ_2026-08-13.json','XZ',intake.site_definitions(),['S016','S020'])
        self.assertEqual(len(frame),45);self.assertEqual(sum(frame['class']=='tree'),27)
    def test_full_synthetic_cli_archives_and_never_freezes(self):
        with tempfile.TemporaryDirectory() as td:
            p,_=self.write_fixture(td);out=Path(td)/'result'
            cmd=[sys.executable,str(ROOT/'scripts/31_receive_independent_labels.py'),'--tim-json',str(p),'--output',str(out),'--synthetic-test']
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            receipt=json.loads((out/'receipt.json').read_text())
            self.assertEqual(receipt['status'],'SYNTHETIC_TEST_ONLY');self.assertFalse(receipt['reference_frozen'])
            self.assertTrue(receipt['source_originals_byte_verified'])
            self.assertTrue((out/'visual_review/index.html').exists())
            self.assertIn('SYNTHETIC TEST ONLY',(out/'agreement.md').read_text())
            with (out/'adjudication_queue.csv').open() as f:self.assertTrue(all(row['decision']=='pending' for row in csv.DictReader(f)))
    def test_completed_zero_object_teacher_sites_are_retained(self):
        with tempfile.TemporaryDirectory() as td:
            p,_=self.write_fixture(td);d=json.loads(p.read_text())
            for rec in d['sites'].values():rec['labels']=[]
            p.write_text(json.dumps(d));out=Path(td)/'zero_result'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/31_receive_independent_labels.py'),'--tim-json',str(p),'--output',str(out),'--synthetic-test'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            metrics=json.loads((out/'metrics.json').read_text())
            self.assertEqual(metrics['n_b'],0);self.assertEqual(metrics['agreement_f1'],0)
            self.assertIsNone(metrics['median_iou']);self.assertIsNone(metrics['class_kappa'])
            self.assertEqual(len(metrics['per_site']),2)
    def test_synthetic_cannot_masquerade_as_teacher(self):
        with tempfile.TemporaryDirectory() as td:
            p,_=self.write_fixture(td);out=Path(td)/'rejected'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/31_receive_independent_labels.py'),'--tim-json',str(p),'--output',str(out)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Synthetic',json.loads((out/'receipt.json').read_text())['error'])
    def test_nonobject_teacher_is_rejected_with_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.json';p.write_text('[]');out=Path(td)/'rejected'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/31_receive_independent_labels.py'),'--tim-json',str(p),'--output',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertEqual(json.loads((out/'receipt.json').read_text())['status'],'validation_or_calculation_failed')


if __name__=='__main__':unittest.main()
