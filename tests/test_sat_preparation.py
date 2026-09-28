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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark.common import sha256,site_definitions

def module(name,file):
    s=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
audit=module('sat_audit','14_audit_sat_output.py')
reconcile=module('sat_reconcile','33_reconcile_sat_output.py')
crowns=module('sat_crowns','20_sat_crown_candidates.py')
conversion_grid=module('sat_conversion_grid','35_compare_crown_conversion_candidates.py')
labels=module('point_labels','21_validate_point_labels.py')

def call(m,args):
    with patch.object(sys,'argv',['synthetic_check',*map(str,args)]),contextlib.redirect_stdout(io.StringIO()):m.main()


class SATSyntheticPreparation(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'outputs/preparation/synthetic_workflow_checks';parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent);self.root=Path(self.temp.name)
        self.site=next(s for s in site_definitions().values() if s['split']=='train')

    def tearDown(self):self.temp.cleanup()

    def test_conversion_grid_configuration_validation(self):
        self.assertEqual(conversion_grid.parse_config(['convex_b050','convex_hull','none','.5'])['buffer_m'],.5)
        self.assertEqual(conversion_grid.parse_config(['concave_r050','concave_hull','.5','0'])['ratio'],.5)
        with self.assertRaisesRegex(ValueError,'ratio must be none'):
            conversion_grid.parse_config(['bad','convex_hull','.5','0'])
        with self.assertRaisesRegex(ValueError,'requires a ratio'):
            conversion_grid.parse_config(['bad','concave_hull','none','0'])
        with self.assertRaisesRegex(ValueError,'requires a ratio'):
            conversion_grid.parse_config(['bad','concave_hull','nan','0'])
        with self.assertRaisesRegex(ValueError,'config ID'):
            conversion_grid.parse_config(['../bad','convex_hull','none','0'])

    def test_restored_coordinate_audit_and_draft_hull(self):
        x0,y0,_,_=self.site['bbox_epsg2193']
        header=laspy.LasHeader(point_format=6,version='1.4');header.add_crs(CRS.from_epsg(2193));header.offsets=[x0,y0,0];header.scales=[.001]*3
        source=laspy.LasData(header);source.x=x0+np.array([10,12,12,10,20]);source.y=y0+np.array([10,10,12,12,20]);source.z=np.array([5]*5)
        src=self.root/'synthetic_input.las';source.write(src)
        pred=laspy.LasData(header.copy());pred.points=source.points.copy()
        pred.add_extra_dim(laspy.ExtraBytesParams(name='PredInstance',type='uint16'))
        pred.add_extra_dim(laspy.ExtraBytesParams(name='PredSemantic',type='uint8'))
        pred['PredInstance']=np.array([1,1,1,1,2]);pred['PredSemantic']=np.array([1]*5)
        output=self.root/'synthetic_prediction.las';pred.write(output)
        report=self.root/'audit.json';call(audit,['--input',src,'--prediction',output,'--output',report])
        result=json.loads(report.read_text());self.assertTrue(result['epsg2193_coordinate_consistency'])
        converted=self.root/'hulls';call(crowns,['--prediction',output,'--coordinate-audit',report,'--tree-semantic-values','1','--output',converted])
        frame=gpd.read_file(converted/'candidate_crowns.gpkg',layer='candidate_crowns')
        self.assertEqual(len(frame),1);self.assertAlmostEqual(frame.geometry.area.iloc[0],4)
        meta=json.loads((converted/'provenance.json').read_text());self.assertFalse(meta['conversion_approved']);self.assertEqual(meta['skipped_instances'][0]['instance_id'],2)
        self.assertEqual(meta['hull_method'],'convex_hull');self.assertEqual(meta['buffer_m'],0)
        merge_receipt=self.root/'merge_receipt.json'
        merge_receipt.write_text(json.dumps({'status':'TRAINING-SITE MULTI-TILE MERGE COMPLETED',
            'split':'train','output_laz_sha256':sha256(output),'benchmark_scores_created':False}))
        merged_converted=self.root/'merged_hulls';call(crowns,['--prediction',output,'--merge-receipt',merge_receipt,'--tree-semantic-values','1','--output',merged_converted])
        merged_meta=json.loads((merged_converted/'provenance.json').read_text())
        self.assertEqual(merged_meta['verification_record_type'],'training_site_merge_receipt')
        self.assertEqual(merged_meta['merge_receipt_sha256'],sha256(merge_receipt))
        bad_receipt=self.root/'bad_merge_receipt.json'
        bad_receipt.write_text(json.dumps({'status':'TRAINING-SITE MULTI-TILE MERGE COMPLETED',
            'split':'train','output_laz_sha256':'0'*64,'benchmark_scores_created':False}))
        with self.assertRaisesRegex(ValueError,'matching completed'):
            call(crowns,['--prediction',output,'--merge-receipt',bad_receipt,'--tree-semantic-values','1','--output',self.root/'bad_merged_hulls'])
        buffered=self.root/'buffered_hulls';call(crowns,['--prediction',output,'--coordinate-audit',report,'--tree-semantic-values','1','--hull-method','concave_hull','--concave-ratio','.5','--buffer-m','.25','--output',buffered])
        buffered_frame=gpd.read_file(buffered/'candidate_crowns.gpkg',layer='candidate_crowns')
        self.assertEqual(len(buffered_frame),1);self.assertGreater(buffered_frame.geometry.area.iloc[0],4)
        buffered_meta=json.loads((buffered/'provenance.json').read_text())
        self.assertEqual(buffered_meta['hull_method'],'concave_hull');self.assertEqual(buffered_meta['concave_ratio'],.5);self.assertEqual(buffered_meta['buffer_m'],.25)
        pred['PredSemantic']=np.array([0]*5);background=self.root/'synthetic_no_trees.las';pred.write(background)
        bg_audit=self.root/'bg_audit.json';call(audit,['--input',src,'--prediction',background,'--output',bg_audit])
        bg_crowns=self.root/'no_crowns';call(crowns,['--prediction',background,'--coordinate-audit',bg_audit,'--tree-semantic-values','1','--output',bg_crowns])
        self.assertEqual(json.loads((bg_crowns/'provenance.json').read_text())['crowns'],0)
        pred.x=np.asarray(pred.x)+1000;shifted=self.root/'synthetic_wrong_origin.las';pred.write(shifted)
        with self.assertRaises(SystemExit):call(audit,['--input',src,'--prediction',shifted,'--output',self.root/'bad_audit.json'])

    def test_point_label_format_and_manual_qa_gate(self):
        upstream_reader = ROOT.parent/'SegmentAnyTree/torch_points3d/modules/KPConv/plyutils.py'
        if not upstream_reader.is_file():
            self.skipTest('upstream SegmentAnyTree checkout is not installed beside the repository')
        ply=self.root/'synthetic_checked_labels.ply'
        header='ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nproperty int semantic_seg\nproperty int treeID\nend_header\n'
        data=np.zeros(4,dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('semantic_seg','<i4'),('treeID','<i4')])
        data['x']=[1,2,3,4];data['y']=[1,2,3,4];data['semantic_seg']=[2,2,1,0];data['treeID']=[0,0,-1,-1]
        ply.write_bytes(header.encode()+data.tobytes())
        x0,y0,_,_=self.site['bbox_epsg2193'];qa=self.root/'synthetic_qa.json'
        record={'ply_sha256':sha256(ply),'point_labels_manually_checked':True,'reviewer':'SYNTHETIC FIXTURE ONLY','site_id':self.site['site_id'],'coordinate_origin_xyz_m':[x0,y0,0]}
        qa.write_text(json.dumps(record));report=self.root/'labels.json'
        call(labels,['--ply',ply,'--qa-record',qa,'--output',report])
        self.assertEqual(json.loads(report.read_text())['tree_instances'],1)
        record['point_labels_manually_checked']=False;qa.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'manual point-level'):call(labels,['--ply',ply,'--qa-record',qa,'--output',self.root/'rejected.json'])

    def test_reconcile_duplicate_cartesian_rows_and_restore_crs(self):
        x0,y0,_,_=self.site['bbox_epsg2193']
        source_header=laspy.LasHeader(point_format=6,version='1.4')
        source_header.add_crs(CRS.from_epsg(2193));source_header.offsets=[x0,y0,0];source_header.scales=[.001]*3
        source=laspy.LasData(source_header)
        source.x=x0+np.array([1,1,2,3]);source.y=y0+np.array([1,1,2,3]);source.z=np.array([4,4,5,6])
        src=self.root/'duplicate_source.las';source.write(src)

        pred_header=laspy.LasHeader(point_format=6,version='1.4')
        pred_header.offsets=[x0+.001,y0+.001,.001];pred_header.scales=[.001]*3
        pred=laspy.LasData(pred_header)
        pred.x=x0+np.array([1,1,1,1,2,3]);pred.y=y0+np.array([1,1,1,1,2,3]);pred.z=np.array([4,4,4,4,5,6])
        pred.add_extra_dim(laspy.ExtraBytesParams(name='PredInstance',type='uint16'))
        pred.add_extra_dim(laspy.ExtraBytesParams(name='PredSemantic',type='uint8'))
        pred['PredInstance']=np.array([7,7,7,7,8,0]);pred['PredSemantic']=np.array([1,1,1,1,1,0])
        upstream=self.root/'expanded_prediction.las';pred.write(upstream)

        fixed=self.root/'reconciled.las';report=self.root/'reconcile.json'
        call(reconcile,['--input',src,'--prediction',upstream,'--output',fixed,'--report',report])
        result=laspy.read(fixed);meta=json.loads(report.read_text())
        self.assertEqual(len(result.points),len(source.points))
        self.assertEqual(result.header.parse_crs().to_epsg(),2193)
        self.assertEqual(list(result['PredInstance']),[7,7,8,0])
        self.assertEqual(meta['removed_cartesian_join_rows'],2)
        audit_path=self.root/'reconciled_audit.json'
        call(audit,['--input',src,'--prediction',fixed,'--output',audit_path])
        self.assertTrue(json.loads(audit_path.read_text())['epsg2193_coordinate_consistency'])


if __name__=='__main__':unittest.main()
