import json
import sys
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark.annotation_coverage import coverage_agreement


class AnnotationCoverageCases(unittest.TestCase):
    def frame(self,objects):
        return gpd.GeoDataFrame([{'label_id':str(i),'class':cls,'geometry':geometry}
                                for i,(cls,geometry) in enumerate(objects)],
                               columns=['label_id','class','geometry'],geometry='geometry',crs=2193)

    def compare(self,a,b,bounds=(0,0,20,20)):
        policy=json.loads((ROOT/'config/independent_coverage.json').read_text())
        return coverage_agreement({'S':(self.frame(a),self.frame(b))},
                                  {'S':{'bbox_epsg2193':bounds}},policy)

    def test_three_trees_vs_one_group_same_extent(self):
        a=[('tree',box(i*4,0,(i+1)*4,4)) for i in range(3)]
        b=[('tree_group',box(0,0,12,4))]
        result=self.compare(a,b)
        self.assertEqual(result['pooled_area']['coverage_iou'],1)
        self.assertEqual(result['pooled_area']['coverage_dice'],1)
        self.assertEqual(result['per_site'][0]['vegetation_objects_a'],3)
        self.assertEqual(result['per_site'][0]['vegetation_objects_b'],1)
        self.assertTrue(all(row['boundary_f1']==1 for row in result['pooled_boundary']))
        self.assertFalse(result['automatic_pass_or_freeze'])

    def test_known_overlap_and_tolerance_does_not_change_area_scores(self):
        result=self.compare([('tree',box(0,0,4,4))],[('tree',box(1,0,5,4))])
        self.assertAlmostEqual(result['pooled_area']['coverage_iou'],.6)
        self.assertAlmostEqual(result['pooled_area']['coverage_dice'],.75)
        boundary=result['pooled_boundary']
        self.assertLess(boundary[0]['boundary_f1'],boundary[1]['boundary_f1'])
        self.assertEqual(boundary[1]['boundary_f1'],1)

    def test_missing_tree_is_not_hidden_by_merging(self):
        result=self.compare([('tree',box(0,0,4,4)),('tree',box(10,0,14,4))],
                            [('tree_group',box(0,0,4,4))])
        self.assertAlmostEqual(result['pooled_area']['coverage_iou'],.5)
        self.assertAlmostEqual(result['pooled_area']['coverage_dice'],2/3)

    def test_not_tree_and_uncertain_do_not_become_vegetation_or_ignore_masks(self):
        for cls in ['not_tree','uncertain']:
            result=self.compare([('tree',box(2,2,4,4))],[(cls,box(2,2,4,4))])
            self.assertEqual(result['pooled_area']['coverage_dice'],0)
            self.assertEqual(result['pooled_boundary'][0]['boundary_f1'],0)
        self.assertEqual(result['per_site'][0]['uncertain_area_b_m2'],4)

    def test_empty_both_sides_remains_undefined(self):
        result=self.compare([],[])
        self.assertIsNone(result['pooled_area']['coverage_iou'])
        self.assertIsNone(result['pooled_area']['coverage_dice'])
        self.assertIsNone(result['pooled_boundary'][0]['boundary_f1'])

    def test_site_clip_does_not_create_artificial_matched_boundaries(self):
        result=self.compare([('tree_group',box(-5,-5,25,25))],
                            [('tree_group',box(-6,-6,26,26))],bounds=(0,0,20,20))
        self.assertEqual(result['pooled_area']['coverage_iou'],1)
        self.assertIsNone(result['pooled_boundary'][0]['boundary_f1'])
        self.assertEqual(result['per_site'][0]['vegetation_outside_site_a_m2'],500)

    def test_pooled_scores_use_area_totals_not_average_site_scores(self):
        policy=json.loads((ROOT/'config/independent_coverage.json').read_text())
        frames={'small':(self.frame([('tree',box(0,0,1,1))]),self.frame([])),
                'large':(self.frame([('tree',box(0,0,10,10))]),self.frame([('tree_group',box(0,0,10,10))]))}
        definitions={sid:{'bbox_epsg2193':(0,0,20,20)} for sid in frames}
        result=coverage_agreement(frames,definitions,policy)
        self.assertAlmostEqual(result['pooled_area']['coverage_iou'],100/101)
        self.assertAlmostEqual(result['pooled_area']['coverage_dice'],200/201)

    def test_wrong_crs_rejected(self):
        policy=json.loads((ROOT/'config/independent_coverage.json').read_text())
        frame=self.frame([('tree',box(0,0,1,1))]).set_crs(4326,allow_override=True)
        with self.assertRaisesRegex(ValueError,'EPSG:2193'):
            coverage_agreement({'S':(frame,frame)},{'S':{'bbox_epsg2193':(0,0,20,20)}},policy)


if __name__=='__main__':unittest.main()
