import importlib.util
import sys
from pathlib import Path
import unittest
from shapely.geometry import box,Polygon
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

spec=importlib.util.spec_from_file_location('group_coverage',Path(__file__).resolve().parents[1]/'scripts/30_diagnose_group_coverage.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class GroupCoverageCases(unittest.TestCase):
    def test_empty_prediction_has_zero_coverage(self):
        r=module.group_coverage(box(0,0,10,10),[])
        self.assertEqual(r['coverage_fraction'],0)
        self.assertEqual(r['linked_union_iou'],0)
    def test_two_halves_and_one_merged_crown_same_area_coverage(self):
        g=box(0,0,10,10)
        split=module.group_coverage(g,[box(0,0,5,10),box(5,0,10,10)])
        merged=module.group_coverage(g,[g])
        self.assertEqual(split['coverage_fraction'],merged['coverage_fraction'])
        self.assertEqual(split['coverage_fraction'],1)
        self.assertEqual(split['linked_prediction_count'],2)
        self.assertEqual(merged['linked_prediction_count'],1)
    def test_overlapping_predictions_not_double_counted(self):
        r=module.group_coverage(box(0,0,10,10),[box(0,0,6,10),box(4,0,8,10)])
        self.assertEqual(r['covered_area_m2'],80)
    def test_spill_reduces_iou_not_coverage(self):
        r=module.group_coverage(box(0,0,10,10),[box(-5,0,10,10),box(20,0,30,10)])
        self.assertEqual(r['coverage_fraction'],1)
        self.assertAlmostEqual(r['linked_union_iou'],2/3)
        self.assertEqual(r['linked_prediction_count'],1)
    def test_invalid_polygon_rejected(self):
        with self.assertRaises(ValueError):module.group_coverage(Polygon([(0,0),(10,10),(0,10),(10,0)]),[])

if __name__=='__main__':unittest.main()
