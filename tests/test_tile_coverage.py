import importlib.util
import sys
import unittest
from pathlib import Path
from shapely.geometry import box

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
spec=importlib.util.spec_from_file_location('coverage',Path(__file__).resolve().parents[1]/'scripts/32_prepare_tile_coverage_plan.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class TileCoverageCases(unittest.TestCase):
    def test_overlap_not_double_counted(self):
        self.assertEqual(c.union_coverage(box(0,0,100,100),[box(0,0,75,100),box(25,0,100,100)]),1)
    def test_gap_and_partial_tile_reported(self):
        self.assertEqual(c.union_coverage(box(0,0,100,100),[box(0,0,30,100),box(50,0,100,100)]),.8)
        self.assertEqual(c.union_coverage(box(0,0,100,100),[]),0)

if __name__=='__main__':unittest.main()
