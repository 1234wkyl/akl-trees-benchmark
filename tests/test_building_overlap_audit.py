import importlib.util
import sys
import unittest
from pathlib import Path
from shapely.geometry import box, GeometryCollection

p = Path(__file__).resolve().parents[1] / 'scripts/70_audit_training_building_overlap.py'
sys.path.insert(0, str(p.parent))
s = importlib.util.spec_from_file_location('buildings70', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


class BuildingOverlap(unittest.TestCase):
    def test_overhang_is_not_centroid_inside(self):
        r = m.overlap_metrics(box(0, 0, 10, 10), box(8, 0, 12, 10))
        self.assertEqual(r['building_overlap_m2'], 20)
        self.assertEqual(r['building_overlap_fraction'], .2)
        self.assertFalse(r['centroid_in_building'])

    def test_inside_and_empty_mask(self):
        crown = box(0, 0, 2, 2)
        inside = m.overlap_metrics(crown, box(-1, -1, 3, 3))
        self.assertEqual(inside['building_overlap_fraction'], 1)
        self.assertTrue(inside['centroid_in_building'])
        empty = m.overlap_metrics(crown, GeometryCollection())
        self.assertEqual(empty['building_overlap_fraction'], 0)
        self.assertFalse(empty['centroid_in_building'])


if __name__ == '__main__':
    unittest.main()
