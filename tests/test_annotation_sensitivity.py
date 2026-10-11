import importlib.util
import sys
import unittest
from pathlib import Path

from shapely.geometry import GeometryCollection, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('sensitivity', ROOT / 'scripts/81_analyse_annotation_sensitivity.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class SensitivityCases(unittest.TestCase):
    def test_containment_correspondence_does_not_assert_good_delineation(self):
        a = [('large', box(0, 0, 10, 10))]
        b = [('small', box(1, 1, 4, 4))]
        self.assertEqual(s.overlap_matches(a, b, .5), [])
        pairs = s.overlap_matches(a, b, .5, smaller=True)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['overlap_value'], 1)
        self.assertAlmostEqual(pairs[0]['iou'], .09)

    def test_split_regions_have_perfect_extent_but_only_one_object_match(self):
        a = [('group', box(0, 0, 10, 10))]
        b = [('left', box(0, 0, 5, 10)), ('right', box(5, 0, 10, 10))]
        self.assertEqual(len(s.overlap_matches(a, b, .5, smaller=True)), 1)
        edges, da, db = s.region_links(a, b, .5)
        self.assertEqual(len(edges), 2); self.assertEqual(da['group'], 2)
        self.assertEqual(sum(v > 0 for v in db.values()), 2)
        score = s.tolerant_score(s.tolerant_area(a[0][1], b[0][1].union(b[1][1]), box(-1, -1, 11, 11), 0))
        self.assertEqual(score['tolerant_area_f'], 1)

    def test_tolerance_has_known_displacement_and_empty_denominators(self):
        a, b, roi = box(0, 0, 10, 10), box(2, 0, 12, 10), box(-5, -5, 20, 20)
        zero = s.tolerant_score(s.tolerant_area(a, b, roi, 0))
        relaxed = s.tolerant_score(s.tolerant_area(a, b, roi, 2))
        self.assertAlmostEqual(zero['tolerant_area_f'], .8)
        self.assertAlmostEqual(relaxed['tolerant_area_f'], 1)
        empty = GeometryCollection()
        self.assertIsNone(s.tolerant_score(s.tolerant_area(empty, empty, roi, 1))['tolerant_area_f'])
        self.assertEqual(s.tolerant_score(s.tolerant_area(a, empty, roi, 1))['tolerant_area_f'], 0)

    def test_touching_and_hole_interiors_never_create_links(self):
        a = [('a', box(0, 0, 10, 10).difference(box(2, 2, 8, 8)))]
        b = [('in_hole', box(3, 3, 7, 7)), ('touching', box(10, 0, 12, 10))]
        self.assertEqual(s.overlap_matches(a, b, .5, smaller=True), [])
        edges, _, _ = s.region_links(a, b, .5)
        self.assertEqual(edges, [])


if __name__ == '__main__':
    unittest.main()
