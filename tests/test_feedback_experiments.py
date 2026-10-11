import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


b = module('building71', '71_test_training_building_filter.py')
w = module('weak72', '72_screen_weak_point_labels.py')


class FeedbackExperiments(unittest.TestCase):
    def test_filter_is_spatial_even_if_prediction_is_matched(self):
        # The implementation must not read truth status to spare a matched tree.
        rows = [{'prediction_id': 'a', 'status': 'matched', 'centroid_in_building': True},
                {'prediction_id': 'b', 'status': 'false_positive', 'centroid_in_building': False}]
        self.assertEqual(b.removal_ids(rows), {'a'})

    def test_partial_and_multi_group_crowns_rejected(self):
        p = {'source_geometry_extends_outside_group_union': True, 'parent_group_ids': ['g1', 'g2']}
        self.assertEqual(w.screen_reason(100, True, p),
                         ['source_crown_only_partly_labelled_by_group', 'multiple_parent_groups'])
        self.assertEqual(w.screen_reason(19, True), ['insufficient_point_support'])

    def test_removed_tree_is_unknown_and_background_preserved(self):
        sem = np.array([2, 2, 1, 0]); ids = np.array([4, 5, -1, -1])
        s, i, removed = w.retain_labels(sem, ids, {4})
        np.testing.assert_array_equal(s, [2, 0, 1, 0])
        np.testing.assert_array_equal(i, [4, -1, -1, -1])
        self.assertEqual(int(removed.sum()), 1)
        np.testing.assert_array_equal(sem, [2, 2, 1, 0])

    def test_crop_checks_whole_crown_and_unknown_points(self):
        xy = np.array([[1., 1.], [1.5, 1.5], [5., 5.]])
        sem = np.array([2, 0, 1]); ids = np.array([4, -1, -1])
        selected, present, truncated, unknown = w.crop_checks(
            xy, sem, ids, box(0, 0, 2, 2), {4: box(0, 0, 3, 3)})
        self.assertEqual(present, [4]); self.assertEqual(truncated, [4]); self.assertEqual(unknown, 1)
        np.testing.assert_array_equal(selected, [True, True, False])


if __name__ == '__main__':
    unittest.main()
