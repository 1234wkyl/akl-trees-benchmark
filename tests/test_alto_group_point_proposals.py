"""Known-answer tests for ALTO candidate membership and split protection."""
import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
from shapely.geometry import box

path = Path(__file__).resolve().parents[1] / 'scripts/69_prepare_alto_group_point_proposals.py'
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location('alto69', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class AltoGroups(unittest.TestCase):
    def test_distinct_cores_preserve_existing_labels_and_xyz_order(self):
        x = np.array([1., 5., 8., 10.]); y = np.ones(4)
        sem, ids, selected, _ = m.assign_group_proposals(x, y, np.full(4, 5),
            np.array([0, 0, 2, 0]), np.array([-1, -1, 3, -1]),
            [box(0, 0, 7, 2)], [], [('a', box(0, 0, 3, 2)), ('b', box(4, 0, 7, 2))], 4, .25)
        self.assertEqual(sem.tolist(), [2, 2, 2, 0])
        self.assertEqual(ids.tolist(), [4, 5, 3, -1])
        self.assertEqual(selected.tolist(), [True, True, False, False])

    def test_overlap_boundary_ground_and_exclusions_abstain(self):
        x = np.array([.1, 1., 2.5, 4., 5., 6.]); y = np.ones(6)
        sem, ids, _, _ = m.assign_group_proposals(x, y, np.array([5, 2, 5, 5, 5, 5]),
            np.zeros(6, dtype='i4'), np.full(6, -1, dtype='i4'),
            [box(0, 0, 7, 2), box(5.8, 0, 6.2, 2)], [box(4.8, 0, 5.2, 2)],
            [('a', box(0, 0, 3, 2)), ('b', box(2, 0, 7, 2))], 0, .25)
        self.assertEqual(sem.tolist(), [0, 0, 0, 2, 0, 0])
        self.assertEqual(ids.tolist(), [-1, -1, -1, 1, -1, -1])

    def test_heldout_and_duplicate_sites_rejected(self):
        definitions = {'S001': {'split': 'train'}, 'S002': {'split': 'test'}}
        self.assertEqual(m.training_sites(None, definitions), ['S001'])
        for ids in [['S002'], ['S001', 'S001']]:
            with self.assertRaises(ValueError):
                m.training_sites(ids, definitions)


if __name__ == '__main__':
    unittest.main()
