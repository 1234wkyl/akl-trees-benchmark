import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location(
    'merge_training_site_tiles', ROOT / 'scripts/46_merge_training_site_tiles.py')
MERGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGER)


def tile(xyz, instance, semantic, margin):
    return {
        'xyz': np.asarray(xyz, dtype=float),
        'instance': np.asarray(instance, dtype=np.int64),
        'semantic': np.asarray(semantic, dtype=np.int64),
        'margin': np.asarray(margin, dtype=float),
    }


class MultiTileMerge(unittest.TestCase):
    def test_exact_overlap_links_instances_but_local_id_collisions_remain_distinct(self):
        first = tile(
            [[0, 0, 1], [1, 0, 1], [3, 0, 1]],
            [1, 1, 2], [2, 2, 2], [1, .1, 1])
        second = tile(
            [[1, 0, 1], [2, 0, 1], [4, 0, 1]],
            [7, 7, 1], [2, 2, 2], [2, 1, 1])

        merged = MERGER.merge_tiles([first, second])
        by_x = {float(x): int(i) for x, i in zip(merged['xyz'][:, 0], merged['instance'])}

        self.assertEqual(merged['output_unique_xyz'], 5)
        self.assertEqual(merged['duplicate_records_removed'], 1)
        self.assertEqual(merged['cross_tile_union_links'], 1)
        self.assertEqual(merged['global_positive_instances'], 3)
        self.assertEqual(by_x[0], by_x[1])
        self.assertEqual(by_x[1], by_x[2])
        self.assertNotEqual(by_x[3], by_x[0])
        self.assertNotEqual(by_x[4], by_x[0])

    def test_semantic_conflict_is_deduplicated_without_linking_instances(self):
        first = tile(
            [[0, 0, 1], [1, 0, 1]], [1, 1], [2, 2], [2, 2])
        second = tile(
            [[0, 0, 1], [2, 0, 1]], [7, 7], [3, 3], [1, 1])

        merged = MERGER.merge_tiles([first, second])
        by_x = {float(x): (int(i), int(s)) for x, i, s in
                zip(merged['xyz'][:, 0], merged['instance'], merged['semantic'])}

        self.assertEqual(merged['semantic_conflict_groups'], 1)
        self.assertEqual(merged['cross_tile_union_links'], 0)
        self.assertEqual(merged['global_positive_instances'], 2)
        self.assertEqual(by_x[0][1], 2)  # stable interior-margin rule selected tile 0
        self.assertNotEqual(by_x[1][0], by_x[2][0])


if __name__ == '__main__':
    unittest.main()
