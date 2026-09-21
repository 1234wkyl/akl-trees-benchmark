"""Known-point check for semantic versus instance support diagnostics."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
SCRIPT = ROOT/'scripts/40_audit_training_reference_points.py'
SPEC = importlib.util.spec_from_file_location('training_reference_points', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TrainingReferencePointTests(unittest.TestCase):
    def test_polygon_inside_counts_separate_semantic_and_instance(self):
        geometry = box(0, 0, 2, 2)
        x = np.array([-1.0, 0.5, 1.5, 2.0, 3.0])
        y = np.ones(5)
        semantic = np.array([1, 1, 0, 1, 1])
        instance = np.array([8, 0, 5, 7, 0])
        result = MODULE.support(geometry, x, y, semantic, instance, (1,))
        self.assertEqual(result, {
            'points': 2, 'tree_semantic_points': 1,
            'positive_instance_points': 1, 'positive_instance_ids': '5',
        })

    def test_tree_semantic_can_exist_without_positive_instance(self):
        geometry = box(0, 0, 2, 2)
        result = MODULE.support(geometry,
                                np.array([0.5, 1.5]), np.array([1.0, 1.0]),
                                np.array([1, 1]), np.array([0, 0]), (1,))
        self.assertEqual(result['tree_semantic_points'], 2)
        self.assertEqual(result['positive_instance_points'], 0)
        self.assertEqual(result['positive_instance_ids'], '')


if __name__ == '__main__':
    unittest.main()
