import importlib.util
import unittest
import sys
from pathlib import Path

p=Path(__file__).resolve().parents[1]/'scripts/52_reconstruct_detection_and_crowns.py'
sys.path.insert(0,str(p.parent))
spec=importlib.util.spec_from_file_location('detection_reconstruction',p)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class CoordinateComparison(unittest.TestCase):
    def test_dense_neighbors_maximize_cardinality_without_reusing_points(self):
        pairs=module.coordinate_pairs([[0,0],[.15,0]],[[.1,0],[-.11,0]],.2)
        self.assertEqual({(i,j) for i,j,d in pairs},{(0,1),(1,0)})
    def test_exact_match_does_not_turn_near_or_duplicate_points_into_identity(self):
        self.assertEqual(len(module.coordinate_pairs([[0,0],[0,0]],[[0,0]],1e-6)),1)
        self.assertEqual(module.coordinate_pairs([[0,0]],[[.001,0]],1e-6),[])
        self.assertEqual(module.coordinate_pairs([],[[0,0]],2),[])
