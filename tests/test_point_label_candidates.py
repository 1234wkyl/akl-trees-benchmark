import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
from shapely.geometry import box
p=Path(__file__).resolve().parents[1]/'scripts/53_prepare_point_label_candidates.py'
sys.path.insert(0,str(p.parent))
spec=importlib.util.spec_from_file_location('point_candidates',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Candidates(unittest.TestCase):
    def test_groups_and_overlapping_crowns_stay_unknown(self):
        x=np.array([1.,2.5,4.,6.]);y=np.ones(4)
        sem,ids,bad=m.assign_candidates(x,y,np.array([5,5,5,5]),
            [box(0,0,3,2),box(2,0,4.5,2)],[box(5,0,7,2)],[])
        self.assertEqual(sem.tolist(),[2,0,2,0]);self.assertEqual(ids.tolist(),[0,-1,1,-1])
    def test_ground_and_hard_negatives_do_not_extrude_tree_columns(self):
        x=np.array([1.,1.,1.,4.,4.]);y=np.ones(5)
        sem,ids,_=m.assign_candidates(x,y,np.array([2,5,1,6,5]),[box(0,0,2,2)],[],[box(3,0,5,2)])
        self.assertEqual(sem.tolist(),[1,2,0,1,0]);self.assertEqual(ids.tolist(),[-1,0,-1,-1,-1])
