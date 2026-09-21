"""Reject dangerous annotation proposals; validate split/remainder topology."""
import importlib.util
from pathlib import Path
import unittest
import copy
import json
import tempfile
from unittest.mock import patch
from shapely.geometry import box
from shapely.ops import unary_union

spec=importlib.util.spec_from_file_location('refinement',Path(__file__).resolve().parents[1]/'scripts/26_refine_tree_groups.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class RefinementSafety(unittest.TestCase):
    def setUp(self):
        self.old=r.OBJECTS.copy();self.old_sites=r.SITES.copy()
        r.SITES['fixture']={'bbox_epsg2193':[0,0,100,100]}
        r.OBJECTS['fixture-parent']=('fixture',{'cls':'tree_group','world':[(10,10),(30,10),(30,30),(10,30)]})
    def tearDown(self):
        r.OBJECTS.clear();r.OBJECTS.update(self.old);r.SITES.clear();r.SITES.update(self.old_sites)
    def proposal(self,geoms):
        return {'action':'split','reason':'Synthetic test for safe geometry processing','crowns':[r.world_px('fixture',p) for p in geoms]}
    def test_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError,'overlap'):
            r.validate('fixture-parent',self.proposal([box(10,10,22,22),box(18,18,30,30)]))
    def test_outside_parent_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside'):
            r.validate('fixture-parent',self.proposal([box(20,20,40,40)]))
    def test_partial_extraction_retains_remainder_with_hole(self):
        children=r.validate('fixture-parent',self.proposal([box(14,14,22,22)]))
        parent=r.parent_geom('fixture-parent');remainder=parent.difference(unary_union(children))
        self.assertEqual(len(remainder.interiors),1)
        self.assertAlmostEqual(remainder.area+sum(p.area for p in children),400)
    def test_self_crossing_crown_rejected(self):
        d={'action':'split','reason':'Synthetic test for invalid self crossing polygon','crowns':[[[50,300],[90,340],[50,340],[90,300]]]}
        with self.assertRaisesRegex(ValueError,'valid'):
            r.validate('fixture-parent',d)
    def test_acceptance_does_not_transfer_to_an_edited_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);source=out/'source.json';source.write_text('{}')
            ds={'fixture-parent':{'action':'single','reason':'Known synthetic reviewed interpretation'}}
            (out/'author_acceptance.json').write_text(json.dumps({'source_sha256':r.sha(source),'accepted_decisions':ds}))
            with patch.object(r,'OUT',out),patch.object(r,'SOURCE',source):
                self.assertIn('fixture-parent',r.accepted_decisions(ds))
                changed=copy.deepcopy(ds);changed['fixture-parent']['reason']+=' Changed interpretation.'
                self.assertNotIn('fixture-parent',r.accepted_decisions(changed))

if __name__=='__main__':unittest.main()
