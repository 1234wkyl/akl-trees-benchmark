import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark.weak_pointgroup_loss import known_embedding_loss


class WeakEmbeddingTest(unittest.TestCase):
    def test_unknown_embedding_targets_have_no_gradient(self):
        x=torch.tensor([[0.,0.],[2.,0.],[4.,0.],[6.,0.],[100.,100.]],requires_grad=True)
        r=known_embedding_loss(x,torch.tensor([1,1,2,2,0]),torch.tensor([1,1,1,1,-1]),torch.zeros(5).long())
        r['ins_loss'].backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad[-1]==0).all())
        self.assertGreater(float(r['ins_var_loss'].detach()),0.)

    def test_instance_identity_is_batch_local_and_empty_targets_are_finite(self):
        x=torch.tensor([[0.],[0.],[100.],[100.]],requires_grad=True)
        r=known_embedding_loss(x,torch.ones(4).long(),torch.ones(4).long(),torch.tensor([0,0,1,1]))
        self.assertEqual(float(r['ins_var_loss'].detach()),0.)
        self.assertEqual(float(r['ins_dist_loss'].detach()),0.)
        empty=known_embedding_loss(x,torch.zeros(4).long(),torch.full((4,),-1),torch.zeros(4).long())
        empty['ins_loss'].backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad==0).all())


if __name__=='__main__':
    unittest.main()
