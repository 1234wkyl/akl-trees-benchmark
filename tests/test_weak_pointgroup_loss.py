import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark.weak_pointgroup_loss import known_instance_ious, weak_proposal_losses, compute_weak_pointgroup_loss


class WeakLoss(unittest.TestCase):
    def test_unknown_points_do_not_reduce_proposal_iou(self):
        y = torch.tensor([1, 1, -1, -1, 0]); ids = torch.tensor([1, 1, 0, 0, 0])
        ious, valid, pairs = known_instance_ious([torch.tensor([0, 1, 2, 3])], ids, y, torch.zeros(5).long())
        self.assertAlmostEqual(float(ious[0, 0]), 1., places=4)
        self.assertTrue(bool(valid[0]))

    def test_all_unknown_proposal_has_no_score_or_mask_gradient(self):
        y = torch.tensor([1, -1, -1]); ids = torch.tensor([1, 0, 0])
        scores = torch.tensor([.6], requires_grad=True)
        masks = torch.tensor([.3, .8], requires_grad=True)
        r = weak_proposal_losses([torch.tensor([1, 2])], scores, ids, y,
                                torch.zeros(3).long(), masks)
        (r['score_loss'] + r['mask_loss']).backward()
        self.assertEqual(float(r['score_loss']), 0)
        self.assertEqual(float(r['mask_loss']), 0)
        self.assertTrue(torch.equal(scores.grad, torch.zeros_like(scores)))
        self.assertTrue(torch.equal(masks.grad, torch.zeros_like(masks)))

    def test_unknown_mask_has_zero_gradient_but_known_background_is_negative(self):
        y = torch.tensor([1, 1, -1, 0]); ids = torch.tensor([1, 1, 0, 0])
        scores = torch.tensor([.6], requires_grad=True)
        masks = torch.tensor([.8, .8, .8, .8], requires_grad=True)
        r = weak_proposal_losses([torch.arange(4)], scores, ids, y, torch.zeros(4).long(), masks)
        r['mask_loss'].backward()
        self.assertEqual(float(masks.grad[2]), 0)
        self.assertGreater(float(masks.grad[3]), 0)
        self.assertLess(float(masks.grad[0]), 0)

    def test_batch_local_ids_are_distinct(self):
        y = torch.ones(4).long(); ids = torch.ones(4).long(); batch = torch.tensor([0, 0, 1, 1])
        ious, valid, pairs = known_instance_ious([torch.tensor([0, 1]), torch.tensor([2, 3])], ids, y, batch)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(float(ious[0, 1]), 0); self.assertEqual(float(ious[1, 0]), 0)
        with self.assertRaises(ValueError):
            known_instance_ious([torch.tensor([0, 2])], ids, y, batch)

    def test_no_gt_and_empty_proposals_remain_finite(self):
        y = torch.tensor([0, -1]); ids = torch.zeros(2).long(); batch = torch.zeros(2).long()
        scores = torch.tensor([.7], requires_grad=True)
        r = weak_proposal_losses([torch.tensor([0, 1])], scores, ids, y, batch)
        self.assertAlmostEqual(float(r['score_loss']), float(F.binary_cross_entropy(scores, torch.zeros(1))), places=6)
        r['score_loss'].backward(); self.assertTrue(torch.isfinite(scores.grad).all())

    def test_adapter_backward_all_heads_ignores_unknown_semantics_and_offsets(self):
        logits = torch.randn(4, 2, requires_grad=True)
        offsets = torch.randn(4, 3, requires_grad=True)
        scores = torch.tensor([.6], requires_grad=True)
        masks = torch.randn(4, 1, requires_grad=True)
        y = torch.tensor([1, 1, -1, 0]); ids = torch.tensor([1, 1, 0, 0])
        model = SimpleNamespace(labels=SimpleNamespace(y=y),
            input=SimpleNamespace(instance_mask=ids>0, instance_labels=ids, batch=torch.zeros(4).long(),
                                  vote_label=torch.ones(4, 3)),
            output=SimpleNamespace(semantic_logits=F.log_softmax(logits, dim=1), offset_logits=offsets,
                cluster_scores=scores, mask_scores=masks, clusters=[torch.arange(4)]),
            opt=SimpleNamespace(loss_weights={k: 1. for k in ['semantic', 'offset_norm_loss', 'offset_dir_loss',
                                                             'score_loss', 'mask_loss']},
                                min_iou_threshold=.25, max_iou_threshold=.75),
            _scorer_type='MLP', mask_supervise=True, cal_iou_based_on_mask=False,
            cal_iou_based_on_mask_start_epoch=0)
        compute_weak_pointgroup_loss(model, 1)
        self.assertTrue(torch.isfinite(model.loss)); model.loss.backward()
        self.assertTrue(torch.equal(logits.grad[2], torch.zeros(2)))
        self.assertTrue(torch.equal(offsets.grad[2], torch.zeros(3)))
        self.assertEqual(float(masks.grad[2]), 0)
        self.assertTrue(torch.isfinite(scores.grad).all())


if __name__ == '__main__':
    unittest.main()
