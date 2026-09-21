"""Known-answer synthetic checks; no real labels or detections created."""
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark.evaluation import aggregate, paired_site_bootstrap, score_site
import importlib.util
s=importlib.util.spec_from_file_location('density',Path(__file__).resolve().parents[1]/'scripts/13_prepare_density.py')
density=importlib.util.module_from_spec(s);s.loader.exec_module(density)


class KnownAnswerEvaluation(unittest.TestCase):
    def test_over_under_segmentation_geometry_diagnostics(self):
        split=score_site([('r',box(0,0,4,2))],[('p1',box(0,0,2,2)),('p2',box(2,0,4,2))])
        self.assertEqual(aggregate([split])['over_segmentation_rate'],1)
        self.assertEqual(aggregate([split])['under_segmentation_rate'],0)
        merged=score_site([('r1',box(0,0,2,2)),('r2',box(2,0,4,2))],[('p',box(0,0,4,2))])
        self.assertEqual(aggregate([merged])['under_segmentation_rate'],1)
        self.assertEqual(aggregate([merged])['over_segmentation_rate'],0)
        tiny=score_site([('r',box(0,0,2,2))],[('p1',box(0,0,2,2)),('p2',box(1.999,0,4,2))])
        self.assertEqual(aggregate([tiny])['over_segmentation_rate'],0)

    def test_duplicate_prediction_and_missing_tree(self):
        result=score_site([('r1',box(0,0,2,2)),('r2',box(20,0,22,2))],
                          [('p1',box(0,0,2,2)),('p2',box(0,0,2,2))])
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,1,1))
        self.assertEqual(result['matches'][0]['prediction_id'],'p1')
        self.assertEqual(aggregate([result])['f1'],.5)

    def test_known_iou_and_even_median(self):
        r=score_site([('r1',box(0,0,4,1)),('r2',box(10,0,19,1))],
                     [('p1',box(1,0,5,1)),('p2',box(11,0,20,1))])
        self.assertAlmostEqual(aggregate([r])['median_matched_iou'],.7)

    def test_distance_limit_inclusive_without_extra_iou_cutoff(self):
        ref=[('r',box(0,0,1,1))]
        self.assertEqual(score_site(ref,[('p',box(2,0,3,1))])['tp'],1)
        self.assertEqual(score_site(ref,[('p',box(2.00001,0,3.00001,1))])['tp'],0)

    def test_match_single_tree_before_ignore(self):
        r=score_site([('r',box(0,0,1,1))],
                     [('p',box(0,0,1,1)),('g',box(10,0,11,1)),('n',box(20,0,21,1))],
                     groups=[box(-1,-1,12,2)],ignore_rule='unmatched_centroid_in_group_or_uncertain')
        self.assertEqual((r['tp'],r['fp'],r['ignored']),(1,1,1))
        self.assertEqual(r['false_positive_ids'],['n'])

    def test_no_predictions_and_empty_site(self):
        r=aggregate([score_site([('r',box(0,0,1,1))],[])])
        self.assertEqual(r['recall'],0);self.assertEqual(r['f1'],0)
        self.assertIsNone(r['precision'])
        e=aggregate([score_site([],[])])
        self.assertIsNone(e['f1']);self.assertIsNone(e['recall'])
        self.assertEqual(aggregate([score_site([],[('p',box(0,0,1,1))])])['precision'],0)

    def test_literature_commission_omission_denominators(self):
        # 18 detections matched, 2 extra predictions, and 6 missed references.
        result = aggregate([{'tp': 18, 'fp': 2, 'fn': 6, 'ious': []}])
        self.assertAlmostEqual(result['commission_rate'], .1)
        self.assertAlmostEqual(result['omission_rate'], .25)
        self.assertAlmostEqual(result['precision'] + result['commission_rate'], 1)
        self.assertAlmostEqual(result['recall'] + result['omission_rate'], 1)
        # Empty denominators must stay undefined, including zero-prediction sites.
        no_predictions = aggregate([{'tp': 0, 'fp': 0, 'fn': 4, 'ious': []}])
        self.assertIsNone(no_predictions['commission_rate'])
        self.assertEqual(no_predictions['omission_rate'], 1)
        no_references = aggregate([{'tp': 0, 'fp': 4, 'fn': 0, 'ious': []}])
        self.assertEqual(no_references['commission_rate'], 1)
        self.assertIsNone(no_references['omission_rate'])

    def test_paired_bootstrap_sign_reproducibility_and_pairing(self):
        perfect=score_site([('r',box(0,0,1,1))],[('p',box(0,0,1,1))])
        missed=score_site([('r',box(0,0,1,1))],[])
        a={'a':perfect,'b':perfect};b={'a':missed,'b':missed}
        positive=paired_site_bootstrap(a,b)
        self.assertEqual(positive['f1_improvement']['lower'],1)
        self.assertEqual(positive['intervals']['commission_rate']['upper'],0)
        self.assertEqual(positive['intervals']['omission_rate']['upper'],0)
        self.assertTrue(positive['positive_improvement_ci_excludes_zero'])
        self.assertFalse(paired_site_bootstrap(b,a)['positive_improvement_ci_excludes_zero'])
        self.assertFalse(paired_site_bootstrap(a,a)['positive_improvement_ci_excludes_zero'])
        self.assertEqual(positive,paired_site_bootstrap(a,b))
        with self.assertRaises(ValueError):paired_site_bootstrap(a,{'a':perfect})

    def test_nested_density_exact_counts_and_records(self):
        selections=density.nested_indices(101,[1,.5,.25],42)
        self.assertEqual([len(selections[f]) for f in [1,.5,.25]],[101,50,25])
        self.assertTrue(set(selections[.25])<=set(selections[.5]))
        self.assertTrue(np.array_equal(selections[.5],density.nested_indices(101,[.5],42)[.5]))
        self.assertTrue(np.all(np.diff(selections[.5])>0))
        with self.assertRaises(ValueError):density.nested_indices(0,[.5],42)


if __name__=='__main__':unittest.main()
