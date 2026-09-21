"""Known-count checks for the training-only crown-conversion comparison."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/39_compare_training_site_grids.py'
COVERAGE = ROOT/'docs/project_status_2026-09-18/tile_coverage_preflight/coverage_plan.json'


def grid(site_id, reference_hash='a'*64, tp=0, fp=1, fn=1, ignored=2):
    tile = {'S017': 'pc_BA31_1000_3849', 'S010': 'pc_BA31_1000_4526',
            'S016': 'pc_BA31_1000_4526'}[site_id]
    return {
        'status': 'DEVELOPMENT_ONLY_TRAINING_SITE_CONVERSION_COMPARISON',
        'site_id': site_id, 'site_split': 'train',
        'reference_sha256': reference_hash,
        'conditions': [{'fraction': 1.0, 'prediction': f'/dummy/{tile}_noise_excluded_reconciled.laz'}],
        'rows': [{
            'fraction': 1.0, 'config_id': 'convex_b000', 'method': 'convex_hull',
            'concave_ratio': None, 'buffer_m': 0.0,
            'tp': tp, 'fp': fp, 'fn': fn,
            'ignored_predictions': ignored,
            'site_predictions': tp+fp+ignored,
        }],
    }


class TrainingSiteGridComparisonTests(unittest.TestCase):
    def invoke(self, first, second):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, out = root/'a.json', root/'b.json', root/'out'
            a.write_text(json.dumps(first))
            b.write_text(json.dumps(second))
            result = subprocess.run(
                [sys.executable, str(SCRIPT), '--grid', str(a), '--grid', str(b),
                 '--coverage-plan', str(COVERAGE),
                 '--config', 'convex_b000', '--output', str(out)],
                cwd=ROOT, capture_output=True, text=True)
            rows = list(csv.DictReader((out/'summary.csv').read_text().splitlines())) if (out/'summary.csv').exists() else None
            summary = json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else None
            return result, rows, summary

    def test_known_micro_counts_and_no_pooled_median(self):
        result, rows, summary = self.invoke(
            grid('S017', tp=4, fp=2, fn=10, ignored=3), grid('S010'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((rows[0]['tp'], rows[0]['fp'], rows[0]['fn']), ('4', '3', '11'))
        self.assertEqual(rows[0]['reference_singles'], '15')
        self.assertAlmostEqual(float(rows[0]['micro_f1']), 8/22)
        self.assertEqual(summary['site_ids'], ['S017', 'S010'])
        self.assertIn('does not report pooled median IoU', summary['matched_iou_note'])

    def test_rejects_mismatched_reference_versions(self):
        result, _, _ = self.invoke(grid('S017'), grid('S010', reference_hash='b'*64))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('different reference versions', result.stderr)

    def test_rejects_held_out_site(self):
        result, _, _ = self.invoke(grid('S017'), grid('S016'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('training-only', result.stderr)

    def test_rejects_unreconciled_prediction_count(self):
        bad = grid('S010')
        bad['rows'][0]['site_predictions'] += 1
        result, _, _ = self.invoke(grid('S017'), bad)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('do not reconcile', result.stderr)

    def test_rejects_prediction_from_another_tile(self):
        bad = grid('S010')
        bad['conditions'][0]['prediction'] = '/dummy/pc_BA31_1000_3849_noise_excluded_reconciled.laz'
        result, _, _ = self.invoke(grid('S017'), bad)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('not a full, safe training tile', result.stderr)


if __name__ == '__main__':
    unittest.main()
