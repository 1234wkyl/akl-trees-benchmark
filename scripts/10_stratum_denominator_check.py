#!/usr/bin/env python
"""What the predeclared per-stratum recall threshold demands of the denominators.

The acceptance rule is "no stratum with recall below 0.70". With only a handful of
reference trees in a stratum, that rule is decided by one or two detections: at n = 5
a single miss moves recall from 1.00 to 0.80, and at n = 1 it moves it to 0.00. This
script turns that into numbers the freeze discussion can use.

It does not propose new sites and it does not change any label. It answers, for every
stratum and split, how much evidence the current reference set actually contains.

    python scripts/10_stratum_denominator_check.py

Writes docs/review_2026-09-17/stratum_denominators.csv and .md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kit import KIT_ROOT, out_dir  # noqa: E402

Z = 1.959963985  # two-sided 95%


def wilson(k: float, n: int, z: float = Z):
    """Wilson score interval; returns (lower, upper) or None when n == 0."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def n_for_lower_bound(p: float, target: float, z: float = Z) -> int | None:
    """Nominal planning calculation at fixed p, not an observed integer count.

    Fractional p*n avoids rounding an assumed recall to a different proportion.
    Actual result intervals must instead use the observed integer TP and n.
    """
    for n in range(1, 4001):
        lo, _ = wilson(p * n, n, z)
        if lo >= target:
            return n
    return None


def max_misses(n: int, threshold: float = 0.70) -> int:
    """Largest number of missed reference trees that still meets the threshold."""
    return int(math.floor((1 - threshold) * n + 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.70,
                    help='predeclared minimum per-stratum recall')
    ap.add_argument('--assumed-recall', type=float, default=0.85,
                    help='recall used only to size the confidence interval')
    ap.add_argument('--robustness', type=int, default=7,
                    help='practical minimum denominator: survives one miss at 0.85 and '
                         'two misses at the stratum threshold')
    ap.add_argument('--output', type=Path,
                    help='report path; CSV is written alongside')
    args = ap.parse_args()
    if not 0 < args.threshold < 1 or not 0 < args.assumed_recall < 1:
        ap.error('threshold and assumed recall must lie strictly between 0 and 1')
    if args.robustness < 1:
        ap.error('robustness must be positive')

    src = KIT_ROOT / 'docs/review_2026-09-17/counts_by_split_stratum.csv'
    rows = list(csv.DictReader(src.read_text().splitlines()))
    need = n_for_lower_bound(args.assumed_recall, args.threshold)

    records = []
    for r in rows:
        n = int(r['tree'])
        lo, hi = wilson(args.assumed_recall * n, n) if n else (None, None)
        records.append({
            'stratum': r['stratum'], 'split': r['split'], 'site': r['sites'],
            'single_trees': n, 'tree_group': int(r['tree_group']),
            'not_tree': int(r['not_tree']), 'uncertain': int(r['uncertain']),
            'max_misses_at_threshold': max_misses(n, args.threshold),
            'recall_after_one_miss': (n - 1) / n if n else None,
            'wilson_lo': lo, 'wilson_hi': hi,
            'shortfall_ci': max(0, need - n) if need is not None else None,
            'shortfall_robust': max(0, args.robustness - n),
        })

    report = args.output or (KIT_ROOT / 'docs/review_2026-09-17/stratum_denominators.md')
    csv_path = report.with_suffix('.csv')
    fields = ['stratum', 'split', 'site', 'single_trees', 'tree_group', 'not_tree',
              'uncertain', 'max_misses_at_threshold', 'recall_after_one_miss',
              'wilson_lo', 'wilson_hi', 'shortfall_ci', 'shortfall_robust']
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for rec in records:
            w.writerow({k: ('' if rec[k] is None else rec[k]) for k in fields})

    def fmt(v, nd=3):
        return 'undefined' if v is None else f'{v:.{nd}f}'

    lines = [
        '# Per-stratum denominators against the predeclared recall threshold', '',
        f'Rule under test: no stratum with recall below {args.threshold:.2f}. '
        f'The interval column uses fixed nominal recall {args.assumed_recall:.2f} '
        'and fractional TP = p*n for illustrative planning; it is not a result. '
        'The binomial approximation ignores correlation between trees within a site. '
        'Neither this interval nor a target sample size adds an acceptance criterion.', '',
        f'Smallest denominator whose Wilson 95% lower bound reaches {args.threshold:.2f} '
        f'at that assumed recall: **n = {need}**.', '',
        '| Stratum | Split | Site | Trees | Max misses | Recall after 1 miss | '
        'Nominal Wilson range | To n=' + str(need) + ' | To n=' + str(args.robustness) + ' |',
        '|---|:--:|:--:|--:|--:|--:|:--:|--:|--:|',
    ]
    order = {'train': 0, 'val': 1, 'test': 2}
    for rec in sorted(records, key=lambda r: (r['stratum'], order[r['split']])):
        ci = 'undefined' if rec['wilson_lo'] is None else \
            f"{rec['wilson_lo']:.2f}-{rec['wilson_hi']:.2f}"
        lines.append(
            f"| {rec['stratum']} | {rec['split']} | {rec['site']} | {rec['single_trees']} "
            f"| {rec['max_misses_at_threshold']} | "
            f"{fmt(rec['recall_after_one_miss'], 2)} | {ci} | "
            f"{rec['shortfall_ci'] if rec['shortfall_ci'] is not None else 'undefined'} | {rec['shortfall_robust']} |")
    lines += [
        '', '## Two different questions', '',
        f'**Illustrative interval planning.** At fixed nominal p = {args.assumed_recall:.2f}, '
        f'the first n with a Wilson lower bound >= {args.threshold:.2f} is {need}. '
        'This is not a power calculation, an approved annotation target or a measured '
        'model result. Small denominators still support an observed recall and a wide '
        'interval, with the assumptions and site correlation stated.', '',
        f'**Discrete sensitivity.** The proposed discussion target is n = {args.robustness}. '
        'For example, n = 7 gives 6/7 = 0.857 after one miss and 5/7 = 0.714 after two. '
        'At n = 5 these are 0.80 and 0.60; at n = 1 any miss fails the threshold. '
        'A target denominator does not justify inventing trees or splitting inseparable '
        'groups; any top-up is conditional on genuine missed trees and supervisor approval.', '',
        '## How to read this', '',
        '`Max misses` is the largest number of reference trees the model may miss in that '
        'stratum before the predeclared threshold fails. The last two columns are how many '
        'single trees would have to be added to reach each target; they are discussion '
        'input for the freeze decision, not a decision to annotate.', '',
        'A stratum below either target is not unusable. It means a pass or fail there rests '
        'on a handful of objects, so the result has to be reported with its denominator.', '',
        'The hard-negative stratum currently has '
        f'with {sum(r["single_trees"] for r in records if r["stratum"] == "hard_negative_built")} '
        'reference trees across all three splits. With a nonzero test denominator, '
        'recall remains computable, although very coarse. Keep the predeclared threshold '
        'and report the denominator. With zero reference trees recall is undefined, '
        'never an automatic pass. No exemption or change to a commission-only measure '
        'has been approved; the separate false-positive precision definition is pending.', '',
        'Tree groups are excluded because the threshold is stated for individual-tree '
        'recall; their counts are shown for context only.',
    ]
    report.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {report}')
    print(f'Wrote {csv_path}')
    print(f'Smallest denominator reaching a {args.threshold:.2f} lower bound at '
          f'{args.assumed_recall:.2f} nominal: n = {need}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
