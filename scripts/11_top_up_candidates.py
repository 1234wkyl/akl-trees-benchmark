#!/usr/bin/env python
"""Render existing labels over RGB and CHM for the thin stratum cells.

With two or three reference trees, per-stratum recall is computable but coarse.
This script does not choose new labels; it draws what is already there so
the decision about a targeted top-up can be made by looking, and so the extra
objects can be added in the labeller rather than by hand-editing JSON.

    python scripts/11_top_up_candidates.py

Writes docs/review_2026-09-17/top_up_candidates/<SITE>_<split>_<stratum>.png
and an index.md listing the sites in priority order.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kit import KIT_ROOT  # noqa: E402

COLOURS = {'tree': (46, 204, 113), 'tree_group': (230, 126, 34),
           'not_tree': (52, 152, 219), 'uncertain': (241, 196, 15)}
SPLIT_ORDER = {'test': 0, 'val': 1, 'train': 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-trees', type=int, default=7,
                    help='render every stratum-split cell with fewer single trees')
    ap.add_argument('--csv', type=Path,
                    default=KIT_ROOT / 'docs/review_2026-09-17/counts_by_split_stratum.csv')
    args = ap.parse_args()

    cells = [r for r in csv.DictReader(args.csv.read_text().splitlines())
             if int(r['tree']) < args.min_trees]
    cells.sort(key=lambda r: (SPLIT_ORDER[r['split']], int(r['tree']), r['stratum']))

    labels = json.loads((KIT_ROOT / 'data/labels/crown_labels_XZ_2026-08-13.json').read_text())
    out = KIT_ROOT / 'docs/review_2026-09-17/top_up_candidates'
    out.mkdir(parents=True, exist_ok=True)

    index = ['# Top-up candidate overlays', '',
             f'Stratum-split cells with fewer than {args.min_trees} individual trees, '
             'test first. Green = tree, orange = tree group, blue = hard negative, '
             'yellow = uncertain. Left is RGB, right is CHM.', '',
             'These images are for deciding whether more individual crowns can be '
             'delineated defensibly. They are not new labels or an approved count target. '
             'Only after a top-up decision, add genuine missed crowns manually in the '
             'labeller and rerun QA; never split inseparable groups to reach a count.', '',
             '| Site | Split | Stratum | Trees now | Shortfall | Overlay |',
             '|---|:--:|---|--:|--:|---|']
    for c in cells:
        sid = c['sites']
        meta = json.loads((KIT_ROOT / f'data/sites/chips/{sid}/meta.json').read_text())
        x0, y0, x1, y1 = meta['bbox_epsg2193']
        res = meta['resolution_m']
        rec = labels['sites'].get(sid, {})
        objs = rec.get('labels', [])

        sheet = Image.new('RGB', (860, 460), 'white')
        for j, layer in enumerate(['rgb', 'chm']):
            sheet.paste(Image.open(
                KIT_ROOT / f'data/sites/chips/{sid}/{layer}.png').convert('RGB'),
                (20 + j * 420, 40))
        draw = ImageDraw.Draw(sheet)
        for lab in objs:
            poly = Polygon(lab['world'])
            if not poly.is_valid:
                poly = poly.buffer(0)
            colour = COLOURS.get(lab['cls'], (0, 0, 0))
            for part in ([poly] if poly.geom_type == 'Polygon' else poly.geoms):
                for j in range(2):
                    pts = [(20 + j * 420 + (x - x0) / res - 0.5,
                            40 + (y1 - y) / res - 0.5) for x, y in part.exterior.coords]
                    draw.line(pts, fill=colour, width=2)
        counts = {}
        for lab in objs:
            counts[lab['cls']] = counts.get(lab['cls'], 0) + 1
        draw.text((20, 8), f"{sid}  {c['split']}  {c['stratum']}  "
                           f"tree={counts.get('tree', 0)} group={counts.get('tree_group', 0)} "
                           f"not_tree={counts.get('not_tree', 0)} "
                           f"uncertain={counts.get('uncertain', 0)}", fill='black')
        draw.text((20, 445), 'RGB left, CHM right; 100 x 100 m', fill='black')
        name = f"{sid}_{c['split']}_{c['stratum']}.png"
        sheet.save(out / name)
        index.append(f"| {sid} | {c['split']} | {c['stratum']} | {c['tree']} | "
                     f"{args.min_trees - int(c['tree'])} | {name} |")

    index += ['', f'{len(cells)} cells rendered. '
                  'Hard-negative sites have few individual trees; keep the predeclared '
                  'recall requirement and report its denominator. A commission-only '
                  'reinterpretation would require explicit approval, which is pending.']
    (out / 'index.md').write_text('\n'.join(index) + '\n')
    print(f'Wrote {len(cells)} overlays to {out}')
    for c in cells:
        print(f"  {c['sites']} {c['split']:<5} {c['stratum']:<22} "
              f"trees={c['tree']} shortfall={args.min_trees - int(c['tree'])}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
