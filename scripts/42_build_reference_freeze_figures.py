#!/usr/bin/env python3
"""Render source-only RGB/CHM evidence for the pre-freeze reference review."""
import argparse
import json
import math
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image

from benchmark.common import sha256, site_definitions, write_json

ROOT = Path(__file__).resolve().parents[1]


def render_sheet(rows, output, title, pad_m, ncols=2):
    sites = site_definitions()
    nrows = math.ceil(len(rows) / ncols)
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(12, 3.5 * nrows), squeeze=False)
    sources = {}
    for index, row in enumerate(rows):
        site = row.site_id
        x0, y0, x1, y1 = sites[site]['bbox_epsg2193']
        bounds = row.geometry.bounds
        view = (max(x0, bounds[0] - pad_m), min(x1, bounds[2] + pad_m),
                max(y0, bounds[1] - pad_m), min(y1, bounds[3] + pad_m))
        for offset, kind in enumerate(('rgb', 'chm')):
            ax = axes[index // ncols, (index % ncols) * 2 + offset]
            image_path = ROOT / 'data/sites/chips' / site / f'{kind}.png'
            with Image.open(image_path) as image:
                ax.imshow(image, extent=(x0, x1, y0, y1), origin='upper')
            sources[str(image_path.relative_to(ROOT))] = sha256(image_path)
            line = row.geometry.boundary
            parts = list(line.geoms) if hasattr(line, 'geoms') else [line]
            for part in parts:
                xs, ys = part.xy
                ax.plot(xs, ys, color='#ffcc00', linewidth=1.8)
            ax.set_xlim(view[0], view[1]); ax.set_ylim(view[2], view[3])
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{row.label_id} | {kind.upper()} | {row.geometry.area:.1f} m²', fontsize=8)
    for index in range(len(rows), nrows * ncols):
        axes[index // ncols, (index % ncols) * 2].axis('off')
        axes[index // ncols, (index % ncols) * 2 + 1].axis('off')
    fig.suptitle(title, fontsize=14, y=.997)
    fig.tight_layout(rect=(0, 0, 1, .975))
    fig.savefig(output, dpi=170, facecolor='white')
    plt.close(fig)
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('use a fresh output directory')
    source = ROOT / 'data/exports/reference_crowns.gpkg'
    frame = gpd.read_file(source, layer='reference_crowns')
    if frame.crs is None or frame.crs.to_epsg() != 2193:
        raise ValueError('EPSG:2193 reference required')
    large = frame[(frame['class'] == 'tree_group') & (frame.geometry.area > 900)].sort_values(['site_id', 'label_id'])
    uncertain = frame[frame['class'] == 'uncertain'].sort_values(['site_id', 'label_id'])
    if len(large) != 11 or len(uncertain) != 4:
        raise ValueError(f'expected 11 large groups and 4 uncertain, received {len(large)} and {len(uncertain)}')
    args.output.mkdir(parents=True)
    source_hashes = {}
    for rows, name, title, pad in ((large, 'large_groups.png', 'Eleven flagged tree groups over 900 square metres', 6),
                                   (uncertain, 'uncertain_regions.png', 'Four unresolved annotations excluded from individual-tree scoring', 12)):
        source_hashes.update(render_sheet(list(rows.itertuples()), args.output / name, title, pad))
    example = ROOT / 'docs/group_refinement_2026-09-18/visual_review/S014-mss0nwrf.png'
    shutil.copy2(example, args.output / 'group_split_example_S014.png')
    write_json(args.output / 'provenance.json', {
        'status': 'PRE-FREEZE REFERENCE REVIEW; NOT INDEPENDENT GROUND TRUTH',
        'reference_path': str(source.relative_to(ROOT)), 'reference_sha256': sha256(source),
        'large_group_ids': list(large.label_id), 'uncertain_ids': list(uncertain.label_id),
        'image_source_sha256': source_hashes,
        'group_split_example_source': str(example.relative_to(ROOT)),
        'group_split_example_source_sha256': sha256(example),
        'outputs_sha256': {p.name: sha256(p) for p in args.output.glob('*.png')},
        'note': 'Only original RGB, CHM and reference geometry are shown. No model detections were used.'})
    print(json.dumps({'large_groups': len(large), 'uncertain': len(uncertain),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
