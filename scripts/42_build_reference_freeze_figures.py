#!/usr/bin/env python3
"""Render source-only RGB/CHM evidence for the pre-freeze reference review."""
import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon

from benchmark.common import sha256, site_definitions, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = (ROOT / 'docs/group_refinement_2026-09-18/'
                  'export_20260922T060150894212Z/'
                  'crown_labels_XZ_2026-09-22-2.json')


def load_labels(path):
    data = json.loads(path.read_text())
    if data.get('schema_version') != 'akl-trees-crown-labels/v1' or data.get('crs') != 'EPSG:2193':
        raise ValueError('expected akl-trees-crown-labels/v1 in EPSG:2193')
    records, geometries = [], []
    for site_id, site in data.get('sites', {}).items():
        for label in site.get('labels', []):
            geometry = Polygon(label['world'])
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if geometry.is_empty or not geometry.is_valid:
                raise ValueError(f'invalid geometry after repair: {label["id"]}')
            records.append({'site_id': site_id, 'label_id': label['id'], 'class': label['cls']})
            geometries.append(geometry)
    return gpd.GeoDataFrame(records, geometry=geometries, crs='EPSG:2193')


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
    ap.add_argument('--labels-json', type=Path, default=DEFAULT_LABELS)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('use a fresh output directory')
    source = args.labels_json.resolve()
    frame = load_labels(source)
    if frame.crs is None or frame.crs.to_epsg() != 2193:
        raise ValueError('EPSG:2193 reference required')
    large = frame[(frame['class'] == 'tree_group') & (frame.geometry.area > 900)].sort_values(['site_id', 'label_id'])
    uncertain = frame[frame['class'] == 'uncertain'].sort_values(['site_id', 'label_id'])
    if len(uncertain) != 4:
        raise ValueError(f'expected 4 uncertain regions, received {len(uncertain)}')
    args.output.mkdir(parents=True)
    source_hashes = {}
    for rows, name, title, pad in ((large, 'large_groups.png', f'{len(large)} current tree groups over 900 square metres', 6),
                                   (uncertain, 'uncertain_regions.png', 'Four unresolved annotations excluded from individual-tree scoring', 12)):
        source_hashes.update(render_sheet(list(rows.itertuples()), args.output / name, title, pad))
    try:
        reference_path = str(source.relative_to(ROOT))
    except ValueError:
        reference_path = str(source)
    write_json(args.output / 'provenance.json', {
        'status': 'PRE-FREEZE REFERENCE REVIEW; NOT INDEPENDENT GROUND TRUTH',
        'reference_path': reference_path, 'reference_sha256': sha256(source),
        'large_group_ids': list(large.label_id), 'uncertain_ids': list(uncertain.label_id),
        'image_source_sha256': source_hashes,
        'outputs_sha256': {p.name: sha256(p) for p in args.output.glob('*.png')},
        'note': 'Only RGB, CHM and the named pre-freeze reference geometry are shown. No model detections were used.'})
    print(json.dumps({'large_groups': len(large), 'uncertain': len(uncertain),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
