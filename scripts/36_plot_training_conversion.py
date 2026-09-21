#!/usr/bin/env python
"""Render training-site crown-conversion geometry against RGB and CHM chips."""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image

from benchmark.common import sha256, site_definitions, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site-id', required=True)
    ap.add_argument('--chips', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--prediction-a', type=Path, required=True)
    ap.add_argument('--prediction-b', type=Path, required=True)
    ap.add_argument('--label-a', default='A')
    ap.add_argument('--label-b', default='B')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sites = site_definitions()
    if args.site_id not in sites or sites[args.site_id]['split'] != 'train':
        ap.error('diagnostic plotting is restricted to training sites')
    if args.output.exists():
        ap.error('use a fresh output directory')
    extent = sites[args.site_id]['bbox_epsg2193']
    image_extent = (extent[0],extent[2],extent[1],extent[3])
    rgb, chm = (args.chips/'rgb.png'), (args.chips/'chm.png')
    layers = {name: gpd.read_file(args.reference, layer=name)
              for name in ('single_trees','tree_groups','hard_negatives')}
    for name, frame in layers.items():
        if frame.crs is None or frame.crs.to_epsg() != 2193:
            raise ValueError(f'{name} reference CRS is not EPSG:2193')
        layers[name] = frame[frame.site_id == args.site_id]
    predictions = [gpd.read_file(path, layer='predictions')
                   for path in (args.prediction_a,args.prediction_b)]
    for frame in predictions:
        if frame.crs is None or frame.crs.to_epsg() != 2193:
            raise ValueError('prediction CRS is not EPSG:2193')
        if set(frame.site_id) - {args.site_id}:
            raise ValueError('prediction file includes another site')

    fig, axes = plt.subplots(2,2,figsize=(14,14),layout='constrained')
    for row, (image_path, image_label) in enumerate(((rgb,'RGB'),(chm,'CHM'))):
        with Image.open(image_path) as im:
            background = im.copy()
        for col, (prediction, label) in enumerate(zip(predictions,(args.label_a,args.label_b))):
            ax = axes[row,col]
            ax.imshow(background,extent=image_extent,origin='upper')
            for name, color, style in (('single_trees','#00ff66','-'),
                                       ('tree_groups','#ffda44','--'),
                                       ('hard_negatives','#ff70d2',':')):
                if not layers[name].empty:
                    layers[name].boundary.plot(ax=ax,color=color,linewidth=1.2,linestyle=style)
            if not prediction.empty:
                prediction.boundary.plot(ax=ax,color='#00d8ff',linewidth=1.0)
            ax.set_xlim(extent[0],extent[2]);ax.set_ylim(extent[1],extent[3])
            ax.set_aspect('equal');ax.set_title(f'{image_label}: {label} ({len(prediction)} crowns)')
            ax.set_xlabel('Easting (EPSG:2193)');ax.set_ylabel('Northing (EPSG:2193)')
    fig.suptitle(f'{args.site_id} training-only, pre-freeze conversion comparison\n'
                 'green=single reference; yellow=group; pink=hard negative; cyan=prediction',fontsize=15)
    args.output.mkdir(parents=True)
    target=args.output/f'{args.site_id}_conversion_comparison.png'
    fig.savefig(target,dpi=180);plt.close(fig)
    write_json(args.output/'provenance.json',{
        'status':'DEVELOPMENT ONLY; not held-out benchmark',
        'site_id':args.site_id,'split':'train',
        'rgb_sha256':sha256(rgb),'chm_png_sha256':sha256(chm),
        'reference_sha256':sha256(args.reference),
        'prediction_a_sha256':sha256(args.prediction_a),
        'prediction_b_sha256':sha256(args.prediction_b),
        'labels':[args.label_a,args.label_b],
        'reference_counts':{name:len(frame) for name,frame in layers.items()},
        'prediction_counts':[len(frame) for frame in predictions],
        'figure_sha256':sha256(target),
        'note':'Image overlay is a visual diagnostic only; geometry and manual labels are not modified.'})
    print(json.dumps({'figure':str(target),'sha256':sha256(target)}))


if __name__=='__main__':
    main()
