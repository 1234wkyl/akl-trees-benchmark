#!/usr/bin/env python
"""Select a training tile, stage unlabelled LAZ, and check local PLY coordinates."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import laspy
import numpy as np
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', default='S001')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--exclude-classes', nargs='*', type=int, default=[], help='explicit candidate preprocessing; preserve raw source')
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        ap.error('use a fresh output folder')
    sites = json.loads((ROOT/'data/sites/reference_sites.json').read_text())['sites']
    selected = next((s for s in sites if s['site_id']==args.site), None)
    if selected is None or selected['split'] != 'train':
        ap.error('smoke preparation requires a training site')
    data = ROOT.parent/'Xi_Zhang_Benchmark/laz'
    index, candidates = [], []
    for path in sorted(data.glob('*.laz')):
        with laspy.open(path) as f:
            h = f.header
            if h.parse_crs().to_epsg() != 2193:
                raise ValueError('unexpected input CRS')
            bounds = box(h.mins[0],h.mins[1],h.maxs[0],h.maxs[1])
            overlaps = []
            for site in sites:
                area = bounds.intersection(box(*site['bbox_epsg2193'])).area
                if area > 0:
                    overlaps.append(site)
                    index.append(dict(site_id=site['site_id'],split=site['split'],tile=path.name,intersection_area_m2=area,header_points=h.point_count))
            if selected in overlaps and not any(s['split'] != 'train' for s in overlaps):
                candidates.append((bounds.intersection(box(*selected['bbox_epsg2193'])).area,path))
    if not candidates:
        raise ValueError('no tile intersecting this train site without val/test-site overlap')
    _, source = sorted(candidates,key=lambda item:(-item[0],item[1].name))[0]
    points = laspy.read(source)
    original_count = len(points)
    z_summary = []
    for cls in np.unique(points.classification):
        z = np.asarray(points.z)[points.classification == cls]
        z_summary.append(dict(classification=int(cls),points=len(z),z_min_m=float(z.min()),z_p01_m=float(np.quantile(z,.01)),z_median_m=float(np.median(z)),z_p99_m=float(np.quantile(z,.99)),z_max_m=float(z.max())))
    keep = ~np.isin(points.classification,args.exclude_classes)
    original_indices = np.flatnonzero(keep)
    if not keep.any():
        raise ValueError('preprocessing removed every point')
    points.points = points.points[keep]
    xyz = np.column_stack((points.x,points.y,points.z))
    if not np.isfinite(xyz).all():
        raise ValueError('non-finite coordinates')
    origin = xyz.min(axis=0)
    local = (xyz-origin).astype('<f4')
    max_error = float(np.abs(local.astype('f8')+origin-xyz).max())
    if max_error > 0.001:
        raise ValueError('coordinate round-trip error exceeds one millimetre')
    out.mkdir(parents=True)
    (out/'input_laz').mkdir(); (out/'diagnostic_ply').mkdir(); (out/'raw_source').mkdir()
    raw_copy = out/'raw_source'/source.name
    shutil.copy2(source,raw_copy)
    assert sha(source)==sha(raw_copy)
    if args.exclude_classes:
        copy = out/'input_laz'/(source.stem+'_noise_excluded.laz')
        points.write(copy)
    else:
        copy = out/'input_laz'/source.name
        shutil.copy2(source,copy)
    dtype = [('x','<f4'),('y','<f4'),('z','<f4'),('intensity','<u2'),('classification','u1'),('return_number','u1'),('point_index','<u4')]
    vertices = np.empty(len(points),dtype=dtype)
    for i,key in enumerate(['x','y','z']):
        vertices[key] = local[:,i]
    vertices['intensity']=points.intensity
    vertices['classification']=points.classification
    vertices['return_number']=points.return_number
    vertices['point_index']=original_indices.astype('u4')
    ply = out/'diagnostic_ply'/(source.stem+'_local.ply')
    header = f'ply\nformat binary_little_endian 1.0\ncomment Unlabelled inference diagnostic; not training labels\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty ushort intensity\nproperty uchar classification\nproperty uchar return_number\nproperty uint point_index\nend_header\n'
    with ply.open('wb') as f:
        f.write(header.encode('ascii'));vertices.tofile(f)
    unique, counts = np.unique(points.classification,return_counts=True)
    metadata = {
        'status':'CPU INPUT PREPARATION ONLY; CUDA inference not run',
        'training_site':args.site,'selected_tile':source.name,'source_sha256':sha(source),
        'original_points':original_count,'excluded_points':original_count-len(points),
        'input_laz_sha256':sha(copy), 'input_laz_filename':copy.name,
        'points':len(points),'crs':'EPSG:2193','las_scales':list(map(float,points.header.scales)),
        'las_offsets':list(map(float,points.header.offsets)),
        'local_origin_xyz_m':list(map(float,origin)), 'max_coordinate_roundtrip_error_m':max_error,
        'classes':dict(zip(map(str,unique),map(int,counts))),
        'local_ply_sha256':sha(ply),
        'held_out_reference_sites_intersecting_selected_tile':[],
        'filtering':{'excluded_asprs_classes':args.exclude_classes,'status':'candidate smoke preprocessing; not final benchmark policy; all retained class values/return numbers unchanged'},
        'training_labels_created':False,
        'height_normalization':'subtract global minimum Z only, not height-above-ground normalization',
        'upstream_inference_input':'input_laz contains one unlabelled input; original source preserved in raw_source',
        'diagnostic_ply_warning':'already local; do not feed to upstream geographic-restoration pipeline without separately applying local_origin_xyz_m',
    }
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with (out/'raw_class_height_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(z_summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(z_summary)
    with (out/'site_tile_index.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(index[0]),lineterminator='\n');writer.writeheader();writer.writerows(index)
    (out/'README.md').write_text('''# SegmentAnyTree smoke input preparation

The selected LAZ tile intersects a train site and no val/test reference-site box. The raw source is preserved in raw_source. If explicit excluded classes are supplied, input_laz contains the corresponding candidate subset; retained native dimensions, class values, scales and offsets are unchanged. Use input_laz as the input to the upstream inference pipeline on Nectar; it contains no reference labels. The diagnostic local PLY has no semantic_seg or treeID attributes and must never be described as training ground truth. Original ASPRS classification is retained as an attribute, not mapped to the upstream training taxonomy. Raw height quantiles by class are recorded separately; no unrecorded height clipping is applied.

The local PLY subtracts XYZ minima before float32 conversion, stores the geographic origin and original point index, and has a measured round-trip error. Minimum-Z translation is not terrain normalization. The upstream loader accepts unlabelled XYZ, but may create placeholder labels internally; any metrics against those placeholders are not real benchmark metrics. Evaluate predictions separately against the accepted independent crown reference.

The container/code environment still requires validation on Nectar. Pin an image digest, bind the recorded code and checkpoint, and retain its driver/package logs. Do not assume the mutable latest image contains the recorded code version or checkpoint. No model inference has run here.
''')
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
