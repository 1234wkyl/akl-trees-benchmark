#!/usr/bin/env python
"""Reconcile an upstream SegmentAnyTree LAZ to the authoritative input points.

The upstream merge joins point, semantic, and instance tables by XYZ. Duplicate
XYZ values can therefore create a Cartesian expansion, and its LAS writer does
not retain the source CRS. This command accepts a result only when unique XYZ
locations form a reciprocal one-to-one match within a small tolerance and all
duplicate prediction rows agree. It then copies prediction fields onto the
original point records and header.
"""
import argparse
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

from benchmark.common import sha256, write_json


def _xyz(las):
    return np.column_stack((las.x, las.y, las.z))


def _unique_rows(xyz):
    unique, inverse, counts = np.unique(
        np.asarray(xyz, dtype=np.float64), axis=0,
        return_inverse=True, return_counts=True)
    return unique, inverse, counts


def _consistent_group_values(inverse, values, groups, name):
    order = np.argsort(inverse, kind='stable')
    sorted_groups = inverse[order]
    sorted_values = np.asarray(values)[order]
    conflict = ((sorted_groups[1:] == sorted_groups[:-1]) &
                (sorted_values[1:] != sorted_values[:-1]))
    if conflict.any():
        bad = np.unique(sorted_groups[1:][conflict])
        raise ValueError(f'{name} conflicts at {len(bad)} duplicate XYZ groups')
    first = np.full(groups, -1, dtype=np.int64)
    first[sorted_groups] = order
    if (first < 0).any():
        raise AssertionError(f'missing {name} group representative')
    return np.asarray(values)[first]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--prediction', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--coordinate-tolerance-m', type=float, default=.002)
    args = ap.parse_args()
    if args.output.exists() or args.report.exists():
        ap.error('use fresh output and report paths')
    if not np.isfinite(args.coordinate_tolerance_m) or args.coordinate_tolerance_m <= 0:
        ap.error('positive coordinate tolerance required')

    source = laspy.read(args.input)
    prediction = laspy.read(args.prediction)
    crs = source.header.parse_crs()
    if crs is None or crs.to_epsg() != 2193:
        raise ValueError('authoritative source CRS must be EPSG:2193')
    dimensions = set(prediction.point_format.dimension_names)
    required = {'PredInstance', 'PredSemantic'}
    if not required <= dimensions:
        raise ValueError(f'missing prediction fields: {sorted(required-dimensions)}')

    source_xyz = _xyz(source)
    prediction_xyz = _xyz(prediction)
    if (not len(source_xyz) or not len(prediction_xyz) or
            not np.isfinite(source_xyz).all() or not np.isfinite(prediction_xyz).all()):
        raise ValueError('empty or non-finite coordinates')
    source_unique, source_inverse, source_counts = _unique_rows(source_xyz)
    pred_unique, pred_inverse, pred_counts = _unique_rows(prediction_xyz)
    if len(source_unique) != len(pred_unique):
        raise ValueError('unique XYZ counts differ; refusing lossy reconciliation')

    pred_instance = _consistent_group_values(
        pred_inverse, prediction['PredInstance'], len(pred_unique), 'PredInstance')
    pred_semantic = _consistent_group_values(
        pred_inverse, prediction['PredSemantic'], len(pred_unique), 'PredSemantic')

    distances, source_to_pred = cKDTree(pred_unique).query(source_unique, k=1)
    reverse_distances, pred_to_source = cKDTree(source_unique).query(pred_unique, k=1)
    reciprocal = np.arange(len(source_unique)) == pred_to_source[source_to_pred]
    bijective = len(np.unique(source_to_pred)) == len(source_unique)
    if ((distances > args.coordinate_tolerance_m).any() or
            (reverse_distances > args.coordinate_tolerance_m).any() or
            not reciprocal.all() or not bijective):
        raise ValueError('unique XYZ mapping is not a reciprocal tolerance-bounded bijection')

    mapped_instance = pred_instance[source_to_pred][source_inverse]
    mapped_semantic = pred_semantic[source_to_pred][source_inverse]
    if mapped_instance.min() < 0 or mapped_instance.max() > np.iinfo(np.uint16).max:
        raise ValueError('PredInstance cannot be represented as uint16')
    if mapped_semantic.min() < 0 or mapped_semantic.max() > np.iinfo(np.uint8).max:
        raise ValueError('PredSemantic cannot be represented as uint8')

    output = laspy.LasData(source.header.copy())
    output.points = source.points.copy()
    output.add_extra_dim(laspy.ExtraBytesParams(name='PredInstance', type='uint16'))
    output.add_extra_dim(laspy.ExtraBytesParams(name='PredSemantic', type='uint8'))
    output['PredInstance'] = mapped_instance.astype(np.uint16)
    output['PredSemantic'] = mapped_semantic.astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.write(args.output, do_compress=args.output.suffix.lower() == '.laz')

    report = {
        'status': 'RECONCILED',
        'source': str(args.input.resolve()),
        'upstream_prediction': str(args.prediction.resolve()),
        'output': str(args.output.resolve()),
        'source_sha256': sha256(args.input),
        'upstream_prediction_sha256': sha256(args.prediction),
        'output_sha256': sha256(args.output),
        'source_points': len(source.points),
        'upstream_prediction_points': len(prediction.points),
        'output_points': len(output.points),
        'source_unique_xyz': len(source_unique),
        'prediction_unique_xyz': len(pred_unique),
        'source_duplicate_excess': int((source_counts - 1).sum()),
        'prediction_duplicate_excess': int((pred_counts - 1).sum()),
        'removed_cartesian_join_rows': len(prediction.points) - len(source.points),
        'coordinate_tolerance_m': args.coordinate_tolerance_m,
        'max_source_to_prediction_unique_distance_m': float(distances.max()),
        'max_prediction_to_source_unique_distance_m': float(reverse_distances.max()),
        'reciprocal_unique_mapping': True,
        'bijective_unique_mapping': True,
        'duplicate_prediction_label_conflicts': 0,
        'output_epsg': output.header.parse_crs().to_epsg(),
        'positive_instances': int(np.unique(mapped_instance[mapped_instance > 0]).size),
        'max_instance_id': int(mapped_instance.max()),
        'semantic_values': list(map(int, np.unique(mapped_semantic))),
        'note': ('Prediction fields were transferred only after reciprocal unique-XYZ '
                 'matching; original source point order, attributes, and CRS were retained.')
    }
    write_json(args.report, report)
    print(f'Reconciled output saved: {args.output}')
    print(f'Reconciliation report saved: {args.report}')


if __name__ == '__main__':
    main()
