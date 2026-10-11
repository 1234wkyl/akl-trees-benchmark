"""Supplementary canopy extent and boundary agreement; no instance matching."""
import math

import numpy as np
from shapely.geometry import GeometryCollection, box
from shapely.ops import unary_union

VEGETATION_CLASSES = {'tree', 'tree_group'}


def union_for_classes(frame, classes):
    geometries = [row.geometry for _, row in frame.iterrows() if row['class'] in classes]
    return unary_union(geometries) if geometries else GeometryCollection()


def area_statistics(area_a, area_b, intersection):
    union = area_a + area_b - intersection
    return {'area_a_m2': area_a, 'area_b_m2': area_b,
            'intersection_m2': intersection, 'union_m2': union,
            'symmetric_difference_m2': area_a + area_b - 2 * intersection,
            'coverage_iou': intersection / union if union else None,
            'coverage_dice': 2 * intersection / (area_a + area_b) if area_a + area_b else None}


def line_parts(geometry):
    if geometry.geom_type in {'LineString', 'LinearRing'}:
        if geometry.length:
            yield geometry
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from line_parts(part)


def sampled_boundary(geometry, other, spacing):
    """Length-weighted midpoint sampling; distances to continuous other linework."""
    distances, weights = [], []
    for line in line_parts(geometry):
        n = max(1, math.ceil(line.length / spacing))
        width = line.length / n
        for index in range(n):
            point = line.interpolate((index + .5) * width)
            distances.append(point.distance(other) if not other.is_empty else float('inf'))
            weights.append(width)
    return np.asarray(distances), np.asarray(weights)


def boundary_statistics(a, b, spacing, tolerances):
    da, wa = sampled_boundary(a, b, spacing)
    db, wb = sampled_boundary(b, a, spacing)
    length_a, length_b = float(wa.sum()), float(wb.sum())
    results = []
    for tolerance in tolerances:
        matched_a = float(wa[da <= tolerance].sum())
        matched_b = float(wb[db <= tolerance].sum())
        pa = matched_a / length_a if length_a else None
        pb = matched_b / length_b if length_b else None
        if not length_a and not length_b:
            f1 = None
        elif not length_a or not length_b or not pa or not pb:
            f1 = 0.0
        else:
            f1 = 2 * pa * pb / (pa + pb)
        results.append({'tolerance_m': tolerance,
                        'boundary_length_a_m': length_a, 'boundary_length_b_m': length_b,
                        'matched_boundary_length_a_m': matched_a, 'matched_boundary_length_b_m': matched_b,
                        'fraction_a_near_b': pa, 'fraction_b_near_a': pb,
                        'boundary_f1': f1})
    return results


def coverage_agreement(frames, definitions, policy):
    if policy['vegetation_classes'] != ['tree', 'tree_group']:
        raise ValueError('coverage view must retain its recorded vegetation classes')
    spacing = policy['boundary_sample_spacing_m']
    tolerances = policy['boundary_tolerances_m']
    if not math.isfinite(spacing) or spacing <= 0 or any(not math.isfinite(t) or t < 0 for t in tolerances):
        raise ValueError('finite positive sampling spacing and nonnegative tolerances required')
    per_site = []
    for sid, (fa, fb) in frames.items():
        for frame in [fa, fb]:
            if frame.crs is None or frame.crs.to_epsg() != 2193:
                raise ValueError('coverage agreement requires declared EPSG:2193')
        roi = box(*definitions[sid]['bbox_epsg2193'])
        ua, ub = union_for_classes(fa, VEGETATION_CLASSES), union_for_classes(fb, VEGETATION_CLASSES)
        ca, cb = ua.intersection(roi), ub.intersection(roi)
        # Clip natural boundary linework, not polygon-first artificial site edges.
        ba = ua.boundary.intersection(roi) if not ua.is_empty else GeometryCollection()
        bb = ub.boundary.intersection(roi) if not ub.is_empty else GeometryCollection()
        per_site.append({'site_id': sid,
                         **area_statistics(ca.area, cb.area, ca.intersection(cb).area),
                         'vegetation_objects_a': int(fa['class'].isin(VEGETATION_CLASSES).sum()),
                         'vegetation_objects_b': int(fb['class'].isin(VEGETATION_CLASSES).sum()),
                         'vegetation_outside_site_a_m2': ua.area - ca.area,
                         'vegetation_outside_site_b_m2': ub.area - cb.area,
                         'uncertain_area_a_m2': union_for_classes(fa, {'uncertain'}).intersection(roi).area,
                         'uncertain_area_b_m2': union_for_classes(fb, {'uncertain'}).intersection(roi).area,
                         'boundary': boundary_statistics(ba, bb, spacing, tolerances)})
    pooled = area_statistics(sum(s['area_a_m2'] for s in per_site),
                             sum(s['area_b_m2'] for s in per_site),
                             sum(s['intersection_m2'] for s in per_site))
    # Pool matched lengths, never average per-site F1 values.
    boundary = []
    for index, tolerance in enumerate(tolerances):
        fields = ['boundary_length_a_m', 'boundary_length_b_m',
                  'matched_boundary_length_a_m', 'matched_boundary_length_b_m']
        row = {field: sum(s['boundary'][index][field] for s in per_site) for field in fields}
        la, lb = row['boundary_length_a_m'], row['boundary_length_b_m']
        pa = row['matched_boundary_length_a_m'] / la if la else None
        pb = row['matched_boundary_length_b_m'] / lb if lb else None
        row.update(tolerance_m=tolerance, fraction_a_near_b=pa, fraction_b_near_a=pb,
                   boundary_f1=(None if not la and not lb else
                                2 * pa * pb / (pa + pb) if pa and pb else 0.0))
        boundary.append(row)
    return {'role': 'supplementary extent/boundary agreement; does not replace primary instance/class results',
            'primary_metric_changed': False, 'automatic_pass_or_freeze': False,
            'policy': policy, 'pooled_area': pooled, 'pooled_boundary': boundary,
            'per_site': per_site}
