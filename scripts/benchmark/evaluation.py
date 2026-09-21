"""Deterministic proposal for crown scoring, plus paired site bootstrap."""
import math
from statistics import median

import numpy as np


def score_site(references, predictions, groups=(), uncertain=(), tolerance=2.0,
               minimum_iou=0.0, ignore_rule='none'):
    """Inputs are (stable_id, valid clipped polygon), never stem/apex labels."""
    if not math.isfinite(tolerance) or tolerance < 0 or not 0 <= minimum_iou <= 1:
        raise ValueError('invalid matching tolerance/IoU')
    if ignore_rule not in {'none', 'unmatched_centroid_in_group_or_uncertain'}:
        raise ValueError('unknown prediction-ignore rule')
    references = [(str(i), g) for i, g in references]
    predictions = [(str(i), g) for i, g in predictions]
    for objects in (references, predictions):
        ids = [x[0] for x in objects]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate object IDs within site')
        if any(g.is_empty or not g.is_valid or g.area <= 0 for _, g in objects):
            raise ValueError('expected valid nonempty crown polygons')
    candidates = []
    for rid, ref in references:
        for pid, pred in predictions:
            distance = ref.centroid.distance(pred.centroid)
            if distance > tolerance:
                continue
            overlap = ref.intersection(pred).area
            iou = overlap / (ref.area + pred.area - overlap)
            if iou >= minimum_iou:
                candidates.append((-iou, distance, str(rid), str(pid)))
    candidates.sort()
    used_r, used_p, matches = set(), set(), []
    for neg_iou, distance, rid, pid in candidates:
        if rid in used_r or pid in used_p:
            continue
        used_r.add(rid); used_p.add(pid)
        matches.append({'reference_id': rid, 'prediction_id': pid,
                        'iou': -neg_iou, 'centroid_distance_m': distance})
    ignored, false_positives = [], []
    for pid, pred in predictions:
        pid = str(pid)
        if pid in used_p:
            continue
        if ignore_rule != 'none' and any(region.covers(pred.centroid)
                                         for region in (*groups, *uncertain)):
            ignored.append(pid)
        else:
            false_positives.append(pid)
    # Geometry-only diagnostics are separate from the 2 m detection matches.
    # A link covers at least 10% of the smaller crown. Ignored predictions are
    # excluded, while matched predictions in ambiguity regions remain scored.
    diagnostic_predictions = [(pid, p) for pid, p in predictions if pid not in ignored]
    reference_degree = {rid: 0 for rid, _ in references}
    prediction_degree = {pid: 0 for pid, _ in diagnostic_predictions}
    for rid, ref in references:
        for pid, pred in diagnostic_predictions:
            if ref.intersection(pred).area / min(ref.area, pred.area) >= 0.10:
                reference_degree[rid] += 1
                prediction_degree[pid] += 1
    return {'tp': len(matches), 'fp': len(false_positives),
            'fn': len(references) - len(matches), 'n_reference': len(references),
            'n_predictions': len(predictions), 'ignored': len(ignored),
            'matches': matches, 'ious': [m['iou'] for m in matches],
            'unmatched_reference_ids': sorted(str(r) for r, _ in references if str(r) not in used_r),
            'false_positive_ids': sorted(false_positives), 'ignored_prediction_ids': sorted(ignored),
            'over_segmented_references': sum(v >= 2 for v in reference_degree.values()),
            'under_segmented_predictions': sum(v >= 2 for v in prediction_degree.values()),
            'diagnostic_overlap_rule': 'intersection / min(crown areas) >= 0.10; no distance filter',
            'reference_overlap_degrees': reference_degree, 'prediction_overlap_degrees': prediction_degree}


def aggregate(records):
    tp, fp, fn = (sum(r[k] for r in records) for k in ('tp', 'fp', 'fn'))
    ious = [v for r in records for v in r['ious']]
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'commission_rate': fp / (tp + fp) if tp + fp else None,
            'omission_rate': fn / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'median_matched_iou': median(ious) if ious else None,
            'over_segmented_references': sum(r.get('over_segmented_references', 0) for r in records),
            'under_segmented_predictions': sum(r.get('under_segmented_predictions', 0) for r in records),
            'over_segmentation_rate': sum(r.get('over_segmented_references', 0) for r in records) / (tp + fn) if tp + fn else None,
            'under_segmentation_rate': sum(r.get('under_segmented_predictions', 0) for r in records) / (tp + fp) if tp + fp else None,
            'ignored_predictions': sum(r.get('ignored', 0) for r in records)}


def paired_site_bootstrap(candidate, baseline=None, iterations=2000, seed=42):
    """Resample whole sites, with the same draws for both methods; micro F1."""
    sites = sorted(candidate)
    if not sites or iterations < 1:
        raise ValueError('at least one site and one iteration required')
    if baseline is not None:
        if set(baseline) != set(sites):
            raise ValueError('models must cover identical sites, including empty sites')
        if any(candidate[s]['tp'] + candidate[s]['fn'] != baseline[s]['tp'] + baseline[s]['fn'] for s in sites):
            raise ValueError('models must use identical reference counts')
    rng = np.random.Generator(np.random.PCG64(seed))
    values = {k: [] for k in ('precision', 'recall', 'commission_rate',
                             'omission_rate', 'f1', 'median_matched_iou',
                             'over_segmentation_rate', 'under_segmentation_rate')}
    deltas = []
    for draw in rng.integers(0, len(sites), size=(iterations, len(sites))):
        c = aggregate([candidate[sites[int(i)]] for i in draw])
        for k in values:
            if c[k] is not None:
                values[k].append(c[k])
        if baseline is not None:
            b = aggregate([baseline[sites[int(i)]] for i in draw])
            if c['f1'] is not None and b['f1'] is not None:
                deltas.append(c['f1'] - b['f1'])
    def interval(v):
        return {'lower': float(np.quantile(v, .025)) if v else None,
                'upper': float(np.quantile(v, .975)) if v else None,
                'valid_replicates': len(v)}
    result = {'unit': 'paired_site', 'stratification': 'none', 'iterations': iterations,
              'seed': seed, 'rng': 'PCG64', 'numpy_version': np.__version__,
              'sites': sites, 'intervals': {k: interval(v) for k, v in values.items()},
              'limitation': 'Few sites; within-site correlation preserved, between-site dependence not modelled. No stratum-specific interval from a single site.'}
    if baseline is not None:
        result['f1_improvement'] = interval(deltas)
        # A negative CI excluding zero is evidence against, not for, superiority.
        result['positive_improvement_ci_excludes_zero'] = bool(
            len(sites) > 1 and len(deltas) == iterations
            and result['f1_improvement']['lower'] > 0)
    return result
