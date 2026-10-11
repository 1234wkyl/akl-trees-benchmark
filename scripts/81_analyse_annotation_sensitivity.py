#!/usr/bin/env python
"""Supplementary object, location and region correspondence of revised labels."""
import argparse
import csv
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw
from shapely.geometry import GeometryCollection, box
from benchmark.common import ROOT, sha256, site_definitions, write_json
from benchmark.annotation_coverage import union_for_classes, area_statistics
from benchmark.evaluation import score_site

spec = importlib.util.spec_from_file_location('intake', ROOT / 'scripts/31_receive_independent_labels.py')
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)
agreement = intake.agreement


def smaller_overlap(a, b):
    denominator = min(a.area, b.area)
    return a.intersection(b).area / denominator if denominator else 0.0


def overlap_matches(a, b, threshold, smaller=False):
    candidates = []
    for i, (ia, ga) in enumerate(a):
        for j, (ib, gb) in enumerate(b):
            value = smaller_overlap(ga, gb) if smaller else agreement.iou(ga, gb)
            if value >= threshold and ga.intersection(gb).area > 0:
                candidates.append((-value, str(ia), str(ib), i, j))
    candidates.sort()
    used_a, used_b, pairs = set(), set(), []
    for neg, ia, ib, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j)
            pairs.append(dict(id_a=ia, id_b=ib, overlap_value=-neg,
                iou=agreement.iou(a[i][1], b[j][1]),
                centroid_distance_m=a[i][1].centroid.distance(b[j][1].centroid)))
    return pairs


def tolerant_area(a, b, roi, distance):
    if distance < 0:
        raise ValueError('nonnegative tolerance required')
    a, b = a.intersection(roi), b.intersection(roi)
    ba = a.buffer(distance).intersection(roi) if distance else a
    bb = b.buffer(distance).intersection(roi) if distance else b
    return dict(tolerance_m=distance, area_a_m2=a.area, area_b_m2=b.area,
        supported_area_a_m2=a.intersection(bb).area,
        supported_area_b_m2=b.intersection(ba).area)


def tolerant_score(row):
    la, lb = row['area_a_m2'], row['area_b_m2']
    pa = row['supported_area_a_m2'] / la if la else None
    pb = row['supported_area_b_m2'] / lb if lb else None
    value = None if not la and not lb else 2 * pa * pb / (pa + pb) if pa and pb else 0.0
    return dict(**row, fraction_a_supported=pa, fraction_b_supported=pb, tolerant_area_f=value)


def region_links(a, b, threshold):
    degree_a = {str(i): 0 for i, _ in a}
    degree_b = {str(i): 0 for i, _ in b}
    edges = []
    for ia, ga in a:
        for ib, gb in b:
            overlap = smaller_overlap(ga, gb)
            if overlap >= threshold and ga.intersection(gb).area > 0:
                edges.append(dict(id_a=str(ia), id_b=str(ib), smaller_overlap=overlap,
                    iou=agreement.iou(ga, gb)))
                degree_a[str(ia)] += 1; degree_b[str(ib)] += 1
    return edges, degree_a, degree_b


def object_summary(rows, name):
    na = sum(r['n_a'] for r in rows); nb = sum(r['n_b'] for r in rows)
    matches = [m for r in rows for m in r['pairs']]
    return dict(method=name, n_a=na, n_b=nb, matched=len(matches),
        f1=2 * len(matches) / (na + nb) if na + nb else None,
        median_matched_iou=median([m['iou'] for m in matches]) if matches else None,
        per_site=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xz-json', type=Path, required=True)
    ap.add_argument('--tim-json', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Existing results preserved; choose a fresh output directory')
    config = ROOT / 'config/annotation_sensitivity_20261009.json'
    policy = json.loads(config.read_text()); definitions = site_definitions()
    export_data=json.loads(args.xz_json.read_text())
    global_trees=sum(l['cls']=='tree' for rec in export_data['sites'].values() for l in rec.get('labels',[]))
    a, ca = agreement.load_independent_json(args.xz_json, 'XZ', definitions, allow_revision_attribution=True)
    b, cb = agreement.load_independent_json(args.tim_json, 'TW', definitions)
    if not set(policy['sites']) <= ca & cb:
        ap.error('Both comparison sites must be complete')
    started = datetime.now(timezone.utc)
    out = args.output; out.mkdir(parents=True); (out / 'inputs').mkdir()
    for source, name in [(args.xz_json, 'XZ_revised.json'), (args.tim_json, 'TW_original.json')]:
        target = out / 'inputs' / name; shutil.copyfile(source, target); target.chmod(0o444)
    shutil.copyfile(config, out / 'policy.json')
    methods = {f'IoU >= {threshold:.2f}': [] for threshold in policy['iou_thresholds']}
    methods['Smaller-area overlap >= 0.50'] = []; methods['Centroid distance <= 2 m'] = []
    regions, links, object_coverage = [], [], []
    tolerance_rows = {d: [] for d in policy['area_tolerances_m']}
    gallery = out / 'visual_review'; gallery.mkdir()
    for sid in policy['sites']:
        fa = a[a.site_id == sid]; fb = b[b.site_id == sid]
        roi = box(*definitions[sid]['bbox_epsg2193'])
        if any(not roi.buffer(5).covers(g) for g in [*fa.geometry, *fb.geometry]):
            raise ValueError('Input geometry exceeds site guard')
        at = [(r.label_id, r.geometry) for _, r in fa.iterrows() if r['class'] == 'tree']
        bt = [(r.label_id, r.geometry) for _, r in fb.iterrows() if r['class'] == 'tree']
        for threshold in policy['iou_thresholds']:
            methods[f'IoU >= {threshold:.2f}'].append(dict(site_id=sid, n_a=len(at), n_b=len(bt),
                pairs=overlap_matches(at, bt, threshold)))
        methods['Smaller-area overlap >= 0.50'].append(dict(site_id=sid, n_a=len(at), n_b=len(bt),
            pairs=overlap_matches(at, bt, policy['smaller_area_overlap_threshold'], smaller=True)))
        location = score_site(at, bt, tolerance=policy['centroid_tolerance_m'], minimum_iou=0, ignore_rule='none')
        methods['Centroid distance <= 2 m'].append(dict(site_id=sid, n_a=len(at), n_b=len(bt),
            pairs=[dict(id_a=m['reference_id'], id_b=m['prediction_id'], iou=m['iou'],
                centroid_distance_m=m['centroid_distance_m']) for m in location['matches']]))
        ar = [(r.label_id, r.geometry.intersection(roi)) for _, r in fa.iterrows() if r['class'] in policy['region_classes']]
        br = [(r.label_id, r.geometry.intersection(roi)) for _, r in fb.iterrows() if r['class'] in policy['region_classes']]
        ua = union_for_classes(fa, set(policy['region_classes'])).intersection(roi)
        ub = union_for_classes(fb, set(policy['region_classes'])).intersection(roi)
        edges, da, db = region_links(ar, br, policy['smaller_area_overlap_threshold'])
        links += [dict(site_id=sid, **e) for e in edges]
        for who, objects, other in [('XZ', ar, ub), ('TW', br, ua)]:
            for lid, geom in objects:
                fraction = geom.intersection(other).area / geom.area if geom.area else None
                object_coverage.append(dict(site_id=sid, rater=who, label_id=lid, area_m2=geom.area,
                    fraction_covered_by_other_union=fraction,
                    majority_covered=fraction is not None and fraction >= policy['covered_object_fraction_threshold']))
        regions.append(dict(site_id=sid, **area_statistics(ua.area, ub.area, ua.intersection(ub).area),
            n_a=len(ar), n_b=len(br), linked_a=sum(v > 0 for v in da.values()), linked_b=sum(v > 0 for v in db.values()),
            multiply_linked_a=sum(v > 1 for v in da.values()), multiply_linked_b=sum(v > 1 for v in db.values()),
            degree_a=da, degree_b=db))
        for d in policy['area_tolerances_m']:
            tolerance_rows[d].append(dict(site_id=sid, **tolerant_score(tolerant_area(ua, ub, roi, d))))
        canvas = Image.new('RGB', (840, 900), 'white'); draw = ImageDraw.Draw(canvas)
        draw.text((10,8),sid+' region comparison | cyan XZ, magenta TW | white intersect, yellow XZ-only, pink TW-only',fill='black')
        for j, kind in enumerate(['rgb', 'chm']):
            chip = Image.open(ROOT / f'data/sites/chips/{sid}/{kind}.png').convert('RGB')
            dr = ImageDraw.Draw(chip)
            for geom, colour in [(ua, '#00ffff'), (ub, '#ff00ff')]:
                if not geom.is_empty: intake.draw_geometry(dr, geom, definitions[sid]['bbox_epsg2193'], (0,0), 1, colour)
            canvas.paste(chip, (10+420*j, 35))
        mask = Image.new('RGBA', (400,400), (25,35,45,255)); dm = ImageDraw.Draw(mask)
        x0,y0,x1,y1 = definitions[sid]['bbox_epsg2193']
        for geom, colour in [(ua.difference(ub), '#f4c542'), (ub.difference(ua), '#e65f9d'), (ua.intersection(ub), '#e9f1f3')]:
            parts = [geom] if geom.geom_type == 'Polygon' else list(getattr(geom, 'geoms', []))
            for p in parts:
                if p.geom_type != 'Polygon': continue
                convert = lambda ring: [((x-x0)*400/(x1-x0)-.5, (y1-y)*400/(y1-y0)-.5) for x,y in ring.coords]
                # Draw each geometry to its own mask so holes never erase other classes.
                layer = Image.new('RGBA',(400,400),(0,0,0,0)); dl = ImageDraw.Draw(layer)
                dl.polygon(convert(p.exterior),fill=colour)
                for hole in p.interiors: dl.polygon(convert(hole),fill=(0,0,0,0))
                mask.alpha_composite(layer)
        canvas.paste(mask.convert('RGB'), (10, 470))
        draw.text((435, 480),'Union coverage; internal tree divisions removed.',fill='black')
        draw.text((435, 510),'Dark: unlabelled by both (not a true-negative count).',fill='black')
        draw.text((435, 540),'Uncertain/non-tree polygons do not create ignore masks.',fill='black')
        canvas.save(gallery / f'{sid}_regions.png')
    summaries = [object_summary(rows, name) for name, rows in methods.items()]
    pooled_area = area_statistics(sum(r['area_a_m2'] for r in regions), sum(r['area_b_m2'] for r in regions),
        sum(r['intersection_m2'] for r in regions))
    pooled_tolerance = []
    for d, rows in tolerance_rows.items():
        fields = ['area_a_m2','area_b_m2','supported_area_a_m2','supported_area_b_m2']
        pooled_tolerance.append(tolerant_score(dict(tolerance_m=d, **{f:sum(r[f] for r in rows) for f in fields})))
    majority = {who: dict(total=sum(r['rater']==who for r in object_coverage),
        majority_covered=sum(r['rater']==who and r['majority_covered'] for r in object_coverage)) for who in ['XZ','TW']}
    result = dict(policy=policy, object_sensitivity=summaries, region_per_site=regions, region_pooled=pooled_area,
        tolerant_area_per_site=tolerance_rows, tolerant_area_pooled=pooled_tolerance,
        majority_coverage=majority, region_links=links, independent_blind_agreement=False,
        posthoc_exploratory=True, original_matching_policy_changed=False, labels_modified=False, reference_frozen=False)
    write_json(out / 'metrics.json', result)
    with (out / 'region_object_coverage.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(object_coverage[0]));writer.writeheader();writer.writerows(object_coverage)
    write_json(out / 'region_links.json', links)
    lines = ['# Exploratory annotation correspondence sensitivity', '',
        f'I retained the original rater protocol and compared the supplied {global_trees}-tree revision with the original TW labels. These are post-revision, post-hoc diagnostics. All tested settings are reported; none certifies passing or reference freeze.', '',
        '## Individual-tree object sensitivity', '', '| Method | XZ | TW | Matched | F1 | Matched median IoU |', '|---|---:|---:|---:|---:|---:|']
    for r in summaries: lines.append(f"| {r['method']} | {r['n_a']} | {r['n_b']} | {r['matched']} | {r['f1']:.4f} | {r['median_matched_iou']:.4f} |")
    lines += ['', 'IoU matching is one-to-one using descending overlap and stable IDs. Smaller-area overlap means intersection / min(area A, area B); containment can score 1 even when full crown extents disagree substantially. The centroid view reuses the benchmark 2 m candidate filter with descending IoU, then distance and IDs, with no ignore regions and no minimum IoU. It tests location, not accurate delineation.', '',
        '## Per-site object results', '', '| Method | Site | Matched | XZ / TW | F1 |', '|---|---|---:|---:|---:|']
    for summary in summaries:
        for row in summary['per_site']:
            lines.append(f"| {summary['method']} | {row['site_id']} | {len(row['pairs'])} | {row['n_a']} / {row['n_b']} | {2*len(row['pairs'])/(row['n_a']+row['n_b']):.4f} |")
    lines += ['', '## Crown-region agreement', '',
        'Tree and tree_group polygons are unioned separately for each rater and clipped to the fixed site domain. Internal instance divisions disappear. Uncertain and not_tree polygons are excluded as classes; confirmed trees are not masked by uncertainty regions.', '',
        '| Site | Region IoU | Region Dice | XZ area (m²) | TW area (m²) |', '|---|---:|---:|---:|---:|']
    for row in [*regions,dict(site_id='Pooled',**pooled_area)]:
        lines.append(f"| {row['site_id']} | {row['coverage_iou']:.4f} | {row['coverage_dice']:.4f} | {row['area_a_m2']:.2f} | {row['area_b_m2']:.2f} |")
    lines += ['', '## Region agreement allowing boundary displacement', '',
        'For tolerance d: A-support = area(A ∩ buffer(B,d))/area(A); B-support = area(B ∩ buffer(A,d))/area(B). The reported score is their harmonic mean. At d = 0 it equals region Dice. It is not buffered IoU, tree F1 or the predeclared matched median IoU. Tolerance is uniform for both raters and all regions, with no disagreement-specific masks.', '',
        '| Tolerance (m) | XZ area supported | TW area supported | Tolerant area score |', '|---|---:|---:|---:|']
    for row in pooled_tolerance:
        lines.append(f"| {row['tolerance_m']} | {row['fraction_a_supported']:.4f} | {row['fraction_b_supported']:.4f} | {row['tolerant_area_f']:.4f} |")
    lines += ['', '## Many-to-many region support', '',
        'An overlap link covers at least 50% of the smaller polygon. Several single trees may link to the same tree_group. Link counts are not true positives and no detection F1 is computed from them.', '',
        '| Site | Linked XZ regions / all | Linked TW regions / all | XZ regions with multiple links | TW regions with multiple links |', '|---|---:|---:|---:|---:|']
    for r in regions: lines.append(f"| {r['site_id']} | {r['linked_a']} / {r['n_a']} | {r['linked_b']} / {r['n_b']} | {r['multiply_linked_a']} | {r['multiply_linked_b']} |")
    lines += ['', 'The union-coverage view also allows several counterparts to cover one polygon. Polygons at least 50% covered by the other rater’s union:']
    for who,r in majority.items(): lines.append(f"- {who}: {r['majority_covered']} / {r['total']}.")
    lines += ['', '## Basis and limits', '',
        '[Weinstein et al., NEON canopy-crown benchmark](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009180) distinguishes location detection from crown delineation and reports IoU-threshold sensitivity. Its 0.4 threshold motivates a sensitivity setting here, not a new acceptance standard. Lower 0.3/0.25 thresholds, containment links and buffered area scores are explicit project diagnostics. Pooled scoring and ≥ threshold inclusion retain this project’s conventions.', '',
        'The 1 m tolerance is tied to the delivered CHM grid; the 2 m sensitivity scale is tied to the existing project location tolerance. Neither proves that these errors are acceptable for individual-tree geometry. The larger the tolerance, the more mismatched boundaries can count as supported.', '',
        'Both revised labels and alternative diagnostics were developed after viewing TW labels; they cannot replace original independent agreement. Two sites do not establish all-strata reliability. No model outputs, annotations, primary evaluation policies or reference freeze state were changed.', '']
    (out / 'report.md').write_text('\n'.join(lines))
    object_html = ''.join(f"<tr><td>{r['method']}</td><td>{r['matched']}</td><td>{r['f1']:.4f}</td><td>{r['median_matched_iou']:.4f}</td></tr>" for r in summaries)
    area_html = ''.join(f"<tr><td>{r['tolerance_m']}</td><td>{r['fraction_a_supported']:.4f}</td><td>{r['fraction_b_supported']:.4f}</td><td>{r['tolerant_area_f']:.4f}</td></tr>" for r in pooled_tolerance)
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Annotation correspondence: object and region sensitivity</title>'
        '<style>body{font:16px system-ui;max-width:1050px;margin:30px auto;padding:0 20px;line-height:1.6}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%}.note{padding:15px;background:#edf4fa}</style>'
        f'<h1>Object vs crown-region correspondence</h1><p>S016 / S020 · {global_trees}-tree revision vs original TW</p>'
        '<p class="note">Exploratory sensitivity after viewing TW labels. All settings are shown. Original independent results and matching rules remain unchanged. No automatic pass or reference freeze.</p>'
        '<h2>Individual-tree correspondence: 48 XZ / 58 TW</h2><table><tr><th>Method</th><th>Matched</th><th>F1</th><th>Matched median IoU</th></tr>'+object_html+'</table>'
        '<p>Lower IoU accepts more boundary disagreement. Smaller-area overlap accepts containment. The 2 m centroid view tests location. Every method still uses one-to-one assignment.</p>'
        f'<h2>Crown-region agreement</h2><p>Tree + tree_group union: exact IoU <b>{pooled_area["coverage_iou"]:.4f}</b>, Dice <b>{pooled_area["coverage_dice"]:.4f}</b>. Internal tree divisions disappear.</p>'
        '<table><tr><th>Boundary tolerance (m)</th><th>XZ area supported</th><th>TW area supported</th><th>Tolerant area score</th></tr>'+area_html+'</table>'
        '<p>Area-support scores use each original region as the denominator. No uncertainty masks. A higher buffered score means proximity of labelled canopy, not correct tree counts.</p>'
        '<p><a href="report.md">Definitions, all per-site results and research basis</a> · <a href="region_object_coverage.csv">Coverage of each region</a> · <a href="metrics.json">All pair and region metrics</a></p>'
        + ''.join(f'<h2>{sid}: region overlay</h2><img src="visual_review/{sid}_regions.png">' for sid in policy['sites']))
    receipt = dict(run_id=started.strftime('%Y%m%dT%H%M%SZ')+'_annotation_sensitivity',
        started_utc=started.isoformat(), completed_utc=datetime.now(timezone.utc).isoformat(),
        exit_code=0, xz_sha256=sha256(args.xz_json), tw_sha256=sha256(args.tim_json),
        policy_sha256=sha256(config), script_sha256=sha256(Path(__file__)),
        command=['scripts/81_analyse_annotation_sensitivity.py','--xz-json',str(args.xz_json),
                 '--tim-json',str(args.tim_json),'--output',str(args.output)],
        primary_policy_changed=False, annotations_modified=False, reference_frozen=False)
    write_json(out / 'receipt.json',receipt)
    write_json(out / 'sha256_manifest.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps(dict(run_id=receipt['run_id'],objects=[{k:v for k,v in r.items() if k!='per_site'} for r in summaries],
        region=pooled_area,tolerant_area=pooled_tolerance,majority_coverage=majority),indent=2))


if __name__ == '__main__':
    main()
