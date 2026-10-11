#!/usr/bin/env python
"""Compare revised XZ labels with original TW labels without replacing blind results."""
import argparse
import csv
import html
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry import box
from benchmark.common import ROOT, sha256, site_definitions, write_json
from benchmark.annotation_coverage import coverage_agreement

spec = importlib.util.spec_from_file_location('intake', ROOT / 'scripts/31_receive_independent_labels.py')
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)
agreement = intake.agreement


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xz-json', type=Path, required=True)
    ap.add_argument('--tim-json', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Existing results preserved; choose a fresh output directory')
    policies = [ROOT / 'config/independent_agreement.json', ROOT / 'config/independent_coverage.json']
    policy, coverage_policy = [json.loads(p.read_text()) for p in policies]
    sites = site_definitions()
    required = policy['sites']
    a, completed_a = agreement.load_independent_json(args.xz_json, 'XZ', sites, allow_revision_attribution=True)
    b, completed_b = agreement.load_independent_json(args.tim_json, 'TW', sites)
    if not set(required) <= completed_a & completed_b:
        ap.error('Both exports must explicitly complete S016 and S020')
    a = a[a.site_id.isin(required)].copy().reset_index(drop=True)
    b = b[b.site_id.isin(required)].copy().reset_index(drop=True)
    geometry_audit = []
    for who, frame in [('XZ', a), ('TW', b)]:
        for _, row in frame.iterrows():
            roi = box(*sites[row.site_id]['bbox_epsg2193'])
            if not roi.buffer(policy['coordinate_guard_margin_m']).covers(row.geometry):
                raise ValueError(f'{row.label_id}: outside coordinate guard')
            geometry_audit.append(dict(rater=who, site_id=row.site_id, label_id=row.label_id,
                used_area_m2=row.geometry.area, interior_rings=sum(len(p.interiors) for p in
                ([row.geometry] if row.geometry.geom_type == 'Polygon' else row.geometry.geoms)),
                outside_site_area_m2=row.geometry.difference(roi).area))
    out = args.output
    out.mkdir(parents=True)
    (out / 'inputs').mkdir()
    for src, name in [(args.xz_json, 'XZ_revised.json'), (args.tim_json, 'TW_original.json')]:
        dest = out / 'inputs' / name
        shutil.copyfile(src, dest)
        dest.chmod(0o444)
    for p in policies:
        shutil.copyfile(p, out / p.name)
    cmd = [sys.executable, str(ROOT / 'scripts/05_interrater_agreement.py'), '--a', 'XZ', '--b', 'TW',
        '--json-a', str(out / 'inputs/XZ_revised.json'), '--json-b', str(out / 'inputs/TW_original.json'),
        '--sites', *required, '--iou', str(policy['iou_threshold']), '--post-revision',
        '--output', str(out / 'agreement.md')]
    started = datetime.now(timezone.utc)
    process = subprocess.run(cmd, text=True, capture_output=True)
    (out / 'calculation.log').write_text(process.stdout + process.stderr)
    write_json(out / 'command.json', cmd)
    if process.returncode:
        raise RuntimeError('Calculation failed; see calculation.log')
    metrics = json.loads((out / 'agreement_provenance.json').read_text())
    frames = {sid: (a[a.site_id == sid].reset_index(drop=True),
                    b[b.site_id == sid].reset_index(drop=True)) for sid in required}
    coverage = coverage_agreement(frames, sites, coverage_policy)
    write_json(out / 'canopy_agreement.json', coverage)
    write_json(out / 'geometry_audit.json', geometry_audit)
    original_dir = ROOT / 'outputs/independent_agreement/TW_2026-10-09_original_comparison'
    original = json.loads((original_dir / 'metrics.json').read_text())
    original_coverage = json.loads((original_dir / 'canopy_agreement.json').read_text())
    primary = metrics['per_class']['tree']
    previous = original['per_class']['tree']
    export_data = json.loads(args.xz_json.read_text())
    global_trees = sum(l['cls']=='tree' for rec in export_data['sites'].values() for l in rec.get('labels', []))
    previous_revision = ROOT / 'data/labels/crown_labels_XZ_2026-10-09_expanded.json'
    prior_data = json.loads(previous_revision.read_text()) if previous_revision.exists() else None
    same_as_prior = prior_data is not None and all(export_data['sites'][sid]['labels']==prior_data['sites'][sid]['labels'] for sid in required)
    cases = []
    for sid, (fa, fb) in frames.items():
        fa = fa[fa['class'] == 'tree'].reset_index(drop=True)
        fb = fb[fb['class'] == 'tree'].reset_index(drop=True)
        matched, ua, ub = agreement.greedy_match(list(fa.geometry), list(fb.geometry),
            policy['iou_threshold'], list(fa.label_id), list(fb.label_id))
        cases += [dict(site_id=sid, id_a=fa.label_id[i], id_b=fb.label_id[j],
            status='matched_tree', iou=value) for i, j, value in matched]
        cases += [dict(site_id=sid, id_a=fa.label_id[i], id_b='', status='only_XZ', iou='')
            for i in range(len(fa)) if i not in ua]
        cases += [dict(site_id=sid, id_a='', id_b=fb.label_id[j], status='only_TW', iou='')
            for j in range(len(fb)) if j not in ub]
    with (out / 'tree_pairs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['site_id', 'id_a', 'id_b', 'status', 'iou'])
        writer.writeheader()
        writer.writerows(cases)
    intake.build_gallery(out, a[a['class'] == 'tree'], b[b['class'] == 'tree'],
        cases, {sid: sites[sid] for sid in required}, 'TW')
    page = out / 'visual_review/index.html'
    text = page.read_text().replace('Independent annotation adjudication', 'Revised annotation correspondence')
    text = text.replace('Original independent comparison', 'Revised XZ / original TW: single-tree correspondence')
    text = text.replace('independent originals', 'revised XZ / original TW')
    text = text.replace('Cyan: original XZ. Magenta: original teacher. No predictions or refined XZ labels. All cases need manual review; images do not grant adjudication or freeze.',
        'Cyan: revised XZ. Magenta: original TW. Only tree polygons are shown. No model predictions. Uncertain labels are excluded from this tree comparison; confirmed trees are not masked by uncertainty regions. This follows inspection of TW labels and is not independent blind agreement. <a href="../index.html">Back to results</a>.')
    page.write_text(text)
    rows = []
    md = ['# Post-revision annotation correspondence', '',
        f'I compared the {global_trees}-tree export ({args.xz_json.name}) with the unchanged original TW export on S016 and S020. This revision follows inspection of TW labels; I retain the original blind comparison separately.', '',
        'Matching is unchanged: class-restricted tree matching, greedy descending IoU with stable ID ties, IoU ≥ 0.50, one-to-one; F1 = 2M / (N_XZ + N_TW). Median IoU uses matched trees only.', '',
        '| Site | Revised XZ trees | TW trees | Matched | Only XZ | Only TW | Original F1 | Revised F1 | Revised matched median IoU |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for now, old in zip([*primary['per_site'], dict(site_id='Pooled', **primary)],
                        [*previous['per_site'], dict(site_id='Pooled', **previous)]):
        values = [now['site_id'], now['n_a'], now['n_b'], now['matched'],
                  now['n_a'] - now['matched'], now['n_b'] - now['matched'],
                  f"{old['f1']:.4f}", f"{now['f1']:.4f}", f"{now['median_iou']:.4f}"]
        md.append('| ' + ' | '.join(map(str, values)) + ' |')
        rows.append('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in values) + '</tr>')
    ca = coverage['pooled_area']
    md += ['', f"The original comparison matched {previous['matched']} of {previous['n_a']}/{previous['n_b']} trees. The revision matches {primary['matched']} of {primary['n_a']}/{primary['n_b']}. The match-count difference is {primary['matched']-previous['matched']}; the XZ tree-count difference is {primary['n_a']-previous['n_a']}. Both counts affect the F1 denominator and numerator. This does not establish whether unmatched trees are incorrect, omitted by the other annotator, or differently delineated.", '',
        'TW has 14 uncertain objects. These are excluded from tree-class matching, not used to remove confirmed tree discrepancies. Non-tree and group objects remain in the all-class appendix.', '',
        f"Supplementary tree + tree_group union coverage IoU is {ca['coverage_iou']:.4f} (original {original_coverage['pooled_area']['coverage_iou']:.4f}); coverage Dice is {ca['coverage_dice']:.4f}. These extent measures do not replace instance agreement.", '',
        '| Boundary tolerance (m) | Supplementary pooled boundary F1 |', '|---|---:|']
    md += [f"| {r['tolerance_m']} | {r['boundary_f1']:.4f} |" for r in coverage['pooled_boundary']]
    md += ['', 'Matched-class kappa is 0.0000 in the all-class appendix: TW matched classes have a constant tree marginal. In the tree-only view kappa is undefined because both matched classes are always tree. Neither is a useful broad classification-reliability measure here.', '',
        'No rater-agreement passing threshold was predeclared. Model acceptance thresholds are not rater acceptance thresholds. This calculation does not freeze the reference or adjudicate disagreement.', '',
        ('S016/S020 objects and geometry are unchanged from the preceding 362-tree export.' if same_as_prior else 'The supplied revision changes S016/S020 relative to the preceding 362-tree export; the current source hash identifies the compared version.'), '',
        'I retained original JSON attribution and polygon holes. Input snapshots, hashes, the exact command, policies, all-class results and all tree pairs accompany this report.', '',
        '[RGB/CHM review](visual_review/index.html) · [All-class appendix](agreement.md) · [Tree pairs](tree_pairs.csv)', '']
    (out / 'report.md').write_text('\n'.join(md))
    headers = ['Site', 'XZ trees', 'TW trees', 'Matches', 'Only XZ', 'Only TW', 'Original F1', 'Revised F1', 'Median IoU']
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Post-revision annotation correspondence</title>'
        '<style>body{font:16px system-ui;max-width:1120px;margin:32px auto;padding:0 20px;line-height:1.6}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.note{background:#f0f5fa;padding:16px}img{max-width:100%}</style>'
        f'<h1>Revised XZ / original TW</h1><p>{args.xz_json.name} · {global_trees}-tree export · S016 and S020</p>'
        '<p class="note">Post-revision correspondence after inspection of TW labels. The original independent blind results remain preserved. Tree-only matching: IoU ≥ 0.50, one-to-one; uncertainty excluded as a class, with no new exclusion masks.</p>'
        '<table><tr>' + ''.join('<th>' + h + '</th>' for h in headers) + '</tr>' + ''.join(rows) + '</table>'
        f'<p>{primary["matched"]} matches, compared with {previous["matched"]} originally. Tree counts and matches both affect F1. These figures do not identify which annotator is correct.</p>'
        f'<p>Supplementary canopy union IoU: <b>{ca["coverage_iou"]:.4f}</b>; Dice: <b>{ca["coverage_dice"]:.4f}</b>. Instance boundaries are removed in this extent view.</p>'
        '<p>No automatic pass or reference freeze.</p>'
        f'<p><a href="visual_review/index.html">Review all {primary["matched"]} matched pairs and {primary["n_a"]+primary["n_b"]-2*primary["matched"]} unmatched records with RGB/CHM</a> · <a href="report.md">Report</a> · <a href="tree_pairs.csv">Pair table</a> · <a href="agreement.md">All-class appendix</a></p>'
        '<h2>S016: tree-only overview</h2><img src="visual_review/S016_overview.png">'
        '<h2>S020: tree-only overview</h2><img src="visual_review/S020_overview.png">')
    receipt = dict(run_id=started.strftime('%Y%m%dT%H%M%SZ') + '_postrevision_correspondence',
        started_utc=started.isoformat(), completed_utc=datetime.now(timezone.utc).isoformat(), exit_code=0,
        xz_source=str(args.xz_json.resolve()), xz_sha256=sha256(args.xz_json),
        tw_source=str(args.tim_json.resolve()), tw_sha256=sha256(args.tim_json),
        comparison_kind='post_revision_correspondence', independent_blind_agreement=False,
        reference_frozen=False, labels_modified=False, command=cmd,
        code_sha256={p.name: sha256(p) for p in [Path(__file__), ROOT / 'scripts/05_interrater_agreement.py',
            ROOT / 'scripts/31_receive_independent_labels.py', ROOT / 'scripts/benchmark/annotation_coverage.py']})
    write_json(out / 'receipt.json', receipt)
    write_json(out / 'metrics.json', dict(primary_tree=primary, original_tree=previous,
        supplementary_coverage=coverage, all_class=metrics, **receipt))
    write_json(out / 'sha256_manifest.json', {str(p.relative_to(out)): sha256(p)
        for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps(dict(run_id=receipt['run_id'], primary_tree=primary,
        coverage_iou=ca['coverage_iou'], output=str(out)), indent=2))


if __name__ == '__main__':
    main()
