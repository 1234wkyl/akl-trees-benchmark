#!/usr/bin/env python
"""Inter-rater agreement between two labellers on the same sites.

Tim double-labels a subset. This measures how far apart you are, which is what
makes the reference set defensible. Report the numbers even if they are poor;
low agreement is a finding about task difficulty, not a failure.

    python scripts/05_interrater_agreement.py --a XZ --b TW

Matching rule: two crowns are the same tree when they overlap with IoU above
the threshold, matched one-to-one by greedy best IoU. Report the threshold you
used, because agreement depends on it.

Writes data/exports/interrater_agreement.md
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kit import KIT_ROOT, load_config, out_dir  # noqa: E402


def iou(a, b) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def greedy_match(ga, gb, thresh, ids_a=None, ids_b=None):
    """One-to-one greedy matching on best IoU."""
    pairs = []
    for i, a in enumerate(ga):
        for j, b in enumerate(gb):
            v = iou(a, b)
            if v >= thresh:
                pairs.append((v, i, j))
    ids_a=ids_a or [str(i) for i in range(len(ga))]
    ids_b=ids_b or [str(i) for i in range(len(gb))]
    pairs.sort(key=lambda p:(-p[0],str(ids_a[p[1]]),str(ids_b[p[2]])))
    used_a, used_b, matched = set(), set(), []
    for v, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j, v))
    return matched, used_a, used_b


def class_kappa(class_pairs):
    """Kappa of matched classes; undefined when expected agreement is one."""
    if not class_pairs:
        return None
    labels = {c for pair in class_pairs for c in pair}
    n = len(class_pairs)
    observed = sum(a == b for a, b in class_pairs) / n
    expected = sum(
        sum(a == c for a, _ in class_pairs) / n
        * sum(b == c for _, b in class_pairs) / n for c in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else None


def format_stat(value):
    return f"{value:.3f}" if value is not None else "Undefined"


def per_class_agreement(frames, thresh, classes):
    """Per-class agreement using class-restricted matching.

    Objects of one class are matched only against objects of the same class,
    the same convention used for per-class scoring in instance segmentation.
    A tree called a group by the other labeller therefore counts as unmatched
    here; that confusion is reported separately by the class crosstab.
    """
    out = {}
    for cls in classes:
        tot_m = tot_a = tot_b = 0
        ious = []
        per_site = []
        for sid, (ga, gb) in frames.items():
            ia = [i for i in range(len(ga)) if ga['class'][i] == cls]
            ib = [j for j in range(len(gb)) if gb['class'][j] == cls]
            matched, ua, ub = greedy_match(
                [ga.geometry[i] for i in ia], [gb.geometry[j] for j in ib], thresh,
                [ga.label_id[i] for i in ia],[gb.label_id[j] for j in ib])
            tot_m += len(matched)
            tot_a += len(ia)
            tot_b += len(ib)
            ious += [v for _, _, v in matched]
            per_site.append({'site_id':sid,'n_a':len(ia),'n_b':len(ib),'matched':len(matched),
                'f1':2*len(matched)/(len(ia)+len(ib)) if len(ia)+len(ib) else None,
                'median_iou':median([v for _,_,v in matched]) if matched else None})
        out[cls] = {
            'n_a': tot_a, 'n_b': tot_b, 'matched': tot_m,
            'f1': 2 * tot_m / (tot_a + tot_b) if tot_a + tot_b else None,
            'median_iou': median(ious) if ious else None,
            'per_site': per_site,
        }
    return out


def class_crosstab(class_pairs):
    """Confusion counts over pooled matched pairs, ordered by incidence."""
    counts = {}
    for a, b in class_pairs:
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def load_independent_json(path, who, sites, allow_revision_attribution=False):
    """Validate a single independent export; keep completed zero-object sites."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    import numpy as np
    data = json.loads(path.read_text())
    if not isinstance(data,dict) or not isinstance(data.get('sites'),dict):
        raise ValueError(f'{path.name}: JSON object with sites mapping required')
    if data.get('schema_version') != 'akl-trees-crown-labels/v1' or data.get('crs') != 'EPSG:2193':
        raise ValueError(f'{path.name}: unsupported schema or CRS')
    if data.get('labeller') != who:
        raise ValueError(f'{path.name}: expected labeller {who}')
    completed, records, geometries, seen = set(), [], [], set()
    for sid, rec in data.get('sites', {}).items():
        if not isinstance(rec,dict) or not isinstance(rec.get('labels',[]),list):
            raise ValueError(f'{path.name}: site record and labels list required')
        if sid not in sites:
            if sid=='undefined' and rec.get('complete') is False and not rec.get('labels'):
                continue  # Known empty non-site key in the original XZ browser export.
            raise ValueError(f'{path.name}: unknown site {sid}')
        if rec.get('complete') is not True:
            continue
        completed.add(sid)
        for label in rec.get('labels', []):
            if not isinstance(label,dict):raise ValueError(f'{path.name}: label object required')
            lid = label.get('id')
            if not isinstance(lid,str) or not lid or lid in seen:
                raise ValueError(f'{path.name}: missing/duplicate label ID')
            seen.add(lid)
            allowed_attributions={None,'',who}
            if allow_revision_attribution and who=='XZ':
                allowed_attributions.add('Codex_AI_refinement_of_XZ')
            if label.get('labeller') not in allowed_attributions:
                raise ValueError(f'{path.name}: mixed labellers')
            if label.get('cls') not in {'tree','tree_group','uncertain','not_tree'}:
                raise ValueError(f'{path.name}: unknown class')
            coords=np.asarray(label.get('world'),dtype=float)
            if coords.ndim!=2 or coords.shape[1]!=2 or len(coords)<3 or not np.isfinite(coords).all():
                raise ValueError(f'{path.name}: finite planar world coordinates required for {lid}')
            holes=label.get('holes_world',[])
            if not isinstance(holes,list):raise ValueError(f'{path.name}: hole list required for {lid}')
            for hole in holes:
                h=np.asarray(hole,dtype=float)
                if h.ndim!=2 or h.shape[1]!=2 or len(h)<3 or not np.isfinite(h).all():
                    raise ValueError(f'{path.name}: finite polygon hole coordinates required for {lid}')
            poly = Polygon(coords,holes=holes)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or not poly.is_valid:
                raise ValueError(f'{path.name}: empty/invalid geometry {lid}')
            records.append(dict(site_id=sid,label_id=lid,labeller=who,**{'class':label['cls']}))
            geometries.append(poly)
    return gpd.GeoDataFrame(records, columns=['site_id','label_id','labeller','class'], geometry=geometries, crs='EPSG:2193'), completed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="labeller A, e.g. XZ")
    ap.add_argument("--b", required=True, help="labeller B, e.g. TW")
    ap.add_argument("--iou", type=float, default=0.5, help="match threshold")
    ap.add_argument("--input", type=Path, help="separate combined-rater GeoPackage; does not change the reference")
    ap.add_argument("--output", type=Path, help="agreement report path")
    ap.add_argument('--json-a', type=Path, help='original independent JSON for labeller A')
    ap.add_argument('--json-b', type=Path, help='original independent JSON for labeller B')
    ap.add_argument('--sites', nargs='+', help='explicit sites to compare, including completed zero-object sites')
    ap.add_argument('--post-revision',action='store_true',help='Non-independent correspondence after author revision; retain recorded author refinement attribution')
    args = ap.parse_args()
    if not 0 < args.iou <= 1:
        ap.error("--iou must be greater than zero and at most one")
    if bool(args.json_a) != bool(args.json_b) or (args.input and args.json_a):
        ap.error('provide both --json-a/--json-b, or --input, not both modes')
    if args.a == args.b:
        ap.error('independent labellers must be distinct')

    cfg = load_config()
    import geopandas as gpd

    if args.json_a:
        definition = json.loads((KIT_ROOT/'data/sites/reference_sites.json').read_text())
        site_definitions = {s['site_id']:s for s in definition['sites']}
        A, complete_a = load_independent_json(args.json_a, args.a, site_definitions,allow_revision_attribution=args.post_revision)
        B, complete_b = load_independent_json(args.json_b, args.b, site_definitions)
        shared = sorted(complete_a & complete_b)
        sources = [args.json_a, args.json_b]
    else:
        src = args.input or out_dir('exports')/'reference_crowns.gpkg'
        if not src.exists():
            print('Reference input missing. Export labels or supply --json-a/--json-b.')
            return 1
        gdf = gpd.read_file(src, layer='reference_crowns')
        if gdf.crs is None or gdf.crs.to_epsg() != 2193:
            raise ValueError('agreement input must use EPSG:2193')
        A, B = gdf[gdf.labeller == args.a], gdf[gdf.labeller == args.b]
        shared = sorted(set(A.site_id) & set(B.site_id))
        sources = [src]
    if args.sites:
        if not set(args.sites) <= set(shared):
            ap.error('each requested site must be independently completed by both labellers')
        shared = sorted(set(args.sites))
    if not shared:
        print("No sites labelled by both people.")
        return 1

    print(f"Comparing {args.a} vs {args.b} on {len(shared)} shared site(s), IoU >= {args.iou}\n")

    tot_m = tot_a = tot_b = 0
    ious, class_pairs = [], []
    per_site, pair_rows = [], []
    frames = {}
    for sid in shared:
        ga = A[A.site_id == sid].reset_index(drop=True)
        gb = B[B.site_id == sid].reset_index(drop=True)
        frames[sid] = (ga, gb)
        matched, ua, ub = greedy_match(list(ga.geometry), list(gb.geometry), args.iou,list(ga.label_id),list(gb.label_id))
        tot_m += len(matched)
        tot_a += len(ga)
        tot_b += len(gb)
        for i, j, v in matched:
            ious.append(v)
            class_pairs.append((ga["class"][i], gb["class"][j]))
            pair_rows.append(dict(site_id=sid,status='matched_same_class' if ga['class'][i]==gb['class'][j] else 'matched_class_disagreement',id_a=ga.label_id[i],id_b=gb.label_id[j],class_a=ga['class'][i],class_b=gb['class'][j],iou=v))
        for frame, used, side in [(ga,ua,'a'),(gb,ub,'b')]:
            for i in sorted(set(range(len(frame)))-used,key=lambda i:str(frame.label_id[i])):
                pair_rows.append(dict(site_id=sid,status='only_'+side,id_a=frame.label_id[i] if side=='a' else '',id_b=frame.label_id[i] if side=='b' else '',class_a=frame['class'][i] if side=='a' else '',class_b=frame['class'][i] if side=='b' else '',iou=''))
        per_site.append({
            "site": sid, "n_a": len(ga), "n_b": len(gb), "matched": len(matched),
            "only_a": len(ga) - len(ua), "only_b": len(gb) - len(ub),
            'agreement_f1':2*len(matched)/(len(ga)+len(gb)) if len(ga)+len(gb) else None,
            'median_iou':median([v for _,_,v in matched]) if matched else None,
            'class_kappa':class_kappa([(ga['class'][i],gb['class'][j]) for i,j,_ in matched]),
        })
        print(f"  {sid}  {args.a}={len(ga):<4} {args.b}={len(gb):<4} "
              f"matched={len(matched):<4} only_{args.a}={len(ga)-len(ua):<3} "
              f"only_{args.b}={len(gb)-len(ub)}")

    # F1-style agreement on detection: symmetric, and the honest headline.
    f1 = 2*tot_m/(tot_a+tot_b) if tot_a+tot_b else None
    med_iou = median(ious) if ious else None

    # Cohen's kappa on the class label, over matched pairs only.
    kappa = class_kappa(class_pairs)

    # Per-class view. The pooled number is dominated by whichever class is most
    # frequent on the site, so it cannot be read as individual-tree agreement.
    classes = ['tree', 'tree_group', 'not_tree', 'uncertain']
    by_class = per_class_agreement(frames, args.iou, classes)
    crosstab = class_crosstab(class_pairs)

    print(f"\n  agreement F1     {format_stat(f1)}   (matched {tot_m} of {tot_a}/{tot_b})")
    print(f"  median IoU       {format_stat(med_iou)}")
    print(f"  class kappa      {format_stat(kappa)}   (on matched objects only)")
    print("\n  per class (class-restricted matching):")
    for cls in classes:
        r = by_class[cls]
        if r['n_a'] + r['n_b'] == 0:
            print(f"    {cls:<11} no objects on either side")
            continue
        print(f"    {cls:<11} {args.a}={r['n_a']:<4} {args.b}={r['n_b']:<4} "
              f"matched={r['matched']:<4} F1={format_stat(r['f1'])} "
              f"median IoU={format_stat(r['median_iou'])}")

    lines = [
        "# Post-revision annotation correspondence" if args.post_revision else "# Inter-rater agreement", "",
        f"Labellers: **{args.a}** and **{args.b}**. Shared sites: {len(shared)}. "
        f"Match rule: one-to-one greedy, IoU >= {args.iou}.", "",
        "| Metric | Value |", "|---|--:|",
        f"| Crowns by {args.a} | {tot_a:,} |",
        f"| Crowns by {args.b} | {tot_b:,} |",
        f"| Matched pairs | {tot_m:,} |",
        f"| Agreement F1 | {format_stat(f1)} |",
        f"| Median IoU of matched objects | {format_stat(med_iou)} |",
        f"| Cohen's kappa on matched classes | {format_stat(kappa)} |",
    ]
    if args.post_revision:
        lines[2:2]=["The author revision follows inspection of returned supervisor labels. This is not independent blind agreement and does not replace the preserved original comparison. Original annotation IDs, geometry and refinement attributions remain in the source JSON.",""]
    lines += ["", "| Site | " + f"{args.a} | {args.b} | matched | only {args.a} | only {args.b} |",
              "|---|--:|--:|--:|--:|--:|"]
    for r in per_site:
        lines.append(f"| {r['site']} | {r['n_a']} | {r['n_b']} | {r['matched']} "
                     f"| {r['only_a']} | {r['only_b']} |")
    lines += ["", "## Per-class agreement (class-restricted matching)", "",
              "Objects of one class are matched only against objects of the same class. "
              "The individual-tree row is the one that corresponds to the benchmark "
              "scoring set; the pooled figure above is dominated by whichever class is "
              "most frequent on these sites and must not be reported as tree agreement.", "",
              "| Class | " + f"{args.a} | {args.b} | matched | F1 | median IoU |",
              "|---|--:|--:|--:|--:|--:|"]
    for cls in classes:
        r = by_class[cls]
        if r['n_a'] + r['n_b'] == 0:
            continue
        lines.append(f"| {cls} | {r['n_a']} | {r['n_b']} | {r['matched']} "
                     f"| {format_stat(r['f1'])} | {format_stat(r['median_iou'])} |")
    if crosstab:
        labels = sorted({c for pair in crosstab for c in pair})
        lines += ["", "## Class confusion on pooled matched pairs", "",
                  "Rows are " + args.a + ", columns are " + args.b + ".", "",
                  "| " + args.a + " \\ " + args.b + " | " + " | ".join(labels) + " |",
                  "|---|" + "--:|" * len(labels)]
        for a in labels:
            lines.append(f"| {a} | " + " | ".join(str(crosstab.get((a, b), 0)) for b in labels) + " |")
    lines += [
        "", "## How to read this", "",
        "Agreement F1 measures repeatability under this annotation and matching protocol; "
        "it is not a mathematical upper bound on model accuracy. This report pools all "
        "label classes, so it must not be described as single-tree model F1. Class kappa "
        "is calculated only on matched pairs and is undefined for a degenerate one-class "
        "comparison. Two sites are a pilot, not a reliability estimate for all strata. "
        "The per-class table restricts matching to one class at a time, so an object one "
        "labeller calls a tree and the other calls a group is unmatched there and appears "
        "in the confusion table instead; use the tree row when the question is how "
        "reproducible individual-tree delineation is, and quote the denominators because "
        "they are small.", "",
        "Crowns matched by only one labeller are the interesting cases. Look at them "
        "before deciding the protocol is finished.",
    ]
    d = out_dir("exports")
    report = args.output or d / "interrater_agreement.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    with report.with_name(report.stem+'_pairs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['site_id','status','id_a','id_b','class_a','class_b','iou'])
        writer.writeheader();writer.writerows(pair_rows)
    metadata={'sources':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
              'sites':shared,'iou_threshold':args.iou,'matching':'one-to-one greedy best IoU, all classes',
              'n_a':tot_a,'n_b':tot_b,'matched':tot_m,'agreement_f1':f1,'median_iou':med_iou,'class_kappa':kappa,
              'per_class':{c:{'n_a':r['n_a'],'n_b':r['n_b'],'matched':r['matched'],
                              'f1':r['f1'],'median_iou':r['median_iou'],'per_site':r['per_site']}
                           for c,r in by_class.items() if r['n_a']+r['n_b']},
              'class_confusion':{f'{a}|{b}':n for (a,b),n in sorted(crosstab.items())},
              'per_site':per_site,
              'matching_tie_rule':'descending IoU then ascending stable A/B label IDs',
              'precision_B_relative_to_A':tot_m/tot_b if tot_b else None,
              'recall_B_relative_to_A':tot_m/tot_a if tot_a else None,
              'directional_metric_note':'Annotation correspondence only; neither rater is declared field truth.',
              'geometry_rule':'buffer(0) repair for invalid JSON polygons; no boundary clipping',
              'canonical_reference_written':False,
              'comparison_kind':'post_revision_correspondence' if args.post_revision else 'original_rater_comparison',
              'independent_blind_agreement':not args.post_revision}
    report.with_name(report.stem+'_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f"\nWrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
