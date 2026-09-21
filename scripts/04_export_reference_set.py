#!/usr/bin/env python
"""Turn exported labels into the reference set, and check them.

Reads every crown_labels_*.json in data/labels/, merges them, writes proper
geospatial files, and runs quality checks that catch the mistakes that are
painful to find later.

    python scripts/04_export_reference_set.py
    python scripts/04_export_reference_set.py --labeller XZ   # one person only

Writes to data/exports/:
    reference_crowns.gpkg      the reference set (QGIS opens this directly)
    reference_crowns.geojson   same, portable
    reference_set_report.md    counts and QA for the report
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kit import KIT_ROOT, load_config, out_dir  # noqa: E402

MIN_SENSIBLE_AREA_M2 = 1.0
MAX_SENSIBLE_AREA_M2 = 900.0


def load_labels(labels_dir: Path, only: str | None):
    files = sorted(labels_dir.glob("crown_labels_*.json"))
    if not files:
        return [], []
    rows, sources = [], []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as exc:
            print(f"  ! skipping {f.name}: {exc}")
            continue
        who = data.get("labeller")
        if only and who != only:
            continue
        sources.append({"file": f.name, "labeller": who,
                        "exported_at": data.get("exported_at")})
        for site_id, rec in (data.get("sites") or {}).items():
            for lab in rec.get("labels", []):
                rows.append({
                    "label_id": lab.get("id"),
                    "site_id": site_id,
                    "class": lab.get("cls"),
                    "area_m2": lab.get("area_m2"),
                    "labeller": lab.get("labeller") or who,
                    "created": lab.get("created"),
                    "site_complete": bool(rec.get("complete")),
                    "world": lab.get("world"),
                })
    return rows, sources


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeller", help="only include this labeller")
    args = ap.parse_args()

    cfg = load_config()
    labels_dir = out_dir("labels")
    rows, sources = load_labels(labels_dir, args.labeller)

    if not rows:
        print(f"No labels found in {labels_dir}")
        print("Export from the labeller and save the file there, then re-run.")
        return 1

    sites_file = out_dir("sites") / "reference_sites.json"
    sites = {s["site_id"]: s for s in json.loads(sites_file.read_text())["sites"]}

    print(f"Loaded {len(rows):,} labels from {len(sources)} export file(s)")
    for s in sources:
        print(f"  {s['file']}  labeller={s['labeller']}  exported={s['exported_at']}")

    import geopandas as gpd
    from shapely.geometry import Polygon

    records, geoms, bad = [], [], []
    for r in rows:
        world = r.pop("world")
        if not world or len(world) < 3:
            bad.append((r["label_id"], "fewer than 3 vertices"))
            continue
        poly = Polygon(world)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            bad.append((r["label_id"], "empty geometry"))
            continue
        site = sites.get(r["site_id"], {})
        r["stratum"] = site.get("stratum")
        r["split"] = site.get("split")
        r["area_m2_geom"] = round(poly.area, 2)
        records.append(r)
        geoms.append(poly)

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=cfg["crs"])

    d = out_dir("exports")
    gdf.to_file(d / "reference_crowns.gpkg", layer="reference_crowns", driver="GPKG")
    gdf.to_file(d / "reference_crowns.geojson", driver="GeoJSON")

    # ---- quality checks ----
    issues: list[str] = []
    for lid, why in bad:
        issues.append(f"dropped {lid}: {why}")

    tiny = gdf[gdf.area_m2_geom < MIN_SENSIBLE_AREA_M2]
    huge = gdf[gdf.area_m2_geom > MAX_SENSIBLE_AREA_M2]
    if len(tiny):
        issues.append(f"{len(tiny)} crown(s) under {MIN_SENSIBLE_AREA_M2} m2, likely stray clicks")
    if len(huge):
        issues.append(f"{len(huge)} crown(s) over {MAX_SENSIBLE_AREA_M2} m2, check for a run-away polygon")

    # Overlaps within a site, per labeller: two crowns for one tree.
    overlaps = 0
    for (site_id, who), grp in gdf.groupby(["site_id", "labeller"]):
        g = grp.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.geometry[i], g.geometry[j]
                if a.intersects(b):
                    inter = a.intersection(b).area
                    if inter > 0.35 * min(a.area, b.area):
                        overlaps += 1
    if overlaps:
        issues.append(f"{overlaps} heavily overlapping crown pair(s) by the same labeller")

    # Crowns outside their own site box.
    outside = 0
    from shapely.geometry import box as shbox

    for _, row in gdf.iterrows():
        s = sites.get(row.site_id)
        if not s:
            continue
        if not shbox(*s["bbox_epsg2193"]).buffer(2.0).contains(row.geometry.centroid):
            outside += 1
    if outside:
        issues.append(f"{outside} crown(s) centred outside their site box")

    incomplete = sorted({r["site_id"] for r in records if not r["site_complete"]})
    unlabelled = [sid for sid in sites if sid not in {r["site_id"] for r in records}]

    # ---- report ----
    by_stratum = defaultdict(int)
    by_split = defaultdict(int)
    by_class = defaultdict(int)
    for r in records:
        by_stratum[r["stratum"]] += 1
        by_split[r["split"]] += 1
        by_class[r["class"]] += 1

    trees = gdf[gdf["class"] == "tree"]
    lines = [
        "# Reference set", "",
        f"{len(gdf):,} labelled polygons across {gdf.site_id.nunique()} sites "
        f"by {gdf.labeller.nunique()} labeller(s).", "",
        "| Class | n |", "|---|--:|",
    ]
    for k, v in sorted(by_class.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v:,} |")
    lines += ["", "| Stratum | n |", "|---|--:|"]
    for k, v in sorted(by_stratum.items()):
        lines.append(f"| {k} | {v:,} |")
    lines += ["", "| Split | n |", "|---|--:|"]
    for k, v in sorted(by_split.items()):
        lines.append(f"| {k} | {v:,} |")

    if len(trees):
        a = trees.area_m2_geom
        lines += ["", "## Crown area, class `tree`", "",
                  f"- n = {len(trees):,}",
                  f"- median {a.median():.1f} m2, mean {a.mean():.1f} m2",
                  f"- range {a.min():.1f} to {a.max():.1f} m2",
                  f"- implied diameter, median {2 * (a.median() / 3.14159) ** 0.5:.1f} m"]

    lines += ["", "## Quality checks", ""]
    if issues:
        lines += [f"- {i}" for i in issues]
    else:
        lines.append("- no automated issues found")
    if incomplete:
        lines.append(f"- {len(incomplete)} site(s) have labels but are not marked complete: "
                     + ", ".join(incomplete))
    if unlabelled:
        lines.append(f"- {len(unlabelled)} site(s) have no labels yet: " + ", ".join(unlabelled))

    lines += ["", "## Provenance", ""]
    for s in sources:
        lines.append(f"- `{s['file']}` labeller {s['labeller']} exported {s['exported_at']}")

    (d / "reference_set_report.md").write_text("\n".join(lines) + "\n")

    print(f"\nWrote {d}/reference_crowns.gpkg, .geojson, reference_set_report.md")
    print(f"\n{len(gdf):,} polygons  |  " + "  ".join(f"{k}={v}" for k, v in sorted(by_split.items())))
    if issues:
        print(f"\n{len(issues)} quality issue(s):")
        for i in issues:
            print(f"   - {i}")
    if unlabelled:
        print(f"\n{len(unlabelled)} site(s) still unlabelled: {', '.join(unlabelled)}")
    print("\nOpen the .gpkg in QGIS to eyeball the polygons over the imagery.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
