#!/usr/bin/env python
"""Select stratified reference sites for hand labelling.

A "site" is a square cell you will label exhaustively: every tree crown inside
it gets a polygon. Sites are chosen to cover eight strata, kept apart from each
other, and assigned to train / validation / test by spatial block so that no
test site sits next to a training site.

    python scripts/01_select_reference_sites.py --dry-run   # show the plan
    python scripts/01_select_reference_sites.py             # write sites

Outputs (data/sites/):
    reference_sites.geojson   site polygons with stratum and split
    reference_sites.json      the same, plus selection metadata
    site_selection_report.md  counts you can paste into the report
    site_map.png              preview map

IMPORTANT, read before you run this twice.
This script encodes MY first guess at the sampling design, not the final word.
The stratum rules below are defensible but arguable, and defending or changing
them is your work, not mine. If you change them, change them here, write down
why in docs/DECISIONS.md, and re-run. Do not hand-edit the output files.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kit import db_connect, load_config, out_dir  # noqa: E402

# Cells smaller than this many trees are not worth a labelling session,
# except for the deliberate hard-negative stratum.
MIN_TREES_PER_SITE = 6
BLOCK_SIZE_M = 4000  # spatial block for the train/val/test split


CACHE_NAME = "cell_aggregate.csv.gz"

COLS = [
    "cx", "cy", "n_trees", "mean_h", "max_h", "mean_crown", "n_register",
    "n_conifer", "n_broadleaf", "n_palm_known", "paved", "built", "pt_density",
]


def load_or_build_cells(cfg: dict, size_m: int, rebuild: bool) -> tuple[list[dict], str]:
    """Get the cell summary, from cache when the database is not reachable.

    The 11 GB database lives on the supervisor's machine. Caching the aggregate
    means you can redesign the sampling on your own laptop, on a plane, with no
    database at all. Changing the stratum rules is your work, so it must not
    depend on somebody else's disk.
    """
    import csv
    import gzip

    cache = out_dir("sites") / CACHE_NAME
    db = Path(cfg["paths"]["database"])

    if not rebuild and cache.exists():
        with gzip.open(cache, "rt", newline="") as fh:
            rows = []
            for r in csv.DictReader(fh):
                rows.append({k: (None if r[k] == "" else
                                 (int(float(r[k])) if k in ("cx", "cy", "n_trees", "n_register",
                                                            "n_conifer", "n_broadleaf",
                                                            "n_palm_known")
                                  else float(r[k]))) for k in COLS})
        return rows, f"cache ({cache.name}, {len(rows):,} cells)"

    if not db.exists():
        raise SystemExit(
            f"No database at {db} and no cache at {cache}.\n\n"
            "You are probably on your own machine rather than the supervisor's.\n"
            "Ask Tim for data/sites/cell_aggregate.csv.gz, drop it in data/sites/,\n"
            "and re-run. Everything in this script works from that file alone."
        )

    con = db_connect(cfg)
    rows = aggregate_cells(con, size_m)
    con.close()

    with gzip.open(cache, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows, f"database, cached to {cache.name}"


def aggregate_cells(con, size_m: int) -> list[dict]:
    """Aggregate every tree into its grid cell and summarise the cell."""
    sql = f"""
    SELECT
      CAST(l.x_2193 / {size_m} AS INT) * {size_m} AS cx,
      CAST(l.y_2193 / {size_m} AS INT) * {size_m} AS cy,
      COUNT(*)                                          AS n_trees,
      AVG(COALESCE(p.canopy_top_m, l.chm_at_point_m))   AS mean_h,
      MAX(COALESCE(p.canopy_top_m, l.chm_at_point_m))   AS max_h,
      AVG(c.crown_area_m2)                              AS mean_crown,
      SUM(CASE WHEN t.source_primary IN
            ('tree_register_points','notable_trees_overlay') THEN 1 ELSE 0 END) AS n_register,
      SUM(CASE WHEN s.growth_form='conifer'             THEN 1 ELSE 0 END) AS n_conifer,
      SUM(CASE WHEN s.growth_form LIKE '%broadleaf'     THEN 1 ELSE 0 END) AS n_broadleaf,
      SUM(CASE WHEN s.growth_form='palm_other'
                AND s.growth_form_confidence='known'    THEN 1 ELSE 0 END) AS n_palm_known,
      AVG(x.fraction_paved_surfaces)                    AS paved,
      AVG(x.fraction_buildings)                         AS built,
      AVG(p.point_density)                              AS pt_density
    FROM tree_lidar_pilot l
    JOIN trees t                    ON t.tree_id = l.tree_id
    LEFT JOIN tree_crown_pilot c    ON c.tree_id = l.tree_id
    LEFT JOIN tree_pointcloud_pilot p ON p.tree_id = l.tree_id
    LEFT JOIN tree_species_attributes s ON s.tree_id = l.tree_id
    LEFT JOIN tree_context_pilot x  ON x.tree_id = l.tree_id
    WHERE l.x_2193 IS NOT NULL AND l.y_2193 IS NOT NULL
    GROUP BY cx, cy
    """
    return [dict(zip(COLS, row)) for row in con.execute(sql)]


def classify(cell: dict) -> str | None:
    """Assign one stratum. Order matters: the rarest classes are tested first.

    These rules use the existing (imperfect) model outputs only to *find*
    candidate sites. They never become labels. Your hand labels are the truth.
    """
    n = cell["n_trees"] or 0
    mean_h = cell["mean_h"] or 0.0
    crown = cell["mean_crown"] or 0.0
    paved = cell["paved"] or 0.0
    built = cell["built"] or 0.0
    conif = cell["n_conifer"] or 0
    broad = cell["n_broadleaf"] or 0
    palm = cell["n_palm_known"] or 0
    reg = cell["n_register"] or 0

    # Deliberate negatives: built-up, almost no canopy. Tests false positives.
    if n <= 3 and (built >= 0.25 or paved >= 0.55):
        return "hard_negative_built"
    if n < MIN_TREES_PER_SITE:
        return None
    if palm >= 1:
        return "palm_present"
    if conif >= 3 and conif > broad:
        return "conifer_dominant"
    if mean_h < 6.0:
        return "small_subcanopy"
    if n >= 25 and mean_h >= 12.0 and paved <= 0.25:
        return "dense_broadleaf"
    if crown >= 60.0 and paved <= 0.30 and n >= 8:
        return "park_open_grown"
    if reg >= 4 and paved >= 0.35:
        return "open_street_trees"
    if 0.25 <= paved <= 0.55 and n >= 10:
        return "canopy_edge"
    return None


def assign_blocks(cells: list[dict], cfg: dict, rng: random.Random) -> dict[tuple, str]:
    """Assign whole spatial blocks to a split so sites cannot leak across."""
    blocks = sorted({(int(c["cx"] // BLOCK_SIZE_M), int(c["cy"] // BLOCK_SIZE_M)) for c in cells})
    rng.shuffle(blocks)
    frac = cfg["reference_set"]["splits"]
    n = len(blocks)
    n_train = int(round(frac["train"] * n))
    n_val = int(round(frac["val"] * n))
    out = {}
    for i, b in enumerate(blocks):
        out[b] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


def build_slots(strata: list[str], cfg: dict) -> list[tuple[str, str]]:
    """Plan exactly which (stratum, split) sites we want, before looking at data.

    Every stratum appears in the test split, otherwise a stratum-level recall
    threshold cannot be evaluated on held-out data.
    """
    target = cfg["reference_set"]["target_sites"]
    frac = cfg["reference_set"]["splits"]
    per_stratum = max(1, target // len(strata))

    # Order chosen so that with 3 sites per stratum the mix lands near 50/20/30.
    order = ["train", "test", "val"] if frac["test"] >= frac["val"] else ["train", "val", "test"]
    slots: list[tuple[str, str]] = []
    for i, s in enumerate(strata):
        for j in range(per_stratum):
            # Rotate the start point per stratum so splits stay balanced overall.
            slots.append((s, order[(i + j) % len(order)]))
    return slots[:target]


def select(cells, strata, cfg, block_split, rng):
    """Fill the planned slots, honouring separation, split buffers and a
    realistic labelling budget per site."""
    size = cfg["reference_set"]["site_size_m"]
    sep = cfg["reference_set"]["min_site_separation_m"]
    buf = cfg["reference_set"]["split_buffer_m"]
    target_sites = cfg["reference_set"]["target_sites"]
    target_crowns = cfg["reference_set"]["target_crowns"]

    # The labelling budget is what makes this finishable. A 100 m cell in dense
    # Auckland canopy can hold 130 crowns; labelling 24 of those is 3,000 crowns
    # and six weeks you do not have. Prefer cells near the per-site budget.
    ideal = max(8, target_crowns // max(1, target_sites))

    for c in cells:
        c["split"] = block_split[(int(c["cx"] // BLOCK_SIZE_M), int(c["cy"] // BLOCK_SIZE_M))]

    index: dict[tuple[str, str], list[dict]] = {}
    for c in cells:
        index.setdefault((c["stratum"], c["split"]), []).append(c)

    # Deterministic tiebreak. Sorting ties by height would systematically pick the
    # tallest cell in every stratum, so the reference set would over-represent big
    # trees and flatter the detectors. Break ties pseudo-randomly instead.
    def jitter(c: dict) -> float:
        return random.Random(f"{cfg['random_seed']}:{c['cx']}:{c['cy']}").random()

    for key, group in index.items():
        stratum = key[0]
        if stratum == "hard_negative_built":
            # Here few trees is the point: these cells test false positives.
            group.sort(key=lambda c: (c["n_trees"] or 0, -(c["built"] or 0), jitter(c)))
        else:
            group.sort(key=lambda c: (abs((c["n_trees"] or 0) - ideal), jitter(c)))

    chosen: list[dict] = []

    def ok(cand: dict) -> bool:
        cxm, cym = cand["cx"] + size / 2, cand["cy"] + size / 2
        for other in chosen:
            oxm, oym = other["cx"] + size / 2, other["cy"] + size / 2
            d = math.hypot(cxm - oxm, cym - oym)
            if d < sep:
                return False
            if other["split"] != cand["split"] and d < buf:
                return False
        return True

    unfilled: list[tuple[str, str]] = []
    for stratum, split in build_slots(strata, cfg):
        for cand in index.get((stratum, split), []):
            if cand.get("_used"):
                continue
            if ok(cand):
                cand["_used"] = True
                chosen.append(cand)
                break
        else:
            unfilled.append((stratum, split))

    # A stratum may have no eligible cell in the wanted split. Rather than skew
    # the design silently, fall back to any split and report it.
    for stratum, split in unfilled:
        for alt in ("test", "train", "val"):
            picked = False
            for cand in index.get((stratum, alt), []):
                if cand.get("_used"):
                    continue
                if ok(cand):
                    cand["_used"] = True
                    cand["_reassigned_from"] = split
                    chosen.append(cand)
                    picked = True
                    break
            if picked:
                break

    return chosen, unfilled, ideal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="re-read the database instead of the cached cell summary")
    args = ap.parse_args()

    cfg = load_config()
    seed = args.seed if args.seed is not None else cfg["random_seed"]
    rng = random.Random(seed)
    size = cfg["reference_set"]["site_size_m"]
    strata = cfg["reference_set"]["strata"]

    print(f"Aggregating trees into {size} m cells (seed {seed}) ...")
    cells, source = load_or_build_cells(cfg, size, args.rebuild_cache)
    print(f"  source: {source}")
    print(f"  {len(cells):,} occupied cells")

    for c in cells:
        c["stratum"] = classify(c)
    eligible = [c for c in cells if c["stratum"]]
    print(f"  {len(eligible):,} cells match a stratum\n")

    counts: dict[str, int] = {}
    for c in eligible:
        counts[c["stratum"]] = counts.get(c["stratum"], 0) + 1
    print("Available cells per stratum:")
    for s in strata:
        print(f"  {s:24s} {counts.get(s, 0):6,}")

    block_split = assign_blocks(eligible, cfg, rng)
    chosen, unfilled, ideal = select(eligible, strata, cfg, block_split, rng)

    est = sum(c["n_trees"] or 0 for c in chosen)
    print(f"\nSelected {len(chosen)} sites, ~{est:,} existing detections inside them.")
    print(f"Labelling budget: aiming for ~{ideal} crowns per site "
          f"(~{cfg['reference_set']['target_crowns']} total).")
    print("(That count is the *current model's*. Your labelling will differ, which is the point.)\n")
    if unfilled:
        print("NOTE: no eligible cell for these (stratum, split) slots; reassigned to another split:")
        for s, sp in unfilled:
            print(f"  - {s} / {sp}")
        print()

    print(f"{'site':<6}{'stratum':<22}{'split':<7}{'trees':>6}{'mean h':>8}{'paved':>7}")
    for i, c in enumerate(chosen, 1):
        print(f"{i:<6}{c['stratum']:<22}{c['split']:<7}{c['n_trees'] or 0:>6}"
              f"{c['mean_h'] or 0:>8.1f}{c['paved'] or 0:>7.2f}")

    split_counts: dict[str, int] = {}
    for c in chosen:
        split_counts[c["split"]] = split_counts.get(c["split"], 0) + 1
    print("\nSplit balance:", ", ".join(f"{k}={v}" for k, v in sorted(split_counts.items())))

    if args.dry_run:
        print("\nDry run, nothing written. Re-run without --dry-run to save.")
        return 0

    d = out_dir("sites")
    features = []
    records = []
    for i, c in enumerate(chosen, 1):
        x0, y0 = float(c["cx"]), float(c["cy"])
        x1, y1 = x0 + size, y0 + size
        site_id = f"S{i:03d}"
        rec = {
            "site_id": site_id,
            "stratum": c["stratum"],
            "split": c["split"],
            "bbox_epsg2193": [x0, y0, x1, y1],
            "centroid_epsg2193": [x0 + size / 2, y0 + size / 2],
            "existing_detections": int(c["n_trees"] or 0),
            "mean_height_m": round(float(c["mean_h"] or 0), 2),
            "mean_crown_area_m2": round(float(c["mean_crown"] or 0), 1),
            "paved_fraction": round(float(c["paved"] or 0), 3),
            "point_density": round(float(c["pt_density"] or 0), 2),
            "labelled": False,
        }
        records.append(rec)
        features.append({
            "type": "Feature",
            "properties": rec,
            "geometry": {"type": "Polygon", "coordinates": [[
                [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
        })

    payload = {
        "schema_version": "akl-trees-reference-sites/v1",
        "generated_by": "01_select_reference_sites.py",
        "seed": seed,
        "site_size_m": size,
        "block_size_m": BLOCK_SIZE_M,
        "min_site_separation_m": cfg["reference_set"]["min_site_separation_m"],
        "split_buffer_m": cfg["reference_set"]["split_buffer_m"],
        "crs": cfg["crs"],
        "sites": records,
    }
    (d / "reference_sites.json").write_text(json.dumps(payload, indent=2))
    (d / "reference_sites.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2193"}},
        "features": features,
    }, indent=2))

    lines = [
        "# Reference site selection", "",
        f"Seed {seed}. Site size {size} m. {len(chosen)} sites selected from "
        f"{len(eligible):,} eligible cells.", "",
        "| Stratum | Sites | Available cells |", "|---|--:|--:|",
    ]
    for s in strata:
        lines.append(f"| {s} | {sum(1 for c in chosen if c['stratum'] == s)} | {counts.get(s, 0):,} |")
    lines += ["", "| Split | Sites |", "|---|--:|"]
    for k, v in sorted(split_counts.items()):
        lines.append(f"| {k} | {v} |")
    lines += ["", f"Minimum separation between sites: {cfg['reference_set']['min_site_separation_m']} m.",
              f"Minimum buffer between different splits: {cfg['reference_set']['split_buffer_m']} m.",
              f"Splits assigned by {BLOCK_SIZE_M} m spatial block, so neighbouring sites",
              "cannot land in different splits and leak information."]
    (d / "site_selection_report.md").write_text("\n".join(lines) + "\n")

    try:
        make_map(chosen, size, d / "site_map.png")
        print(f"\nWrote site_map.png")
    except Exception as exc:
        print(f"\n(map skipped: {exc})")

    print(f"Wrote {d}/reference_sites.geojson, .json, site_selection_report.md")
    print("\nNext:  python scripts/02_extract_site_chips.py")
    return 0


def make_map(chosen, size, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    colours = {"train": "#2a78d6", "val": "#eda100", "test": "#e34948"}
    fig, ax = plt.subplots(figsize=(9, 8))
    for c in chosen:
        ax.add_patch(Rectangle((c["cx"], c["cy"]), size * 12, size * 12,
                               color=colours[c["split"]], alpha=0.85))
    xs = [c["cx"] for c in chosen]
    ys = [c["cy"] for c in chosen]
    pad = 3000
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_title("Reference sites by split (markers not to scale)", fontsize=12)
    ax.set_xlabel("Easting (EPSG:2193)")
    ax.set_ylabel("Northing (EPSG:2193)")
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=v, label=k) for k, v in colours.items()]
    ax.legend(handles=handles, frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
