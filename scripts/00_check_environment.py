#!/usr/bin/env python
"""Day one: prove your environment and data access work.

Run this first, and run it again any time something breaks. It checks every
library, path and dataset the rest of the kit needs, and tells you exactly what
to fix. It reads nothing but metadata, changes nothing, and is safe to re-run.

    python scripts/00_check_environment.py
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kit import KIT_ROOT, load_config, say  # noqa: E402

REQUIRED_LIBS = [
    ("numpy", "arrays"),
    ("pandas", "tables"),
    ("rasterio", "reading the CHM and imagery"),
    ("geopandas", "vector data and the reference set"),
    ("shapely", "crown polygons"),
    ("pyproj", "coordinate transforms"),
    ("laspy", "reading LAZ point clouds"),
    ("sklearn", "metrics and splits"),
    ("matplotlib", "figures"),
]
OPTIONAL_LIBS = [
    ("torch", "GPU models, needed from week 6"),
    ("pdal", "point cloud pipelines, alternative to laspy"),
]


def main() -> int:
    print("\n=== Auckland tree benchmark: environment check ===\n")
    fails: list[str] = []
    warns: list[str] = []

    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        fails.append("Python 3.10+ required")
        say(False, "python version", "need 3.10 or newer")
    else:
        say(True, "python version")

    print("\n-- required libraries --")
    for mod, why in REQUIRED_LIBS:
        try:
            m = importlib.import_module(mod)
            say(True, mod, getattr(m, "__version__", ""))
        except Exception:
            say(False, mod, f"needed for {why}")
            fails.append(f"missing library: {mod}")

    print("\n-- optional libraries --")
    for mod, why in OPTIONAL_LIBS:
        try:
            m = importlib.import_module(mod)
            say(True, mod, getattr(m, "__version__", ""))
        except Exception:
            say(None, mod, f"not installed. {why}")
            warns.append(f"optional library missing: {mod}")

    print("\n-- configuration --")
    try:
        cfg = load_config()
        say(True, "config/project.json parsed")
    except Exception as exc:
        say(False, "config/project.json", str(exc))
        print("\nCannot continue without config. Fix it and re-run.")
        return 1

    seed = cfg.get("random_seed")
    say(seed is not None, "random seed", str(seed))
    thr = cfg.get("acceptance_thresholds", {})
    say(bool(thr.get("declared_on")), "thresholds predeclared", f"on {thr.get('declared_on')}")

    # Which mode are we in? The full source data lives on the supervisor's
    # machine. On your own laptop the labelling half still works from the
    # shipped chips, and that is the critical path for weeks 1 to 8.
    chips_index = KIT_ROOT / "data" / "sites" / "chips" / "index.json"
    repo = Path(cfg["paths"]["source_repo"])
    standalone = not (repo.is_dir() and Path(cfg["paths"]["database"]).exists())
    print("\n-- mode --")
    if standalone:
        say(None, "STANDALONE", "full frozen database/source access unavailable; existing chips and cell cache support local labelling")
    else:
        say(True, "FULL", "source repo reachable")

    print("\n-- data paths --")
    paths = cfg["paths"]

    def need(ok: bool, label: str, detail: str, what: str) -> None:
        """Blocking in full mode, informational in standalone mode."""
        if ok:
            say(True, label, detail)
        elif standalone:
            say(None, label, "not on this machine, only needed to regenerate sites")
            warns.append(f"{what} unavailable (standalone)")
        else:
            say(False, label, detail)
            fails.append(f"{what} not found")

    need(repo.is_dir(), "source repo", str(repo), "source_repo")

    db = Path(paths["database"])
    need(db.exists(), "database",
         f"{db.stat().st_size / 1e9:.1f} GB" if db.exists() else str(db), "database")

    chm = Path(paths["chm_vrt"])
    need(chm.exists(), "CHM raster", str(chm.name), "chm_vrt")

    cache = KIT_ROOT / "data" / "sites" / "cell_aggregate.csv.gz"
    if cache.exists():
        say(True, "cell cache", f"{cache.stat().st_size/1e6:.1f} MB, site selection works offline")
    elif standalone:
        say(None, "cell cache", "absent, ask Tim for data/sites/cell_aggregate.csv.gz")
        warns.append("no cell cache, cannot re-run site selection offline")

    img = Path(paths["imagery_tiles_z18"])
    if img.is_dir():
        say(True, "imagery tiles", f"{img}")
    else:
        say(None, "imagery tiles", "source XYZ archive absent; existing chips support labelling without regeneration")
        warns.append("local imagery tiles not found")

    laz = Path(paths["laz_root"])
    if laz.is_dir():
        n = sum(1 for _ in laz.glob("*.laz"))
        say(n > 0, "LAZ point cloud", f"{n:,} tiles in {laz}")
        if n == 0:
            warns.append("LAZ directory present but empty")
    else:
        say(None, "LAZ point cloud", f"not mounted: {laz}")
        warns.append("LAZ root not mounted. External drive may be unplugged.")

    plan = Path(paths["laz_tile_plan"])
    if plan.exists():
        try:
            tiles = json.loads(plan.read_text())["tiles"]
            say(True, "LAZ tile plan", f"{len(tiles):,} tiles planned")
        except Exception as exc:
            say(False, "LAZ tile plan", str(exc))
            fails.append("tile plan unreadable")
    else:
        say(None, "LAZ tile plan", "not found")
        warns.append("tile plan not found")

    print("\n-- labelling assets (what you need on day one) --")
    if chips_index.exists():
        try:
            chips = json.loads(chips_index.read_text())
            say(True, "site chips", f"{len(chips)} sites ready to label")
        except Exception as exc:
            say(False, "site chips", str(exc))
            fails.append("chips index unreadable")
    else:
        say(None, "site chips", "none yet, run scripts 01 and 02 (needs full mode)")
        warns.append("no site chips")

    print("\n-- database contents --")
    if db.exists():
        try:
            from _kit import db_connect

            con = db_connect(cfg)
            cur = con.cursor()
            for table, expect in [
                ("trees", 1_000_000),
                ("tree_lidar_pilot", 1_000_000),
                ("tree_crown_pilot", 1_000_000),
                ("tree_pointcloud_pilot", 1_000_000),
                ("tree_species_attributes", 1_000_000),
            ]:
                try:
                    n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    say(n >= expect, table, f"{n:,} rows")
                except Exception:
                    say(False, table, "missing")
                    fails.append(f"table missing: {table}")
            con.close()
        except Exception as exc:
            say(False, "database read", str(exc))
            fails.append("cannot read database")

    print("\n-- kit folders --")
    for sub in ["data/sample", "data/sites", "data/labels", "data/exports", "labeller", "docs"]:
        say((KIT_ROOT / sub).exists(), sub)

    print("\n" + "=" * 55)
    if fails:
        print(f"{len(fails)} BLOCKING problem(s):")
        for f in fails:
            print(f"   - {f}")
        print("\nFix these before continuing. If a path is wrong, edit config/project.json.")
        print("If a library is missing:  conda env update -f environment.yml")
        print("If you are stuck for more than a day, email Tim. Do not sit on it.")
        return 1

    print("Environment OK. No blocking problems.")
    if warns:
        print(f"\n{len(warns)} warning(s), none blocking day-one work:")
        for w in warns:
            print(f"   - {w}")

    if standalone:
        print("\nSTANDALONE MODE. This is expected on your own machine.")
        print("  You CAN:    label crowns, export the reference set, measure agreement")
        print("              (scripts 03, 04, 05) and re-run site selection if the")
        print("              cell cache is present.")
        print("  You CANNOT: cut new image chips (script 02), which needs the CHM and")
        print("              imagery on Tim's machine. Ask him to run it, or work there.")
        print("\nNext:  python scripts/03_serve_labeller.py")
    else:
        print("\nNext:  python scripts/01_select_reference_sites.py --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
