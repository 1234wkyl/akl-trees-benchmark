"""Shared helpers for the starter kit. Import this, don't copy from it."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = KIT_ROOT / "config" / "project.json"


def load_config(path: Path | None = None) -> dict:
    """Load project.json and expand {source_repo} placeholders in paths."""
    cfg = json.loads((path or CONFIG_PATH).read_text())
    paths = cfg["paths"]
    repo = paths["source_repo"]
    for key, value in paths.items():
        if isinstance(value, str):
            expanded = Path(value.replace("{source_repo}", repo)).expanduser()
            paths[key] = str(expanded if expanded.is_absolute() else (KIT_ROOT / expanded).resolve())
    return cfg


def db_connect(cfg: dict) -> sqlite3.Connection:
    """Open the shared database READ-ONLY.

    The database is 11 GB, shared, and the live research store. Never open it
    writable from the kit. Write your own outputs to data/ instead.
    """
    db = Path(cfg["paths"]["database"])
    if not db.exists():
        raise SystemExit(
            f"Database not found: {db}\n\n"
            "If you are on your own machine, this is expected. The 11 GB database\n"
            "lives on the supervisor's machine, and this step needs it.\n\n"
            "  Labelling does NOT need the database. Run:\n"
            "      python scripts/03_serve_labeller.py\n\n"
            "  To cut chips for NEW sites, either work on Tim's machine or send him\n"
            "  data/sites/reference_sites.json and ask him to run script 02.\n\n"
            "If you ARE on the machine with the data, fix the path in config/project.json."
        )
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def out_dir(name: str) -> Path:
    d = KIT_ROOT / "data" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def say(ok: bool | None, label: str, detail: str = "") -> bool:
    mark = {True: "  ok  ", False: " FAIL ", None: " warn "}[ok]
    print(f"[{mark}] {label}" + (f"  {detail}" if detail else ""))
    return bool(ok)


def site_grid_cell(x: float, y: float, size_m: int) -> tuple[int, int]:
    """Snap a coordinate to the origin of its grid cell."""
    return (int(x // size_m) * size_m, int(y // size_m) * size_m)
