#!/usr/bin/env python
"""Cut the imagery, canopy height and point cloud for each reference site.

Everything is resampled onto ONE grid in EPSG:2193, so a pixel in the labeller
maps to a real coordinate with no approximation:

    x = x0 + (col + 0.5) * res
    y = y1 - (row + 0.5) * res

    python scripts/02_extract_site_chips.py                 # all sites, local imagery
    python scripts/02_extract_site_chips.py --site S001     # one site
    python scripts/02_extract_site_chips.py --with-points   # also clip the LAZ
    python scripts/02_extract_site_chips.py --imagery linz  # sharper, needs LINZ_API_KEY

Writes data/sites/chips/<SITE>/ containing rgb.png, chm.png, chm.tif,
meta.json, existing_detections.json and optionally points.npz.

Note on imagery. The local Esri tiles are about 0.48 m/pixel, so a 6 m crown is
roughly 12 pixels across. That is usable but coarse. LINZ 0.075 m imagery is far
better for delineating crowns and is what your final reference set should use.
Discuss with Tim before spending a day downloading it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kit import db_connect, load_config, out_dir  # noqa: E402

TILE_SIZE = 256
ESRI_ZOOM = 18
LINZ_ZOOM = 21
LINZ_LAYER = 121752


def lonlat_to_world_px(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = max(-85.05112878, min(85.05112878, lat))
    scale = (2 ** zoom) * TILE_SIZE
    x = (lon + 180.0) / 360.0 * scale
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * scale
    return x, y


def mercator_bounds_of_tile(tx: int, ty: int, zoom: int) -> tuple[float, float, float, float]:
    """Exact EPSG:3857 bounds of an XYZ tile."""
    n = 2 ** zoom
    half = 20037508.342789244
    w = 2 * half / n
    return (-half + tx * w, half - (ty + 1) * w, -half + (tx + 1) * w, half - ty * w)


def build_rgb_mosaic(site_bbox, cfg, source: str, api_key: str | None):
    """Mosaic XYZ tiles covering the site, returned as an array + 3857 transform."""
    x0, y0, x1, y1 = site_bbox
    to_wgs = Transformer.from_crs(cfg["crs"], "EPSG:4326", always_xy=True)
    lons, lats = [], []
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        lo, la = to_wgs.transform(x, y)
        lons.append(lo)
        lats.append(la)

    zoom = ESRI_ZOOM if source == "esri" else LINZ_ZOOM
    px = [lonlat_to_world_px(lo, la, zoom) for lo, la in zip(lons, lats)]
    txs = [int(p[0] // TILE_SIZE) for p in px]
    tys = [int(p[1] // TILE_SIZE) for p in px]
    tx0, tx1 = min(txs) - 1, max(txs) + 1
    ty0, ty1 = min(tys) - 1, max(tys) + 1

    from PIL import Image

    cols, rows = tx1 - tx0 + 1, ty1 - ty0 + 1
    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE), (32, 32, 32))
    found = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            img = load_tile(tx, ty, zoom, cfg, source, api_key)
            if img is None:
                continue
            mosaic.paste(img, ((tx - tx0) * TILE_SIZE, (ty - ty0) * TILE_SIZE))
            found += 1

    west, _, _, north = mercator_bounds_of_tile(tx0, ty0, zoom)
    _, south, east, _ = mercator_bounds_of_tile(tx1, ty1, zoom)
    res_x = (east - west) / mosaic.width
    res_y = (north - south) / mosaic.height
    transform = from_origin(west, north, res_x, res_y)
    return np.array(mosaic).transpose(2, 0, 1), transform, found


_missing_warned = set()


def load_tile(tx, ty, zoom, cfg, source, api_key):
    from PIL import Image

    if source == "esri":
        p = Path(cfg["paths"]["imagery_tiles_z18"]) / f"{zoom}_{tx}_{ty}.jpg"
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                return None
        return None

    cache = Path(cfg["paths"]["source_repo"]) / "data/raw/linz_urban_imagery/tiles_z21"
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"{zoom}_{tx}_{ty}.png"
    if p.exists():
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return None
    if not api_key:
        if "key" not in _missing_warned:
            print("  ! LINZ_API_KEY not set, cannot download high-res tiles")
            _missing_warned.add("key")
        return None
    import requests

    url = (f"https://basemaps.linz.govt.nz/v1/tiles/aerial/"
           f"layer={LINZ_LAYER}/EPSG:3857/{zoom}/{tx}/{ty}.png?api={api_key}")
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and r.content:
            p.write_bytes(r.content)
            return Image.open(p).convert("RGB")
    except Exception:
        return None
    return None


def warp_to_site_grid(src_array, src_transform, src_crs, bbox, res, dst_crs, resampling):
    x0, y0, x1, y1 = bbox
    width = int(round((x1 - x0) / res))
    height = int(round((y1 - y0) / res))
    dst_transform = from_origin(x0, y1, res, res)
    bands = src_array.shape[0]
    dst = np.zeros((bands, height, width), dtype=src_array.dtype)
    for b in range(bands):
        reproject(
            source=src_array[b],
            destination=dst[b],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )
    return dst, dst_transform, width, height


def colourise_chm(chm: np.ndarray, nodata: float, vmax: float = 25.0) -> np.ndarray:
    """Height ramp for the labeller.

    A light-on-light ramp makes a 6 m building look like a 6 m tree, which is
    exactly the distinction you are labelling. Use a dark-to-bright sequential
    ramp instead, hold everything below 1 m at the dark end as ground, and mark
    the 3 m canopy threshold the classical detector uses.
    """
    import matplotlib

    valid = chm != nodata
    v = np.where(valid, chm, 0.0)
    v = np.clip(v, 0.0, vmax) / vmax
    rgb = (matplotlib.colormaps["viridis"](v)[..., :3] * 255).astype(np.uint8)

    ground = valid & (chm < 1.0)
    rgb[ground] = (18, 18, 24)
    rgb[~valid] = (60, 20, 20)  # nodata reads as dark red, not as ground

    # Thin contour at the 3 m canopy threshold, so the detector's blind spot is visible.
    near3 = valid & (np.abs(chm - 3.0) < 0.15)
    rgb[near3] = (255, 255, 255)
    return rgb.transpose(2, 0, 1)


def clip_points(site_bbox, cfg, decimate: int = 2):
    """Clip the classified point cloud to the site."""
    plan_path = Path(cfg["paths"]["laz_tile_plan"])
    laz_root = Path(cfg["paths"]["laz_root"])
    if not plan_path.exists() or not laz_root.is_dir():
        return None
    import laspy

    x0, y0, x1, y1 = site_bbox
    plan = json.loads(plan_path.read_text())
    xs, ys, zs, cls, ret = [], [], [], [], []
    for t in plan["tiles"]:
        b = t["bbox_epsg2193"]
        if b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1:
            continue
        f = laz_root / t["output_name"]
        if not f.exists():
            continue
        try:
            with laspy.open(f) as fh:
                pts = fh.read()
        except Exception:
            continue
        m = (pts.x >= x0) & (pts.x < x1) & (pts.y >= y0) & (pts.y < y1)
        if not m.any():
            continue
        xs.append(np.asarray(pts.x)[m][::decimate])
        ys.append(np.asarray(pts.y)[m][::decimate])
        zs.append(np.asarray(pts.z)[m][::decimate])
        cls.append(np.asarray(pts.classification)[m][::decimate])
        try:
            ret.append(np.asarray(pts.return_number)[m][::decimate])
        except Exception:
            ret.append(np.zeros(m.sum(), dtype=np.uint8)[::decimate])
    if not xs:
        return None
    return {
        "x": np.concatenate(xs), "y": np.concatenate(ys), "z": np.concatenate(zs),
        "classification": np.concatenate(cls), "return_number": np.concatenate(ret),
    }


def existing_detections(con, bbox):
    x0, y0, x1, y1 = bbox
    sql = """
      SELECT l.tree_id, l.x_2193, l.y_2193,
             COALESCE(p.canopy_top_m, l.chm_at_point_m) AS h,
             c.crown_area_m2, t.source_primary, s.growth_form, s.growth_form_confidence
      FROM tree_lidar_pilot l
      JOIN trees t ON t.tree_id = l.tree_id
      LEFT JOIN tree_crown_pilot c ON c.tree_id = l.tree_id
      LEFT JOIN tree_pointcloud_pilot p ON p.tree_id = l.tree_id
      LEFT JOIN tree_species_attributes s ON s.tree_id = l.tree_id
      WHERE l.x_2193 >= ? AND l.x_2193 < ? AND l.y_2193 >= ? AND l.y_2193 < ?
    """
    keys = ["tree_id", "x", "y", "height_m", "crown_area_m2",
            "source_primary", "growth_form", "growth_form_confidence"]
    return [dict(zip(keys, r)) for r in con.execute(sql, (x0, x1, y0, y1))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", help="single site id, e.g. S001")
    ap.add_argument("--res", type=float, default=0.25, help="chip resolution, m/pixel")
    ap.add_argument("--imagery", choices=["esri", "linz"], default="esri")
    ap.add_argument("--with-points", action="store_true", help="also clip LAZ (needs the drive)")
    args = ap.parse_args()

    cfg = load_config()
    sites_file = out_dir("sites") / "reference_sites.json"
    if not sites_file.exists():
        print("No sites yet. Run scripts/01_select_reference_sites.py first.")
        return 1
    sites = json.loads(sites_file.read_text())["sites"]
    if args.site:
        sites = [s for s in sites if s["site_id"] == args.site]
        if not sites:
            print(f"No such site: {args.site}")
            return 1

    api_key = None
    if args.imagery == "linz":
        import os

        api_key = os.environ.get("LINZ_API_KEY")
        env = Path(cfg["paths"]["source_repo"]) / ".env"
        if not api_key and env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("LINZ_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

    chips_root = out_dir("sites") / "chips"
    chips_root.mkdir(exist_ok=True)
    con = db_connect(cfg)
    from PIL import Image

    print(f"Extracting {len(sites)} site(s) at {args.res} m/pixel from {args.imagery} imagery\n")
    for s in sites:
        sid = s["site_id"]
        bbox = s["bbox_epsg2193"]
        d = chips_root / sid
        d.mkdir(exist_ok=True)

        rgb_src, rgb_tf, n_tiles = build_rgb_mosaic(bbox, cfg, args.imagery, api_key)
        rgb, dst_tf, W, H = warp_to_site_grid(
            rgb_src, rgb_tf, CRS.from_epsg(3857), bbox, args.res,
            CRS.from_string(cfg["crs"]), Resampling.cubic)
        Image.fromarray(rgb.transpose(1, 2, 0)).save(d / "rgb.png")

        with rasterio.open(cfg["paths"]["chm_vrt"]) as src:
            win = rasterio.windows.from_bounds(*bbox, transform=src.transform)
            chm_src = src.read(1, window=win, boundless=True, fill_value=src.nodata)
            chm_tf = src.window_transform(win)
            nodata = src.nodata
        chm, _, _, _ = warp_to_site_grid(
            chm_src[None, ...], chm_tf, CRS.from_string(cfg["crs"]), bbox, args.res,
            CRS.from_string(cfg["crs"]), Resampling.bilinear)
        chm = chm[0]
        Image.fromarray(colourise_chm(chm, nodata).transpose(1, 2, 0)).save(d / "chm.png")
        with rasterio.open(d / "chm.tif", "w", driver="GTiff", height=H, width=W,
                           count=1, dtype="float32", crs=cfg["crs"], transform=dst_tf,
                           nodata=nodata) as dst:
            dst.write(chm.astype("float32"), 1)

        dets = existing_detections(con, bbox)
        (d / "existing_detections.json").write_text(json.dumps(dets, indent=1))

        n_pts = 0
        if args.with_points:
            pts = clip_points(bbox, cfg)
            if pts:
                np.savez_compressed(d / "points.npz", **pts)
                n_pts = len(pts["x"])

        valid = chm[chm != nodata]
        meta = {
            "site_id": sid, "stratum": s["stratum"], "split": s["split"],
            "crs": cfg["crs"], "bbox_epsg2193": bbox,
            "resolution_m": args.res, "width_px": W, "height_px": H,
            "imagery_source": args.imagery, "imagery_tiles_found": n_tiles,
            "pixel_to_world": {
                "x": "x0 + (col + 0.5) * resolution_m",
                "y": "y1 - (row + 0.5) * resolution_m",
                "x0": bbox[0], "y1": bbox[3],
            },
            "chm_stats": {
                "min": float(valid.min()) if valid.size else None,
                "max": float(valid.max()) if valid.size else None,
                "mean": float(valid.mean()) if valid.size else None,
            },
            "existing_detection_count": len(dets),
            "point_count": n_pts,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  {sid}  {s['stratum']:<20} {W}x{H}px  tiles={n_tiles:<3} "
              f"dets={len(dets):<4} chm_max={meta['chm_stats']['max'] or 0:.1f}m"
              + (f"  pts={n_pts:,}" if args.with_points else ""))

    con.close()
    index = [{"site_id": s["site_id"], "stratum": s["stratum"], "split": s["split"]} for s in sites]
    (chips_root / "index.json").write_text(json.dumps(index, indent=2))
    write_required_tiles(sites, cfg)
    print(f"\nWrote {chips_root}")
    print("\nNext:  python scripts/03_serve_labeller.py")
    return 0


def write_required_tiles(sites, cfg) -> None:
    """List only the LAZ tiles the sites actually need.

    You do not need all 3,113 metropolitan tiles on the cluster. Copying the
    whole set wastes a day and most of your disk quota. This is the short list.
    """
    plan_path = Path(cfg["paths"]["laz_tile_plan"])
    if not plan_path.exists():
        return
    plan = json.loads(plan_path.read_text())
    laz_root = Path(cfg["paths"]["laz_root"])
    needed, missing = set(), set()
    for s in sites:
        x0, y0, x1, y1 = s["bbox_epsg2193"]
        for t in plan["tiles"]:
            b = t["bbox_epsg2193"]
            if b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1:
                continue
            needed.add(t["output_name"])
            if laz_root.is_dir() and not (laz_root / t["output_name"]).exists():
                missing.add(t["output_name"])

    d = out_dir("sites")
    (d / "required_tiles.txt").write_text("\n".join(sorted(needed)) + "\n")
    print(f"\n{len(needed)} LAZ tile(s) cover these sites, of 3,113 metropolitan tiles.")
    print(f"  list: {d / 'required_tiles.txt'}")
    if missing:
        print(f"  ! {len(missing)} not present locally: {', '.join(sorted(missing)[:4])}"
              + (" ..." if len(missing) > 4 else ""))
        (d / "missing_tiles.txt").write_text("\n".join(sorted(missing)) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
