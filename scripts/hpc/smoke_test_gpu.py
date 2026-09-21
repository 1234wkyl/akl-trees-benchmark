#!/usr/bin/env python
"""Prove the GPU box works BEFORE you try to install SegmentAnyTree.

Most cluster time is lost to a broken CUDA/PyTorch pairing that only shows up
deep inside somebody else's training script. This checks the stack bottom-up so
a failure points at one thing.

    python scripts/hpc/smoke_test_gpu.py
    python scripts/hpc/smoke_test_gpu.py --laz /path/to/tile.laz

Run it on the Nectar instance, not your laptop. Paste the output into your lab
notebook: it is the record of what the environment actually was on that date.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def head(t: str) -> None:
    print(f"\n-- {t} " + "-" * max(0, 52 - len(t)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laz", help="optional LAZ tile to test point cloud reading")
    args = ap.parse_args()

    print("=" * 60)
    print("GPU smoke test")
    print(f"time  {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"host  {platform.node()}")
    print(f"os    {platform.platform()}")
    print(f"py    {sys.version.split()[0]}")
    print("=" * 60)

    ok = True

    head("nvidia-smi")
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,memory.used,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
            print(out.stdout.strip() or out.stderr.strip())
        except Exception as exc:
            print(f"failed: {exc}")
            ok = False
    else:
        print("nvidia-smi NOT FOUND. You are not on a GPU node, or the driver is missing.")
        print("Use the reserved standalone Nectar GPU VM and check its NVIDIA driver.")
        ok = False

    head("torch")
    try:
        import torch
        print(f"torch          {torch.__version__}")
        print(f"compiled CUDA  {torch.version.cuda}")
        print(f"cuda available {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"device count   {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  [{i}] {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
        else:
            print("\nCUDA NOT AVAILABLE. The usual cause is a PyTorch build that does not")
            print("match the driver. Check the CUDA version in nvidia-smi against")
            print("torch.version.cuda above, then reinstall PyTorch for that CUDA.")
            ok = False
    except ImportError:
        print("torch NOT INSTALLED")
        return 1

    if torch.cuda.is_available():
        head("gpu compute")
        try:
            import time
            n = 4096
            a = torch.randn(n, n, device="cuda")
            b = torch.randn(n, n, device="cuda")
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(6):
                c = a @ b
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 6
            tflops = 2 * n ** 3 / dt / 1e12
            print(f"{n}x{n} matmul  {dt*1000:.1f} ms  ~{tflops:.1f} TFLOP/s")
            print(f"peak memory     {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
            del a, b, c
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"FAILED: {exc}")
            ok = False

        head("gradient step")
        try:
            import torch.nn as nn
            model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 4)).cuda()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            x = torch.randn(64, 256, device="cuda")
            y = torch.randint(0, 4, (64,), device="cuda")
            before = None
            for step in range(12):
                loss = nn.functional.cross_entropy(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
                if before is None:
                    before = loss.item()
            print(f"loss {before:.4f} -> {loss.item():.4f}  (must decrease)")
            if loss.item() >= before:
                print("WARNING: loss did not decrease. Autograd or the optimiser is wrong.")
                ok = False
        except Exception as exc:
            print(f"FAILED: {exc}")
            ok = False

    head("geospatial stack")
    for mod in ["numpy", "rasterio", "geopandas", "shapely", "laspy", "sklearn"]:
        try:
            m = __import__(mod)
            print(f"  {mod:<12} {getattr(m, '__version__', '?')}")
        except Exception:
            print(f"  {mod:<12} MISSING")
            ok = False

    if args.laz:
        head("read a real LAZ tile")
        try:
            import laspy, numpy as np
            with laspy.open(args.laz) as fh:
                pts = fh.read()
            x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
            area = max((x.max() - x.min()) * (y.max() - y.min()), 1.0)
            print(f"points     {len(x):,}")
            print(f"extent     {x.min():.1f},{y.min():.1f} to {x.max():.1f},{y.max():.1f}")
            print(f"z range    {z.min():.1f} to {z.max():.1f} m")
            print(f"density    {len(x)/area:.2f} points/m2")
            cls, counts = np.unique(np.asarray(pts.classification), return_counts=True)
            names = {1: "unclassified", 2: "ground", 3: "low veg", 4: "med veg",
                     5: "high veg", 6: "building", 9: "water"}
            for c, n in zip(cls, counts):
                print(f"  class {c:<3}{names.get(int(c),''):<14}{n:>10,}  {100*n/len(x):5.1f}%")
            print("\nNote the density. The published models train far denser than this.")
            print("That gap is your research question, so record the real number.")
        except Exception as exc:
            print(f"FAILED: {exc}")
            ok = False

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED, fix the above before going further")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
