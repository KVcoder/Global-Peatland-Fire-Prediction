#!/usr/bin/env python3
"""
viz_valid_geotiff.py

Visualize a GeoTIFF and highlight all VALID pixels (not NoData, finite).
- Default: show raster values (masked where invalid) + dark-green valid overlay
- --mask-only: show ONLY the valid footprint (dark green = valid)

Valid pixel rule:
  valid = np.isfinite(value) AND value != nodata

Deps:
  pip install rasterio numpy matplotlib

Examples:
  python viz_valid_geotiff.py path/to/file.tif
  python viz_valid_geotiff.py path/to/file.tif --band 1 --out viz.png
  python viz_valid_geotiff.py path/to/file.tif --mask-only --out mask.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from matplotlib.colors import ListedColormap


# 0 = transparent, 1 = dark green
VALID_DARK_GREEN = ListedColormap([
    (0, 0, 0, 0),          # invalid: transparent
    (0.0, 0.35, 0.0, 1.0), # valid: dark green
])


def _pick_out_shape(w: int, h: int, max_dim: int) -> tuple[int, int]:
    """Choose an output shape (h, w) that fits within max_dim while preserving aspect."""
    if max(w, h) <= max_dim:
        return h, w
    scale = max_dim / float(max(w, h))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return out_h, out_w


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize a GeoTIFF and highlight valid pixels (NoData=-9999 by default)."
    )
    ap.add_argument("tif", help="Path to GeoTIFF.")
    ap.add_argument("--band", type=int, default=1, help="1-based band index to visualize (default: 1).")
    ap.add_argument("--nodata", type=float, default=-9999.0, help="NoData value (default: -9999.0).")
    ap.add_argument("--mask-only", action="store_true", help="Show only the valid footprint (dark green).")
    ap.add_argument("--overlay-alpha", type=float, default=0.35, help="Alpha for valid overlay (default: 0.35).")
    ap.add_argument("--max-dim", type=int, default=2000, help="Downsample largest dimension to this for display (default: 2000).")
    ap.add_argument("--title", default=None, help="Optional plot title.")
    ap.add_argument("--out", default=None, help="Output PNG path. If omitted, opens a window.")
    args = ap.parse_args()

    tif = Path(args.tif)

    with rasterio.open(tif) as ds:
        if args.band < 1 or args.band > ds.count:
            raise SystemExit(f"--band must be in [1, {ds.count}] for this file.")

        out_h, out_w = _pick_out_shape(ds.width, ds.height, args.max_dim)

        # Read values with bilinear (nicer for continuous fields)
        arr_val = ds.read(
            args.band,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32, copy=False)

        # Read a second copy using nearest to compute the mask reliably
        arr_near = ds.read(
            args.band,
            out_shape=(out_h, out_w),
            resampling=Resampling.nearest,
        ).astype(np.float32, copy=False)

        nodata = args.nodata
        valid = np.isfinite(arr_near) & (arr_near != nodata)

        total = valid.size
        n_valid = int(valid.sum())
        pct_valid = 100.0 * (n_valid / total if total else 0.0)

        # Build a metadata-ish subtitle
        crs = ds.crs.to_string() if ds.crs else "None"
        res = ds.res if ds.res else (None, None)
        meta_line = f"{tif.name} | band {args.band} | display {out_w}x{out_h} | nodata={nodata} | valid={pct_valid:.2f}% | CRS={crs} | res={res}"

    fig, ax = plt.subplots(figsize=(10, 6))

    if args.mask_only:
        ax.imshow(valid.astype(np.uint8), cmap=VALID_DARK_GREEN, interpolation="nearest")
        ax.set_title(args.title or f"Valid footprint (dark green)\n{meta_line}")
    else:
        masked = np.ma.array(arr_val, mask=~valid)
        im = ax.imshow(masked, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Value (masked where invalid)")

        # Overlay valid footprint in dark green
        ax.imshow(valid.astype(np.uint8), cmap=VALID_DARK_GREEN, interpolation="nearest", alpha=args.overlay_alpha)

        ax.set_title(args.title or f"Values + valid overlay (dark green)\n{meta_line}")

    ax.set_xlabel("Column (display pixels)")
    ax.set_ylabel("Row (display pixels)")
    ax.set_aspect("equal")

    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
