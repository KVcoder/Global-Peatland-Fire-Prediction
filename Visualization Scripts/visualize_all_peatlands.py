#!/usr/bin/env python3
"""
Visualize a GeoTIFF and highlight pixels with valid (non-nodata) data.

- Reads a selected band (default: 1)
- Computes valid mask: isfinite & != nodata
- Saves a PNG with an overlay highlighting valid pixels

Example:
  python viz_valid_geotiff.py --tif input.tif --out valid_overlay.png
"""

import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt


def robust_minmax(a: np.ndarray, valid: np.ndarray, pmin: float, pmax: float):
    """Percentile-based min/max for display scaling."""
    vals = a[valid]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = np.percentile(vals, pmin)
    vmax = np.percentile(vals, pmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
    return float(vmin), float(vmax)


def main():
    ap = argparse.ArgumentParser(description="Visualize a GeoTIFF and highlight valid pixels.")
    ap.add_argument("--tif", required=True, help="Path to input GeoTIFF")
    ap.add_argument("--out", default=None, help="Output PNG path (default: <tif_basename>_valid.png)")
    ap.add_argument("--band", type=int, default=1, help="Band index (1-based). Default: 1")
    ap.add_argument("--nodata", type=float, default=None, help="Override nodata value (default: read from file)")
    ap.add_argument("--alpha", type=float, default=0.35, help="Overlay alpha for valid mask (default: 0.35)")
    ap.add_argument("--pmin", type=float, default=2.0, help="Display scaling percentile min (default: 2)")
    ap.add_argument("--pmax", type=float, default=98.0, help="Display scaling percentile max (default: 98)")
    ap.add_argument("--mask-only", action="store_true", help="If set, only render the valid mask (no base raster).")
    args = ap.parse_args()

    with rasterio.open(args.tif) as src:
        if args.band < 1 or args.band > src.count:
            raise SystemExit(f"--band must be in [1, {src.count}] for this file.")
        arr = src.read(args.band).astype(np.float32)
        nodata = src.nodata if args.nodata is None else args.nodata

    # Valid where finite and not nodata (if nodata is defined)
    valid = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        valid &= (arr != nodata)

    # Output name
    if args.out is None:
        base = args.tif.rsplit(".", 1)[0]
        args.out = f"{base}_valid.png"

    # Plot
    plt.figure(figsize=(10, 8))

    if args.mask_only:
        # Show mask as black/white
        plt.imshow(valid.astype(np.uint8), interpolation="nearest")
        plt.title("Valid-data mask (1=valid, 0=nodata/invalid)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(args.out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {args.out}")
        return

    # Base raster display (robust scaling on valid pixels)
    vmin, vmax = robust_minmax(arr, valid, args.pmin, args.pmax)
    base = np.clip(arr, vmin, vmax)

    plt.imshow(base, vmin=vmin, vmax=vmax, interpolation="nearest")
    # Overlay valid pixels in red
    overlay = np.zeros((valid.shape[0], valid.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = 1.0  # R
    overlay[..., 3] = valid.astype(np.float32) * float(args.alpha)  # alpha only where valid
    plt.imshow(overlay, interpolation="nearest")

    total = valid.size
    n_valid = int(valid.sum())
    pct = (n_valid / total * 100.0) if total else 0.0

    plt.title(f"Valid pixels highlighted (red): {n_valid:,}/{total:,} ({pct:.2f}%)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
