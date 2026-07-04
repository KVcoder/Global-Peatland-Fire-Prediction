#!/usr/bin/env python3
"""
view_peat_mask_zarr.py

Visualize a peat_mask-only Zarr store (output from extract_and_mask_crop_smap_peat_mask.py).
Highlights "valid" values (inside bbox) vs NoData / outside bbox.

What it does:
- Opens Zarr v2 directory store OR .zip store
- Loads `peat_mask` (must be 2D)
- Auto-detects NoData from array attributes if present, else uses --nodata
- Builds a boolean "valid" mask:
    valid = (peat_mask != nodata) AND finite
  (for integer masks, this is effectively peat_mask != nodata)
- Displays:
  1) The peat_mask image
  2) The valid-mask overlay highlight
  3) A zoomed-in view around valid pixels (optional, enabled by default)
- Prints counts of valid pixels and unique values

Requirements:
  pip install zarr fsspec numpy matplotlib

Examples:
  python view_peat_mask_zarr.py --zarr 2023_SE_Asia_mask.zarr
  python view_peat_mask_zarr.py --zarr 2023_SE_Asia_mask.zarr --nodata 255
  python view_peat_mask_zarr.py --zarr 2023_SE_Asia_mask.zarr --no-zoom
  python view_peat_mask_zarr.py --zarr 2023_SE_Asia_mask.zip
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import zarr
import fsspec
import matplotlib.pyplot as plt


def open_zarr_group(path: str):
    # Support .zip zarr or directory store
    if path.lower().endswith(".zip"):
        store = fsspec.get_mapper(f"zip://{path}")
        # force v2 (your stores are v2)
        return zarr.open_group(store=store, mode="r", zarr_version=2)
    return zarr.open_group(path, mode="r", zarr_version=2)


def get_nodata(arr, nodata_arg: Optional[float]) -> Optional[float]:
    # Try common attribute names we set
    attrs = dict(getattr(arr, "attrs", {}))
    for k in ["nodata_value", "_FillValue", "fill_value", "nodata"]:
        if k in attrs:
            return attrs[k]
    return nodata_arg


def tight_bbox(mask: np.ndarray, pad: int = 10) -> Optional[Tuple[slice, slice]]:
    """Return a padded tight bbox around True pixels, or None if empty."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(mask.shape[0], y1 + pad)
    x1 = min(mask.shape[1], x1 + pad)
    return slice(y0, y1), slice(x0, x1)


def main():
    ap = argparse.ArgumentParser(description="Visualize peat_mask-only Zarr and highlight valid values.")
    ap.add_argument("--zarr", required=True, help="Path to peat_mask-only store (.zarr dir or .zip)")
    ap.add_argument("--nodata", type=float, default=None, help="Override nodata if not in attrs (e.g., 255)")
    ap.add_argument("--no-zoom", action="store_true", help="Disable zoomed-in view around valid pixels")
    ap.add_argument("--pad", type=int, default=10, help="Padding (pixels) around valid region for zoom view")
    ap.add_argument("--max-unique", type=int, default=20, help="Print up to this many unique values")
    ap.add_argument("--title", type=str, default=None, help="Custom plot title")
    args = ap.parse_args()

    g = open_zarr_group(args.zarr)

    if "peat_mask" not in g:
        raise KeyError(f"'peat_mask' not found. Arrays: {list(g.array_keys())}")

    arr = g["peat_mask"]
    if arr.ndim != 2:
        raise ValueError(f"Expected peat_mask to be 2D, got shape={arr.shape} (ndim={arr.ndim})")

    peat = arr[:]  # load into memory (2D only, should be fine)
    nodata = get_nodata(arr, args.nodata)

    # Build valid mask
    if nodata is None:
        valid = np.isfinite(peat)  # floats
        print("[warn] No nodata found; treating all finite pixels as valid.")
    else:
        # cast safely to compare
        try:
            nod = np.array(nodata).astype(peat.dtype).item()
        except Exception:
            nod = nodata
        valid = np.isfinite(peat) & (peat != nod)

    # Stats
    total = peat.size
    n_valid = int(valid.sum())
    print(f"[info] shape={peat.shape} dtype={peat.dtype}")
    print(f"[info] nodata={nodata}")
    print(f"[info] valid pixels: {n_valid}/{total} ({(100.0*n_valid/total):.4f}%)")

    # Unique values summary (for integer masks this is helpful)
    try:
        uniq = np.unique(peat[valid]) if n_valid > 0 else np.array([])
        if uniq.size > 0:
            show = uniq[: args.max_unique]
            print(f"[info] unique values among valid (showing up to {args.max_unique}): {show}")
            if uniq.size > args.max_unique:
                print(f"[info] ... ({uniq.size - args.max_unique} more)")
        else:
            print("[info] No valid pixels to summarize.")
    except Exception as e:
        print(f"[warn] Could not compute unique values: {e}")

    # Plot 1: peat_mask raw
    fig1 = plt.figure()
    plt.imshow(peat, interpolation="nearest")
    plt.colorbar()
    plt.title(args.title or "peat_mask (raw values)")
    plt.tight_layout()

    # Plot 2: overlay valid mask highlight
    fig2 = plt.figure()
    plt.imshow(peat, interpolation="nearest")
    # Overlay: mask False as transparent, True as solid highlight (colormap default)
    overlay = np.ma.masked_where(~valid, valid.astype(np.uint8))
    plt.imshow(overlay, interpolation="nearest", alpha=0.6)
    plt.title("Valid pixels highlighted (overlay)")
    plt.tight_layout()

    # Plot 3: zoomed bbox around valid pixels
    if not args.no_zoom:
        bb = tight_bbox(valid, pad=args.pad)
        if bb is None:
            print("[info] Zoom view skipped: no valid pixels.")
        else:
            ys, xs = bb
            peat_zoom = peat[ys, xs]
            valid_zoom = valid[ys, xs]

            fig3 = plt.figure()
            plt.imshow(peat_zoom, interpolation="nearest")
            overlay2 = np.ma.masked_where(~valid_zoom, valid_zoom.astype(np.uint8))
            plt.imshow(overlay2, interpolation="nearest", alpha=0.6)
            plt.title(f"Zoom around valid region (pad={args.pad})")
            plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
