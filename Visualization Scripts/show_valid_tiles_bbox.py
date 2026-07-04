#!/usr/bin/env python3
"""
Interactive GeoTIFF viewer with draggable/resizable bounding box that displays
the number of valid pixels inside the box live.

Usage:
  python interactive_valid_bbox.py --tif input.tif
  python interactive_valid_bbox.py --tif input.tif --band 1 --nodata -9999

Notes:
- Requires a GUI matplotlib backend (Qt5Agg, TkAgg, etc.)
- If running in Jupyter, try: %matplotlib qt
"""

import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


def robust_minmax(a: np.ndarray, valid: np.ndarray, pmin: float, pmax: float):
    vals = a[valid]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = np.percentile(vals, pmin)
    vmax = np.percentile(vals, pmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return 0.0, 1.0
    return float(vmin), float(vmax)


def clamp_int(v, lo, hi):
    return int(max(lo, min(hi, v)))


def extents_to_slices(extents, width, height):
    """
    extents: (xmin, xmax, ymin, ymax) in image data coords (pixels).
    Returns slices (yslice, xslice) clamped to array bounds.
    """
    xmin, xmax, ymin, ymax = extents
    x0 = int(np.floor(min(xmin, xmax)))
    x1 = int(np.ceil(max(xmin, xmax)))
    y0 = int(np.floor(min(ymin, ymax)))
    y1 = int(np.ceil(max(ymin, ymax)))

    # Clamp to bounds. Note slice end is exclusive.
    x0 = clamp_int(x0, 0, width)
    x1 = clamp_int(x1, 0, width)
    y0 = clamp_int(y0, 0, height)
    y1 = clamp_int(y1, 0, height)

    # Ensure non-empty
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)

    return slice(y0, y1), slice(x0, x1), (x0, x1, y0, y1)


def main():
    ap = argparse.ArgumentParser(description="Interactive GeoTIFF bbox valid-pixel counter.")
    ap.add_argument("--tif", required=True, help="Path to input GeoTIFF")
    ap.add_argument("--band", type=int, default=1, help="Band index (1-based). Default: 1")
    ap.add_argument("--nodata", type=float, default=None, help="Override nodata (default: read from file)")
    ap.add_argument("--pmin", type=float, default=2.0, help="Display scaling percentile min (default: 2)")
    ap.add_argument("--pmax", type=float, default=98.0, help="Display scaling percentile max (default: 98)")
    args = ap.parse_args()

    with rasterio.open(args.tif) as src:
        if args.band < 1 or args.band > src.count:
            raise SystemExit(f"--band must be in [1, {src.count}] for this file.")
        arr = src.read(args.band).astype(np.float32)
        nodata = src.nodata if args.nodata is None else args.nodata

    valid = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        valid &= (arr != nodata)

    h, w = arr.shape
    total_valid = int(valid.sum())

    vmin, vmax = robust_minmax(arr, valid, args.pmin, args.pmax)
    base = np.clip(arr, vmin, vmax)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title("Drag/resize/move the box — valid pixels counted live")
    im = ax.imshow(base, vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    # Text overlay
    info = ax.text(
        0.01, 0.99,
        "",
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
        fontsize=10
    )

    def update_from_extents(extents):
        ysl, xsl, (x0, x1, y0, y1) = extents_to_slices(extents, w, h)
        n = int(valid[ysl, xsl].sum())
        area = int((y1 - y0) * (x1 - x0))
        pct = (n / area * 100.0) if area else 0.0
        info.set_text(
            f"Box pixels: {area:,}\n"
            f"Valid in box: {n:,} ({pct:.2f}%)\n"
            f"Coords (px): x[{x0}:{x1})  y[{y0}:{y1})\n"
            f"Total valid in image: {total_valid:,}"
        )
        fig.canvas.draw_idle()

    # Initialize with a default box
    default_extents = (w * 0.25, w * 0.75, h * 0.25, h * 0.75)
    update_from_extents(default_extents)

    def onselect(eclick, erelease):
        # Called on mouse release
        update_from_extents(rs.extents)

    # Some matplotlib versions support live move callback:
    def onmove(eclick, erelease):
        update_from_extents(rs.extents)

    # Create selector (try with live move support, fall back if older mpl)
    try:
        rs = RectangleSelector(
            ax,
            onselect=onselect,
            interactive=True,
            useblit=True,
            button=[1],
            minspanx=1, minspany=1,
            spancoords="pixels",
            drag_from_anywhere=True,
            onmove_callback=onmove,  # live updates while dragging (if supported)
        )
    except TypeError:
        # Older matplotlib: no onmove_callback
        rs = RectangleSelector(
            ax,
            onselect=onselect,
            interactive=True,
            useblit=True,
            button=[1],
            minspanx=1, minspany=1,
            spancoords="pixels",
            drag_from_anywhere=True,
        )

        # Best-effort live updates via motion events
        last = {"extents": None}

        def motion(_event):
            ext = getattr(rs, "extents", None)
            if ext is None:
                return
            # Avoid redrawing if nothing changed
            if last["extents"] is not None and np.allclose(ext, last["extents"]):
                return
            last["extents"] = tuple(ext)
            update_from_extents(ext)

        fig.canvas.mpl_connect("motion_notify_event", motion)

    # Set initial rectangle
    rs.extents = default_extents

    print(
        "Controls:\n"
        "  - Left-drag to draw/select box\n"
        "  - Drag edges/handles to resize\n"
        "  - Drag inside box to move (if supported by your mpl version)\n"
        "Close the window to exit."
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
