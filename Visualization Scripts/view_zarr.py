#!/usr/bin/env python3
"""
view_zarr_interactive.py

Interactive viewer for a Zarr store with:
  - x (lon) 1-D axis
  - y (lat) 1-D axis (can be increasing or decreasing)
  - main var (T, C, H, W), e.g. "field", "era5land", "smap_wtd"
  - optional peat_mask (H, W)

Features:
  - Matplotlib image display of a chosen time index
  - Slider to change time (t)
  - RectangleSelector to drag/resize a bbox in lon/lat
  - Live valid pixel count inside the bbox from the full-res data
"""

import argparse
import numpy as np
import zarr
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RectangleSelector, CheckButtons

def is_increasing(a: np.ndarray) -> bool:
    return bool(np.all(np.diff(a) >= 0))

def coord_to_index(coord: np.ndarray, value: float) -> int:
    """
    Map a coordinate value to nearest index, handling increasing or decreasing coord arrays.
    Uses searchsorted on a monotonic axis.
    """
    inc = is_increasing(coord)
    if inc:
        i = int(np.searchsorted(coord, value, side="left"))
        i = max(0, min(i, len(coord) - 1))
        # snap to nearest neighbor
        if 0 < i < len(coord) - 1:
            if abs(coord[i] - value) > abs(coord[i - 1] - value):
                i -= 1
        return i
    else:
        # decreasing: search on reversed increasing copy
        rev = coord[::-1]
        j = int(np.searchsorted(rev, value, side="left"))
        j = max(0, min(j, len(rev) - 1))
        if 0 < j < len(rev) - 1:
            if abs(rev[j] - value) > abs(rev[j - 1] - value):
                j -= 1
        return (len(coord) - 1) - j

def clamp_slice(a: int, b: int, n: int) -> slice:
    lo = max(0, min(a, b))
    hi = min(n, max(a, b) + 1)
    return slice(lo, hi)

def finite_valid_mask(arr: np.ndarray, fill_value) -> np.ndarray:
    m = np.isfinite(arr)
    if fill_value is not None and np.isfinite(fill_value):
        m &= (arr != fill_value)
    return m

def robust_vmin_vmax(img: np.ndarray) -> tuple[float, float]:
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
    return float(lo), float(hi)

class ZarrViewer:
    def __init__(self, zarr_path: str, var_name: str, t0: int, c0: int,
                 max_display_pixels: int = 1200):
        self.z = zarr.open_group(zarr_path, mode="r")
        if "x" not in self.z or "y" not in self.z:
            raise KeyError(f"Expected 'x' and 'y' arrays. Found: {list(self.z.array_keys())}")

        self.x = self.z["x"][:]
        self.y = self.z["y"][:]
        self.x_inc = is_increasing(self.x)
        self.y_inc = is_increasing(self.y)

        if var_name not in self.z:
            raise KeyError(f"'{var_name}' not found. Arrays: {list(self.z.array_keys())}")

        self.var_name = var_name
        self.a = self.z[var_name]
        if self.a.ndim != 4:
            raise ValueError(f"Expected {var_name} to be 4D (T,C,H,W). Got {self.a.shape}")
        self.T, self.C, self.H, self.W = self.a.shape

        self.fillv = getattr(self.a, "fill_value", None)

        self.pm = self.z["peat_mask"] if "peat_mask" in self.z else None
        self.show_pm = False

        # time/channel init
        self.t = int(np.clip(t0, 0, self.T - 1))
        self.c = int(np.clip(c0, 0, self.C - 1))

        # Display downsample steps (keep display manageable)
        step_y = max(1, int(np.ceil(self.H / max_display_pixels)))
        step_x = max(1, int(np.ceil(self.W / max_display_pixels)))
        self.step_y = step_y
        self.step_x = step_x
        self.ys_disp = slice(0, self.H, self.step_y)
        self.xs_disp = slice(0, self.W, self.step_x)

        # initial bbox: center-ish (in coords)
        self.bbox = None  # (lon0, lon1, lat0, lat1) in coords

        # figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.22, right=0.82)

        self.im = None
        self.pm_im = None
        self.text = self.fig.text(0.02, 0.02, "", fontsize=10, family="monospace")

        self._build_ui()

    def _extent_for_display(self):
        # extent = [xmin, xmax, ymin, ymax]
        x0 = float(self.x[self.xs_disp.start])
        x1 = float(self.x[min(self.W - 1, self.xs_disp.stop - 1)])
        y0 = float(self.y[self.ys_disp.start])
        y1 = float(self.y[min(self.H - 1, self.ys_disp.stop - 1)])

        # Matplotlib expects extent as (left,right,bottom,top). We can set origin accordingly.
        # We'll use origin='upper' if y is decreasing so it visually matches common raster orientation.
        if self.y_inc:
            extent = [x0, x1, y0, y1]
            origin = "lower"
        else:
            extent = [x0, x1, y1, y0]  # swap so bottom<top
            origin = "upper"
        return extent, origin

    def _read_display_image(self, t: int):
        img = self.a[t, self.c, self.ys_disp, self.xs_disp]
        img = np.asarray(img)
        return img

    def _read_display_peatmask(self):
        if self.pm is None:
            return None
        pm = self.pm[self.ys_disp, self.xs_disp]
        return np.asarray(pm)

    def _build_ui(self):
        self.ax.set_title(f"{self.var_name} (t={self.t}, c={self.c})")
        self.ax.set_xlabel("Longitude (x)")
        self.ax.set_ylabel("Latitude (y)")
        self.ax.grid(False)

        extent, origin = self._extent_for_display()
        img = self._read_display_image(self.t)
        vmin, vmax = robust_vmin_vmax(img)

        self.im = self.ax.imshow(
            img,
            extent=extent,
            origin=origin,
            interpolation="nearest",
            aspect="auto",
        )
        self.im.set_clim(vmin, vmax)
        self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)

        # peat_mask overlay (optional)
        if self.pm is not None:
            pm = self._read_display_peatmask()
            self.pm_im = self.ax.imshow(
                pm,
                extent=extent,
                origin=origin,
                interpolation="nearest",
                alpha=0.0,  # start hidden
                aspect="auto",
            )

        # Time slider
        ax_slider = plt.axes([0.12, 0.10, 0.62, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label="t",
            valmin=0,
            valmax=self.T - 1,
            valinit=self.t,
            valstep=1
        )
        self.slider.on_changed(self._on_slider)

        # Checkbox for peat_mask
        ax_check = plt.axes([0.84, 0.72, 0.14, 0.12])
        labels = []
        actives = []
        if self.pm is not None:
            labels.append("peat_mask overlay")
            actives.append(False)
        self.check = CheckButtons(ax_check, labels, actives) if labels else None
        if self.check is not None:
            self.check.on_clicked(self._on_check)

        # Rectangle selector (interactive bbox)
        self.rs = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[1],  # left click
            minspanx=0.1,
            minspany=0.1,
            spancoords="data",
            interactive=True,
            drag_from_anywhere=True,
        )

        # Initial info text
        self._update_text(None)

    def _on_check(self, label):
        if label == "peat_mask overlay" and self.pm_im is not None:
            self.show_pm = not self.show_pm
            self.pm_im.set_alpha(0.35 if self.show_pm else 0.0)
            self.fig.canvas.draw_idle()

    def _on_slider(self, val):
        self.t = int(val)
        self._update_image()
        self._update_text(self.bbox)

    def _on_select(self, eclick, erelease):
        # coords in lon/lat
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 is None or x2 is None or y1 is None or y2 is None:
            return
        lon0, lon1 = float(min(x1, x2)), float(max(x1, x2))
        lat0, lat1 = float(min(y1, y2)), float(max(y1, y2))
        self.bbox = (lon0, lon1, lat0, lat1)
        self._update_text(self.bbox)

    def _update_image(self):
        img = self._read_display_image(self.t)
        vmin, vmax = robust_vmin_vmax(img)
        self.im.set_data(img)
        self.im.set_clim(vmin, vmax)
        self.ax.set_title(f"{self.var_name} (t={self.t}, c={self.c})")

        if self.pm_im is not None:
            pm = self._read_display_peatmask()
            self.pm_im.set_data(pm)

        self.fig.canvas.draw_idle()

    def _count_valid_in_bbox(self, bbox):
        lon0, lon1, lat0, lat1 = bbox

        # Convert bbox coords -> index slices
        x0 = coord_to_index(self.x, lon0)
        x1 = coord_to_index(self.x, lon1)
        y0 = coord_to_index(self.y, lat0)
        y1 = coord_to_index(self.y, lat1)

        xs = clamp_slice(x0, x1, self.W)
        ys = clamp_slice(y0, y1, self.H)

        # Read just that window at full resolution
        window = np.asarray(self.a[self.t, self.c, ys, xs])
        valid = finite_valid_mask(window, self.fillv)
        valid_count = int(valid.sum())

        pm_count = None
        if self.pm is not None:
            pmw = np.asarray(self.pm[ys, xs])
            pm_count = int((pmw != 0).sum())

        # Report also the "actual" coordinate edges of the selected indices
        lon_lo = float(self.x[xs.start])
        lon_hi = float(self.x[xs.stop - 1])
        lat_lo = float(self.y[ys.start])
        lat_hi = float(self.y[ys.stop - 1])

        return {
            "xs": xs, "ys": ys,
            "valid_count": valid_count,
            "pm_count": pm_count,
            "lon_lo": lon_lo, "lon_hi": lon_hi,
            "lat_lo": lat_lo, "lat_hi": lat_hi,
            "shape": window.shape
        }

    def _update_text(self, bbox):
        lines = []
        lines.append(f"Zarr var: {self.var_name}  shape={self.a.shape}  chunks={getattr(self.a,'chunks',None)}")
        lines.append(f"Axes: y(len={len(self.y)}) [{self.y.min():.6f}, {self.y.max():.6f}] ({'inc' if self.y_inc else 'dec'})"
                     f" | x(len={len(self.x)}) [{self.x.min():.6f}, {self.x.max():.6f}] ({'inc' if self.x_inc else 'dec'})")
        lines.append(f"t={self.t}/{self.T-1}  c={self.c}/{self.C-1}  fill_value={self.fillv}")

        if bbox is None:
            lines.append("BBox: (drag to select) — counts will appear here.")
        else:
            try:
                out = self._count_valid_in_bbox(bbox)
                lines.append(f"BBox (drawn): lon[{bbox[0]:.4f},{bbox[1]:.4f}] lat[{bbox[2]:.4f},{bbox[3]:.4f}]")
                lines.append(f"BBox (snapped): lon[{out['lon_lo']:.6f},{out['lon_hi']:.6f}] lat[{out['lat_lo']:.6f},{out['lat_hi']:.6f}]")
                lines.append(f"Window (H,W)={out['shape']}  valid_pixels={out['valid_count']:,}")
                if out["pm_count"] is not None:
                    lines.append(f"peat_mask_nonzero_in_bbox={out['pm_count']:,}")
            except Exception as e:
                lines.append(f"BBox: error counting valid pixels: {e}")

        self.text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-zarr", required=True, help="Path to input Zarr store")
    ap.add_argument("--var-name", default="field", help="Main var name (T,C,H,W)")
    ap.add_argument("--t", type=int, default=0, help="Initial time index")
    ap.add_argument("--c", type=int, default=0, help="Channel index (default 0)")
    ap.add_argument("--max-display-pixels", type=int, default=1200,
                    help="Downsample target for display (per axis). Full-res is used for bbox counts.")
    args = ap.parse_args()

    viewer = ZarrViewer(
        zarr_path=args.in_zarr,
        var_name=args.var_name,
        t0=args.t,
        c0=args.c,
        max_display_pixels=args.max_display_pixels,
    )
    viewer.show()

if __name__ == "__main__":
    main()
