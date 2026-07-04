#!/usr/bin/env python3
"""
Extract a few daily layers from ERA5-Land monthly stacks and display them.

Usage:
  python extract_days_show.py \
    --tif era5land_monthly_stack_2019_06.tif era5land_monthly_stack_2019_07.tif \
    --var t2m_C \
    --dates 2019-06-05,2019-06-06,2019-07-01 \
    --save-dir out_days  # optional: writes per-day GeoTIFFs

Notes
- Variable names are the converted aliases (e.g., t2m_C, tp_mm, u10_ms).
- Dates can be YYYYMMDD or YYYY-MM-DD.
- If you packed to Int16, the script decodes using known scale factors.
"""

import argparse
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
import matplotlib.pyplot as plt

# Known Int16 packing divisors (match your export script PACK table).
# If your files are Float32 (no packing), these are ignored.
DIVISOR = {
    # ÷100
    "t2m_C": 100, "d2m_C": 100, "tsk_C": 100,
    "stl1_C": 100, "stl2_C": 100, "stl3_C": 100, "stl4_C": 100,
    "u10_ms": 100, "v10_ms": 100,
    "lai_hv": 100, "lai_lv": 100,
    # ÷10
    "tp_mm": 10, "sf_mm": 10, "sm_mm": 10, "ro_mm": 10,
    "sro_mm": 10, "ssro_mm": 10, "sdwe_mm": 10, "src_mm": 10, "evap_mm": 10,
}

def norm_date(s: str) -> str:
    s = s.strip()
    if re.fullmatch(r"\d{8}", s):
        return s
    # allow YYYY-MM-DD
    return datetime.strptime(s, "%Y-%m-%d").strftime("%Y%m%d")

def collect_matches(
    tifs: List[str], var_alias: str, yyyymmdds: List[str]
) -> List[Tuple[str, int, str]]:
    """
    Return list of (tif_path, band_index_1based, band_name) for requested dates+var.
    """
    needed = set(yyyymmdds)
    found = []
    suffix = f"_{var_alias}"
    for path in tifs:
        with rasterio.open(path) as ds:
            names = list(ds.descriptions or [])
            for i, name in enumerate(names, start=1):
                if not name:  # sometimes descriptions may be None
                    continue
                if name.endswith(suffix) and name[:8] in needed:
                    found.append((path, i, name))
    # Keep results ordered by date
    found.sort(key=lambda x: x[2])
    missing = sorted(needed - set(n[:8] for _,_,n in found))
    if missing:
        print(f"Warning: missing {var_alias} for date(s): {', '.join(missing)}")
    return found

def read_stack(matches: List[Tuple[str, int, str]]) -> Tuple[np.ndarray, Dict, List[str]]:
    """
    Read matched bands into a stack (T,H,W). Returns data, profile, and date strings.
    """
    arrays = []
    dates = []
    profile = None
    for path, idx, name in matches:
        with rasterio.open(path) as ds:
            arr = ds.read(idx)  # 2D
            if profile is None:
                profile = ds.profile.copy()
                # force single-band for future writes
                profile.update(count=1)
            arrays.append(arr)
            dates.append(name.split("_")[0])
    if not arrays:
        raise SystemExit("No requested bands found. Check --dates and --var exist in the files.")
    data = np.stack(arrays, axis=0)  # (T,H,W)
    return data, profile, dates

def maybe_decode(data: np.ndarray, var_alias: str, dtype) -> np.ndarray:
    """
    If data is Int16 and var has a known divisor, decode to float.
    """
    if np.issubdtype(dtype, np.integer) and var_alias in DIVISOR:
        div = DIVISOR[var_alias]
        out = data.astype("float32") / float(div)
        return out
    return data  # already float or unknown packing

def show_days(data: np.ndarray, dates: List[str], var_alias: str):
    """
    Display each day in a row (or grid if many).
    """
    T = data.shape[0]
    ncols = min(T, 4)
    nrows = (T + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)
    vmin, vmax = np.nanpercentile(data[np.isfinite(data)], [2, 98]) if np.isfinite(data).any() else (None, None)
    k = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if k < T:
                im = ax.imshow(data[k], vmin=vmin, vmax=vmax)
                ax.set_title(f"{dates[k]}  {var_alias}")
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis("off")
            k += 1
    plt.tight_layout()
    plt.show()

def save_geotiffs(
    data: np.ndarray, profile: Dict, dates: List[str],
    var_alias: str, out_dir: str
):
    os.makedirs(out_dir, exist_ok=True)
    for i, d in enumerate(dates):
        path = os.path.join(out_dir, f"{var_alias}_{d}.tif")
        with rasterio.open(path, "w", **profile) as dst:
            dst.set_descriptions([f"{d}_{var_alias}"])
            dst.write(data[i], 1)
        print("Wrote", path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", nargs="+", required=True, help="Path(s) to monthly stack GeoTIFFs")
    ap.add_argument("--var", required=True, help="Variable alias, e.g., t2m_C, tp_mm, u10_ms")
    ap.add_argument("--dates", required=True, help="Comma-separated dates (YYYYMMDD or YYYY-MM-DD)")
    ap.add_argument("--save-dir", default=None, help="Optional output directory to write per-day GeoTIFFs")
    args = ap.parse_args()

    dates = [norm_date(s) for s in args.dates.split(",")]
    matches = collect_matches(args.tif, args.var, dates)
    data, profile, found_dates = read_stack(matches)

    # Decode if Int16-packed
    data = maybe_decode(data, args.var, profile.get("dtype"))
    # Update profile to Float32 if decoded
    if data.dtype.kind == "f":
        profile.update(dtype="float32")

    # Display
    show_days(data, found_dates, args.var)

    # Optional write per-day files
    if args.save_dir:
        save_geotiffs(data, profile, found_dates, args.var, args.save_dir)

if __name__ == "__main__":
    main()
