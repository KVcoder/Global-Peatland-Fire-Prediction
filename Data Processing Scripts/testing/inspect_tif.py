#!/usr/bin/env python3
"""
inspect_tif.py — quick, human-friendly inspector for GeoTIFFs

Usage:
  python inspect_tif.py /path/to/file.tif [--dump-npy] [--dump-csv] [--sample 10]

What it does:
- Opens the .tif with rasterio
- Prints core metadata (size, bands, dtype(s), CRS, transform, bounds)
- Shows global & per-band tags (if any)
- Computes per-band stats (min/max/mean/std, valid/NaN counts)
- Reports masks, overviews, and approximate pixel size
- Saves a JSON report next to the input file
- Optionally dumps arrays:
    --dump-npy  : saves each band as .npy (fast & lossless)
    --dump-csv  : saves each band as CSV (only if total pixels <= 5e6 by default)
- Optionally prints a center sample window (--sample N prints NxN values for band 1)

Notes:
- Very large rasters can be huge to dump as CSV; the script will refuse unless small
  (you can override the size threshold by setting environment variable CSV_MAX_PX).
"""

import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path    
CSV_MAX_PX = int(os.environ.get("CSV_MAX_PX", "5000000"))  # 5 million px default


@dataclass
class BandStats:
    band: int
    dtype: str
    nodata: Optional[float]
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    valid_count: int
    nan_count: int
    has_mask: bool
    mask_valid_pct: Optional[float]


def human_crs(crs) -> Dict[str, Any]:
    if crs is None:
        return {"wkt": None, "epsg": None}
    epsg = None
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    wkt_short = None
    try:
        wkt_short = crs.to_wkt("WKT2_2019_SIMPLE")  # compact-ish
    except Exception:
        try:
            wkt_short = crs.to_wkt()
        except Exception:
            wkt_short = None
    return {"wkt": wkt_short, "epsg": epsg, "proj4": str(crs)}


def pixel_size_from_affine(transform):
    # Approximate square pixel size (meters or degrees depending on CRS)
    # Works for north-up rasters (no rotation/shear)
    try:
        sx = abs(transform.a)
        sy = abs(transform.e)
        return sx, sy
    except Exception:
        return None, None


def infer_dates_from_name(name: str) -> Dict[str, Any]:
    """
    Makes BEST-EFFORT guesses about temporal semantics from common filename patterns.
    Returns a dict of guesses (could be empty).
    """
    guess = {}
    # e.g., ..._2015_03.tif  (monthly)
    m = re.search(r'(\d{4})_(\d{2})\.tif$', name)
    if m:
        guess["pattern"] = "YYYY_MM"
        guess["year"] = int(m.group(1))
        guess["month"] = int(m.group(2))
    # e.g., ...yearly_2015.tif (daily stacks across year or multi-day bands)
    m2 = re.search(r'yearly_(\d{4})\.tif$', name)
    if m2:
        guess["pattern"] = "yearly_YYYY"
        guess["year"] = int(m2.group(1))
    return guess


def read_band_stats(ds, band_index: int) -> BandStats:
    # Read the band into memory for accurate stats. For huge rasters this can be big; if memory is tight,
    # consider doing a chunked stats pass (not implemented here for simplicity).
    arr = ds.read(band_index).astype("float64")  # safer for NaNs
    nodata = ds.nodatavals[band_index - 1] if ds.nodatavals else None

    if nodata is not None:
        arr[arr == nodata] = np.nan

    valid_mask = np.isfinite(arr)
    valid = arr[valid_mask]

    if valid.size == 0:
        stats = BandStats(
            band=band_index,
            dtype=str(arr.dtype),
            nodata=nodata,
            min=None,
            max=None,
            mean=None,
            std=None,
            valid_count=0,
            nan_count=arr.size,
            has_mask=ds.mask_flag_enums[band_index - 1] is not None if ds.mask_flag_enums else False,
            mask_valid_pct=None,
        )
    else:
        stats = BandStats(
            band=band_index,
            dtype=str(arr.dtype),
            nodata=nodata,
            min=float(np.nanmin(valid)),
            max=float(np.nanmax(valid)),
            mean=float(np.nanmean(valid)),
            std=float(np.nanstd(valid)),
            valid_count=int(valid.size),
            nan_count=int(np.count_nonzero(~valid_mask)),
            has_mask=ds.mask_flag_enums[band_index - 1] is not None if ds.mask_flag_enums else False,
            mask_valid_pct=None,
        )

    try:
        # If there is an alpha/mask band or internal mask, estimate valid % from it (0/255 semantics).
        m = ds.read_masks(band_index)
        stats.mask_valid_pct = float(np.count_nonzero(m) / m.size * 100.0)
    except Exception:
        pass

    return stats


def summarize_tif(path: str, dump_npy: bool, dump_csv: bool, sample: Optional[int]) -> Dict[str, Any]:
    with rasterio.open(path) as ds:
        basic = {
            "driver": ds.driver,
            "width": ds.width,
            "height": ds.height,
            "count_bands": ds.count,
            "dtypes": [str(dt) for dt in ds.dtypes],
            "nodatavals": list(ds.nodatavals) if ds.nodatavals else [None] * ds.count,
            "crs": human_crs(ds.crs),
            "transform": tuple(ds.transform),
            "bounds": ds.bounds._asdict() if hasattr(ds, "bounds") else None,
            "colorinterp": [str(ci) for ci in getattr(ds, "colorinterp", [])],
            "descriptions": list(ds.descriptions) if ds.descriptions else [None] * ds.count,
            "overviews_per_band": [ds.overviews(i + 1) for i in range(ds.count)],
            "res": getattr(ds, "res", None),
        }

        sx, sy = pixel_size_from_affine(ds.transform)
        basic["approx_pixel_size"] = (sx, sy)

        tags_global = ds.tags() or {}
        tags_per_band = [ds.tags(i + 1) for i in range(ds.count)]

        # Band stats
        band_stats: List[BandStats] = []
        for i in range(1, ds.count + 1):
            band_stats.append(read_band_stats(ds, i))

        # Optional small sample (center window on band 1)
        sample_window_info: Optional[Dict[str, Any]] = None
        if sample and sample > 0 and ds.count >= 1:
            c = sample
            row_c = ds.height // 2
            col_c = ds.width // 2
            r0 = max(0, row_c - c // 2); c0 = max(0, col_c - c // 2)
            win = Window.from_slices((r0, min(r0 + c, ds.height)), (c0, min(c0 + c, ds.width)))
            arr = ds.read(1, window=win)
            sample_window_info = {
                "window_rc": [int(r0), int(c0), int(win.height), int(win.width)],
                "values": arr.tolist(),
            }

        # Optional dumps
        out_dir = Path(path).with_suffix("")  # /path/to/file -> /path/to/file
        out_dir.mkdir(exist_ok=True, parents=True)

        dumped = []
        total_px = ds.width * ds.height
        if dump_csv and total_px > CSV_MAX_PX:
            print(f"[WARN] CSV dump skipped: raster has {total_px:,} px which exceeds CSV_MAX_PX={CSV_MAX_PX:,}. Use --dump-npy or raise CSV_MAX_PX.")
            dump_csv = False

        for i in range(1, ds.count + 1):
            arr = ds.read(i)
            # Apply nodata as NaN for dumps to be consistent with stats
            nod = ds.nodatavals[i - 1] if ds.nodatavals else None
            if nod is not None:
                arr = arr.astype("float64")
                arr[arr == nod] = np.nan

            if dump_npy:
                f = out_dir / f"band{i:02d}.npy"
                np.save(f, arr)
                dumped.append(str(f))

            if dump_csv:
                f = out_dir / f"band{i:02d}.csv"
                # For large rasters this will be big; enabled only if below threshold
                np.savetxt(f, arr, delimiter=",", fmt="%.10g")
                dumped.append(str(f))

        # Assemble report
        report = {
            "file": path,
            "name": Path(path).name,
            "basic": basic,
            "tags_global": tags_global,
            "tags_per_band": tags_per_band,
            "band_stats": [asdict(s) for s in band_stats],
            "sample_window": sample_window_info,
            "filename_time_guess": infer_dates_from_name(Path(path).name),
            "dumped_outputs": dumped,
        }

        # Write JSON report next to file (e.g., file.report.json)
        report_path = str(Path(path).with_suffix(".report.json"))
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report, report_path


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    path = argv[1]
    dump_npy = "--dump-npy" in argv
    dump_csv = "--dump-csv" in argv

    sample = None
    if "--sample" in argv:
        try:
            idx = argv.index("--sample")
            sample = int(argv[idx + 1])
        except Exception:
            print("[WARN] --sample given but no/invalid integer; ignoring.")

    if not Path(path).exists():
        print(f"[ERR] File not found: {path}")
        return 2

    report, report_path = summarize_tif(path, dump_npy, dump_csv, sample)
    # Pretty print core info to stdout
    basic = report["basic"]
    print("=" * 80)
    print(f"File: {report['file']}")
    print(f"Driver: {basic['driver']} | Size: {basic['width']} x {basic['height']} | Bands: {basic['count_bands']}")
    print(f"Dtypes: {basic['dtypes']} | NoData per band: {basic['nodatavals']}")
    print(f"CRS: EPSG={basic['crs']['epsg']} | PROJ4/WKT: {basic['crs']['proj4']}")
    print(f"Transform (affine): {basic['transform']} | Pixel size (approx): {basic['approx_pixel_size']}")
    print(f"Bounds: {basic['bounds']}")
    if report['filename_time_guess']:
        print(f"Filename time guess: {report['filename_time_guess']}")
    print("-" * 80)
    print("Global tags:")
    if report["tags_global"]:
        for k, v in report["tags_global"].items():
            print(f"  {k}: {v}")
    else:
        print("  (none)")

    print("-" * 80)
    print("Per-band summary:")
    for s in report["band_stats"]:
        print(f"  Band {s['band']:>2} | dtype={s['dtype']} | nodata={s['nodata']} | "
              f"min={s['min']} | max={s['max']} | mean={s['mean']} | std={s['std']} | "
              f"valid={s['valid_count']:,} | NaN={s['nan_count']:,} | "
              f"mask={s['has_mask']} ({s['mask_valid_pct']:.2f}% valid if available)")

    if basic.get("overviews_per_band"):
        print("-" * 80)
        print("Overviews per band (decimation factors):")
        for i, ovs in enumerate(basic["overviews_per_band"], start=1):
            print(f"  Band {i}: {ovs if ovs else '(none)'}")

    if report["sample_window"]:
        r0, c0, h, w = report["sample_window"]["window_rc"]
        print("-" * 80)
        print(f"Center sample window [r0={r0}, c0={c0}, h={h}, w={w}] (band 1):")
        # Print small table-like block (truncate if giant)
        vals = report["sample_window"]["values"]
        max_rows = min(20, len(vals))
        for row in vals[:max_rows]:
            print("  " + ", ".join(f"{x:.6g}" for x in row[:40]))

    print("-" * 80)
    print(f"Wrote JSON report → {report_path}")

    if report["dumped_outputs"]:
        print("Dumped data files:")
        for p in report["dumped_outputs"]:
            print(f"  {p}")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
