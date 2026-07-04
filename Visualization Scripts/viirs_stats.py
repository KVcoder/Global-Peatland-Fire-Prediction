#!/usr/bin/env python3
"""
Summarize VIIRS → Peat Grid daily CSVs

What it does
------------
- Scans daily CSVs like: viirs_grid_YYYY-MM-DD.csv (your tool's output)
- Counts "fire cell-days" (unique (row,col) per day with ≥1 detection)
- Estimates "non-fire cell-days" as:
    (# peat cells) * (# days with CSVs) - (total fire cell-days)
  (i.e., all peat cells on all present days that had no detections)
- Aggregates totals by year, sensor flags, and day/night flags
- Reports detection intensity (sum of counts), FRP & confidence summary stats
- Finds hotspot cells (most fire days)

Outputs
-------
- Console summary
- CSVs in --report-dir (default: ./viirs_grid_summary):
    per_day_stats.csv
    per_year_stats.csv
    hotspots_top_cells.csv

Assumptions
-----------
- Daily CSV schema from your generator:
  date,row,col,lon_center,lat_center,count,max_frp,max_confidence,
  first_epoch,last_epoch,has_day,has_night,has_snpp,has_noaa20
- Booleans may be True/False or 0/1 as strings; this script normalizes them.
- "Non-fire" is an estimate based on peat mask cell count and observed days.
  If some days are missing (no CSV written), those days are *not* counted.

Speed tips
----------
- Install pyarrow: 2–4× faster CSV reads
- Use Python 3.10+ if possible
"""

import os, sys, glob, argparse, re
import numpy as np
import pandas as pd

# Progress bars
try:
    from tqdm.auto import tqdm
    _tqdm = True
except Exception:
    _tqdm = False
    def tqdm(x, **kwargs): return x

# Optional Arrow CSV engine
try:
    import pyarrow as pa  # noqa: F401
    _arrow = True
except Exception:
    _arrow = False

# Raster handling to count peat cells
try:
    import rasterio
except Exception:
    print("Please install deps:\n  pip install pandas rasterio tqdm pyarrow", file=sys.stderr)
    raise

FNAME_RE = re.compile(r"viirs_grid_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)

def parse_args():
    ap = argparse.ArgumentParser(description="Summarize VIIRS→Peat per-day grid CSVs.")
    ap.add_argument("--indir",   required=True, help="Folder with viirs_grid_YYYY-MM-DD.csv files")
    ap.add_argument("--mask",    required=True, help="Peat mask GeoTIFF used to generate the CSVs")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Peat if mask_value > threshold (must match your processing)")
    ap.add_argument("--pattern", default="viirs_grid_*.csv", help="Glob pattern for daily CSVs")
    ap.add_argument("--report-dir", default="viirs_grid_summary", help="Where to write summary CSVs")
    ap.add_argument("--topk", type=int, default=50, help="How many hotspot cells to output")
    ap.add_argument("--sample-days", type=int, default=0,
                    help="Optional: only process the first N day-files (for quick tests)")
    ap.add_argument("--quiet", action="store_true", help="Reduce console output")
    return ap.parse_args()

def count_peat_cells(mask_path: str, threshold: float) -> int:
    with rasterio.open(mask_path) as ds:
        band = ds.read(1, masked=False).astype("float64")
        nodata = ds.nodata
        keep = np.isfinite(band)
        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            keep &= (band != nodata)
        keep &= (band > threshold)
        return int(np.count_nonzero(keep))

def list_day_files(indir: str, pattern: str):
    files = sorted(glob.glob(os.path.join(indir, pattern)))
    # Keep only those that look like viirs_grid_YYYY-MM-DD.csv
    out = []
    for f in files:
        if FNAME_RE.search(os.path.basename(f)):
            out.append(f)
    return out

def parse_date_from_name(path: str) -> str:
    m = FNAME_RE.search(os.path.basename(path))
    if m: return m.group(1)
    return ""

def to_bool_series(s: pd.Series) -> pd.Series:
    # Accept True/False, 1/0, "true"/"false", "1"/"0"
    if s is None:
        return pd.Series([False] * 0, dtype="bool")
    if s.dtype == bool:
        return s
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(["true", "t", "1", "yes", "y"])

def read_day(path: str):
    usecols = ["date","row","col","count","max_frp","max_confidence",
               "has_day","has_night","has_snpp","has_noaa20","lon_center","lat_center"]
    dtypes = {
        "row": "Int32", "col": "Int32",
        "count": "Int64",
        "max_frp": "float64",
        "max_confidence": "float64",
        "lon_center": "float64",
        "lat_center": "float64",
    }
    kwargs = dict(usecols=[c for c in usecols], dtype=dtypes, low_memory=False)
    if _arrow:
        kwargs["engine"] = "pyarrow"
    df = pd.read_csv(path, **kwargs)

    # Normalize date (prefer file name if column missing or empty)
    if "date" not in df.columns or df["date"].isna().all():
        df["date"] = parse_date_from_name(path)
    else:
        # Ensure YYYY-MM-DD
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Boolean normalization
    for col in ["has_day","has_night","has_snpp","has_noaa20"]:
        if col in df.columns:
            df[col] = to_bool_series(df[col]).astype("bool")
        else:
            df[col] = False

    # Guard: collapse accidental duplicates within a day
    # (if not using --merge-on-write in the generator)
    grouped = (df.groupby(["row","col"], as_index=False, sort=False)
                 .agg(date=("date","first"),
                      lon_center=("lon_center","first"),
                      lat_center=("lat_center","first"),
                      count=("count","sum"),
                      max_frp=("max_frp","max"),
                      max_confidence=("max_confidence","max"),
                      has_day=("has_day","any"),
                      has_night=("has_night","any"),
                      has_snpp=("has_snpp","any"),
                      has_noaa20=("has_noaa20","any")))
    return grouped

def main():
    args = parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    day_files = list_day_files(args.indir, args.pattern)
    if not day_files:
        print(f"No files match {os.path.join(args.indir, args.pattern)}", file=sys.stderr)
        sys.exit(2)
    if args.sample_days > 0:
        day_files = day_files[:args.sample_days]

    # Count peat cells
    peat_cells = count_peat_cells(args.mask, args.threshold)

    # Aggregation holders
    per_day_rows = []
    # For hotspots: count of days per (row,col)
    hotspot_counts = {}
    hotspot_meta = {}  # store one lon/lat example for the cell
    # Streaming sums for overall stats
    total_fire_cell_days = 0
    total_detection_events = 0  # sum of 'count'
    frp_vals = []          # sampled FRP for quick quantiles (reservoir)
    conf_vals = []         # sampled confidence
    rng = np.random.default_rng(123)
    reservoir_size = 500000  # adjust if you want tighter quantiles

    iterator = tqdm(day_files, desc="Days", unit="day") if _tqdm and not args.quiet else day_files
    for f in iterator:
        df = read_day(f)
        if df.empty:
            continue
        day = df["date"].iloc[0]

        # Fire cell-days for this day
        n_fire_cells = len(df)
        total_fire_cell_days += n_fire_cells

        # Sum of detections (intensity)
        day_detection_events = int(pd.to_numeric(df["count"], errors="coerce").fillna(0).sum())
        total_detection_events += day_detection_events

        # Day/Night/Sensor splits
        n_day = int(df["has_day"].sum())
        n_night = int(df["has_night"].sum())
        n_snpp = int(df["has_snpp"].sum())
        n_n20  = int(df["has_noaa20"].sum())

        # Non-fire estimate for this day
        n_non_fire_cells = max(peat_cells - n_fire_cells, 0)

        # Store per-day
        per_day_rows.append({
            "date": day,
            "fire_cell_days": n_fire_cells,
            "non_fire_cell_days_est": n_non_fire_cells,
            "detections_sum": day_detection_events,
            "has_day_cells": n_day,
            "has_night_cells": n_night,
            "has_SNPP_cells": n_snpp,
            "has_NOAA20_cells": n_n20
        })

        # Hotspots: increment each unique cell once per day
        w = df[["row","col"]].to_numpy(dtype=np.int64)
        # pack into a single int key (assuming width unknown — use tuple as key instead)
        for (r, c), lc, lt in zip(w, df["lon_center"], df["lat_center"]):
            key = (int(r), int(c))
            hotspot_counts[key] = hotspot_counts.get(key, 0) + 1
            if key not in hotspot_meta:
                # record a lon/lat example for the cell
                hotspot_meta[key] = (float(lc), float(lt))

        # Streaming reservoir sample for FRP and confidence
        # (for quantiles without storing everything)
        for col, store in (("max_frp", frp_vals), ("max_confidence", conf_vals)):
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            if len(store) < reservoir_size:
                # just extend
                take = int(min(reservoir_size - len(store), len(vals)))
                if take == len(vals):
                    store.extend(vals.tolist())
                else:
                    idx = rng.choice(len(vals), size=take, replace=False)
                    store.extend(vals[idx].tolist())
            else:
                # reservoir replacement
                for v in vals:
                    j = rng.integers(0, 10_000_000)
                    if j < reservoir_size:
                        store[j] = float(v)

    # Build per-day DataFrame
    per_day = pd.DataFrame(per_day_rows)
    if per_day.empty:
        print("No non-empty day files processed.", file=sys.stderr)
        sys.exit(3)

    # Totals
    n_days = len(per_day)
    total_non_fire_est = int(peat_cells * n_days - total_fire_cell_days)
    total_non_fire_est = max(total_non_fire_est, 0)

    # Per-year rollups
    per_day["year"] = pd.to_datetime(per_day["date"]).dt.year
    per_year = (per_day.groupby("year", as_index=False)
                .agg(days=("date","count"),
                     fire_cell_days=("fire_cell_days","sum"),
                     non_fire_cell_days_est=("non_fire_cell_days_est","sum"),
                     detections_sum=("detections_sum","sum"),
                     day_cells=("has_day_cells","sum"),
                     night_cells=("has_night_cells","sum"),
                     SNPP_cells=("has_SNPP_cells","sum"),
                     NOAA20_cells=("has_NOAA20_cells","sum")))

    # Hotspots top-K
    if hotspot_counts:
        hot_df = (pd.DataFrame(
            [(r, c, cnt, *hotspot_meta.get((r,c),(np.nan,np.nan))) for (r,c), cnt in hotspot_counts.items()],
            columns=["row","col","fire_days","lon_center","lat_center"])
            .sort_values("fire_days", ascending=False)
            .head(args.topk)
        )
    else:
        hot_df = pd.DataFrame(columns=["row","col","fire_days","lon_center","lat_center"])

    # FRP / confidence quantiles (approx from reservoir)
    def qtiles(arr):
        if len(arr) == 0:
            return {}
        qs = [0, 5, 25, 50, 75, 95, 100]
        vals = np.percentile(np.asarray(arr, dtype=np.float64), qs, method="linear")
        return dict(zip([f"p{q}" for q in qs], vals))

    frp_q = qtiles(frp_vals)
    conf_q = qtiles(conf_vals)

    # Write reports
    per_day_path  = os.path.join(args.report_dir, "per_day_stats.csv")
    per_year_path = os.path.join(args.report_dir, "per_year_stats.csv")
    hot_path      = os.path.join(args.report_dir, "hotspots_top_cells.csv")
    per_day.sort_values("date").to_csv(per_day_path, index=False)
    per_year.sort_values("year").to_csv(per_year_path, index=False)
    hot_df.to_csv(hot_path, index=False)

    # Console summary
    if not args.quiet:
        first_day = per_day["date"].min()
        last_day  = per_day["date"].max()
        print("\n=== VIIRS→Peat Daily Grid Summary ===")
        print(f"Files processed      : {n_days}")
        print(f"Date range           : {first_day} → {last_day}")
        print(f"Peat cells (mask)    : {peat_cells:,}")
        print(f"Fire cell-days (+)   : {total_fire_cell_days:,}")
        print(f"Non-fire cell-days ~ : {total_non_fire_est:,}   (estimate)")
        ratio = (total_fire_cell_days / (total_fire_cell_days + total_non_fire_est)
                 if (total_fire_cell_days + total_non_fire_est) > 0 else 0.0)
        print(f"Class balance (+ rate): {ratio:.6f}")
        print(f"Detections sum       : {total_detection_events:,} (sum of 'count' over all fire cell-days)")
        print("\nFRP max per cell-day (approx quantiles from reservoir):")
        for k in ["p0","p5","p25","p50","p75","p95","p100"]:
            if k in frp_q:
                print(f"  {k:>4}: {frp_q[k]:.3f}")
        print("Confidence max per cell-day (approx quantiles):")
        for k in ["p0","p5","p25","p50","p75","p95","p100"]:
            if k in conf_q:
                print(f"  {k:>4}: {conf_q[k]:.3f}")

        print("\nTop hotspots (by # fire days):")
        if not hot_df.empty:
            for i, row in hot_df.head(min(10, len(hot_df))).iterrows():
                print(f"  #{i+1:02d} cell (r{row.row},c{row.col})  days={int(row.fire_days)}  "
                      f"@ ({row.lat_center:.3f},{row.lon_center:.3f})")
        else:
            print("  (none)")

        print("\nWrote:")
        print(f"  {per_day_path}")
        print(f"  {per_year_path}")
        print(f"  {hot_path}")
        print()

if __name__ == "__main__":
    main()
