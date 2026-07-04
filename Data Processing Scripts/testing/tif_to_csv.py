
#!/usr/bin/env python3
"""
tif_to_csv.py

Convert a stacked GeoTIFF (e.g., ERA5-style stack) to a CSV you can read.

By default, writes a LONG format CSV with columns:
  band,time_iso,var_label,row,col,lat,lon,value

Options to control size:
  --every-n N         Export every Nth pixel in both rows and cols (subsample)
  --points points.csv Only export values at specified lon/lat points (CSV with columns: lon,lat)
  --max-rows N        Stop after writing approximately N data rows (safety)

The script tries to infer timestamps and variable labels from per-band
descriptions/tags or the filename. If none found, fields are left blank.

Usage:
  python tif_to_csv.py stack.tif -o out.csv
  python tif_to_csv.py stack.tif -o out.csv --every-n 10
  python tif_to_csv.py stack.tif -o out.csv --points stations.csv
"""

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import xy

ISO_DATE_RE = re.compile(
    r'(?P<y>\d{4})[-_/]?(?P<m>\d{2})(?:[-_/]?(?P<d>\d{2}))?(?:[ T](?P<h>\d{2}):(?P<min>\d{2})(?::(?P<s>\d{2}))?)?'
)
YEAR_MONTH_ONLY_RE = re.compile(r'(?P<y>\d{4})[-_/]?(?P<m>\d{2})(?![-_/]?\d{2})')

KNOWN_VAR_HINTS = [
    "t2m", "d2m", "u10", "v10", "sp", "tp", "ssr", "msl", "tcc", "t2m_max", "t2m_min",
    "precip", "precipitation", "total_precipitation", "skin_temperature", "sst",
    "wind_u", "wind_v", "u_component", "v_component", "dewpoint", "surface_pressure",
    "total_cloud_cover", "evaporation", "runoff"
]

def parse_date_strings(text: str) -> List[datetime]:
    dates = []
    for m in ISO_DATE_RE.finditer(text):
        y = int(m.group('y'))
        mo = int(m.group('m'))
        d = int(m.group('d') or 1)
        h = int(m.group('h') or 0)
        mi = int(m.group('min') or 0)
        s = int(m.group('s') or 0)
        try:
            dates.append(datetime(y, mo, d, h, mi, s))
        except ValueError:
            continue
    if not dates:
        for m in YEAR_MONTH_ONLY_RE.finditer(text):
            y = int(m.group('y')); mo = int(m.group('m'))
            try:
                dates.append(datetime(y, mo, 1))
            except ValueError:
                continue
    return dates

def extract_band_meta(ds: rasterio.io.DatasetReader, tif_path: Path) -> Tuple[List[Optional[str]], List[Optional[str]]]:
    """
    Return (time_iso_per_band, var_label_per_band), lists length ds.count.
    """
    time_iso = [None] * ds.count
    var_labels = [None] * ds.count

    # file-level tags (rarely per-band, but might include a global time range)
    try:
        file_tags = ds.tags()
        # fall through: these are not per-band, so not used directly
    except Exception:
        file_tags = {}

    try:
        descs = ds.descriptions or [None] * ds.count
    except Exception:
        descs = [None] * ds.count

    for b in range(1, ds.count + 1):
        pieces = []
        d = descs[b - 1]
        if d:
            pieces.append(d)
        try:
            bt = ds.tags(b)
            if bt:
                pieces.extend([f"{k}={v}" for k, v in bt.items()])
        except Exception:
            pass
        merged = " | ".join([p for p in pieces if p])
        # time
        t = None
        if merged:
            ds_ = parse_date_strings(merged)
            if ds_:
                t = ds_[0].isoformat()
        if not t:
            # fallback: try filename substring
            fn_ds = parse_date_strings(tif_path.name)
            if fn_ds:
                t = fn_ds[0].isoformat()
        time_iso[b - 1] = t

        # var
        var = None
        low = merged.lower()
        for hint in KNOWN_VAR_HINTS:
            if hint in low:
                var = hint
                break
        if var is None and merged:
            m = re.search(r'(?:variable|var|name|short_name)\s*[:=]\s*([A-Za-z0-9_]+)', merged, re.IGNORECASE)
            if m:
                var = m.group(1)
        var_labels[b - 1] = var

    return time_iso, var_labels

def load_points(points_csv: Path):
    pts = []
    with open(points_csv, newline="") as f:
        reader = csv.DictReader(f)
        if "lon" not in reader.fieldnames or "lat" not in reader.fieldnames:
            raise ValueError("Points CSV must have columns: lon,lat")
        for row in reader:
            pts.append((float(row["lon"]), float(row["lat"])))
    return pts

def nearest_rowcol(ds, lon, lat):
    # transform geographic lon/lat to row/col (assumes CRS is geographic; if not, warn)
    if ds.crs is None or ds.crs.to_string().lower() not in ("epsg:4326", "ogc:crs84", "wgs84"):
        # For simplicity, assume GeoTIFF is lon/lat; in other CRSs this is approximate.
        pass
    # use inverse transform
    row, col = ~ds.transform * (lon, lat)
    # round to nearest pixel
    return int(round(row)), int(round(col))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif_path", type=Path, help="Path to the GeoTIFF stack")
    ap.add_argument("-o", "--out", type=Path, required=True, help="Output CSV path")
    ap.add_argument("--every-n", type=int, default=None, help="Export every Nth pixel (subsample)")
    ap.add_argument("--points", type=Path, default=None, help="CSV with columns lon,lat to sample specific points")
    ap.add_argument("--max-rows", type=int, default=None, help="Approximate cap on number of output rows")
    args = ap.parse_args()

    tif_path: Path = args.tif_path
    if not tif_path.exists():
        print(f"ERROR: File not found: {tif_path}")
        return

    with rasterio.open(tif_path) as ds, open(args.out, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["band", "time_iso", "var_label", "row", "col", "lat", "lon", "value"])

        time_iso, var_labels = extract_band_meta(ds, tif_path)

        # Determine pixel iterator
        H, W = ds.height, ds.width
        if args.points:
            pts = load_points(args.points)
            # precompute row/col for each point
            rc = [nearest_rowcol(ds, lon, lat) for lon, lat in pts]
            targets = {}
            for i, (lon, lat) in enumerate(pts):
                r, c = rc[i]
                if 0 <= r < H and 0 <= c < W:
                    targets[(r, c)] = (lon, lat)
        else:
            step = args.every_n if args.every_n and args.every_n > 1 else 1

        rows_written = 0

        # Iterate bands
        for b in range(1, ds.count + 1):
            # read whole band into memory in blocks to avoid huge memory for very large rasters
            if args.points:
                # sample only target pixels
                for (r, c), (lon, lat) in targets.items():
                    window = rasterio.windows.Window(c, r, 1, 1)
                    arr = ds.read(b, window=window, masked=True)
                    val = float(arr[0, 0]) if arr.size else np.nan
                    writer.writerow([b, time_iso[b-1] or "", var_labels[b-1] or "", r, c, lat, lon, val])
                    rows_written += 1
                    if args.max_rows and rows_written >= args.max_rows:
                        print(f"Reached --max-rows={args.max_rows}, stopping.")
                        return
            else:
                # iterate with stride
                for r in range(0, H, step):
                    # read one row window at stride to keep mem low
                    row_window = rasterio.windows.Window(0, r, W, 1)
                    arr = ds.read(b, window=row_window, masked=True).squeeze(0)
                    for c in range(0, W, step):
                        val = float(arr[0, c]) if arr.ndim == 2 else float(arr[c])  # handle possible shape (1, W)
                        lon, lat = xy(ds.transform, r, c, offset="center")
                        writer.writerow([b, time_iso[b-1] or "", var_labels[b-1] or "", r, c, lat, lon, val])
                        rows_written += 1
                        if args.max_rows and rows_written >= args.max_rows:
                            print(f"Reached --max-rows={args.max_rows}, stopping.")
                            return

    print(f"Wrote CSV: {args.out}")

if __name__ == "__main__":
    main()
