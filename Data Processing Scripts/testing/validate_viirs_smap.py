#!/usr/bin/env python3
"""
validate_viirs_grid_csvs.py
---------------------------
Audits per-day VIIRS→peat-grid CSVs produced by the "VIIRS → Peat Grid (CSV-only)" script.

Checks performed per CSV:
  1) Filename date matches the 'date' column (all rows).
  2) Required columns exist: date,row,col,lon_center,lat_center,count,
     max_frp,max_confidence,first_epoch,last_epoch,has_day,has_night,has_snpp,has_noaa20.
  3) (row,col) are integers, in-bounds for the peat mask; no duplicates within a file.
  4) lon_center/lat_center equal the raster pixel centers for (row,col) in EPSG:4326 (within tolerance).
  5) Mask at (row,col) is peat (value > threshold and not NoData).
  6) Basic aggregator integrity: count>=1; first_epoch<=last_epoch; both epochs fall within that day's UTC range.
  7) Optional sensor-window sanity: if you keep default SNPP/NOAA-20 windows (no temporal overlap),
     flags has_snpp/has_noaa20 will be checked for unlikely overlap.

Exit code is 0 if all files pass (no errors). With --strict, any warning triggers non-zero exit.

Dependencies:
  pip install pandas numpy rasterio pyproj tqdm pyarrow

Author: You + ChatGPT
"""

import os, re, sys, glob, argparse, math
from dataclasses import dataclass
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

# Raster + CRS
import rasterio
from rasterio.transform import xy
from pyproj import Transformer


RE_DATE_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})")

REQUIRED_COLS = [
    "date","row","col","lon_center","lat_center","count","max_frp","max_confidence",
    "first_epoch","last_epoch","has_day","has_night","has_snpp","has_noaa20"
]

BOOL_COLS = ["has_day","has_night","has_snpp","has_noaa20"]


@dataclass
class AuditConfig:
    mask_path: str
    threshold: float = 0.0
    tol_deg: float = 1e-6
    strict: bool = False
    snpp_start: str = "2012-01-20"
    snpp_end: str   = "2018-12-31"
    noaa20_start: str = "2019-01-01"
    noaa20_end: str   = "2024-12-31"
    preload_mask: bool = False


class MaskHelper:
    def __init__(self, mask_path: str, preload: bool = False):
        if not os.path.exists(mask_path):
            raise FileNotFoundError(mask_path)
        self.ds = rasterio.open(mask_path)
        self.transform = self.ds.transform
        self.height = self.ds.height
        self.width = self.ds.width
        self.crs = self.ds.crs
        self.nodata = self.ds.nodata

        # Transformer maskCRS -> WGS84 for pixel center lon/lat
        self.inv = None
        if self.crs and self.crs.to_string().upper() not in ("EPSG:4326", "OGC:CRS84"):
            self.inv = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

        # preload band
        self.preload = preload
        self.band = self.ds.read(1) if preload else None

        # block sizes for on-demand reads
        if not preload:
            try:
                self.block_h, self.block_w = self.ds.block_shapes[0]
            except Exception:
                self.block_h, self.block_w = 512, self.width
            self.blocks_w = (self.width + self.block_w - 1)//self.block_w

    def centers_wgs84(self, rows: np.ndarray, cols: np.ndarray):
        xs, ys = xy(self.transform, rows, cols, offset="center")
        xs = np.asarray(xs); ys = np.asarray(ys)
        if self.inv is not None:
            lon, lat = self.inv.transform(xs, ys)
        else:
            lon, lat = xs, ys
        return np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")

    def values_at(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        if self.preload:
            return self.band[rows, cols].astype("float64")
        # on-demand block reads
        bh = getattr(self, "block_h", 512)
        bw = getattr(self, "block_w", self.width)
        h, w = self.height, self.width
        blocks_w = getattr(self, "blocks_w", (self.width + bw - 1)//bw)
        br = rows // bh
        bc = cols // bw
        keys = np.unique(br * blocks_w + bc)
        out = np.empty(len(rows), dtype="float64")
        for key in keys:
            brk = int(key // blocks_w)
            bck = int(key %  blocks_w)
            r0, r1 = int(brk * bh), int(min((brk + 1) * bh, h))
            c0, c1 = int(bck * bw), int(min((bck + 1) * bw, w))
            tile = self.ds.read(1, window=((r0, r1), (c0, c1)))
            sel = np.where((br == brk) & (bc == bck))[0]
            rr = rows[sel] - r0
            cc = cols[sel] - c0
            out[sel] = tile[rr, cc].astype("float64")
        return out


def nearly_equal(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    diff = np.abs(a - b)
    # handle NaNs conservatively: NaN vs number -> False
    return (diff <= tol) & np.isfinite(a) & np.isfinite(b)


def coerce_bool(s: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
        return s.astype("boolean")
    m = s.astype(str).str.strip().str.lower()
    mapped = m.map({
        "true": True, "t": True, "1": True, "yes": True, "y": True,
        "false": False, "f": False, "0": False, "no": False, "n": False
    })
    return mapped.astype("boolean")


def is_integral_series(s: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(s):
        return True
    if pd.api.types.is_float_dtype(s):
        # allow floats that are actually ints (e.g., 12.0)
        return np.all(np.isfinite(s.to_numpy()) & (np.floor(s) == s))
    return False


def extract_date_from_name(path: str) -> str | None:
    m = RE_DATE_IN_NAME.search(os.path.basename(path))
    return m.group(1) if m else None


def audit_csv(path: str, mh: MaskHelper, cfg: AuditConfig):
    errors, warns = [], []

    # read
    read_kwargs = dict(low_memory=False)
    if _arrow:
        read_kwargs["engine"] = "pyarrow"

    try:
        df = pd.read_csv(path, **read_kwargs)
    except Exception as e:
        errors.append(f"Cannot read CSV: {e}")
        return errors, warns, 0

    # 1) required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors, warns, 0

    nrows = len(df)

    # 2) filename date == column 'date' (all rows)
    date_in_name = extract_date_from_name(path)
    unique_dates = set(df["date"].astype(str).unique().tolist())
    if len(unique_dates) != 1:
        errors.append(f"Multiple 'date' values present: {sorted(unique_dates)[:5]} ...")
    elif date_in_name and (date_in_name not in unique_dates):
        errors.append(f"Filename date '{date_in_name}' != CSV date {list(unique_dates)[0]}")
    elif not date_in_name:
        warns.append("Filename does not contain YYYY-MM-DD; cannot cross-check name vs column.")

    # parsed day window (UTC)
    day_str = list(unique_dates)[0] if unique_dates else None
    day_start = pd.Timestamp(day_str, tz="UTC") if day_str else None
    day_end = (day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)) if day_start is not None else None

    # 3) row/col: integral & in-bounds; duplicates
    row = df["row"]; col = df["col"]

    if not is_integral_series(row) or not is_integral_series(col):
        errors.append("Columns 'row' and/or 'col' are not integral values.")
    else:
        r = row.to_numpy(dtype=np.int64)
        c = col.to_numpy(dtype=np.int64)
        oob = (r < 0) | (r >= mh.height) | (c < 0) | (c >= mh.width)
        if np.any(oob):
            k = int(np.count_nonzero(oob))
            errors.append(f"{k} rows have out-of-bounds (row,col) w.r.t. mask ({mh.height}x{mh.width}).")

        dups = pd.Series(list(zip(r, c))).duplicated(keep=False)
        if dups.any():
            k = int(dups.sum())
            errors.append(f"{k} duplicate (row,col) entries within file (expected unique per day).")

    # 4) centers match raster centers
    try:
        exp_lon, exp_lat = mh.centers_wgs84(df["row"].to_numpy(dtype=int),
                                            df["col"].to_numpy(dtype=int))
        lon_ok = nearly_equal(df["lon_center"].to_numpy(dtype=float), exp_lon, cfg.tol_deg)
        lat_ok = nearly_equal(df["lat_center"].to_numpy(dtype=float), exp_lat, cfg.tol_deg)
        bad = ~(lon_ok & lat_ok)
        if np.any(bad):
            k = int(np.count_nonzero(bad))
            max_lon_err = float(np.nanmax(np.abs(df["lon_center"].to_numpy(dtype=float)[bad] - exp_lon[bad])))
            max_lat_err = float(np.nanmax(np.abs(df["lat_center"].to_numpy(dtype=float)[bad] - exp_lat[bad])))
            errors.append(f"{k} rows have lon/lat centers not matching raster centers "
                          f"(max |Δlon|={max_lon_err:.2e}, |Δlat|={max_lat_err:.2e} > tol {cfg.tol_deg}).")
    except Exception as e:
        errors.append(f"Failed center-coordinate check: {e}")

    # 5) mask peatness check
    try:
        vals = mh.values_at(df["row"].to_numpy(dtype=int), df["col"].to_numpy(dtype=int))
        keep = np.isfinite(vals) & (vals > cfg.threshold)
        if mh.nodata is not None and not (isinstance(mh.nodata, float) and math.isnan(mh.nodata)):
            keep &= (vals != mh.nodata)
        bad = ~keep
        if np.any(bad):
            k = int(np.count_nonzero(bad))
            errors.append(f"{k} rows map to non-peat / NoData mask cells (<= threshold {cfg.threshold} or NoData).")
    except Exception as e:
        errors.append(f"Failed mask-value check: {e}")

    # 6) aggregator integrity
    # count >= 1
    bad_count = (~pd.to_numeric(df["count"], errors="coerce").fillna(-1).ge(1))
    if bad_count.any():
        k = int(bad_count.sum())
        errors.append(f"{k} rows have count < 1 or invalid.")

    # first_epoch <= last_epoch and inside the day window (if available)
    fe = pd.to_numeric(df["first_epoch"], errors="coerce")
    le = pd.to_numeric(df["last_epoch"], errors="coerce")
    bad_order = ~(fe <= le)
    if bad_order.any():
        k = int(bad_order.sum())
        errors.append(f"{k} rows have first_epoch > last_epoch.")

    if (day_start is not None) and fe.notna().any() and le.notna().any():
        s_epoch = int(day_start.timestamp())
        e_epoch = int(day_end.timestamp())
        out_of_day = (~fe.between(s_epoch, e_epoch)) | (~le.between(s_epoch, e_epoch))
        out_of_day = out_of_day.fillna(False)
        if out_of_day.any():
            k = int(out_of_day.sum())
            warns.append(f"{k} rows have epoch bounds outside the CSV day window "
                         f"({day_str} UTC). May be parsing/rounding issues.")

    # 7) boolean flags & sensor window sanity
    for bcol in BOOL_COLS:
        try:
            df[bcol] = coerce_bool(df[bcol], bcol)
            if df[bcol].isna().any():
                warns.append(f"Column '{bcol}' contains values that are not clearly boolean; coerced with NaNs.")
        except Exception as e:
            errors.append(f"Failed to coerce boolean column '{bcol}': {e}")

    # SNPP/NOAA20 overlap sanity (based on configured windows)
    try:
        if day_str:
            day = pd.to_datetime(day_str).date()
            snpp_ok = pd.to_datetime(cfg.snpp_start).date() <= day <= pd.to_datetime(cfg.snpp_end).date()
            n20_ok  = pd.to_datetime(cfg.noaa20_start).date() <= day <= pd.to_datetime(cfg.noaa20_end).date()
            if snpp_ok and n20_ok:
                # This shouldn't happen with default windows; warn if both flags present in same day.
                both_true = (df["has_snpp"].fillna(False) & df["has_noaa20"].fillna(False)).any()
                if both_true:
                    warns.append("Both has_snpp and has_noaa20 are True on a day that belongs to both windows. "
                                 "If your sources truly overlap, ignore; else verify inputs.")
            else:
                if snpp_ok and df["has_noaa20"].fillna(False).any():
                    warns.append("Day in SNPP window but has_noaa20=True present.")
                if n20_ok and df["has_snpp"].fillna(False).any():
                    warns.append("Day in NOAA-20 window but has_snpp=True present.")
    except Exception as e:
        warns.append(f"Sensor-window sanity check skipped due to error: {e}")

    return errors, warns, nrows


def parse_args():
    ap = argparse.ArgumentParser(description="Validate VIIRS daily grid CSVs against a peat mask.")
    ap.add_argument("--mask", required=True, help="Path to peat mask GeoTIFF used for gridding.")
    ap.add_argument("--indir", required=True,
                    help="Directory containing viirs_grid_YYYY-MM-DD.csv files OR a glob to CSVs.")
    ap.add_argument("--threshold", type=float, default=0.0, help="Peat threshold; peat if value > threshold.")
    ap.add_argument("--tol-deg", type=float, default=1e-6, help="Tolerance for lon/lat center checks (degrees).")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors for exit code.")
    ap.add_argument("--preload-mask", action="store_true", help="Preload mask band into RAM for faster checks.")
    ap.add_argument("--snpp-start", default="2012-01-20")
    ap.add_argument("--snpp-end",   default="2018-12-31")
    ap.add_argument("--noaa20-start", default="2019-01-01")
    ap.add_argument("--noaa20-end",   default="2024-12-31")
    return ap.parse_args()


def expand_inputs(indir: str):
    if os.path.isdir(indir):
        return sorted(glob.glob(os.path.join(indir, "viirs_grid_*.csv")))
    # otherwise treat as glob
    return sorted(glob.glob(indir))


def main():
    args = parse_args()
    files = expand_inputs(args.indir)
    if not files:
        print("No CSVs found. Provide a directory with viirs_grid_*.csv or a matching glob.", file=sys.stderr)
        sys.exit(2)

    cfg = AuditConfig(
        mask_path=args.mask,
        threshold=args.threshold,
        tol_deg=args.tol_deg,
        strict=args.strict,
        snpp_start=args.snpp_start,
        snpp_end=args.snpp_end,
        noaa20_start=args.noaa20_start,
        noaa20_end=args.noaa20_end,
        preload_mask=args.preload_mask,
    )

    mh = MaskHelper(cfg.mask_path, preload=cfg.preload_mask)

    total_rows = 0
    total_errors = 0
    total_warns = 0

    iterator = tqdm(files, desc="Auditing CSVs", unit="file") if _tqdm else files
    for f in iterator:
        errors, warns, nrows = audit_csv(f, mh, cfg)
        total_rows += nrows
        total_errors += len(errors)
        total_warns += len(warns)

        status = "OK"
        if errors:
            status = "ERROR"
        elif warns:
            status = "WARN"

        print(f"\n[{status}] {os.path.basename(f)}  rows={nrows}")
        for e in errors:
            print(f"  ✖ {e}")
        for w in warns:
            print(f"  ! {w}")

    print("\nSummary")
    print("-------")
    print(f"Files checked : {len(files)}")
    print(f"Total rows    : {total_rows:,}")
    print(f"Errors        : {total_errors}")
    print(f"Warnings      : {total_warns}")

    exit_bad = (total_errors > 0) or (cfg.strict and total_warns > 0)
    sys.exit(1 if exit_bad else 0)


if __name__ == "__main__":
    main()
