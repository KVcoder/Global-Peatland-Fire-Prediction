#!/usr/bin/env python3
"""
VIIRS → Peat Grid (CSV-only)
----------------------------
Bins FIRMS VIIRS points onto the exact pixel grid of your peat mask GeoTIFF
(0.1° EPSG:4326 as exported from EE), aggregates per (date,row,col) with
fast reducers (max FRP, max confidence, counts, time range, day/night flags,
sensor flags), applies the peat mask at the SAME pixel resolution, and writes
one CSV per day. No GeoTIFFs are written.

Designed for speed:
  - Optional Arrow CSV engine (2–4× faster parsing if pyarrow is installed)
  - Large chunk streaming (tune --chunksize)
  - Peat mask preloading into RAM (--preload-mask)
  - Single integer cell id grouping path
  - Per-file, in-memory daily aggregation (each day written once per file)
  - Optional on-disk per-day merge to avoid duplicates across files (--merge-on-write)

Assumptions:
  - Your peat mask TIFF has a valid CRS/transform (EE export shown below).
  - SNPP covers 2012-01-20 → 2018-12-31, NOAA-20 covers 2019-01-01 → 2024-12-31
    (no temporal overlap is fine and avoids cross-sensor conflicts).

Example EE export (for context):
// Export.image.toDrive with crs EPSG:4326 and 0.1° transform

Author: You + ChatGPT
"""

import os, sys, glob, argparse
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

# Raster handling
try:
    import rasterio
    from rasterio.transform import rowcol, xy
    from rasterio.warp import transform_bounds
    from pyproj import Transformer
except Exception as e:
    print("Please install deps:\n  pip install pandas rasterio pyproj tqdm pyarrow", file=sys.stderr)
    raise


# ----------------------------- CLI -----------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Pool VIIRS points to peat mask grid and write daily CSVs.")
    ap.add_argument("--snpp",   required=True, help="Comma-separated globs for S-NPP CSVs (e.g., './SNPP/*.csv')")
    ap.add_argument("--noaa20", required=True, help="Comma-separated globs for NOAA-20 CSVs (e.g., './NOAA20/*.csv')")
    ap.add_argument("--mask",   required=True, help="Path to peat mask GeoTIFF (binary 0/1 at 0.1° EPSG:4326)")
    ap.add_argument("--outdir", default="viirs_daily_grid", help="Output folder for per-day pooled CSVs")
    ap.add_argument("--chunksize", type=int, default=5_000_000, help="Rows per CSV chunk")
    ap.add_argument("--snpp-start",   default="2012-01-20")
    ap.add_argument("--snpp-end",     default="2018-12-31")  # inclusive
    ap.add_argument("--noaa20-start", default="2019-01-01")
    ap.add_argument("--noaa20-end",   default="2024-12-31")  # inclusive
    ap.add_argument("--dedupe-within-chunk", action="store_true",
                    help="Drop exact duplicates (sensor,lon,lat,acq_epoch) within each file chunk")
    ap.add_argument("--preload-mask", action="store_true",
                    help="Load peat mask band into RAM (fastest). If omitted, uses on-demand reads.")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Mask threshold; peat if mask_value > threshold (default 0.0)")
    ap.add_argument("--merge-on-write", action="store_true",
                    help="If a day CSV exists, merge/aggregate with it and overwrite (prevents duplicate row/col).")
    return ap.parse_args()


# -------------------------- Utilities --------------------------------

def expand_globs(globs_csv: str):
    paths = []
    for g in [g.strip() for g in globs_csv.split(",") if g.strip()]:
        paths.extend(glob.glob(g, recursive=True))
    return sorted({p for p in paths if p.lower().endswith(".csv")})

def detect_header_mapping(path):
    """
    Probe a CSV header & return a canonical->actual column mapping.
    Ensures required canonicals 'latitude','longitude','acq_date','acq_time' exist.
    """
    hdr = pd.read_csv(path, nrows=0)
    actual = list(hdr.columns)

    candidates = {
        "latitude":  ["latitude","lat","LATITUDE","LAT"],
        "longitude": ["longitude","lon","long","LONGITUDE","LON","LONG"],
        "frp":       ["frp","FRP"],
        "confidence":["confidence","CONFIDENCE"],
        "acq_date":  ["acq_date","acq date","Acq_Date","ACQ_DATE"],
        "acq_time":  ["acq_time","acq time","Acq_Time","ACQ_TIME"],
        "daynight":  ["daynight","day/night","DayNight","DAYNIGHT"],
        "satellite": ["satellite","Satellite","SATELLITE"],
    }

    mapping = {k: None for k in candidates.keys()}
    for canon, cands in candidates.items():
        for c in cands:
            if c in actual:
                mapping[canon] = c
                break

    required = ["latitude","longitude","acq_date","acq_time"]
    missing = [r for r in required if mapping[r] is None]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}; found header: {actual}")

    # Build list of columns to read (skip Nones)
    usecols = [v for v in mapping.values() if v is not None]
    return mapping, usecols

def vector_acq_epoch(dates_str: pd.Series, times_str: pd.Series) -> np.ndarray:
    """Vectorized epoch from date YYYY-MM-DD and time HHmm (FIRMS UTC)."""
    if dates_str is None or times_str is None:
        return np.full(0, np.nan)
    dt = pd.to_datetime(dates_str, errors="coerce", utc=True)
    t = times_str.astype(str).str.zfill(4)
    hh = pd.to_numeric(t.str[:-2].replace("", "0"), errors="coerce").fillna(0).astype("int64")
    mm = pd.to_numeric(t.str[-2:], errors="coerce").fillna(0).astype("int64")
    m = hh * 60 + mm
    epoch = (dt + pd.to_timedelta(m, unit="m")).astype("int64") // 1_000_000_000
    return epoch.to_numpy(dtype="float64")


# ---------------------- Mask + Grid Sampler --------------------------

class MaskIndexSampler:
    """
    - bbox prefilter in WGS84
    - row/col mapping via integer floor (aligns exactly to mask pixels)
    - explicit NoData handling
    - optional mask preload
    - helpers for (row,col) ↔ lon/lat centers in EPSG:4326
    """
    def __init__(self, tif_path: str, threshold: float = 0.0, preload: bool = False):
        if not os.path.exists(tif_path):
            raise FileNotFoundError(tif_path)
        self.path = tif_path
        self.threshold = threshold
        self.ds = rasterio.open(tif_path)
        self.crs = self.ds.crs
        self.transform = self.ds.transform
        self.height = self.ds.height
        self.width = self.ds.width
        self.nodata = self.ds.nodata

        # Bounds in WGS84 for early drop
        self.wgs84_bounds = transform_bounds(self.crs, "EPSG:4326", *self.ds.bounds, densify_pts=16)

        # Transformers if CRS != EPSG:4326
        self.fwd = None
        if self.crs and self.crs.to_string().upper() not in ("EPSG:4326", "OGC:CRS84"):
            self.fwd = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self.inv = None
        if self.crs and self.crs.to_string().upper() not in ("EPSG:4326", "OGC:CRS84"):
            self.inv = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

        # Preload
        self.preload = preload
        if preload:
            self.band = self.ds.read(1)
        else:
            self.band = None
            try:
                self.block_h, self.block_w = self.ds.block_shapes[0]
            except Exception:
                self.block_h, self.block_w = 512, self.width
            self.blocks_w = (self.width + self.block_w - 1) // self.block_w

    def _bbox_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        minx, miny, maxx, maxy = self.wgs84_bounds
        lon = df["longitude"].to_numpy()
        lat = df["latitude"].to_numpy()
        keep = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
        return df.loc[keep]

    def lonlat_to_rowcol(self, lon: np.ndarray, lat: np.ndarray):
        if self.fwd is not None:
            xs, ys = self.fwd.transform(lon, lat)
        else:
            xs, ys = lon, lat
        rows, cols = rowcol(self.transform, xs, ys, op=lambda v: int(np.floor(v)))
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        in_bounds = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        return rows, cols, in_bounds

    def rowscols_to_lonlat_centers(self, rows: np.ndarray, cols: np.ndarray):
        xs, ys = xy(self.transform, rows, cols, offset="center")
        xs = np.asarray(xs); ys = np.asarray(ys)
        if self.inv is not None:
            lon, lat = self.inv.transform(xs, ys)
        else:
            lon, lat = xs, ys
        return np.asarray(lon), np.asarray(lat)

    def values_at_rowscols(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
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
            win = ((r0, r1), (c0, c1))
            tile = self.ds.read(1, window=win)
            sel = np.where((br == brk) & (bc == bck))[0]
            rr = rows[sel] - r0
            cc = cols[sel] - c0
            out[sel] = tile[rr, cc].astype("float64")
        return out

    def mask_keep_cells(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        vals = self.values_at_rowscols(rows, cols)
        keep = np.isfinite(vals) & (vals > self.threshold)
        nd = self.nodata
        if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
            keep &= (vals != nd)
        return keep


# --------------- Fast pooling (per date, per cell) -------------------

def pool_to_grid(df: pd.DataFrame, sampler: MaskIndexSampler) -> pd.DataFrame:
    """
    Input: normalized VIIRS rows (date, lon, lat, frp, confidence, acq_epoch, DayNight, sensor)
    Output: pooled per (date,row,col) with fast reducers (no lon/lat centers here).
    """
    if df.empty:
        return df

    # Early bbox drop
    df = sampler._bbox_filter(df)
    if df.empty:
        return df

    rows, cols, inb = sampler.lonlat_to_rowcol(df["longitude"].to_numpy(),
                                               df["latitude"].to_numpy())
    if not inb.any():
        return df.iloc[0:0].copy()

    sub = df.iloc[np.nonzero(inb)[0]].copy()
    sub["row"] = rows[inb]
    sub["col"] = cols[inb]

    # Fast flags (robust to missing DayNight)
    dn_series = sub["DayNight"] if "DayNight" in sub.columns else pd.Series("", index=sub.index)
    dn0 = dn_series.astype(str).str.upper().str[0]
    sub["is_day"] = (dn0 == "D")
    sub["is_night"] = (dn0 == "N")
    sub["has_snpp"] = (sub["sensor"] == "SNPP")
    sub["has_noaa20"] = (sub["sensor"] == "NOAA20")

    # Single integer cell id
    w = sampler.width
    sub["cell"] = (sub["row"].to_numpy(dtype=np.int64) * w + sub["col"].to_numpy(dtype=np.int64))

    # Fast built-in reducers (C-accelerated), no sort
    grp = (sub.groupby(["date","cell"], sort=False, as_index=False)
              .agg(count=("frp","size"),
                   max_frp=("frp","max"),
                   max_confidence=("confidence","max"),
                   first_epoch=("acq_epoch","min"),
                   last_epoch=("acq_epoch","max"),
                   has_day=("is_day","any"),
                   has_night=("is_night","any"),
                   has_snpp=("has_snpp","any"),
                   has_noaa20=("has_noaa20","any")))

    # Recover row/col (int32 to save mem)
    cell = grp["cell"].to_numpy(dtype=np.int64)
    grp["row"] = (cell // w).astype(np.int32)
    grp["col"] = (cell %  w).astype(np.int32)
    grp.drop(columns=["cell"], inplace=True)
    return grp


# --------------- Per-file daily aggregation & writes ------------------

def _merge_day_aggregates(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two per-day aggregates that each have columns:
      row,col,count,max_frp,max_confidence,first_epoch,last_epoch,has_day,has_night,has_snpp,has_noaa20
    Group by (row,col) and combine with correct reducers.
    """
    merged = pd.concat([existing, incoming], ignore_index=True)
    out = (merged.groupby(["row","col"], as_index=False)
           .agg(count=("count","sum"),
                max_frp=("max_frp","max"),
                max_confidence=("max_confidence","max"),
                first_epoch=("first_epoch","min"),
                last_epoch=("last_epoch","max"),
                has_day=("has_day","any"),
                has_night=("has_night","any"),
                has_snpp=("has_snpp","any"),
                has_noaa20=("has_noaa20","any")))
    return out

def _read_day_csv(path: str) -> pd.DataFrame:
    dtypes = {
        "row": "Int32",
        "col": "Int32",
        "lon_center": "float64",
        "lat_center": "float64",
        "count": "int64",
        "max_frp": "float64",
        "max_confidence": "float64",
        "first_epoch": "float64",
        "last_epoch": "float64",
        "has_day": "boolean",
        "has_night": "boolean",
        "has_snpp": "boolean",
        "has_noaa20": "boolean",
    }
    # 'date' read as string and ignored for merge; recomputed on write
    return pd.read_csv(path, dtype=dtypes)

def _write_day_csv(day: str, sub: pd.DataFrame, sampler: MaskIndexSampler, outdir: str,
                   merge_on_write: bool):
    """
    sub: DataFrame with columns (row,col,count,max_frp,max_confidence,first_epoch,last_epoch,has_day,has_night,has_snpp,has_noaa20)
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"viirs_grid_{day}.csv")

    # If merging on write and file exists: read, merge, then overwrite
    if merge_on_write and os.path.exists(path):
        try:
            old = _read_day_csv(path)
            # Ensure only aggregator columns + row/col
            old_sub = old[["row","col","count","max_frp","max_confidence","first_epoch","last_epoch",
                           "has_day","has_night","has_snpp","has_noaa20"]].copy()
            sub = _merge_day_aggregates(old_sub, sub)
        except Exception as e:
            print(f"[WARN] Cannot merge with existing {path}: {e}. Overwriting with new data.", file=sys.stderr)

    # Sort and attach centers
    sub = sub.sort_values(["row","col"])
    lonc, latc = sampler.rowscols_to_lonlat_centers(sub["row"].to_numpy(),
                                                    sub["col"].to_numpy())
    out_df = pd.DataFrame({
        "date":        [day] * len(sub),
        "row":         sub["row"].to_numpy(),
        "col":         sub["col"].to_numpy(),
        "lon_center":  lonc,
        "lat_center":  latc,
        "count":       sub["count"].to_numpy(),
        "max_frp":     sub["max_frp"].to_numpy(),
        "max_confidence": sub["max_confidence"].to_numpy(),
        "first_epoch": sub["first_epoch"].to_numpy(),
        "last_epoch":  sub["last_epoch"].to_numpy(),
        "has_day":     sub["has_day"].to_numpy(),
        "has_night":   sub["has_night"].to_numpy(),
        "has_snpp":    sub["has_snpp"].to_numpy(),
        "has_noaa20":  sub["has_noaa20"].to_numpy(),
    })

    mode = "w"
    header = True
    if not merge_on_write and os.path.exists(path):
        # Append if not merging (possible duplicates)
        mode = "a"
        header = False

    out_df.to_csv(path, index=False, mode=mode, header=header)
    return len(out_df)


def process_file(path, sensor_label, sampler: MaskIndexSampler,
                 outdir, chunksize, date_start, date_end, dedupe_within_chunk, merge_on_write):
    # Detect columns & build reader kwargs once
    try:
        mapping, usecols = detect_header_mapping(path)
    except Exception as e:
        print(f"[WARN] {e}", file=sys.stderr)
        return 0

    read_kwargs = dict(chunksize=chunksize, low_memory=False, usecols=usecols)
    if _arrow:
        read_kwargs["engine"] = "pyarrow"

    # Per-file in-memory daily buffer: day -> aggregated DataFrame (no 'date' col inside)
    agg_by_day = {}

    try:
        chunk_iter = pd.read_csv(path, **read_kwargs)
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}", file=sys.stderr)
        return 0

    for chunk in chunk_iter:
        # Extract series via mapping (robust to varying headers)
        lat = chunk[mapping["latitude"]]
        lon = chunk[mapping["longitude"]]
        frp = chunk[mapping["frp"]] if mapping["frp"] is not None else np.nan
        conf_raw = chunk[mapping["confidence"]] if mapping["confidence"] is not None else None
        ad = chunk[mapping["acq_date"]]
        at = chunk[mapping["acq_time"]]
        dn = chunk[mapping["daynight"]] if mapping["daynight"] is not None else ""

        # Confidence handling
        if conf_raw is None:
            conf_num = np.nan
        else:
            if pd.api.types.is_numeric_dtype(conf_raw):
                conf_num = pd.to_numeric(conf_raw, errors="coerce")
            else:
                conf_num = (conf_raw.astype(str).str.strip().str.lower()
                            .map({"l":0,"low":0,"n":1,"nominal":1,"h":2,"high":2}))

        # Build normalized frame (vectorized)
        out = pd.DataFrame({
            "date": pd.to_datetime(ad, errors="coerce").dt.strftime("%Y-%m-%d"),
            "sensor": sensor_label,
            "longitude": pd.to_numeric(lon, errors="coerce"),
            "latitude":  pd.to_numeric(lat, errors="coerce"),
            "frp":       pd.to_numeric(frp, errors="coerce"),
            "confidence": pd.to_numeric(conf_num, errors="coerce"),
            "acq_time":  at.astype(str),
            "acq_epoch": vector_acq_epoch(ad, at),
            "DayNight":  dn.astype(str) if mapping["daynight"] is not None else ""
        })

        # quick cleaning + geographic sanity
        out = out.dropna(subset=["date","longitude","latitude"])
        out = out[(out["longitude"] >= -180) & (out["longitude"] <= 180) &
                  (out["latitude"]  >=  -90) & (out["latitude"]  <=  90)]

        # date filter (inclusive)
        if date_start or date_end:
            dts = pd.to_datetime(out["date"], errors="coerce").dt.date
            if date_start: out = out[dts >= pd.to_datetime(date_start).date()]
            if date_end:   out = out[dts <= pd.to_datetime(date_end).date()]
            if out.empty:
                continue

        if dedupe_within_chunk:
            out = out.drop_duplicates(subset=["sensor","longitude","latitude","acq_epoch"])

        # Pool to mask grid (per date, per cell)
        pooled = pool_to_grid(out, sampler)
        if pooled.empty:
            continue

        # Apply peat mask at cell resolution
        keep = sampler.mask_keep_cells(pooled["row"].to_numpy(),
                                       pooled["col"].to_numpy())
        pooled = pooled.iloc[keep]
        if pooled.empty:
            continue

        # Split by day and accumulate in-memory (merge aggregates)
        for day, sub in pooled.groupby("date"):
            sub_nodate = sub.drop(columns=["date"]).copy()
            if day in agg_by_day:
                agg_by_day[day] = _merge_day_aggregates(agg_by_day[day], sub_nodate)
            else:
                agg_by_day[day] = sub_nodate

    # After finishing this file: write each day once
    rows_written_csv = 0
    for day, sub in agg_by_day.items():
        rows_written_csv += _write_day_csv(day, sub, sampler, outdir, merge_on_write)

    return rows_written_csv


# ------------------------------ MAIN ---------------------------------

def main():
    args = parse_args()

    snpp_files = expand_globs(args.snpp)
    n20_files  = expand_globs(args.noaa20)
    if not snpp_files:
        print("No S-NPP CSVs matched. Check --snpp.", file=sys.stderr); sys.exit(2)
    if not n20_files:
        print("No NOAA-20 CSVs matched. Check --noaa20.", file=sys.stderr); sys.exit(2)
    if not os.path.exists(args.mask):
        print(f"[FATAL] Mask not found: {args.mask}", file=sys.stderr); sys.exit(2)

    sampler = MaskIndexSampler(args.mask, threshold=args.threshold, preload=args.preload_mask)

    total_snpp = 0
    iterator = tqdm(snpp_files, desc="SNPP files", unit="file") if _tqdm else snpp_files
    for f in iterator:
        total_snpp += process_file(
            f, "SNPP", sampler, args.outdir, args.chunksize,
            args.snpp_start, args.snpp_end, args.dedupe_within_chunk, args.merge_on_write
        )

    total_n20 = 0
    iterator = tqdm(n20_files, desc="NOAA-20 files", unit="file") if _tqdm else n20_files
    for f in iterator:
        total_n20 += process_file(
            f, "NOAA20", sampler, args.outdir, args.chunksize,
            args.noaa20_start, args.noaa20_end, args.dedupe_within_chunk, args.merge_on_write
        )

    print(f"Done. Grid rows written: SNPP={total_snpp:,}, NOAA-20={total_n20:,}.")
    print(f"Daily CSVs in: {args.outdir}")


if __name__ == "__main__":
    main()
