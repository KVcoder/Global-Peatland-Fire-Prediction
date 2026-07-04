#!/usr/bin/env python3
"""
Convert stacked daily GeoTIFFs (e.g., EE yearly stacks) into one CSV per day,
filtered to a peat mask grid (0/1, same resolution/transform/CRS).

Parallel + robust:
  - --workers-files: process multiple input TIFFs in parallel (process pool).
  - --workers-bands: optional per-file band threads (IO heavy; keep small).
  - Atomic writes with lock files for cross-process safety.
  - Optional --merge-on-write to dedupe (row,col) if multiple stacks overlap.

Speed:
  - Windowed (block) reads only where peat exists (sparse mode).
  - Dense fallback when mask is dense.
  - GDAL threads via rasterio.Env + larger cache.

Outputs (one file per day):
  outdir/<prefix>_YYYY-MM-DD.csv[.gz]
Columns:
  date,row,col,lon_center,lat_center,<value_name>
"""

import os
import re
import sys
import glob
import time
import uuid
import argparse
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Progress bars (optional)
try:
    from tqdm.auto import tqdm
    def _tqdm(x, **kw): return tqdm(x, **kw)
except Exception:
    def _tqdm(x, **kw): return x

# Raster IO
import rasterio
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ------------------------------ CLI ----------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Daily CSV export from stacked GeoTIFFs masked by peat grid (parallel, accelerated).")
    ap.add_argument("--tifs", required=True,
                    help="Comma-separated globs for merged GeoTIFFs (e.g., '/data/merged/*.tif')")
    ap.add_argument("--mask", required=True,
                    help="Path to peat mask GeoTIFF (binary 0/1, aligned to stacks)")
    ap.add_argument("--outdir", default="daily_csv",
                    help="Directory to write daily CSVs")
    ap.add_argument("--value-name", default="value",
                    help="Column name for the variable (e.g., WTD, Tair, VPD)")
    ap.add_argument("--prefix", default="grid",
                    help="Filename prefix, e.g., 'wtd_grid' → wtd_grid_YYYY-MM-DD.csv")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Mask threshold: keep cells where mask_value > threshold (default 0.0)")
    ap.add_argument("--gzip", action="store_true",
                    help="Write .csv.gz instead of .csv (uses pandas compression='gzip').")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip writing if the day's CSV already exists")
    ap.add_argument("--merge-on-write", action="store_true",
                    help="If an output day already exists, merge new rows into it (dedupe by row,col).")
    ap.add_argument("--start", default=None, help="Earliest date to export (YYYY-MM-DD), inclusive")
    ap.add_argument("--end",   default=None, help="Latest date to export (YYYY-MM-DD), inclusive")
    ap.add_argument("--year-start", default=None,
                    help="Optional first date for each file when band names lack dates (e.g., '2015-03-31'). "
                         "If not set, will assume Jan 1 of the parsed year from filename.")
    ap.add_argument("--gdal-cache-mb", type=int, default=1024,
                    help="GDAL cache size in MB (default 1024).")
    ap.add_argument("--dense-threshold", type=float, default=0.5,
                    help="If kept fraction >= this, use whole-band fast path (default 0.5).")
    ap.add_argument("--workers-files", type=int, default=0,
                    help="Process multiple files in parallel. 0=auto (#CPU). 1=disabled.")
    ap.add_argument("--workers-bands", type=int, default=1,
                    help="Threads per file to process bands concurrently. >1 may thrash slow disks.")
    return ap.parse_args()

# -------------------------- Utilities --------------------------------

def expand_globs(globs_csv: str):
    paths = []
    for g in [g.strip() for g in globs_csv.split(",") if g.strip()]:
        paths.extend(glob.glob(g, recursive=True))
    return sorted({p for p in paths if p.lower().endswith((".tif", ".tiff"))})

_date_rx = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def band_dates_from_descriptions(ds: rasterio.io.DatasetReader):
    desc = list(ds.descriptions) if ds.descriptions is not None else []
    if len(desc) != ds.count:
        return None
    out = []
    for d in desc:
        d = (d or "").strip()
        if _date_rx.match(d):
            out.append(d)
        else:
            return None
    return out

def year_from_filename(path: str):
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d{4})(?!.*\d)", base)  # last 4-digit group
    return int(m.group(1)) if m else None

def infer_dates_for_stack(path: str, count: int, explicit_first: str = None):
    if explicit_first:
        start = pd.to_datetime(explicit_first).date()
    else:
        yr = year_from_filename(path)
        if not yr:
            return None
        start = pd.to_datetime(f"{yr}-01-01").date()
    rng = pd.date_range(start, periods=count, freq="D")
    return [d.strftime("%Y-%m-%d") for d in rng]

def ensure_alignment_meta(a: rasterio.io.DatasetReader, mask_transform, mask_crs, mask_w, mask_h):
    if a.crs != mask_crs:
        raise ValueError(f"CRS mismatch: {a.crs} vs mask {mask_crs}")
    if a.transform != mask_transform:
        raise ValueError("Transform mismatch between stack and mask.")
    if a.width != mask_w or a.height != mask_h:
        raise ValueError("Shape mismatch between stack and mask.")

def read_mask_bool(mask_path, threshold: float):
    with rasterio.open(mask_path) as m:
        arr = m.read(1)
        keep = np.isfinite(arr)
        if m.nodata is not None and not (isinstance(m.nodata, float) and np.isnan(m.nodata)):
            keep &= (arr != m.nodata)
        keep &= (arr > threshold)
        return keep, m.transform, m.crs, m.width, m.height

def build_segments(ds, keep_bool):
    """
    Precompute segments of kept pixels grouped by ds block windows.
    Returns: segments list[(win, rr, cc, start, end)], rows, cols, lonc, latc
    Order is row-major by block, then within-block.
    """
    rows_all = []
    cols_all = []
    segments = []
    cursor = 0

    for _, win in ds.block_windows(1):
        sub = keep_bool[win.row_off:win.row_off + win.height,
                        win.col_off:win.col_off + win.width]
        rr, cc = np.nonzero(sub)
        if rr.size:
            rr = rr.astype(np.int32, copy=False)
            cc = cc.astype(np.int32, copy=False)
            rows = rr + win.row_off
            cols = cc + win.col_off
            rows_all.append(rows)
            cols_all.append(cols)
            n = rr.size
            segments.append((win, rr, cc, cursor, cursor + n))
            cursor += n

    if cursor == 0:
        return [], np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([]), np.array([])

    rows_all = np.concatenate(rows_all).astype(np.int32, copy=False)
    cols_all = np.concatenate(cols_all).astype(np.int32, copy=False)

    xs, ys = rasterio.transform.xy(ds.transform, rows_all, cols_all, offset="center")
    lonc = np.asarray(xs, dtype="float64")
    latc = np.asarray(ys, dtype="float64")

    return segments, rows_all, cols_all, lonc, latc

# ---------- Concurrency-safe writer (atomic + optional merge) ----------

def _acquire_lock(lock_path, timeout=60.0, poll=0.05):
    """Create a lock file atomically with 'x' mode. Wait up to timeout seconds."""
    start = time.time()
    while True:
        try:
            f = os.fdopen(os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644), 'w')
            f.write(str(os.getpid()))
            f.close()
            return True
        except FileExistsError:
            if time.time() - start > timeout:
                return False
            time.sleep(poll)

def _release_lock(lock_path):
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass

def write_day_csv_atomic(outdir, prefix, date_str, rows, cols, lonc, latc, vals,
                         value_name, gzip=False, skip_existing=False, merge_on_write=False):
    os.makedirs(outdir, exist_ok=True)
    ext = ".csv.gz" if gzip else ".csv"
    path = os.path.join(outdir, f"{prefix}_{date_str}{ext}")
    lock_path = path + ".lock"
    tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"

    # Fast path: don't touch lock if skipping and exists
    if skip_existing and os.path.exists(path) and not merge_on_write:
        return "skip-existing", path, 0

    compression = "gzip" if gzip else None
    ok = np.isfinite(vals)

    if not ok.any():
        # Empty header as completion marker
        header_df = pd.DataFrame(columns=["date","row","col","lon_center","lat_center",value_name])
        # Acquire lock briefly to avoid clobber races
        if not _acquire_lock(lock_path):
            # If can't lock, give up quietly
            return "lock-timeout", path, 0
        try:
            if skip_existing and os.path.exists(path) and not merge_on_write:
                _release_lock(lock_path)
                return "skip-existing", path, 0
            header_df.to_csv(tmp_path, index=False, mode="w", compression=compression)
            os.replace(tmp_path, path)
        finally:
            _release_lock(lock_path)
        return "written-empty", path, 0

    # Build dataframe for non-empty day
    sub = pd.DataFrame({
        "date":        date_str,
        "row":         rows[ok],
        "col":         cols[ok],
        "lon_center":  lonc[ok],
        "lat_center":  latc[ok],
        value_name:    vals[ok].astype("float64", copy=False),
    })

    # Acquire lock and write/merge atomically
    if not _acquire_lock(lock_path):
        return "lock-timeout", path, 0
    try:
        if os.path.exists(path):
            if skip_existing and not merge_on_write:
                return "skip-existing", path, 0
            if merge_on_write:
                try:
                    existing = pd.read_csv(path, compression=compression)
                    # concat & dedupe (prefer non-NaN values)
                    df = pd.concat([existing, sub], ignore_index=True)
                    # Sort so non-NaN values come first, then drop duplicates on (row,col)
                    df["_isnan"] = df[value_name].isna()
                    df.sort_values(by=["_isnan"], inplace=True)  # False (good values) first
                    df.drop(columns=["_isnan"], inplace=True)
                    df.drop_duplicates(subset=["row","col"], keep="first", inplace=True)
                    df.to_csv(tmp_path, index=False, compression=compression)
                    os.replace(tmp_path, path)
                    return "merged", path, len(sub)
                except Exception:
                    # Fallback: if merge fails for any reason, write a sidecar file
                    sidecar = os.path.join(outdir, f"{prefix}_{date_str}__{uuid.uuid4().hex}{ext}")
                    sub.to_csv(sidecar, index=False, compression=compression)
                    return "sidecar", sidecar, len(sub)
        # Fresh write
        sub.to_csv(tmp_path, index=False, compression=compression)
        os.replace(tmp_path, path)
        return "written", path, len(sub)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        _release_lock(lock_path)

# ------------------------------ Per-file worker ------------------------------

def process_one_file(tif_path, mask_path, args_dict):
    """Runs in a separate process when workers-files > 1."""
    # Rehydrate args as a simple namespace-like dict access
    ARG = type("ARG", (), args_dict)

    # Global date window
    dmin = pd.to_datetime(ARG.start).date() if ARG.start else None
    dmax = pd.to_datetime(ARG.end).date()   if ARG.end   else None

    # Read mask + metadata locally in this process
    keep_bool, mask_transform, mask_crs, mask_w, mask_h = read_mask_bool(mask_path, ARG.threshold)
    kept_frac = float(keep_bool.sum()) / float(keep_bool.size)
    sparse_mode = kept_frac < ARG.dense_threshold

    days_written = 0
    rows_written = 0

    # RasterIO / GDAL env
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS",
                      GDAL_CACHEMAX=ARG.gdal_cache_mb):
        with rasterio.open(tif_path) as ds:
            ensure_alignment_meta(ds, mask_transform, mask_crs, mask_w, mask_h)

            dates = band_dates_from_descriptions(ds)
            if dates is None:
                dates = infer_dates_for_stack(tif_path, ds.count, explicit_first=ARG.year_start)
            if dates is None or len(dates) != ds.count:
                return (os.path.basename(tif_path), "no-dates", 0, 0, sparse_mode, kept_frac)

            # Precompute geometry per file
            if sparse_mode:
                segments, keep_rows, keep_cols, lonc, latc = build_segments(ds, keep_bool)
                if len(segments) == 0:
                    return (os.path.basename(tif_path), "empty-mask", 0, 0, sparse_mode, kept_frac)
            else:
                rows, cols = np.nonzero(keep_bool)
                keep_rows = rows.astype(np.int32, copy=False)
                keep_cols = cols.astype(np.int32, copy=False)
                xs, ys = rasterio.transform.xy(ds.transform, keep_rows, keep_cols, offset="center")
                lonc = np.asarray(xs, dtype="float64")
                latc = np.asarray(ys, dtype="float64")

            nodata = ds.nodata
            band_indices = list(range(1, ds.count + 1))
            date_objs = [pd.to_datetime(d).date() for d in dates]

            tasks = []
            def do_band(b, d):
                if (dmin and d < dmin) or (dmax and d > dmax):
                    return ("skipped", None, 0)
                if sparse_mode:
                    vals = np.full(keep_rows.shape[0], np.nan, dtype="float64")
                    for (win, rr, cc, s, e) in segments:
                        block = ds.read(b, window=win, out_dtype="float32", masked=False)
                        v = block[rr, cc].astype("float64", copy=False)
                        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
                            v = np.where(v == nodata, np.nan, v)
                        vals[s:e] = v
                else:
                    arr = ds.read(b, out_dtype="float32", masked=False)
                    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
                        arr = np.where(arr == nodata, np.nan, arr)
                    vals = arr[keep_bool].astype("float64", copy=False)

                return write_day_csv_atomic(
                    ARG.outdir, ARG.prefix, d.strftime("%Y-%m-%d"),
                    keep_rows, keep_cols, lonc, latc, vals, ARG.value_name,
                    gzip=ARG.gzip, skip_existing=ARG.skip_existing, merge_on_write=ARG.merge_on_write
                )

            if ARG.workers_bands > 1:
                with ThreadPoolExecutor(max_workers=ARG.workers_bands) as tpool:
                    futs = [tpool.submit(do_band, b, d) for b, d in zip(band_indices, date_objs)]
                    for f in as_completed(futs):
                        status, path, nrows = f.result()
                        if status.startswith(("written", "merged", "sidecar", "written-empty")):
                            days_written += 1
                            rows_written += nrows
            else:
                for b, d in zip(band_indices, date_objs):
                    status, path, nrows = do_band(b, d)
                    if status.startswith(("written", "merged", "sidecar", "written-empty")):
                        days_written += 1
                        rows_written += nrows

    return (os.path.basename(tif_path), "ok", days_written, rows_written, sparse_mode, kept_frac)

# ------------------------------ MAIN ---------------------------------

def main():
    args = parse_args()

    tif_paths = expand_globs(args.tifs)
    if not tif_paths:
        print("No GeoTIFFs matched. Check --tifs.", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.mask):
        print(f"Mask not found: {args.mask}", file=sys.stderr)
        sys.exit(2)

    # Normalize workers
    if args.workers_files == 0:
        try:
            import multiprocessing as mp
            args.workers_files = max(1, mp.cpu_count())
        except Exception:
            args.workers_files = 1

    # Pack args for child processes (must be JSON/pickle-friendly)
    args_dict = dict(
        outdir=args.outdir,
        prefix=args.prefix,
        value_name=args.value_name,
        gzip=bool(args.gzip),
        skip_existing=bool(args.skip_existing),
        merge_on_write=bool(args.merge_on_write),
        start=args.start,
        end=args.end,
        year_start=args.year_start,
        gdal_cache_mb=int(args.gdal_cache_mb),
        dense_threshold=float(args.dense_threshold),
        threshold=float(args.threshold),
        workers_bands=int(args.workers_bands),
    )

    total_days = 0
    total_rows = 0

    if args.workers_files > 1 and len(tif_paths) > 1:
        print(f"[INFO] Parallel across files: {args.workers_files} workers | per-file band threads: {args.workers_bands}")
        with ProcessPoolExecutor(max_workers=args.workers_files) as pool:
            futs = [pool.submit(process_one_file, tif, args.mask, args_dict) for tif in tif_paths]
            for f in _tqdm(as_completed(futs), total=len(futs), desc="Files", unit="file"):
                name, status, days, rows, sparse_mode, kept_frac = f.result()
                if status == "ok":
                    total_days += days
                    total_rows += rows
                    print(f"  ✓ {name}: {days} days, {rows:,} rows | mode={'sparse' if sparse_mode else 'dense'} ({kept_frac:.1%})")
                elif status == "no-dates":
                    print(f"  ⚠ {name}: could not infer per-band dates, skipped.")
                elif status == "empty-mask":
                    print(f"  ⚠ {name}: mask kept zero pixels, skipped.")
                else:
                    print(f"  ? {name}: status={status}")
    else:
        print(f"[INFO] Serial over files | per-file band threads: {args.workers_bands}")
        for tif in _tqdm(tif_paths, desc="Files", unit="file"):
            name, status, days, rows, sparse_mode, kept_frac = process_one_file(tif, args.mask, args_dict)
            if status == "ok":
                total_days += days
                total_rows += rows
                print(f"  ✓ {name}: {days} days, {rows:,} rows | mode={'sparse' if sparse_mode else 'dense'} ({kept_frac:.1%})")
            elif status == "no-dates":
                print(f"  ⚠ {name}: could not infer per-band dates, skipped.")
            elif status == "empty-mask":
                print(f"  ⚠ {name}: mask kept zero pixels, skipped.")
            else:
                print(f"  ? {name}: status={status}")

    print(f"\nDone. Wrote/updated ~{total_days} day files with ~{total_rows:,} total rows.")
    print(f"Output folder: {args.outdir}")

if __name__ == "__main__":
    main()
