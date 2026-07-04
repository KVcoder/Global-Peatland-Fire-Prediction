#!/usr/bin/env python3
"""
QC for merged ERA5-Land monthly stacks (accelerated + tqdm progress).

Improvements vs. previous draft
- Robust to 0-band files (won't crash on "Image has no bands").
- Correct stats for integer rasters (avoid .filled(np.nan) crash by upcasting to float32).
- Faster, fully vectorized band stats with safe NaN handling.
- Optional per-band CSV; stable ordering; resilient progress even on errors.

Usage example (20 cores machine, but keep I/O contention modest):
  python qc_era5l_monthly_stacks.py \
    --input-dir ERA5_LAND_MERGED \
    --name-contains era5land_monthly_stack_ \
    --report qc_report.csv \
    --per-band-report qc_per_band.csv \
    --thumb-size 256 \
    --workers 4 \
    --verbose

Notes
- For GDAL-backed rasters, heavy multi-threading can contend on I/O. Start with --workers 2-4.
- If you hit MemoryError on very band-rich files, reduce --thumb-size (e.g., 128) or --workers.
"""

from __future__ import annotations
import argparse
import calendar
import csv
import os
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import rasterio
from rasterio.enums import Resampling

# --- tqdm: graceful fallback if not installed ---
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)

from concurrent.futures import ThreadPoolExecutor, as_completed

PAT = re.compile(
    r"""(?x)
    (?P<prefix>.*?era5land_monthly_stack_)?
    (?P<year>\d{4})_(?P<month>\d{2})
    (?:_(?P<alias>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*))?
    \.tif$"""
)


def parse_name(p: Path) -> Optional[Dict]:
    m = PAT.search(p.name)
    if not m:
        return None
    d = m.groupdict()
    d["year"] = int(d["year"]) 
    d["month"] = int(d["month"]) 
    d["alias"] = d["alias"] or ""  # '' if not split-by-variable
    return d


def days_in_month(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]


def list_inputs(in_dir: Path, contains: Optional[str]) -> List[Path]:
    pats = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
    files: List[Path] = []
    for pat in pats:
        files.extend(in_dir.rglob(pat))
    if contains:
        files = [f for f in files if contains in f.name]
    files = [f for f in files if parse_name(f)]
    files.sort()
    return files


def compute_out_shape(h: int, w: int, max_dim: int) -> Tuple[int, int]:
    scale = max(h, w) / float(max_dim)
    if scale <= 1:
        return h, w
    out_h = max(1, int(round(h / scale)))
    out_w = max(1, int(round(w / scale)))
    return out_h, out_w


def read_all_bands_thumb(src: rasterio.DatasetReader, max_dim: int) -> np.ma.MaskedArray:
    """Read all bands as one masked array of shape (bands, out_h, out_w)."""
    h, w = src.height, src.width
    out_h, out_w = compute_out_shape(h, w, max_dim)
    # Choose resampling: average better for floats; nearest for ints (avoids implicit cast errors)
    try:
        dtype0 = np.dtype(src.dtypes[0]) if src.count > 0 else np.float32
    except Exception:
        dtype0 = np.float32
    resamp = Resampling.average if np.issubdtype(dtype0, np.floating) else Resampling.nearest
    marr = src.read(
        indexes=None,  # read all bands
        out_shape=(src.count, out_h, out_w),
        resampling=resamp,
        masked=True,
    )
    return marr  # masked array (bands, out_h, out_w)


def band_stats_vectorized(marr: np.ma.MaskedArray) -> Dict[str, np.ndarray]:
    # Upcast to float32 so we can safely use NaN for masking regardless of input dtype
    marr_f = marr.astype(np.float32, copy=False)
    mask = np.ma.getmaskarray(marr_f)  # shape: (bands, H, W)
    total = mask.shape[1] * mask.shape[2]
    # Count valid (unmasked) pixels per band
    n_valid = np.sum(~mask, axis=(1, 2)).astype(np.int64)
    frac_valid = np.where(total > 0, n_valid / total, 0.0)

    # Convert to plain ndarray with NaNs where masked
    data = marr_f.filled(np.nan)
    with np.errstate(all="ignore"):
        vmin = np.nanmin(data, axis=(1, 2))
        vmax = np.nanmax(data, axis=(1, 2))

    all_nodata = (n_valid == 0)
    tol = 0.0
    # Any non-zero anywhere (excluding NaNs)
    with np.errstate(invalid="ignore"):
        any_nonzero = np.any(np.abs(data) > tol, axis=(1, 2))
    all_zero = (~all_nodata) & (~any_nonzero)

    # Constant band if min == max (and not all-NaN)
    with np.errstate(invalid="ignore"):
        constant = (~all_nodata) & np.isclose(vmin, vmax, equal_nan=False)

    # Replace min/max with None for all-NaN bands (friendlier CSV)
    vmin_out = vmin.astype(object)
    vmax_out = vmax.astype(object)
    vmin_out[all_nodata] = None
    vmax_out[all_nodata] = None

    return {
        "frac_valid": frac_valid,
        "min": vmin_out,
        "max": vmax_out,
        "all_nodata": all_nodata,
        "all_zero": all_zero,
        "constant": constant,
        "n_valid": n_valid,
    }


def infer_num_variables(band_count: int, dim: int) -> Optional[int]:
    if dim <= 0 or band_count % dim != 0:
        return None
    v = band_count // dim
    return v if 1 <= v <= 40 else None


def process_file(
    p: Path,
    thumb_size: int,
    verbose: bool,
    want_per_band_rows: bool,
) -> Tuple[Dict, List[Dict], Tuple[str, Tuple[float, float, float, float, float, float], int, int, object]]:
    meta = parse_name(p)
    y, m, alias = meta["year"], meta["month"], meta["alias"]
    dim = days_in_month(y, m)

    with rasterio.open(p) as src:
        bcount = src.count
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height
        nodata = src.nodata
        dtype = src.dtypes[0] if bcount > 0 else None

        # Signature for grid consistency checks (include nodata too)
        sig = (str(crs), tuple(transform)[:6], width, height, nodata)

        split_by_var = (alias != "")
        if bcount == 0:
            inferred_vars = None
            expected = 0
            ok_bands = False
            # Minimal stats (no bands)
            suspicious = 0
            all_nodata_bands = 0
            all_zero_bands = 0
            constant_bands = 0
            band_rows: List[Dict] = []
        else:
            # Derive expected band count
            if split_by_var:
                expected = dim
                inferred_vars = 1
                ok_bands = (bcount == expected)
            else:
                inferred_vars = infer_num_variables(bcount, dim)
                expected = dim * (inferred_vars or 0)
                ok_bands = (inferred_vars is not None and bcount == expected)

            # Read thumbnail & compute stats
            try:
                marr = read_all_bands_thumb(src, max_dim=thumb_size)
            except MemoryError:
                # Helpful message encouraging smaller thumb size / fewer workers
                raise MemoryError(
                    f"MemoryError while reading thumbnail for {p.name}. "
                    f"Try reducing --thumb-size (e.g., 128) and/or --workers."
                )
            stats = band_stats_vectorized(marr)

            if verbose:
                for b in tqdm(range(bcount), leave=False, desc=f"{p.name} bands", unit="band"):
                    print(
                        f"{p.name} band {b+1:>3}: valid={stats['frac_valid'][b]:.3f} "
                        f"min={stats['min'][b]} max={stats['max'][b]} "
                        f"{'ALL_NODATA' if stats['all_nodata'][b] else ''} "
                        f"{'ALL_ZERO' if stats['all_zero'][b] else ''} "
                        f"{'CONST' if stats['constant'][b] else ''}"
                    )

            suspicious = int(np.count_nonzero(stats["all_nodata"] | stats["all_zero"] | stats["constant"]))
            all_nodata_bands = int(np.count_nonzero(stats["all_nodata"]))
            all_zero_bands = int(np.count_nonzero(stats["all_zero"]))
            constant_bands = int(np.count_nonzero(stats["constant"]))

            band_rows = []
            if want_per_band_rows or verbose:
                for b in tqdm(range(bcount), leave=False, desc=f"{p.name} per-band rows", unit="band"):
                    band_rows.append(
                        {
                            "file": p.name,
                            "year": y,
                            "month": m,
                            "alias": alias or "(multi)",
                            "band": b + 1,
                            "frac_valid": float(stats["frac_valid"][b]),
                            "min": stats["min"][b],
                            "max": stats["max"][b],
                            "all_nodata": bool(stats["all_nodata"][b]),
                            "all_zero": bool(stats["all_zero"][b]),
                            "constant": bool(stats["constant"][b]),
                        }
                    )

    row = {
        "file": p.name,
        "year": y,
        "month": m,
        "alias": alias or "(multi)",
        "split_by_variable": split_by_var,
        "bands": bcount,
        "expected_bands": expected,
        "days_in_month": dim,
        "inferred_variables": inferred_vars,
        "band_count_ok": ok_bands,
        "dtype": dtype,
        "nodata": nodata,
        "crs": str(crs),
        "width": width,
        "height": height,
        "transform0": transform.a,
        "transform1": transform.b,
        "transform2": transform.c,
        "transform3": transform.d,
        "transform4": transform.e,
        "transform5": transform.f,
        "suspicious_bands": suspicious,
        "all_nodata_bands": all_nodata_bands,
        "all_zero_bands": all_zero_bands,
        "constant_bands": constant_bands,
    }

    return row, band_rows, sig


def main():
    ap = argparse.ArgumentParser(description="QC merged ERA5-Land monthly stacks (accelerated + tqdm).")
    ap.add_argument("--input-dir", required=True, help="Directory with merged GeoTIFFs")
    ap.add_argument("--name-contains", default="era5land_monthly_stack_", help="Filename substring filter")
    ap.add_argument("--report", default="qc_report.csv", help="Output CSV (file-level summary)")
    ap.add_argument("--per-band-report", default=None, help="Optional per-band CSV (may be large)")
    ap.add_argument("--thumb-size", type=int, default=256, help="Max thumbnail dimension for stats")
    ap.add_argument("--verbose", action="store_true", help="Print per-band findings (with tqdm)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel file reads; 1 = serial (safest for GDAL).")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    files = list_inputs(in_dir, args.name_contains)
    if not files:
        print("No matching files found.")
        return

    rows: List[Dict] = []
    band_rows_all: List[Dict] = []
    first_sig = None
    crs_mismatch = 0
    transform_mismatch = 0
    nodata_mismatch = 0

    def handle_result(row, band_rows, sig):
        nonlocal first_sig, crs_mismatch, transform_mismatch, nodata_mismatch
        rows.append(row)
        if args.per_band_report is not None or args.verbose:
            band_rows_all.extend(band_rows)
        if first_sig is None:
            first_sig = sig
        else:
            if sig[0] != first_sig[0]:
                crs_mismatch += 1
            if sig[1:4] != first_sig[1:4]:  # transform (a..f sliced to 6), width, height
                transform_mismatch += 1
            if sig[4] != first_sig[4]:
                nodata_mismatch += 1

    # --- Progress over files ---
    if args.workers > 1:
        # Limit workers to avoid oversubscription on machines with huge core counts
        max_workers = max(1, min(args.workers, (os.cpu_count() or 8)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(process_file, p, args.thumb_size, args.verbose,
                              (args.per_band_report is not None)): p for p in files}
            with tqdm(total=len(futs), desc="Files", unit="file") as pbar:
                for fut in as_completed(futs):
                    p = futs[fut]
                    try:
                        row, band_rows, sig = fut.result()
                        handle_result(row, band_rows, sig)
                    except Exception as e:
                        # Still advance the bar so you see progress
                        print(f"ERROR: {p.name} -> {e}")
                    finally:
                        pbar.update(1)
    else:
        for p in tqdm(files, desc="Files", unit="file"):
            try:
                row, band_rows, sig = process_file(
                    p,
                    thumb_size=args.thumb_size,
                    verbose=args.verbose,
                    want_per_band_rows=(args.per_band_report is not None),
                )
                handle_result(row, band_rows, sig)
            except Exception as e:
                print(f"ERROR: {p.name} -> {e}")

    # Keep file order stable
    rows.sort(key=lambda r: (r["year"], r["month"], r["file"]))

    # Write summary CSV
    with tqdm(total=1, desc="Write summary CSV", unit="step", leave=False) as _:
        with open(args.report, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Write per-band CSV (optional)
    if band_rows_all and args.per_band_report:
        # Keep stable ordering
        band_rows_all.sort(key=lambda r: (r["year"], r["month"], r["file"], r["band"]))
        # Chunked write to keep responsive feel on very large outputs
        with tqdm(total=len(band_rows_all), desc="Write per-band CSV", unit="row") as pbar:
            with open(args.per_band_report, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(band_rows_all[0].keys()))
                w.writeheader()
                chunk = 10000
                for i in range(0, len(band_rows_all), chunk):
                    w.writerows(band_rows_all[i:i + chunk])
                    pbar.update(min(chunk, len(band_rows_all) - i))

    # Console summary
    print(f"\nChecked {len(rows)} file(s).")
    if crs_mismatch or transform_mismatch or nodata_mismatch:
        print(
            "WARNING: grid mismatch across files "
            f"(CRS mismatches: {crs_mismatch}, transform/size mismatches: {transform_mismatch}, nodata mismatches: {nodata_mismatch})."
        )
    bad_bandcount = [r for r in rows if not r["band_count_ok"]]
    many_suspicious = [r for r in rows if r["suspicious_bands"] > 0]
    if bad_bandcount:
        print(f"Band-count errors in {len(bad_bandcount)} file(s):")
        for r in bad_bandcount[:10]:
            print(
                f"  - {r['file']}: bands={r['bands']} expected={r['expected_bands']} "
                f"(days={r['days_in_month']}, inferred_vars={r['inferred_variables']})"
            )
        if len(bad_bandcount) > 10:
            print(f"  ... and {len(bad_bandcount) - 10} more")
    if many_suspicious:
        print(
            f"Suspicious content detected in {len(many_suspicious)} file(s) "
            f"(any all-NaN/zero/constant bands). See CSV for counts."
        )
    print(f"CSV written: {args.report}")
    if args.per_band_report:
        print(f"Per-band CSV written: {args.per_band_report}")
    print("Done.")


if __name__ == "__main__":
    main()
