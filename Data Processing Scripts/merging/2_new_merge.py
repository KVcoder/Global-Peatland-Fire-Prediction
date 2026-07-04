#!/usr/bin/env python3
"""
new_merge.py — Build daily peat-only GeoTIFFs from ERA5-Land monthly stacked tiles,
then validate each output and log its properties to CSV.

Deps:
  conda/pip install rioxarray xarray rasterio numpy tqdm
"""

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import rasterio
from tqdm import tqdm

import rioxarray as riox
import xarray as xr
from rioxarray.merge import merge_arrays

# ---------- Parsers ----------
FN_MONTH_RE = re.compile(r"monthly_stack_(\d{4})_(\d{2})", re.IGNORECASE)
DATE_RE = re.compile(r"(20\d{6})")  # YYYYMMDD in band description

def sanitize_var(desc: str) -> str:
    if not desc:
        return "band"
    name = DATE_RE.sub("", desc)
    name = re.sub(r"[^\w]+", "_", name).strip("_")
    return name or "band"

# ---------- Discovery ----------
def find_tiles(input_dir: Path, pattern: str):
    files = sorted(Path(input_dir).rglob(pattern))
    if not files:
        raise SystemExit(f"No files found under {input_dir} matching pattern: {pattern}")
    return files

def group_by_month(files):
    groups = defaultdict(list)
    for f in files:
        m = FN_MONTH_RE.search(f.name)
        if m:
            y, mth = m.group(1), m.group(2)
            groups[(y, mth)].append(f)
    if not groups:
        raise SystemExit("Could not parse (year, month) from filenames; check your pattern.")
    return OrderedDict(sorted(groups.items()))

def build_month_catalog(files):
    """
    Return:
      ordered catalog: date -> {"vars": [var...], "map": var -> [(path, band_index)...]}
      common dtype, crs
    """
    catalog = {}
    common_dtype = None
    common_crs = None

    for path in files:
        with rasterio.open(path) as ds:
            if common_dtype is None:
                common_dtype = ds.dtypes[0]
            elif ds.dtypes[0] != common_dtype:
                raise RuntimeError(f"Inconsistent dtype: {path} has {ds.dtypes[0]}, expected {common_dtype}")

            if common_crs is None:
                common_crs = ds.crs
            elif ds.crs != common_crs:
                raise RuntimeError(f"Inconsistent CRS in {path}: {ds.crs} vs {common_crs}")

            descs = list(ds.descriptions or [None] * ds.count)
            for bidx in range(1, ds.count + 1):
                desc = descs[bidx - 1] or f"b{bidx}"
                m = DATE_RE.search(desc)
                if not m:
                    raise RuntimeError(f"{path.name} band {bidx} lacks YYYYMMDD in description: '{desc}'")
                date = m.group(1)
                var = sanitize_var(desc)
                if date not in catalog:
                    catalog[date] = {"vars": set(), "map": defaultdict(list)}
                catalog[date]["vars"].add(var)
                catalog[date]["map"][var].append((path, bidx))

    for date, d in catalog.items():
        d["vars"] = sorted(d["vars"])

    ordered = OrderedDict(sorted(catalog.items()))
    return ordered, common_dtype, common_crs

# ---------- Helpers ----------
def choose_nodata(dtype, sample_nodata):
    if sample_nodata is not None:
        return sample_nodata
    return -9999.0 if np.issubdtype(np.dtype(dtype), np.floating) else -9999

def mosaic_var_for_date(entries, nodata_val):
    """
    entries = [(file_path, band_index), ...]
    Returns DataArray (y,x) merged geospatially (first-valid wins).
    """
    arrays = []
    for p, bidx in entries:
        da = riox.open_rasterio(str(p), masked=True)  # (band, y, x)
        da1 = da.isel(band=bidx-1).squeeze(drop=True)
        da1 = da1.rio.write_nodata(nodata_val, inplace=False)
        arrays.append(da1)

    merged = merge_arrays(arrays, nodata=nodata_val)  # rioxarray merge wrapper. :contentReference[oaicite:1]{index=1}
    return merged  # 2D DataArray

def dataarrays_to_multiband(arrays, varnames):
    """
    arrays: list of 2D DataArrays (y,x) with same grid → 3D DataArray ('band','y','x')
    """
    stacked = xr.concat([a.expand_dims({"band": [i+1]}) for i, a in enumerate(arrays)], dim="band")
    stacked = stacked.transpose("band", "y", "x")

    # Persist CRS/transform CF metadata correctly. :contentReference[oaicite:2]{index=2}
    stacked = stacked.rio.write_crs(arrays[0].rio.crs, inplace=True)
    # Usually coordinates exist; write_transform is only needed if not saving x/y coords. :contentReference[oaicite:3]{index=3}
    try:
        stacked = stacked.rio.write_transform(arrays[0].rio.transform(), inplace=True)
    except Exception:
        pass
    stacked = stacked.rio.write_coordinate_system(inplace=True)  # signature is (inplace=...). :contentReference[oaicite:4]{index=4}

    # Make band index explicit (1..N)
    stacked = stacked.assign_coords(band=np.arange(1, len(varnames)+1))
    return stacked

def write_daily(out_path: Path, da3d, varnames, dtype, nodata_val, compress="DEFLATE"):
    """
    da3d: DataArray ('band','y','x') with CRS/transform set.
    """
    # Ensure nodata & descriptions metadata are consistent
    da3d = da3d.rio.write_nodata(nodata_val, inplace=False)
    # Avoid rioxarray's long_name length mismatch by setting it to our varnames (or remove it). :contentReference[oaicite:5]{index=5}
    da3d = da3d.copy()
    da3d.attrs.pop("long_name", None)
    da3d.attrs["long_name"] = tuple(varnames)

    da3d = da3d.astype(dtype, copy=False)
    da3d.rio.to_raster(out_path, tiled=True, BIGTIFF="YES", compress=compress)  # 2D/3D supported. :contentReference[oaicite:6]{index=6}

    # Also set RasterIO band descriptions explicitly (most robust). :contentReference[oaicite:7]{index=7}
    with rasterio.open(out_path, "r+") as dst:
        dst.descriptions = list(varnames)

def validate_and_summarize(tif_path: Path, expected_count: int, expected_crs, expected_dtype: str,
                           expected_nodata, valid_pixels: int | None):
    """Open the written GeoTIFF and return a summary dict + boolean 'ok' flag."""
    errs = []
    with rasterio.open(tif_path) as ds:
        if ds.count != expected_count:
            errs.append(f"count {ds.count} != {expected_count}")
        if expected_crs and ds.crs != expected_crs:
            errs.append("CRS mismatch")
        if ds.dtypes[0] != expected_dtype:
            errs.append(f"dtype {ds.dtypes[0]} != {expected_dtype}")
        # descriptions length check
        descs = ds.descriptions or tuple([None]*ds.count)
        if len(descs) != ds.count:
            errs.append("descriptions length mismatch")
        if expected_nodata is not None and ds.nodata != expected_nodata:
            errs.append(f"nodata {ds.nodata} != {expected_nodata}")

        # Basic geotransform sanity
        transform = ds.transform
        if transform.a == 0 or transform.e == 0:
            errs.append("degenerate transform")

        # assemble summary
        epsg = ds.crs.to_epsg() if ds.crs else None
        summary = {
            "file": str(tif_path.name),
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtype": ds.dtypes[0],
            "crs_epsg": epsg or "",
            "nodata": "" if ds.nodata is None else ds.nodata,
            "pixel_size_x": transform.a,
            "pixel_size_y": transform.e,
            "origin_x": transform.c,
            "origin_y": transform.f,
            "valid_pixels": "" if valid_pixels is None else int(valid_pixels),
            "band_names": "|".join([d or "" for d in descs]),
            "ok": "yes" if not errs else "no",
            "errors": ";".join(errs),
        }
    return summary

def append_csv(csv_path: Path, row: dict):
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Split ERA5-Land peat-masked monthly stacks into daily GeoTIFFs; validate & log to CSV.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pattern", default="era5land_monthly_stack_*.tif")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compress", default="DEFLATE")
    ap.add_argument("--report", action="store_true", help="Print per-day valid pixel counts to stderr")
    ap.add_argument("--csv", default=None, help="Path to CSV log (default: <output-dir>/daily_index.csv)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else (out_dir / "daily_index.csv")

    files = find_tiles(in_dir, args.pattern)
    by_month = group_by_month(files)

    total_written = 0
    for (yyyy, mm), month_files in by_month.items():
        month_tag = f"{yyyy}-{mm}"
        catalog, dtype, crs = build_month_catalog(month_files)

        with rasterio.open(str(month_files[0])) as ref:
            nodata_val = choose_nodata(dtype, ref.nodata)

        for date, payload in tqdm(catalog.items(), desc=f"{month_tag}", unit="day"):
            out_path = out_dir / f"era5land_daily_{date}.tif"
            if out_path.exists() and not args.overwrite:
                # still record to CSV if file exists — open to summarize
                with rasterio.open(out_path) as ds:
                    descs = ds.descriptions or tuple([None]*ds.count)
                row = validate_and_summarize(out_path, expected_count=len(descs), expected_crs=ds.crs,
                                             expected_dtype=ds.dtypes[0], expected_nodata=ds.nodata,
                                             valid_pixels=None)
                append_csv(csv_path, row)
                continue

            var_arrays = []
            varnames = []
            for var in payload["vars"]:
                entries = payload["map"][var]
                if not entries:
                    continue
                merged_da = mosaic_var_for_date(entries, nodata_val=nodata_val)
                var_arrays.append(merged_da)
                varnames.append(var)

            if not var_arrays:
                continue

            # Build 3D array & write
            da3d = dataarrays_to_multiband(var_arrays, varnames)
            write_daily(out_path, da3d, varnames, dtype=dtype, nodata_val=nodata_val, compress=args.compress)

            # Count valid pixels (at least one band valid) from in-memory arrays for CSV/report
            data_stack = np.stack([va.values for va in var_arrays], axis=0)
            valid_mask = np.any(data_stack != nodata_val, axis=0) if not np.issubdtype(np.dtype(dtype), np.floating) \
                         else (np.any(data_stack != nodata_val, axis=0) & np.isfinite(np.nanmin(data_stack, axis=0)))
            n_valid = int(valid_mask.sum())

            if args.report:
                import sys
                print(f"{date}: valid_pixels={n_valid}", file=sys.stderr)

            # Validate written file & append to CSV
            row = validate_and_summarize(out_path, expected_count=len(varnames), expected_crs=crs,
                                         expected_dtype=dtype, expected_nodata=nodata_val, valid_pixels=n_valid)
            append_csv(csv_path, row)
            total_written += 1

    print(f"Done. Wrote {total_written} daily file(s) to {out_dir}\nLog: {csv_path}")

if __name__ == "__main__":
    main()
