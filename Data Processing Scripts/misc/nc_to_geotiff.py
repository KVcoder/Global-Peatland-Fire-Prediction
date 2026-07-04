#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def pick_data_var(ds, varname=None):
    if varname:
        if varname not in ds.data_vars:
            raise KeyError(f"Variable '{varname}' not found. Available: {list(ds.data_vars)}")
        return ds[varname]

    candidates = []
    for v in ds.data_vars:
        da = ds[v]
        if da.ndim >= 2 and np.issubdtype(da.dtype, np.floating):
            candidates.append(v)
    if not candidates:
        for v in ds.data_vars:
            if ds[v].ndim >= 2:
                candidates.append(v)
    if not candidates:
        raise ValueError("No suitable data variable found in NetCDF.")
    return ds[candidates[0]]


def infer_lat_lon_names(da):
    coord_names = set(list(da.coords) + list(da.dims))
    lat_names = ["lat", "latitude", "y"]
    lon_names = ["lon", "longitude", "x"]
    lat = next((n for n in lat_names if n in coord_names), None)
    lon = next((n for n in lon_names if n in coord_names), None)
    return lat, lon


def build_transform_from_1d_latlon(lat_vals, lon_vals):
    lat_vals = np.asarray(lat_vals)
    lon_vals = np.asarray(lon_vals)

    dlon = np.median(np.diff(np.sort(lon_vals)))
    dlat = np.median(np.diff(np.sort(lat_vals)))

    west = lon_vals.min() - dlon / 2.0
    east = lon_vals.max() + dlon / 2.0
    south = lat_vals.min() - dlat / 2.0
    north = lat_vals.max() + dlat / 2.0

    height = lat_vals.size
    width = lon_vals.size

    transform = from_bounds(west, south, east, north, width, height)
    return transform, (height, width)


def ensure_band_first(arr, dims):
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]

    if arr.ndim == 3:
        spatial_names = {"lat", "latitude", "lon", "longitude", "x", "y"}
        band_dim = None
        for i, d in enumerate(dims):
            if d not in spatial_names:
                band_dim = i
                break
        if band_dim is None:
            band_dim = 0
        if band_dim != 0:
            arr = np.moveaxis(arr, band_dim, 0)
        return arr

    raise ValueError(f"Unsupported array ndim={arr.ndim}; expected 2D or 3D.")


def convert_one_nc(nc_path, out_dir, varname, crs, nodata):
    base = os.path.splitext(os.path.basename(nc_path))[0]
    out_path = os.path.join(out_dir, f"{base}.tif")

    ds = xr.open_dataset(nc_path, decode_coords="all")
    try:
        da = pick_data_var(ds, varname)
        lat_name, lon_name = infer_lat_lon_names(da)

        if lat_name not in da.coords or lon_name not in da.coords:
            raise ValueError(
                f"[{os.path.basename(nc_path)}] Expected 1D lat/lon coords. "
                f"Found lat='{lat_name}', lon='{lon_name}', coords={list(da.coords)}"
            )

        lat_vals = da[lat_name].values
        lon_vals = da[lon_name].values
        if lat_vals.ndim != 1 or lon_vals.ndim != 1:
            raise ValueError(f"[{os.path.basename(nc_path)}] lat/lon are not 1D.")

        transform, (h, w) = build_transform_from_1d_latlon(lat_vals, lon_vals)

        arr = da.values
        dims = list(da.dims)

        # Flip lat if ascending so row 0 = north
        if lat_name in dims:
            lat_axis = dims.index(lat_name)
            if lat_vals[0] < lat_vals[-1]:
                arr = np.flip(arr, axis=lat_axis)

        arr = np.asarray(arr, dtype=np.float32)
        arr = ensure_band_first(arr, dims)

        # NaN -> nodata
        arr = np.where(np.isfinite(arr), arr, np.float32(nodata))

        bands, height, width = arr.shape
        if height != h or width != w:
            raise ValueError(
                f"[{os.path.basename(nc_path)}] Shape mismatch: data={arr.shape}, coords={(h, w)}"
            )

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": bands,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "nodata": np.float32(nodata),
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        with rasterio.open(out_path, "w", **profile) as dst:
            for b in range(bands):
                dst.write(arr[b, :, :], b + 1)

        return (nc_path, out_path, None)
    except Exception as e:
        return (nc_path, None, str(e))
    finally:
        ds.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--var", default=None)
    ap.add_argument("--crs", default="EPSG:4326")
    ap.add_argument("--nodata", type=float, default=-9999.0)
    ap.add_argument("--pattern", default="*.nc")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    nc_files = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    if not nc_files:
        raise FileNotFoundError(f"No files found in {args.in_dir} matching {args.pattern}")

    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(convert_one_nc, f, args.out_dir, args.var, args.crs, args.nodata)
            for f in nc_files
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="NC→GeoTIFF"):
            in_path, out_path, err = fut.result()
            if err:
                failures.append((in_path, err))

    if failures:
        print("\nSome files failed:")
        for f, err in failures[:20]:
            print(f"- {os.path.basename(f)}: {err}")
        if len(failures) > 20:
            print(f"... and {len(failures) - 20} more")
        raise SystemExit(1)

    print(f"\nDone. Wrote {len(nc_files)} GeoTIFFs to {args.out_dir}")


if __name__ == "__main__":
    main()
