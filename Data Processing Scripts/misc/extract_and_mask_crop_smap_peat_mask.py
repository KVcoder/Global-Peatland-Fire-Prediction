#!/usr/bin/env python3
"""
extract_and_mask_crop_smap_peat_mask.py

Goal
----
1) Read `peat_mask` from a full GeoTIFF→Zarr store (created by geotiff_to_zarr.py).
2) Write a NEW Zarr v2 store that contains ONLY ONE 2D array named `peat_mask`.
3) "Crop" by lat/lon bbox WITHOUT shrinking:
   - keep original (H, W) shape
   - set everything OUTSIDE bbox to NoData = -9999
   - inside bbox, preserve the *actual peat mask values* from the input `peat_mask`
     (i.e., peatlands remain 1, non-peat remains 0; we do NOT mark the whole bbox as peat)

Important
---------
- NoData is fixed to -9999.
- If the input peat_mask is uint8/uint16 (common 0/1 mask), it cannot represent -9999.
  In that case we automatically write output as int16 to support -9999.

Requirements: zarr, fsspec, numpy
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import List, Tuple

import numpy as np
import fsspec
import zarr


NODATA_VALUE = -9999


def _open_group(path_or_store, consolidated: bool = True):
    """
    Open Zarr group for reading.
    Tries consolidated metadata first (if requested), then falls back.
    Forces v2 on fallback so numcodecs-based stores open cleanly.
    """
    store = path_or_store
    if isinstance(path_or_store, str) and path_or_store.lower().endswith(".zip"):
        store = fsspec.get_mapper(f"zip://{path_or_store}")

    if consolidated:
        try:
            return zarr.open_consolidated(store, mode="r")
        except Exception:
            pass

    return zarr.open_group(store, mode="r", zarr_version=2)


def _make_output_group(out_path: str, overwrite: bool, zip_store: bool = False):
    """
    Create output as Zarr v2 (so numcodecs compressors work).
    """
    if zip_store:
        if overwrite and os.path.exists(out_path):
            os.remove(out_path)
        store = fsspec.get_mapper(f"zip://{out_path}")
        return zarr.open_group(store=store, mode="w", zarr_version=2), store

    if overwrite and os.path.exists(out_path):
        shutil.rmtree(out_path)
    return zarr.open_group(out_path, mode="w", zarr_version=2), out_path


def _get_axis(zin, names: List[str]) -> np.ndarray:
    for n in names:
        if n in zin:
            return np.asarray(zin[n][:])
    raise KeyError(f"Could not find any of axes {names} in input store. Found arrays: {list(zin.array_keys())}")


def _normalize_lon_to_axis(lon: float, x_axis: np.ndarray) -> float:
    """
    If axis looks like [0..360] and lon is in [-180..180], map negatives to +360.
    """
    xmin = float(np.nanmin(x_axis))
    xmax = float(np.nanmax(x_axis))
    if xmin >= -1.0 and xmax > 180.0 and lon < 0.0:
        return lon + 360.0
    return lon


def _find_index_slice_1d(coord_1d: np.ndarray, vmin: float, vmax: float, name: str) -> slice:
    """
    Returns slice(start, stop) covering all coord values in [min(vmin,vmax), max(vmin,vmax)].
    Works for increasing or decreasing axes.
    """
    coord = np.asarray(coord_1d)
    lo = min(vmin, vmax)
    hi = max(vmin, vmax)
    mask = (coord >= lo) & (coord <= hi)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(
            f"No {name} values found in range [{lo}, {hi}]. "
            f"Actual min/max: [{float(np.min(coord))}, {float(np.max(coord))}]"
        )
    return slice(int(idx.min()), int(idx.max()) + 1)


def _compute_slices_for_bbox(
    y_axis: np.ndarray,
    x_axis: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Tuple[slice, List[slice]]:
    """
    Returns:
      y_slice (contiguous)
      x_slices (one slice, or two slices if bbox wraps in 0..360)
    """
    y_slice = _find_index_slice_1d(y_axis, lat_min, lat_max, name="lat(y)")

    lon_min_n = _normalize_lon_to_axis(lon_min, x_axis)
    lon_max_n = _normalize_lon_to_axis(lon_max, x_axis)

    if lon_min_n <= lon_max_n:
        x_slices = [_find_index_slice_1d(x_axis, lon_min_n, lon_max_n, name="lon(x)")]
    else:
        # wrap: [lon_min..max] U [min..lon_max]
        x_slices = [
            _find_index_slice_1d(x_axis, lon_min_n, float(np.max(x_axis)), name="lon(x)-hi"),
            _find_index_slice_1d(x_axis, float(np.min(x_axis)), lon_max_n, name="lon(x)-lo"),
        ]

    return y_slice, x_slices


def _choose_output_dtype(in_dtype: np.dtype) -> np.dtype:
    """
    Ensure output dtype can represent -9999.
    - If input is unsigned int: promote to int16 (or int32 if needed).
    - If input is signed int: keep it if it can represent -9999, else promote.
    - If input is float: keep it.
    """
    dt = np.dtype(in_dtype)

    if np.issubdtype(dt, np.floating):
        return dt

    if np.issubdtype(dt, np.unsignedinteger):
        # peat masks are typically 0/1; int16 is plenty and supports -9999
        return np.dtype(np.int16)

    if np.issubdtype(dt, np.signedinteger):
        info = np.iinfo(dt)
        if info.min <= NODATA_VALUE <= info.max:
            return dt
        # promote
        if np.iinfo(np.int16).min <= NODATA_VALUE <= np.iinfo(np.int16).max:
            return np.dtype(np.int16)
        return np.dtype(np.int32)

    # fallback
    return np.dtype(np.int16)


def _copy_compressor_and_chunks(in_arr):
    """
    Best-effort reuse of compressor/chunks from the input Zarr v2 array.
    (Newer zarr warns that 'compressor' is deprecated; that's fine for v2 output.)
    """
    compressor = getattr(in_arr, "compressor", None)
    chunks = getattr(in_arr, "chunks", None)
    return compressor, chunks


def main():
    ap = argparse.ArgumentParser(
        description="Extract peat_mask-only Zarr (v2), mask outside bbox to NoData=-9999 (no shrinking)."
    )
    ap.add_argument("--in-zarr", required=True, help="Input full Zarr store (.zarr dir or .zip)")
    ap.add_argument("--out-zarr", required=True, help="Output Zarr store (.zarr dir or .zip)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")

    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)

    ap.add_argument(
        "--zip-store",
        action="store_true",
        help="Write output as a single .zip store (out-zarr should end with .zip).",
    )
    ap.add_argument(
        "--consolidated",
        action="store_true",
        help="Prefer consolidated metadata when reading input (tries, then falls back).",
    )

    args = ap.parse_args()

    # ---- Open input ----
    zin = _open_group(args.in_zarr, consolidated=args.consolidated)

    if "peat_mask" not in zin:
        raise KeyError(f"Input store does not contain 'peat_mask'. Arrays: {list(zin.array_keys())}")

    pm_in = zin["peat_mask"]
    if pm_in.ndim != 2:
        raise ValueError(f"Expected peat_mask to be 2D (H,W). Got shape={pm_in.shape} (ndim={pm_in.ndim})")

    # Axes only for bbox -> indices (NOT written to output)
    y_axis = _get_axis(zin, ["y", "lat", "latitude"])
    x_axis = _get_axis(zin, ["x", "lon", "longitude"])

    H, W = pm_in.shape
    if y_axis.ndim != 1 or x_axis.ndim != 1:
        raise ValueError(f"Expected 1D x/y axes. Got y.ndim={y_axis.ndim}, x.ndim={x_axis.ndim}")

    if len(y_axis) != H or len(x_axis) != W:
        print(
            f"[warn] Axis lengths do not match peat_mask shape: peat_mask={pm_in.shape}, "
            f"len(y)={len(y_axis)}, len(x)={len(x_axis)}. Proceeding, but coords may not align."
        )

    # ---- Compute slices ----
    y_slice, x_slices = _compute_slices_for_bbox(
        y_axis=y_axis,
        x_axis=x_axis,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
    )

    y0, y1 = y_slice.start, y_slice.stop
    print(f"[INFO] y slice: {y0}:{y1}  (lat approx [{y_axis[y0]:.6f}, {y_axis[y1-1]:.6f}])")
    for i, xs in enumerate(x_slices):
        x0, x1 = xs.start, xs.stop
        print(f"[INFO] x slice {i+1}/{len(x_slices)}: {x0}:{x1}  (lon approx [{x_axis[x0]:.6f}, {x_axis[x1-1]:.6f}])")

    # ---- Output dtype + NoData ----
    out_dtype = _choose_output_dtype(pm_in.dtype)
    if out_dtype != np.dtype(pm_in.dtype):
        print(f"[INFO] Promoting peat_mask dtype {pm_in.dtype} -> {out_dtype} so NoData=-9999 is representable.")

    compressor, chunks = _copy_compressor_and_chunks(pm_in)

    # ---- Create output group (Zarr v2) ----
    zout, _ = _make_output_group(args.out_zarr, overwrite=args.overwrite, zip_store=args.zip_store)

    # ---- Create ONLY ONE array: peat_mask (H,W), filled with -9999 everywhere ----
    create_kwargs = dict(
        name="peat_mask",
        shape=(H, W),
        dtype=out_dtype,
        overwrite=True,
        fill_value=np.array(NODATA_VALUE, dtype=out_dtype).item(),
    )
    if chunks is not None:
        create_kwargs["chunks"] = chunks
    if compressor is not None:
        create_kwargs["compressor"] = compressor  # ok for v2 output (warning is fine)

    pm_out = zout.create(**create_kwargs)

    # Copy attrs (optional) and stamp our metadata
    try:
        pm_out.attrs.update(dict(pm_in.attrs))
    except Exception:
        pass
    pm_out.attrs.update(
        {
            "nodata_value": int(NODATA_VALUE),
            "masking_behavior": "outside_bbox_set_to_-9999_no_shrink",
            "note": "Inside bbox preserves input peat_mask values (peat=1, non-peat=0), not a full-bbox fill.",
            "bbox_requested": {
                "lat_min": float(args.lat_min),
                "lat_max": float(args.lat_max),
                "lon_min": float(args.lon_min),
                "lon_max": float(args.lon_max),
            },
            "source_in_zarr": str(args.in_zarr),
        }
    )

    # ---- Write inside bbox: copy exact peat_mask values from input ----
    # This keeps peatlands ONLY where the input peat_mask says so.
    print("[INFO] Writing input peat_mask values inside bbox region(s)...")
    for xs in x_slices:
        block = np.asarray(pm_in[y_slice, xs])
        pm_out[y_slice, xs] = block.astype(out_dtype, copy=False)

    # ---- Sanity checks ----
    if pm_out.ndim != 2:
        raise RuntimeError(f"[bug] Output peat_mask is not 2D: ndim={pm_out.ndim}")
    if pm_out.shape != pm_in.shape:
        raise RuntimeError(f"[bug] Output shape changed unexpectedly: {pm_out.shape} vs {pm_in.shape}")

    # Quick spot-check: ensure we didn't accidentally fill bbox with 1s
    # (We expect some zeros inside bbox unless your bbox is entirely peatland.)
    try:
        sample = np.asarray(pm_out[y_slice, x_slices[0]])
        unique = np.unique(sample[:: max(1, sample.shape[0] // 128), :: max(1, sample.shape[1] // 128)])
        print(f"[INFO] Sample unique values inside bbox (downsampled): {unique[:20]}")
    except Exception:
        pass

    print(f"[DONE] Wrote peat_mask-only store: {args.out_zarr}")
    print(f"[DONE] peat_mask shape: {pm_out.shape}, dtype: {pm_out.dtype}, nodata: {NODATA_VALUE}")
    print("[NOTE] Output contains ONLY one array named 'peat_mask' (no x/y/time/band variables).")


if __name__ == "__main__":
    main()
