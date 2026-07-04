#!/usr/bin/env python3
"""
crop_zarr_by_xy.py

Crop a global Zarr store (with 1-D x/y coordinate axes) to a
lat/lon bounding box, writing a new Zarr v2 store with the same
compressor and metadata where possible.

Assumes input Zarr (v2-style) layout something like:

  *.zarr/
    .zattrs
    .zgroup
    .zmetadata
    <main_var>/   # (T, C, H, W), e.g. "era5land" or "smap_wtd" or "field"
    peat_mask/    # (H, W)         [optional]
    time/         # (T,)           [optional]
    band/         # (C,)           [optional]
    x/            # (W,)
    y/            # (H,)
    spatial_ref   # scalar or small array [optional]

Output will mirror this structure but spatially cropped.
"""

import argparse
import numpy as np
import zarr
from tqdm.auto import tqdm


def find_index_slice(coord_1d, vmin, vmax, name="coord"):
    """
    coord_1d: 1-D array (e.g. y or x)
    vmin, vmax: desired value range
    Returns a Python slice(start, stop) that covers all points in [vmin, vmax].
    """
    coord = np.asarray(coord_1d)
    mask = (coord >= vmin) & (coord <= vmax)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(
            f"No {name} values found in range [{vmin}, {vmax}]. "
            f"Actual min/max: [{coord.min()}, {coord.max()}]"
        )
    start = int(idx.min())
    stop = int(idx.max()) + 1  # slice stop is exclusive
    return slice(start, stop)


def create_like(in_arr, out_group, name, new_shape, new_chunks=None):
    """
    Create an output Zarr array in out_group with metadata copied from in_arr.

    new_shape: desired shape
    new_chunks: optional new chunking tuple; if None, infer from in_arr.chunks.
    """
    compressor = getattr(in_arr, "compressor", None)
    dtype = in_arr.dtype
    fill_value = getattr(in_arr, "fill_value", None)
    chunks = getattr(in_arr, "chunks", None)

    if new_chunks is not None:
        chunks_out = new_chunks
    elif chunks is not None:
        # shrink chunks if they exceed new_shape along any axis
        chunks_out = tuple(
            min(c, s) if (c is not None) else None
            for c, s in zip(chunks, new_shape)
        )
    else:
        chunks_out = None

    out_arr = out_group.create(
        name,
        shape=new_shape,
        chunks=chunks_out,
        dtype=dtype,
        compressor=compressor,  # reuse original compressor (e.g. Blosc lz4 clevel=1)
        fill_value=fill_value,
        overwrite=True,
    )
    # copy attrs
    try:
        out_arr.attrs.update(dict(in_arr.attrs))
    except Exception:
        pass
    return out_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-zarr",  required=True,
                    help="Path to input Zarr store (v2)")
    ap.add_argument("--out-zarr", required=True,
                    help="Path to output cropped Zarr store (v2)")
    ap.add_argument("--lat-min",  type=float, required=True)
    ap.add_argument("--lat-max",  type=float, required=True)
    ap.add_argument("--lon-min",  type=float, required=True)
    ap.add_argument("--lon-max",  type=float, required=True)
    ap.add_argument(
        "--var-name",
        default="era5land",
        help="Name of main 4D data variable (e.g. 'era5land', 'smap_wtd', 'field')",
    )
    args = ap.parse_args()

    main_var = args.var_name

    # Open input root group (force Zarr v2 so numcodecs.Blosc works)
    zin = zarr.open_group(args.in_zarr, mode="r", zarr_version=2)

    # ---- 1) Get 1-D y (lat) and x (lon) axes ----
    if "y" not in zin or "x" not in zin:
        raise KeyError(
            f"Expected 'y' and 'x' arrays in {args.in_zarr}, "
            f"found: {list(zin.array_keys())}"
        )

    y_axis = zin["y"][:]   # latitudes
    x_axis = zin["x"][:]   # longitudes

    # Compute index slices
    y_slice = find_index_slice(y_axis, args.lat_min, args.lat_max, name="lat(y)")
    x_slice = find_index_slice(x_axis, args.lon_min, args.lon_max, name="lon(x)")

    y0, y1 = y_slice.start, y_slice.stop
    x0, x1 = x_slice.start, x_slice.stop

    print(
        f"[INFO] y index slice: {y0}:{y1} "
        f"(lat approx [{y_axis[y0]:.3f}, {y_axis[y1-1]:.3f}])"
    )
    print(
        f"[INFO] x index slice: {x0}:{x1} "
        f"(lon approx [{x_axis[x0]:.3f}, {x_axis[x1-1]:.3f}])"
    )

    # ---- 2) Prepare output group (also Zarr v2) ----
    zout = zarr.open_group(args.out_zarr, mode="w", zarr_version=2)

    # copy root attrs if any
    try:
        zout.attrs.update(dict(zin.attrs))
    except Exception:
        pass

    # ---- 3) Crop coordinate / static arrays ----

    # y (lat axis)
    y_in = zin["y"]
    y_out = create_like(y_in, zout, "y", (y1 - y0,))
    y_out[...] = y_in[y_slice]

    # x (lon axis)
    x_in = zin["x"]
    x_out = create_like(x_in, zout, "x", (x1 - x0,))
    x_out[...] = x_in[x_slice]

    # time axis (copied fully)
    if "time" in zin:
        t_in = zin["time"]
        t_out = create_like(t_in, zout, "time", t_in.shape)
        t_out[...] = t_in[...]

    # band axis (copied fully)
    if "band" in zin:
        b_in = zin["band"]
        b_out = create_like(b_in, zout, "band", b_in.shape)
        b_out[...] = b_in[...]

    # spatial_ref (usually scalar), copy as-is
    if "spatial_ref" in zin:
        s_in = zin["spatial_ref"]
        s_out = create_like(s_in, zout, "spatial_ref", s_in.shape)
        s_out[...] = s_in[...]

    # peat_mask (H, W) -> crop in y,x
    if "peat_mask" in zin:
        pm_in = zin["peat_mask"]
        if pm_in.ndim != 2:
            raise ValueError(
                f"Expected 'peat_mask' to have shape (H, W), got {pm_in.shape}"
            )
        pm_out = create_like(pm_in, zout, "peat_mask", (y1 - y0, x1 - x0))
        print("[INFO] Copying peat_mask...")
        pm_out[:, :] = pm_in[y_slice, x_slice]

    # ---- 4) Crop main data array with tqdm over time ----
    if main_var not in zin:
        raise KeyError(
            f"'{main_var}' array not found in {args.in_zarr}. "
            f"Available arrays: {list(zin.array_keys())}"
        )

    data_in = zin[main_var]
    if data_in.ndim != 4:
        raise ValueError(
            f"Expected '{main_var}' shape (T, C, H, W), got {data_in.shape}"
        )
    T, C, H, W = data_in.shape
    H_out = y1 - y0
    W_out = x1 - x0

    data_out = create_like(
        data_in,
        zout,
        main_var,
        (T, C, H_out, W_out),
        # optional: keep same chunking pattern but cropped spatial dimensions
        new_chunks=None,
    )

    print(
        f"[INFO] Copying '{main_var}' with crop "
        f"(T={T}, C={C}, H={H_out}, W={W_out})..."
    )
    for t in tqdm(range(T), desc=f"Copying {main_var}[time]"):
        slab = data_in[t, :, y_slice, x_slice]  # (C, H_out, W_out)
        data_out[t, :, :, :] = slab

    print(f"[DONE] Cropped store written to: {args.out_zarr}")


if __name__ == "__main__":
    main()
