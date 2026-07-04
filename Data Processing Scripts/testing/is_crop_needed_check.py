#!/usr/bin/env python3
import argparse
import numpy as np
import zarr

def monotonic(a):
    da = np.diff(a)
    if np.all(da >= 0): return "increasing"
    if np.all(da <= 0): return "decreasing"
    return "non-monotonic"

def nearest_index(coord, v):
    coord = np.asarray(coord)
    return int(np.argmin(np.abs(coord - v)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-zarr", required=True)
    ap.add_argument("--var-name", default="era5land")
    ap.add_argument("--lat-min", type=float, default=None)
    ap.add_argument("--lat-max", type=float, default=None)
    ap.add_argument("--lon-min", type=float, default=None)
    ap.add_argument("--lon-max", type=float, default=None)
    args = ap.parse_args()

    zin = zarr.open_group(args.in_zarr, mode="r")

    if "y" not in zin or "x" not in zin:
        raise KeyError(f"Expected 'y' and 'x' arrays. Found arrays: {list(zin.array_keys())}")

    y = zin["y"][:]
    x = zin["x"][:]

    print("=== AXES ===")
    print(f"y: len={len(y)}  first/last=({y[0]:.6f}, {y[-1]:.6f})  min/max=({y.min():.6f}, {y.max():.6f})  {monotonic(y)}")
    print(f"x: len={len(x)}  first/last=({x[0]:.6f}, {x[-1]:.6f})  min/max=({x.min():.6f}, {x.max():.6f})  {monotonic(x)}")

    if args.var_name in zin:
        a = zin[args.var_name]
        print("\n=== MAIN VAR ===")
        print(f"{args.var_name}: shape={a.shape} dtype={a.dtype} chunks={getattr(a,'chunks',None)}")

        if a.ndim == 4:
            T, C, H, W = a.shape
            okH = (H == len(y))
            okW = (W == len(x))
            print(f"matches y/x? H==len(y): {okH}  W==len(x): {okW}")
        else:
            print("WARNING: main var is not 4D (expected T,C,H,W).")

    if "peat_mask" in zin:
        pm = zin["peat_mask"]
        print("\n=== peat_mask ===")
        print(f"peat_mask: shape={pm.shape} dtype={pm.dtype} chunks={getattr(pm,'chunks',None)}")
        if pm.ndim == 2:
            print(f"matches y/x? H==len(y): {pm.shape[0]==len(y)}  W==len(x): {pm.shape[1]==len(x)}")

    if args.lat_min is not None and args.lat_max is not None:
        lo, hi = min(args.lat_min, args.lat_max), max(args.lat_min, args.lat_max)
        imin = nearest_index(y, lo)
        imax = nearest_index(y, hi)
        print("\n=== BBOX CHECK (LAT) ===")
        print(f"requested lat range: [{lo}, {hi}]")
        print(f"nearest indices: {imin} (y={y[imin]:.6f}), {imax} (y={y[imax]:.6f})")
        print(f"covered by axis min/max? {y.min() <= lo <= y.max() and y.min() <= hi <= y.max()}")

    if args.lon_min is not None and args.lon_max is not None:
        lo, hi = min(args.lon_min, args.lon_max), max(args.lon_min, args.lon_max)
        imin = nearest_index(x, lo)
        imax = nearest_index(x, hi)
        print("\n=== BBOX CHECK (LON) ===")
        print(f"requested lon range: [{lo}, {hi}]")
        print(f"nearest indices: {imin} (x={x[imin]:.6f}), {imax} (x={x[imax]:.6f})")
        print(f"covered by axis min/max? {x.min() <= lo <= x.max() and x.min() <= hi <= x.max()}")
        print("NOTE: this assumes bbox does NOT cross the dateline and lon convention matches the file.")

if __name__ == "__main__":
    main()
