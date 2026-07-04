#!/usr/bin/env python3
import argparse
import numpy as np
import zarr

def find_index_slice(coord_1d, vmin, vmax):
    coord = np.asarray(coord_1d)
    lo, hi = (vmin, vmax) if vmin <= vmax else (vmax, vmin)
    m = (coord >= lo) & (coord <= hi)
    idx = np.where(m)[0]
    if idx.size == 0:
        raise ValueError(f"No coord values in [{lo}, {hi}] (coord min/max: {coord.min()}, {coord.max()})")
    return slice(int(idx.min()), int(idx.max()) + 1)

def bbox_from_mask(mask2d):
    """Return (r0,r1,c0,c1) bounding box of True pixels; r1/c1 are exclusive."""
    rows = np.where(mask2d.any(axis=1))[0]
    cols = np.where(mask2d.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    return r0, r1, c0, c1

def finite_mask(arr, fill_value=None):
    m = np.isfinite(arr)
    if fill_value is not None and np.isfinite(fill_value):
        m &= (arr != fill_value)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-zarr", required=True)
    ap.add_argument("--var-name", default="field")
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--sample-times", default="0,mid,last",
                    help="Which timesteps to sample for data validity: e.g. '0,100,200' or '0,mid,last'")
    args = ap.parse_args()

    z = zarr.open_group(args.in_zarr, mode="r")
    y = z["y"][:]
    x = z["x"][:]
    ys = find_index_slice(y, args.lat_min, args.lat_max)
    xs = find_index_slice(x, args.lon_min, args.lon_max)

    print("=== AXIS EXTENTS ===")
    print(f"y min/max: {y.min():.6f}, {y.max():.6f}  (len={len(y)})")
    print(f"x min/max: {x.min():.6f}, {x.max():.6f}  (len={len(x)})")
    print(f"Requested bbox slices: y[{ys.start}:{ys.stop}] x[{xs.start}:{xs.stop}]")
    print(f"Requested bbox approx: lat [{y[ys.start]:.6f}, {y[ys.stop-1]:.6f}]  lon [{x[xs.start]:.6f}, {x[xs.stop-1]:.6f}]")

    # --- peat_mask check (fast + definitive if peat_mask encodes domain) ---
    if "peat_mask" in z:
        pm = z["peat_mask"][:]
        pm_valid = (pm != 0)
        inside = pm_valid[ys, xs].sum()
        total = pm_valid.sum()
        outside = total - inside

        print("\n=== PEAT_MASK DOMAIN CHECK ===")
        print(f"peat_mask nonzero total:  {int(total):,}")
        print(f"peat_mask nonzero inside: {int(inside):,}")
        print(f"peat_mask nonzero outside:{int(outside):,}")

        bb = bbox_from_mask(pm_valid)
        if bb is None:
            print("peat_mask has no nonzero pixels.")
        else:
            r0, r1, c0, c1 = bb
            print("peat_mask nonzero bbox (indices): "
                  f"rows {r0}:{r1}, cols {c0}:{c1}")
            print("peat_mask nonzero bbox (coords): "
                  f"lat [{y[r1-1]:.6f}, {y[r0]:.6f}]  lon [{x[c0]:.6f}, {x[c1-1]:.6f}]")

    # --- data validity check (sample a few timesteps so it's not huge) ---
    if args.var_name not in z:
        raise KeyError(f"{args.var_name} not found. Arrays: {list(z.array_keys())}")

    a = z[args.var_name]
    if a.ndim != 4:
        raise ValueError(f"Expected 4D (T,C,H,W). Got {a.shape}")

    T = a.shape[0]
    fillv = getattr(a, "fill_value", None)

    def parse_times(spec):
        out = []
        for token in spec.split(","):
            token = token.strip().lower()
            if token == "mid":
                out.append(T // 2)
            elif token == "last":
                out.append(T - 1)
            elif token != "":
                out.append(int(token))
        # unique, in range
        out = sorted(set([t for t in out if 0 <= t < T]))
        return out

    times = parse_times(args.sample_times)

    print("\n=== DATA VALIDITY CHECK (sample times) ===")
    print(f"Sampling t indices: {times} (T={T})  fill_value={fillv}")

    for t in times:
        slab = a[t, 0, :, :]  # assuming C=1; if more, this still checks the first channel
        # Read two parts to keep memory predictable
        inside_arr = slab[ys, xs]
        inside_valid = finite_mask(inside_arr, fillv)
        inside_count = int(inside_valid.sum())

        # Outside count computed as total valid minus inside valid
        # (Compute total valid for this timestep)
        all_arr = slab[:, :]
        all_valid = finite_mask(all_arr, fillv)
        total_count = int(all_valid.sum())
        outside_count = total_count - inside_count

        # Some quick stats inside if any valid
        if inside_count > 0:
            vals = inside_arr[inside_valid]
            vmin, vmax = float(vals.min()), float(vals.max())
        else:
            vmin, vmax = float("nan"), float("nan")

        print(f"t={t:4d}: valid total={total_count:,}  inside={inside_count:,}  outside={outside_count:,}  inside[min,max]=[{vmin},{vmax}]")

if __name__ == "__main__":
    main()
