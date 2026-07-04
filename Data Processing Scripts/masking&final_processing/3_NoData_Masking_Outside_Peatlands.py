#!/usr/bin/env python3
"""
mask_outside_to_nodata.py

For each ERA5-Land *daily* GeoTIFF:
  - Resample the peat mask (nearest neighbor) to the file's grid.
  - Set all pixels OUTSIDE the (resampled) mask to NoData.
  - Preserve data inside the mask as-is.
  - Write to an output folder, keeping band descriptions and metadata.

Works block-by-block to keep memory usage low.

Usage examples
--------------
# Folder of daily files -> masked copies in OUT_DIR
python mask_outside_to_nodata.py \
  --src-dir WEATHER_MERGE \
  --mask SMAP_L4_peat_mask_latlon_0p1deg_v8-1.tif \
  --out-dir WEATHER_MERGE_MASKED

# Glob pattern, force overwrite and a specific NoData value for floats
python mask_outside_to_nodata.py \
  --src-glob "WEATHER_MERGE/era5land_daily_*.tif" \
  --mask SMAP_L4_peat_mask_latlon_0p1deg_v8-1.tif \
  --out-dir WEATHER_MERGE_MASKED \
  --overwrite \
  --nodata -9999.0
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window


def list_sources(src_dir: Optional[str], src_glob: Optional[str]) -> List[Path]:
    if src_glob:
        return sorted(Path().glob(src_glob))
    if src_dir:
        return sorted(Path(src_dir).rglob("*.tif"))
    raise SystemExit("[ERR] Provide --src-dir or --src-glob")


def choose_nodata(dtype: str, existing: Optional[float], user_val: Optional[float]) -> float:
    """
    Prefer: user_val > existing > sensible default by dtype.
    For unsigned integers, pick the max value of the type.
    For floats, default -9999.0 (explicit nodata tag).
    """
    if user_val is not None:
        return user_val

    if existing is not None:
        return existing

    kind = np.dtype(dtype).kind
    if kind == "f":  # float
        return -9999.0
    if kind in ("i", "u"):
        dt = np.dtype(dtype)
        if kind == "u":
            return float(np.iinfo(dt).max)  # e.g., 255, 65535, 4294967295
        else:
            # signed int: use a large negative
            mn = np.iinfo(dt).min
            return float(mn if mn < -9999 else -9999)
    # fallback
    return -9999.0


def resample_mask_nn(mask_path: Path, ref_ds) -> np.ndarray:
    """Nearest-neighbor resample mask to ref_ds → boolean peat grid (True=peat)."""
    with rasterio.open(mask_path) as mds:
        src = mds.read(1, masked=False)
        dst = np.zeros((ref_ds.height, ref_ds.width), dtype=np.float32)
        reproject(
            source=src,
            destination=dst,
            src_transform=mds.transform,
            src_crs=mds.crs,
            src_nodata=mds.nodata,
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            dst_nodata=0.0,
            resampling=Resampling.nearest,
        )
    peat_bool = np.isfinite(dst) & (dst > 0)
    return peat_bool


def process_one(src_path: Path, mask_path: Path, out_dir: Path, overwrite: bool,
                user_nodata: Optional[float], report: bool):
    out_path = out_dir / src_path.name
    if out_path.exists() and not overwrite:
        if report:
            print(f"[SKIP] {out_path} exists (use --overwrite to replace)")
        return

    with rasterio.open(src_path) as src:
        peat_bool = resample_mask_nn(mask_path, src)
        nodata_val = choose_nodata(src.dtypes[0], src.nodata, user_nodata)

        profile = src.profile.copy()
        profile.update(nodata=nodata_val)

        # Keep compression/tiled settings if present
        # profile already carries them from src

        with rasterio.open(out_path, "w", **profile) as dst:
            # Copy per-band descriptions (labels)
            try:
                dst.descriptions = src.descriptions
            except Exception:
                pass

            # Stream by blocks/windows to avoid big memory use
            # Use band 1's tiling as iterator
            for ji, window in src.block_windows(1):
                r0, c0 = int(window.row_off), int(window.col_off)
                r1 = r0 + int(window.height)
                c1 = c0 + int(window.width)

                # Windowed mask slice (True=peat; False=outside)
                peat_w = peat_bool[r0:r1, c0:c1]
                outside_w = ~peat_w

                for b in range(1, src.count + 1):
                    arr = src.read(b, window=window, masked=False)

                    # Set outside to nodata; preserve inside untouched
                    if np.issubdtype(arr.dtype, np.floating):
                        arr = arr.astype(np.float32, copy=False)
                        # Write the exact nodata literal (not NaN) so the nodata tag applies
                        arr[outside_w] = nodata_val
                    else:
                        # Ensure we can represent nodata in this dtype
                        dt = arr.dtype
                        as_int = int(nodata_val)
                        iinfo = np.iinfo(dt) if dt.kind in ("i", "u") else None
                        if iinfo and (as_int < iinfo.min or as_int > iinfo.max):
                            raise SystemExit(
                                f"[ERR] nodata={as_int} not representable in {dt} for {src_path.name}. "
                                f"Provide a suitable --nodata for integer type."
                            )
                        arr[outside_w] = as_int

                    dst.write(arr, indexes=b, window=window)

            # Copy global tags (nice-to-have)
            try:
                dst.update_tags(**src.tags())
            except Exception:
                pass

    if report:
        print(f"[OK] Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Clamp values OUTSIDE a peat mask to NoData for ERA5 daily GeoTIFFs.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--src-dir", help="Folder containing era5land_daily_*.tif")
    g.add_argument("--src-glob", help="Glob pattern, e.g., 'WEATHER_MERGE/era5land_daily_*.tif'")
    ap.add_argument("--mask", required=True, help="Peat mask .tif (resampled NN to each src)")
    ap.add_argument("--out-dir", required=True, help="Output folder (will be created if missing)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--nodata", type=float, default=None,
                    help="Force this NoData value (otherwise use src nodata or a safe default by dtype)")
    ap.add_argument("--quiet", action="store_true", help="Less console output")
    args = ap.parse_args()

    srcs = list_sources(args.src_dir, args.src_glob)
    if not srcs:
        raise SystemExit("[ERR] No input .tif files found.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = Path(args.mask)
    if not mask_path.exists():
        raise SystemExit(f"[ERR] Mask not found: {mask_path}")

    for i, p in enumerate(srcs, 1):
        if not args.quiet:
            print(f"[{i}/{len(srcs)}] {p.name}")
        process_one(p, mask_path, out_dir, args.overwrite, args.nodata, report=not args.quiet)


if __name__ == "__main__":
    main()
