#!/usr/bin/env python3
"""
unscale_folder_fast.py — divide all bands by a factor (default 100) and write Float32 GeoTIFFs.

Speed-ups vs. baseline:
- Windowed (block) I/O to avoid loading whole rasters in RAM (faster with compressed GTiffs).
- Vectorized in-place divide using numpy.where mask semantics.
- Optional multi-process over files (--jobs).
- GDAL internal threading control (--threads) to avoid over-subscription.
- Skips existing outputs if desired.

Usage:
  python unscale_folder_fast.py --src ERA5LAND_FINAL --dst ERA5LAND_FINAL_UNSCALED
  python unscale_folder_fast.py --src ERA5LAND_FINAL --dst OUT --factor 100 --jobs 4 --threads 1 --skip-existing
  python unscale_folder_fast.py --src IN --dst OUT --compress ZSTD --predictor 3  # (optional override)

Notes:
- By default, keeps original compression/tiling if present. Use --compress to override.
- BIGTIFF is set to IF_SAFER automatically.
"""

import argparse
from pathlib import Path
import os
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Folder of .tif files")
    p.add_argument("--dst", required=True, help="Output folder")
    p.add_argument("--factor", type=float, default=100.0, help="Divide raw values by this factor")
    p.add_argument("--nodata", type=float, default=-9999.0, help="NoData to preserve if dataset has none")
    p.add_argument("--jobs", type=int, default=1, help="Number of parallel worker processes over files")
    p.add_argument("--threads", default=None,
                   help="GDAL_NUM_THREADS (e.g., 1, 2, 4, or ALL_CPUS). Default: ALL_CPUS if --jobs=1 else 1")
    p.add_argument("--skip-existing", action="store_true", help="Skip if output file already exists")
    p.add_argument("--compress", default=None, help="Override output compression (e.g., ZSTD, DEFLATE, LZW)")
    p.add_argument("--predictor", type=int, default=None, choices=[1,2,3],
                   help="GTiff PREDICTOR (2 = horizontal differencing for ints, 3 = for floats).")
    return p.parse_args()

def _count_windows(ds):
    # How many block windows? (for progress bars)
    return sum(1 for _ in ds.block_windows(1))

def _process_one(tif_path, dst_dir, factor, default_nodata, threads, compress, predictor, skip_existing, show_inner_bar):
    tif = Path(tif_path)
    out = Path(dst_dir) / tif.name
    if skip_existing and out.exists() and out.stat().st_size > 0:
        return str(out)

    # Avoid GDAL oversubscription: default threads depends on --jobs
    if threads is None:
        gdal_threads = "ALL_CPUS"
    else:
        gdal_threads = str(threads)

    with rasterio.Env(GDAL_NUM_THREADS=gdal_threads):
        with rasterio.open(tif) as ds:
            # Build destination profile
            src_prof = ds.profile.copy()
            nodata = float(ds.nodata) if ds.nodata is not None else float(default_nodata)
            dst_prof = src_prof.copy()
            dst_prof.update(dtype="float32", nodata=nodata, BIGTIFF="IF_SAFER")
            if compress:
                dst_prof.update(compress=compress)
                if predictor is not None:
                    dst_prof.update(predictor=predictor)

            # Create destination
            with rasterio.open(out, "w", **dst_prof) as dst:
                # Copy tags & band tags/descriptions
                if ds.tags():
                    dst.update_tags(**ds.tags())
                for i, desc in enumerate(ds.descriptions or (), start=1):
                    if desc:
                        dst.set_band_description(i, desc)
                    tags_i = ds.tags(i)
                    if tags_i:
                        dst.update_tags(i, **tags_i)

                # Iterate over the dataset's native blocks (windows)
                total_wins = _count_windows(ds) if show_inner_bar else None
                win_iter = ds.block_windows(1)
                if show_inner_bar:
                    win_iter = tqdm(win_iter, total=total_wins, leave=False, desc=tif.name)

                is_nan_nodata = math.isnan(nodata)

                for _, window in win_iter:
                    # Read all bands in this window in source dtype (fastest), then cast to float32 once.
                    data = ds.read(window=window)  # (bands, rows, cols), src dtype
                    out_block = data.astype("float32", copy=False)

                    # Build mask for nodata (using source dtype comparison to be exact)
                    if is_nan_nodata:
                        # Only applies if source already float; otherwise no NaNs to preserve
                        mask = np.isnan(out_block)
                    else:
                        mask = (data == nodata)

                    # Divide in-place where valid
                    # where=~mask: leaves nodata values unchanged
                    np.divide(out_block, float(factor), out=out_block, where=~mask)

                    # Ensure nodata is exactly nodata (in case factor==0 or any shenanigans)
                    if is_nan_nodata:
                        # nothing to set; NaNs already in mask positions
                        pass
                    else:
                        out_block[mask] = nodata

                    dst.write(out_block, window=window)

                # Done writing blocks
    return str(out)

def main():
    args = parse_args()
    src = Path(args.src); dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.glob("*.tif")])
    if not files:
        print("No .tif files found in --src.", flush=True)
        return

    # Default threading behavior
    threads = args.threads
    if threads is None:
        threads = "ALL_CPUS" if args.jobs == 1 else "1"

    if args.jobs > 1:
        # Parallel over files; use a single top-level bar
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {
                ex.submit(
                    _process_one,
                    str(tif),
                    str(dst),
                    args.factor,
                    args.nodata,
                    str(threads),
                    args.compress,
                    args.predictor,
                    args.skip_existing,
                    False,  # don't render inner bars in parallel (too noisy)
                ): tif
                for tif in files
            }
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                pass
    else:
        # Serial with per-file inner window bars for nice feedback
        for tif in tqdm(files, desc="Processing files"):
            _process_one(
                str(tif),
                str(dst),
                args.factor,
                args.nodata,
                str(threads),
                args.compress,
                args.predictor,
                args.skip_existing,
                True,  # show inner bars
            )

if __name__ == "__main__":
    main()
