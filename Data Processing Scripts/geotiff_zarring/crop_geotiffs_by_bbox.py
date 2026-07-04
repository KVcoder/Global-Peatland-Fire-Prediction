#!/usr/bin/env python3
"""
crop_geotiffs_by_bbox.py

Crop GeoTIFF(s) to a lat/lon bounding box (default bbox CRS: EPSG:4326),
writing cropped GeoTIFFs while preserving metadata/options where possible.

Examples:
  python crop_geotiffs_by_bbox.py \
    --in-dir /data/tifs --out-dir /data/tifs_cropped \
    --lat-min -10 --lat-max 10 --lon-min 95 --lon-max 125 \
    --pattern "*.tif" --recursive --workers 8

Notes:
- Crops only (no reprojection). Bbox is transformed into each raster CRS.
- If bbox doesn't overlap a raster, that file is skipped.
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from tqdm.auto import tqdm


def _maybe_wrap_bbox_lon_to_360(lon_min, lon_max, src_bounds):
    """
    Heuristic: If raster bounds look like 0..360 and bbox has negative lon,
    shift bbox longitudes into 0..360.
    """
    left, bottom, right, top = src_bounds
    raster_looks_0360 = (left >= 0) and (right > 180)
    bbox_has_negative = (lon_min < 0) or (lon_max < 0)
    if raster_looks_0360 and bbox_has_negative:
        lon_min = lon_min % 360
        lon_max = lon_max % 360
    return lon_min, lon_max


def crop_one(
    in_path: str,
    out_path: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    bbox_crs: str = "EPSG:4326",
):
    in_path = str(in_path)
    out_path = str(out_path)

    with rasterio.Env():
        with rasterio.open(in_path) as src:
            if src.crs is None:
                raise ValueError(f"{in_path}: Missing CRS; cannot interpret lat/lon bbox.")

            # Ensure proper ordering
            bottom = min(lat_min, lat_max)
            top = max(lat_min, lat_max)
            left = min(lon_min, lon_max)
            right = max(lon_min, lon_max)

            # Handle common "0..360 lon" rasters if bbox is -180..180
            left, right = _maybe_wrap_bbox_lon_to_360(left, right, src.bounds)

            # Transform bbox (in bbox_crs) -> src.crs
            # densify_pts helps with rotated/projection distortion
            tb = transform_bounds(
                bbox_crs, src.crs, left, bottom, right, top,
                densify_pts=21
            )
            left_s, bottom_s, right_s, top_s = tb

            # Compute crop window in pixel coords
            win = from_bounds(left_s, bottom_s, right_s, top_s, transform=src.transform)

            # Intersect with raster extent to avoid boundless reads
            full = rasterio.windows.Window(col_off=0, row_off=0, width=src.width, height=src.height)
            win = win.intersection(full)

            if win.width <= 0 or win.height <= 0:
                return {"in": in_path, "out": out_path, "status": "skipped_no_overlap"}

            # Read all bands within window
            data = src.read(window=win)  # (bands, h, w)

            # Update profile for output
            profile = src.profile.copy()
            profile.update(
                height=int(win.height),
                width=int(win.width),
                transform=rasterio.windows.transform(win, src.transform),
                driver="GTiff",
            )
            # Safer for large outputs
            profile.setdefault("BIGTIFF", "IF_SAFER")

            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)

                # Copy dataset + band tags
                try:
                    dst.update_tags(**src.tags())
                except Exception:
                    pass
                for b in range(1, src.count + 1):
                    try:
                        dst.update_tags(b, **src.tags(b))
                    except Exception:
                        pass

            return {"in": in_path, "out": out_path, "status": "ok"}


def iter_files(in_dir: Path, pattern: str, recursive: bool):
    if recursive:
        yield from in_dir.rglob(pattern)
    else:
        yield from in_dir.glob(pattern)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Folder containing GeoTIFFs")
    ap.add_argument("--out-dir", required=True, help="Output folder for cropped GeoTIFFs")
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--bbox-crs", default="EPSG:4326", help="CRS of the input bbox (default EPSG:4326)")
    ap.add_argument("--pattern", default="*.tif", help="Glob pattern (default: *.tif)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--workers", type=int, default=1, help="Number of processes")
    ap.add_argument("--keep-rel-path", action="store_true",
                    help="Mirror input subfolder structure under out-dir")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    files = [p for p in iter_files(in_dir, args.pattern, args.recursive) if p.is_file()]
    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {in_dir} (recursive={args.recursive})")

    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for p in files:
            if args.keep_rel_path:
                rel = p.relative_to(in_dir)
                out_path = out_dir / rel
            else:
                out_path = out_dir / p.name

            futures.append(ex.submit(
                crop_one,
                str(p), str(out_path),
                args.lat_min, args.lat_max, args.lon_min, args.lon_max,
                args.bbox_crs
            ))

        ok = skipped = failed = 0
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Cropping GeoTIFFs"):
            try:
                res = fut.result()
                if res["status"] == "ok":
                    ok += 1
                elif res["status"].startswith("skipped"):
                    skipped += 1
            except Exception as e:
                failed += 1
                # Keep it short but useful
                print(f"[ERROR] {e}")

    print(f"[DONE] ok={ok} skipped={skipped} failed={failed} out_dir={out_dir}")


if __name__ == "__main__":
    main()
