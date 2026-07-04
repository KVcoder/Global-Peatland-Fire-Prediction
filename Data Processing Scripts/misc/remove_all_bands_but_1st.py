#!/usr/bin/env python3
"""
Keep only the 1st band in every GeoTIFF in a folder.

Examples:
  python keep_band1.py --in_dir /path/to/tifs --out_dir /path/to/out
  python keep_band1.py --in_dir /path/to/tifs --inplace  # overwrites (via temp+replace)
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import rasterio


def process_one(src_path: Path, dst_path: Path, compress: str | None = "deflate") -> None:
    with rasterio.open(src_path) as src:
        if src.count < 1:
            raise RuntimeError("GeoTIFF has no bands?")

        band1 = src.read(1)  # (H, W)

        profile = src.profile.copy()
        profile.update(
            count=1,
        )

        # Preserve nodata for band 1 if present
        # (Rasterio stores nodata in profile["nodata"] for single nodata value)
        nodata = src.nodata
        if nodata is not None:
            profile["nodata"] = nodata

        # Optional compression + tiling (keep simple and compatible)
        if compress:
            profile["compress"] = compress
        # Some files may be internally tiled already; leaving block sizes as-is can fail
        # if they're incompatible for count=1 in some drivers. Safer: drop tiling keys.
        for k in ("tiled", "blockxsize", "blockysize", "interleave"):
            profile.pop(k, None)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(band1, 1)

        # Copy colormap (rare, but possible for categorical rasters)
        try:
            cmap = src.colormap(1)
            if cmap:
                with rasterio.open(dst_path, "r+") as dst:
                    dst.write_colormap(1, cmap)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing .tif/.tiff files (walks recursively)")
    ap.add_argument("--out_dir", default=None, help="Output folder (required unless --inplace)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite originals safely (temp file then replace)")
    ap.add_argument("--suffix", default="_band1", help="Suffix for output filenames (ignored with --inplace)")
    ap.add_argument("--compress", default="deflate", help="GeoTIFF compression (deflate, lzw, none)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        print(f"ERROR: in_dir does not exist: {in_dir}", file=sys.stderr)
        sys.exit(2)

    if not args.inplace and args.out_dir is None:
        print("ERROR: Provide --out_dir or use --inplace", file=sys.stderr)
        sys.exit(2)

    compress = None if args.compress.lower() in ("none", "off", "false", "0") else args.compress.lower()

    tif_paths = sorted(
        list(in_dir.rglob("*.tif")) + list(in_dir.rglob("*.tiff"))
    )

    if not tif_paths:
        print(f"No .tif/.tiff files found under {in_dir}")
        return

    out_dir = Path(args.out_dir) if args.out_dir else None

    n_ok, n_fail = 0, 0
    for src_path in tif_paths:
        try:
            if args.inplace:
                # Write to a temp file in same directory then replace
                tmp_fd, tmp_name = tempfile.mkstemp(suffix=".tif", prefix=src_path.stem + "_tmp_", dir=str(src_path.parent))
                os.close(tmp_fd)
                tmp_path = Path(tmp_name)

                try:
                    process_one(src_path, tmp_path, compress=compress)
                    tmp_path.replace(src_path)
                finally:
                    # If something went wrong before replace, clean up temp
                    if tmp_path.exists() and tmp_path != src_path:
                        try:
                            tmp_path.unlink()
                        except Exception:
                            pass
                dst_path = src_path
            else:
                rel = src_path.relative_to(in_dir)
                dst_name = src_path.stem + args.suffix + src_path.suffix
                dst_path = (out_dir / rel.parent / dst_name)
                process_one(src_path, dst_path, compress=compress)

            n_ok += 1
            print(f"[OK] {src_path} -> {dst_path}")
        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {src_path}: {e}", file=sys.stderr)

    print(f"\nDone. OK={n_ok}  FAIL={n_fail}")


if __name__ == "__main__":
    main()
