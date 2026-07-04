#!/usr/bin/env python3
"""
stack_pair_folders.py — Combine same-named .tif files from two folders
into single multi-band outputs (bands from folder1, then folder2).

Deps:
  pip/conda install rasterio tqdm

Example:
  python stack_pair_folders.py \
    --folder1 /path/to/folder_A \
    --folder2 /path/to/folder_B \
    --out /path/to/OUT \
    --compress LZW --tiled --overwrite
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

def list_tifs(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff")])

def ensure_same_grid(src1, src2, fname):
    problems = []
    if src1.width != src2.width or src1.height != src2.height:
        problems.append("width/height differ")
    if src1.crs != src2.crs:
        problems.append("CRS differ")
    if src1.transform != src2.transform:
        problems.append("affine transform differ")
    if src1.dtypes[0] != src2.dtypes[0]:
        problems.append(f"dtype differ ({src1.dtypes[0]} vs {src2.dtypes[0]})")
    if problems:
        raise ValueError(f"[{fname}] Incompatible rasters: " + ", ".join(problems))

def build_profile(base_ds, total_bands, args):
    profile = base_ds.profile.copy()
    profile.update(
        count=total_bands,
        compress=args.compress,
        tiled=args.tiled,
        BIGTIFF="YES" if args.bigtiff or (base_ds.width * base_ds.height * total_bands > 4_000_000_000 // rasterio.dtypes.dtype_ranges[base_ds.dtypes[0]][1]) else "IF_NEEDED",
    )
    if args.blockxsize and args.blockysize:
        profile.update(blockxsize=args.blockxsize, blockysize=args.blockysize, tiled=True)
    if args.nodata is not None:
        profile.update(nodata=args.nodata)
    return profile

def combined_descriptions(ds1, ds2):
    d1 = list(ds1.descriptions or [])
    d2 = list(ds2.descriptions or [])
    # Fill missing descriptions with generic names
    d1 = [d if (d and d.strip()) else f"f1_band{idx+1}" for idx, d in enumerate(d1 or [None]*ds1.count)]
    d2 = [d if (d and d.strip()) else f"f2_band{idx+1}" for idx, d in enumerate(d2 or [None]*ds2.count)]
    return tuple(d1 + d2)

def copy_band_tags(src, dst, dst_band_offset):
    # Copy per-band tags if present
    for b in range(1, src.count + 1):
        tags = src.tags(b)
        if tags:
            dst.update_tags(dst_band_offset + b, **tags)

def process_pair(path1: Path, path2: Path, out_path: Path, args):
    with rasterio.open(path1) as ds1, rasterio.open(path2) as ds2:
        ensure_same_grid(ds1, ds2, path1.name)

        total_bands = ds1.count + ds2.count
        profile = build_profile(ds1, total_bands, args)

        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            return  # skip

        with rasterio.open(out_path, "w", **profile) as dst:
            # Optional: set dataset-level tags (prefix to avoid collisions)
            dst.update_tags(**{f"src1_{k}": v for k, v in ds1.tags().items()})
            dst.update_tags(**{f"src2_{k}": v for k, v in ds2.tags().items()})

            # Write folder1 bands
            for i in range(1, ds1.count + 1):
                data = ds1.read(i)  # full band read; typically OK for 0.1° global
                dst.write(data, i)

            # Write folder2 bands
            for j in range(1, ds2.count + 1):
                data = ds2.read(j)
                dst.write(data, ds1.count + j)

            # Descriptions
            try:
                dst.descriptions = combined_descriptions(ds1, ds2)
            except Exception:
                pass

            # nodata: prefer explicit arg, else carry ds1’s if present
            if args.nodata is None and ds1.nodata is not None:
                dst.update_tags(**{"_note": "nodata inherited from folder1"})
            elif args.nodata is not None:
                dst.update_tags(**{"_note": f"nodata set by user to {args.nodata}"})

            # Per-band tags
            copy_band_tags(ds1, dst, 0)
            copy_band_tags(ds2, dst, ds1.count)

def main():
    ap = argparse.ArgumentParser(description="Stack same-named GeoTIFFs from two folders into combined multi-band outputs.")
    ap.add_argument("--folder1", required=True, type=Path, help="Path to first folder of .tif files")
    ap.add_argument("--folder2", required=True, type=Path, help="Path to second folder of .tif files")
    ap.add_argument("--out", required=True, type=Path, help="Output folder")
    ap.add_argument("--compress", default="LZW", help="GDAL compression (e.g., LZW, DEFLATE, ZSTD, NONE)")
    ap.add_argument("--tiled", action="store_true", help="Write tiled GeoTIFFs")
    ap.add_argument("--blockxsize", type=int, default=None, help="Tile width (requires --tiled)")
    ap.add_argument("--blockysize", type=int, default=None, help="Tile height (requires --tiled)")
    ap.add_argument("--bigtiff", action="store_true", help="Force BIGTIFF=YES")
    ap.add_argument("--nodata", type=float, default=None, help="Override nodata value for output")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    f1 = list_tifs(args.folder1)
    f2 = list_tifs(args.folder2)

    names1 = {p.name for p in f1}
    names2 = {p.name for p in f2}
    common = sorted(list(names1 & names2))

    missing_in_2 = sorted(list(names1 - names2))
    missing_in_1 = sorted(list(names2 - names1))
    if missing_in_2:
        print(f"WARNING: {len(missing_in_2)} files present in folder1 but missing in folder2 (e.g., {missing_in_2[:3]})")
    if missing_in_1:
        print(f"WARNING: {len(missing_in_1)} files present in folder2 but missing in folder1 (e.g., {missing_in_1[:3]})")

    map1 = {p.name: p for p in f1}
    map2 = {p.name: p for p in f2}

    args.out.mkdir(parents=True, exist_ok=True)

    for name in tqdm(common, desc="Combining"):
        out_path = args.out / name
        try:
            process_pair(map1[name], map2[name], out_path, args)
        except Exception as e:
            print(f"ERROR [{name}]: {e}")

    print(f"Done. Wrote {len(common)} files to: {args.out}")

if __name__ == "__main__":
    main()
