#!/usr/bin/env python3
"""
Check GeoTIFF format consistency for three folders:
  - era5land
  - smap_wtd
  - viirs

By default, reads only 1 file per folder.
Optionally, scan all files in each folder and confirm they share the same format.

"Format" here means:
  - CRS
  - transform (georeferencing)
  - resolution (pixel size)
  - width, height
  - number of bands
  - dtype
  - nodata

Additionally, this script compares the canonical formats BETWEEN the three folders
to check whether they share the same grid/spec.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import rasterio


def affine_to_tuple(transform) -> Tuple[float, ...]:
    """Convert rasterio.Affine to a plain tuple for comparisons/printing."""
    return tuple(transform)


def summarize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the pieces of metadata we care about for format comparison."""
    transform = meta.get("transform")
    if transform is not None:
        transform_tuple = affine_to_tuple(transform)
        res = (transform.a, transform.e)  # (pixel_width, pixel_height)
    else:
        transform_tuple = None
        res = None

    crs = meta.get("crs")
    if crs is not None:
        crs_str = crs.to_string()
    else:
        crs_str = None

    return {
        "width": meta.get("width"),
        "height": meta.get("height"),
        "count": meta.get("count"),      # number of bands
        "dtype": meta.get("dtype"),
        "crs": crs_str,
        "transform": transform_tuple,
        "res": res,                      # (pixel_width, pixel_height)
        "nodata": meta.get("nodata"),
    }


def formats_equal(a: Dict[str, Any], b: Dict[str, Any], tol: float = 1e-9) -> bool:
    """
    Compare two format summaries with a small tolerance for transform/res floats.
    """
    if a is None or b is None:
        return False

    keys = ["width", "height", "count", "dtype", "crs", "nodata"]
    for k in keys:
        if a.get(k) != b.get(k):
            return False

    # Compare transform and res with tolerance
    for key in ["transform", "res"]:
        va = a.get(key)
        vb = b.get(key)
        if va is None and vb is None:
            continue
        if (va is None) != (vb is None):
            return False
        if len(va) != len(vb):
            return False
        for x, y in zip(va, vb):
            if x is None and y is None:
                continue
            if x is None or y is None:
                return False
            if abs(x - y) > tol:
                return False

    return True


def find_geotiffs(folder: Path) -> List[Path]:
    """Return a sorted list of GeoTIFF files in a folder."""
    return sorted(
        list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    )


def check_folder(folder: Path, full_scan: bool = False) -> Tuple[Optional[Dict[str, Any]], List[Tuple[Path, Dict[str, Any]]]]:
    """
    Check one folder:
      - Read one file (or all, if full_scan)
      - Print the canonical format
      - If full_scan=True, verify that all files share that format

    Returns:
      (canonical_format_dict or None, mismatches_list)
    """
    print(f"\n=== Folder: {folder} ===")

    if not folder.exists() or not folder.is_dir():
        print("  ! Folder does not exist or is not a directory.")
        return None, []

    tiffs = find_geotiffs(folder)
    if not tiffs:
        print("  ! No GeoTIFF files (.tif/.tiff) found.")
        return None, []

    if not full_scan:
        tiffs_to_check = [tiffs[0]]
        print(f"  Checking only 1 file (set --full-scan to check all).")
    else:
        tiffs_to_check = tiffs
        print(f"  Checking all {len(tiffs_to_check)} files.")

    canonical_format: Optional[Dict[str, Any]] = None
    mismatches: List[Tuple[Path, Dict[str, Any]]] = []

    for i, tif_path in enumerate(tiffs_to_check, start=1):
        with rasterio.open(tif_path) as src:
            meta = src.meta.copy()
        fmt = summarize_meta(meta)

        if canonical_format is None:
            canonical_format = fmt
            print(f"  Reference file: {tif_path.name}")
        else:
            if not formats_equal(canonical_format, fmt):
                mismatches.append((tif_path, fmt))

        if not full_scan and i == 1:
            break

    # Print canonical format
    if canonical_format is None:
        print("  ! Could not read any metadata.")
        return None, mismatches

    print("\n  Canonical format for this folder:")
    print(f"    CRS:        {canonical_format['crs']}")
    print(f"    Size:       {canonical_format['width']} x {canonical_format['height']}")
    print(f"    Bands:      {canonical_format['count']}")
    print(f"    Dtype:      {canonical_format['dtype']}")
    if canonical_format["transform"] is not None:
        tx = canonical_format["transform"]
        print(f"    Transform:  {tx}")
    if canonical_format["res"] is not None:
        print(f"    Resolution: {canonical_format['res'][0]} x {canonical_format['res'][1]} (units of CRS)")
    print(f"    Nodata:     {canonical_format['nodata']}")

    # If full_scan, report consistency
    if full_scan:
        if not mismatches:
            print("\n  ✓ All checked files match the canonical format.")
        else:
            print(f"\n  ! Found {len(mismatches)} file(s) with mismatched format:")
            for path, fmt in mismatches:
                print(f"    - {path.name}")
                print(f"      CRS:   {fmt['crs']}")
                print(f"      Size:  {fmt['width']} x {fmt['height']}")
                print(f"      Bands: {fmt['count']}, Dtype: {fmt['dtype']}")
                if fmt["transform"] is not None:
                    print(f"      Transform:  {fmt['transform']}")
                if fmt["res"] is not None:
                    print(f"      Resolution: {fmt['res'][0]} x {fmt['res'][1]}")
                print(f"      Nodata: {fmt['nodata']}")

    return canonical_format, mismatches


def compare_folders(canonicals: Dict[str, Optional[Dict[str, Any]]]) -> None:
    """
    Compare canonical formats between folders and print a summary.
    """
    print("\n=== Cross-folder format comparison ===")

    names = list(canonicals.keys())
    any_missing = False
    for name in names:
        if canonicals[name] is None:
            print(f"  - {name}: no canonical format (folder missing or no GeoTIFFs).")
            any_missing = True

    if any_missing:
        print("  (Skipping comparisons for folders without valid format.)")

    # Pairwise comparisons
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name = names[i]
            b_name = names[j]
            a_fmt = canonicals[a_name]
            b_fmt = canonicals[b_name]

            if a_fmt is None or b_fmt is None:
                continue

            same = formats_equal(a_fmt, b_fmt)
            if same:
                print(f"  ✓ {a_name} and {b_name} share the same format.")
            else:
                print(f"  ! {a_name} and {b_name} differ in format.")
                # Optionally, point out the main differences:
                for key in ["crs", "width", "height", "count", "dtype", "res", "nodata"]:
                    if a_fmt.get(key) != b_fmt.get(key):
                        print(f"      - {key}: {a_name}={a_fmt.get(key)}, {b_name}={b_fmt.get(key)}")
                # For transform, you may want to see the full tuple:
                if a_fmt.get("transform") != b_fmt.get("transform"):
                    print(f"      - transform differs:")
                    print(f"          {a_name}: {a_fmt.get('transform')}")
                    print(f"          {b_name}: {b_fmt.get('transform')}")


def main():
    parser = argparse.ArgumentParser(
        description="Check GeoTIFF format consistency for era5land, smap_wtd, and viirs folders, "
                    "and compare formats across folders."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory containing era5land, smap_wtd, and viirs subfolders (default: current directory).",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="If set, scan all files in each folder instead of just one.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()

    canonicals: Dict[str, Optional[Dict[str, Any]]] = {}
    for name in ["era5land", "smap_wtd", "viirs"]:
        folder = base / name
        canonical_format, mismatches = check_folder(folder, full_scan=args.full_scan)
        canonicals[name] = canonical_format

    compare_folders(canonicals)


if __name__ == "__main__":
    main()
