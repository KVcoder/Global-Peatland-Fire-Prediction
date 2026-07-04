#!/usr/bin/env python3
"""
check_geotiff_names.py

Inspect one or more GeoTIFF files and print band / feature names.

For each file, this script prints:
  - basic info (size, CRS, band count)
  - for each band:
      * band index
      * "name" inferred from common metadata fields
      * raw description and a few common tag keys

Requires: rasterio
    pip install rasterio
"""

import argparse
import sys
from pathlib import Path

import rasterio


CANDIDATE_TAG_KEYS = [
    "long_name",
    "standard_name",
    "short_name",
    "variable",
    "var_name",
    "band_name",
    "name",
    "DESCRIPTION",
]


def inspect_geotiff(path: Path):
    print(f"\n=== {path} ===")

    if not path.is_file():
        print("  [ERROR] Not a file or does not exist.")
        return

    try:
        with rasterio.open(path) as src:
            print(f"  Driver:     {src.driver}")
            print(f"  Size:       {src.width} x {src.height}")
            print(f"  Bands:      {src.count}")
            print(f"  CRS:        {src.crs}")
            print(f"  Data type:  {src.dtypes}")

            # Dataset-level tags might also contain variable info
            ds_tags = src.tags()
            if ds_tags:
                print("  Dataset tags (subset):")
                for k, v in list(ds_tags.items())[:10]:
                    print(f"    {k} = {v}")

            print("\n  Per-band info:")
            for bidx in range(1, src.count + 1):
                desc = src.descriptions[bidx - 1] or ""
                tags = src.tags(bidx)

                # Infer a "nice" name from description or common tag keys
                name = desc.strip() if desc else ""
                if not name:
                    for key in CANDIDATE_TAG_KEYS:
                        if key in tags and tags[key].strip():
                            name = tags[key].strip()
                            break
                if not name:
                    name = f"band_{bidx}"

                print(f"    Band {bidx}: {name}")
                if desc:
                    print(f"      description: {desc}")
                common = {k: v for k, v in tags.items() if k in CANDIDATE_TAG_KEYS}
                if common:
                    print("      tags:")
                    for k, v in common.items():
                        print(f"        {k} = {v}")
    except Exception as e:
        print(f"  [ERROR] Failed to open {path}: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check GeoTIFFs for band / feature names."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more GeoTIFF files to inspect",
    )
    args = parser.parse_args(argv)

    for f in args.files:
        inspect_geotiff(Path(f))


if __name__ == "__main__":
    main()
