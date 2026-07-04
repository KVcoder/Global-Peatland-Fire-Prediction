#!/usr/bin/env python3
"""
bands_count.py — Print how many bands a .tif/.tiff has (optionally list names).

Requires: rasterio  (pip install rasterio)
"""

import argparse
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser(description="Count bands in a GeoTIFF.")
    p.add_argument("tif", help="Path to .tif/.tiff file")
    p.add_argument("--list", action="store_true",
                   help="Also list band index and description/name if present")
    args = p.parse_args()

    tif_path = Path(args.tif)
    if not tif_path.is_file():
        print(f"ERROR: File not found: {tif_path}", file=sys.stderr)
        sys.exit(1)

    try:
        import rasterio
    except ImportError:
        print("ERROR: rasterio is required. Install with: pip install rasterio", file=sys.stderr)
        sys.exit(2)

    try:
        with rasterio.open(tif_path) as ds:
            print(ds.count)
            if args.list:
                for i in range(1, ds.count + 1):
                    desc = ds.descriptions[i - 1] if ds.descriptions else None
                    name = desc if (desc and desc.strip()) else f"Band {i}"
                    print(f"{i}\t{name}")
    except Exception as e:
        print(f"ERROR: Could not read {tif_path}: {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
