#!/usr/bin/env python3
"""
Stitch tiled Earth Engine GeoTIFFs into one file per mosaic.

- Groups tiles by stripping the trailing "-<num>-<num>.tif"
  e.g. "WTD_only_Op1deg_WTD_daily_mean_yearly_2016-0000000000-0000001792.tif"
  → group key: "WTD_only_Op1deg_WTD_daily_mean_yearly_2016"
- Creates a VRT and then a single compressed (BigTIFF) GeoTIFF per group.

Usage:
  python stitch_eetiles.py /path/to/input_tiles /path/to/output_merged
"""

import os
import re
import sys
import glob
from osgeo import gdal

def stitch_all(in_dir: str, out_dir: str, delete_vrt: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Match "...-<num>-<num>.tif" and capture the base before the two indices.
    rx = re.compile(r'^(?P<base>.+)-\d+-\d+\.tif$', re.IGNORECASE)

    files = sorted(glob.glob(os.path.join(in_dir, '*.tif')))
    groups = {}
    for f in files:
        m = rx.match(os.path.basename(f))
        if not m:
            continue
        base = m.group('base')
        groups.setdefault(base, []).append(f)

    if not groups:
        print('No tiled files found that match pattern "-<num>-<num>.tif".')
        return

    print(f'Found {len(groups)} mosaics to build.')

    for base, tile_list in sorted(groups.items()):
        tile_list.sort()
        first = tile_list[0]

        # Try to carry through the source NoData value (if any).
        nodata = None
        ds0 = gdal.Open(first, gdal.GA_ReadOnly)
        if ds0:
            b1 = ds0.GetRasterBand(1)
            nodata = b1.GetNoDataValue()
            ds0 = None

        vrt_path = os.path.join(out_dir, os.path.basename(base) + '.vrt')
        tif_path = os.path.join(out_dir, os.path.basename(base) + '.tif')

        if os.path.exists(tif_path):
            print(f'[skip] {os.path.basename(tif_path)} already exists.')
            continue

        print(f'→ {os.path.basename(base)}: {len(tile_list)} tiles → VRT → GeoTIFF')

        vrt_opts = gdal.BuildVRTOptions(
            srcNodata=nodata, VRTNodata=nodata, resolution='highest'
        )
        vrt_ds = gdal.BuildVRT(vrt_path, tile_list, options=vrt_opts)
        if vrt_ds is None:
            print(f'  ERROR: failed to build VRT for {base}')
            continue
        vrt_ds = None

        # Write a single compressed, tiled BigTIFF
        tr_opts = gdal.TranslateOptions(
            format='GTiff',
            creationOptions=['BIGTIFF=YES', 'TILED=YES', 'COMPRESS=LZW']
            # If GDAL >= 3.1 and you want COGs instead, use:
            # format='COG', creationOptions=['COMPRESS=LZW']
        )
        out_ds = gdal.Translate(tif_path, vrt_path, options=tr_opts)
        if out_ds is None:
            print(f'  ERROR: failed to write GeoTIFF for {base}')
        else:
            out_ds = None
            if delete_vrt:
                try:
                    os.remove(vrt_path)
                except OSError:
                    pass
            print(f'  Done: {tif_path}')

    print('All mosaics complete.')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    stitch_all(sys.argv[1], sys.argv[2])
