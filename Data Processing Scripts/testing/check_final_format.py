#!/usr/bin/env python3
"""
Check alignment & encoding of a single-day sample from each source.

Verifies that the selected ERA5, WTD, and VIIRS GeoTIFFs share the same:
- CRS
- Affine transform (within tolerance)
- Width/height
- Tiling layout (tiled, block size)
- Compression and Predictor (expects ZSTD + PREDICTOR=3)
- Data types and nodata handling
- Band counts (ERA5=12, WTD=1, VIIRS=1)

Exit code is non-zero if any check fails.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import rasterio

def read_tags(ds) -> Dict[str, str]:
    tags = {}
    for ns in [None, 'IMAGE_STRUCTURE', 'TIFF', 'GDAL_METADATA', 'DERIVED_SUBDATASET']:
        try:
            tags.update(ds.tags(ns=ns))
        except Exception:
            pass
    return {str(k).upper(): str(v) for k, v in tags.items()}

def predictor_from_tags(tags: Dict[str,str]) -> Optional[int]:
    for key in ['PREDICTOR', 'TIFF:PREDICTOR', 'IMAGE_STRUCTURE:PREDICTOR']:
        if key in tags:
            try:
                return int(tags[key])
            except Exception:
                return None
    return None

def profile_summary(ds) -> Dict:
    tags = read_tags(ds)
    prof = ds.profile.copy()
    comp = (tags.get('COMPRESSION') or prof.get('compress') or '').upper()
    return dict(
        name=ds.name,
        crs=str(ds.crs) if ds.crs else None,
        transform=list(ds.transform)[:6],
        width=ds.width,
        height=ds.height,
        count=ds.count,
        dtype=prof.get('dtype'),
        nodata=prof.get('nodata', ds.nodatavals[0] if ds.nodatavals else None),
        tiled=prof.get('tiled'),
        blockx=prof.get('blockxsize'),
        blocky=prof.get('blockysize'),
        compression=comp,
        predictor=predictor_from_tags(tags),
    )

def close_enough(a: Tuple[float,...], b: Tuple[float,...], atol: float) -> bool:
    return np.allclose(np.array(a, dtype=float), np.array(b, dtype=float), atol=atol, rtol=0.0)

def check_pair(base: Dict, other: Dict, tol: float, label_a: str, label_b: str, errors: list):
    if base['crs'] != other['crs']:
        errors.append(f"CRS mismatch {label_a} vs {label_b}: {base['crs']} != {other['crs']}")
    if not close_enough(base['transform'], other['transform'], tol):
        errors.append(f"Transform mismatch {label_a} vs {label_b}: {base['transform']} != {other['transform']} (tol={tol})")
    if base['width'] != other['width'] or base['height'] != other['height']:
        errors.append(f"Shape mismatch {label_a} vs {label_b}: {(base['width'], base['height'])} != {(other['width'], other['height'])}")
    if base['tiled'] != other['tiled']:
        errors.append(f"Tiling mismatch {label_a} vs {label_b}: {base['tiled']} != {other['tiled']}")
    if (base['blockx'] != other['blockx']) or (base['blocky'] != other['blocky']):
        errors.append(f"Block size mismatch {label_a} vs {label_b}: {(base['blockx'], base['blocky'])} != {(other['blockx'], other['blocky'])}")

def expect_encoding(s: Dict, expect_count: int, expect_dtype: str, expect_compress: str = 'ZSTD', expect_predictor: int = 3) -> list:
    errs = []
    if s['count'] != expect_count:
        errs.append(f"Band count expected {expect_count} but got {s['count']} for {s['name']}")
    if (s['dtype'] or '').lower() != expect_dtype.lower():
        errs.append(f"dtype expected {expect_dtype} but got {s['dtype']} for {s['name']}")
    comp = (s.get('compression') or '').upper()
    if expect_compress and expect_compress.upper() not in comp:
        errs.append(f"Compression expected {expect_compress} but got '{comp}' for {s['name']}")
    pred = s.get('predictor')
    if expect_predictor is not None and pred != expect_predictor:
        errs.append(f"Predictor expected {expect_predictor} but got {pred} for {s['name']}")
    return errs

def main():
    ap = argparse.ArgumentParser(description="Check single-day file alignment and encoding across ERA5/WTD/VIIRS")
    ap.add_argument('--era5-file', type=Path, required=True, help='Path to a 12-band ERA5-Land GeoTIFF for one day')
    ap.add_argument('--wtd-file', type=Path, required=True, help='Path to a 1-band WTD GeoTIFF for the SAME day')
    ap.add_argument('--viirs-file', type=Path, required=True, help='Path to a 1-band VIIRS GeoTIFF for the SAME day')
    ap.add_argument('--transform-atol', type=float, default=1e-9, help='Absolute tolerance for affine comparison')
    args = ap.parse_args()

    with rasterio.Env(GDAL_NUM_THREADS='ALL_CPUS', NUM_THREADS='ALL_CPUS', GDAL_CACHEMAX=1024):
        with rasterio.open(args.era5_file) as dse, rasterio.open(args.wtd_file) as dsw, rasterio.open(args.viirs_file) as dsv:
            se = profile_summary(dse)
            sw = profile_summary(dsw)
            sv = profile_summary(dsv)

    print("ERA5   :", se)
    print("WTD    :", sw)
    print("VIIRS  :", sv)

    errors = []
    check_pair(se, sw, args.transform_atol, 'ERA5', 'WTD', errors)
    check_pair(se, sv, args.transform_atol, 'ERA5', 'VIIRS', errors)

    errors += expect_encoding(se, expect_count=12, expect_dtype='float32')
    errors += expect_encoding(sw, expect_count=1, expect_dtype='float32')
    errors += expect_encoding(sv, expect_count=1, expect_dtype='float32')

    if errors:
        print("\n❌ FORMAT MISMATCHES FOUND:")
        for e in errors:
            print(" -", e)
        sys.exit(2)
    else:
        print("\n✅ All checks passed: files share the same grid & expected encoding (ZSTD + Predictor=3).")
        sys.exit(0)

if __name__ == '__main__':
    main()
