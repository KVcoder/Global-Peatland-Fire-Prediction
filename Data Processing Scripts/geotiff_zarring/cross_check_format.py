#!/usr/bin/env python3
"""
cross_check_format.py
---------------------

Check the *format* of two GeoTIFFs: a SAMPLE file (e.g., ERA5 template) and an OUTPUT file
(e.g., SMAP resampled result). Prints metadata side-by-side and validates grid compatibility.

Exit code:
- 0 if grid checks pass (CRS, size, resolution, transform)
- 1 if they fail (use --no-strict-exit to always exit 0)

Requires: rasterio>=1.2, numpy
"""

import sys
import json
import math
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def pixel_size(transform) -> Tuple[float, float]:
    return (abs(transform.a), abs(transform.e))


def crs_to_str(crs) -> str | None:
    if crs is None:
        return None
    try:
        epsg = crs.to_epsg()
        if epsg is not None:
            return f"EPSG:{epsg}"
    except Exception:
        pass
    try:
        return crs.to_string()
    except Exception:
        return str(crs)


def get_imgstruct_tag(ds, key: str) -> Any:
    keyU = key.upper()
    # profile first
    val = ds.profile.get(key.lower())
    if val is not None:
        return val
    # IMAGE_STRUCTURE domain
    try:
        t = ds.tags(ns="IMAGE_STRUCTURE")
        if keyU in t:
            return t[keyU]
    except Exception:
        pass
    # dataset-level tags
    try:
        t = ds.tags()
        if keyU in t:
            return t[keyU]
    except Exception:
        pass
    # band-level tags (e.g., PREDICTOR sometimes here)
    try:
        t = ds.tags(1)
        if keyU in t:
            return t[keyU]
    except Exception:
        pass
    return None


def dataset_meta(path: Path) -> Dict[str, Any]:
    with rasterio.open(path) as ds:
        resx, resy = pixel_size(ds.transform)

        # Tiling / block size (per-band; take band 1 if present)
        blockx = blocky = None
        tiled = None
        try:
            if ds.block_shapes and len(ds.block_shapes) >= 1 and ds.block_shapes[0]:
                blocky, blockx = ds.block_shapes[0]
                tiled = (blockx is not None and blocky is not None
                         and blockx < ds.width and blocky < ds.height)
        except Exception:
            pass

        compress = (get_imgstruct_tag(ds, "COMPRESS")
                    or get_imgstruct_tag(ds, "COMPRESSION")
                    or ds.profile.get("compress"))
        predictor = get_imgstruct_tag(ds, "PREDICTOR")
        zstd_level = ds.profile.get("zstd_level")
        bigtiff = ds.profile.get("bigtiff")

        # FIX: Use Rasterio's .descriptions property (tuple of strings or None)
        try:
            band_descs = list(ds.descriptions) if getattr(ds, "descriptions", None) is not None else [None] * ds.count
        except Exception:
            band_descs = [None] * ds.count

        # Normalize predictor & zstd_level to ints when possible
        def as_int_or(val):
            if isinstance(val, (int, np.integer)):
                return int(val)
            if isinstance(val, str) and val.isdigit():
                return int(val)
            return val

        meta = {
            "path": str(path),
            "driver": ds.driver,
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtype": ds.dtypes[0] if ds.count > 0 else None,
            "nodata": ds.nodata,
            "crs": crs_to_str(ds.crs),
            "transform": tuple(ds.transform) if ds.transform is not None else None,
            "resx": resx,
            "resy": resy,
            "tiled": bool(tiled) if tiled is not None else None,
            "blockx": blockx,
            "blocky": blocky,
            "compress": (compress.upper() if isinstance(compress, str) else compress),
            "predictor": as_int_or(predictor),
            "zstd_level": as_int_or(zstd_level),
            "bigtiff": bigtiff,
            "band_descriptions": band_descs,
        }
        return meta


def close_enough(a: float, b: float, rtol: float, atol: float) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def compare(sample: Dict[str, Any], output: Dict[str, Any], rtol: float, atol: float) -> Dict[str, Any]:
    checks = {}

    checks["crs_match"] = (sample["crs"] == output["crs"])
    checks["shape_match"] = (sample["width"] == output["width"] and sample["height"] == output["height"])
    checks["resx_match"] = close_enough(sample["resx"], output["resx"], rtol, atol)
    checks["resy_match"] = close_enough(sample["resy"], output["resy"], rtol, atol)

    # Transform element-wise
    tx_ok = False
    if sample["transform"] and output["transform"] and len(sample["transform"]) == len(output["transform"]):
        tx_ok = all(close_enough(float(a), float(b), rtol, atol)
                    for a, b in zip(sample["transform"], output["transform"]))
    checks["transform_match"] = tx_ok

    # Helpful extras
    checks["dtype_same"] = (sample["dtype"] == output["dtype"])
    checks["tiled_output"] = bool(output["tiled"])

    checks["overall_pass"] = (checks["crs_match"] and checks["shape_match"]
                              and checks["resx_match"] and checks["resy_match"]
                              and checks["transform_match"])
    return checks


def pretty_print(sample: Dict[str, Any], output: Dict[str, Any], checks: Dict[str, Any]) -> None:
    def row(k, a, b):
        print(f"{k:>16}: {str(a):<30} | {str(b):<30}")

    print("\n=== FORMAT CHECK ===\n")
    print(f"{'Field':>16}  {'SAMPLE':<30} | {'OUTPUT':<30}")
    print("-" * 82)
    row("path", sample["path"], output["path"])
    row("driver", sample["driver"], output["driver"])
    row("size", f"{sample['width']} x {sample['height']}", f"{output['width']} x {output['height']}")
    row("bands", sample["count"], output["count"])
    row("dtype", sample["dtype"], output["dtype"])
    row("nodata", sample["nodata"], output["nodata"])
    row("crs", sample["crs"], output["crs"])
    row("res", f"{sample['resx']:.9f},{sample['resy']:.9f}", f"{output['resx']:.9f},{output['resy']:.9f}")
    row("tiled", sample["tiled"], output["tiled"])
    row("block", f"{sample['blockx']}x{sample['blocky']}", f"{output['blockx']}x{output['blocky']}")
    row("compress", sample["compress"], output["compress"])
    row("predictor", sample["predictor"], output["predictor"])
    row("zstd_level", sample["zstd_level"], output["zstd_level"])
    row("bigtiff", sample["bigtiff"], output["bigtiff"])

    print("\nTransform (sample):", sample["transform"])
    print("Transform (output):", output["transform"])

    print("\n--- Checks ---")
    for k, v in checks.items():
        print(f"{k:>16}: {v}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Check format of a sample GeoTIFF vs an output GeoTIFF.")
    ap.add_argument("--sample", required=True, help="Path to sample/template GeoTIFF")
    ap.add_argument("--output", required=True, help="Path to output/resampled GeoTIFF")
    ap.add_argument("--rtol", type=float, default=1e-9, help="Relative tolerance for comparisons")
    ap.add_argument("--atol", type=float, default=1e-9, help="Absolute tolerance for comparisons")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of pretty text")
    ap.add_argument("--no-strict-exit", action="store_true", help="Always exit 0 even if checks fail")
    args = ap.parse_args()

    sample_path = Path(args.sample)
    output_path = Path(args.output)
    if not sample_path.exists():
        print(f"[ERR] Sample not found: {sample_path}", file=sys.stderr)
        sys.exit(2)
    if not output_path.exists():
        print(f"[ERR] Output not found: {output_path}", file=sys.stderr)
        sys.exit(2)

    sample = dataset_meta(sample_path)
    output = dataset_meta(output_path)
    checks = compare(sample, output, args.rtol, args.atol)

    if args.json:
        print(json.dumps({"sample": sample, "output": output, "checks": checks}, indent=2))
    else:
        pretty_print(sample, output, checks)

    if args.no_strict_exit:
        sys.exit(0)
    sys.exit(0 if checks["overall_pass"] else 1)


if __name__ == "__main__":
    main()
