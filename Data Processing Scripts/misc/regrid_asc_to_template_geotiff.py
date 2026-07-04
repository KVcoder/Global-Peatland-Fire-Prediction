#!/usr/bin/env python3
"""
regrid_asc_to_template_geotiff.py

Regrid GRIP4 AAIGrid (*.asc) rasters to exactly match a template GeoTIFF grid.

Modes:
  - simple: direct resample onto template grid (e.g., average for densities).
  - density_preserving (recommended for GRIP DENS layers):
      density (m/km2) * area_land (km2) -> road length (m) per source cell
      aggregate road length and land area by SUM to template grid
      density_out = length_sum / area_sum

Outputs:
  - GeoTIFFs that match the template's grid (CRS, transform, width/height),
    and inherit template tiling/compression/nodata settings where possible.

Dependencies:
  pip install rasterio numpy tqdm

Example:
  python regrid_asc_to_template_geotiff.py \
    --template smap_wtd_2016-01-01.tif \
    --indir GRIP4_density_total \
    --outdir GRIP_total_regrid \
    --mode density_preserving \
    --include_area_land \
    --pattern "grip4_*.asc"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
from tqdm import tqdm


DENSITY_SUFFIX = "_dens_m_km2.asc"
AREA_LAND_NAME = "grip4_area_land_km2.asc"


def read_raster(path: Path, fallback_crs: CRS | None = None) -> Tuple[np.ndarray, Dict]:
    """
    Read a single-band raster. Many ASCII grids have no embedded CRS; use fallback_crs.
    Returns array and meta dict with crs/transform/nodata.
    """
    with rasterio.open(path) as src:
        arr = src.read(1)
        crs = src.crs if src.crs is not None else fallback_crs
        if crs is None:
            raise rasterio.errors.CRSError(f"Missing CRS for {path} and no fallback CRS provided.")
        meta = {
            "crs": crs,
            "transform": src.transform,
            "nodata": src.nodata,
        }
    return arr, meta


def load_template(template_path: Path) -> Tuple[Dict, Dict]:
    """Load template profile + grid params (crs/transform/width/height/nodata)."""
    with rasterio.open(template_path) as tmp:
        profile = tmp.profile.copy()
        grid = {
            "crs": tmp.crs,
            "transform": tmp.transform,
            "width": tmp.width,
            "height": tmp.height,
            "nodata": tmp.nodata,
        }
    return profile, grid


def reproject_to_template(
    src_arr: np.ndarray,
    src_meta: Dict,
    tpl_grid: Dict,
    resampling: Resampling,
    dst_dtype: np.dtype,
    dst_nodata: float,
) -> np.ndarray:
    """Reproject/resample src_arr onto the template grid."""
    dst = np.full((tpl_grid["height"], tpl_grid["width"]), dst_nodata, dtype=dst_dtype)

    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        src_nodata=src_meta["nodata"],
        dst_transform=tpl_grid["transform"],
        dst_crs=tpl_grid["crs"],
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    return dst


def write_like_template(out_path: Path, template_profile: Dict, data: np.ndarray, nodata: float) -> None:
    """
    Write a GeoTIFF matching the template profile as much as possible.
    Keeps template compression/tiling/blocksize settings if present.
    """
    profile = template_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=str(data.dtype),
        nodata=nodata,
        bigtiff="if_safer",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)


def is_density_file(p: Path) -> bool:
    """Case-insensitive check for GRIP density rasters."""
    return p.name.lower().endswith(DENSITY_SUFFIX)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, required=True, help="Template GeoTIFF (grid/CRS).")
    ap.add_argument("--indir", type=Path, required=True, help="Directory containing GRIP4 *.asc.")
    ap.add_argument("--outdir", type=Path, required=True, help="Output directory for GeoTIFFs.")
    ap.add_argument("--mode", choices=["simple", "density_preserving"], default="density_preserving")
    ap.add_argument(
        "--simple_resampling",
        choices=["nearest", "bilinear", "average"],
        default="average",
        help="Resampling for simple mode (densities usually average).",
    )
    ap.add_argument(
        "--pattern",
        default="*.asc",
        help="Glob pattern for ASC files (default: *.asc).",
    )
    ap.add_argument(
        "--include_area_land",
        action="store_true",
        help="Also output area_land regridded to template grid.",
    )
    ap.add_argument(
        "--area_land_name",
        default=AREA_LAND_NAME,
        help=f"Filename for land area ASC inside indir (default: {AREA_LAND_NAME}).",
    )
    args = ap.parse_args()

    tpl_profile, tpl_grid = load_template(args.template)

    # Use template CRS as fallback CRS for ASC grids missing CRS (GRIP is WGS84/EPSG:4326)
    fallback_crs = tpl_grid["crs"]
    if fallback_crs is None:
        # Extremely rare for a template GeoTIFF, but handle anyway
        fallback_crs = CRS.from_epsg(4326)
        tpl_grid["crs"] = fallback_crs
        tpl_profile["crs"] = fallback_crs

    # Define output nodata: prefer template, else default to -9999 (matches your template)
    dst_nodata = float(tpl_grid["nodata"] if tpl_grid["nodata"] is not None else -9999.0)
    tpl_profile["nodata"] = dst_nodata
    tpl_grid["nodata"] = dst_nodata

    asc_files = sorted(args.indir.glob(args.pattern))
    if not asc_files:
        raise FileNotFoundError(f"No files matched {args.pattern} in {args.indir}")

    # Simple resampling map
    simple_resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "average": Resampling.average,
    }
    simple_resampling = simple_resampling_map[args.simple_resampling]

    if args.mode == "simple":
        for src_path in tqdm(asc_files, desc="Regridding (simple)", unit="file"):
            src_arr, src_meta = read_raster(src_path, fallback_crs=fallback_crs)

            dst = reproject_to_template(
                src_arr=src_arr.astype(np.float32, copy=False),
                src_meta=src_meta,
                tpl_grid=tpl_grid,
                resampling=simple_resampling,
                dst_dtype=np.float32,
                dst_nodata=dst_nodata,
            )

            out_path = args.outdir / (src_path.with_suffix("").name + "_templategrid.tif")
            write_like_template(out_path, tpl_profile, dst, nodata=dst_nodata)

        print("Done.")
        return

    # -------------------------
    # density_preserving mode
    # -------------------------
    area_land_path = args.indir / args.area_land_name
    if not area_land_path.exists():
        raise FileNotFoundError(f"Missing required land-area raster: {area_land_path}")

    # Read area_land once
    area_src_arr, area_src_meta = read_raster(area_land_path, fallback_crs=fallback_crs)

    # Aggregate land area to template grid by SUM
    area_sum_tpl = reproject_to_template(
        src_arr=area_src_arr.astype(np.float64),
        src_meta=area_src_meta,
        tpl_grid=tpl_grid,
        resampling=Resampling.sum,
        dst_dtype=np.float64,
        dst_nodata=np.nan,  # internal accumulation nodata
    )
    area_sum_tpl = np.where(np.isfinite(area_sum_tpl), area_sum_tpl, 0.0)

    if args.include_area_land:
        out_area = args.outdir / "area_land_km2_templategrid.tif"
        area_out = area_sum_tpl.astype(np.float32)
        area_out = np.where(area_sum_tpl > 0, area_out, dst_nodata).astype(np.float32)
        write_like_template(out_area, tpl_profile, area_out, nodata=dst_nodata)

    # Process only density rasters
    dens_files: List[Path] = [p for p in asc_files if is_density_file(p)]
    if not dens_files:
        raise FileNotFoundError(
            f"No density files found. Expected filenames ending with {DENSITY_SUFFIX}. "
            f"Files seen: {[p.name for p in asc_files]}"
        )

    for dens_path in tqdm(dens_files, desc="Regridding (density_preserving)", unit="file"):
        dens_src_arr, dens_src_meta = read_raster(dens_path, fallback_crs=fallback_crs)

        dens = dens_src_arr.astype(np.float64)
        area = area_src_arr.astype(np.float64)

        # Build validity mask
        valid = np.ones(dens.shape, dtype=bool)
        if dens_src_meta["nodata"] is not None:
            valid &= dens != dens_src_meta["nodata"]
        if area_src_meta["nodata"] is not None:
            valid &= area != area_src_meta["nodata"]
        valid &= area > 0

        # density (m/km2) * area (km2) -> length (m) per source cell
        length = np.zeros_like(dens, dtype=np.float64)
        length[valid] = dens[valid] * area[valid]

        # Aggregate length by SUM to template grid
        length_sum_tpl = reproject_to_template(
            src_arr=length,
            src_meta=dens_src_meta,
            tpl_grid=tpl_grid,
            resampling=Resampling.sum,
            dst_dtype=np.float64,
            dst_nodata=np.nan,
        )
        length_sum_tpl = np.where(np.isfinite(length_sum_tpl), length_sum_tpl, 0.0)

        # Compute output density (m/km2) on template grid
        out = np.full((tpl_grid["height"], tpl_grid["width"]), dst_nodata, dtype=np.float32)
        m = area_sum_tpl > 0
        out[m] = (length_sum_tpl[m] / area_sum_tpl[m]).astype(np.float32)

        out_path = args.outdir / (dens_path.with_suffix("").name + "_templategrid.tif")
        write_like_template(out_path, tpl_profile, out, nodata=dst_nodata)

    print("Done.")


if __name__ == "__main__":
    main()
