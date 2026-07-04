#!/usr/bin/env python3
"""
globe_tif_viewer.py

Interactive GeoTIFF-on-globe viewer using Rasterio + PyVista, with
Cartopy/Natural Earth background layers (ocean, land, coastlines, country borders).

This version fixes the "land colored like ocean" artifact by:
- keeping the ocean as a full ellipsoid
- replacing polygon-triangulated land fill with a rasterized land mask mesh
- still drawing coastlines and country borders from Natural Earth vectors

Features
--------
- Reads a single-band GeoTIFF / .tif
- Masks NoData from file metadata and explicit -9999 values
- Draws only valid tiles; invalid tiles are transparent/missing
- Colors valid tiles by value with configurable colormap and scale
- Renders on a rotatable 3D oblate globe
- Supports a square interactive window
- Saves high-resolution poster-ready PNGs
- Optional interactive HTML export
- Background layers: ocean, land, coastlines, country borders

Hotkeys
-------
P : save a high-resolution PNG
C : print current camera settings
E : export an interactive HTML scene

Install
-------
pip install numpy rasterio pyvista matplotlib cartopy shapely
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject

from cartopy.io import shapereader as shpreader
from shapely.geometry import (
    Polygon,
    MultiPolygon,
    LineString,
    MultiLineString,
    GeometryCollection,
)

# WGS84 ellipsoid, normalized so equatorial radius = 1.0
WGS84_A = 1.0
WGS84_B = 6356752.314245 / 6378137.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive rotatable globe viewer for a GeoTIFF."
    )

    parser.add_argument("tif", type=str, help="Path to input GeoTIFF.")
    parser.add_argument("--band", type=int, default=7, help="Band to read (default: 1).")
    parser.add_argument(
        "--nodata",
        type=float,
        default=None,
        help="Override nodata value from file metadata.",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=4,
        help=(
            "Downsample factor for interactive rendering. "
            "1 = full resolution, 2 = every 2nd cell, etc. Default: 4"
        ),
    )

    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap name.")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum value for color scale.")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum value for color scale.")
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use logarithmic color mapping (valid values must be > 0).",
    )

    parser.add_argument(
        "--height-scale",
        type=float,
        default=0.0,
        help=(
            "Optional radial exaggeration relative to globe radius. "
            "0.0 means no height extrusion. Try 0.01 to 0.03 for a subtle effect."
        ),
    )
    parser.add_argument(
        "--data-offset",
        type=float,
        default=0.003,
        help="Small height offset to lift TIFF tiles above the background globe.",
    )

    parser.add_argument("--center-lon", type=float, default=0.0, help="Starting center longitude.")
    parser.add_argument("--center-lat", type=float, default=20.0, help="Starting center latitude.")
    parser.add_argument("--roll", type=float, default=0.0, help="Camera roll in degrees.")
    parser.add_argument(
        "--distance",
        type=float,
        default=2.8,
        help="Camera distance from the globe center.",
    )
    parser.add_argument("--zoom", type=float, default=1.0, help="Additional zoom factor.")

    parser.add_argument(
        "--background-resolution",
        type=str,
        default="110m",
        choices=["110m", "50m", "10m"],
        help="Natural Earth resolution for background layers.",
    )
    parser.add_argument("--ocean-color", type=str, default="#9ecae1", help="Ocean color.")
    parser.add_argument("--land-color", type=str, default="#e8dfc8", help="Land color.")
    parser.add_argument("--coastline-color", type=str, default="#666666", help="Coastline color.")
    parser.add_argument(
        "--country-border-color",
        type=str,
        default="#777777",
        help="Country border color.",
    )
    parser.add_argument("--coastline-width", type=float, default=1.0, help="Coastline width.")
    parser.add_argument(
        "--country-border-width",
        type=float,
        default=0.7,
        help="Country border width.",
    )
    parser.add_argument("--hide-land", action="store_true", help="Hide land fill.")
    parser.add_argument("--hide-country-borders", action="store_true", help="Hide country borders.")
    parser.add_argument("--hide-coastlines", action="store_true", help="Hide coastlines.")

    parser.add_argument("--show-edges", action="store_true", help="Draw tile boundaries.")
    parser.add_argument("--edge-color", type=str, default="black", help="Tile edge color.")
    parser.add_argument("--line-width", type=float, default=0.2, help="Tile edge width.")

    parser.add_argument("--background", type=str, default="white", help="Window background color.")
    parser.add_argument(
        "--graticule-step",
        type=int,
        default=30,
        help="Spacing in degrees for graticule lines. Default: 30",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=1400,
        help="Square render window size in pixels. Default: 1400",
    )
    parser.add_argument(
        "--image-scale",
        type=int,
        default=4,
        help="Multiplier used when saving PNGs. Saved size = window_size * image_scale.",
    )
    parser.add_argument("--save", type=str, default=None, help="PNG path for saved image.")
    parser.add_argument(
        "--export-html",
        type=str,
        default=None,
        help="HTML path for interactive scene export.",
    )
    parser.add_argument(
        "--off-screen",
        action="store_true",
        help="Render off-screen and save immediately instead of opening the viewer.",
    )

    return parser.parse_args()


def read_raster(path: str, band: int = 7, nodata_override: float | None = None):
    """
    Read one raster band.
    Reproject to EPSG:4326 if needed.
    Build valid-data mask from dataset_mask() plus explicit nodata checks.
    """
    dst_crs = CRS.from_epsg(4326)

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("Input TIFF has no CRS. A georeferenced raster is required.")

        data = src.read(band).astype(np.float32)
        ds_mask = src.dataset_mask().astype(np.uint8)
        transform = src.transform
        crs = src.crs
        nodata = nodata_override if nodata_override is not None else src.nodata

        if crs != dst_crs:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                crs, dst_crs, src.width, src.height, *src.bounds
            )

            dst_data = np.empty((dst_height, dst_width), dtype=np.float32)
            dst_mask = np.empty((dst_height, dst_width), dtype=np.uint8)

            reproject(
                source=data,
                destination=dst_data,
                src_transform=transform,
                src_crs=crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
            reproject(
                source=ds_mask,
                destination=dst_mask,
                src_transform=transform,
                src_crs=crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )

            data = dst_data
            ds_mask = dst_mask
            transform = dst_transform

    valid = ds_mask > 0
    valid &= np.isfinite(data)
    valid &= data != -9999

    if nodata is not None:
        if np.isnan(nodata):
            valid &= ~np.isnan(data)
        else:
            valid &= data != nodata

    return data, valid, transform


def make_edge_indices(n: int, step: int) -> np.ndarray:
    step = max(1, int(step))
    idx = np.arange(0, n + 1, step, dtype=np.int32)
    if idx[-1] != n:
        idx = np.append(idx, n)
    return idx


def sample_cells(data: np.ndarray, valid: np.ndarray, step: int):
    row_edges = make_edge_indices(data.shape[0], step)
    col_edges = make_edge_indices(data.shape[1], step)

    row_centers = np.clip((row_edges[:-1] + row_edges[1:] - 1) // 2, 0, data.shape[0] - 1)
    col_centers = np.clip((col_edges[:-1] + col_edges[1:] - 1) // 2, 0, data.shape[1] - 1)

    sampled_values = data[np.ix_(row_centers, col_centers)]
    sampled_valid = valid[np.ix_(row_centers, col_centers)]

    return row_edges, col_edges, sampled_values, sampled_valid


def affine_xy(transform, rows: np.ndarray, cols: np.ndarray):
    cc, rr = np.meshgrid(cols, rows)
    x = transform.c + cc * transform.a + rr * transform.b
    y = transform.f + cc * transform.d + rr * transform.e
    return x.astype(np.float64), y.astype(np.float64)


def geodetic_to_ecef(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    a: float = WGS84_A,
    b: float = WGS84_B,
    h: float | np.ndarray = 0.0,
):
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    e2 = 1.0 - (b * b) / (a * a)
    N = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)

    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = ((b * b / (a * a)) * N + h) * sin_lat
    return x, y, z


def corner_heights_from_cells(
    cell_values: np.ndarray,
    cell_valid: np.ndarray,
    vmin: float,
    vmax: float,
    height_scale: float,
    base_offset: float = 0.0,
) -> np.ndarray:
    ny, nx = cell_values.shape
    out = np.full((ny + 1, nx + 1), base_offset, dtype=np.float64)

    if height_scale == 0.0 or vmax <= vmin:
        return out

    normed = np.zeros_like(cell_values, dtype=np.float64)
    normed[cell_valid] = (cell_values[cell_valid] - vmin) / (vmax - vmin)
    normed = np.clip(normed, 0.0, 1.0)

    acc = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    cnt = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    vals = np.where(cell_valid, normed, 0.0)

    for di in (0, 1):
        for dj in (0, 1):
            acc[di : di + ny, dj : dj + nx] += vals
            cnt[di : di + ny, dj : dj + nx] += cell_valid

    avg = np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)
    out += height_scale * avg
    return out


def build_valid_quad_mesh(
    lon_corners: np.ndarray,
    lat_corners: np.ndarray,
    cell_values: np.ndarray,
    cell_valid: np.ndarray,
    vmin: float,
    vmax: float,
    height_scale: float = 0.0,
    base_offset: float = 0.0,
) -> pv.PolyData:
    corner_heights = corner_heights_from_cells(
        cell_values=cell_values,
        cell_valid=cell_valid,
        vmin=vmin,
        vmax=vmax,
        height_scale=height_scale,
        base_offset=base_offset,
    )

    x, y, z = geodetic_to_ecef(lon_corners, lat_corners, h=corner_heights)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)

    ny, nx = cell_values.shape
    ncols = nx + 1

    def idx(i: int, j: int) -> int:
        return i * ncols + j

    faces = []
    values = []

    for i in range(ny):
        for j in range(nx):
            if not cell_valid[i, j]:
                continue

            faces.extend([4, idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)])
            values.append(float(cell_values[i, j]))

    if not values:
        raise ValueError("No valid cells remain after masking nodata values.")

    mesh = pv.PolyData(points, np.asarray(faces, dtype=np.int64))
    mesh.cell_data["value"] = np.asarray(values, dtype=np.float32)
    return mesh


def make_polyline(points: np.ndarray) -> pv.PolyData:
    n = len(points)
    lines = np.hstack(([n], np.arange(n, dtype=np.int64)))
    return pv.PolyData(points.astype(np.float32), lines=lines)


def add_graticule(
    plotter: pv.Plotter,
    step_deg: int = 30,
    color: str = "#888888",
    width: float = 1.0,
    opacity: float = 0.35,
    height_offset: float = 0.0015,
):
    step_deg = max(1, int(step_deg))

    lats = np.linspace(-89.999, 89.999, 361)
    for lon in range(-180, 181, step_deg):
        lon_arr = np.full_like(lats, lon, dtype=float)
        x, y, z = geodetic_to_ecef(lon_arr, lats, h=height_offset)
        line = make_polyline(np.column_stack((x, y, z)))
        plotter.add_mesh(line, color=color, line_width=width, opacity=opacity, lighting=False)

    lons = np.linspace(-180, 180, 721)
    for lat in range(-90 + step_deg, 90, step_deg):
        lat_arr = np.full_like(lons, lat, dtype=float)
        x, y, z = geodetic_to_ecef(lons, lat_arr, h=height_offset)
        line = make_polyline(np.column_stack((x, y, z)))
        plotter.add_mesh(line, color=color, line_width=width, opacity=opacity, lighting=False)


def _split_dateline(coords: np.ndarray, max_jump: float = 180.0):
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return []

    parts = []
    current = [coords[0]]

    for p in coords[1:]:
        if abs(p[0] - current[-1][0]) > max_jump:
            if len(current) > 1:
                parts.append(np.array(current))
            current = [p]
        else:
            current.append(p)

    if len(current) > 1:
        parts.append(np.array(current))

    return parts


def add_lines_from_geom(
    plotter: pv.Plotter,
    geom,
    color: str = "black",
    width: float = 1.0,
    height: float = 0.0015,
):
    if geom.is_empty:
        return

    if isinstance(geom, LineString):
        parts = _split_dateline(np.asarray(geom.coords))
        for part in parts:
            if len(part) < 2:
                continue
            x, y, z = geodetic_to_ecef(part[:, 0], part[:, 1], h=height)
            line = make_polyline(np.column_stack((x, y, z)))
            plotter.add_mesh(line, color=color, line_width=width, lighting=False)

    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            add_lines_from_geom(plotter, g, color=color, width=width, height=height)

    elif isinstance(geom, Polygon):
        add_lines_from_geom(
            plotter,
            LineString(geom.exterior.coords),
            color=color,
            width=width,
            height=height,
        )
        for ring in geom.interiors:
            add_lines_from_geom(
                plotter,
                LineString(ring.coords),
                color=color,
                width=width,
                height=height,
            )

    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            add_lines_from_geom(plotter, g, color=color, width=width, height=height)

    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            add_lines_from_geom(plotter, g, color=color, width=width, height=height)


def land_mask_cell_size_deg(resolution: str) -> float:
    """
    Pick a rasterization cell size for the land mask.
    Smaller = smoother coastline but more quads.
    """
    if resolution == "10m":
        return 0.10
    if resolution == "50m":
        return 0.25
    return 0.75  # 110m


def build_rasterized_land_mesh(
    geoms,
    resolution: str = "110m",
    height: float = 0.0010,
) -> pv.PolyData | None:
    """
    Rasterize land polygons to a regular lon/lat mask and convert that
    mask to a quad mesh on the globe.

    This avoids polygon triangulation artifacts that can leave ocean-colored
    gaps inside continents.
    """
    cell_deg = land_mask_cell_size_deg(resolution)

    ncols = int(round(360.0 / cell_deg))
    nrows = int(round(180.0 / cell_deg))

    transform = from_origin(-180.0, 90.0, cell_deg, cell_deg)

    land_mask = rasterize(
        ((geom, 1) for geom in geoms if not geom.is_empty),
        out_shape=(nrows, ncols),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype=np.uint8,
    ).astype(bool)

    if not np.any(land_mask):
        return None

    row_edges = np.arange(0, nrows + 1, dtype=np.int32)
    col_edges = np.arange(0, ncols + 1, dtype=np.int32)
    lon_corners, lat_corners = affine_xy(transform, row_edges, col_edges)

    dummy_values = np.ones((nrows, ncols), dtype=np.float32)

    mesh = build_valid_quad_mesh(
        lon_corners=lon_corners,
        lat_corners=lat_corners,
        cell_values=dummy_values,
        cell_valid=land_mask,
        vmin=0.0,
        vmax=1.0,
        height_scale=0.0,
        base_offset=height,
    )
    return mesh


def add_background_layers(
    plotter: pv.Plotter,
    resolution: str = "110m",
    ocean_color: str = "#9ecae1",
    land_color: str = "#e8dfc8",
    coastline_color: str = "#666666",
    country_border_color: str = "#777777",
    coastline_width: float = 1.0,
    country_border_width: float = 0.7,
    show_land: bool = True,
    show_coastlines: bool = True,
    show_country_borders: bool = True,
):
    """
    Add ocean ellipsoid plus optional land fill, coastlines, and country borders.

    Land fill is rasterized from Natural Earth admin_0_countries polygons into a
    global lon/lat mask, then drawn as quads on the globe.
    """
    base = pv.Sphere(radius=1.0, theta_resolution=180, phi_resolution=180)
    base.scale([1.0, 1.0, WGS84_B / WGS84_A], inplace=True)
    plotter.add_mesh(
        base,
        color=ocean_color,
        lighting=False,
        smooth_shading=False,
        opacity=1.0,
    )

    country_geoms = None
    if show_land or show_country_borders:
        countries_path = shpreader.natural_earth(
            resolution=resolution,
            category="cultural",
            name="admin_0_countries",
        )
        country_geoms = list(shpreader.Reader(countries_path).geometries())

    if show_land and country_geoms is not None:
        land_mesh = build_rasterized_land_mesh(
            country_geoms,
            resolution=resolution,
            height=0.0010,
        )
        if land_mesh is not None:
            plotter.add_mesh(
                land_mesh,
                color=land_color,
                lighting=False,
                smooth_shading=False,
                opacity=1.0,
            )

    if show_coastlines:
        coast_path = shpreader.natural_earth(
            resolution=resolution,
            category="physical",
            name="coastline",
        )
        for geom in shpreader.Reader(coast_path).geometries():
            add_lines_from_geom(
                plotter,
                geom,
                color=coastline_color,
                width=coastline_width,
                height=0.0014,
            )

    if show_country_borders:
        border_path = shpreader.natural_earth(
            resolution=resolution,
            category="cultural",
            name="admin_0_boundary_lines_land",
        )
        for geom in shpreader.Reader(border_path).geometries():
            add_lines_from_geom(
                plotter,
                geom,
                color=country_border_color,
                width=country_border_width,
                height=0.0018,
            )


def camera_from_lonlat(center_lon: float, center_lat: float, distance: float, roll_deg: float):
    vx, vy, vz = geodetic_to_ecef(np.array(center_lon), np.array(center_lat))
    view_dir = np.array([vx, vy, vz], dtype=float)
    view_dir = view_dir / np.linalg.norm(view_dir)

    lon = math.radians(center_lon)
    lat = math.radians(center_lat)

    east = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=float)
    north = np.array(
        [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
        dtype=float,
    )

    roll = math.radians(roll_deg)
    up = north * math.cos(roll) + east * math.sin(roll)
    up = up / np.linalg.norm(up)

    position = view_dir * float(distance)
    focal_point = np.array([0.0, 0.0, 0.0], dtype=float)

    return [tuple(position), tuple(focal_point), tuple(up)]


def auto_output_png(input_path: str) -> str:
    stem = Path(input_path).stem
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{stamp}.png"


def auto_output_html(input_path: str) -> str:
    stem = Path(input_path).stem
    return f"{stem}_scene.html"


def main():
    args = parse_args()

    data, valid, transform = read_raster(
        args.tif,
        band=args.band,
        nodata_override=args.nodata,
    )

    valid_values = data[valid]
    if valid_values.size == 0:
        raise ValueError("No valid data values were found in the input raster.")

    if args.log_scale and np.any(valid_values <= 0):
        raise ValueError("--log-scale requires all valid data values to be > 0.")

    vmin = float(args.vmin) if args.vmin is not None else float(np.nanmin(valid_values))
    vmax = float(args.vmax) if args.vmax is not None else float(np.nanmax(valid_values))

    if vmax <= vmin:
        raise ValueError("Invalid scale: vmax must be greater than vmin.")

    row_edges, col_edges, cell_values, cell_valid = sample_cells(data, valid, step=args.step)
    lon_corners, lat_corners = affine_xy(transform, row_edges, col_edges)

    mesh = build_valid_quad_mesh(
        lon_corners=lon_corners,
        lat_corners=lat_corners,
        cell_values=cell_values,
        cell_valid=cell_valid,
        vmin=vmin,
        vmax=vmax,
        height_scale=args.height_scale,
        base_offset=args.data_offset,
    )

    plotter = pv.Plotter(
        window_size=(args.window_size, args.window_size),
        off_screen=args.off_screen,
    )
    plotter.set_background(args.background)

    add_background_layers(
        plotter,
        resolution=args.background_resolution,
        ocean_color=args.ocean_color,
        land_color=args.land_color,
        coastline_color=args.coastline_color,
        country_border_color=args.country_border_color,
        coastline_width=args.coastline_width,
        country_border_width=args.country_border_width,
        show_land=not args.hide_land,
        show_coastlines=not args.hide_coastlines,
        show_country_borders=not args.hide_country_borders,
    )

    add_graticule(
        plotter,
        step_deg=args.graticule_step,
        color="#888888",
        width=1.0,
        opacity=0.35,
        height_offset=max(0.001, args.data_offset * 0.5),
    )

    plotter.add_mesh(
        mesh,
        scalars="value",
        preference="cell",
        cmap=args.cmap,
        clim=(vmin, vmax),
        log_scale=args.log_scale,
        show_edges=args.show_edges,
        edge_color=args.edge_color,
        line_width=args.line_width,
        lighting=False,
        smooth_shading=False,
        scalar_bar_args={
            "title": "Value\n\n\n\n\n\n\n",
            "vertical": True,
            "position_x": 0.88,
            "position_y": 0.18,
            "height": 0.62,
            "width": 0.06,
            "title_font_size": 14,
            "label_font_size": 11,
            "fmt": "%.3g",
        },
    )

    plotter.camera_position = camera_from_lonlat(
        center_lon=args.center_lon,
        center_lat=args.center_lat,
        distance=args.distance,
        roll_deg=args.roll,
    )
    if args.zoom != 1.0:
        plotter.camera.zoom(args.zoom)

    text_color = "white" if args.background.lower() in {"black", "#000", "#000000"} else "black"
    plotter.add_text(
        "Mouse: rotate | scroll/right-drag: zoom | P: save PNG | C: print camera | E: export HTML",
        position="upper_left",
        font_size=10,
        color=text_color,
    )

    def save_png():
        out = args.save if args.save else auto_output_png(args.tif)
        plotter.screenshot(out, scale=args.image_scale)
        print(f"Saved PNG: {Path(out).resolve()}")

    def print_camera():
        print("\nCurrent camera_position =")
        print(plotter.camera_position)
        print()

    def export_html():
        out = args.export_html if args.export_html else auto_output_html(args.tif)
        try:
            plotter.export_html(out)
            print(f"Exported HTML scene: {Path(out).resolve()}")
        except Exception as exc:
            print(f"HTML export failed: {exc}")
            print("You may need extra PyVista/Trame HTML export dependencies installed.")

    plotter.add_key_event("p", save_png)
    plotter.add_key_event("c", print_camera)
    plotter.add_key_event("e", export_html)

    if args.off_screen:
        if args.save is None:
            args.save = f"{Path(args.tif).stem}.png"
        save_png()
        return

    plotter.show(title=Path(args.tif).name)


if __name__ == "__main__":
    main()