#!/usr/bin/env python3
import argparse
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
import matplotlib.pyplot as plt

# --- Optional imports used by specific modes ---
try:
    import cartopy.crs as ccrs
except Exception:
    ccrs = None

try:
    import folium
    from PIL import Image
except Exception:
    folium = None
    Image = None


def read_geotiff(path, band=1):
    ds = rasterio.open(path)
    data = ds.read(band)
    nodata = ds.nodata
    if nodata is not None:
        data = np.ma.masked_equal(data, nodata)
    # If nodata wasn't set, try masking NaNs
    if np.issubdtype(data.dtype, np.floating):
        data = np.ma.masked_invalid(data)
    return ds, data


def get_cartopy_crs(rio_crs):
    """Best-effort conversion of rasterio CRS to a Cartopy CRS."""
    if ccrs is None:
        raise RuntimeError("Cartopy isn't installed. Install it to use --mode static.")
    epsg = None
    try:
        epsg = rio_crs.to_epsg()
    except Exception:
        pass
    if epsg is not None:
        return ccrs.epsg(epsg)
    # Fallback: assume geographic lon/lat
    warnings.warn("CRS has no EPSG code; assuming geographic lon/lat (PlateCarree).")
    return ccrs.PlateCarree()


def show_static_map(tif_path, band=1, projection="Robinson", cmap="viridis"):
    if ccrs is None:
        raise RuntimeError("Cartopy isn't installed. Install it to use static mode.")
    ds, data = read_geotiff(tif_path, band=band)

    # Compute data extent in its native CRS
    h, w = data.shape[-2], data.shape[-1]
    left, bottom, right, top = array_bounds(h, w, ds.transform)
    extent = (left, right, bottom, top)

    data_crs = get_cartopy_crs(ds.crs)

    # Pick a nice world projection for the map
    proj_map = {
        "PlateCarree": ccrs.PlateCarree(),
        "Robinson": ccrs.Robinson(),
        "Mercator": ccrs.Mercator(),
        "Mollweide": ccrs.Mollweide(),
        "EqualEarth": ccrs.EqualEarth(),
    }.get(projection, ccrs.Robinson())

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=proj_map)
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    gl = ax.gridlines(draw_labels=False, linewidth=0.2, color="black", alpha=0.25)

    # Stretch contrast a bit using percentiles (ignore masked)
    finite_vals = np.asarray(data.compressed()) if np.ma.isMaskedArray(data) else data[np.isfinite(data)]
    if finite_vals.size:
        vmin, vmax = np.nanpercentile(finite_vals, [2, 98])
        if vmin == vmax:
            vmin, vmax = None, None
    else:
        vmin, vmax = None, None

    im = ax.imshow(
        data,
        origin="upper",
        extent=extent,      # in the data CRS coordinates
        transform=data_crs, # tell Cartopy what CRS the extent/data are in
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        zorder=0,
    )
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05)
    cb.set_label(f"{Path(tif_path).name} — band {band}")
    ax.set_title("GeoTIFF on a world map")

    plt.tight_layout()
    plt.show()
    ds.close()


def save_png_for_folium(data, out_png, cmap="viridis"):
    """Convert a (masked) array to an RGBA PNG using a matplotlib colormap."""
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    arr = np.array(data, dtype=float)
    if np.ma.isMaskedArray(data):
        mask = data.mask
    else:
        mask = ~np.isfinite(arr)

    # Normalize using robust percentiles
    finite = arr[~mask]
    if finite.size:
        vmin, vmax = np.nanpercentile(finite, [2, 98])
        if vmin == vmax:
            vmin, vmax = np.nanmin(finite), np.nanmax(finite)
    else:
        vmin, vmax = 0, 1

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(arr, bytes=True)  # uint8 RGBA
    rgba[mask] = [0, 0, 0, 0]               # transparent for masked

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(out_png)


def show_interactive_map(tif_path, band=1, out_html="map.html", cmap="viridis"):
    if folium is None or Image is None:
        raise RuntimeError("folium and Pillow are required for interactive mode. Install them first.")
    ds, data = read_geotiff(tif_path, band=band)

    # Compute image bounds in WGS84 (lat/lon)
    try:
        bounds_wgs84 = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
    except Exception:
        # If CRS transform fails, assume it's already lon/lat
        warnings.warn("CRS to WGS84 transform failed; assuming data is already lon/lat.")
        bounds_wgs84 = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)

    # Save a colorized PNG (with transparency on NoData)
    out_png = Path(out_html).with_suffix(".png")
    save_png_for_folium(data, out_png, cmap=cmap)

    # Folium wants bounds as [[south, west], [north, east]]
    west, south, east, north = bounds_wgs84
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=2, tiles="CartoDB positron")
    folium.raster_layers.ImageOverlay(
        name=Path(tif_path).name,
        image=str(out_png),
        bounds=[[south, west], [north, east]],
        opacity=0.85,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    print(f"✅ Saved interactive map: {out_html}")


def main():
    p = argparse.ArgumentParser(description="Visualize a GeoTIFF on a world map (static with Cartopy or interactive with Folium).")
    p.add_argument("tif_path", help="Path to your .tif (GeoTIFF) file")
    p.add_argument("--band", type=int, default=1, help="Band index to plot (default: 1)")
    p.add_argument("--mode", choices=["static", "interactive"], default="static", help="Plot mode")
    p.add_argument("--projection", default="Robinson", help="Cartopy projection for static mode (Robinson, PlateCarree, Mercator, Mollweide, EqualEarth)")
    p.add_argument("--cmap", default="viridis", help="Matplotlib colormap name (e.g., viridis, magma)")
    p.add_argument("--html", default="map.html", help="Output HTML filename for interactive mode")
    args = p.parse_args()

    tif_path = Path(args.tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(f"{tif_path} not found")

    if args.mode == "static":
        show_static_map(str(tif_path), band=args.band, projection=args.projection, cmap=args.cmap)
    else:
        show_interactive_map(str(tif_path), band=args.band, out_html=args.html, cmap=args.cmap)


if __name__ == "__main__":
    main()
