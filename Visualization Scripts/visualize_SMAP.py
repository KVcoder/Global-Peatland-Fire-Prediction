#!/usr/bin/env python3
"""
Interactive SMAP Water Table Depth (WTD) Viewer for your exported GeoTIFF stacks.

What it does
------------
• Scans a folder for your merged yearly GeoTIFFs (from your stitch script).
• Lets you pick a file (year), choose a date (band), and visualize it on a web map.
• Click anywhere on the map to get a full time series for that location across the selected file.
• Adjust color scaling (percentile stretch) and transparency.

Assumptions
-----------
• Each GeoTIFF is a multi-band stack where band order corresponds to consecutive days
  (e.g., yearly stacks with daily bands). Band count should match the number of days in that year
  (clamped by data availability, e.g., 2015 starts on 2015‑03‑31).
• CRS is EPSG:4326 at ~0.1° resolution.
• Filenames contain the year label like "..._2016.tif" (best effort fallback if band descriptions
  don't contain dates).

Dependencies
------------
    pip install streamlit leafmap localtileserver rioxarray rasterio xarray pandas numpy streamlit-folium

Run
---
    streamlit run smap_wtd_viewer.py

Then, in the sidebar, set the input folder that contains your merged .tif files.

"""
from __future__ import annotations
import os
import re
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import rioxarray as rxr
import xarray as xr

import streamlit as st
from streamlit_folium import st_folium
import folium
import leafmap.foliumap as leafmap

DATASET_START = datetime(2015, 3, 31)

YEAR_RE = re.compile(r"_(?P<year>\d{4})(?=\.tif$)", re.IGNORECASE)
DATE8_RE = re.compile(r"^(?P<d>\d{8})$")


def find_tifs(folder: Path) -> List[Path]:
    exts = (".tif", ".tiff")
    return sorted([p for p in folder.glob("*.tif*") if p.suffix.lower() in exts])


def parse_year_from_filename(path: Path) -> Optional[int]:
    m = YEAR_RE.search(path.name)
    if m:
        return int(m.group("year"))
    return None


def get_band_dates(path: Path) -> List[pd.Timestamp]:
    """Try to recover per-band dates, first from band descriptions, then by filename-year fallback."""
    with rasterio.open(path) as src:
        count = src.count
        descs = src.descriptions or ()
        # Attempt 1: band descriptions are YYYYMMDD
        parsed: List[pd.Timestamp] = []
        if descs and len(descs) == count:
            ok = True
            for d in descs:
                if d is None:
                    ok = False
                    break
                d = d.strip()
                m = DATE8_RE.match(d)
                if not m:
                    ok = False
                    break
                parsed.append(pd.to_datetime(m.group("d"), format="%Y%m%d"))
            if ok and len(parsed) == count:
                return parsed
        # Attempt 2: fallback by year inferred from filename
        yr = parse_year_from_filename(path)
        if yr is None:
            # As a last resort, assume the first band's date is dataset start (not ideal)
            start = DATASET_START
        else:
            start = datetime(yr, 1, 1)
            if start < DATASET_START:
                start = max(start, DATASET_START)
        # Build consecutive daily dates for band count
        return list(pd.date_range(start=start, periods=count, freq="D"))


def robust_percentiles(path: Path, band: int, lo: float = 2.0, hi: float = 98.0) -> Tuple[float, float]:
    """Fast-ish percentile stretch from a downsampled read."""
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        # target small shape while preserving aspect
        target_h = min(1024, h)
        target_w = int(w * (target_h / h))
        arr = src.read(band, out_shape=(target_h, target_w), resampling=Resampling.bilinear)
        nodata = src.nodata
    data = arr.astype("float32")
    if nodata is not None:
        data[data == nodata] = np.nan
    data = data[np.isfinite(data)]
    if data.size == 0:
        return (0.0, 1.0)
    vmin = float(np.nanpercentile(data, lo))
    vmax = float(np.nanpercentile(data, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return (float(np.nanmin(data)), float(np.nanmax(data)))
    return vmin, vmax


def sample_timeseries(tif_path: Path, lat: float, lon: float) -> pd.DataFrame:
    """Return a DataFrame with columns ['date','wtd_m'] by sampling all bands at one location."""
    da = rxr.open_rasterio(tif_path, masked=True)  # dims: band, y, x
    # Select nearest pixel for the given lon/lat (x=lon, y=lat in EPSG:4326)
    series = da.sel(x=lon, y=lat, method="nearest").values  # shape: (bands,)
    dates = get_band_dates(tif_path)
    # Guard for length mismatches
    n = min(len(series), len(dates))
    df = pd.DataFrame({"date": pd.to_datetime(dates[:n]), "wtd_m": np.asarray(series[:n], dtype=float)})
    # Mask nodata
    nodata = getattr(da, "rio").nodatavals[0] if getattr(da, "rio", None) else None
    if nodata is not None:
        df.loc[df["wtd_m"] == nodata, "wtd_m"] = np.nan
    return df


# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="SMAP WTD Viewer", layout="wide")
st.title("SMAP Water Table Depth (WTD) Viewer")

with st.sidebar:
    st.header("Settings")
    folder_str = st.text_input("Folder with merged .tif files", value=str(Path.cwd()))
    folder = Path(folder_str).expanduser().resolve()
    refresh = st.button("Scan folder")

# Scan immediately or when button clicked
if folder.exists() and folder.is_dir():
    tifs = find_tifs(folder)
else:
    tifs = []

if not tifs:
    st.info("Point the sidebar to the folder that contains your merged yearly GeoTIFFs (one .tif per mosaic).")
    st.stop()

# Build selection with years when possible
labels = []
for p in tifs:
    yr = parse_year_from_filename(p)
    labels.append(f"{p.name}" + (f"  (year {yr})" if yr else ""))

sel = st.selectbox("Select a mosaic (usually a year)", options=list(range(len(tifs))), format_func=lambda i: labels[i])
sel_path = tifs[sel]

# Determine per-band dates
band_dates = get_band_dates(sel_path)
nbands = len(band_dates)

col1, col2, col3 = st.columns([2,1,1])
with col1:
    # Date slider
    idx = st.slider("Date (band)", min_value=0, max_value=nbands-1, value=0, format="")
    selected_date = pd.to_datetime(band_dates[idx]).date()
    st.write(f"**Selected date:** {selected_date}")
with col2:
    opacity = st.slider("Layer opacity", 0.0, 1.0, 0.85, step=0.05)
with col3:
    pct_lo, pct_hi = st.slider("Percentile stretch", 0.0, 100.0, (2.0, 98.0), step=0.5)

# Compute stretch for the selected band
vmin, vmax = robust_percentiles(sel_path, idx+1, pct_lo, pct_hi)

# Map
m = leafmap.Map(center=[10, 0], zoom=2, draw_control=False, measure_control=False, fullscreen_control=True)
try:
    # Note: leafmap's add_raster uses localtileserver under the hood.
    m.add_raster(
        str(sel_path),
        indexes=idx+1,
        vmin=vmin,
        vmax=vmax,
        colormap="viridis",
        opacity=opacity,
        layer_name=f"WTD {selected_date}",
    )
except Exception as e:
    st.error(f"Failed to add raster: {e}")

m.add_layer_control()

st_map_event = st_folium(m, height=650)

# Click to sample time series
if st_map_event and isinstance(st_map_event, dict) and st_map_event.get("last_clicked"):
    lat = float(st_map_event["last_clicked"]["lat"])  # folium uses lat/lon
    lon = float(st_map_event["last_clicked"]["lng"])  # a.k.a. lng
    st.markdown(f"**Last clicked:** lat={lat:.4f}, lon={lon:.4f}")

    with st.spinner("Sampling time series at clicked point..."):
        df = sample_timeseries(sel_path, lat, lon)
        if df.empty:
            st.warning("No valid data at this location (likely outside peat mask). Try another point.")
        else:
            st.line_chart(df.set_index("date")["wtd_m"], height=250)
            st.caption("WTD in meters (positive = deeper water table). Nodata values are omitted.")
else:
    st.caption("Click the map to plot a full time series for that location.")

with st.expander("File & metadata"):
    with rasterio.open(sel_path) as src:
        st.write({
            "file": sel_path.name,
            "bands": src.count,
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "nodata": src.nodata,
        })
    st.write("First & last dates in this stack:", str(pd.to_datetime(band_dates[0]).date()), "→", str(pd.to_datetime(band_dates[-1]).date()))

st.success("Ready. Adjust the date slider, then click the map to explore time series.")
