#!/usr/bin/env python3
"""
ERA5-Land DAILY_AGGR → single-month stacked GeoTIFF(s) over a mask.

This script exports exactly ONE month per run. You specify the month with
--month YYYY-MM. It can export to Google Drive or Google Cloud Storage.

Key points
----------
- No date ranges. Exactly one month is exported per run.
- Uses export region (no .clip()), with optional --use-bounds rectangle.
- Optional Int16 packing to shrink files (before or after stacking).
- Optional split-by-variable to write one file per variable.
- Supports skipEmptyTiles and fileDimensions for large regions.

Example
-------
# To Google Drive (one file per variable)
python ERA5L_one_month.py \
  --project peatfireforcasting \
  --mask-asset projects/peatfireforcasting/assets/SMAP_L4_peat_mask_latlon_0p1deg_v8-1 \
  --drive-folder ERA5L_monthly \
  --month 2019-06 \
  --file-dim 8192 --skip-empty-tiles \
  --pack-int16-before-stack --split-by-variable --monitor \
  --select "temperature_2m,dewpoint_temperature_2m,u_component_of_wind_10m,v_component_of_wind_10m,total_precipitation_sum,runoff_sum"

# To Google Cloud Storage (COG by default)
python ERA5L_one_month.py \
  --project peatfireforcasting \
  --mask-asset projects/peatfireforcasting/assets/SMAP_L4_peat_mask_latlon_0p1deg_v8-1 \
  --gcs-bucket peat-exports \
  --month 2019-06 \
  --file-dim 8192 --skip-empty-tiles \
  --pack-int16-before-stack --split-by-variable --monitor \
  --select "temperature_2m,dewpoint_temperature_2m,u_component_of_wind_10m,v_component_of_wind_10m,total_precipitation_sum,runoff_sum"
"""

import argparse
import time
from typing import Tuple, List, Optional, Set, Dict
from datetime import datetime, timezone
import calendar
import re

import ee
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ----------------- Defaults -----------------
DEFAULT_SCALE = 11132            # ≈ ERA5-Land DAILY_AGGR native (~0.1°)
DESC_PREFIX = "era5land_monthly_stack_"

# Linear conversions: raw_band -> (new_name, multiply, add)
BAND_SPEC = {
    # Temperatures (K → °C)
    "temperature_2m": ("t2m_C", 1, -273.15),
    "dewpoint_temperature_2m": ("d2m_C", 1, -273.15),
    "skin_temperature": ("tsk_C", 1, -273.15),
    "soil_temperature_level_1": ("stl1_C", 1, -273.15),
    "soil_temperature_level_2": ("stl2_C", 1, -273.15),
    "soil_temperature_level_3": ("stl3_C", 1, -273.15),
    "soil_temperature_level_4": ("stl4_C", 1, -273.15),
    "temperature_of_snow_layer": ("snow_temp_C", 1, -273.15),
    "lake_ice_temperature": ("lake_ice_temp_C", 1, -273.15),
    "lake_mix_layer_temperature": ("lake_mxlyr_temp_C", 1, -273.15),
    "lake_total_layer_temperature": ("lake_total_temp_C", 1, -273.15),
    "lake_bottom_temperature": ("lake_bottom_temp_C", 1, -273.15),

    # Wind (m/s)
    "u_component_of_wind_10m": ("u10_ms", 1, 0),
    "v_component_of_wind_10m": ("v10_ms", 1, 0),

    # Depth/flux sums (m → mm). ECMWF vertical flux sign is downward positive.
    "total_precipitation_sum": ("tp_mm", 1000, 0),
    "snowfall_sum": ("sf_mm", 1000, 0),
    "snowmelt_sum": ("sm_mm", 1000, 0),
    "runoff_sum": ("ro_mm", 1000, 0),
    "surface_runoff_sum": ("sro_mm", 1000, 0),
    "sub_surface_runoff_sum": ("ssro_mm", 1000, 0),
    "snow_depth_water_equivalent": ("sdwe_mm", 1000, 0),
    "skin_reservoir_content": ("src_mm", 1000, 0),  # not a sum but depth-like

    # Total evaporation (upward evap is negative in ECMWF). Flip sign and convert.
    "total_evaporation_sum": ("evap_mm", -1000, 0),

    # LAI (unitless m²/m²; monthly climatology)
    "leaf_area_index_high_vegetation": ("lai_hv", 1, 0),
    "leaf_area_index_low_vegetation": ("lai_lv", 1, 0),
}

# Optional packing to Int16: alias -> (scale_factor_for_storage, offset, dtype, nodata)
PACK = {
  "t2m_C":  (100, 0, "int16", -32768),  # 0.01 °C
  "d2m_C":  (100, 0, "int16", -32768),
  "tsk_C":  (100, 0, "int16", -32768),
  "stl1_C": (100, 0, "int16", -32768),
  "stl2_C": (100, 0, "int16", -32768),
  "stl3_C": (100, 0, "int16", -32768),
  "stl4_C": (100, 0, "int16", -32768),
  "u10_ms": (100, 0, "int16", -32768),  # 0.01 m/s
  "v10_ms": (100, 0, "int16", -32768),
  "tp_mm":  (10,  0, "int16", -32768),  # 0.1 mm
  "sf_mm":  (10,  0, "int16", -32768),
  "sm_mm":  (10,  0, "int16", -32768),
  "ro_mm":  (10,  0, "int16", -32768),
  "sro_mm": (10,  0, "int16", -32768),
  "ssro_mm":(10,  0, "int16", -32768),
  "sdwe_mm":(10,  0, "int16", -32768),
  "src_mm": (10,  0, "int16", -32768),
  "evap_mm":(10,  0, "int16", -32768),
  "lai_hv": (100, 0, "int16", -32768),
  "lai_lv": (100, 0, "int16", -32768),
}

# ----------------- Helpers -----------------
# Task throttle helper (matches by description prefix)
def active_task_count(prefix: str) -> int:
    n = 0
    for t in ee.batch.Task.list():
        try:
            s = t.status()
        except Exception:
            continue
        desc = (s.get("description", "") or "")
        state = s.get("state")
        if desc.startswith(prefix) and state in ("READY", "RUNNING", "SUBMITTED", "QUEUED"):
            n += 1
    return n


def _month_slices(start_ym: ee.Date, window_days: int):
    """Return [(slice_start, slice_end), ...] covering one month. end is exclusive."""
    end_ym = start_ym.advance(1, "month")
    if not window_days or window_days <= 0:
        return [(start_ym, end_ym)]

    n_days = int(end_ym.difference(start_ym, "day").getInfo())
    slices = []
    for offset in range(0, n_days, window_days):
        s_start = start_ym.advance(offset, "day")
        s_end_candidate = s_start.advance(window_days, "day")
        s_end = ee.Date(ee.Algorithms.If(
            s_end_candidate.millis().gt(end_ym.millis()), end_ym, s_end_candidate
        ))
        slices.append((s_start, s_end))
    return slices


def init_ee(service_account: str = None, private_key: str = None, project: str = None):
    if not project:
        raise SystemExit("You must pass --project <gcp-project-id> (Earth Engine API enabled).")
    if service_account and private_key:
        creds = ee.ServiceAccountCredentials(service_account, private_key)
        ee.Initialize(credentials=creds, project=project)
    else:
        ee.Authenticate()
        ee.Initialize(project=project)

def _bounded_rect_from_mask(mask01: ee.Image, scale_for_bounds: int = 5000) -> ee.Geometry:
    world_rect = ee.Geometry.Rectangle([-179.999999, -89.999999, 179.999999, 89.999999], geodesic=False)
    latlon = ee.Image.pixelLonLat().updateMask(mask01)
    stats = latlon.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=world_rect,
        scale=scale_for_bounds,
        bestEffort=True,
        maxPixels=1e13,
        tileScale=4,
    )
    lon_min = ee.Number(stats.get('longitude_min'))
    lat_min = ee.Number(stats.get('latitude_min'))
    lon_max = ee.Number(stats.get('longitude_max'))
    lat_max = ee.Number(stats.get('latitude_max'))
    return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], geodesic=False)

def load_mask_and_region(mask_arg: str, use_bounds: bool, simplify_m: Optional[float] = None):
    """
    Returns:
      mask01 : ee.Image  -> binary mask (1 where mask>0), self-masked
      region : ee.Geometry -> bounded, non-geodesic geometry (rectangle if use_bounds=True)
    """
    if mask_arg.startswith("gs://"):
        src = ee.Image.loadGeoTIFF(mask_arg)
    elif mask_arg.startswith("users/") or mask_arg.startswith("projects/"):
        src = ee.Image(mask_arg)
    else:
        raise SystemExit("Mask must be an EE asset id (users/... or projects/...) or a GCS COG (gs://...).")
    mask01 = src.gt(0).selfMask()

    # Fail fast if mask empty
    world_rect = ee.Geometry.Rectangle([-179.999999, -89.999999, 179.999999, 89.999999], geodesic=False)
    try:
        cnt = (mask01.rename("mask")
               .reduceRegion(ee.Reducer.count(), world_rect, scale=10000,
                             bestEffort=True, maxPixels=1e13, tileScale=4)
               .get("mask").getInfo())
    except Exception:
        cnt = None
    if not cnt:
        raise SystemExit("Mask has zero valid pixels (all 0/NoData). Check --mask-asset and its values.")

    rect = _bounded_rect_from_mask(mask01, scale_for_bounds=5000)
    if use_bounds:
        region = rect
    else:
        footprint = ee.Geometry(mask01.geometry(ee.ErrorMargin(1))).buffer(0, ee.ErrorMargin(1))
        region = footprint.intersection(rect, ee.ErrorMargin(1))
        if simplify_m and simplify_m > 0:
            region = region.simplify(simplify_m)

    return mask01, region

def parse_select_arg(select_arg: Optional[str]) -> Optional[Set[str]]:
    if not select_arg:
        return None
    items = [x.strip() for x in select_arg.replace("\n", ",").split(",") if x.strip()]
    return set(items) if items else None

def build_daily(mask01: ee.Image, start: str, end: str,
                selected_raw: Optional[Set[str]] = None,
                pack_int16_before_stack: bool = False
               ) -> Tuple[ee.ImageCollection, List[str], List[str]]:
    # Make end inclusive by advancing one day.
    end_plus_1 = ee.Date(end).advance(1, "day")
    src = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
           .filterDate(start, end_plus_1)
           .map(lambda img: img.updateMask(mask01)))

    # Guard: bail early if empty.
    if src.size().getInfo() == 0:
        raise SystemExit("No ERA5-Land images found in the requested date range.")

    server_bands = ee.Image(src.first()).bandNames().getInfo()
    spec_keys = set(BAND_SPEC.keys())

    if selected_raw:
        selected_raw = {b for b in selected_raw if b in server_bands}
        if not selected_raw:
            raise SystemExit("None of the requested --select bands exist in ERA5-Land DAILY_AGGR for this date range.")
        passthrough = [b for b in selected_raw if b not in spec_keys]
        converted   = [b for b in selected_raw if b in spec_keys]
    else:
        passthrough = [b for b in server_bands if b not in spec_keys]
        converted   = [b for b in server_bands if b in spec_keys]

    if not passthrough and not converted:
        raise SystemExit("No bands selected after filtering; check --select.")

    def _convert(img):
        pieces = []
        if passthrough:
            pieces.append(img.select(passthrough))
        for b in converted:
            new_name, mul, add = BAND_SPEC[b]
            tmp = img.select(b).multiply(mul).add(add).rename(new_name)
            if pack_int16_before_stack and new_name in PACK:
                s, off, _, _ = PACK[new_name]
                tmp = tmp.multiply(s).add(off).round().toInt16().rename(new_name)
            pieces.append(tmp)
        return ee.Image.cat(pieces).copyProperties(img, ["system:time_start"])

    daily = src.map(_convert)
    return daily, passthrough, converted


def _pack_int16_image(stacked: ee.Image) -> ee.Image:
    """
    Quantize bands like 'YYYYMMdd_alias' using PACK[alias].
    Leaves bands not in PACK unmodified.
    """
    band_names = stacked.bandNames().getInfo()  # one client call
    out_imgs = []
    for b in band_names:
        # Split "YYYYMMdd_alias" into alias; date is 8 chars + underscore.
        try:
            alias = b.split("_", 1)[1]
        except Exception:
            alias = b
        if alias in PACK:
            s, off, _, nodata = PACK[alias]
            q = (stacked.select([b]).multiply(s).add(off).round().toInt16()
                 .rename(b)
                 .updateMask(stacked.select([b]).mask()))
            # Optional per-band metadata
            q = q.set({f"{b}:scale": 1.0/s, f"{b}:offset": -off/float(s), f"{b}:nodata": nodata})
            out_imgs.append(q)
        else:
            out_imgs.append(stacked.select([b]))
    return ee.Image.cat(out_imgs)


def export_image(
    image: ee.Image,
    desc: str,
    region: ee.Geometry,
    scale: int,
    drive_folder: Optional[str],
    gcs_bucket: Optional[str],
    file_dim: Optional[int],
    skip_empty: bool,
):
    params = dict(
        image=image,
        description=desc,
        region=region,
        scale=scale,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    if file_dim:
        params["fileDimensions"] = file_dim
    if skip_empty:
        params["skipEmptyTiles"] = True

    if gcs_bucket:
        params.update(bucket=gcs_bucket, fileNamePrefix=desc)
        task = ee.batch.Export.image.toCloudStorage(**params)
    else:
        if drive_folder:
            params.update(folder=drive_folder, fileNamePrefix=desc)
        task = ee.batch.Export.image.toDrive(**params)
    task.start()
    return task


def export_one_month(
    daily: ee.ImageCollection,
    region: ee.Geometry,
    month_key: str,
    scale: int,
    drive_folder: str = None,
    gcs_bucket: str = None,
    file_dim: Optional[int] = None,
    skip_empty: bool = False,
    split_by_variable: bool = False,
    pack_int16: bool = False,
    pack_int16_before_stack: bool = False,
    desc_prefix: str = DESC_PREFIX,
    window_days: int = 0,
    max_parallel: int = 2,
    progress: bool = False,
) -> List[Dict]:
    """Stack and export exactly one month, with optional window slicing.
    Returns a list of tasks (1+ per slice, and per variable if split).
    """
    if daily.size().getInfo() == 0:
        raise SystemExit("No images in the specified month for ERA5-Land DAILY_AGGR.")

    # Build slices for the month
    y, m = [int(x) for x in month_key.split("_")]
    start_ym = ee.Date.fromYMD(y, m, 1)
    slices = _month_slices(start_ym, window_days)

    tasks: List[Dict] = []
    default_proj = ee.Image(daily.first()).projection()

    iterable = enumerate(slices, start=1)
    if progress and tqdm is not None:
        iterable = tqdm(list(iterable), desc=f"Submitting {month_key}", unit="slice")

    for w_idx, (s_start, s_end) in iterable:
        # Throttle concurrent tasks across all previously submitted slices
        while active_task_count(desc_prefix) >= max_parallel:
            time.sleep(30)

        mon = (daily
               .filterDate(s_start, s_end)
               .map(lambda i: i.set("system:index", ee.Date(i.get("system:time_start")).format("YYYYMMdd")))
               .sort("system:time_start"))

        if mon.size().getInfo() == 0:
            # Nothing in this slice; skip quietly
            continue

        stacked_all = mon.toBands().setDefaultProjection(default_proj)
        slice_suffix = f"_w{w_idx:02d}" if window_days and window_days > 0 else ""

        if split_by_variable:
            names = ee.Image(stacked_all).bandNames().getInfo()
            aliases: Set[str] = set()
            for bn in names:
                parts = bn.split("_", 1)
                alias = parts[1] if len(parts) == 2 else bn
                aliases.add(alias)

            for alias in sorted(aliases):
                pattern = f".*_{re.escape(alias)}$"
                v_img = stacked_all.select([pattern])
                if pack_int16 and not pack_int16_before_stack:
                    v_img = _pack_int16_image(v_img)
                desc = f"{desc_prefix}{month_key}{slice_suffix}_{alias}"
                task = export_image(
                    v_img, desc, region, scale,
                    drive_folder, gcs_bucket, file_dim, skip_empty
                )
                tasks.append({"desc": desc, "task": task})
        else:
            img = stacked_all
            if pack_int16 and not pack_int16_before_stack:
                img = _pack_int16_image(img)
            desc = f"{desc_prefix}{month_key}{slice_suffix}"
            task = export_image(
                img, desc, region, scale,
                drive_folder, gcs_bucket, file_dim, skip_empty
            )
            tasks.append({"desc": desc, "task": task})

    return tasks


def monitor_tasks(tasks: List[Dict], poll_seconds: int = 30):
    if not tasks:
        print("No tasks to monitor.")
        return
    by_id: Dict[str, Dict] = {}
    for t in tasks:
        try:
            tid = t["task"].status().get("id") or t["desc"]
        except Exception:
            tid = t["desc"]
        by_id[tid] = {"desc": t["desc"], "task": t["task"], "done": False}

    completed_states = {"COMPLETED", "FAILED", "CANCELLED"}
    done_count = 0
    bar = tqdm(total=len(by_id), desc="Completed exports", unit="task") if tqdm is not None else None
    if not bar:
        print(f"Monitoring {len(by_id)} task(s)...")

    while done_count < len(by_id):
        for tid, info in list(by_id.items()):
            if info["done"]:
                continue
            try:
                s = info["task"].status()
                state = s.get("state", "UNKNOWN")
            except Exception:
                state = "UNKNOWN"
            if state in completed_states:
                info["done"] = True
                done_count += 1
                if bar:
                    bar.update(1)
                    bar.set_postfix_str(f"last={state}")
                else:
                    print(f"[{state}] {info['desc']}")
        time.sleep(poll_seconds)
    if bar:
        bar.close()

    failures = []
    for info in by_id.values():
        try:
            st = info["task"].status()
            if st.get("state") == "FAILED":
                failures.append((info["desc"], st.get("error_message", "UNKNOWN ERROR")))
        except Exception:
            failures.append((info["desc"], "STATUS ERROR"))
    if failures:
        print("\nSome tasks failed:")
        for desc, msg in failures:
            print(f"- {desc}: {msg}")

# ----------------- Main -----------------
def month_bounds(month_str: str) -> Tuple[str, str, str]:
    """Return (start_date, end_date_inclusive, month_key) for YYYY-MM."""
    try:
        dt = datetime.strptime(month_str, "%Y-%m")
    except ValueError:
        raise SystemExit("--month must be in the format YYYY-MM, e.g., 2019-06")
    year, month = dt.year, dt.month
    start_date = f"{year:04d}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    end_date = f"{year:04d}-{month:02d}-{last_day:02d}"
    month_key = f"{year:04d}_{month:02d}"
    return start_date, end_date, month_key


def main():
    ap = argparse.ArgumentParser(description="Export ONE MONTH of ERA5-Land DAILY_AGGR stacks over a raster mask.")
    ap.add_argument("--project", required=True, help="GCP project id with the Earth Engine API enabled")
    ap.add_argument("--mask-asset", required=True,
                    help="EE asset id (users/... or projects/...) OR GCS COG (gs://bucket/file.tif)")
    ap.add_argument("--month", required=True, help="Month to export in YYYY-MM (e.g., 2019-06)")
    ap.add_argument("--scale", type=int, default=DEFAULT_SCALE, help="Export scale in meters (native ≈ 11132)")
    ap.add_argument("--drive-folder", default=None, help="Drive folder name (optional)")
    ap.add_argument("--gcs-bucket", default=None, help="If set, export to this GCS bucket instead of Drive")
    ap.add_argument("--use-bounds", action="store_true",
                    help="Export a simple rectangle around the mask geometry (faster for complex shapes).")
    ap.add_argument("--simplify-m", type=float, default=None,
                    help="Simplify footprint by this many meters (ignored with --use-bounds).")
    ap.add_argument("--select", default=None,
                    help="Comma-separated list of RAW ERA5-Land band IDs to include "
                         "(e.g., temperature_2m,total_precipitation_sum). If omitted, includes all.")
    ap.add_argument("--monitor", action="store_true",
                    help="Keep running and monitor the EE export task(s) until completion.")
    ap.add_argument("--service-account", default=None, help="Service account email (optional)")
    ap.add_argument("--private-key", default=None, help="Path to service account JSON key (optional)")

    # Size/IO toggles
    ap.add_argument("--file-dim", type=int, default=8192,
                    help="Tile size (pixels) for exports; enables skipEmptyTiles & multi-file grids.")
    ap.add_argument("--skip-empty-tiles", action="store_true",
                    help="Don't write fully masked tiles (works with --file-dim).")

    # Packing/splitting toggles
    ap.add_argument("--pack-int16", action="store_true",
                    help="Quantize known bands to Int16 with scale factors (~2× smaller) AFTER stacking.")
    ap.add_argument("--pack-int16-before-stack", action="store_true",
                    help="Quantize known bands to Int16 BEFORE stacking (reduces monthly stack IO).")
    ap.add_argument("--split-by-variable", action="store_true",
                    help="Export one file per variable instead of a single giant stack.")

    # Naming & slicing/throttle
    ap.add_argument("--desc-prefix", default=DESC_PREFIX,
                    help=f"Description prefix to use (default '{DESC_PREFIX}').")
    ap.add_argument("--window-days", type=int, default=0,
                    help="If >0, split the month into N-day windows (emits multiple files with _wNN suffix).")
    ap.add_argument("--max-parallel", type=int, default=2,
                    help="Max concurrent EE tasks to allow while submitting slices (throttle).")
    ap.add_argument("--progress", action="store_true",
                    help="Show a tqdm progress bar while submitting slices.")

    args = ap.parse_args()

    # Compute the month's bounds
    start_date, end_date, month_key = month_bounds(args.month)
    print(f"Exporting month {month_key} → {start_date} … {end_date}")

    # Initialize EE and prepare region
    init_ee(args.service_account, args.private_key, project=args.project)
    mask01, region = load_mask_and_region(args.mask_asset, args.use_bounds, args.simplify_m)

    # Strict sanity check: throws if unbounded/invalid
    try:
        _ = ee.Geometry(region).bounds(ee.ErrorMargin(1)).area(ee.ErrorMargin(1)).getInfo()
    except Exception as e:
        raise SystemExit(f"Export region is invalid/unbounded: {e}")

    # Parse selection
    selected_raw = parse_select_arg(args.select)

    # Build daily collection with conversions (mask applied once, end inclusive)
    daily, passthrough, converted = build_daily(
        mask01, start_date, end_date,
        selected_raw=selected_raw,
        pack_int16_before_stack=args.pack_int16_before_stack
    )

    if converted:
        renamed = [BAND_SPEC[b][0] for b in converted]
        print("Converted bands (renamed):", ", ".join(renamed))
    if passthrough:
        print("Passthrough bands (unchanged):", ", ".join(passthrough))

    # Fire off exactly one month's export (auto-started)
    tasks = export_one_month(
        daily=daily,
        region=region,
        month_key=month_key,
        scale=args.scale,
        drive_folder=args.drive_folder,
        gcs_bucket=args.gcs_bucket,
        file_dim=args.file_dim,
        skip_empty=args.skip_empty_tiles,
        split_by_variable=args.split_by_variable,
        pack_int16=args.pack_int16,
        pack_int16_before_stack=args.pack_int16_before_stack,
        desc_prefix=args.desc_prefix,
        window_days=args.window_days,
        max_parallel=args.max_parallel,
        progress=args.progress,
    )

    print(f"Submitted {len(tasks)} export task(s) for {month_key}.")

    if args.monitor:
        monitor_tasks(tasks, poll_seconds=30)

    print("Export submitted." if not args.monitor else "Finished monitoring.")


if __name__ == "__main__":
    main()
