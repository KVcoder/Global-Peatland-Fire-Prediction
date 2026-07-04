#!/usr/bin/env python3
"""
ERA5-Land DAILY_AGGR → monthly (or windowed) stacked GeoTIFFs over a mask.

What's new (faster, non-commercial safe):
- Exports to GCS or Drive (no custom task priority anywhere).
- Avoids .clip(); relies on export region (less backend work).
- Optional early Int16 quantization before stacking (--pack-int16-before-stack).
- Optional month slicing (--window-days) to increase parallelism.
- Still supports Drive, split-by-variable, skipEmptyTiles, fileDimensions, COG.

Usage example
-------------
# To Google Drive
python ERA5L_fast_gcs.py \
  --project peatfireforcasting \
  --mask-asset projects/peatfireforcasting/assets/SMAP_L4_peat_mask_latlon_0p1deg_v8-1 \
  --drive-folder ERA5L_monthly \
  --start 2015-04-01 --end 2024-12-31 \
  --max-parallel 4 --use-bounds --progress --monitor \
  --file-dim 8192 --skip-empty-tiles \
  --pack-int16-before-stack --split-by-variable \
  --window-days 10 \
  --select "temperature_2m,dewpoint_temperature_2m,u_component_of_wind_10m,v_component_of_wind_10m,total_precipitation_sum,runoff_sum" \
  --restart-month 2019_06

# To Google Cloud Storage (COG by default)
python ERA5L_fast_gcs.py \
  --project peatfireforcasting \
  --mask-asset projects/peatfireforcasting/assets/SMAP_L4_peat_mask_latlon_0p1deg_v8-1 \
  --gcs-bucket peat-exports \
  --start 2015-03-31 --end 2024-12-31 \
  --max-parallel 6 --use-bounds --progress --monitor \
  --file-dim 8192 --skip-empty-tiles --pack-int16-before-stack --split-by-variable \
  --window-days 10 \
  --select "temperature_2m,dewpoint_temperature_2m,u_component_of_wind_10m,v_component_of_wind_10m,total_precipitation_sum,runoff_sum" \
  --restart-month 2019_06
"""

import argparse
import time
from typing import Tuple, List, Optional, Set, Dict
from datetime import datetime, timezone
import re

import ee
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ----------------- Defaults -----------------
DEFAULT_START = "2015-03-31"
DEFAULT_END   = "2024-12-31"
DEFAULT_SCALE = 11132            # ≈ ERA5-Land DAILY_AGGR native (~0.1°)
DEFAULT_MAX_PARALLEL = 2
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

def active_task_count(prefix: str) -> int:
    n = 0
    for t in ee.batch.Task.list():
        s = t.status()
        desc = s.get("description", "") or ""
        state = s.get("state")
        if desc.startswith(prefix) and state in ("READY", "RUNNING", "SUBMITTED", "QUEUED"):
            n += 1
    return n

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

def months_with_data(daily: ee.ImageCollection) -> List[Tuple[int,int]]:
    """Return sorted [(year, month), ...] that actually exist in the collection (1 client call)."""
    ts_list = daily.aggregate_array('system:time_start').getInfo()  # list of ms since epoch
    months = {(datetime.fromtimestamp(ts/1000, tz=timezone.utc).year,
               datetime.fromtimestamp(ts/1000, tz=timezone.utc).month) for ts in ts_list}
    return sorted(months)

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

def _month_slices(start_ym: ee.Date, window_days: int):
    """Return [(slice_start, slice_end), ...] covering one month. end is exclusive."""
    end_ym = start_ym.advance(1, "month")
    if not window_days or window_days <= 0:
        return [(start_ym, end_ym)]

    # One cheap client call per month to get day count
    n_days = int(end_ym.difference(start_ym, "day").getInfo())

    slices = []
    for offset in range(0, n_days, window_days):
        s_start = start_ym.advance(offset, "day")
        s_end_candidate = s_start.advance(window_days, "day")
        # min(s_end_candidate, end_ym) without extra client calls
        s_end = ee.Date(ee.Algorithms.If(
            s_end_candidate.millis().gt(end_ym.millis()),
            end_ym, s_end_candidate
        ))
        slices.append((s_start, s_end))
    return slices

def export_months(
    daily: ee.ImageCollection,
    region: ee.Geometry,
    start: str,
    end: str,
    scale: int,
    drive_folder: str = None,
    gcs_bucket: str = None,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    progress: bool = False,
    file_dim: Optional[int] = None,
    skip_empty: bool = False,
    split_by_variable: bool = False,
    pack_int16: bool = False,
    pack_int16_before_stack: bool = False,
    window_days: int = 0,
    resume_after: Optional[str] = None,
    restart_month: Optional[str] = None,   # NEW
    ignore_active: bool = False,           # NEW
    desc_prefix: str = DESC_PREFIX,        # NEW
) -> List[Dict]:
    """
    Start monthly (or windowed) exports; return a list of {'desc', 'task'}.

    Resuming:
      - If `resume_after` is provided (e.g., "2019_06_w03" or "2019_06"),
        all slices with base suffix <= that key are skipped.
        Base suffix is "YYYY_MM" or "YYYY_MM_wNN" when window_days>0.

    Restarting a month:
      - If `restart_month` is provided (e.g., "2019_06"),
        months with key < restart_month are skipped entirely,
        but the month == restart_month is exported from scratch
        (all its slices), then it continues normally.
    """
    months = months_with_data(daily)  # only months that exist
    tasks: List[Dict] = []
    default_proj = ee.Image(daily.first()).projection()

    iterator = months
    if progress and tqdm is not None:
        iterator = tqdm(months, desc="Submitting exports", unit="month")

    for y, m in iterator:
        month_key = f"{y}_{m:02d}"

        # Skip earlier months when restarting a specific month
        if restart_month and month_key < restart_month:
            if progress and tqdm is not None:
                tqdm.write(f"Skipping month {month_key} (< restart-month {restart_month})")
            continue

        # Throttle concurrent tasks before submitting new ones (unless ignored)
        if not ignore_active:
            while active_task_count(desc_prefix) >= max_parallel:
                if progress and tqdm is not None and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(active=active_task_count(desc_prefix))
                time.sleep(30)

        start_ym = ee.Date.fromYMD(y, m, 1)
        slices = _month_slices(start_ym, window_days)

        for w_idx, (s_start, s_end) in enumerate(slices, start=1):
            slice_suffix = f"_w{w_idx:02d}" if window_days and window_days > 0 else ""
            base_sfx = f"{month_key}{slice_suffix}"

            # Apply resume logic: skip everything up to and including resume_after
            if resume_after and base_sfx <= resume_after:
                msg = f"Skipping {base_sfx} (<= resume-after {resume_after})"
                if progress and tqdm is not None:
                    tqdm.write(msg)
                else:
                    print(msg)
                continue

            # Build the slice collection (no clipping; export region handles cropping)
            mon = (daily
                   .filterDate(s_start, s_end)
                   .map(lambda i: i.set("system:index",
                                        ee.Date(i.get("system:time_start")).format("YYYYMMdd")))
                   .sort("system:time_start"))

            # Stack once per slice; keep projection explicit
            stacked_all = mon.toBands().setDefaultProjection(default_proj)

            if split_by_variable:
                # Group band names by alias (string after the first underscore)
                names = ee.Image(stacked_all).bandNames().getInfo()
                aliases: Set[str] = set()
                for bn in names:
                    parts = bn.split("_", 1)
                    alias = parts[1] if len(parts) == 2 else bn
                    aliases.add(alias)

                for alias in sorted(aliases):
                    pattern = f".*_{re.escape(alias)}$"
                    v_img = stacked_all.select([pattern])

                    # Only pack here if we did NOT already pack before stacking
                    if pack_int16 and not pack_int16_before_stack:
                        v_img = _pack_int16_image(v_img)

                    desc = f"{desc_prefix}{month_key}{slice_suffix}_{alias}"
                    task = export_image(
                        v_img, desc, region, scale,
                        drive_folder, gcs_bucket, file_dim, skip_empty
                    )
                    tasks.append({"desc": desc, "task": task})
                    if not progress or tqdm is None:
                        print("Started:", desc)
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
                if not progress or tqdm is None:
                    print("Started:", desc)

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
        print(f"Monitoring {len(by_id)} tasks...")

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
def main():
    ap = argparse.ArgumentParser(description="Export ERA5-Land DAILY_AGGR monthly stacks over a raster mask (fast, GCS/Drive).")
    ap.add_argument("--project", required=True, help="GCP project id with the Earth Engine API enabled")
    ap.add_argument("--mask-asset", required=True,
                    help="EE asset id (users/... or projects/...) OR GCS COG (gs://bucket/file.tif)")
    ap.add_argument("--start", default=DEFAULT_START, help="YYYY-MM-DD")
    ap.add_argument("--end", default=DEFAULT_END, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--scale", type=int, default=DEFAULT_SCALE, help="Export scale in meters (native ≈ 11132)")
    ap.add_argument("--drive-folder", default=None, help="Drive folder name (optional)")
    ap.add_argument("--gcs-bucket", default=None, help="If set, export to this GCS bucket instead of Drive")
    ap.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL, help="Max concurrent exports")
    ap.add_argument("--use-bounds", action="store_true",
                    help="Export a simple rectangle around the mask geometry (faster for complex shapes).")
    ap.add_argument("--simplify-m", type=float, default=None,
                    help="Simplify footprint by this many meters (ignored with --use-bounds).")
    ap.add_argument("--select", default=None,
                    help="Comma-separated list of RAW ERA5-Land band IDs to include "
                         "(e.g., temperature_2m,total_precipitation_sum). If omitted, includes all.")
    ap.add_argument("--progress", action="store_true", help="Show tqdm bars while submitting months.")
    ap.add_argument("--monitor", action="store_true",
                    help="Keep running and monitor EE tasks until completion.")
    ap.add_argument("--service-account", default=None, help="Service account email (optional)")
    ap.add_argument("--private-key", default=None, help="Path to service account JSON key (optional)")

    # Speedup toggles
    ap.add_argument("--file-dim", type=int, default=8192,
                    help="Tile size (pixels) for exports; enables skipEmptyTiles & multi-file grids.")
    ap.add_argument("--skip-empty-tiles", action="store_true",
                    help="Don't write fully masked tiles (works with --file-dim).")
    ap.add_argument("--pack-int16", action="store_true",
                    help="Quantize known bands to Int16 with scale factors (~2× smaller) AFTER stacking.")
    ap.add_argument("--pack-int16-before-stack", action="store_true",
                    help="Quantize known bands to Int16 BEFORE stacking (reduces monthly stack IO).")
    ap.add_argument("--split-by-variable", action="store_true",
                    help="Export one file per variable per slice instead of a single giant stack.")
    ap.add_argument("--window-days", type=int, default=0,
                    help="If >0, split each month into N-day windows to increase parallelism.")
    ap.add_argument("--resume-after", default=None,
                help="Skip all months/slices whose suffix (YYYY_MM or YYYY_MM_wNN) "
                     "is <= this value, e.g., 2019_06_w03")

    # NEW controls
    ap.add_argument("--restart-month", default=None,
                    help="YYYY_MM. Skip months < this key, but re-export this month and onward (ignores window slices).")
    ap.add_argument("--ignore-active", action="store_true",
                    help="Do not throttle against existing Earth Engine tasks.")
    ap.add_argument("--desc-prefix", default=DESC_PREFIX,
                    help=f"Description prefix to use (default '{DESC_PREFIX}'). Throttling matches this prefix.")

    args = ap.parse_args()

    if args.resume_after:
        print(f"Resuming after {args.resume_after} ...")
    if args.restart_month:
        print(f"Restarting at month {args.restart_month} (re-export this month and continue) ...")

    init_ee(args.service_account, args.private_key, project=args.project)

    # Preload mask ONCE and reuse
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
        mask01, args.start, args.end,
        selected_raw=selected_raw,
        pack_int16_before_stack=args.pack_int16_before_stack
    )

    if converted:
        renamed = [BAND_SPEC[b][0] for b in converted]
        print("Converted bands (renamed):", ", ".join(renamed))
    if passthrough:
        print("Passthrough bands (unchanged):", ", ".join(passthrough))

    # Fire off exports (auto-started, throttled)
    tasks = export_months(
        daily=daily,
        region=region,
        start=args.start,
        end=args.end,
        scale=args.scale,
        drive_folder=args.drive_folder,
        gcs_bucket=args.gcs_bucket,
        max_parallel=args.max_parallel,
        progress=args.progress,
        file_dim=args.file_dim,
        skip_empty=args.skip_empty_tiles,
        split_by_variable=args.split_by_variable,
        pack_int16=args.pack_int16,
        pack_int16_before_stack=args.pack_int16_before_stack,
        window_days=args.window_days,
        resume_after=args.resume_after,
        restart_month=args.restart_month,      # NEW
        ignore_active=args.ignore_active,      # NEW
        desc_prefix=args.desc_prefix,          # NEW
    )

    print(f"Submitted {len(tasks)} export task(s).")
    
    if args.monitor:
        monitor_tasks(tasks, poll_seconds=30)

    print("All exports submitted." if not args.monitor else "All exports finished monitoring.")

if __name__ == "__main__":
    main()
