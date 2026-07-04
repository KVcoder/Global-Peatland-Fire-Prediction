#!/usr/bin/env python3
"""
Compute daily Canadian Fire Weather Index (FWI) rasters from a folder of
single-day GeoTIFFs that contain these five bands:
  - d2m_C
  - t2m_C
  - tp_mm
  - u10_ms
  - v10_ms

Multicore note
--------------
The FWI system is sequential across days (FFMC/DMC/DC carry over from the
previous day), so days cannot be processed in parallel. However, within a
given day each pixel/chunk is independent once the previous-day state is
known. This script can therefore parallelize *row chunks within each day*
using multiple CPU cores via ThreadPoolExecutor.

The implementation follows the standard Canadian Forest Fire Weather Index
(CFFDRS FWI) equations as documented by Van Wagner & Pickett (1985),
Van Wagner (1987), and mirrored in the open-source `cffdrs` reference
implementation.

Important notes
---------------
1) The FWI system is sequential: FFMC, DMC, and DC carry from one day to the
   next. This script therefore sorts files chronologically and requires a
   continuous daily sequence.
2) Standard documented start-up values are used by default:
      FFMC=85, DMC=6, DC=15
   You can override them from the CLI.
3) Input weather must represent local-noon conditions, with precipitation equal
   to the previous 24 h total ending at local noon.
4) Output is written only where the reference valid mask is true. By default,
   the valid mask is taken from the first input day and every subsequent day is
   required to have the exact same valid footprint.

Dependencies
------------
- numpy
- rasterio
- tqdm  (optional but recommended)
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

NODATA_DEFAULT = -9999.0
DATE_RE = re.compile(r"(\d{8})")

# Band names expected in the input GeoTIFF descriptions.
REQUIRED_BANDS = {
    "d2m_c": "d2m_C",
    "t2m_c": "t2m_C",
    "tp_mm": "tp_mm",
    "u10_ms": "u10_ms",
    "v10_ms": "v10_ms",
}

# Constants used by the daily FFMC equations in cffdrs.
FFMC_COEFFICIENT = 250.0 * 59.5 / 101.0


@dataclass(frozen=True)
class DailyFile:
    dt: date
    path: Path


# -----------------------------
# Helpers
# -----------------------------
def normalize_band_name(name: str | None) -> str:
    if name is None:
        return ""
    return re.sub(r"\s+", "", name.strip().lower())


def parse_date_from_name(path: Path) -> date:
    m = DATE_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not find YYYYMMDD date in filename: {path.name}")
    return date.fromisoformat(f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}")


def list_daily_files(folder: Path, pattern: str) -> List[DailyFile]:
    files = [DailyFile(parse_date_from_name(p), p) for p in folder.glob(pattern)]
    if not files:
        raise FileNotFoundError(f"No files found in {folder} with pattern {pattern!r}")
    files.sort(key=lambda x: x.dt)
    return files


def find_date_gaps(files: List[DailyFile]) -> List[Tuple[DailyFile, DailyFile, int]]:
    gaps: List[Tuple[DailyFile, DailyFile, int]] = []
    for prev, cur in zip(files[:-1], files[1:]):
        delta = (cur.dt - prev.dt).days
        if delta != 1:
            gaps.append((prev, cur, delta - 1))
    return gaps


def list_missing_dates(files: List[DailyFile]) -> List[date]:
    missing: List[date] = []
    for prev, cur in zip(files[:-1], files[1:]):
        d = prev.dt + timedelta(days=1)
        while d < cur.dt:
            missing.append(d)
            d += timedelta(days=1)
    return missing


def write_missing_dates_report(path: Path, missing_dates: List[date], gaps: List[Tuple[DailyFile, DailyFile, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        if not missing_dates:
            f.write('No missing dates found.\n')
            return

        f.write('# Missing daily input dates for FWI run\n')
        f.write(f'# Total missing dates: {len(missing_dates)}\n')
        f.write(f'# Total gap segments: {len(gaps)}\n\n')

        f.write('# Gap summary\n')
        for prev, cur, missing_n in gaps:
            f.write(
                f'# after {prev.path.name} ({prev.dt.isoformat()}) '
                f'before {cur.path.name} ({cur.dt.isoformat()}) -> missing {missing_n} day(s)\n'
            )

        f.write('\n# Missing dates (YYYY-MM-DD)\n')
        for d in missing_dates:
            f.write(d.isoformat() + '\n')


def ensure_daily_continuity(files: List[DailyFile]) -> None:
    gaps = find_date_gaps(files)
    if gaps:
        parts = []
        for prev, cur, missing_n in gaps[:10]:
            parts.append(
                f"Gap between {prev.path.name} ({prev.dt}) and {cur.path.name} ({cur.dt}); missing {missing_n} day(s)"
            )
        more = "" if len(gaps) <= 10 else f" ... plus {len(gaps)-10} more gap(s)"
        raise ValueError("Input dates are not continuous. " + " | ".join(parts) + more)


def band_map_from_descriptions(src: rasterio.io.DatasetReader) -> Dict[str, int]:
    descs = list(src.descriptions or [])
    descs += [None] * max(0, src.count - len(descs))
    mapping: Dict[str, int] = {}
    for i in range(1, src.count + 1):
        key = normalize_band_name(descs[i - 1])
        if key:
            mapping[key] = i

    missing = [k for k in REQUIRED_BANDS if k not in mapping]
    if missing:
        if src.count >= 5:
            fallback = {
                "d2m_c": 1,
                "t2m_c": 2,
                "tp_mm": 3,
                "u10_ms": 4,
                "v10_ms": 5,
            }
            if all(k not in mapping for k in REQUIRED_BANDS):
                return fallback
        wanted = ", ".join(REQUIRED_BANDS.values())
        found = ", ".join([d for d in descs if d]) or "<none>"
        raise ValueError(
            f"Could not map all required bands. Need: {wanted}. Found descriptions: {found}"
        )
    return {k: mapping[k] for k in REQUIRED_BANDS}


def row_center_latitudes(transform: rasterio.Affine, height: int) -> np.ndarray:
    rows = np.arange(height, dtype=np.float64)
    return (transform.f + (rows + 0.5) * transform.e).astype(np.float32)


def saturation_vapor_pressure_pa(t_c: np.ndarray) -> np.ndarray:
    t_k = t_c.astype(np.float64) + 273.15
    return 611.21 * np.exp(17.502 * (t_k - 273.16) / (t_k - 32.19))


def rh_from_t_td(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    rh = 100.0 * saturation_vapor_pressure_pa(td_c) / saturation_vapor_pressure_pa(t_c)
    return np.clip(rh, 0.0, 100.0).astype(np.float32)


# -----------------------------
# FWI equations (array form)
# -----------------------------

def fine_fuel_moisture_code(ffmc_yda, temp, rh, ws_kmh, prec):
    ffmc_yda = ffmc_yda.astype(np.float64, copy=False)
    temp = temp.astype(np.float64, copy=False)
    rh = rh.astype(np.float64, copy=False)
    ws_kmh = ws_kmh.astype(np.float64, copy=False)
    prec = prec.astype(np.float64, copy=False)

    wmo = FFMC_COEFFICIENT * (101.0 - ffmc_yda) / (59.5 + ffmc_yda)

    rain_mask = prec > 0.5
    if np.any(rain_mask):
        ra = prec[rain_mask] - 0.5
        wmo_r = wmo[rain_mask]
        rain_term = 42.5 * ra * np.exp(-100.0 / (251.0 - wmo_r)) * (1.0 - np.exp(-6.93 / ra))
        heavy_extra = 0.0015 * np.square(np.maximum(wmo_r - 150.0, 0.0)) * np.sqrt(ra)
        wmo_rain = wmo_r + rain_term + np.where(wmo_r > 150.0, heavy_extra, 0.0)
        wmo[rain_mask] = np.minimum(wmo_rain, 250.0)
    else:
        wmo = np.minimum(wmo, 250.0)

    ed = (
        0.942 * np.power(rh, 0.679)
        + 11.0 * np.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
    )
    ew = (
        0.618 * np.power(rh, 0.753)
        + 10.0 * np.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
    )

    dry_mask = (wmo < ed) & (wmo < ew)
    z = np.zeros_like(wmo)
    z = np.where(
        dry_mask,
        0.424 * (1.0 - np.power((100.0 - rh) / 100.0, 1.7))
        + 0.0694 * np.sqrt(ws_kmh) * (1.0 - np.power((100.0 - rh) / 100.0, 8.0)),
        z,
    )
    x = z * 0.581 * np.exp(0.0365 * temp)
    wm = np.where(dry_mask, ew - (ew - wmo) / np.power(10.0, x), wmo)

    wet_mask = wmo > ed
    z = np.where(
        wet_mask,
        0.424 * (1.0 - np.power(rh / 100.0, 1.7))
        + 0.0694 * np.sqrt(ws_kmh) * (1.0 - np.power(rh / 100.0, 8.0)),
        z,
    )
    x = z * 0.581 * np.exp(0.0365 * temp)
    wm = np.where(wet_mask, ed + (wmo - ed) / np.power(10.0, x), wm)

    ffmc1 = 59.5 * (250.0 - wm) / (FFMC_COEFFICIENT + wm)
    ffmc1 = np.clip(ffmc1, 0.0, 101.0)
    return ffmc1.astype(np.float32)


def duff_moisture_code(dmc_yda, temp, rh, prec, lat, mon, lat_adjust=True):
    dmc_yda = dmc_yda.astype(np.float64, copy=False)
    temp = temp.astype(np.float64, copy=False)
    rh = rh.astype(np.float64, copy=False)
    prec = prec.astype(np.float64, copy=False)
    lat = lat.astype(np.float64, copy=False)

    ell01 = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0], dtype=np.float64)
    ell02 = np.array([7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8], dtype=np.float64)
    ell03 = np.array([10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2], dtype=np.float64)
    ell04 = np.array([11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8], dtype=np.float64)

    temp = np.maximum(temp, -1.1)
    rk = 1.894 * (temp + 1.1) * (100.0 - rh) * ell01[mon - 1] * 1.0e-4

    if lat_adjust:
        rk = np.where((lat <= 30.0) & (lat > 10.0), 1.894 * (temp + 1.1) * (100.0 - rh) * ell02[mon - 1] * 1.0e-4, rk)
        rk = np.where((lat <= -10.0) & (lat > -30.0), 1.894 * (temp + 1.1) * (100.0 - rh) * ell03[mon - 1] * 1.0e-4, rk)
        rk = np.where((lat <= -30.0) & (lat >= -90.0), 1.894 * (temp + 1.1) * (100.0 - rh) * ell04[mon - 1] * 1.0e-4, rk)
        rk = np.where((lat <= 10.0) & (lat > -10.0), 1.894 * (temp + 1.1) * (100.0 - rh) * 9.0 * 1.0e-4, rk)

    pr = dmc_yda.copy()
    rain_mask = prec > 1.5
    if np.any(rain_mask):
        ra = prec[rain_mask]
        rw = 0.92 * ra - 1.27
        d_prev = dmc_yda[rain_mask]
        wmi = 20.0 + 280.0 / np.exp(0.023 * d_prev)

        b = np.empty_like(d_prev)
        m1 = d_prev <= 33.0
        m2 = (d_prev > 33.0) & (d_prev <= 65.0)
        m3 = d_prev > 65.0
        b[m1] = 100.0 / (0.5 + 0.3 * d_prev[m1])
        b[m2] = 14.0 - 1.3 * np.log(d_prev[m2])
        b[m3] = 6.2 * np.log(d_prev[m3]) - 17.2

        wmr = wmi + 1000.0 * rw / (48.77 + b * rw)
        pr_r = 43.43 * (5.6348 - np.log(np.maximum(wmr - 20.0, 1.0e-12)))
        pr[rain_mask] = pr_r

    pr = np.maximum(pr, 0.0)
    dmc1 = np.maximum(pr + rk, 0.0)
    return dmc1.astype(np.float32)

def drought_code(dc_yda, temp, rh_unused, prec, lat, mon, lat_adjust=True):
    dc_yda = dc_yda.astype(np.float64, copy=False)
    temp = temp.astype(np.float64, copy=False)
    prec = prec.astype(np.float64, copy=False)
    lat = lat.astype(np.float64, copy=False)

    fl01 = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6], dtype=np.float64)
    fl02 = np.array([6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8], dtype=np.float64)

    temp = np.maximum(temp, -2.8)
    pe = (0.36 * (temp + 2.8) + fl01[mon - 1]) / 2.0
    if lat_adjust:
        pe = np.where(lat <= -20.0, (0.36 * (temp + 2.8) + fl02[mon - 1]) / 2.0, pe)
        pe = np.where((lat > -20.0) & (lat <= 20.0), (0.36 * (temp + 2.8) + 1.4) / 2.0, pe)
    pe = np.maximum(pe, 0.0)

    ra = prec
    rw = 0.83 * ra - 1.27
    smi = 800.0 * np.exp(-dc_yda / 400.0)

    dr = dc_yda.copy()
    wet = prec > 2.8
    if np.any(wet):
        arg = 1.0 + 3.937 * rw[wet] / smi[wet]
        # Guard against roundoff/invalid values; this branch is only used for wet pixels.
        arg = np.maximum(arg, np.finfo(np.float64).tiny)
        dr0 = dc_yda[wet] - 400.0 * np.log(arg)
        dr[wet] = np.maximum(dr0, 0.0)

    dc1 = np.maximum(dr + pe, 0.0)
    return dc1.astype(np.float32)


def initial_spread_index(ffmc, ws_kmh):
    ffmc = ffmc.astype(np.float64, copy=False)
    ws_kmh = ws_kmh.astype(np.float64, copy=False)

    fm = FFMC_COEFFICIENT * (101.0 - ffmc) / (59.5 + ffmc)
    f_w = np.exp(0.05039 * ws_kmh)
    f_f = 91.9 * np.exp(-0.1386 * fm) * (1.0 + np.power(fm, 5.31) / 49300000.0)
    isi = 0.208 * f_w * f_f
    return isi.astype(np.float32)



def buildup_index(dmc, dc):
    dmc = dmc.astype(np.float64, copy=False)
    dc = dc.astype(np.float64, copy=False)

    denom = dmc + 0.4 * dc
    bui1 = np.zeros_like(dmc)
    nz = denom != 0.0
    bui1[nz] = 0.8 * dc[nz] * dmc[nz] / denom[nz]

    p = np.zeros_like(dmc)
    nz_dmc = dmc != 0.0
    p[nz_dmc] = (dmc[nz_dmc] - bui1[nz_dmc]) / dmc[nz_dmc]

    cc = 0.92 + np.power(0.0114 * dmc, 1.7)
    bui0 = np.maximum(dmc - cc * p, 0.0)
    bui1 = np.where(bui1 < dmc, bui0, bui1)
    return bui1.astype(np.float32)


def fire_weather_index(isi, bui):
    isi = isi.astype(np.float64, copy=False)
    bui = bui.astype(np.float64, copy=False)

    bb = np.where(
        bui > 80.0,
        0.1 * isi * (1000.0 / (25.0 + 108.64 / np.exp(0.023 * bui))),
        0.1 * isi * (0.626 * np.power(bui, 0.809) + 2.0),
    )
    fwi = bb.copy()
    mask = bb > 1.0
    if np.any(mask):
        log_term = 0.434 * np.log(bb[mask])
        fwi[mask] = np.exp(2.72 * np.power(log_term, 0.647))
    return fwi.astype(np.float32)

def compute_reference_mask(
    first_path: Path,
    band_map: Dict[str, int],
    chunk_rows: int,
    strict_nodata_value: float | None,
) -> np.ndarray:
    with rasterio.open(first_path) as src:
        h, w = src.height, src.width
        ref_mask = np.zeros((h, w), dtype=bool)
        nodata_values = src.nodatavals

        for row0 in range(0, h, chunk_rows):
            nrows = min(chunk_rows, h - row0)
            win = Window(0, row0, w, nrows)
            valid = np.ones((nrows, w), dtype=bool)
            for key in REQUIRED_BANDS:
                arr = src.read(band_map[key], window=win).astype(np.float32)
                nd = strict_nodata_value
                if nd is None:
                    nd = nodata_values[band_map[key] - 1]
                band_valid = np.isfinite(arr)
                if nd is not None:
                    band_valid &= arr != nd
                valid &= band_valid
            ref_mask[row0:row0 + nrows, :] = valid

        if not ref_mask.any():
            raise ValueError("Reference valid mask is empty. Check nodata handling and inputs.")
        return ref_mask


def read_band_chunk(src, band_idx: int, win: Window) -> np.ndarray:
    return src.read(band_idx, window=win).astype(np.float32)


def output_name(dt: date, prefix: str, suffix: str) -> str:
    return f"{prefix}{dt.strftime('%Y%m%d')}{suffix}"


def valid_mask_for_chunk(
    d2m: np.ndarray,
    t2m: np.ndarray,
    tp: np.ndarray,
    u10: np.ndarray,
    v10: np.ndarray,
    nodata_values: Tuple[float | None, float | None, float | None, float | None, float | None],
    input_nodata: float | None,
) -> np.ndarray:
    current_valid = np.isfinite(d2m) & np.isfinite(t2m) & np.isfinite(tp) & np.isfinite(u10) & np.isfinite(v10)
    if input_nodata is not None:
        current_valid &= (d2m != input_nodata) & (t2m != input_nodata) & (tp != input_nodata) & (u10 != input_nodata) & (v10 != input_nodata)
    else:
        for arr, nd in zip((d2m, t2m, tp, u10, v10), nodata_values):
            if nd is not None:
                current_valid &= arr != nd
    return current_valid


def process_chunk_task(
    *,
    d2m: np.ndarray,
    t2m: np.ndarray,
    tp: np.ndarray,
    u10: np.ndarray,
    v10: np.ndarray,
    ref_valid: np.ndarray,
    ffmc_prev_chunk: np.ndarray,
    dmc_prev_chunk: np.ndarray,
    dc_prev_chunk: np.ndarray,
    lat_chunk: np.ndarray,
    mon: int,
    nodata_values: Tuple[float | None, float | None, float | None, float | None, float | None],
    input_nodata: float | None,
    out_nodata: float,
    allow_mask_changes: bool,
    no_lat_adjust: bool,
    daily_name: str,
    row0: int,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current_valid = valid_mask_for_chunk(d2m, t2m, tp, u10, v10, nodata_values, input_nodata)

    if not allow_mask_changes:
        if not np.array_equal(current_valid, ref_valid):
            diff = int(np.count_nonzero(current_valid != ref_valid))
            raise ValueError(
                f"Valid mask changed on {daily_name} in rows {row0}:{row0 + ref_valid.shape[0]}; "
                f"{diff} cells differ from the reference mask. "
                "Use --allow-mask-changes only if this is intentional."
            )
        active = ref_valid
    else:
        active = ref_valid & current_valid

    out = np.full(d2m.shape, np.float32(out_nodata), dtype=np.float32)
    ffmc_new_chunk = ffmc_prev_chunk.copy()
    dmc_new_chunk = dmc_prev_chunk.copy()
    dc_new_chunk = dc_prev_chunk.copy()

    if np.any(active):
        td = d2m[active]
        t = t2m[active]
        p = tp[active]
        ws_kmh = (np.hypot(u10[active], v10[active]) * 3.6).astype(np.float32)
        rh = rh_from_t_td(t, td)
        rh = np.minimum(rh, np.float32(99.9999))

        ffmc_prev = ffmc_prev_chunk[active]
        dmc_prev = dmc_prev_chunk[active]
        dc_prev = dc_prev_chunk[active]
        lat_active = lat_chunk[active]

        ffmc_new = fine_fuel_moisture_code(ffmc_prev, t, rh, ws_kmh, p)
        dmc_new = duff_moisture_code(dmc_prev, t, rh, p, lat_active, mon, lat_adjust=not no_lat_adjust)
        dc_new = drought_code(dc_prev, t, rh, p, lat_active, mon, lat_adjust=not no_lat_adjust)
        isi = initial_spread_index(ffmc_new, ws_kmh)
        bui = buildup_index(dmc_new, dc_new)
        fwi = fire_weather_index(isi, bui)

        ffmc_new_chunk[active] = ffmc_new
        dmc_new_chunk[active] = dmc_new
        dc_new_chunk[active] = dc_new
        out[active] = fwi.astype(np.float32)

    return row0, out, ffmc_new_chunk, dmc_new_chunk, dc_new_chunk


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute daily FWI GeoTIFFs from local-noon weather GeoTIFFs.")
    ap.add_argument("input_folder", type=Path, help="Folder containing daily 5-band GeoTIFFs")
    ap.add_argument("output_folder", type=Path, help="Folder to write daily 1-band FWI GeoTIFFs")
    ap.add_argument("--pattern", default="*.tif", help="Glob for input files (default: *.tif)")
    ap.add_argument("--chunk-rows", type=int, default=512, help="Rows per processing chunk (default: 512)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="Worker threads per day (default: CPU count)")
    ap.add_argument("--prefix", default="FWI_", help="Output filename prefix (default: FWI_)")
    ap.add_argument("--suffix", default=".tif", help="Output filename suffix (default: .tif)")
    ap.add_argument("--nodata", type=float, default=NODATA_DEFAULT, help=f"Output nodata (default: {NODATA_DEFAULT})")
    ap.add_argument("--input-nodata", type=float, default=None, help="Override input nodata instead of reading from GeoTIFF metadata")
    ap.add_argument("--init-ffmc", type=float, default=85.0, help="Initial FFMC for first day (default: 85)")
    ap.add_argument("--init-dmc", type=float, default=6.0, help="Initial DMC for first day (default: 6)")
    ap.add_argument("--init-dc", type=float, default=15.0, help="Initial DC for first day (default: 15)")
    ap.add_argument(
        "--gap-mode",
        choices=["error", "reinit"],
        default="error",
        help="How to handle missing dates: error (default) stops, reinit restarts FFMC/DMC/DC after each gap segment.",
    )
    ap.add_argument(
        "--no-lat-adjust",
        action="store_true",
        help="Disable latitude/day-length adjustment for DMC/DC (not recommended for global runs)",
    )
    ap.add_argument(
        "--allow-mask-changes",
        action="store_true",
        help="Allow per-day valid footprint to differ from day 1. If set, each day uses the intersection of day-1 valid mask and current valid data.",
    )
    ap.add_argument(
        "--missing-dates-txt",
        type=Path,
        default=None,
        help="Optional path for a text report listing all missing input dates. Defaults to <output_folder>/missing_dates.txt when gaps exist.",
    )
    args = ap.parse_args()

    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")

    files = list_daily_files(args.input_folder, args.pattern)
    gaps = find_date_gaps(files)
    missing_dates = list_missing_dates(files)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    if args.missing_dates_txt is None:
        missing_dates_txt = args.output_folder / "missing_dates.txt"
    else:
        missing_dates_txt = args.missing_dates_txt

    if gaps:
        write_missing_dates_report(missing_dates_txt, missing_dates, gaps)
        print(f"Missing-dates report written to: {missing_dates_txt}")

    if args.gap_mode == "error":
        ensure_daily_continuity(files)
    elif gaps:
        print("Warning: input dates are not continuous.")
        for prev, cur, missing_n in gaps:
            print(
                f"  Gap after {prev.path.name} ({prev.dt}) before {cur.path.name} ({cur.dt}); "
                f"missing {missing_n} day(s). Reinitializing FFMC/DMC/DC at {cur.dt}."
            )

    with rasterio.open(files[0].path) as src0:
        band_map = band_map_from_descriptions(src0)
        profile = src0.profile.copy()
        height, width = src0.height, src0.width
        transform = src0.transform
        crs = src0.crs
        lat_rows = row_center_latitudes(transform, height)

        ref_mask = compute_reference_mask(
            files[0].path,
            band_map,
            chunk_rows=args.chunk_rows,
            strict_nodata_value=args.input_nodata,
        )

        ffmc_state = np.full((height, width), np.float32(args.init_ffmc), dtype=np.float32)
        dmc_state = np.full((height, width), np.float32(args.init_dmc), dtype=np.float32)
        dc_state = np.full((height, width), np.float32(args.init_dc), dtype=np.float32)

        ffmc_state[~ref_mask] = np.nan
        dmc_state[~ref_mask] = np.nan
        dc_state[~ref_mask] = np.nan

        out_profile = profile.copy()
        out_profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=np.float32(args.nodata),
            compress="deflate",
            predictor=3,
            tiled=True,
            interleave="band",
        )

        nodata_values = (
            src0.nodatavals[band_map["d2m_c"] - 1],
            src0.nodatavals[band_map["t2m_c"] - 1],
            src0.nodatavals[band_map["tp_mm"] - 1],
            src0.nodatavals[band_map["u10_ms"] - 1],
            src0.nodatavals[band_map["v10_ms"] - 1],
        )

        iterator: Iterable[DailyFile]
        if tqdm is not None:
            iterator = tqdm(files, desc="FWI days", unit="day")
        else:
            iterator = files

        prev_dt = None
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for daily in iterator:
                if prev_dt is not None and (daily.dt - prev_dt).days != 1:
                    if args.gap_mode == "reinit":
                        ffmc_state[:, :] = np.float32(args.init_ffmc)
                        dmc_state[:, :] = np.float32(args.init_dmc)
                        dc_state[:, :] = np.float32(args.init_dc)
                        ffmc_state[~ref_mask] = np.nan
                        dmc_state[~ref_mask] = np.nan
                        dc_state[~ref_mask] = np.nan
                    else:
                        raise ValueError(
                            f"Input dates are not continuous before {daily.path.name} ({daily.dt})."
                        )
                with rasterio.open(daily.path) as src:
                    if src.width != width or src.height != height:
                        raise ValueError(f"Raster shape mismatch in {daily.path.name}")
                    if src.transform != transform:
                        raise ValueError(f"Transform mismatch in {daily.path.name}")
                    if src.crs != crs:
                        raise ValueError(f"CRS mismatch in {daily.path.name}")

                    out_path = args.output_folder / output_name(daily.dt, args.prefix, args.suffix)
                    with rasterio.open(out_path, "w", **out_profile) as dst:
                        dst.set_band_description(1, "FWI")
                        dst.update_tags(
                            1,
                            long_name="Canadian Fire Weather Index",
                            source_date=daily.dt.isoformat(),
                            input_file=daily.path.name,
                            algorithm="Van Wagner 1987 / cffdrs daily FWI",
                        )

                        futures = []
                        for row0 in range(0, height, args.chunk_rows):
                            nrows = min(args.chunk_rows, height - row0)
                            win = Window(0, row0, width, nrows)

                            d2m = read_band_chunk(src, band_map["d2m_c"], win)
                            t2m = read_band_chunk(src, band_map["t2m_c"], win)
                            tp = read_band_chunk(src, band_map["tp_mm"], win)
                            u10 = read_band_chunk(src, band_map["u10_ms"], win)
                            v10 = read_band_chunk(src, band_map["v10_ms"], win)

                            ref_valid = ref_mask[row0:row0 + nrows, :].copy()
                            ffmc_prev_chunk = ffmc_state[row0:row0 + nrows, :].copy()
                            dmc_prev_chunk = dmc_state[row0:row0 + nrows, :].copy()
                            dc_prev_chunk = dc_state[row0:row0 + nrows, :].copy()
                            lat_chunk = np.broadcast_to(lat_rows[row0:row0 + nrows, None], (nrows, width)).copy()

                            futures.append(
                                ex.submit(
                                    process_chunk_task,
                                    d2m=d2m,
                                    t2m=t2m,
                                    tp=tp,
                                    u10=u10,
                                    v10=v10,
                                    ref_valid=ref_valid,
                                    ffmc_prev_chunk=ffmc_prev_chunk,
                                    dmc_prev_chunk=dmc_prev_chunk,
                                    dc_prev_chunk=dc_prev_chunk,
                                    lat_chunk=lat_chunk,
                                    mon=daily.dt.month,
                                    nodata_values=nodata_values,
                                    input_nodata=args.input_nodata,
                                    out_nodata=args.nodata,
                                    allow_mask_changes=args.allow_mask_changes,
                                    no_lat_adjust=args.no_lat_adjust,
                                    daily_name=daily.path.name,
                                    row0=row0,
                                )
                            )

                        for fut in as_completed(futures):
                            row0, out, ffmc_new_chunk, dmc_new_chunk, dc_new_chunk = fut.result()
                            nrows = out.shape[0]
                            win = Window(0, row0, width, nrows)
                            ffmc_state[row0:row0 + nrows, :] = ffmc_new_chunk
                            dmc_state[row0:row0 + nrows, :] = dmc_new_chunk
                            dc_state[row0:row0 + nrows, :] = dc_new_chunk
                            dst.write(out, 1, window=win)
                prev_dt = daily.dt

    print("Done.")
    print(f"Inputs : {len(files)} daily GeoTIFFs")
    print(f"Outputs: {args.output_folder}")
    print(f"Date span: {files[0].dt} -> {files[-1].dt}")
    print(f"Init state used for first day: FFMC={args.init_ffmc}, DMC={args.init_dmc}, DC={args.init_dc}")
    print(f"Workers used per day: {args.workers}")
    print(f"Gap mode: {args.gap_mode}")
    if gaps:
        print(f"Missing dates listed in: {missing_dates_txt}")
    else:
        print("Missing dates listed in: none (no gaps found)")


if __name__ == "__main__":
    main()
