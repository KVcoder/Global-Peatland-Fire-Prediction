#!/usr/bin/env python3
"""
Convert daily GeoTIFFs (ERA5-Land / SMAP / VIIRS etc.) → Zarr (T, C, H, W) with:
- Validation of CRS / transform / shape
- Calendar enforcement over a given date range
- Optional handling of *missing* days by filling them with all-NaN slices
- Peat mask derivation (from first non-missing day)
- Post-write NaN-aware pixel checks
- Optional tiny read benchmark

(Parallel-enabled)
- Optional Dask LocalCluster (multi-process or multi-thread) to parallelize the
  xarray/dask graph during to_zarr() and any eager reads.
- Optional GDAL internal threading via --gdal-threads.

Requires: rasterio, rioxarray, xarray, dask[array], zarr, numcodecs, numpy
Recommended (for best parallelism): dask[distributed]
"""

from __future__ import annotations
import os
import re
import sys
import json
import math
import time
import random
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import rasterio
from rasterio.windows import Window
import xarray as xr
import rioxarray as rxr
import dask.array as da  # noqa: F401 (triggers dask-backed IO in rioxarray/xarray)
import zarr
import fsspec
from numcodecs import Blosc, Zstd
from tqdm import tqdm
from tqdm.dask import TqdmCallback

# =========================
# DASK PARALLELISM HELPERS
# =========================
def start_dask_cluster(
    scheduler: str,
    workers: int,
    threads_per_worker: int,
    worker_mem: str,
    dash: bool,
):
    """
    Returns a context manager that sets up parallel execution.
    Prefer 'distributed' if available; otherwise fall back to threaded/process pool
    via dask's single-process scheduler.
    """
    class _NoopCtx:
        def __enter__(self):
            import dask
            if scheduler == "threads":
                dask.config.set(scheduler="threads")
            elif scheduler == "processes":
                dask.config.set(scheduler="processes")
            else:
                dask.config.set(scheduler="threads")
            return None

        def __exit__(self, *exc):
            return False

    if scheduler != "distributed":
        return _NoopCtx()

    try:
        from dask.distributed import LocalCluster, Client
    except Exception:
        print(
            "[warn] dask.distributed not available; falling back to local threads scheduler.",
            file=sys.stderr,
        )
        return _NoopCtx()

    class _DistCtx:
        def __enter__(self):
            processes = True
            dashboard = ":0" if dash else None
            self.cluster = LocalCluster(
                n_workers=int(workers),
                threads_per_worker=int(threads_per_worker),
                processes=processes,
                memory_limit=None if worker_mem == "auto" else worker_mem,
                dashboard_address=dashboard,
            )
            self.client = Client(self.cluster)
            try:
                link = getattr(self.client, "dashboard_link", None)
                if link:
                    print(f"[dask] Dashboard: {link}")
            except Exception:
                pass
            print(
                f"[dask] LocalCluster up: workers={workers}, threads/worker={threads_per_worker}, "
                f"processes={processes}, mem/worker={worker_mem}"
            )
            return self.client

        def __exit__(self, *exc):
            try:
                self.client.close()
            finally:
                self.cluster.close()
            print("[dask] Cluster closed.")
            return False

    return _DistCtx()

# -----------------------
# Filename → date parsing
# -----------------------
DATE_RE = re.compile(r"(\d{4})[^\d]?(\d{2})[^\d]?(\d{2})")  # flexible YYYYMMDD in filename


def find_tiffs(tiff_dir: str) -> List[str]:
    files = sorted(
        [
            os.path.join(tiff_dir, f)
            for f in os.listdir(tiff_dir)
            if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
        ]
    )
    if not files:
        raise FileNotFoundError(f"No .tif files found in {tiff_dir}")
    return files


def extract_date_key(path: str) -> Tuple[int, int, int]:
    m = DATE_RE.search(os.path.basename(path))
    if not m:
        ts = os.path.getmtime(path)
        d = dt.date.fromtimestamp(ts)
        return (d.year, d.month, d.day)
    y, mth, d = map(int, m.groups())
    return (y, mth, d)


def expected_dates(start: str, end: str) -> List[str]:
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    out: List[str] = []
    d = s
    while d <= e:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def enforce_calendar(
    paths: List[str],
    start: str,
    end: str,
    allow_missing: bool = False,
) -> Tuple[List[Optional[str]], List[str]]:
    """
    Map TIFFs to a daily calendar from start..end (inclusive).

    Returns
    -------
    ordered : list[Optional[str]]
        One entry per calendar day. If allow_missing is False, all entries
        are real paths. If allow_missing is True, missing days get None.
    want : list[str]
        ISO dates corresponding to each entry in `ordered`.
    """
    def to_time_str(p: str) -> str:
        y, m, d = extract_date_key(p)
        return f"{y:04d}-{m:02d}-{d:02d}"

    by_date: Dict[str, List[str]] = {}
    for p in paths:
        t = to_time_str(p)
        by_date.setdefault(t, []).append(p)

    want = expected_dates(start, end)
    missing = [d for d in want if d not in by_date]
    duplicates = {d: v for d, v in by_date.items() if len(v) > 1}
    extras = [d for d in by_date.keys() if d not in want]

    if duplicates:
        example = [(k, len(v)) for k, v in list(duplicates.items())[:3]]
        raise ValueError(f"Duplicate daily files for {len(duplicates)} day(s), e.g. {example}")

    if extras:
        print(
            f"[warn] Found {len(extras)} extra day(s) not in {start}..{end}, e.g. {extras[:3]}",
            file=sys.stderr,
        )

    if missing:
        if not allow_missing:
            raise ValueError(f"Missing {len(missing)} day(s), e.g. {missing[:5]} ...")
        else:
            print(
                f"[warn] Allowing {len(missing)} missing day(s); they will be filled with NaNs in the stack.",
                file=sys.stderr,
            )

    ordered: List[Optional[str]] = []
    for d in want:
        if d in by_date:
            ordered.append(by_date[d][0])
        else:
            ordered.append(None)

    return ordered, want

# -----------------------
# Raster signature & checks
# -----------------------
@dataclass
class RasterSignature:
    crs: str
    transform: Tuple[float, float, float, float, float, float]
    width: int
    height: int
    count: int
    dtype: str
    nodata: Optional[float]


def read_signature(path: str) -> RasterSignature:
    with rasterio.open(path) as ds:
        return RasterSignature(
            crs=str(ds.crs) if ds.crs else "",
            transform=tuple(ds.transform)[:6],
            width=ds.width,
            height=ds.height,
            count=ds.count,
            dtype=ds.dtypes[0],
            nodata=ds.nodata,
        )


def almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def check_consistency(paths: List[str]) -> RasterSignature:
    sig0 = read_signature(paths[0])
    for p in paths[1:]:
        s = read_signature(p)
        if s.crs != sig0.crs:
            raise ValueError(f"CRS mismatch: {p} {s.crs} vs {sig0.crs}")
        for i, (aa, bb) in enumerate(zip(s.transform, sig0.transform)):
            if not almost_equal(aa, bb, tol=1e-9):
                raise ValueError(
                    f"Transform mismatch at idx {i} in {p}: {s.transform} vs {sig0.transform}"
                )
        if (s.width, s.height) != (sig0.width, sig0.height):
            raise ValueError(
                f"Shape mismatch in {p}: {(s.width, s.height)} vs {(sig0.width, sig0.height)}"
            )
        if s.count != sig0.count:
            raise ValueError(f"Band count mismatch in {p}: {s.count} vs {sig0.count}")
        if s.dtype != sig0.dtype:
            raise ValueError(f"Dtype mismatch in {p}: {s.dtype} vs {sig0.dtype}")
        if s.nodata != sig0.nodata:
            print(
                f"[warn] nodata mismatch: {p}: {s.nodata} vs {sig0.nodata}",
                file=sys.stderr,
            )
    return sig0

# -----------------------
# Compression utils
# -----------------------
def make_compressor(kind: str, clevel: int):
    kind = kind.lower()
    if kind == "zstd":
        try:
            return Zstd(level=clevel)
        except Exception:
            return Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.SHUFFLE)
    elif kind == "blosc-zstd":
        return Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.SHUFFLE)
    elif kind == "lz4":
        return Blosc(cname="lz4", clevel=clevel, shuffle=Blosc.SHUFFLE)
    else:
        raise ValueError(f"Unsupported compressor: {kind}")

# -----------------------
# Build stacked DataArray with NaN + NaN for missing days
# -----------------------
def build_time_stack_with_nan(
    paths: List[Optional[str]],
    date_strings: List[str],
    chunks_xy: int,
    nodata_fallback: float = -9999,
) -> xr.DataArray:
    """
    Open each GeoTIFF; coerce NoData/-9999 to NaN; concat along time.
    If a path is None, synthesize a full-NaN slice with matching shape.

    Output dims: (time, band, y, x). Time is numpy datetime64[D].
    """
    # Template from first non-missing file
    template_da = None
    for p in paths:
        if p is not None:
            tmp = rxr.open_rasterio(
                p, chunks={"band": -1, "y": chunks_xy, "x": chunks_xy}
            )
            nd = tmp.rio.nodata
            if nd is None:
                nd = nodata_fallback
            tmp = tmp.where(tmp != nd)
            template_da = tmp
            break

    if template_da is None:
        raise RuntimeError("No valid GeoTIFFs found to derive template from.")

    opened: List[xr.DataArray] = []
    times_py: List[str] = []

    for p, tstr in tqdm(
        list(zip(paths, date_strings)),
        desc="Open/mask GeoTIFFs (with NaN fill for missing)",
        unit="file",
    ):
        if p is None:
            da_ = xr.full_like(template_da, np.nan)
        else:
            da_ = rxr.open_rasterio(
                p, chunks={"band": -1, "y": chunks_xy, "x": chunks_xy}
            )
            nd = da_.rio.nodata
            if nd is None:
                nd = nodata_fallback
            da_ = da_.where(da_ != nd)

        times_py.append(tstr)
        opened.append(da_)

    stacked = xr.concat(opened, dim="time")  # (time, band, y, x)
    times_np = np.array(times_py, dtype="datetime64[D]")
    stacked = stacked.assign_coords(time=("time", times_np))
    return stacked


def derive_and_check_peat_mask(sample_tif: str) -> np.ndarray:
    with rasterio.open(sample_tif) as ds:
        nd = ds.nodata if ds.nodata is not None else -9999
        block = ds.read(1)  # (H,W) first band
        mask = (block != nd).astype(np.uint8)  # 1=peatland(valid), 0=non-peat
        return mask

# -----------------------
# Write dataset (folder or zip store)
# -----------------------
def write_dataset_with_mask(
    da_tc: xr.DataArray,
    peat_mask: np.ndarray,
    zarr_out: str,
    chunk_t: int,
    chunk_y: int,
    chunk_x: int,
    compressor,
    consolidate: bool,
    overwrite: bool,
    zip_store: bool,
    zarr_format: int,
) -> Tuple[int, int, int, int]:
    da_tc = da_tc.transpose("time", "band", "y", "x")
    T = da_tc.sizes["time"]
    C = da_tc.sizes["band"]
    H = da_tc.sizes["y"]
    W = da_tc.sizes["x"]

    target = da_tc.chunk(
        {
            "time": max(1, chunk_t),
            "band": C,
            "y": max(1, chunk_y),
            "x": max(1, chunk_x),
        }
    )

    ds = target.to_dataset(name="field")
    ds["field"].attrs.update(
        {
            "long_name": "Daily stack (NaN outside peat; NaN on missing days)",
            "layout": "T,C,H,W via dims (time,band,y,x)",
            "nodata_handling": "NoData/-9999 coerced to NaN; missing days → all-NaN slice",
        }
    )

    ds["peat_mask"] = xr.DataArray(
        peat_mask,
        dims=("y", "x"),
        attrs={
            "long_name": "Peatland mask (1=peatland valid, 0=non-peat)",
            "dtype": "uint8",
        },
    )

    if int(zarr_format) == 3:
        era_enc = {
            "compressors": [compressor],
            "chunks": (chunk_t, C, chunk_y, chunk_x),
            "_FillValue": None,
        }
        mask_enc = {
            "compressors": [compressor],
            "chunks": (chunk_y, chunk_x),
            "dtype": "uint8",
        }
    else:
        era_enc = {
            "compressor": compressor,
            "chunks": (chunk_t, C, chunk_y, chunk_x),
            "_FillValue": None,
        }
        mask_enc = {
            "compressor": compressor,
            "chunks": (chunk_y, chunk_x),
            "dtype": "uint8",
        }

    encoding = {"field": era_enc, "peat_mask": mask_enc}

    if zip_store:
        if os.path.exists(zarr_out):
            os.remove(zarr_out)
        store = fsspec.get_mapper(f"zip://{zarr_out}")
        with TqdmCallback(desc="Writing Zarr (zip)", unit="task"):
            ds.to_zarr(
                store=store,
                consolidated=False,
                mode="w",
                encoding=encoding,
                zarr_format=int(zarr_format),
                compute=True,
            )
        if consolidate and int(zarr_format) == 2:
            zarr.consolidate_metadata(store)
        elif consolidate and int(zarr_format) == 3:
            print(
                "[note] Consolidation is Zarr v2 only; skipping for zarr_format=3."
            )
    else:
        if overwrite and os.path.exists(zarr_out):
            import shutil

            shutil.rmtree(zarr_out)
        with TqdmCallback(desc="Writing Zarr", unit="task"):
            ds.to_zarr(
                zarr_out,
                consolidated=False,
                mode="w",
                encoding=encoding,
                zarr_format=int(zarr_format),
                compute=True,
            )
        if consolidate and int(zarr_format) == 2:
            zarr.consolidate_metadata(zarr_out)
        elif consolidate and int(zarr_format) == 3:
            print(
                "[note] Consolidation is Zarr v2 only; skipping for zarr_format=3."
            )

    return T, C, H, W

# -----------------------
# Post-write tests and utilities
# -----------------------
def open_zarr_main_array(zarr_path_or_store: str):
    """
    Cross-version Zarr (v2/v3) opener that returns (largest_array, key_path).
    Handles directory stores and .zip via fsspec.
    """
    import zarr as _z

    store_input = zarr_path_or_store
    if isinstance(zarr_path_or_store, str) and zarr_path_or_store.lower().endswith(
        ".zip"
    ):
        store_input = fsspec.get_mapper(f"zip://{zarr_path_or_store}")

    # Try consolidated first, then plain group
    try:
        root = _z.open_consolidated(store_input, mode="r")
    except Exception:
        root = _z.open_group(store_input, mode="r")

    def _iter_arrays(g, prefix=""):
        if hasattr(g, "array_keys"):
            for k in g.array_keys():
                try:
                    yield prefix + k, g[k]
                except Exception:
                    pass
        if hasattr(g, "group_keys"):
            for k in g.group_keys():
                try:
                    sub = g[k]
                except Exception:
                    continue
                yield from _iter_arrays(sub, prefix + k + "/")

        if hasattr(g, "arrays"):
            for k, arr in g.arrays():
                yield prefix + k, arr
        if hasattr(g, "groups"):
            for k, sub in g.groups():
                yield from _iter_arrays(sub, prefix + k + "/")

        if hasattr(g, "keys"):
            for k in g.keys():
                try:
                    obj = g[k]
                except Exception:
                    continue
                try:
                    if isinstance(obj, _z.Array):
                        yield prefix + k, obj
                    else:
                        yield from _iter_arrays(obj, prefix + k + "/")
                except Exception:
                    continue

    best_key, best_arr, best_n = None, None, -1
    for key, arr in _iter_arrays(root, ""):
        try:
            n = int(np.prod(arr.shape))
        except Exception:
            continue
        if n > best_n:
            best_key, best_arr, best_n = key, arr, n

    if best_arr is None:
        raise RuntimeError("No arrays found in zarr store.")

    return best_arr, best_key


def pixel_match_check_nanaware(
    paths: List[Optional[str]], date_strings: List[str], zarr_out: str, trials: int = 3
) -> None:
    """
    Randomly pick time indices where we actually have a GeoTIFF (paths[t] is not None),
    and compare a patch vs. the Zarr array at the same time index.

    This works even when some days are all-NaN (missing input files).
    """
    arr, key = open_zarr_main_array(zarr_out)
    T, C, H, W = arr.shape
    print(f"[check] Zarr main array '{key}' shape: (T={T},C={C},H={H},W={W})")

    # Time indices where we have a real file
    candidates = [i for i, p in enumerate(paths) if p is not None]
    if not candidates:
        print("[check] No non-missing days to validate against GeoTIFFs.")
        return

    rng = random.Random(1234)
    trials = min(trials, len(candidates))
    for _ in tqdm(range(trials), desc="Pixel checks", unit="win"):
        t = rng.choice(candidates)
        p = paths[t]
        assert p is not None

        with rasterio.open(p) as ds:
            nd = ds.nodata if ds.nodata is not None else -9999
            patch = min(128, ds.width, ds.height)
            y0 = max(0, ds.height // 2 - patch // 2)
            x0 = max(0, ds.width // 2 - patch // 2)
            tif_block = ds.read(window=Window(x0, y0, patch, patch))  # (C,H,W)

        z_block = arr.get_orthogonal_selection(
            (
                slice(t, t + 1),
                slice(0, C),
                slice(y0, y0 + patch),
                slice(x0, x0 + patch),
            )
        )[0]

        tif_nan = np.where(tif_block == nd, np.nan, tif_block)
        m = np.isfinite(tif_nan) & np.isfinite(z_block)
        ok = True if not m.any() else np.allclose(
            tif_nan[m], z_block[m], rtol=0, atol=1e-6
        )
        if not ok:
            raise AssertionError(f"Pixel mismatch at t={t} (date {date_strings[t]})")
        print(f"[check] t={t} ({date_strings[t]}): NaN-aware pixel window check passed.")


def tiny_bench_read(zarr_out: str, n_patches: int = 20, patch: int = 256):
    arr, _ = open_zarr_main_array(zarr_out)
    T, C, H, W = arr.shape
    rng = random.Random(99)
    t0 = time.time()
    for _ in range(n_patches):
        t = rng.randrange(0, T)
        y = rng.randrange(0, max(1, H - patch + 1))
        x = rng.randrange(0, max(1, W - patch + 1))
        _ = arr.get_orthogonal_selection(
            (
                slice(t, t + 1),
                slice(0, C),
                slice(y, y + patch),
                slice(x, x + patch),
            )
        )
    dt = time.time() - t0
    if dt > 0:
        print(
            f"[bench] Zarr random patch reads: {n_patches / dt:.1f} patches/sec (patch={patch})"
        )


def estimate_zarr_size_sample(
    paths: List[Optional[str]],
    chunk_y: int,
    chunk_x: int,
    compressor,
    samples: int = 8,
) -> None:
    real_paths = [p for p in paths if p is not None]
    if not real_paths:
        print("[estimate] No real paths available to estimate size.")
        return

    sig = read_signature(real_paths[0])
    H, W, C = sig.height, sig.width, sig.count
    tiles_y = math.ceil(H / chunk_y)
    tiles_x = math.ceil(W / chunk_x)
    total_tiles_per_t = tiles_y * tiles_x
    total_tiles = total_tiles_per_t * len(paths)

    rng = random.Random(321)
    from numcodecs.compat import ensure_ndarray

    compressed_bytes: List[int] = []
    for _ in range(min(samples, len(real_paths))):
        p = rng.choice(real_paths)
        y0 = rng.randrange(0, max(1, H - chunk_y + 1))
        x0 = rng.randrange(0, max(1, W - chunk_x + 1))
        with rasterio.open(p) as ds:
            block = ds.read(window=Window(x0, y0, chunk_x, chunk_y))  # (C, h, w)
        arr = np.ascontiguousarray(block)
        buf = compressor.encode(ensure_ndarray(arr))
        compressed_bytes.append(len(buf))

    if compressed_bytes:
        avg = sum(compressed_bytes) / len(compressed_bytes)
        approx_total = avg * total_tiles
        print(
            f"[estimate] Approx Zarr data bytes ~ {approx_total / 1e9:.2f} GB "
            f"(avg tile {avg / 1024:.1f} KiB, tiles {total_tiles})"
        )
    else:
        print("[estimate] Not enough samples to estimate size.")

# -----------------------
# Main
# -----------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="GeoTIFF (daily) → Zarr (ERA5-Land/SMAP/VIIRS) with mask & checks (parallel-enabled)"
    )
    ap.add_argument("--tiff-dir", required=True, help="Folder with daily .tif files")
    ap.add_argument("--zarr-out", required=True, help="Output Zarr store path (.zarr folder or .zip)")
    ap.add_argument("--chunk-t", type=int, default=16)
    ap.add_argument("--chunk-y", type=int, default=256)
    ap.add_argument("--chunk-x", type=int, default=256)
    ap.add_argument("--compressor", choices=["zstd", "blosc-zstd", "lz4"], default="zstd")
    ap.add_argument("--clevel", type=int, default=3)
    ap.add_argument("--consolidate", action="store_true", help="Write consolidated metadata (dir store)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--estimate-compression",
        action="store_true",
        help="Sample tiles to estimate Zarr size",
    )
    ap.add_argument(
        "--bench-read", action="store_true", help="Quick random-patch read bench on Zarr"
    )
    ap.add_argument(
        "--start",
        default="2016-01-01",
        help="Expected calendar start date (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--end",
        default="2024-12-31",
        help="Expected calendar end date (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--zip-store",
        action="store_true",
        help="Write a single .zip Zarr file instead of a folder",
    )
    ap.add_argument(
        "--zarr-format",
        type=int,
        default=2,
        choices=[2, 3],
        help="Zarr storage format version (2 or 3)",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing daily GeoTIFFs and fill them with NaN slices in Zarr",
    )

    # --- DASK / PARALLELISM ---
    ap.add_argument(
        "--scheduler",
        choices=["distributed", "processes", "threads"],
        default="distributed",
        help="Parallel execution backend",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) // 4),
        help="Number of Dask workers (processes for 'distributed')",
    )
    ap.add_argument(
        "--threads-per-worker",
        type=int,
        default=2,
        help="Threads per Dask worker",
    )
    ap.add_argument(
        "--worker-mem",
        type=str,
        default="auto",
        help="Memory limit per worker (e.g. 8GB) or 'auto'",
    )
    ap.add_argument(
        "--dask-dashboard",
        action="store_true",
        help="Enable dashboard if using 'distributed'",
    )
    ap.add_argument(
        "--gdal-threads",
        type=int,
        default=1,
        help="Internal GDAL threads per rasterio read",
    )

    args = ap.parse_args()

    # DASK: start cluster/context
    with start_dask_cluster(
        scheduler=args.scheduler,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        worker_mem=args.worker_mem,
        dash=args.dask_dashboard,
    ):
        gdal_env = {
            "GDAL_NUM_THREADS": str(max(1, int(args.gdal_threads))),
            "GDAL_DISABLE_READDIR_ON_OPEN": "YES",
            "CPL_VSIL_CURL_USE_HEAD": "NO",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
        }

        with rasterio.Env(**gdal_env):
            # 1) discover files and enforce calendar
            raw_tiffs = find_tiffs(args.tiff_dir)
            tiffs, day_strings = enforce_calendar(
                raw_tiffs,
                start=args.start,
                end=args.end,
                allow_missing=args.allow_missing,
            )
            n_missing = sum(p is None for p in tiffs)
            print(
                f"[info] Expected calendar {args.start}..{args.end} with {len(day_strings)} days."
            )
            print(
                f"[info] Found {len(raw_tiffs)} TIFF files mapped to {len(tiffs)} days "
                f"({n_missing} missing days)."
            )

            # 2) validate stackability using non-missing files
            real_paths = [p for p in tiffs if p is not None]
            if not real_paths:
                raise RuntimeError("No non-missing files available for signature check.")
            sig = check_consistency(real_paths)
            print("[info] Signature:")
            print(
                json.dumps(
                    {
                        "crs": sig.crs,
                        "transform": sig.transform,
                        "resolution_deg_approx": (
                            abs(sig.transform[0]),
                            abs(sig.transform[4]),
                        ),
                        "width": sig.width,
                        "height": sig.height,
                        "bands(C)": sig.count,
                        "dtype": sig.dtype,
                        "nodata": sig.nodata,
                    },
                    indent=2,
                )
            )

            # 3) compressor & storage estimate
            comp = make_compressor(args.compressor, args.clevel)
            if args.estimate_compression:
                estimate_zarr_size_sample(
                    tiffs, args.chunk_y, args.chunk_x, comp, samples=min(12, len(tiffs))
                )

            # 4) build xarray stack lazily with NaN coercion (and NaN on missing days)
            print(
                "[info] Building stack with NoData/(-9999) → NaN (and NaN slices for missing days) …"
            )
            da_tc = build_time_stack_with_nan(
                tiffs, day_strings, chunks_xy=max(args.chunk_y, args.chunk_x)
            )
            print(f"[info] Logical stacked dims: {dict(da_tc.sizes)}")

            # 4b) derive peat_mask from first non-missing day
            first_real = real_paths[0]
            peat_mask = derive_and_check_peat_mask(first_real)
            print(
                f"[info] peat_mask: {peat_mask.shape} (1=peatland, 0=non-peat; derived from {first_real})"
            )

            # 5) write dataset
            print("[info] Writing Zarr with peat_mask …")
            T, C, H, W = write_dataset_with_mask(
                da_tc=da_tc,
                peat_mask=peat_mask,
                zarr_out=args.zarr_out,
                chunk_t=args.chunk_t,
                chunk_y=args.chunk_y,
                chunk_x=args.chunk_x,
                compressor=comp,
                consolidate=args.consolidate,
                overwrite=args.overwrite,
                zip_store=args.zip_store,
                zarr_format=args.zarr_format,
            )
            print(
                f"[done] Wrote Zarr: {args.zarr_out}  shape=(T={T},C={C},H={H},W={W})  "
                f"chunks=({args.chunk_t},{C},{args.chunk_y},{args.chunk_x})"
            )

            # 6) correctness checks
            print("[test] Validating pixel equivalence on a few samples …")
            pixel_match_check_nanaware(
                tiffs, day_strings, args.zarr_out, trials=min(3, len(tiffs))
            )

            # 7) optional micro-benchmark
            if args.bench_read:
                tiny_bench_read(args.zarr_out, n_patches=20, patch=min(256, H, W))

            print("[ok] All checks passed.")


if __name__ == "__main__":
    main()
