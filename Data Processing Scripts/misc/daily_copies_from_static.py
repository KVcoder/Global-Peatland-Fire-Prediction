#!/usr/bin/env python3
"""
static_to_daily_copies_parallel.py

Expand ONE static raster into daily GeoTIFF files by copying the same data to every day.
Parallelized across multiple CPU cores by splitting the day list into chunks.

Band description behavior:
- If --set-band-desc is set, band 1 description is set to the value of --prefix
  (e.g., "pop_") for EVERY output file.

Deps:
  pip install rasterio numpy
(Optional progress bar)
  pip install tqdm

Examples:
  # Full year (2016), parallel
  python static_to_daily_copies_parallel.py \
    --src-tif POP_on_smap/pop_static.tif \
    --year 2016 \
    --out-dir POP_daily \
    --prefix "pop_" \
    --set-band-desc \
    --workers 8 \
    --chunk-days 31

  # Explicit date range (inclusive)
  python static_to_daily_copies_parallel.py \
    --src-tif POP_on_smap/pop_static.tif \
    --start 2016-01-01 --end 2018-12-31 \
    --out-dir POP_daily \
    --prefix "pop_" \
    --set-band-desc \
    --workers 6 \
    --chunk-days 45
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import rasterio

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_yyyy_mm_dd(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid date {s!r}. Expected YYYY-MM-DD.") from e


def days_in_year(year: int) -> List[date]:
    d = date(year, 1, 1)
    out: List[date] = []
    while d.year == year:
        out.append(d)
        d += timedelta(days=1)
    return out


def days_in_range(start: date, end: date) -> List[date]:
    if end < start:
        raise ValueError("--end must be >= --start")
    out: List[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def chunk_list(lst: List[date], n: int) -> List[List[date]]:
    if n <= 0:
        raise ValueError("--chunk-days must be >= 1")
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _worker_write_chunk(
    src_tif: str,
    band: int,
    out_dir: str,
    prefix: str,
    suffix: str,
    fmt: str,
    overwrite: bool,
    compress: str,
    set_band_desc: bool,
    days_chunk: List[date],
) -> tuple[int, int]:
    """
    Worker: opens source once, reads array once, writes all days in this chunk.
    Returns (wrote, skipped).
    """
    src_path = Path(src_tif)
    out_dir_p = Path(out_dir)

    wrote = 0
    skipped = 0

    with rasterio.open(src_path) as src:
        if band < 1 or band > src.count:
            raise ValueError(f"--band {band} out of range for {src_path.name} (has {src.count} bands)")

        arr = src.read(band)  # read once per worker
        profile = src.profile.copy()
        profile.update(count=1)

        # Compression settings
        if compress == "none":
            profile.pop("compress", None)
            profile.pop("predictor", None)
            profile.pop("zlevel", None)
        else:
            profile.update(compress=compress)
            dtype = str(profile.get("dtype", ""))
            profile.update(predictor=3 if dtype.startswith("float") else 2)

        # Requested behavior: band desc = prefix (fallback if empty)
        band_desc_value = prefix if prefix else "band1"

        for d in days_chunk:
            YYYY = f"{d.year:04d}"
            MM = f"{d.month:02d}"
            DD = f"{d.day:02d}"
            datestr = fmt.format(YYYY=YYYY, MM=MM, DD=DD)

            out_name = f"{prefix}{datestr}{suffix}.tif"
            out_path = out_dir_p / out_name

            if out_path.exists() and not overwrite:
                skipped += 1
                continue

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(arr, 1)
                if set_band_desc:
                    dst.set_band_description(1, band_desc_value)

            wrote += 1

    return wrote, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy ONE static raster into daily GeoTIFFs over a date range (parallel).")
    ap.add_argument("--src-tif", required=True, help="Path to the single source GeoTIFF to copy from.")
    ap.add_argument("--out-dir", required=True, help="Output folder for daily GeoTIFFs.")

    # Choose either --year OR (--start and --end)
    ap.add_argument("--year", type=int, help="Year to generate daily copies for (e.g., 2016).")
    ap.add_argument("--start", help="Start date (YYYY-MM-DD), inclusive.")
    ap.add_argument("--end", help="End date (YYYY-MM-DD), inclusive.")

    ap.add_argument("--band", type=int, default=1, help="1-based band to copy from source (default 1).")
    ap.add_argument("--prefix", default="", help='Output filename prefix (default ""). Example: "pop_".')
    ap.add_argument("--suffix", default="", help='Output filename suffix before .tif (default "").')
    ap.add_argument("--format", default="{YYYY}{MM}{DD}", help='Date token format (default "{YYYY}{MM}{DD}").')

    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing daily outputs.")
    ap.add_argument(
        "--compress",
        default="deflate",
        choices=["none", "lzw", "deflate", "zstd"],
        help="GeoTIFF compression for outputs (default deflate).",
    )
    ap.add_argument(
        "--set-band-desc",
        action="store_true",
        help='Set band description to the value of --prefix (same for every output).',
    )
    ap.add_argument("--quiet", action="store_true", help="Less logging.")

    # Parallel knobs
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes. 0 = use cpu_count()-1 (default 0). Use 1 to disable parallelism.",
    )
    ap.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        help="How many days each task handles (default 30). Larger = less overhead.",
    )

    args = ap.parse_args()

    src_path = Path(args.src_tif)
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    if not src_path.exists():
        raise FileNotFoundError(f"Missing --src-tif: {src_path}")

    # Build day list
    if args.year is not None:
        if args.start or args.end:
            raise SystemExit("Use either --year OR (--start and --end), not both.")
        all_days = days_in_year(args.year)
    else:
        if not (args.start and args.end):
            raise SystemExit("Provide --year OR both --start and --end.")
        all_days = days_in_range(parse_yyyy_mm_dd(args.start), parse_yyyy_mm_dd(args.end))

    if not all_days:
        raise SystemExit("No days to process.")

    # Determine workers
    workers = args.workers
    if workers == 0:
        workers = max(1, (os.cpu_count() or 2) - 1)
    if workers < 1:
        raise SystemExit("--workers must be >= 1")

    # Chunk days
    day_chunks = chunk_list(all_days, args.chunk_days)

    # If single worker, run in-process (simpler + sometimes faster on slow disks)
    total_wrote = 0
    total_skipped = 0

    if workers == 1 or len(day_chunks) == 1:
        for chunk in (tqdm(day_chunks, desc="Chunks", unit="chunk") if (tqdm and not args.quiet) else day_chunks):
            wrote, skipped = _worker_write_chunk(
                str(src_path),
                args.band,
                str(out_dir),
                args.prefix,
                args.suffix,
                args.format,
                args.overwrite,
                args.compress,
                args.set_band_desc,
                chunk,
            )
            total_wrote += wrote
            total_skipped += skipped
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    _worker_write_chunk,
                    str(src_path),
                    args.band,
                    str(out_dir),
                    args.prefix,
                    args.suffix,
                    args.format,
                    args.overwrite,
                    args.compress,
                    args.set_band_desc,
                    chunk,
                )
                for chunk in day_chunks
            ]

            fut_iter = as_completed(futures)
            if tqdm is not None and not args.quiet:
                fut_iter = tqdm(fut_iter, total=len(futures), desc="Chunks", unit="chunk")

            for fut in fut_iter:
                wrote, skipped = fut.result()
                total_wrote += wrote
                total_skipped += skipped

    if not args.quiet:
        print(f"Done. Wrote {total_wrote} files, skipped {total_skipped} (existing).")
        print(f"Workers: {workers} | chunk-days: {args.chunk_days} | outputs: {out_dir}")


if __name__ == "__main__":
    main()
