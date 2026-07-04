#!/usr/bin/env python3
"""
Visualize daily 1-band prediction GeoTIFFs using percentile-based color mapping.

Default behavior
----------------
- Reads a folder of daily 1-band GeoTIFFs (e.g., FWI_YYYYMMDD.tif)
- Computes a *global* percentile scale from a sampled distribution across all files
- Renders one PNG per day using a continuous color gradient:
    gray -> blue -> green -> yellow -> orange -> red
- Masks invalid/nodata pixels so only valid regions are drawn
- Writes a percentile scale report for reproducibility

Why global percentile scaling?
------------------------------
Global scaling keeps colors comparable across days. A red pixel in January and a red
pixel in August both mean "high percentile relative to the whole run", not just a
high value relative to that single day.

Notes
-----
The global percentile CDF is estimated from a random sample of valid pixels across the
folder for speed and memory efficiency. For large global rasters, this is the practical
option and is usually visually stable.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Use a non-interactive backend for multiprocessing / headless runs.
import matplotlib
matplotlib.use("Agg")


DATE_RE = re.compile(r"(\d{8})")


@dataclass
class RenderConfig:
    out_dir: str
    dpi: int
    vmin_p: float
    vmax_p: float
    title_prefix: str
    colorbar_label: str
    transparent: bool
    show_axes: bool
    figsize_w: float
    figsize_h: float
    scale_mode: str
    q_probs: Optional[np.ndarray]
    q_vals: Optional[np.ndarray]
    day_quantile_resolution: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render daily GeoTIFF predictions as percentile-colored PNG maps."
    )
    p.add_argument("input_dir", help="Folder containing daily 1-band GeoTIFF files")
    p.add_argument("output_dir", help="Folder to write PNG visualizations")
    p.add_argument(
        "--pattern",
        default="FWI_*.tif",
        help='Glob pattern for inputs (default: "FWI_*.tif")',
    )
    p.add_argument(
        "--scale-mode",
        choices=["global", "per-day"],
        default="global",
        help="Percentile scaling mode (default: global)",
    )
    p.add_argument(
        "--sample-per-file",
        type=int,
        default=10000,
        help="Max valid pixels to sample per file when building the global percentile scale (default: 10000)",
    )
    p.add_argument(
        "--global-quantile-resolution",
        type=int,
        default=1000,
        help="Number of quantile intervals used to approximate the global percentile CDF (default: 1000)",
    )
    p.add_argument(
        "--day-quantile-resolution",
        type=int,
        default=1000,
        help="Number of quantile intervals used in per-day mode (default: 1000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for global sampling (default: 42)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG DPI (default: 200)",
    )
    p.add_argument(
        "--figsize",
        default="14,6",
        help='Figure size in inches as "width,height" (default: 14,6)',
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for PNG rendering (default: 1)",
    )
    p.add_argument(
        "--title-prefix",
        default="FWI",
        help='Title prefix, e.g. "FWI" (default: FWI)',
    )
    p.add_argument(
        "--colorbar-label",
        default="Percentile",
        help='Colorbar label (default: "Percentile")',
    )
    p.add_argument(
        "--percentile-range",
        default="0,100",
        help='Percentile display range as "min,max" (default: 0,100)',
    )
    p.add_argument(
        "--transparent",
        action="store_true",
        help="Save PNGs with transparent background / nodata",
    )
    p.add_argument(
        "--hide-axes",
        action="store_true",
        help="Hide lon/lat axes for cleaner images",
    )
    p.add_argument(
        "--report-file",
        default=None,
        help="Optional path for the percentile-scale report text file (default: <output_dir>/percentile_scale_report.txt)",
    )
    return p.parse_args()


def get_files(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern {pattern!r} in {input_dir}")
    return files


def parse_date_str(path: Path) -> str:
    m = DATE_RE.search(path.name)
    if m:
        s = m.group(1)
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return path.stem


def make_cmap() -> LinearSegmentedColormap:
    colors = [
        "#8c8c8c",  # gray
        "#2b83ba",  # blue
        "#1a9850",  # green
        "#fee08b",  # yellow
        "#f46d43",  # orange
        "#d73027",  # red
    ]
    cmap = LinearSegmentedColormap.from_list("gray_blue_green_yellow_orange_red", colors, N=256)
    cmap.set_bad((1, 1, 1, 0))
    return cmap


def read_valid_values(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32, copy=False)
        nd = src.nodata
        valid = np.isfinite(arr)
        if nd is not None:
            valid &= arr != nd
        vals = arr[valid]
        return vals


def sample_values(vals: np.ndarray, sample_per_file: int, rng: np.random.Generator) -> np.ndarray:
    n = vals.size
    if n == 0:
        return vals
    if n <= sample_per_file:
        return vals.astype(np.float32, copy=False)
    idx = rng.choice(n, size=sample_per_file, replace=False)
    return vals[idx].astype(np.float32, copy=False)


def build_global_quantiles(
    files: Sequence[Path],
    sample_per_file: int,
    quantile_resolution: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    samples = []
    total_valid = 0
    used_files = 0

    for path in files:
        vals = read_valid_values(path)
        total_valid += int(vals.size)
        if vals.size == 0:
            continue
        samples.append(sample_values(vals, sample_per_file, rng))
        used_files += 1

    if not samples:
        raise ValueError("No valid pixels found across the input files.")

    sample = np.concatenate(samples).astype(np.float32, copy=False)
    q_probs = np.linspace(0.0, 100.0, quantile_resolution + 1, dtype=np.float32)
    q_vals = np.percentile(sample, q_probs).astype(np.float32)

    stats = {
        "sample_size": int(sample.size),
        "used_files": int(used_files),
        "total_valid_pixels_seen": int(total_valid),
        "min": float(np.min(sample)),
        "max": float(np.max(sample)),
        "mean": float(np.mean(sample)),
    }
    return q_probs, q_vals, stats


def percentile_map_from_quantiles(
    values: np.ndarray,
    q_probs: np.ndarray,
    q_vals: np.ndarray,
) -> np.ndarray:
    # Map raw values -> approximate percentile rank using monotonic quantile anchors.
    # np.interp handles repeated q_vals fine by using the last matching interval.
    return np.interp(values, q_vals, q_probs, left=q_probs[0], right=q_probs[-1]).astype(np.float32)


def make_day_quantiles(values: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    q_probs = np.linspace(0.0, 100.0, resolution + 1, dtype=np.float32)
    q_vals = np.percentile(values, q_probs).astype(np.float32)
    return q_probs, q_vals


def get_extent(src: rasterio.io.DatasetReader) -> tuple[float, float, float, float]:
    bounds = src.bounds
    return (bounds.left, bounds.right, bounds.bottom, bounds.top)


def render_one(path_str: str, cfg: RenderConfig) -> str:
    path = Path(path_str)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.png"

    cmap = make_cmap()

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32, copy=False)
        nd = src.nodata
        valid = np.isfinite(arr)
        if nd is not None:
            valid &= arr != nd

        if not np.any(valid):
            raise ValueError(f"No valid pixels in {path}")

        if cfg.scale_mode == "global":
            assert cfg.q_probs is not None and cfg.q_vals is not None
            pvals = percentile_map_from_quantiles(arr[valid], cfg.q_probs, cfg.q_vals)
        else:
            q_probs, q_vals = make_day_quantiles(arr[valid], cfg.day_quantile_resolution)
            pvals = percentile_map_from_quantiles(arr[valid], q_probs, q_vals)

        vis = np.full(arr.shape, np.nan, dtype=np.float32)
        vis[valid] = pvals

        extent = get_extent(src)

    vmin = cfg.vmin_p
    vmax = cfg.vmax_p
    date_str = parse_date_str(path)

    fig, ax = plt.subplots(figsize=(cfg.figsize_w, cfg.figsize_h), constrained_layout=True)
    if cfg.transparent:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((1, 1, 1, 0))

    im = ax.imshow(
        np.ma.masked_invalid(vis),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="upper",
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_title(f"{cfg.title_prefix} — {date_str}")
    if cfg.show_axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        ax.set_axis_off()

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05)
    cbar.set_label(cfg.colorbar_label)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])

    fig.savefig(out_path, dpi=cfg.dpi, transparent=cfg.transparent)
    plt.close(fig)
    return str(out_path)


def write_report(
    report_path: Path,
    files: Sequence[Path],
    args: argparse.Namespace,
    global_stats: Optional[dict],
    q_probs: Optional[np.ndarray],
    q_vals: Optional[np.ndarray],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Daily percentile visualization report\n")
        f.write(f"Input dir: {Path(args.input_dir).resolve()}\n")
        f.write(f"Output dir: {Path(args.output_dir).resolve()}\n")
        f.write(f"Pattern: {args.pattern}\n")
        f.write(f"Files matched: {len(files)}\n")
        f.write(f"Scale mode: {args.scale_mode}\n")
        f.write(f"Color gradient: gray -> blue -> green -> yellow -> orange -> red\n")
        f.write(f"Percentile display range: {args.percentile_range}\n")
        f.write(f"Sample per file: {args.sample_per_file}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Workers: {args.workers}\n")
        f.write("\n")

        if args.scale_mode == "global" and global_stats is not None and q_probs is not None and q_vals is not None:
            f.write("# Global sampled percentile scale\n")
            for k, v in global_stats.items():
                f.write(f"{k}: {v}\n")
            f.write("\n# Percentile anchors\n")
            for qp, qv in zip(q_probs, q_vals):
                if int(qp) == qp:
                    f.write(f"P{int(qp):03d} = {float(qv):.8g}\n")
                else:
                    f.write(f"P{float(qp):07.3f} = {float(qv):.8g}\n")
        elif args.scale_mode == "per-day":
            f.write("# Per-day scaling selected\n")
            f.write("Each PNG was colored using percentiles computed from that day only.\n")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        figsize_w, figsize_h = [float(x) for x in args.figsize.split(",")]
    except Exception as e:
        raise ValueError("--figsize must look like '14,6'") from e

    try:
        vmin_p, vmax_p = [float(x) for x in args.percentile_range.split(",")]
    except Exception as e:
        raise ValueError("--percentile-range must look like '0,100'") from e
    if not (0.0 <= vmin_p < vmax_p <= 100.0):
        raise ValueError("--percentile-range must satisfy 0 <= min < max <= 100")

    files = get_files(input_dir, args.pattern)

    q_probs = None
    q_vals = None
    global_stats = None
    if args.scale_mode == "global":
        print("Building global sampled percentile scale...")
        q_probs, q_vals, global_stats = build_global_quantiles(
            files=files,
            sample_per_file=args.sample_per_file,
            quantile_resolution=args.global_quantile_resolution,
            seed=args.seed,
        )
        print(
            f"Global sample built from {global_stats['used_files']} files; "
            f"sample_size={global_stats['sample_size']} "
            f"min={global_stats['min']:.6g} max={global_stats['max']:.6g}"
        )

    cfg = RenderConfig(
        out_dir=str(output_dir),
        dpi=args.dpi,
        vmin_p=vmin_p,
        vmax_p=vmax_p,
        title_prefix=args.title_prefix,
        colorbar_label=args.colorbar_label,
        transparent=args.transparent,
        show_axes=not args.hide_axes,
        figsize_w=figsize_w,
        figsize_h=figsize_h,
        scale_mode=args.scale_mode,
        q_probs=q_probs,
        q_vals=q_vals,
        day_quantile_resolution=args.day_quantile_resolution,
    )

    print(f"Rendering {len(files)} daily PNGs...")
    if args.workers <= 1:
        for i, path in enumerate(files, start=1):
            out = render_one(str(path), cfg)
            if i == 1 or i % 25 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] {Path(out).name}")
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(render_one, str(path), cfg): path for path in files}
            for fut in as_completed(futures):
                _ = fut.result()
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == len(files):
                    print(f"[{completed}/{len(files)}] done")

    report_path = Path(args.report_file) if args.report_file else (output_dir / "percentile_scale_report.txt")
    write_report(report_path, files, args, global_stats, q_probs, q_vals)
    print(f"Done. PNGs written to: {output_dir}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
