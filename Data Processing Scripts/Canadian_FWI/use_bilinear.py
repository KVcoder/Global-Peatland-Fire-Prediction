#!/usr/bin/env python3
"""
use_bilinear.py

Align a folder of daily GeoTIFFs to a template by searching for the best integer
translation (dy, dx) using the UNION of per-file valid masks.

Masking logic:
- For each source TIFF, validity is computed using all bands EXCEPT one ignored band
  (default: band 3).
- A pixel is valid iff:
    * all kept bands are finite
    * all kept bands are not nodata
    * NOT(all kept bands == 0)

Outputs keep ALL source bands.
For each band:
- original integer-shift placement is done first
- optional bilinear fill fills only still-missing pixels
- optional IDW fill fills only still-missing pixels
- already placed pixels are never overwritten

Additional behavior:
- Every source band is divided by 100 before alignment / fill / output.

Pixels outside template_valid remain nodata in every output band.

Also writes PNG footprints:
  _template_valid.png
  _aligned_union_valid.png
  _aligned_first_valid.png
  _template_missing_in_era.png
  _template_missing_after_fills.png

Deps:
  pip install rasterio numpy matplotlib affine scipy tqdm
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from affine import Affine
from rasterio.warp import reproject, Resampling
from scipy.spatial import cKDTree
from tqdm import tqdm


DATE_RE = re.compile(r"(\d{8})")


def extract_date_key(p: Path) -> str:
    m = DATE_RE.search(p.name)
    return m.group(1) if m else p.name


def check_same_grid(ref_ds: rasterio.DatasetReader, ds: rasterio.DatasetReader, path: Path) -> None:
    if ds.width != ref_ds.width or ds.height != ref_ds.height:
        raise SystemExit(
            f"Grid mismatch vs first file: {path}\n"
            f"  size {ds.width}x{ds.height} != {ref_ds.width}x{ref_ds.height}"
        )
    if ds.crs != ref_ds.crs:
        raise SystemExit(f"CRS mismatch vs first file: {path}\n  {ds.crs} != {ref_ds.crs}")
    if ds.transform != ref_ds.transform:
        raise SystemExit(f"Transform mismatch vs first file (alignment differs): {path}")


def compute_valid_mask_single_band(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    if nodata is None:
        return np.isfinite(arr)
    if isinstance(nodata, float) and np.isnan(nodata):
        return np.isfinite(arr)
    return np.isfinite(arr) & (arr != nodata)


def compute_multiband_valid_mask(
    ds: rasterio.DatasetReader,
    nodata: Optional[float],
    ignore_zero_band: int = 3,
) -> np.ndarray:
    band_ids = list(range(1, ds.count + 1))
    keep_bands = [b for b in band_ids if b != ignore_zero_band]

    if not keep_bands:
        raise SystemExit(
            f"No bands left after ignoring band {ignore_zero_band}. "
            f"Source has {ds.count} band(s)."
        )

    stacked = ds.read(keep_bands).astype(np.float32)

    all_finite = np.all(np.isfinite(stacked), axis=0)

    if nodata is None:
        all_not_nodata = np.ones(all_finite.shape, dtype=bool)
    elif isinstance(nodata, float) and np.isnan(nodata):
        all_not_nodata = np.all(np.isfinite(stacked), axis=0)
    else:
        all_not_nodata = np.all(stacked != nodata, axis=0)

    all_zero_except_ignored = np.all(stacked == 0, axis=0)

    valid_mask = all_finite & all_not_nodata & (~all_zero_except_ignored)
    return valid_mask


def place_with_shift_bool(src: np.ndarray, dst_shape: Tuple[int, int], dy: int, dx: int) -> np.ndarray:
    dst_h, dst_w = dst_shape
    src_h, src_w = src.shape
    dst = np.zeros((dst_h, dst_w), dtype=bool)

    dst_y0 = max(dy, 0)
    dst_x0 = max(dx, 0)
    src_y0 = max(-dy, 0)
    src_x0 = max(-dx, 0)

    copy_h = min(src_h - src_y0, dst_h - dst_y0)
    copy_w = min(src_w - src_x0, dst_w - dst_x0)
    if copy_h <= 0 or copy_w <= 0:
        return dst

    dst[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = src[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
    return dst


def place_with_shift_values(src: np.ndarray, dst_shape: Tuple[int, int], dy: int, dx: int, fill: float) -> np.ndarray:
    dst_h, dst_w = dst_shape
    src_h, src_w = src.shape
    dst = np.full((dst_h, dst_w), fill, dtype=np.float32)

    dst_y0 = max(dy, 0)
    dst_x0 = max(dx, 0)
    src_y0 = max(-dy, 0)
    src_x0 = max(-dx, 0)

    copy_h = min(src_h - src_y0, dst_h - dst_y0)
    copy_w = min(src_w - src_x0, dst_w - dst_x0)
    if copy_h <= 0 or copy_w <= 0:
        return dst

    dst[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = src[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w].astype(np.float32)
    return dst


def best_shift_fft(template_mask: np.ndarray, era_mask: np.ndarray, downsample: int = 8) -> Tuple[int, int, float]:
    ds = max(1, int(downsample))
    A = template_mask[::ds, ::ds].astype(np.float32)
    B = era_mask[::ds, ::ds].astype(np.float32)

    Ha, Wa = A.shape
    Hb, Wb = B.shape
    shape = (Ha + Hb - 1, Wa + Wb - 1)

    Bf = np.flipud(np.fliplr(B))

    FA = np.fft.rfft2(A, shape)
    FB = np.fft.rfft2(Bf, shape)
    conv = np.fft.irfft2(FA * FB, shape)

    iy, ix = np.unravel_index(np.argmax(conv), conv.shape)
    dy_ds = int(iy - (Hb - 1))
    dx_ds = int(ix - (Wb - 1))
    peak = float(conv[iy, ix])

    return dy_ds * ds, dx_ds * ds, peak


def score_masks(template_valid: np.ndarray, other_valid: np.ndarray) -> Dict[str, float]:
    ttot = int(template_valid.sum())
    otot = int(other_valid.sum())
    inter = int((template_valid & other_valid).sum())
    union = ttot + otot - inter
    iou = (inter / union) if union else 0.0
    coverage = (inter / ttot) if ttot else 0.0
    return {
        "template_total": float(ttot),
        "other_total": float(otot),
        "intersection": float(inter),
        "union": float(union),
        "iou": float(iou),
        "coverage": float(coverage),
    }


def extent_from_transform(transform: Affine, width: int, height: int):
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (width, height)
    xmin, xmax = (x0, x1) if x0 < x1 else (x1, x0)
    ymin, ymax = (y1, y0) if y1 < y0 else (y0, y1)
    return (xmin, xmax, ymin, ymax)


def plot_mask_png(mask: np.ndarray, transform: Affine, out_png: Path, title: str):
    img = mask.astype(np.uint8)
    cmap = ListedColormap([(1, 1, 1, 1), (0, 0, 0, 1)])
    ext = extent_from_transform(transform, mask.shape[1], mask.shape[0])

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_title(title)
    ax.imshow(img, cmap=cmap, interpolation="nearest", origin="upper", extent=ext, vmin=0, vmax=1)
    ax.set_xlabel("X (lon / projected)")
    ax.set_ylabel("Y (lat / projected)")
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)


def plot_diff_png(template_valid: np.ndarray, aligned_valid: np.ndarray, transform: Affine, out_png: Path, title: str):
    both = template_valid & aligned_valid
    t_only = template_valid & (~aligned_valid)
    a_only = aligned_valid & (~template_valid)

    img = np.zeros(template_valid.shape, dtype=np.uint8)
    img[both] = 1
    img[t_only] = 2
    img[a_only] = 3

    cmap = ListedColormap([
        (1, 1, 1, 1),
        (0, 0, 0, 1),
        (0.85, 0.2, 0.75, 1),
        (1.0, 0.55, 0.0, 1),
    ])

    ext = extent_from_transform(transform, template_valid.shape[1], template_valid.shape[0])

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_title(title)
    ax.imshow(img, cmap=cmap, interpolation="nearest", origin="upper", extent=ext)
    ax.set_xlabel("X (lon / projected)")
    ax.set_ylabel("Y (lat / projected)")
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)


def plot_red_missing_png(mask: np.ndarray, transform: Affine, out_png: Path, title: str):
    img = mask.astype(np.uint8)
    cmap = ListedColormap([
        (1, 1, 1, 1),
        (1.0, 0.0, 0.0, 1)
    ])
    ext = extent_from_transform(transform, mask.shape[1], mask.shape[0])

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_title(title)
    ax.imshow(img, cmap=cmap, interpolation="nearest", origin="upper", extent=ext, vmin=0, vmax=1)
    ax.set_xlabel("X (lon / projected)")
    ax.set_ylabel("Y (lat / projected)")
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)


def compute_union_valid_mask_from_folder(
    folder: Path,
    pattern: str,
    nodata: Optional[float],
    ignore_zero_band: int,
) -> Tuple[List[Path], np.ndarray, rasterio.profiles.Profile]:
    files: List[Path] = sorted(folder.glob(pattern), key=extract_date_key)
    if not files:
        raise SystemExit(f"No files matched {pattern} in {folder}")

    with rasterio.open(files[0]) as ref:
        use_nodata = nodata if nodata is not None else ref.nodata
        ref_profile = ref.profile.copy()

        valid_any = compute_multiband_valid_mask(ref, use_nodata, ignore_zero_band=ignore_zero_band)

        for p in files[1:]:
            with rasterio.open(p) as ds:
                check_same_grid(ref, ds, p)
                v = compute_multiband_valid_mask(ds, use_nodata, ignore_zero_band=ignore_zero_band)
                valid_any |= v

    return files, valid_any, ref_profile


def bilinear_fill_missing_only(
    src_band: np.ndarray,
    src_valid_mask: np.ndarray,
    src_transform: Affine,
    src_crs,
    dst_shape: Tuple[int, int],
    dst_transform: Affine,
    dst_crs,
    dy: int,
    dx: int,
    nodata_out: float,
) -> np.ndarray:
    src_for_reproj = np.full(src_band.shape, nodata_out, dtype=np.float32)
    src_for_reproj[src_valid_mask] = src_band[src_valid_mask].astype(np.float32)

    dst = np.full(dst_shape, nodata_out, dtype=np.float32)

    shifted_src_transform = src_transform * Affine.translation(dx, dy)

    reproject(
        source=src_for_reproj,
        destination=dst,
        src_transform=shifted_src_transform,
        src_crs=src_crs,
        src_nodata=nodata_out,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=nodata_out,
        resampling=Resampling.bilinear,
    )
    return dst


def idw_fill_missing_only(
    arr: np.ndarray,
    template_valid: np.ndarray,
    nodata_out: float,
    k: int = 8,
    power: float = 2.0,
    max_dist_px: Optional[float] = None,
    chunk_size: int = 200000,
) -> Tuple[np.ndarray, int]:
    out = arr.copy()

    seed_mask = template_valid & np.isfinite(out) & (out != nodata_out)
    missing_mask = template_valid & (~seed_mask)

    n_seed = int(seed_mask.sum())
    n_missing = int(missing_mask.sum())

    if n_missing == 0 or n_seed == 0:
        return out, 0

    seed_rc = np.column_stack(np.nonzero(seed_mask)).astype(np.float32)
    seed_vals = out[seed_mask].astype(np.float32)
    miss_rc = np.column_stack(np.nonzero(missing_mask)).astype(np.float32)

    tree = cKDTree(seed_rc)

    k_eff = max(1, min(int(k), n_seed))
    eps = 1e-12
    filled_total = 0

    for start in range(0, n_missing, chunk_size):
        stop = min(start + chunk_size, n_missing)
        q = miss_rc[start:stop]

        dists, idxs = tree.query(q, k=k_eff)

        if k_eff == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        weights = 1.0 / np.maximum(dists, eps) ** float(power)

        if max_dist_px is not None:
            weights[dists > float(max_dist_px)] = 0.0

        denom = weights.sum(axis=1)
        ok = denom > 0
        if not np.any(ok):
            continue

        vals = seed_vals[idxs]
        filled = np.sum(weights[ok] * vals[ok], axis=1) / denom[ok]

        q_ok = q[ok].astype(int)
        out[q_ok[:, 0], q_ok[:, 1]] = filled.astype(np.float32)
        filled_total += int(ok.sum())

    return out, filled_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--pattern", default="*.tif")
    ap.add_argument("--nodata", type=float, default=None)
    ap.add_argument("--ignore-zero-band", type=int, default=3)

    ap.add_argument("--template", required=True)
    ap.add_argument("--template-band", type=int, default=1)
    ap.add_argument("--template-nodata", type=float, default=None)
    ap.add_argument("--template-zero-invalid", action="store_true")
    ap.add_argument("--template-valid-min", type=float, default=None)

    ap.add_argument("--downsample", type=int, default=8)
    ap.add_argument("--refine", type=int, default=10)

    ap.add_argument("--fill-missing-bilinear", action="store_true")
    ap.add_argument("--fill-missing-idw", action="store_true")
    ap.add_argument("--idw-k", type=int, default=8)
    ap.add_argument("--idw-power", type=float, default=2.0)
    ap.add_argument("--idw-max-dist-px", type=float, default=None)
    ap.add_argument("--idw-chunk-size", type=int, default=200000)

    ap.add_argument("--out-folder", required=True)
    ap.add_argument("--out-diff", default=None)
    ap.add_argument("--stats-csv", default=None)

    args = ap.parse_args()

    NODATA_OUT = -9999.0
    SCALE_DIVISOR = 100.0

    folder = Path(args.folder)
    out_folder = Path(args.out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    files, valid_any_era, _era_profile = compute_union_valid_mask_from_folder(
        folder=folder,
        pattern=args.pattern,
        nodata=args.nodata,
        ignore_zero_band=args.ignore_zero_band,
    )

    print(f"ERA files: {len(files)}")
    print(f"ERA ever-valid px (multi-band, ignore band {args.ignore_zero_band} for zero-test): {int(valid_any_era.sum())}")

    with rasterio.open(Path(args.template)) as tds:
        tnodata = args.template_nodata if args.template_nodata is not None else tds.nodata
        tarr = tds.read(args.template_band).astype(np.float32)

        template_valid = compute_valid_mask_single_band(tarr, tnodata)
        if args.template_zero_invalid:
            template_valid &= (tarr != 0)
        if args.template_valid_min is not None:
            template_valid &= (tarr > float(args.template_valid_min))

        template_shape = tarr.shape
        template_transform = tds.transform
        template_profile = tds.profile.copy()
        template_crs = tds.crs

    print(f"Template valid px: {int(template_valid.sum())}")
    plot_mask_png(template_valid, template_transform, out_folder / "_template_valid.png", "Template valid footprint")

    dy0, dx0, peak = best_shift_fft(template_valid, valid_any_era, downsample=max(1, args.downsample))
    print(f"[FFT-fixed] dy,dx=({dy0},{dx0}) peak~{peak:.3g}")

    best_dy, best_dx = dy0, dx0
    best_aligned = place_with_shift_bool(valid_any_era, template_shape, best_dy, best_dx)
    best_stats = score_masks(template_valid, best_aligned)

    if args.refine and args.refine > 0:
        R = int(args.refine)
        best = (best_stats["coverage"], best_stats["iou"], best_dy, best_dx, best_aligned)
        for dy in range(dy0 - R, dy0 + R + 1):
            for dx in range(dx0 - R, dx0 + R + 1):
                aligned = place_with_shift_bool(valid_any_era, template_shape, dy, dx)
                st = score_masks(template_valid, aligned)
                cov, iou = st["coverage"], st["iou"]
                if (cov > best[0]) or (cov == best[0] and iou > best[1]):
                    best = (cov, iou, dy, dx, aligned)
        best_dy, best_dx, best_aligned = best[2], best[3], best[4]
        best_stats = score_masks(template_valid, best_aligned)
        print(f"[Refine] dy,dx=({best_dy},{best_dx})")

    print("\n=== Search-mask alignment ===")
    print(f"Shift dy,dx: ({best_dy},{best_dx})")
    print(f"IoU: {best_stats['iou']:.6f} | Coverage: {best_stats['coverage']:.6f} | Intersection: {int(best_stats['intersection'])}")

    if args.out_diff:
        plot_diff_png(
            template_valid,
            best_aligned,
            template_transform,
            Path(args.out_diff),
            f"Diff search-mask | shift=({best_dy},{best_dx})"
        )

    valid_any_on_template = place_with_shift_bool(valid_any_era, template_shape, best_dy, best_dx)

    print("\n=== Mask counts after shift ===")
    print(f"valid_any_on_template px: {int(valid_any_on_template.sum())}")
    print(f"template_valid px: {int(template_valid.sum())}")
    print(f"inter(template, valid_any_on_template): {int((template_valid & valid_any_on_template).sum())}")

    missing_in_era = template_valid & (~valid_any_on_template)
    print(f"template_missing_in_era px: {int(missing_in_era.sum())}")

    plot_red_missing_png(
        missing_in_era,
        template_transform,
        out_folder / "_template_missing_in_era.png",
        "Template-valid pixels missing from aligned ERA footprint"
    )

    per_rows: List[Dict[str, str]] = []
    union_valid = np.zeros(template_shape, dtype=bool)
    first_day_mask: Optional[np.ndarray] = None

    print("\n=== Writing aligned outputs ===")
    pbar = tqdm(files, desc="Writing aligned outputs", unit="file")

    for i, p in enumerate(pbar, 1):
        with rasterio.open(p) as ds:
            use_nodata = args.nodata if args.nodata is not None else ds.nodata

            out_profile = template_profile.copy()
            out_profile.update(
                dtype="float32",
                count=ds.count,
                nodata=NODATA_OUT,
                compress="deflate"
            )

            v = compute_multiband_valid_mask(ds, use_nodata, ignore_zero_band=args.ignore_zero_band)
            v_shift = place_with_shift_bool(v, template_shape, best_dy, best_dx)
            usable_base = v_shift & template_valid

            out_cube = np.full((ds.count, template_shape[0], template_shape[1]), NODATA_OUT, dtype=np.float32)

            filled_bilinear_total = 0
            filled_idw_total = 0

            for band_idx in range(1, ds.count + 1):
                a = ds.read(band_idx).astype(np.float32) / SCALE_DIVISOR

                a_shift = place_with_shift_values(a, template_shape, best_dy, best_dx, fill=NODATA_OUT)

                out_arr = np.full(template_shape, NODATA_OUT, dtype=np.float32)
                out_arr[usable_base] = a_shift[usable_base]

                if args.fill_missing_bilinear:
                    bilinear_arr = bilinear_fill_missing_only(
                        src_band=a,
                        src_valid_mask=v,
                        src_transform=ds.transform,
                        src_crs=ds.crs,
                        dst_shape=template_shape,
                        dst_transform=template_transform,
                        dst_crs=template_crs,
                        dy=best_dy,
                        dx=best_dx,
                        nodata_out=NODATA_OUT,
                    )

                    bilinear_valid = np.isfinite(bilinear_arr) & (bilinear_arr != NODATA_OUT)
                    current_valid = np.isfinite(out_arr) & (out_arr != NODATA_OUT)
                    fill_mask = template_valid & (~current_valid) & bilinear_valid

                    out_arr[fill_mask] = bilinear_arr[fill_mask]
                    filled_bilinear_total += int(fill_mask.sum())

                if args.fill_missing_idw:
                    out_arr, filled_idw = idw_fill_missing_only(
                        arr=out_arr,
                        template_valid=template_valid,
                        nodata_out=NODATA_OUT,
                        k=args.idw_k,
                        power=args.idw_power,
                        max_dist_px=args.idw_max_dist_px,
                        chunk_size=args.idw_chunk_size,
                    )
                    filled_idw_total += filled_idw

                out_cube[band_idx - 1] = out_arr

            usable = template_valid & np.any(np.isfinite(out_cube) & (out_cube != NODATA_OUT), axis=0)

            out_path = out_folder / p.name
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(out_cube)

            if first_day_mask is None:
                first_day_mask = usable.copy()

            st = score_masks(template_valid, usable)
            union_valid |= usable

            row = {
                "file": p.name,
                "iou": f"{st['iou']:.8f}",
                "coverage": f"{st['coverage']:.8f}",
                "intersection": str(int(st["intersection"])),
                "output_valid_total": str(int(st["other_total"])),
            }
            if args.fill_missing_bilinear:
                row["filled_bilinear_px_total"] = str(filled_bilinear_total)
            if args.fill_missing_idw:
                row["filled_idw_px_total"] = str(filled_idw_total)
            per_rows.append(row)

            pbar.set_postfix({
                "file": p.name[-18:],
                "cov": f"{st['coverage']:.4f}",
                "bil": filled_bilinear_total if args.fill_missing_bilinear else 0,
                "idw": filled_idw_total if args.fill_missing_idw else 0,
            })

    plot_mask_png(
        union_valid,
        template_transform,
        out_folder / "_aligned_union_valid.png",
        "Union valid footprint (all aligned outputs)"
    )

    if first_day_mask is not None:
        plot_mask_png(
            first_day_mask,
            template_transform,
            out_folder / "_aligned_first_valid.png",
            "First-day valid footprint (aligned output)"
        )

    missing_after_fills = template_valid & (~union_valid)
    plot_red_missing_png(
        missing_after_fills,
        template_transform,
        out_folder / "_template_missing_after_fills.png",
        "Template-valid pixels still missing after all fills"
    )

    union_stats = score_masks(template_valid, union_valid)
    print("\n=== UNION stats across outputs ===")
    print(f"Union IoU: {union_stats['iou']:.6f}")
    print(f"Union Coverage: {union_stats['coverage']:.6f}")
    print(f"Union intersection: {int(union_stats['intersection'])}")
    print(f"Union valid total: {int(union_stats['other_total'])}")
    print(f"Union missing after fills: {int(missing_after_fills.sum())}")

    if args.stats_csv and per_rows:
        csv_path = Path(args.stats_csv)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
            w.writeheader()
            for r in per_rows:
                w.writerow(r)
        print(f"Wrote CSV: {csv_path.resolve()}")

    print(f"\nDone. Output folder: {out_folder.resolve()}")
    print(f"Output nodata: {NODATA_OUT:g}")
    print(f"All output band values were divided by {SCALE_DIVISOR:g}")


if __name__ == "__main__":
    main()