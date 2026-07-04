#!/usr/bin/env python3
"""
viz_gradient_and_maskcheck.py

Visualize a GeoTIFF band with a gradient where higher values are darker,
and optionally compare its valid-pixel mask against a template GeoTIFF.

Main features
-------------
1) Value visualization (default):
   - Invalid pixels (NoData / non-finite) are masked out
   - Uses a colormap that can make higher values darker (default: Greens)
   - Optional faint valid-mask overlay

2) Mask-only visualization:
   - Shows only valid footprint (valid pixels highlighted)

3) Template mask comparison (optional):
   - Confirms whether valid pixels match a template TIFF
   - Reports:
       * exact equality of valid masks
       * template-valid => input-valid (subset check; your example)
       * input-valid => template-valid (reverse subset)
   - Can also render a mismatch map:
       * dark green = valid in both
       * magenta   = valid in template only (FAIL for your example)
       * orange    = valid in input only

Deps
----
  pip install rasterio numpy matplotlib

Examples
--------
# 1) Show values with darker-high gradient
python viz_gradient_and_maskcheck.py input.tif --out values.png

# 2) Mask-only view
python viz_gradient_and_maskcheck.py input.tif --mask-only --out mask.png

# 3) Compare valid mask to template, print PASS/FAIL
python viz_gradient_and_maskcheck.py input.tif --template template.tif --mask-only

# 4) Compare + save mismatch visualization
python viz_gradient_and_maskcheck.py input.tif \
  --template template.tif \
  --compare-mask-plot compare_mask.png

# 5) Override NoData values
python viz_gradient_and_maskcheck.py input.tif \
  --template template.tif \
  --nodata -9999 --template-nodata -9999
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from affine import Affine


def extent_from_transform(transform: Affine, width: int, height: int) -> Tuple[float, float, float, float]:
    """Matplotlib extent = (xmin, xmax, ymin, ymax)"""
    x0, y0 = transform * (0, 0)               # top-left
    x1, y1 = transform * (width, height)      # bottom-right
    xmin, xmax = (x0, x1) if x0 < x1 else (x1, x0)
    ymin, ymax = (y1, y0) if y1 < y0 else (y0, y1)
    return (xmin, xmax, ymin, ymax)


def read_downsampled(
    ds: rasterio.DatasetReader,
    band: int,
    max_dim: int,
    resampling: Resampling,
) -> Tuple[np.ndarray, Affine]:
    """Read band possibly downsampled; returns (data, adjusted_transform)."""
    H, W = ds.height, ds.width
    scale = max(H / max_dim, W / max_dim, 1.0)
    out_h = max(1, int(round(H / scale)))
    out_w = max(1, int(round(W / scale)))

    data = ds.read(
        band,
        out_shape=(out_h, out_w),
        resampling=resampling,
    )

    # Adjust transform for the resampled grid
    transform = ds.transform * Affine.scale(W / out_w, H / out_h)
    return data, transform


def compute_valid_mask(data: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """True where valid (finite and not equal to nodata if nodata is defined)."""
    if nodata is None:
        return np.isfinite(data)
    return np.isfinite(data) & (data != nodata)


def transforms_close(t1: Affine, t2: Affine, atol: float = 1e-9) -> bool:
    a = (t1.a, t1.b, t1.c, t1.d, t1.e, t1.f)
    b = (t2.a, t2.b, t2.c, t2.d, t2.e, t2.f)
    return all(abs(x - y) <= atol for x, y in zip(a, b))


def same_grid(ds_a: rasterio.DatasetReader, ds_b: rasterio.DatasetReader, atol: float = 1e-9) -> Tuple[bool, str]:
    """Check whether two datasets are on the same pixel grid (strict)."""
    if ds_a.width != ds_b.width or ds_a.height != ds_b.height:
        return False, f"size mismatch: input=({ds_a.width}x{ds_a.height}), template=({ds_b.width}x{ds_b.height})"
    if ds_a.crs != ds_b.crs:
        return False, f"CRS mismatch: input={ds_a.crs}, template={ds_b.crs}"
    if not transforms_close(ds_a.transform, ds_b.transform, atol=atol):
        return False, "transform mismatch (pixel alignment differs)"
    return True, "same grid"


def compare_valid_masks_streaming(
    ds_input: rasterio.DatasetReader,
    band_input: int,
    nodata_input: Optional[float],
    ds_template: rasterio.DatasetReader,
    band_template: int,
    nodata_template: Optional[float],
) -> dict:
    """
    Compare valid masks block-by-block (memory-friendly).
    Assumes same grid has already been verified.
    """
    total_pixels = 0
    input_valid_n = 0
    template_valid_n = 0
    both_valid_n = 0
    template_only_n = 0
    input_only_n = 0
    both_invalid_n = 0

    # Iterate using input blocks; same window read from template
    for _, window in ds_input.block_windows(band_input):
        a = ds_input.read(band_input, window=window)
        b = ds_template.read(band_template, window=window)

        valid_a = compute_valid_mask(a, nodata_input)
        valid_b = compute_valid_mask(b, nodata_template)

        total_pixels += valid_a.size

        both_valid = valid_a & valid_b
        template_only = valid_b & (~valid_a)   # template valid but input invalid (FAIL for template=>input)
        input_only = valid_a & (~valid_b)
        both_invalid = (~valid_a) & (~valid_b)

        input_valid_n += int(valid_a.sum())
        template_valid_n += int(valid_b.sum())
        both_valid_n += int(both_valid.sum())
        template_only_n += int(template_only.sum())
        input_only_n += int(input_only.sum())
        both_invalid_n += int(both_invalid.sum())

    exact_equal = (template_only_n == 0) and (input_only_n == 0)
    template_subset_of_input = (template_only_n == 0)   # template valid => input valid
    input_subset_of_template = (input_only_n == 0)

    return {
        "total_pixels": total_pixels,
        "input_valid_n": input_valid_n,
        "template_valid_n": template_valid_n,
        "both_valid_n": both_valid_n,
        "template_only_n": template_only_n,
        "input_only_n": input_only_n,
        "both_invalid_n": both_invalid_n,
        "exact_equal": exact_equal,
        "template_subset_of_input": template_subset_of_input,
        "input_subset_of_template": input_subset_of_template,
    }


def print_compare_report(stats: dict) -> None:
    total = stats["total_pixels"]
    def pct(n: int) -> float:
        return 100.0 * n / total if total else 0.0

    print("\n=== Valid-mask comparison report ===")
    print(f"Total pixels          : {total}")
    print(f"Input valid           : {stats['input_valid_n']} ({pct(stats['input_valid_n']):.4f}%)")
    print(f"Template valid        : {stats['template_valid_n']} ({pct(stats['template_valid_n']):.4f}%)")
    print(f"Valid in BOTH         : {stats['both_valid_n']} ({pct(stats['both_valid_n']):.4f}%)")
    print(f"Template-only valid   : {stats['template_only_n']} ({pct(stats['template_only_n']):.4f}%)")
    print(f"Input-only valid      : {stats['input_only_n']} ({pct(stats['input_only_n']):.4f}%)")
    print(f"Invalid in BOTH       : {stats['both_invalid_n']} ({pct(stats['both_invalid_n']):.4f}%)")

    print("\nChecks:")
    if stats["exact_equal"]:
        print("  PASS: exact valid-mask match (same highlighted pixels).")
    else:
        print("  FAIL: valid masks are not exactly the same.")

    if stats["template_subset_of_input"]:
        print("  PASS: template-valid => input-valid (your requested condition).")
    else:
        print("  FAIL: some pixels are valid in template but invalid in input (template-only valid > 0).")

    if stats["input_subset_of_template"]:
        print("  PASS: input-valid => template-valid.")
    else:
        print("  NOTE: input has extra valid pixels not present in template.")


def build_compare_preview(
    ds_input: rasterio.DatasetReader,
    band_input: int,
    nodata_input: Optional[float],
    ds_template: rasterio.DatasetReader,
    band_template: int,
    nodata_template: Optional[float],
    max_dim: int,
) -> Tuple[np.ndarray, Affine]:
    """
    Build a downsampled categorical comparison image:
      0 = both invalid
      1 = both valid
      2 = template-only valid (template valid, input invalid)
      3 = input-only valid    (input valid, template invalid)
    """
    # Nearest resampling preserves masks better for preview
    a, t = read_downsampled(ds_input, band_input, max_dim=max_dim, resampling=Resampling.nearest)
    b, t2 = read_downsampled(ds_template, band_template, max_dim=max_dim, resampling=Resampling.nearest)

    # same_grid already validated at full res; downsampled shapes should match, but guard anyway
    if a.shape != b.shape:
        raise RuntimeError(f"Downsampled shapes differ: input={a.shape}, template={b.shape}")
    if not transforms_close(t, t2, atol=1e-9):
        # This should generally match, but not fatal if dimensions match and the two previews align visually.
        pass

    valid_a = compute_valid_mask(a, nodata_input)
    valid_b = compute_valid_mask(b, nodata_template)

    cmp_img = np.zeros(a.shape, dtype=np.uint8)
    cmp_img[valid_a & valid_b] = 1
    cmp_img[valid_b & (~valid_a)] = 2  # template-only valid (FAIL for template=>input)
    cmp_img[valid_a & (~valid_b)] = 3  # input-only valid
    return cmp_img, t


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize GeoTIFF values with dark-high gradient and optionally compare valid mask to a template."
    )
    ap.add_argument("tif", help="Input GeoTIFF path (the file you want to visualize/check).")
    ap.add_argument("--band", type=int, default=1, help="Input band index (1-based). Default 1.")

    # Template compare
    ap.add_argument("--template", default=None, help="Template GeoTIFF for valid-mask comparison.")
    ap.add_argument("--template-band", type=int, default=1, help="Template band index (1-based). Default 1.")
    ap.add_argument("--template-nodata", type=float, default=None, help="Override NoData for template TIFF.")

    # Nodata override
    ap.add_argument("--nodata", type=float, default=None, help="Override NoData for input TIFF.")

    # Visualization options
    ap.add_argument("--out", default=None, help="Output PNG path. If omitted, opens a window (if supported).")
    ap.add_argument("--title", default=None, help="Plot title override.")
    ap.add_argument("--max-dim", type=int, default=1400, help="Downsample max dimension for visualization speed.")
    ap.add_argument("--resampling", default="nearest", help="nearest|bilinear (default nearest).")
    ap.add_argument("--mask-only", action="store_true", help="Show only valid footprint (valid=dark green).")
    ap.add_argument("--mask-alpha", type=float, default=0.20, help="Alpha for valid-mask overlay (default 0.20).")
    ap.add_argument("--mask-color", default="#006400", help="Valid mask color for overlay / mask-only (default dark green).")
    ap.add_argument("--show-bounds", action="store_true", help="Draw dataset bounds outline.")

    # Gradient styling (dark-high)
    ap.add_argument("--cmap", default="Greens", help="Matplotlib colormap (default Greens; higher values appear darker).")
    ap.add_argument("--reverse-cmap", action="store_true", help="Reverse the chosen colormap (useful if it darkens low values).")
    ap.add_argument("--pmin", type=float, default=2.0, help="Lower percentile for display scaling (default 2).")
    ap.add_argument("--pmax", type=float, default=98.0, help="Upper percentile for display scaling (default 98).")
    ap.add_argument("--vmin", type=float, default=None, help="Fixed min value for display (overrides percentile min).")
    ap.add_argument("--vmax", type=float, default=None, help="Fixed max value for display (overrides percentile max).")

    # Compare visualization
    ap.add_argument(
        "--compare-mask-plot",
        default=None,
        help="Optional PNG path for mismatch visualization (requires --template).",
    )
    ap.add_argument(
        "--fail-on-mask-mismatch",
        action="store_true",
        help="Exit with code 2 if exact valid-mask match fails.",
    )
    ap.add_argument(
        "--fail-on-template-not-subset",
        action="store_true",
        help="Exit with code 3 if template-valid => input-valid fails (your requested condition).",
    )

    args = ap.parse_args()

    tif_path = Path(args.tif)
    if not tif_path.exists():
        raise SystemExit(f"File not found: {tif_path}")

    if args.template is not None:
        template_path = Path(args.template)
        if not template_path.exists():
            raise SystemExit(f"Template file not found: {template_path}")
    else:
        template_path = None

    # resampling for value display
    resampling_name = args.resampling.lower().strip()
    if resampling_name == "nearest":
        rs = Resampling.nearest
    elif resampling_name == "bilinear":
        rs = Resampling.bilinear
    else:
        raise SystemExit("--resampling must be nearest or bilinear")

    exit_code = 0

    with rasterio.open(tif_path) as ds:
        if args.band < 1 or args.band > ds.count:
            raise SystemExit(f"--band must be in [1, {ds.count}] (got {args.band})")

        input_nodata = args.nodata if args.nodata is not None else ds.nodata

        # --- Optional template comparison (full-res, strict grid) ---
        compare_stats = None
        if template_path is not None:
            with rasterio.open(template_path) as dtmp:
                if args.template_band < 1 or args.template_band > dtmp.count:
                    raise SystemExit(f"--template-band must be in [1, {dtmp.count}] (got {args.template_band})")

                template_nodata = args.template_nodata if args.template_nodata is not None else dtmp.nodata

                ok, reason = same_grid(ds, dtmp)
                if not ok:
                    raise SystemExit(
                        "Cannot compare valid pixels exactly because grids differ.\n"
                        f"Reason: {reason}\n"
                        "Make sure both TIFFs have the same width/height/CRS/transform (same pixel alignment)."
                    )

                compare_stats = compare_valid_masks_streaming(
                    ds_input=ds,
                    band_input=args.band,
                    nodata_input=input_nodata,
                    ds_template=dtmp,
                    band_template=args.template_band,
                    nodata_template=template_nodata,
                )
                print_compare_report(compare_stats)

                if args.fail_on_mask_mismatch and not compare_stats["exact_equal"]:
                    exit_code = max(exit_code, 2)
                if args.fail_on_template_not_subset and not compare_stats["template_subset_of_input"]:
                    exit_code = max(exit_code, 3)

                # Optional compare visualization PNG
                if args.compare_mask_plot:
                    cmp_img, cmp_t = build_compare_preview(
                        ds_input=ds,
                        band_input=args.band,
                        nodata_input=input_nodata,
                        ds_template=dtmp,
                        band_template=args.template_band,
                        nodata_template=template_nodata,
                        max_dim=args.max_dim,
                    )
                    ext_cmp = extent_from_transform(cmp_t, cmp_img.shape[1], cmp_img.shape[0])

                    fig2, ax2 = plt.subplots(figsize=(12, 7))
                    ax2.set_title(f"Valid-mask comparison: {tif_path.name} vs {template_path.name}")

                    cmp_cmap = ListedColormap([
                        (0, 0, 0, 0),      # 0 both invalid (transparent)
                        (0.0, 0.39, 0.0, 1.0),  # 1 both valid (dark green)
                        (0.85, 0.20, 0.75, 1.0), # 2 template-only valid (magenta)
                        (1.00, 0.55, 0.00, 1.0), # 3 input-only valid (orange)
                    ])
                    ax2.imshow(cmp_img, extent=ext_cmp, origin="upper", interpolation="nearest", cmap=cmp_cmap)

                    legend_handles = [
                        Patch(facecolor=(0.0, 0.39, 0.0, 1.0), edgecolor='none', label="Valid in both"),
                        Patch(facecolor=(0.85, 0.20, 0.75, 1.0), edgecolor='none', label="Template-only valid"),
                        Patch(facecolor=(1.00, 0.55, 0.00, 1.0), edgecolor='none', label="Input-only valid"),
                    ]
                    ax2.legend(handles=legend_handles, loc="lower left", framealpha=0.95)

                    ax2.set_xlabel("X (lon / projected)")
                    ax2.set_ylabel("Y (lat / projected)")
                    ax2.grid(True, linewidth=0.3)
                    plt.tight_layout()

                    cmp_out = Path(args.compare_mask_plot)
                    cmp_out.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(cmp_out, dpi=200)
                    plt.close(fig2)
                    print(f"Wrote compare mask plot: {cmp_out.resolve()}")

        # --- Main visualization of input TIFF ---
        data, t2 = read_downsampled(ds, args.band, max_dim=args.max_dim, resampling=rs)
        valid = compute_valid_mask(data, input_nodata)

        h2, w2 = data.shape
        ext = extent_from_transform(t2, w2, h2)

        title = args.title or f"{tif_path.name} (band {args.band})"
        if input_nodata is not None:
            title += f" | nodata={input_nodata:g}"
        if compare_stats is not None:
            # short summary in title
            title += " | mask="
            title += "MATCH" if compare_stats["exact_equal"] else "MISMATCH"

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_title(title)

        if args.mask_only:
            mask_img = valid.astype(np.uint8)
            mask_cmap = ListedColormap([(0, 0, 0, 0), args.mask_color])  # 0 transparent, 1 chosen color
            ax.imshow(
                mask_img,
                extent=ext,
                origin="upper",
                interpolation="nearest",
                cmap=mask_cmap,
            )
        else:
            if not np.any(valid):
                raise SystemExit("No valid pixels found in input TIFF after applying nodata/non-finite mask.")

            valid_vals = data[valid].astype(np.float64)

            # Display scaling (percentile-based unless vmin/vmax provided)
            if args.vmin is not None:
                vmin = float(args.vmin)
            else:
                vmin = float(np.nanpercentile(valid_vals, args.pmin))

            if args.vmax is not None:
                vmax = float(args.vmax)
            else:
                vmax = float(np.nanpercentile(valid_vals, args.pmax))

            if not np.isfinite(vmin) or not np.isfinite(vmax):
                raise SystemExit("Computed vmin/vmax are not finite.")
            if vmax < vmin:
                vmin, vmax = vmax, vmin
            if vmax == vmin:
                # avoid zero-range normalization
                vmax = vmin + 1e-12

            cmap_name = args.cmap + ("_r" if args.reverse_cmap else "")
            mdata = np.ma.array(data, mask=~valid)

            im = ax.imshow(
                mdata,
                extent=ext,
                origin="upper",
                interpolation="nearest",
                cmap=cmap_name,
                norm=Normalize(vmin=vmin, vmax=vmax, clip=True),
            )
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Value (higher = darker)")

            # faint valid-mask overlay so footprint is obvious
            mask_img = valid.astype(np.uint8)
            overlay_cmap = ListedColormap([(0, 0, 0, 0), args.mask_color])
            ax.imshow(
                mask_img,
                extent=ext,
                origin="upper",
                interpolation="nearest",
                cmap=overlay_cmap,
                alpha=args.mask_alpha,
            )

        if args.show_bounds:
            b = ds.bounds
            xs = [b.left, b.right, b.right, b.left, b.left]
            ys = [b.bottom, b.bottom, b.top, b.top, b.bottom]
            ax.plot(xs, ys, linewidth=1.5)

        ax.set_xlabel("X (lon / projected)")
        ax.set_ylabel("Y (lat / projected)")
        ax.grid(True, linewidth=0.3)

        plt.tight_layout()

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=200)
            print(f"Wrote: {out_path.resolve()}")
            plt.close(fig)
        else:
            plt.show()

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()