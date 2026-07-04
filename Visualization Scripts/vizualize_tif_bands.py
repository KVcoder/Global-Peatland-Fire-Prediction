#!/usr/bin/env python3
import os
import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as mcolors  # NEW


def main():
    ap = argparse.ArgumentParser(description="Export one PNG per band from a (multi-band) GeoTIFF.")
    ap.add_argument("--tif", required=True, help="Path to input GeoTIFF")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: <tif_basename>_bands next to file)")
    ap.add_argument("--nodata", type=float, default=None,
                    help="Override nodata (default: read from GeoTIFF). Common: -9999")
    ap.add_argument("--vmin", type=float, default=0.00, help="Color scale min (default 0.0)")
    ap.add_argument("--vmax", type=float, default=0.05, help="Color scale max (default 1.0)")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap (default viridis)")
    ap.add_argument("--dpi", type=int, default=200, help="PNG DPI (default 200)")
    ap.add_argument("--clip01", action="store_true",
                    help="Clip values to [0,1] before plotting (useful for probabilities)")
    ap.add_argument("--mask_outside01", action="store_true",
                    help="Mask values outside [0,1] as invalid (instead of clipping)")
    ap.add_argument("--no_axes", action="store_true", help="Hide axes for cleaner images")
    args = ap.parse_args()

    tif_path = args.tif
    if not os.path.isfile(tif_path):
        raise FileNotFoundError(tif_path)

    base = os.path.splitext(os.path.basename(tif_path))[0]
    out_dir = args.out_dir or os.path.join(os.path.dirname(tif_path), f"{base}_bands")
    os.makedirs(out_dir, exist_ok=True)

    # NEW: build a colormap where "bad" (masked) == color at value 0.0 for this vmin/vmax
    norm = mcolors.Normalize(vmin=args.vmin, vmax=args.vmax, clip=True)
    try:
        cmap = plt.colormaps[args.cmap].copy()  # mpl>=3.5
    except Exception:
        cmap = plt.cm.get_cmap(args.cmap).copy()  # fallback

    zero_color = cmap(norm(0.0))      # rgba for value 0.0 under your scale
    cmap.set_bad(zero_color)          # masked pixels become "0-color"
    # (optional, but nice): ensure values below vmin also use the lowest color
    cmap.set_under(cmap(0.0))

    with rasterio.open(tif_path) as src:
        nodata = src.nodata if args.nodata is None else args.nodata
        count = src.count

        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        for b in tqdm(range(1, count + 1), desc="Exporting bands"):
            arr = src.read(b).astype(np.float32)

            mask = ~np.isfinite(arr)
            if nodata is not None:
                mask |= (arr == nodata)

            if args.mask_outside01:
                mask |= (arr < 0.0) | (arr > 1.0)
            elif args.clip01:
                arr = np.clip(arr, 0.0, 1.0)

            arr_masked = np.ma.array(arr, mask=mask)

            # NEW: set figure/axes background to the same "0-color"
            fig, ax = plt.subplots()
            fig.patch.set_facecolor(zero_color)
            ax.set_facecolor(zero_color)

            im = ax.imshow(
                arr_masked, extent=extent, origin="upper",
                vmin=args.vmin, vmax=args.vmax, cmap=cmap
            )
            ax.set_title(f"{base} — band {b}")

            cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label("PoF")

            if args.no_axes:
                ax.axis("off")
            else:
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

            out_png = os.path.join(out_dir, f"{base}_band{b:02d}.png")
            fig.savefig(
                out_png, dpi=args.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none"
            )
            plt.close(fig)

    print(f"Saved {count} PNGs to: {out_dir}")


if __name__ == "__main__":
    main()
