#!/usr/bin/env python3
"""
Count valid pixels ("tiles") in a GeoTIFF.

A pixel is considered VALID if:
- it is not nodata (from the file, or overridden by --nodata), AND
- it is not NaN/inf (for float rasters), AND
- (optional) it is not zero if --exclude-zero is set.

For multi-band rasters, you can define pixel validity as:
- --pixel-valid-mode any : pixel is valid if ANY band is valid (default)
- --pixel-valid-mode all : pixel is valid only if ALL bands are valid

Reads in blocks/windows to avoid loading the whole GeoTIFF into memory.
"""

import argparse
import numpy as np
import rasterio


def _valid_mask(arr: np.ndarray, nodata, exclude_zero: bool) -> np.ndarray:
    """
    arr: ndarray with shape (h,w) for one band
    returns: bool mask (h,w) True where valid
    """
    valid = np.ones(arr.shape, dtype=bool)

    # nodata check (only if nodata provided / known)
    if nodata is not None:
        valid &= (arr != nodata)

    # NaN/Inf check for float rasters
    if np.issubdtype(arr.dtype, np.floating):
        valid &= np.isfinite(arr)

    if exclude_zero:
        valid &= (arr != 0)

    return valid


def count_valid_pixels(
    tif_path: str,
    nodata_override=None,
    pixel_valid_mode: str = "any",
    exclude_zero: bool = False,
    band: int | None = None,
) -> tuple[int, int]:
    """
    Returns (valid_pixels, total_pixels) where total_pixels is height*width.
    For multi-band, total_pixels is still height*width (pixel locations).
    """
    if pixel_valid_mode not in ("any", "all"):
        raise ValueError("pixel_valid_mode must be 'any' or 'all'")

    with rasterio.open(tif_path) as ds:
        height, width = ds.height, ds.width
        total_pixels = int(height * width)

        # choose which bands to read
        if band is not None:
            if band < 1 or band > ds.count:
                raise ValueError(f"--band must be between 1 and {ds.count}")
            bands_to_read = [band]
        else:
            bands_to_read = list(range(1, ds.count + 1))

        nodata = ds.nodata if nodata_override is None else nodata_override

        valid_count = 0

        # Iterate in dataset-native blocks for speed/memory
        # Use band 1 for window layout (works for all bands)
        for _, window in ds.block_windows(1):
            data = ds.read(bands_to_read, window=window)  # shape: (B, h, w)

            # per-band valid masks
            band_valid = []
            for b in range(data.shape[0]):
                band_valid.append(_valid_mask(data[b], nodata, exclude_zero))

            if len(band_valid) == 1:
                pix_valid = band_valid[0]
            else:
                stack = np.stack(band_valid, axis=0)  # (B,h,w)
                if pixel_valid_mode == "any":
                    pix_valid = np.any(stack, axis=0)
                else:  # all
                    pix_valid = np.all(stack, axis=0)

            valid_count += int(pix_valid.sum())

        return valid_count, total_pixels


def main():
    ap = argparse.ArgumentParser(description="Count valid pixels (pixels treated as tiles) in a GeoTIFF.")
    ap.add_argument("--tif", required=True, help="Path to input GeoTIFF")
    ap.add_argument("--nodata", type=float, default=None, help="Override nodata value (default: use GeoTIFF nodata)")
    ap.add_argument("--pixel-valid-mode", choices=["any", "all"], default="any",
                    help="Multi-band rule: any band valid (default) vs all bands valid")
    ap.add_argument("--exclude-zero", action="store_true", help="Also treat zeros as invalid")
    ap.add_argument("--band", type=int, default=None,
                    help="Only use this 1-based band to determine validity (default: use all bands)")

    args = ap.parse_args()

    valid, total = count_valid_pixels(
        tif_path=args.tif,
        nodata_override=args.nodata,
        pixel_valid_mode=args.pixel_valid_mode,
        exclude_zero=args.exclude_zero,
        band=args.band,
    )

    invalid = total - valid
    frac = (valid / total) if total > 0 else 0.0

    print("==== Valid Pixel Count ====")
    print(f"TIF:           {args.tif}")
    print(f"Valid pixels:  {valid}")
    print(f"Total pixels:  {total}")
    print(f"Invalid:       {invalid}")
    print(f"Valid frac:    {frac:.6f}")
    print(f"Nodata used:   {args.nodata}")
    print(f"Mode:          pixel-valid-mode={args.pixel_valid_mode}, exclude-zero={args.exclude_zero}, band={args.band}")


if __name__ == "__main__":
    main()
