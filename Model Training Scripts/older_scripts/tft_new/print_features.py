#!/usr/bin/env python3
"""
print_feature_names.py

Prints per-channel feature names for:
- ERA5-Land (era5land)
- SMAP (smap_wtd)
- VIIRS (field)

Tries to read names from common Zarr attributes like:
  'feature_names', 'variable_names', 'var_names', 'band_names',
  'channel_names', 'channels'.

If none are found, falls back to generic 'channel_0', 'channel_1', ...
"""

from joint_peat_dataset_builder import _open_zarr_array
import numpy as np


# TODO: replace these with your actual Zarr directory paths
ERA5_ZARR = "SOUTHEAST_ASIA_CROP/era5land_SE_Asia_t32_p32.zarr"
SMAP_ZARR = "SOUTHEAST_ASIA_CROP/smap_wtd_SE_Asia_t32_p32.zarr"
VIIRS_ZARR = "SOUTHEAST_ASIA_CROP/viirs_SE_Asia_t32_p32.zarr"


def get_channel_names(arr, prefix: str):
    """
    Try to extract human-readable channel names from Zarr array attrs.
    If none found, fall back to prefix:channel_<idx>.
    """
    attrs = getattr(arr, "attrs", {}) or {}

    candidate_keys = [
        "feature_names",
        "features",
        "variable_names",
        "var_names",
        "band_names",
        "channel_names",
        "channels",
    ]

    for key in candidate_keys:
        if key in attrs:
            names = attrs[key]
            # Often stored as list / tuple / np.ndarray / comma-separated string
            if isinstance(names, str):
                names = [n.strip() for n in names.split(",") if n.strip()]
            elif isinstance(names, np.ndarray):
                names = names.tolist()

            if isinstance(names, (list, tuple)):
                return [f"{prefix}:{name}" for name in names]

    # Fallback: generic names based on C dimension (T, C, H, W)
    if len(arr.shape) != 4:
        raise ValueError(f"Expected 4D array (T, C, H, W), got shape {arr.shape}")

    C = arr.shape[1]
    return [f"{prefix}:channel_{i}" for i in range(C)]


def main():
    era5_arr, _ = _open_zarr_array(ERA5_ZARR, "era5land")
    smap_arr, _ = _open_zarr_array(SMAP_ZARR, "smap_wtd")
    viirs_arr, _ = _open_zarr_array(VIIRS_ZARR, "field")

    era5_names = get_channel_names(era5_arr, "era5")
    smap_names = get_channel_names(smap_arr, "smap")
    viirs_names = get_channel_names(viirs_arr, "viirs")

    print("\nERA5-Land features (input):")
    for name in era5_names:
        print("  -", name)

    print("\nSMAP features (input):")
    for name in smap_names:
        print("  -", name)

    print("\nVIIRS features (target):")
    for name in viirs_names:
        print("  -", name)

    # This matches the order of channels in JointPeatDataset.x
    print("\nCombined input feature order in JointPeatDataset.x (ERA5 + SMAP):")
    for i, name in enumerate(era5_names + smap_names):
        print(f"  [{i:02d}] {name}")


if __name__ == "__main__":
    main()
