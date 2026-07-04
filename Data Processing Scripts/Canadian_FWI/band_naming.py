from pathlib import Path
import rasterio

# folder containing your .tif files
folder = Path("ERA5_ALIGNED")

# descriptions for the 5 bands, in order
band_descriptions = [
    "d2m_C",
    "t2m_C",
    "tp_mm",
    "u10_ms",
    "v10_ms"
]

for tif in folder.glob("*.tif"):
    with rasterio.open(tif, "r+") as dst:
        if dst.count != len(band_descriptions):
            print(f"Skipping {tif.name}: expected 5 bands, found {dst.count}")
            continue

        for i, desc in enumerate(band_descriptions, start=1):
            dst.set_band_description(i, desc)

    print(f"Updated: {tif.name}")