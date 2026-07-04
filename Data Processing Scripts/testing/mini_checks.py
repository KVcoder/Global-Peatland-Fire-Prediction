# Run in your project root with your paths
import pandas as pd, rasterio, os, numpy as np
outdir = "runs/pipeline_only"

cal = pd.read_parquet(os.path.join(outdir,"calendar.parquet"))
print(cal.head(500)); print(cal[['has_era5','has_wtd','has_viirs']].mean())  # fraction of days present

cells = pd.read_parquet(os.path.join(outdir,"cells.parquet"))
print("Peat cells M =", len(cells))

# Pick one ERA5 file and check bands
era5_dir = "era5land"
fn = next(p for p in os.listdir(era5_dir) if p.endswith(".tif"))
with rasterio.open(os.path.join(era5_dir, fn)) as ds:
    print("ERA5 bands =", ds.count)

# Check VIIRS nodata handling on one file
viirs_dir = "viirs"
vf = next(p for p in os.listdir(viirs_dir) if p.endswith(".tif"))
with rasterio.open(os.path.join(viirs_dir, vf)) as ds:
    arr = ds.read(1).astype(np.float32)
    nod = ds.nodata
    valid = (np.isfinite(arr) & (arr != nod)) if nod is not None else np.isfinite(arr)
    print("VIIRS valid px:", valid.sum(), " nonzero (fires):", ((arr!=0)&valid).sum())
