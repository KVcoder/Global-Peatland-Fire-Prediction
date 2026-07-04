import xarray as xr

ds = xr.open_zarr("full_data/viirs.zarr", consolidated=False)  # try consolidated=False if this fails

print(ds)  # overview

var = list(ds.data_vars)[0]   # pick a variable (or set var = "your_var_name")
da = ds[var]

print("\nVariable:", var)
print("dims :", da.dims)
print("shape:", da.shape)
print("chunks:", da.chunks)   # chunk sizes per dim (Dask-style)
