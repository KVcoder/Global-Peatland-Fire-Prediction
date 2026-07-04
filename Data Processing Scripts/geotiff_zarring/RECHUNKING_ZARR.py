#!/usr/bin/env python3
"""
RECHUNKING_ZARR.py

Rechunk an existing Zarr dataset to better match training access patterns.

Assumes source has:
  - MAIN VAR (configurable via --var-name), e.g.:
      * 'field'        (VIIRS)
      * 'era5land'     (ERA5-Land)
      * 'smap_wtd'     (SMAP)
    with dims (time, band, y, x) or similar.
  - optional 'peat_mask' : (y, x)

Writes a new Zarr with user-specified chunks (chunk_t, chunk_y, chunk_x)
and compressor (lz4 / blosc-zstd / zstd).
"""

from __future__ import annotations
import os
import sys
import shutil
import argparse

import xarray as xr
import zarr
import fsspec
from numcodecs import Blosc, Zstd
from tqdm.dask import TqdmCallback

from dask.distributed import Client, LocalCluster


# =========================
# DASK PARALLELISM HELPERS
# =========================
def start_dask_cluster(
    scheduler: str,
    workers: int,
    threads_per_worker: int,
    worker_mem: str,
    dash: bool,
):
    """
    Simple helper to optionally start a distributed Dask cluster.

    If scheduler == 'distributed':
        - Start LocalCluster + Client
    Else:
        - Return a no-op context, and rely on xarray/dask default scheduler.
    """
    if scheduler != "distributed":
        class _Noop:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False

        return _Noop()

    class _DistCtx:
        def __enter__(self):
            dashboard = ":0" if dash else None
            self.cluster = LocalCluster(
                n_workers=int(workers),
                threads_per_worker=int(threads_per_worker),
                processes=True,
                memory_limit=None if worker_mem == "auto" else worker_mem,
                dashboard_address=dashboard,
            )
            self.client = Client(self.cluster)
            print(
                f"[dask] LocalCluster up: workers={workers}, "
                f"threads/worker={threads_per_worker}, mem/worker={worker_mem}"
            )
            try:
                link = getattr(self.client, "dashboard_link", None)
                if link:
                    print(f"[dask] Dashboard: {link}")
            except Exception:
                pass
            return self.client

        def __exit__(self, *exc):
            try:
                self.client.close()
            finally:
                self.cluster.close()
            print("[dask] Cluster closed.")
            return False

    return _DistCtx()


# =========================
# COMPRESSION HELPER
# =========================
def make_compressor(kind: str, clevel: int):
    """
    kind:
      - 'zstd'        -> numcodecs.Zstd
      - 'blosc-zstd'  -> Blosc(zstd) (threads controlled via BLOSC_NTHREADS env)
      - 'lz4'         -> Blosc(lz4)  (threads controlled via BLOSC_NTHREADS env)

    NOTE: We do NOT pass 'nthreads=' here because some numcodecs versions
    do not accept it. Use the BLOSC_NTHREADS environment variable instead.
    """
    kind = kind.lower()
    if kind == "zstd":
        return Zstd(level=clevel)
    elif kind in ("blosc-zstd", "lz4"):
        cname = "zstd" if kind == "blosc-zstd" else "lz4"
        return Blosc(
            cname=cname,
            clevel=clevel,
            shuffle=Blosc.SHUFFLE,
        )
    else:
        raise ValueError(f"Unsupported compressor: {kind}")


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser(description="Rechunk existing Zarr store for faster training")
    ap.add_argument("--src", required=True, help="Source Zarr path (.zarr dir or .zip)")
    ap.add_argument("--dst", required=True, help="Destination Zarr path (.zarr dir or .zip)")

    # NEW: main data variable name
    ap.add_argument(
        "--var-name",
        type=str,
        default="field",
        help="Name of main data variable to rechunk (e.g. 'field', 'era5land', 'smap_wtd').",
    )

    ap.add_argument("--chunk-t", type=int, default=32, help="Time chunk size (days)")
    ap.add_argument("--chunk-y", type=int, default=256)
    ap.add_argument("--chunk-x", type=int, default=256)

    ap.add_argument("--compressor", choices=["zstd", "blosc-zstd", "lz4"], default="lz4")
    ap.add_argument("--clevel", type=int, default=1)

    ap.add_argument(
        "--blosc-threads",
        type=int,
        default=max(1, (os.cpu_count() or 8) // 2),
        help="Threads for Blosc compression/decompression (via BLOSC_NTHREADS env).",
    )

    ap.add_argument("--overwrite", action="store_true", help="Overwrite dst if it exists")
    ap.add_argument("--zarr-format", type=int, default=2, choices=[2, 3])
    ap.add_argument(
        "--zip-store",
        action="store_true",
        help="Write destination as zip store (NOT recommended for training)",
    )
    ap.add_argument(
        "--consolidate",
        action="store_true",
        help="Consolidate metadata (v2, dir store only)",
    )

    # Dask options
    ap.add_argument("--scheduler", choices=["distributed", "threads", "processes"], default="distributed")
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) // 4),
        help="Number of Dask workers (for 'distributed' scheduler)",
    )
    ap.add_argument("--threads-per-worker", type=int, default=2)
    ap.add_argument("--worker-mem", type=str, default="auto")
    ap.add_argument("--dask-dashboard", action="store_true")

    args = ap.parse_args()

    # Blosc threading: uses BLOSC_NTHREADS instead of nthreads= argument
    os.environ.setdefault("BLOSC_NTHREADS", str(max(1, int(args.blosc_threads))))
    print(f"[blosc] BLOSC_NTHREADS={os.environ['BLOSC_NTHREADS']}")

    # Handle dst path
    if os.path.exists(args.dst):
        if args.overwrite:
            print(f"[info] Removing existing dst: {args.dst}")
            if os.path.isdir(args.dst):
                shutil.rmtree(args.dst)
            else:
                os.remove(args.dst)
        else:
            print(
                f"[error] Destination {args.dst} exists. Use --overwrite to replace.",
                file=sys.stderr,
            )
            sys.exit(1)

    comp = make_compressor(args.compressor, args.clevel)

    # Dask context
    with start_dask_cluster(
        scheduler=args.scheduler,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        worker_mem=args.worker_mem,
        dash=args.dask_dashboard,
    ):
        # Try consolidated metadata first
        try:
            ds = xr.open_zarr(args.src, consolidated=True)
            print("[info] Opened src with consolidated metadata.")
        except Exception:
            ds = xr.open_zarr(args.src, consolidated=False)
            print("[info] Opened src without consolidated metadata.")

        var_name = args.var_name
        if var_name not in ds:
            raise KeyError(
                f"Expected variable '{var_name}' in source dataset. "
                f"Available variables: {list(ds.data_vars.keys())}"
            )

        data_var = ds[var_name]
        dims = data_var.dims
        print(f"[info] Source '{var_name}' dims: {dims}")
        print(f"[info] Source '{var_name}' shape: {dict(data_var.sizes)}")

        # Expect (time, band, y, x); if not, we still proceed but warn
        if len(dims) != 4:
            raise ValueError(
                f"Expected '{var_name}' to have 4 dims (time, band, y, x-ish), "
                f"got dims={dims}"
            )

        time_dim = dims[0]
        band_dim = dims[1]
        y_dim = dims[2]
        x_dim = dims[3]

        if dims != ("time", "band", "y", "x"):
            print(
                f"[warn] Unexpected dim order for '{var_name}': {dims}. "
                "Proceeding assuming order is (time, band, y, x).",
                file=sys.stderr,
            )

        # We always keep band dimension unchunked across entire C
        C = data_var.sizes[band_dim]

        data_re = data_var.chunk(
            {
                time_dim: max(1, args.chunk_t),
                band_dim: C,
                y_dim: max(1, args.chunk_y),
                x_dim: max(1, args.chunk_x),
            }
        )

        new_ds = data_re.to_dataset(name=var_name)

        # Rechunk peat_mask if present and if it is (y, x)
        if "peat_mask" in ds:
            pm = ds["peat_mask"]
            if set(pm.dims) == {"y", "x"}:
                pm_re = pm.chunk(
                    {
                        "y": max(1, args.chunk_y),
                        "x": max(1, args.chunk_x),
                    }
                )
                new_ds["peat_mask"] = pm_re
            else:
                print(
                    f"[warn] peat_mask dims={pm.dims} not (y,x); copying without rechunk.",
                    file=sys.stderr,
                )
                new_ds["peat_mask"] = pm

        # Build encoding dict
        if args.zarr_format == 3:
            data_enc = {
                "compressors": [comp],
                "chunks": (args.chunk_t, C, args.chunk_y, args.chunk_x),
                "_FillValue": None,
            }
            mask_enc = {
                "compressors": [comp],
                "chunks": (args.chunk_y, args.chunk_x),
                "dtype": "uint8",
            }
        else:
            data_enc = {
                "compressor": comp,
                "chunks": (args.chunk_t, C, args.chunk_y, args.chunk_x),
                "_FillValue": None,
            }
            mask_enc = {
                "compressor": comp,
                "chunks": (args.chunk_y, args.chunk_x),
                "dtype": "uint8",
            }

        encoding = {var_name: data_enc}
        if "peat_mask" in new_ds:
            encoding["peat_mask"] = mask_enc

        # Choose store target
        if args.zip_store:
            store = fsspec.get_mapper(f"zip://{args.dst}")
            target = store
            desc = "Writing rechunked Zarr (zip)"
        else:
            target = args.dst
            desc = "Writing rechunked Zarr"

        print(
            f"[info] Writing dst={args.dst} with chunks="
            f"({args.chunk_t}, C={C}, {args.chunk_y}, {args.chunk_x}), "
            f"compressor={args.compressor}, clevel={args.clevel}, "
            f"zarr_format={args.zarr_format}, var_name={var_name}"
        )

        # Write out the new Zarr
        with TqdmCallback(desc=desc, unit="task"):
            new_ds.to_zarr(
                target,
                mode="w",
                consolidated=False,
                encoding=encoding,
                zarr_format=args.zarr_format,
                compute=True,
            )

        # Consolidate metadata if requested (only for v2 dir stores)
        if args.consolidate and (not args.zip_store) and args.zarr_format == 2:
            print("[info] Consolidating metadata...")
            zarr.consolidate_metadata(args.dst)

        print("[done] Rechunked Zarr written successfully.")


if __name__ == "__main__":
    main()
