#!/usr/bin/env python3
"""
compare_tif_vs_zarr.py

Directly compare DataLoader speed for:
- GeoTIFF-backed PeatDataset
- Zarr-backed Era5ZarrDataset

Both are benchmarked with the same:
- t_hist (history length)
- batch_size
- num_workers
- (optionally) prefetch_factor

Usage example:

  python compare_tif_vs_zarr.py \
    --era5 data_geotiffs/era5land \
    --wtd  data_geotiffs/smap_wtd \
    --viirs data_geotiffs/viirs \
    --outdir data_output \
    --zarr era5land.zarr \
    --t-hist 30 \
    --patch 256 \
    --batch-size 16 \
    --workers 8 \
    --prefetch 2 \
    --batches 50
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Your project imports
from data.dataset import DatasetPaths, PeatDataset
from era5land_zarr_pytorch_dataset import Era5ZarrDataset


# -------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------

def collate(batch):
    """
    Same collate logic as in your training script:
    - Ensures all keys are present
    - Stacks each field into a batch tensor
    """
    if not batch:
        raise RuntimeError("Received empty batch in collate; check dataset filtering and batch_size.")
    keys = batch[0].keys()
    for k in keys:
        if any(k not in b for b in batch):
            raise KeyError(f"Missing key '{k}' in batch elements")
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def benchmark_loader(loader, label: str, n_warmup: int = 3, n_batches: int = 50):
    """
    Benchmark a PyTorch DataLoader.

    - Runs a few warmup batches
    - Then times n_batches and prints:
        • total samples
        • samples/sec
        • ms/batch
    """
    print(f"\n==============================")
    print(f"  Benchmark: {label}")
    print(f"==============================")

    it = iter(loader)

    # ------------------------
    # Warmup (do not time)
    # ------------------------
    warmup_count = 0
    for i in range(n_warmup):
        try:
            batch = next(it)
        except StopIteration:
            break
        _ = batch["x"] if isinstance(batch, dict) else batch
        warmup_count += 1
    print(f"[warmup] Ran {warmup_count} batches")

    # ------------------------
    # Timed loop
    # ------------------------
    total_samples = 0
    total_batches = 0
    start = time.time()

    for batch_idx, batch in enumerate(loader):
        if total_batches >= n_batches:
            break
        x = batch["x"] if isinstance(batch, dict) else batch
        total_samples += x.shape[0]
        total_batches += 1

    elapsed = time.time() - start

    if total_batches == 0 or elapsed <= 0:
        print("[result] Not enough batches to benchmark.")
        return

    samples_per_sec = total_samples / elapsed
    ms_per_batch = (elapsed / total_batches) * 1000.0

    print(f"[result] Batches:       {total_batches}")
    print(f"[result] Total samples: {total_samples}")
    print(f"[result] Wall time:     {elapsed:.2f} s")
    print(f"[result] Samples/sec:   {samples_per_sec:.1f}")
    print(f"[result] ms/batch:      {ms_per_batch:.1f}")


# -------------------------------------------------------------
# GeoTIFF-backed loader (PeatDataset)
# -------------------------------------------------------------

def make_tif_loader(
    era5_dir: str,
    wtd_dir: str,
    viirs_dir: str,
    outdir: str,
    t_hist: int = 30,
    batch_size: int = 16,
    workers: int = 8,
    prefetch: int = 2,
):
    """
    Build a train DataLoader for your GeoTIFF-backed PeatDataset
    using reasonable defaults that match your training script.
    """
    print("\n[make_tif_loader] Initializing PeatDataset (GeoTIFF)...")

    paths = DatasetPaths(era5_dir, wtd_dir, viirs_dir, outdir)
    splits = dict(
        train_start="2016-01-01",
        train_end="2020-12-31",
        val_start="2021-01-01",
        val_end="2022-12-31",
        test_start="2023-01-01",
        test_end="2024-12-31",
    )

    # Adjust these to match your normal training config if needed
    horizons = [1, 3, 7, 14]
    K = 8
    cache_items = 64

    ds_train = PeatDataset(
        paths,
        T_hist=t_hist,
        horizons=horizons,
        K=K,
        split="train",
        split_dates=splits,
        cache_items=cache_items,
        rebuild_artifacts=False,
        persist_valid_t0=True,
    )

    print(f"[make_tif_loader] Train samples: {len(ds_train)}")

    pin = torch.cuda.is_available()
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        collate_fn=collate,
        persistent_workers=(workers > 0),
    )
    if workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch

    loader = DataLoader(ds_train, **loader_kwargs)
    print(f"[make_tif_loader] Batches per epoch: {len(loader)}")
    return loader


# -------------------------------------------------------------
# Zarr-backed loader (Era5ZarrDataset)
# -------------------------------------------------------------

def make_zarr_loader(
    zarr_path: str,
    array_path: Optional[str],
    t_hist: int = 30,
    patch: int = 256,
    batch_size: int = 16,
    workers: int = 8,
):
    """
    Build a train DataLoader for your Zarr-backed Era5ZarrDataset.

    Important:
    - Make sure `patch` matches the spatial patch size you conceptually
      use in PeatDataset (e.g. 256x256).
    - `t_hist` should match your training history length (e.g. 30).
    """
    print("\n[make_zarr_loader] Initializing Era5ZarrDataset (Zarr)...")

    ds = Era5ZarrDataset(
        store_path=zarr_path,
        array_path=array_path,
        t_hist=t_hist,
        patch=patch,
        stride=None,               # non-overlapping patches
        time_stack="separate",
        mode="train",
        split=0.9,
        seed=42,
        normalize="per_channel",   # or None, but be consistent when comparing
        fixed_stats=None,
        time_index=None,
        transform=None,
        max_samples=None,
        skip_nan_patches=False,    # set True and tune nan_threshold if desired
        nan_check_time=0,
        nan_threshold=1.0,
    )

    print(f"[make_zarr_loader] Train samples: {len(ds)}")
    print(f"[make_zarr_loader] Underlying array: {ds.arr_path}")
    print(f"[make_zarr_loader] Shape (T, C, H, W) = ({ds.T}, {ds.C}, {ds.H}, {ds.W})")

    pin = torch.cuda.is_available()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        collate_fn=collate,            # keep behavior similar to PeatDataset
        persistent_workers=(workers > 0),
    )
    print(f"[make_zarr_loader] Batches per epoch: {len(loader)}")
    return loader


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Compare DataLoader speed: GeoTIFF (PeatDataset) vs Zarr (Era5ZarrDataset)"
    )
    # GeoTIFF roots
    ap.add_argument("--era5", required=True, help="ERA5 GeoTIFF directory")
    ap.add_argument("--wtd", required=True, help="WTD GeoTIFF directory")
    ap.add_argument("--viirs", required=True, help="VIIRS GeoTIFF directory")
    ap.add_argument("--outdir", required=True, help="Output/cache directory (same as training)")

    # Zarr store
    ap.add_argument("--zarr", required=True, help="Path to ERA5-Land Zarr store")
    ap.add_argument("--array-path", default=None, help="Array path inside Zarr (auto-detect if omitted)")

    # Shared hyperparams
    ap.add_argument("--t-hist", type=int, default=30, help="History length (days)")
    ap.add_argument("--patch", type=int, default=256, help="Patch size for Zarr dataset (H=W=patch)")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size")
    ap.add_argument("--workers", type=int, default=8, help="Number of DataLoader workers")
    ap.add_argument("--prefetch", type=int, default=2, help="Prefetch factor for GeoTIFF loader")
    ap.add_argument("--batches", type=int, default=50, help="Number of timed batches per benchmark")

    return ap.parse_args()


def main():
    args = parse_args()

    print("==========================================")
    print("  GeoTIFF vs Zarr DataLoader Benchmark")
    print("==========================================")
    print(f"t_hist      = {args.t_hist}")
    print(f"patch       = {args.patch} (Zarr only)")
    print(f"batch_size  = {args.batch_size}")
    print(f"workers     = {args.workers}")
    print(f"prefetch    = {args.prefetch} (GeoTIFF only)")
    print(f"batches     = {args.batches}")
    print("")

    # Build loaders
    tif_loader = make_tif_loader(
        era5_dir=args.era5,
        wtd_dir=args.wtd,
        viirs_dir=args.viirs,
        outdir=args.outdir,
        t_hist=args.t_hist,
        batch_size=args.batch_size,
        workers=args.workers,
        prefetch=args.prefetch,
    )

    zarr_loader = make_zarr_loader(
        zarr_path=args.zarr,
        array_path=args.array_path,
        t_hist=args.t_hist,
        patch=args.patch,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    # Run benchmarks
    benchmark_loader(
        tif_loader,
        label=f"GeoTIFF / PeatDataset (workers={args.workers})",
        n_warmup=3,
        n_batches=args.batches,
    )
    benchmark_loader(
        zarr_loader,
        label=f"Zarr / Era5ZarrDataset (workers={args.workers})",
        n_warmup=3,
        n_batches=args.batches,
    )


if __name__ == "__main__":
    main()
