#!/usr/bin/env python3
"""
Sanity checks for the PeatDataset input pipeline — FAST + VERBOSE.

What this validates:
  1) Required parquet/json artifacts exist in --outdir and basic schema assumptions.
  2) Dataset construction for train/val with provided split dates.
  3) __getitem__ returns properly shaped tensors and valid dtypes.
  4) DataLoader batching & collate, including masks and star-graph node count (center + K).
  5) Quick integrity probes (no NaNs/Infs, masks are {0,1}, targets in {0,1} when mask=1).
  6) Rudimentary throughput timing for a few warmup + measured batches.

Exit codes:
  0 = all checks passed
  2 = soft warnings only (e.g., slow throughput)
  3 = hard assertion failed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

# Optional: nice progress bars
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---------------------------
# Project imports
# ---------------------------
try:
    from data.dataset import DatasetPaths, PeatDataset
except Exception as e:
    print("[FATAL] Could not import data.dataset.PeatDataset:", e)
    sys.exit(3)


# ---------------------------
# Utils
# ---------------------------

def eprint(*a, **k):
    print(*a, **k, file=sys.stderr)


def ok(msg: str):
    print(f"✅ {msg}")


def warn(msg: str):
    print(f"⚠️  {msg}")


def fail(msg: str):
    print(f"❌ {msg}")


def assert_true(cond: bool, msg: str):
    if not cond:
        fail(msg)
        sys.exit(3)


def path_exists(p: str) -> bool:
    return os.path.exists(p) and os.path.isfile(p)


def try_load_json(p: str) -> Dict:
    with open(p, "r") as f:
        return json.load(f)


def human(x: float) -> str:
    if x >= 1e6: return f"{x/1e6:.2f} M/s"
    if x >= 1e3: return f"{x/1e3:.2f} k/s"
    return f"{x:.2f} /s"


# ---------------------------
# Shape & value checks
# ---------------------------

def check_item_shapes(item: Dict, T_hist: int, K: int, C_dyn: int, pos_dim: int, H: int):
    req_keys = ["x_dyn", "x_pos", "y", "y_mask"]
    for k in req_keys:
        assert_true(k in item, f"Missing key '{k}' in dataset item")

    x_dyn = item["x_dyn"]  # [N,T,C]
    x_pos = item["x_pos"]  # [N,pos_dim]
    y     = item["y"]      # [H]
    y_m   = item["y_mask"] # [H]

    assert_true(torch.is_tensor(x_dyn) and torch.is_tensor(x_pos), "x_dyn/x_pos must be tensors")
    assert_true(torch.is_tensor(y) and torch.is_tensor(y_m), "y/y_mask must be tensors")

    assert_true(x_dyn.ndim == 3, f"x_dyn rank should be 3 [N,T,C], got {x_dyn.shape}")
    assert_true(x_pos.ndim == 2, f"x_pos rank should be 2 [N,pos], got {x_pos.shape}")
    assert_true(y.ndim == 1,     f"y rank should be 1 [H], got {y.shape}")
    assert_true(y_m.ndim == 1,   f"y_mask rank should be 1 [H], got {y_m.shape}")

    N, T, C = x_dyn.shape
    Np, Pd  = x_pos.shape
    Hh      = y.shape[0]

    assert_true(T == T_hist, f"T mismatch: expected {T_hist}, got {T}")
    assert_true(C == C_dyn,  f"C_dyn mismatch: expected {C_dyn}, got {C}")
    assert_true(N == (K + 1), f"Node count must be center + K neighbors = {K+1}, got {N}")
    assert_true(Np == N,      f"x_pos N mismatch: expected {N}, got {Np}")
    assert_true(Pd == pos_dim, f"pos_dim mismatch: expected {pos_dim}, got {Pd}")
    assert_true(Hh == H,      f"Horizons H mismatch: expected {H}, got {Hh}")

    # dtypes
    assert_true(x_dyn.dtype in (torch.float32, torch.float16, torch.bfloat16), f"x_dyn should be float, got {x_dyn.dtype}")
    assert_true(x_pos.dtype in (torch.float32, torch.float16, torch.bfloat16), f"x_pos should be float, got {x_pos.dtype}")
    assert_true(y.dtype in (torch.float32, torch.float16, torch.bfloat16), f"y should be float logits-target, got {y.dtype}")
    assert_true(y_m.dtype in (torch.float32, torch.float16, torch.bfloat16, torch.uint8, torch.bool), f"y_mask dtype odd: {y_m.dtype}")

    # value checks
    for name, tens in (("x_dyn", x_dyn), ("x_pos", x_pos), ("y", y)):
        assert_true(torch.isfinite(tens).all().item(), f"{name} contains NaN/Inf")

    # mask ∈ {0,1}
    if y_m.dtype.is_floating_point:
        assert_true(((y_m == 0) | (y_m == 1)).all().item(), "y_mask (float) must be 0/1")
    elif y_m.dtype == torch.bool:
        pass
    else:
        assert_true(((y_m == 0) | (y_m == 1)).all().item(), "y_mask must be 0/1")

    # If a horizon is valid (mask==1), y should be in [0,1] or {0,1}
    if y.numel() > 0 and (y_m.bool() if y_m.dtype == torch.bool else (y_m > 0)).any().item():
        y_valid = y[y_m.bool()] if y_m.dtype == torch.bool else y[y_m > 0]
        in_range = ((y_valid >= 0) & (y_valid <= 1)).all().item()
        is_binary = ((y_valid == 0) | (y_valid == 1)).all().item()
        assert_true(in_range or is_binary, "targets (where mask=1) should be within [0,1] or binary")


def check_batch_shapes(batch: Dict, B: int, T_hist: int, K: int, C_dyn: int, pos_dim: int, H: int):
    for k in ("x_dyn", "x_pos", "y", "y_mask"):
        assert_true(k in batch, f"Missing '{k}' in batch")

    x_dyn = batch["x_dyn"]  # [B,N,T,C]
    x_pos = batch["x_pos"]  # [B,N,pos]
    y     = batch["y"]      # [B,H]
    y_m   = batch["y_mask"] # [B,H]

    assert_true(x_dyn.ndim == 4, f"x_dyn rank 4 expected [B,N,T,C], got {x_dyn.shape}")
    assert_true(x_pos.ndim == 3, f"x_pos rank 3 expected [B,N,pos], got {x_pos.shape}")
    assert_true(y.ndim == 2 and y_m.ndim == 2, f"y/y_mask rank 2 expected [B,H], got {y.shape}/{y_m.shape}")

    Bn, N, T, C = x_dyn.shape
    Bp, Np, Pd  = x_pos.shape
    By, Hh      = y.shape
    Bm, Hm      = y_m.shape

    assert_true(Bn == B, f"Batch size mismatch for x_dyn: expected {B}, got {Bn}")
    assert_true(Bp == B and By == B and Bm == B, f"Batch size mismatch among tensors")

    assert_true(N == K + 1, f"N mismatch: expected {K+1}, got {N}")
    assert_true(Np == N,    f"x_pos N mismatch: expected {N}, got {Np}")
    assert_true(T == T_hist and C == C_dyn, "T/C mismatch")
    assert_true(Pd == pos_dim, f"pos_dim mismatch: expected {pos_dim}, got {Pd}")
    assert_true(Hh == H and Hm == H, f"H mismatch: expected {H}, got {Hh}/{Hm}")

    for name, tens in (("x_dyn", x_dyn), ("x_pos", x_pos), ("y", y)):
        assert_true(torch.isfinite(tens).all().item(), f"{name} contains NaN/Inf in batch")


# ---------------------------
# Main test routine
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PeatDataset input pipeline checks")
    # Roots
    p.add_argument("--era5", required=True, help="Path to ERA5LAND_UNSCALED_TRIMMED")
    p.add_argument("--wtd",  required=True, help="Path to RESAMPLED_SMAP_WTD_DAILY_GEOTIFF_TRIMMED")
    p.add_argument("--viirs", required=True, help="Path to RESAMPLED_NEW_VIIRS_DAILY_TRIMMED")
    p.add_argument("--outdir", required=True, help="Output directory with parquet/json artifacts")

    # Splits
    p.add_argument("--train-start", default="2016-01-01")
    p.add_argument("--train-end",   default="2020-12-31")
    p.add_argument("--val-start",   default="2021-01-01")
    p.add_argument("--val-end",     default="2022-12-31")

    # Dataset hyperparams
    p.add_argument("--T-hist", type=int, default=30)
    p.add_argument("--horizons", type=int, nargs="+", default=[1,3,7,14])
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--cache-items", type=int, default=64)
    p.add_argument("--rebuild-artifacts", action="store_true")
    p.add_argument("--persist-valid-t0", action="store_true")

    # Loader / system knobs
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-batches", type=int, default=3, help="Measured batches per split")
    p.add_argument("--warmup-batches", type=int, default=1, help="Warmup (not timed)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--gdal-cache", type=int, default=512, help="MB; exported to GDAL_CACHEMAX")

    # Progress control
    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars")

    return p.parse_args()


def main():
    args = parse_args()

    # Threading & GDAL cache
    try:
        torch.set_num_threads(max(1, int(args.torch_threads)))
    except Exception:
        pass
    os.environ["GDAL_CACHEMAX"] = str(int(args.gdal_cache))

    # Quick artifact presence check
    req_files = [
        "cells.parquet",
        "neighbors.parquet",
        "calendar.parquet",
        "train_stats.json",
    ]
    valid_t0_present = any(
        fn.startswith("valid_t0_") and fn.endswith(".parquet")
        for fn in os.listdir(args.outdir)
    ) if os.path.isdir(args.outdir) else False

    assert_true(os.path.isdir(args.outdir), print(f"--outdir does not exist: {args.outdir}"))
    for f in (tqdm(req_files, desc="Checking artifacts", leave=False) if (tqdm and not args.no_tqdm) else req_files):
        pth = os.path.join(args.outdir, f)
        assert_true(path_exists(pth), f"Missing required artifact: {pth}")
    assert_true(
        valid_t0_present or args.persist_valid_t0,
        "valid_t0_*.parquet not found. Rebuild with persist or pass --persist-valid-t0 here to let the dataset write it."
    )
    ok("Artifacts present.")

    # grid_meta/train_stats (grid_meta.n_cells is now a WARNING, not fatal)
    try:
        gm = try_load_json(os.path.join(args.outdir, "grid_meta.json"))
        if "n_cells" in gm and gm["n_cells"] > 0:
            ok(f"grid_meta.json: n_cells={gm['n_cells']}")
        else:
            warn("grid_meta.json missing n_cells (or zero). Continuing without it.")
    except Exception as e:
        warn(f"Non-fatal: could not parse grid_meta.json: {e}")

    try:
        _ = try_load_json(os.path.join(args.outdir, "train_stats.json"))
        ok("train_stats.json loaded.")
    except Exception as e:
        warn(f"Non-fatal: could not parse train_stats.json: {e}")

    # Build datasets
    paths = DatasetPaths(args.era5, args.wtd, args.viirs, args.outdir)
    splits = dict(
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
    )

    C_DYN   = 13
    POS_DIM = 36
    H       = len(args.horizons)

    def make(split: str) -> PeatDataset:
        return PeatDataset(
            paths,
            T_hist=args.T_hist,
            horizons=args.horizons,
            K=args.K,
            split=split,
            split_dates=splits,
            cache_items=args.cache_items,
            rebuild_artifacts=args.rebuild_artifacts,
            persist_valid_t0=args.persist_valid_t0,
        )

    print("== Constructing datasets")
    splits_to_build = [("train",), ("val",)]
    iterator = tqdm(splits_to_build, desc="Datasets", leave=False) if (tqdm and not args.no_tqdm) else splits_to_build

    for i, _ in enumerate(iterator):
        # using index to assign after the loop to keep names concise
        pass
    # build explicitly to preserve variable names
    ds_train = make("train")
    if tqdm and not args.no_tqdm:
        tqdm(total=2, desc="Datasets", leave=False).update(1)
    ds_val   = make("val")
    if tqdm and not args.no_tqdm:
        tqdm(total=2, desc="Datasets", leave=False).update(2)

    assert_true(len(ds_train) > 0, "train dataset has zero samples")
    assert_true(len(ds_val) > 0, "val dataset has zero samples")
    ok("Datasets constructed.")

    # Single-item shape probe (random index)
    import random
    probe_pairs = [("train", ds_train), ("val", ds_val)]
    probe_iter = tqdm(probe_pairs, desc="Single-item probes", leave=False) if (tqdm and not args.no_tqdm) else probe_pairs
    for name, ds in probe_iter:
        idx = random.randrange(0, len(ds))
        it = ds[idx]
        check_item_shapes(it, args.T_hist, args.K, C_DYN, POS_DIM, H)
        ok(f"{name} __getitem__ shape/value checks passed (idx={idx}).")

    # DataLoaders
    def make_loader(ds):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        )

    train_loader = make_loader(ds_train)
    val_loader   = make_loader(ds_val)
    ok("DataLoaders constructed.")

    # Warmup + measured passes with tqdm
    def run_loader(loader, name: str):
        B     = args.batch_size
        nb    = args.num_batches
        warm  = args.warmup_batches

        # Warmup
        it = iter(loader)
        if warm > 0:
            warm_iter = tqdm(range(warm), desc=f"{name}: warmup", leave=False) if (tqdm and not args.no_tqdm) else range(warm)
            for _ in warm_iter:
                try:
                    batch = next(it)
                    check_batch_shapes(batch, B=B, T_hist=args.T_hist, K=args.K, C_dyn=C_DYN, pos_dim=POS_DIM, H=H)
                except StopIteration:
                    break

        # Measured
        it = iter(loader)
        start = time.perf_counter()
        total_samples = 0
        iterator = range(nb)
        pbar = None
        if tqdm and not args.no_tqdm:
            pbar = tqdm(iterator, desc=f"{name}: measuring {nb} batches", leave=False)
        for _ in (pbar if pbar is not None else iterator):
            try:
                batch = next(it)
            except StopIteration:
                break
            check_batch_shapes(batch, B=B, T_hist=args.T_hist, K=args.K, C_dyn=C_DYN, pos_dim=POS_DIM, H=H)
            bs = batch["y"].shape[0]
            total_samples += bs
            if pbar is not None:
                pbar.set_postfix_str(f"samples={total_samples}")
        if pbar is not None:
            pbar.close()

        elapsed = time.perf_counter() - start
        sps = (total_samples / elapsed) if elapsed > 0 else float("inf")
        print(f"{name}: {total_samples} samples in {elapsed:.2f}s  →  {human(sps)}")
        return sps

    print("== Iterating train/val")
    sps_train = run_loader(train_loader, "train")
    sps_val   = run_loader(val_loader,   "val")

    status = 0
    if sps_train < 10 or sps_val < 10:
        warn("Throughput seems low (<10 samples/sec). Consider increasing --workers, --gdal-cache, enabling --pin-memory, or verifying disk speed.")
        status = max(status, 2)

    ok("All checks passed.")
    sys.exit(status)


if __name__ == "__main__":
    main()
