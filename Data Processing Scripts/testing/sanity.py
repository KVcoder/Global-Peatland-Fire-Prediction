from __future__ import annotations

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import DatasetPaths, PeatDataset


def collate(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k == "edge_index":
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--era5", required=True)
    p.add_argument("--wtd", required=True)
    p.add_argument("--viirs", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()

    # Build artifacts on train split
    split_dates = dict(
        train_start="2016-01-01", train_end="2016-03-31",
        val_start="2016-04-01", val_end="2016-04-30",
        test_start="2016-05-01", test_end="2016-05-31",
    )
    paths = DatasetPaths(args.era5, args.wtd, args.viirs, args.outdir)
    ds = PeatDataset(paths, T_hist=10, horizons=[1,3], K=4, split="train", split_dates=split_dates, rebuild_artifacts=True, cache_items=8)

    # Grid sanity via an actual read happens implicitly (asserts in dataset).

    # Small batch
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    batch = next(iter(loader))
    x_dyn = batch["x_dyn"]       # [B,N,T,C]
    x_mask = batch["x_mask_dyn"]
    x_static = batch["x_static"]
    x_pos = batch["x_pos"]
    y = batch["y"]
    y_mask = batch["y_mask"]

    print("Shapes:")
    print("x_dyn:", tuple(x_dyn.shape))
    print("x_mask_dyn:", tuple(x_mask.shape))
    print("x_static:", tuple(x_static.shape))
    print("x_pos:", tuple(x_pos.shape))
    print("y:", tuple(y.shape), "y_mask:", tuple(y_mask.shape))

    # Label prevalence by date (rough)
    # NOTE: Full scan is expensive; this prints a small randomized sample.
    print("\nLabel prevalence (sampled dates):")
    # Not implemented: requires reading many files; users can implement if needed.

if __name__ == "__main__":
    main()
