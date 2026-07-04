#!/usr/bin/env python3
"""
Spatiotemporal Peat Ignition — XGBoost per-tile model + optional post-hoc calibration

Replaces the per-tile MLP with per-horizon XGBoost binary classifiers.
Each pixel (tile) is treated independently:
  features: time×channels (flattened) + any extra features the dataset already provides
  label: fire (0/1) for each horizon

Calibration (optional; per-horizon):
  - Platt: p' = sigmoid(a * logit(p) + b)
  - Isotonic: p' = iso(p) using histogram bins + PAV (memory-safe)

Also reports counts of POST-CALIBRATION probabilities above a percent threshold:
  --prob-max-pct 0.06 means 0.06% => prob threshold = 0.0006

NOTE ON MEMORY:
  Training uses *sampled* valid pixels from patches. Control with:
    --xgb-sample-frac, --xgb-max-train-tiles

Requires:
  - xgboost
  - torch, numpy, psutil
  - joint_peat_dataset_builder.JointPeatDataset
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, List

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import xgboost as xgb

from joint_peat_dataset_builder import JointPeatDataset


# ----------------------------------------------------------------------
# Small utils
# ----------------------------------------------------------------------

def exists(x):
    return x is not None


def human_int(n: int) -> str:
    n = float(n)
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            s = f"{n:.1f}{unit}"
            return s.rstrip("0").rstrip(".")
        n /= 1000.0
    return f"{n:.1f}T"


def get_ram_usage() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


def get_gpu_memory(device_id: int = 0) -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device_id) / (1024 ** 3)
    return 0.0


def print_diagnostic_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_diagnostic_item(label: str, value: Any, indent: int = 0):
    prefix = "  " * indent + "• "
    print(f"{prefix}{label:.<40} {value}")


class RAMMonitor:
    def __init__(self, device_id=0, update_interval=0.5):
        self.device_id = device_id
        self.update_interval = update_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.pbar = None
        self.current_status = "Initializing..."

    def start(self):
        self.running = True
        self.pbar = tqdm(total=0, position=0, bar_format="{desc}", leave=True)
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.pbar:
            self.pbar.close()

    def set_status(self, status: str):
        self.current_status = status

    def _monitor_loop(self):
        while self.running:
            ram = get_ram_usage()
            desc = f"💾 RAM: {ram:.2f}GB"
            if torch.cuda.is_available():
                desc += f" | 🎮 GPU: {get_gpu_memory(self.device_id):.2f}GB"
            desc += f" | {self.current_status}"
            if self.pbar:
                self.pbar.set_description_str(desc)
            time.sleep(self.update_interval)


class TimedDataLoader:
    def __init__(self, loader: DataLoader, desc: str = "batch", use_tqdm: bool = True, position: int = 1):
        self.loader = loader
        self.desc = desc
        self.use_tqdm = use_tqdm
        self.position = position
        self.batch_times: deque = deque(maxlen=50)
        self.pbar = None

    def __iter__(self):
        it = iter(self.loader)
        total = None
        try:
            total = len(self.loader)
        except TypeError:
            total = None

        if self.use_tqdm:
            self.pbar = tqdm(total=total, desc=self.desc, leave=True, position=self.position)

        while True:
            try:
                t0 = time.time()
                batch = next(it)
                dt = time.time() - t0
                self.batch_times.append(dt)
                if self.pbar:
                    avg = sum(self.batch_times) / len(self.batch_times)
                    self.pbar.set_postfix({"load_ms": f"{dt*1000:.0f}", "avg_ms": f"{avg*1000:.0f}"})
                    self.pbar.update(1)
                yield batch
            except StopIteration:
                break

        if self.pbar:
            self.pbar.close()

    def __len__(self):
        return len(self.loader)


def collate(batch):
    if not batch:
        raise RuntimeError("Received empty batch in collate; check dataset filtering and batch_size.")
    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def _choose_mp_context(requested: str) -> Optional[str]:
    if requested != "auto":
        return requested
    if sys.platform.startswith("linux"):
        return "forkserver"
    return "spawn"


def make_loader(ds, batch_size, shuffle, args, sampler=None):
    pin = torch.cuda.is_available()
    kw: Dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=pin,
        collate_fn=collate,
        persistent_workers=(args.workers > 0),
    )
    if sampler is not None:
        kw["sampler"] = sampler
        kw["shuffle"] = False
    else:
        kw["shuffle"] = shuffle

    if args.workers > 0:
        kw["prefetch_factor"] = args.prefetch
        ctx = _choose_mp_context(args.mp_context)
        if ctx:
            kw["multiprocessing_context"] = ctx

    return DataLoader(ds, **kw)


def wrap_loader(loader, desc: str, use_tqdm: bool, position: int = 1):
    return TimedDataLoader(loader, desc=desc, use_tqdm=use_tqdm, position=position)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_time_if_needed(x: torch.Tensor) -> torch.Tensor:
    """batch x -> (B,F,H,W)"""
    if x.dim() == 4:
        return x
    if x.dim() == 5:
        B, T, C, H, W = x.shape
        return x.view(B, T * C, H, W)
    raise ValueError(f"Expected x with 4 or 5 dims, got {tuple(x.shape)}")


# ----------------------------------------------------------------------
# Calibration (Platt / Isotonic)
# ----------------------------------------------------------------------

def _pav_weighted(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    y = y.astype(np.float64)
    w = w.astype(np.float64)

    means, weights, sizes = [], [], []
    for yi, wi in zip(y, w):
        means.append(float(yi))
        weights.append(float(wi))
        sizes.append(1)

        while len(means) >= 2 and means[-2] > means[-1]:
            w_new = weights[-2] + weights[-1]
            m_new = (means[-2] * weights[-2] + means[-1] * weights[-1]) / max(w_new, 1e-12)
            s_new = sizes[-2] + sizes[-1]

            means[-2] = float(m_new)
            weights[-2] = float(w_new)
            sizes[-2] = int(s_new)

            means.pop()
            weights.pop()
            sizes.pop()

    yhat = np.concatenate([np.full(sz, m, dtype=np.float64) for m, sz in zip(means, sizes)], axis=0)
    return yhat


@dataclass
class ProbCalibratorXGB:
    method: str
    horizons: List[int]
    a: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    iso_x: Optional[List[Optional[np.ndarray]]] = None
    iso_y: Optional[List[Optional[np.ndarray]]] = None

    @property
    def K(self) -> int:
        return len(self.horizons)

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """probs: (N,K) in [0,1]"""
        if self.method == "none" or self.method is None:
            return probs

        probs = np.clip(probs, 1e-8, 1.0 - 1e-8)

        if self.method == "platt":
            if self.a is None or self.b is None:
                return probs
            logit = np.log(probs / (1.0 - probs))
            z = logit * self.a.reshape(1, -1) + self.b.reshape(1, -1)
            out = 1.0 / (1.0 + np.exp(-z))
            return np.clip(out, 0.0, 1.0)

        if self.method == "isotonic":
            if self.iso_x is None or self.iso_y is None:
                return probs
            out = np.empty_like(probs)
            for k in range(probs.shape[1]):
                x = self.iso_x[k]
                y = self.iso_y[k]
                if x is None or y is None or x.size < 2:
                    out[:, k] = probs[:, k]
                else:
                    out[:, k] = np.interp(probs[:, k], x, y)
            return np.clip(out, 0.0, 1.0)

        return probs


@torch.no_grad()
def fit_calibrator_xgb(models: List[xgb.XGBClassifier], calib_loader, args, monitor: Optional[RAMMonitor] = None) -> Optional[ProbCalibratorXGB]:
    method = getattr(args, "calibration_method", "none")
    if method == "none" or calib_loader is None:
        return None

    K = len(args.horizons)
    cal = ProbCalibratorXGB(method=method, horizons=[int(h) for h in args.horizons], iso_x=[None]*K, iso_y=[None]*K)

    if method == "platt":
        # Fit a,b on minibatches of predicted probs (memory-safe)
        device = torch.device("cuda" if (torch.cuda.is_available() and args.platt_on_gpu) else "cpu")
        a = torch.ones(K, device=device, dtype=torch.float32, requires_grad=True)
        b = torch.zeros(K, device=device, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([a, b], lr=float(args.calib_platt_lr))

        steps = max(1, int(args.calib_platt_steps))
        it = iter(calib_loader)

        for s in range(steps):
            if monitor is not None:
                monitor.set_status(f"Platt calib step {s+1}/{steps}")

            try:
                batch = next(it)
            except StopIteration:
                it = iter(calib_loader)
                batch = next(it)

            x = _flatten_time_if_needed(batch["x"])  # (B,F,H,W)
            y = (batch["y"] > 0.5)  # (B,K,H,W)
            m = (batch["mask"] > 0.5)  # (B,K,H,W)

            B, F, H, W = x.shape
            # We'll create per-horizon minibatches of valid pixels.
            losses = []
            for k in range(K):
                valid = m[:, k]  # (B,H,W)
                if valid.sum().item() == 0:
                    continue

                # Sample a subset of valid pixels for stability/speed
                idx = torch.nonzero(valid, as_tuple=False)
                if idx.shape[0] == 0:
                    continue
                if args.platt_max_tiles > 0 and idx.shape[0] > args.platt_max_tiles:
                    perm = torch.randperm(idx.shape[0])[: args.platt_max_tiles]
                    idx = idx[perm]

                bs = idx[:, 0]
                ys = idx[:, 1]
                xs = idx[:, 2]

                feat = x[bs, :, ys, xs].contiguous().cpu().numpy().astype(np.float32)
                p = models[k].predict_proba(feat)[:, 1]
                p = np.clip(p, 1e-8, 1.0 - 1e-8)
                logit_p = np.log(p / (1.0 - p)).astype(np.float32)

                t = y[bs, k, ys, xs].float().to(device)
                z = torch.from_numpy(logit_p).to(device) * a[k] + b[k]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z, t)
                losses.append(loss)

            if not losses:
                continue

            loss_all = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss_all.backward()
            opt.step()

        cal.a = a.detach().float().cpu().numpy()
        cal.b = b.detach().float().cpu().numpy()
        return cal

    if method == "isotonic":
        nb = max(16, int(args.isotonic_bins))
        counts = np.zeros((K, nb), dtype=np.float64)
        true_sums = np.zeros((K, nb), dtype=np.float64)

        iterator = wrap_loader(calib_loader, desc="calib(iso)", use_tqdm=not args.no_tqdm, position=2)
        for batch in iterator:
            if monitor is not None:
                monitor.set_status("Isotonic calib")

            x = _flatten_time_if_needed(batch["x"])  # (B,F,H,W)
            y = (batch["y"] > 0.5)
            m = (batch["mask"] > 0.5)

            B, F, H, W = x.shape
            for k in range(K):
                valid = m[:, k]
                if valid.sum().item() == 0:
                    continue

                idx = torch.nonzero(valid, as_tuple=False)
                if idx.shape[0] == 0:
                    continue

                # Optional cap (prevents giant memory spikes on very big patches)
                if args.iso_max_tiles > 0 and idx.shape[0] > args.iso_max_tiles:
                    perm = torch.randperm(idx.shape[0])[: args.iso_max_tiles]
                    idx = idx[perm]

                bs = idx[:, 0]
                ys = idx[:, 1]
                xs = idx[:, 2]

                feat = x[bs, :, ys, xs].contiguous().cpu().numpy().astype(np.float32)
                p = models[k].predict_proba(feat)[:, 1]
                t = y[bs, k, ys, xs].contiguous().cpu().numpy().astype(np.float64)

                p = np.clip(p, 0.0, 1.0)
                bin_idx = np.floor(p * nb).astype(np.int64)
                bin_idx = np.clip(bin_idx, 0, nb - 1)

                counts[k] += np.bincount(bin_idx, minlength=nb).astype(np.float64)
                true_sums[k] += np.bincount(bin_idx, weights=t, minlength=nb).astype(np.float64)

        bin_centers = (np.arange(nb, dtype=np.float64) + 0.5) / nb

        iso_x: List[Optional[np.ndarray]] = [None] * K
        iso_y: List[Optional[np.ndarray]] = [None] * K
        for k in range(K):
            nz = counts[k] > 0
            if nz.sum() < 2:
                continue
            xk = bin_centers[nz]
            y_mean = true_sums[k][nz] / np.maximum(counts[k][nz], 1e-12)
            wk = counts[k][nz]
            y_iso = _pav_weighted(y_mean, wk)
            iso_x[k] = xk.astype(np.float64)
            iso_y[k] = y_iso.astype(np.float64)

        cal.iso_x = iso_x
        cal.iso_y = iso_y
        return cal

    return None


# ----------------------------------------------------------------------
# Metrics + evaluation
# ----------------------------------------------------------------------

def _compute_basic_metrics(tp, fp, fn, tn):
    support = tp + fp + fn + tn
    if support == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": support,
        }
    accuracy = (tp + tn) / support
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "support": int(support),
    }


def _bce_logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


@torch.no_grad()
def evaluate_xgb(
    models: List[xgb.XGBClassifier],
    loader,
    args,
    calibrator: Optional[ProbCalibratorXGB] = None,
    use_tqdm: bool = True,
    monitor: Optional[RAMMonitor] = None,
):
    K = len(args.horizons)

    # args.prob_max_pct is in percent units (0.06 means 0.06%)
    prob_thresh = float(args.prob_max_pct) / 100.0

    tp_total = fp_total = fn_total = tn_total = 0
    tp_h = [0 for _ in range(K)]
    fp_h = [0 for _ in range(K)]
    fn_h = [0 for _ in range(K)]
    tn_h = [0 for _ in range(K)]

    gt_count_h = np.zeros(K, dtype=np.float64)
    total_prob_h = np.zeros(K, dtype=np.float64)

    loss_sum = 0.0
    loss_n = 0

    iterator = wrap_loader(loader, desc="eval", use_tqdm=use_tqdm, position=3)

    for batch in iterator:
        if monitor is not None:
            monitor.set_status("Evaluating")

        x = _flatten_time_if_needed(batch["x"])  # (B,F,H,W)
        y = (batch["y"] > 0.5)
        m = (batch["mask"] > 0.5)

        B, F, H, W = x.shape

        # For each horizon, predict only on valid pixels
        for k in range(K):
            valid = m[:, k]
            if valid.sum().item() == 0:
                continue

            idx = torch.nonzero(valid, as_tuple=False)
            if idx.shape[0] == 0:
                continue

            # Optional cap for speed/memory (evaluation remains approximate if capped)
            if args.eval_max_tiles > 0 and idx.shape[0] > args.eval_max_tiles:
                perm = torch.randperm(idx.shape[0])[: args.eval_max_tiles]
                idx = idx[perm]

            bs = idx[:, 0]
            ys = idx[:, 1]
            xs = idx[:, 2]

            feat = x[bs, :, ys, xs].contiguous().cpu().numpy().astype(np.float32)
            t = y[bs, k, ys, xs].contiguous().cpu().numpy().astype(np.int64)

            p = models[k].predict_proba(feat)[:, 1].astype(np.float64)
            p = np.clip(p, 0.0, 1.0)

            # Apply post-hoc calibration (per horizon)
            if calibrator is not None:
                p2 = calibrator.apply(p.reshape(-1, 1))[:, 0]
            else:
                p2 = p

            # loss
            loss_sum += _bce_logloss(p2, t.astype(np.float64)) * float(t.size)
            loss_n += int(t.size)

            pred = (p2 >= float(args.metrics_threshold))
            tp = int(((pred == 1) & (t == 1)).sum())
            fp = int(((pred == 1) & (t == 0)).sum())
            fn = int(((pred == 0) & (t == 1)).sum())
            tn = int(((pred == 0) & (t == 0)).sum())

            tp_total += tp
            fp_total += fp
            fn_total += fn
            tn_total += tn

            tp_h[k] += tp
            fp_h[k] += fp
            fn_h[k] += fn
            tn_h[k] += tn

            total_prob_h[k] += float(p2.size)
            gt_count_h[k] += float((p2 > prob_thresh).sum())

    val_loss = float("nan") if loss_n <= 0 else float(loss_sum / max(1, loss_n))

    overall = _compute_basic_metrics(tp_total, fp_total, fn_total, tn_total)
    per_horizon_metrics: Dict[int, Dict[str, float]] = {}
    for idx, h in enumerate(args.horizons):
        per_horizon_metrics[int(h)] = _compute_basic_metrics(tp_h[idx], fp_h[idx], fn_h[idx], tn_h[idx])

    gt_total = float(gt_count_h.sum())
    total_total = float(total_prob_h.sum())
    gt_frac = float("nan") if total_total <= 0 else float(gt_total / total_total)

    gt_per_h = {}
    for idx, h in enumerate(args.horizons):
        tot = float(total_prob_h[idx])
        cnt = float(gt_count_h[idx])
        frac = float("nan") if tot <= 0 else float(cnt / tot)
        gt_per_h[int(h)] = {"count": cnt, "total": tot, "fraction": frac}

    metrics = {
        "accuracy": overall["accuracy"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "tp": overall["tp"],
        "fp": overall["fp"],
        "fn": overall["fn"],
        "tn": overall["tn"],
        "support": overall["support"],
        "per_horizon": per_horizon_metrics,
        "prob_gt": {
            "threshold_prob": prob_thresh,
            "threshold_pct": float(args.prob_max_pct),
            "count": gt_total,
            "total": total_total,
            "fraction": gt_frac,
            "per_horizon": gt_per_h,
        },
    }

    return val_loss, metrics


# ----------------------------------------------------------------------
# Dataset helpers
# ----------------------------------------------------------------------

def build_datasets(args):
    common = dict(
        era5_zarr=args.era5_zarr,
        smap_zarr=args.smap_zarr,
        extra_zarr=args.smap_l4_zarr,
        viirs_zarr=args.viirs_zarr,
        era5_array=args.era5_array,
        smap_array=args.smap_array,
        extra_array=args.smap_l4_array,
        viirs_array=args.viirs_array,
        t_hist=args.T_hist,
        horizons=args.horizons,
        patch=args.patch,
        stride=args.stride,
        time_stack=args.stack_time,
        split=args.split,
        val_frac=args.val_frac,
        seed=args.seed,
        normalize_inputs=(None if args.normalize_inputs == "none" else args.normalize_inputs),
        max_samples=args.max_samples,
        skip_nonpeat_patches=not args.no_skip_nonpeat,
        peat_min_fraction=args.peat_min_fraction,
        time_index=None,
        coord_as_features=True,
        return_coords=False,
    )

    print_diagnostic_header("Building datasets")

    train_ds = JointPeatDataset(mode="train", **common)
    print_diagnostic_item("Train patches", len(train_ds), indent=1)

    val_ds = JointPeatDataset(mode="val", **common)
    print_diagnostic_item("Val patches", len(val_ds), indent=1)

    test_ds = None
    if args.val_frac is not None and (1.0 - args.split - args.val_frac) > 0:
        test_ds = JointPeatDataset(mode="test", **common)
        print_diagnostic_item("Test patches", len(test_ds), indent=1)

    return train_ds, val_ds, test_ds


def split_train_for_calibration(train_ds, args):
    frac = float(getattr(args, "calib_frac", 0.0))
    if frac <= 0.0:
        return train_ds, None

    N = len(train_ds)
    n_cal = int(round(N * frac))
    n_cal = max(1, min(n_cal, N - 1))

    rng = np.random.RandomState(args.seed + 1337)
    idx = np.arange(N)
    rng.shuffle(idx)

    cal_idx = idx[:n_cal]
    tr_idx = idx[n_cal:]

    calib_ds = Subset(train_ds, cal_idx)
    train_ds2 = Subset(train_ds, tr_idx)

    print_diagnostic_header("Calibration split")
    print_diagnostic_item("Train (after split)", len(train_ds2), indent=1)
    print_diagnostic_item("Calib", len(calib_ds), indent=1)
    return train_ds2, calib_ds


# ----------------------------------------------------------------------
# Tabular sampling for XGBoost
# ----------------------------------------------------------------------

@dataclass
class TabularSamples:
    X: List[np.ndarray]  # per horizon (N_k, F)
    y: List[np.ndarray]  # per horizon (N_k,)


@torch.no_grad()
def collect_tabular_samples(ds, args, split_name: str, monitor: Optional[RAMMonitor] = None) -> TabularSamples:
    """Sample valid pixels from patches to create manageable XGB training matrices."""
    K = len(args.horizons)
    X_list: List[List[np.ndarray]] = [[] for _ in range(K)]
    y_list: List[List[np.ndarray]] = [[] for _ in range(K)]
    n_collected = [0 for _ in range(K)]

    loader = make_loader(ds, batch_size=args.batch_size, shuffle=True, args=args)
    iterator = wrap_loader(loader, desc=f"sample({split_name})", use_tqdm=not args.no_tqdm, position=2)

    for batch in iterator:
        if monitor is not None:
            monitor.set_status(f"Sampling {split_name}")

        x = _flatten_time_if_needed(batch["x"])  # (B,F,H,W)
        y = (batch["y"] > 0.5)
        m = (batch["mask"] > 0.5)

        B, F, H, W = x.shape

        for k in range(K):
            if args.xgb_max_train_tiles > 0 and n_collected[k] >= args.xgb_max_train_tiles:
                continue

            # sample per sample in batch for better control
            for b in range(B):
                valid_b = m[b, k]  # (H,W)
                nv = int(valid_b.sum().item())
                if nv <= 0:
                    continue

                # choose how many to sample from this patch
                if args.xgb_tiles_per_patch > 0:
                    n_take = min(args.xgb_tiles_per_patch, nv)
                else:
                    frac = float(args.xgb_sample_frac)
                    n_take = int(max(1, round(nv * frac)))

                # global cap
                if args.xgb_max_train_tiles > 0:
                    remaining = args.xgb_max_train_tiles - n_collected[k]
                    if remaining <= 0:
                        break
                    n_take = min(n_take, remaining)

                # positions
                idx = torch.nonzero(valid_b, as_tuple=False)  # (Nv,2) [y,x]
                if idx.shape[0] == 0:
                    continue

                if idx.shape[0] > n_take:
                    perm = torch.randperm(idx.shape[0])[:n_take]
                    idx = idx[perm]

                ys = idx[:, 0]
                xs = idx[:, 1]

                feat_t = x[b, :, ys, xs]
                if feat_t.dim() == 2 and feat_t.shape[0] == x.shape[1]:
                    feat_t = feat_t.permute(1, 0)
                feat = feat_t.contiguous().cpu().numpy().astype(np.float32)

                lab = y[b, k, ys, xs].contiguous().cpu().numpy().astype(np.int64)  # (n_take,)

                X_list[k].append(feat)
                y_list[k].append(lab)
                n_collected[k] += int(lab.size)

        if args.xgb_max_train_tiles > 0 and all(n >= args.xgb_max_train_tiles for n in n_collected):
            break

    X_out: List[np.ndarray] = []
    y_out: List[np.ndarray] = []

    for k in range(K):
        if X_list[k]:
            Xk = np.concatenate(X_list[k], axis=0)
            yk = np.concatenate(y_list[k], axis=0)
        else:
            Xk = np.zeros((0, 1), dtype=np.float32)
            yk = np.zeros((0,), dtype=np.int64)
        X_out.append(Xk)
        y_out.append(yk)

    print_diagnostic_header(f"Tabular samples: {split_name}")
    for k, h in enumerate(args.horizons):
        print_diagnostic_item(f"h={h} samples", human_int(int(y_out[k].size)), indent=1)

    return TabularSamples(X=X_out, y=y_out)


# ----------------------------------------------------------------------
# CSV logging
# ----------------------------------------------------------------------

def init_metrics_csv(logdir: str) -> str:
    os.makedirs(logdir, exist_ok=True)
    path = os.path.join(logdir, "metrics_log.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch", "split", "loss", "acc", "prec", "rec", "f1",
                "calib", "prob_thresh_pct", "prob_thresh_prob", "gt_count", "gt_total", "gt_frac"
            ])
    return path


def append_metrics_csv(path: str, epoch: int, split: str, loss: float, metrics: Dict[str, Any], calib_method: str):
    pr = metrics.get("prob_gt", {}) or {}
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            epoch, split, loss,
            metrics.get("accuracy", float("nan")),
            metrics.get("precision", float("nan")),
            metrics.get("recall", float("nan")),
            metrics.get("f1", float("nan")),
            calib_method,
            pr.get("threshold_pct", float("nan")),
            pr.get("threshold_prob", float("nan")),
            pr.get("count", float("nan")),
            pr.get("total", float("nan")),
            pr.get("fraction", float("nan")),
        ])


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # --- Zarr data roots ---
    p.add_argument("--era5-zarr", required=True)
    p.add_argument("--smap-zarr", required=True)
    p.add_argument("--smap_l4-zarr", required=True)
    p.add_argument("--viirs-zarr", required=True)

    p.add_argument("--era5-array", default="field")
    p.add_argument("--smap-array", default="field")
    p.add_argument("--smap_l4-array", default="field")
    p.add_argument("--viirs-array", default="field")

    # --- Dataset hyperparameters ---
    p.add_argument("--T-hist", type=int, default=30)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 7, 14])
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--stack-time", choices=["separate", "channel"], default="channel")
    p.add_argument("--split", type=float, default=0.9)
    p.add_argument("--val-frac", type=float, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--normalize-inputs", choices=["none", "per_channel"], default="none")
    p.add_argument("--no-skip-nonpeat", action="store_true")
    p.add_argument("--peat-min-fraction", type=float, default=0.01)

    # --- Loader/seed ---
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size for patch sampling/eval. For big patches, keep this small.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--mp-context", choices=["auto", "spawn", "fork", "forkserver"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-tqdm", action="store_true")

    # --- XGBoost training sampling ---
    p.add_argument("--xgb-sample-frac", type=float, default=0.02,
                   help="Fraction of valid pixels per patch to include in training samples (if tiles_per_patch=0).")
    p.add_argument("--xgb-tiles-per-patch", type=int, default=0,
                   help="If >0, sample this many valid pixels per patch (overrides xgb_sample_frac).")
    p.add_argument("--xgb-max-train-tiles", type=int, default=300_000,
                   help="Max training samples PER HORIZON (caps memory).")

    # --- XGBoost hyperparameters ---
    p.add_argument("--xgb-n-estimators", type=int, default=400)
    p.add_argument("--xgb-learning-rate", type=float, default=0.05)
    p.add_argument("--xgb-max-depth", type=int, default=6)
    p.add_argument("--xgb-subsample", type=float, default=0.8)
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    p.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    p.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    p.add_argument("--xgb-gamma", type=float, default=0.0)
    p.add_argument("--xgb-max-bin", type=int, default=256)
    p.add_argument("--xgb-verbosity", type=int, default=1)

    # GPU/CPU selection. XGBoost 2.x supports tree_method='hist' + device='cuda'.
    # For older XGBoost, use tree_method='gpu_hist'.
    p.add_argument("--xgb-tree-method", choices=["hist", "gpu_hist"], default="hist")
    p.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cuda")

    # Imbalance handling
    p.add_argument("--xgb-use-scale-pos-weight", action="store_true",
                   help="If set, uses scale_pos_weight = n_neg/n_pos computed from sampled training labels.")

    # --- Metrics ---
    p.add_argument("--metrics-threshold", type=float, default=0.5)

    # --- Post-hoc calibration ---
    p.add_argument("--calibration-method", choices=["none", "platt", "isotonic"], default="none")
    p.add_argument("--calib-frac", type=float, default=0.0)
    p.add_argument("--calib-platt-steps", type=int, default=200)
    p.add_argument("--calib-platt-lr", type=float, default=0.05)
    p.add_argument("--isotonic-bins", type=int, default=400)

    # Calibration speed caps
    p.add_argument("--platt-on-gpu", action="store_true",
                   help="If set, optimize Platt a,b on CUDA (still predicts with XGB on CPU/GPU per xgb_device).")
    p.add_argument("--platt-max-tiles", type=int, default=50_000,
                   help="Max valid tiles per batch+horizon used for Platt step.")
    p.add_argument("--iso-max-tiles", type=int, default=200_000,
                   help="Max valid tiles per batch+horizon used for isotonic histogram updates.")

    # Evaluation speed cap (0 => exact on all valid pixels)
    p.add_argument("--eval-max-tiles", type=int, default=0,
                   help="If >0, caps evaluated valid pixels per batch+horizon (approximate metrics).")

    # TensorBoard-like logdir + saving
    p.add_argument("--logdir", default="runs/peat_xgb")

    # Threshold counting
    p.add_argument(
        "--prob-max-pct",
        type=float,
        default=0.06,
        help=(
            "Probability threshold expressed in percent. 0.06 means 0.06% => prob=0.0006. "
            "We count probabilities > (prob_max_pct/100)."
        ),
    )

    return p.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    print_diagnostic_header("Threshold sanity check")
    prob_thresh = float(args.prob_max_pct) / 100.0
    print_diagnostic_item("--prob-max-pct", f"{args.prob_max_pct}%", indent=1)
    print_diagnostic_item("Converted probability threshold", f"{prob_thresh:.8f}", indent=1)
    print("NOTE: 0.06% => 0.0006. 6% would be 0.06.")

    monitor = RAMMonitor(device_id=0)
    if not args.no_tqdm:
        monitor.start()

    with contextlib.ExitStack() as stack:
        start_ram = get_ram_usage()

        if monitor is not None:
            monitor.set_status("Dataset init")

        train_ds, val_ds, test_ds = build_datasets(args)
        train_ds, calib_ds = split_train_for_calibration(train_ds, args)

        if len(train_ds) == 0:
            raise RuntimeError("Train dataset is empty after filtering/splitting.")

        # Determine input feature count from one sample
        sample = train_ds[0]
        x0 = sample["x"]
        if x0.dim() == 4:  # (T,C,H,W)
            T, C, _, _ = x0.shape
            in_features = T * C
        elif x0.dim() == 3:  # (C,H,W)
            in_features = x0.shape[0]
        else:
            raise ValueError(f"Unexpected x shape from dataset: {tuple(x0.shape)}")

        print_diagnostic_header("Model")
        print_diagnostic_item("XGBoost per-horizon classifiers", "binary:logistic", indent=1)
        print_diagnostic_item("in_features", in_features, indent=1)
        print_diagnostic_item("horizons", args.horizons, indent=1)

        print_diagnostic_header("Sampling training tiles")
        if monitor is not None:
            monitor.set_status("Sampling train tiles")

        train_samples = collect_tabular_samples(train_ds, args, split_name="train", monitor=monitor)

        # Train per horizon
        models: List[xgb.XGBClassifier] = []
        for k, h in enumerate(args.horizons):
            Xk = train_samples.X[k]
            yk = train_samples.y[k]
            if yk.size == 0:
                raise RuntimeError(f"No training samples collected for horizon {h}. Increase xgb_sample_frac or relax filtering.")

            # imbalance
            scale_pos_weight = None
            if args.xgb_use_scale_pos_weight:
                n_pos = float((yk == 1).sum())
                n_neg = float((yk == 0).sum())
                if n_pos > 0:
                    scale_pos_weight = max(1.0, n_neg / max(n_pos, 1.0))

            params = dict(
                n_estimators=int(args.xgb_n_estimators),
                learning_rate=float(args.xgb_learning_rate),
                max_depth=int(args.xgb_max_depth),
                subsample=float(args.xgb_subsample),
                colsample_bytree=float(args.xgb_colsample_bytree),
                min_child_weight=float(args.xgb_min_child_weight),
                reg_lambda=float(args.xgb_reg_lambda),
                gamma=float(args.xgb_gamma),
                max_bin=int(args.xgb_max_bin),
                objective="binary:logistic",
                eval_metric="logloss",
                verbosity=int(args.xgb_verbosity),
                tree_method=str(args.xgb_tree_method),
            )

            # device handling
            if args.xgb_tree_method == "hist":
                # XGBoost 2.x path
                params["device"] = str(args.xgb_device)

            if scale_pos_weight is not None:
                params["scale_pos_weight"] = float(scale_pos_weight)

            print_diagnostic_header(f"Training XGB (horizon={h})")
            print_diagnostic_item("Samples", human_int(int(yk.size)), indent=1)
            if scale_pos_weight is not None:
                print_diagnostic_item("scale_pos_weight", f"{scale_pos_weight:.2f}", indent=1)

            if monitor is not None:
                monitor.set_status(f"Training XGB h={h}")

            clf = xgb.XGBClassifier(**params)
            clf.fit(Xk, yk)
            models.append(clf)

        # Optional calibration
        calib_loader = make_loader(calib_ds, batch_size=args.batch_size, shuffle=False, args=args) if calib_ds is not None else None
        calibrator = None
        calib_method = "none"
        if calib_loader is not None and args.calibration_method != "none":
            calib_method = args.calibration_method
            print_diagnostic_header(f"Calibration ({calib_method})")
            calibrator = fit_calibrator_xgb(models, calib_loader, args, monitor=monitor)

        # Eval
        metrics_csv_path = init_metrics_csv(args.logdir)

        val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False, args=args) if val_ds is not None else None
        test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False, args=args) if test_ds is not None else None

        if val_loader is not None:
            if monitor is not None:
                monitor.set_status("Eval(val)")
            val_loss, val_metrics = evaluate_xgb(models, val_loader, args, calibrator=calibrator, use_tqdm=not args.no_tqdm, monitor=monitor)

            pr = val_metrics.get("prob_gt", {}) or {}
            print_diagnostic_header("Validation")
            print(f"[Val] calib={calib_method} loss={val_loss:.6f} acc={val_metrics['accuracy']:.4f} "
                  f"prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}")
            print(f"[Val] P(prob > {pr['threshold_pct']}% = {pr['threshold_prob']:.8f}) : "
                  f"{pr['count']:.0f}/{pr['total']:.0f} ({100.0*pr['fraction']:.4f}%)")
            for h, hh in (pr.get("per_horizon", {}) or {}).items():
                print(f"  [Val h={h}] {hh['count']:.0f}/{hh['total']:.0f} ({100.0*hh['fraction']:.4f}%)")

            append_metrics_csv(metrics_csv_path, 1, "val", val_loss, val_metrics, calib_method)

        if test_loader is not None:
            if monitor is not None:
                monitor.set_status("Eval(test)")
            test_loss, test_metrics = evaluate_xgb(models, test_loader, args, calibrator=calibrator, use_tqdm=not args.no_tqdm, monitor=monitor)

            pr = test_metrics.get("prob_gt", {}) or {}
            print_diagnostic_header("Test")
            print(f"[Test] calib={calib_method} loss={test_loss:.6f} acc={test_metrics['accuracy']:.4f} "
                  f"prec={test_metrics['precision']:.4f} rec={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}")
            print(f"[Test] P(prob > {pr['threshold_pct']}% = {pr['threshold_prob']:.8f}) : "
                  f"{pr['count']:.0f}/{pr['total']:.0f} ({100.0*pr['fraction']:.4f}%)")
            for h, hh in (pr.get("per_horizon", {}) or {}).items():
                print(f"  [Test h={h}] {hh['count']:.0f}/{hh['total']:.0f} ({100.0*hh['fraction']:.4f}%)")

            append_metrics_csv(metrics_csv_path, 1, "test", test_loss, test_metrics, calib_method)

        # Save
        os.makedirs(args.logdir, exist_ok=True)
        for k, h in enumerate(args.horizons):
            path = os.path.join(args.logdir, f"xgb_h{h}.json")
            models[k].get_booster().save_model(path)
        if calibrator is not None:
            with open(os.path.join(args.logdir, "calibrator.json"), "w") as f:
                json.dump({
                    "method": calibrator.method,
                    "horizons": calibrator.horizons,
                    "a": None if calibrator.a is None else calibrator.a.tolist(),
                    "b": None if calibrator.b is None else calibrator.b.tolist(),
                    "iso_x": None if calibrator.iso_x is None else [None if v is None else v.tolist() for v in calibrator.iso_x],
                    "iso_y": None if calibrator.iso_y is None else [None if v is None else v.tolist() for v in calibrator.iso_y],
                }, f)

        print_diagnostic_header("Done")
        print_diagnostic_item("Logdir", args.logdir, indent=1)
        print_diagnostic_item("RAM delta", f"{(get_ram_usage() - start_ram):+.2f} GB", indent=1)

    if monitor is not None:
        monitor.stop()


if __name__ == "__main__":
    main()
