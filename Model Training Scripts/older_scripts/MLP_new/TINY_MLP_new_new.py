#!/usr/bin/env python3
"""Spatiotemporal Peat Ignition — Tiny per-tile MLP (speed & memory)

Includes:
- Optional post-hoc calibration on a held-out calibration split (Platt / Isotonic)
- Real-time RAM/GPU monitoring + dataloader timing
- Script-2-style metrics:
    logloss / ROC-AUC / ECE / MCE / Brier
    calibration curves (bin_pred/bin_true/bin_count + ROC arrays)
    prob_range counts in [--reliability-bin-min, --reliability-bin-max]
- Script-2-style artifacts:
    metrics_log.csv (overall + per-horizon rows)
    metrics_history.json
    matplotlib plots (history curves) + ROC curves
    reliability_bias_{split}_epoch{epoch}.npz

Restored CLI features:
- --dropout
- --patch-sampling (none|weighted)
- --plot-metrics (save Script-2-style plots)
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
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from joint_peat_dataset_builder import JointPeatDataset

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:
    print("SOMETHING WRONG W LINE 52 go check")

# ----------------------------------------------------------------------
# Small utils
# ----------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def human_int(n: int) -> str:
    n = float(n)
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            s = f"{n:.1f}{unit}"
            return s.rstrip("0").rstrip(".")
        n /= 1000.0
    return f"{n:.1f}T"


def get_ram_usage() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def get_gpu_memory(device_id: int = 0) -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device_id) / (1024**3)
    return 0.0


def header(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def amp_autocast(device: torch.device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


# ----------------------------------------------------------------------
# Monitoring
# ----------------------------------------------------------------------

class RAMMonitor:
    def __init__(self, device_id=0, update_interval=0.5):
        self.device_id = device_id
        self.update_interval = update_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.pbar = None
        self.status = "Initializing..."

    def start(self):
        self.running = True
        self.pbar = tqdm(total=0, position=0, bar_format="{desc}", leave=True)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.pbar:
            self.pbar.close()

    def set(self, status: str):
        self.status = status

    def _loop(self):
        while self.running:
            desc = f"💾 RAM: {get_ram_usage():.2f}GB"
            if torch.cuda.is_available():
                desc += f" | 🎮 GPU: {get_gpu_memory(self.device_id):.2f}GB"
            desc += f" | {self.status}"
            if self.pbar:
                self.pbar.set_description_str(desc)
            time.sleep(self.update_interval)


class TimedDataLoader:
    def __init__(self, loader: DataLoader, desc: str, use_tqdm: bool, position: int):
        self.loader = loader
        self.desc = desc
        self.use_tqdm = use_tqdm
        self.position = position
        self.batch_times: deque = deque(maxlen=50)

    def __iter__(self):
        it = iter(self.loader)
        total = None
        try:
            total = len(self.loader)
        except TypeError:
            pass

        pbar = None
        if self.use_tqdm:
            pbar = tqdm(total=total, desc=self.desc, leave=True, position=self.position)

        while True:
            try:
                t0 = time.time()
                batch = next(it)
                dt = time.time() - t0
                self.batch_times.append(dt)
                if pbar:
                    avg = sum(self.batch_times) / len(self.batch_times)
                    pbar.set_postfix({"load_ms": f"{dt*1000:.0f}", "avg_ms": f"{avg*1000:.0f}"})
                    pbar.update(1)
                yield batch
            except StopIteration:
                break

        if pbar:
            pbar.close()


def _choose_mp_context(requested: str) -> Optional[str]:
    if requested != "auto":
        return requested
    return "forkserver" if sys.platform.startswith("linux") else "spawn"


def collate(batch):
    if not batch:
        raise RuntimeError("Empty batch in collate (dataset filtering too aggressive?).")
    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals, 0) if torch.is_tensor(vals[0]) else vals
    return out


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


# ----------------------------------------------------------------------
# Model: Per-tile MLP (2-layer)
# ----------------------------------------------------------------------

class TileMLP2Layer(nn.Module):
    """Per-pixel independent MLP.

    Input:
      - batch['x']: (B,T,C,H,W) or (B,F,H,W)
    Output:
      - logits: (B,K,H,W)
    """

    def __init__(
        self,
        in_features: int,
        horizons: Sequence[int],
        hidden: int = 64,
        dropout: float = 0.10,
        chunk_tiles: int = 0,
        norm: str = "bn",
        bn_momentum: float = 0.10,
        bn_eps: float = 1e-5,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.horizons = [int(h) for h in horizons]
        self.K = len(self.horizons)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.chunk_tiles = int(chunk_tiles)

        self.fc1 = nn.Linear(self.in_features, self.hidden)
        if norm == "bn":
            self.norm1 = nn.BatchNorm1d(self.hidden, momentum=float(bn_momentum), eps=float(bn_eps))
        elif norm == "ln":
            self.norm1 = nn.LayerNorm(self.hidden)
        else:
            self.norm1 = nn.Identity()

        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=self.dropout)
        self.fc2 = nn.Linear(self.hidden, self.K)

    @staticmethod
    def _flatten_time_if_needed(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.view(B, T * C, H, W)
        raise ValueError(f"Expected x with 4 or 5 dims, got {tuple(x.shape)}")

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x = self._flatten_time_if_needed(batch["x"])  # (B,F,H,W)
        B, F, H, W = x.shape
        x2 = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, F)  # (N,F)

        if self.chunk_tiles > 0 and x2.shape[0] > self.chunk_tiles:
            outs = []
            for i in range(0, x2.shape[0], self.chunk_tiles):
                xi = x2[i : i + self.chunk_tiles]
                hi = self.drop(self.act(self.norm1(self.fc1(xi))))
                outs.append(self.fc2(hi))
            out = torch.cat(outs, dim=0)
        else:
            h1 = self.drop(self.act(self.norm1(self.fc1(x2))))
            out = self.fc2(h1)

        return out.view(B, H, W, self.K).permute(0, 3, 1, 2).contiguous()


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------

class MaskedBCEWithLogits(nn.Module):
    def __init__(self, enable_class_weights: bool = False, max_pos_weight: float = 100.0):
        super().__init__()
        self.enable_class_weights = bool(enable_class_weights)
        self.max_pos_weight = float(max_pos_weight)

    def forward(self, logits, targets, mask):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.enable_class_weights:
            with torch.no_grad():
                valid = mask > 0.5
                pos = (targets > 0.5) & valid
                neg = (~pos) & valid
                n_pos = pos.sum().item()
                n_neg = neg.sum().item()
                if n_pos > 0 and n_neg > 0:
                    w_fire = min(n_neg / max(float(n_pos), 1.0), self.max_pos_weight)
                    weights = torch.ones_like(loss)
                    weights[pos] = w_fire
                    weights[~(pos | neg)] = 0.0
                else:
                    weights = torch.ones_like(loss)
            loss = loss * weights
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)


class MaskedFocalLossWithLogits(nn.Module):
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        enable_class_weights: bool = False,
        max_pos_weight: float = 100.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.enable_class_weights = bool(enable_class_weights)
        self.max_pos_weight = float(max_pos_weight)

    def forward(self, logits, targets, mask):
        probs = torch.sigmoid(logits)
        t = (targets > 0.5).float()
        p_t = probs * t + (1.0 - probs) * (1.0 - t)
        alpha_t = self.alpha * t + (1.0 - self.alpha) * (1.0 - t)
        focal_term = (1.0 - p_t).pow(self.gamma)
        loss = -alpha_t * focal_term * torch.log(p_t.clamp(min=1e-6))

        if self.enable_class_weights:
            with torch.no_grad():
                valid = mask > 0.5
                pos = (targets > 0.5) & valid
                neg = (~pos) & valid
                n_pos = pos.sum().item()
                n_neg = neg.sum().item()
                if n_pos > 0 and n_neg > 0:
                    w_fire = min(n_neg / max(float(n_pos), 1.0), self.max_pos_weight)
                    weights = torch.ones_like(loss)
                    weights[pos] = w_fire
                    weights[~(pos | neg)] = 0.0
                else:
                    weights = torch.ones_like(loss)
            loss = loss * weights

        return (loss * mask).sum() / mask.sum().clamp(min=1.0)


class MaskedFocalTverskyLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, logits, targets, mask):
        probs = torch.sigmoid(logits)
        t = (targets > 0.5).float()
        m = (mask > 0.5).float()
        probs = probs * m
        t = t * m
        tp = (probs * t).sum(dim=(0, 2, 3))
        fp = (probs * (1.0 - t)).sum(dim=(0, 2, 3))
        fn = ((1.0 - probs) * t).sum(dim=(0, 2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - tversky).pow(self.gamma).mean()


def make_criterion(loss_name: str, args) -> nn.Module:
    if loss_name == "bce":
        return MaskedBCEWithLogits(args.enable_pixel_class_weights, args.max_fire_class_weight)
    if loss_name == "focal":
        return MaskedFocalLossWithLogits(
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
            enable_class_weights=args.enable_pixel_class_weights,
            max_pos_weight=args.max_fire_class_weight,
        )
    if loss_name == "focal_tversky":
        return MaskedFocalTverskyLossWithLogits(
            alpha=args.tversky_alpha,
            beta=args.tversky_beta,
            gamma=args.tversky_gamma,
        )
    raise ValueError(f"Unknown loss: {loss_name}")


# ----------------------------------------------------------------------
# Post-hoc probability calibration (Platt / Isotonic)
# ----------------------------------------------------------------------

class ProbCalibrator:
    def __init__(self, method: str, horizons: Sequence[int]):
        self.method = str(method)
        self.horizons = [int(h) for h in horizons]
        self.K = len(self.horizons)
        self.a: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.iso_x: list[Optional[torch.Tensor]] = [None] * self.K
        self.iso_y: list[Optional[torch.Tensor]] = [None] * self.K

    def to(self, device: torch.device):
        if self.method == "platt":
            if self.a is not None:
                self.a = self.a.to(device)
            if self.b is not None:
                self.b = self.b.to(device)
        elif self.method == "isotonic":
            for k in range(self.K):
                if self.iso_x[k] is not None:
                    self.iso_x[k] = self.iso_x[k].to(device)
                if self.iso_y[k] is not None:
                    self.iso_y[k] = self.iso_y[k].to(device)
        return self

    def apply_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.method in (None, "none"):
            return torch.sigmoid(logits)
        if self.method == "platt" and self.a is not None and self.b is not None:
            a = self.a.view(1, -1, 1, 1)
            b = self.b.view(1, -1, 1, 1)
            return torch.sigmoid(logits * a + b)

        # isotonic
        p = torch.sigmoid(logits)
        out = torch.empty_like(p)
        for k in range(p.shape[1]):
            x = self.iso_x[k]
            y = self.iso_y[k]
            out[:, k] = p[:, k] if x is None or y is None or x.numel() < 2 else _torch_interp1d(p[:, k], x, y)
        return out


def _torch_interp1d(p: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    M = x.numel()
    idx = torch.bucketize(p, x)
    idx0 = (idx - 1).clamp(0, M - 1)
    idx1 = idx.clamp(0, M - 1)
    x0, x1 = x[idx0], x[idx1]
    y0, y1 = y[idx0], y[idx1]
    denom = (x1 - x0).clamp(min=1e-12)
    t = ((p - x0) / denom).clamp(0.0, 1.0)
    return (y0 + t * (y1 - y0)).clamp(0.0, 1.0)


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
            means[-2], weights[-2], sizes[-2] = float(m_new), float(w_new), int(s_new)
            means.pop(), weights.pop(), sizes.pop()

    return np.concatenate([np.full(sz, m, dtype=np.float64) for m, sz in zip(means, sizes)], axis=0)


@torch.no_grad()
def fit_calibrator(model, calib_loader, device, args) -> Optional[ProbCalibrator]:
    method = args.calibration_method
    if method == "none" or calib_loader is None:
        return None

    model.eval()
    K = len(args.horizons)
    cal = ProbCalibrator(method=method, horizons=args.horizons)

    if method == "platt":
        with torch.enable_grad():
            a = torch.ones(K, device=device, dtype=torch.float32, requires_grad=True)
            b = torch.zeros(K, device=device, dtype=torch.float32, requires_grad=True)
            opt = torch.optim.Adam([a, b], lr=float(args.calib_platt_lr))

            it = iter(calib_loader)
            for _ in range(max(1, int(args.calib_platt_steps))):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(calib_loader)
                    batch = next(it)

                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
                with torch.no_grad():
                    logits = model(batch)

                y = (batch["y"] > 0.5).float()
                m = (batch["mask"] > 0.5).float()

                z = logits * a.view(1, -1, 1, 1) + b.view(1, -1, 1, 1)
                loss_map = F.binary_cross_entropy_with_logits(z, y, reduction="none")
                loss = (loss_map * m).sum() / m.sum().clamp(min=1.0)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        cal.a, cal.b = a.detach().cpu(), b.detach().cpu()
        return cal

    if method == "isotonic":
        nb = max(16, int(args.isotonic_bins))
        counts = np.zeros((K, nb), dtype=np.float64)
        true_sums = np.zeros((K, nb), dtype=np.float64)

        for batch in calib_loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            probs = torch.sigmoid(model(batch))
            targets = batch["y"] > 0.5
            valid = batch["mask"] > 0.5

            B, K_eff, H, W = probs.shape
            K_eff = min(K_eff, K)
            for k in range(K_eff):
                v = valid[:, k].reshape(-1)
                if v.sum().item() == 0:
                    continue
                p = probs[:, k].reshape(-1)[v].detach().cpu().numpy().astype(np.float64)
                y = targets[:, k].reshape(-1)[v].detach().cpu().numpy().astype(np.float64)

                idx = np.floor(p * nb).astype(np.int64)
                idx = np.clip(idx, 0, nb - 1)
                counts[k] += np.bincount(idx, minlength=nb).astype(np.float64)
                true_sums[k] += np.bincount(idx, weights=y, minlength=nb).astype(np.float64)

        bin_centers = (np.arange(nb, dtype=np.float64) + 0.5) / nb
        for k in range(K):
            nz = counts[k] > 0
            if nz.sum() < 2:
                continue
            x = bin_centers[nz]
            y_mean = true_sums[k][nz] / np.maximum(counts[k][nz], 1e-12)
            y_iso = _pav_weighted(y_mean, counts[k][nz])
            cal.iso_x[k] = torch.tensor(x, dtype=torch.float32).cpu()
            cal.iso_y[k] = torch.tensor(y_iso, dtype=torch.float32).cpu()

        return cal

    return None


# ----------------------------------------------------------------------
# Patch sampling (restored)
# ----------------------------------------------------------------------

@torch.no_grad()
def _patch_has_any_fire(sample: Dict[str, Any]) -> bool:
    y = sample.get("y")
    m = sample.get("mask")
    if y is None or m is None or (not torch.is_tensor(y)) or (not torch.is_tensor(m)):
        return False
    return bool(((y > 0.5) & (m > 0.5)).any().item())


def compute_patch_weights(ds, pos_weight: float, max_items: Optional[int] = None) -> torch.DoubleTensor:
    """Binary patch weights: positive patches get pos_weight, others 1.0."""
    N = len(ds)
    if N <= 0:
        raise RuntimeError("Empty dataset for weight computation.")

    if max_items is not None and N > max_items:
        idx = np.random.RandomState(123).choice(N, size=max_items, replace=False)
        flag = np.zeros(N, dtype=np.bool_)
        for i in tqdm(idx, desc="scan weights (subsample)"):
            flag[i] = _patch_has_any_fire(ds[int(i)])
        weights = np.ones(N, dtype=np.float64)
        weights[flag] = float(pos_weight)
        return torch.tensor(weights, dtype=torch.double)

    flag = np.zeros(N, dtype=np.bool_)
    for i in tqdm(range(N), desc="scan weights"):
        flag[i] = _patch_has_any_fire(ds[i])

    weights = np.ones(N, dtype=np.float64)
    weights[flag] = float(pos_weight)
    return torch.tensor(weights, dtype=torch.double)


# ----------------------------------------------------------------------
# CSV logging (match Script 2)
# ----------------------------------------------------------------------

def init_metrics_csv(logdir: str) -> str:
    os.makedirs(logdir, exist_ok=True)
    path = os.path.join(logdir, "metrics_log.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch",
                "split",
                "loss",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "logloss",
                "roc_auc",
                "tp",
                "fp",
                "fn",
                "tn",
                "support",
                "horizon",
                "calibration_method",
            ])
    return path


def append_metrics_csv(path: str, epoch: int, split: str, loss: float, metrics: Dict[str, Any], calib_method: str):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)

        # overall row
        w.writerow([
            epoch,
            split,
            loss,
            metrics.get("accuracy", float("nan")),
            metrics.get("precision", float("nan")),
            metrics.get("recall", float("nan")),
            metrics.get("f1", float("nan")),
            metrics.get("logloss", float("nan")),
            metrics.get("roc_auc", float("nan")),
            metrics.get("tp", ""),
            metrics.get("fp", ""),
            metrics.get("fn", ""),
            metrics.get("tn", ""),
            metrics.get("support", ""),
            "",
            calib_method,
        ])

        # per-horizon rows
        per_h = metrics.get("per_horizon", {}) or {}
        for h, m in per_h.items():
            w.writerow([
                epoch,
                f"{split}_h{int(h)}",
                "",
                m.get("accuracy", float("nan")),
                m.get("precision", float("nan")),
                m.get("recall", float("nan")),
                m.get("f1", float("nan")),
                m.get("logloss", float("nan")),
                m.get("roc_auc", float("nan")),
                m.get("tp", ""),
                m.get("fp", ""),
                m.get("fn", ""),
                m.get("tn", ""),
                m.get("support", ""),
                int(h),
                calib_method,
            ])


# ----------------------------------------------------------------------
# Script-2-style history dumping + matplotlib plots
# ----------------------------------------------------------------------

def plot_reliability_bias_curve_fine(calib_fine: Dict[str, Any], out_path: str, args, title: str):
    import matplotlib.pyplot as plt
    import numpy as np

    xs = np.asarray(calib_fine.get("bin_centers", []), dtype=np.float64)
    ys = np.asarray(calib_fine.get("bias_pct", []), dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        return

    ys = np.ma.masked_invalid(ys)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhline(0.0, linewidth=1)
    ax.plot(xs, ys, linewidth=1.0)  # <-- NO markers, smooth line
    ax.set_title(title)
    ax.set_xlabel("Predicted probability bin center")
    ax.set_ylabel("Bias (%)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.plot_dpi, format=args.plot_file_format)
    plt.close(fig)

def tb_log_reliability_bias_curve_fine(writer: SummaryWriter, calib_fine: Dict[str, Any], epoch: int, tag: str, title: str):
    import matplotlib.pyplot as plt
    import numpy as np

    xs = np.asarray(calib_fine.get("bin_centers", []), dtype=np.float64)
    ys = np.asarray(calib_fine.get("bias_pct", []), dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        return

    ys = np.ma.masked_invalid(ys)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhline(0.0, linewidth=1)
    ax.plot(xs, ys, linewidth=1.0)  # <-- NO markers
    ax.set_title(title)
    ax.set_xlabel("Predicted probability bin center")
    ax.set_ylabel("Bias (%)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


def make_reliability_bias_cmap():
    # min (under) -> neutral (0) -> max (over)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "reliability_bias_exact",
        ["#1895b3", "#edd005", "#e01c29"],
        N=256,
    )
    cmap.set_bad(color="white")  # keep masked bins white
    return cmap


def plot_reliability_bias_heatmap_1d(calib: Dict[str, Any], out_path: str, args, title: str):
    import matplotlib.pyplot as plt
    import numpy as np

    xs = np.asarray(calib.get("bin_centers_slice", []), dtype=np.float64)
    ys = np.asarray(calib.get("reliability_bias_pct", []), dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        return

    xs_pct = xs * 100.0
    Z = ys.reshape(1, -1)  # (1, nbins)
    Z = np.ma.masked_invalid(Z)  # <-- critical: makes cmap.set_bad(color="white") actually apply

    cmap = make_reliability_bias_cmap()

    vmax = float(getattr(args, "reliability_bias_vmax", 0.8))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(9, 2.0))
    im = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[xs_pct.min(), xs_pct.max(), 0, 1],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",  # crisp & reliable
    )

    ax.set_yticks([0.5])
    ax.set_yticklabels(["All"])
    ax.set_xlabel("Probability of Fire (%)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Reliability Bias % (Model − Obs)")
    cbar.set_ticks([vmin, vmin / 2, 0.0, vmax / 2, vmax])

    fig.tight_layout()
    fig.savefig(out_path, dpi=args.plot_dpi, format=args.plot_file_format)
    plt.close(fig)


def tb_log_reliability_bias_heatmap_1d(
    writer: SummaryWriter,
    calib: Dict[str, Any],
    epoch: int,
    tag: str,
    title: str,
    args,
):
    import matplotlib.pyplot as plt
    import numpy as np

    xs = np.asarray(calib.get("bin_centers_slice", []), dtype=np.float64)
    ys = np.asarray(calib.get("reliability_bias_pct", []), dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        return

    xs_pct = xs * 100.0
    Z = ys.reshape(1, -1)
    Z = np.ma.masked_invalid(Z)  # <-- critical

    cmap = make_reliability_bias_cmap()

    vmax = float(getattr(args, "reliability_bias_vmax", 0.8))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(9, 2.0))
    im = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[xs_pct.min(), xs_pct.max(), 0, 1],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # --- Probability gradient key (maps x-position to probability visually) ---
    prob_ax = inset_axes(
        ax,
        width="100%",
        height="18%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.55, 1.0, 1.0),   # pushes it below the main axis
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )

    # Make a 1xN horizontal gradient over the same x-range as the heatmap
    x0, x1 = float(xs_pct.min()), float(xs_pct.max())
    grad = np.linspace(0.0, 1.0, 512, dtype=np.float64)[None, :]  # values just to colorize the strip

    prob_ax.imshow(
        grad,
        aspect="auto",
        origin="lower",
        extent=[x0, x1, 0, 1],
        cmap="viridis",          # any sequential cmap is fine
        interpolation="nearest",
    )

    prob_ax.set_yticks([])
    prob_ax.set_xlim(x0, x1)

    # Choose a few clean tick marks (edit these to taste)
    ticks = [x0, 1.0, 2.0, 4.0, 6.0]
    ticks = [t for t in ticks if x0 <= t <= x1]
    prob_ax.set_xticks(ticks)
    prob_ax.set_xticklabels([f"{t:.1f}%" for t in ticks])
    prob_ax.set_xlabel("Probability key (x-axis)")

    # Optional: mark the slice endpoints clearly
    prob_ax.axvline(x0, linewidth=1.0)
    prob_ax.axvline(x1, linewidth=1.0)

    
    ax.set_yticks([0.5])
    ax.set_yticklabels(["All"])
    ax.set_xlabel("Probability of Fire (%)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Reliability Bias % (Model − Obs)")
    cbar.set_ticks([vmin, vmin / 2, 0.0, vmax / 2, vmax])

    fig.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)




def save_history_json(history: Dict[str, Any], logdir: str) -> str:
    os.makedirs(logdir, exist_ok=True)
    path = os.path.join(logdir, "metrics_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Plots] Saved raw history JSON to {path}")
    return path


def _plot_multi_curves(curves, title: str, xlabel: str, ylabel: str, out_path: str, args):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Plots] matplotlib unavailable ({e}); skipping {title}.")
        return

    fig, ax = plt.subplots()
    has_data = False
    for xs, ys, label in curves:
        if not xs or not ys:
            continue
        n = min(len(xs), len(ys))
        if n <= 0:
            continue
        ax.plot(xs[:n], ys[:n], marker="o", label=label)
        has_data = True

    if not has_data:
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.plot_dpi, format=args.plot_file_format)
    plt.close(fig)
    print(f"[Plots] Saved {title} to {out_path}")


def plot_metric_history(history: Dict[str, Any], args):
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)

    train = history.get("train", {})
    val = history.get("val", {})

    _plot_multi_curves(
        [(train.get("epoch", []), train.get("loss", []), "train_loss"),
         (val.get("epoch", []), val.get("loss", []), "val_loss")],
        title="Train vs Val Loss",
        xlabel="Epoch",
        ylabel="Loss",
        out_path=os.path.join(logdir, f"loss_curves.{args.plot_file_format}"),
        args=args,
    )

    _plot_multi_curves(
        [(val.get("epoch", []), val.get("precision", []), "val_precision"),
         (val.get("epoch", []), val.get("recall", []), "val_recall"),
         (val.get("epoch", []), val.get("f1", []), "val_f1")],
        title="Validation Precision / Recall / F1",
        xlabel="Epoch",
        ylabel="Score",
        out_path=os.path.join(logdir, f"val_core_metrics.{args.plot_file_format}"),
        args=args,
    )

    _plot_multi_curves(
        [(val.get("epoch", []), val.get("tp", []), "TP"),
         (val.get("epoch", []), val.get("fp", []), "FP"),
         (val.get("epoch", []), val.get("fn", []), "FN"),
         (val.get("epoch", []), val.get("tn", []), "TN")],
        title="Validation Confusion Counts",
        xlabel="Epoch",
        ylabel="Count",
        out_path=os.path.join(logdir, f"val_confusion_counts.{args.plot_file_format}"),
        args=args,
    )

    _plot_multi_curves(
        [(val.get("epoch", []), val.get("logloss", []), "val_logloss")],
        title="Validation Log Loss",
        xlabel="Epoch",
        ylabel="Log Loss",
        out_path=os.path.join(logdir, f"val_logloss.{args.plot_file_format}"),
        args=args,
    )

    _plot_multi_curves(
        [(val.get("epoch", []), val.get("roc_auc", []), "val_roc_auc")],
        title="Validation ROC AUC",
        xlabel="Epoch",
        ylabel="AUC",
        out_path=os.path.join(logdir, f"val_roc_auc.{args.plot_file_format}"),
        args=args,
    )

    per_h = (val.get("per_horizon", {}) or {})
    for h, hh in per_h.items():
        _plot_multi_curves(
            [(hh.get("epoch", []), hh.get("precision", []), f"h={h} precision"),
             (hh.get("epoch", []), hh.get("recall", []), f"h={h} recall"),
             (hh.get("epoch", []), hh.get("f1", []), f"h={h} f1")],
            title=f"Validation Metrics (horizon={h})",
            xlabel="Epoch",
            ylabel="Score",
            out_path=os.path.join(logdir, f"val_h{h}_metrics.{args.plot_file_format}"),
            args=args,
        )


def plot_roc_curves_from_metrics(metrics: Dict[str, Any], args, split: str, epoch: Optional[int] = None):
    calib = metrics.get("calibration", {}) or {}
    roc_overall = calib.get("roc", {}) or {}

    curves = []
    fpr_o = roc_overall.get("fpr", [])
    tpr_o = roc_overall.get("tpr", [])
    auc_o = roc_overall.get("auc", float("nan"))
    label = "overall"
    if isinstance(auc_o, (int, float)) and not math.isnan(float(auc_o)):
        label += f" (AUC={float(auc_o):.3f})"
    curves.append((fpr_o, tpr_o, label))

    per_h = calib.get("per_horizon", {}) or {}
    for h, hc in per_h.items():
        roc_h = (hc.get("roc", {}) or {})
        fpr_h = roc_h.get("fpr", [])
        tpr_h = roc_h.get("tpr", [])
        auc_h = roc_h.get("auc", float("nan"))
        label_h = f"h={h}"
        if isinstance(auc_h, (int, float)) and not math.isnan(float(auc_h)):
            label_h += f" (AUC={float(auc_h):.3f})"
        curves.append((fpr_h, tpr_h, label_h))

    suffix = split
    if epoch is not None:
        suffix += f"_epoch{epoch}"

    out_path = os.path.join(args.logdir, f"roc_curves_{suffix}.{args.plot_file_format}")
    _plot_multi_curves(
        curves,
        title=f"ROC Curves ({split})",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        out_path=out_path,
        args=args,
    )


# ----------------------------------------------------------------------
# Reliability-bias saving/logging (npz + TensorBoard)
# ----------------------------------------------------------------------

def save_reliability_bias_npz(calib: Dict[str, Any], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = {}
    data["bin_centers"] = np.asarray(calib.get("bin_centers_slice", []), dtype=np.float64)
    data["bias_pct"] = np.asarray(calib.get("reliability_bias_pct", []), dtype=np.float64)
    data["count"] = np.asarray(calib.get("reliability_bias_count", []), dtype=np.float64)

    per_h = calib.get("per_horizon", {}) or {}
    for h, hc in per_h.items():
        h = int(h)
        data[f"h{h}_bin_centers"] = np.asarray(
            hc.get("bin_centers_slice", calib.get("bin_centers_slice", [])), dtype=np.float64
        )
        data[f"h{h}_bias_pct"] = np.asarray(hc.get("reliability_bias_pct", []), dtype=np.float64)
        data[f"h{h}_count"] = np.asarray(hc.get("reliability_bias_count", []), dtype=np.float64)

    np.savez_compressed(out_path, **data)


def tb_log_reliability_bias_curve(writer: SummaryWriter, calib: Dict[str, Any], epoch: int, tag: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    xs = np.asarray(calib.get("bin_centers_slice", []), dtype=np.float64)
    ys = np.asarray(calib.get("reliability_bias_pct", []), dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhline(0.0, linewidth=1)
    ax.plot(xs, ys, marker="o")
    ax.set_title("Reliability Bias % (Model − Obs)")
    ax.set_xlabel("Predicted probability bin center")
    ax.set_ylabel("Bias (%)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


def save_and_log_calibration(split: str, metrics: Dict[str, Any], epoch: int, args, writer: Optional[SummaryWriter]):
    calib = metrics.get("calibration", {}) or {}

    calib_fine = (calib.get("fine", {}) or {})
    if calib_fine:
        try:
            out_img_fine = os.path.join(
                args.logdir,
                f"reliability_bias_finecurve_{split}_epoch{epoch}.{args.plot_file_format}",
            )
            plot_reliability_bias_curve_fine(
                calib_fine,
                out_img_fine,
                args,
                title=f"Fine Reliability Bias % (bin={calib_fine.get('bin_width','?')})",
            )
            if writer is not None:
                tb_log_reliability_bias_curve_fine(
                    writer,
                    calib_fine,
                    epoch,
                    tag=f"{split}/reliability_bias_finecurve",
                    title="Fine Reliability Bias % (Model − Obs)",
                )
        except Exception as e:
            print(f"[ReliabilityBias] Failed to write fine curve ({split}): {e}")

        # ✅ NEW: per-horizon fine curves (file + TensorBoard)
        try:
            fine_ph = (calib_fine.get("per_horizon", {}) or {})
            for h, hc in fine_ph.items():
                if not isinstance(hc, dict):
                    continue
                h_int = int(h)

                out_img_fine_h = os.path.join(
                    args.logdir,
                    f"reliability_bias_finecurve_{split}_h{h_int}_epoch{epoch}.{args.plot_file_format}",
                )
                plot_reliability_bias_curve_fine(
                    hc,
                    out_img_fine_h,
                    args,
                    title=f"Fine Reliability Bias % (h={h_int}, bin={calib_fine.get('bin_width','?')})",
                )

                if writer is not None:
                    tb_log_reliability_bias_curve_fine(
                        writer,
                        hc,
                        epoch,
                        tag=f"{split}/h{h_int}_reliability_bias_finecurve",
                        title=f"Fine Reliability Bias % (h={h_int})",
                    )
        except Exception as e:
            print(f"[ReliabilityBias] Failed per-horizon fine curves ({split}): {e}")



    out_npz = os.path.join(args.logdir, f"reliability_bias_{split}_epoch{epoch}.npz")
    try:
        save_reliability_bias_npz(calib, out_npz)
    except Exception as e:
        print(f"[ReliabilityBias] Failed to save NPZ ({split}): {e}")

    # --- NEW: single 1D heatmap output (file + TB), and ONLY this plot ---
    try:
        out_img = os.path.join(
            args.logdir,
            f"reliability_bias_heatmap_{split}_epoch{epoch}.{args.plot_file_format}",
        )
        plot_reliability_bias_heatmap_1d(
            calib,
            out_img,
            args,
            title="Reliability Bias % (Model − Obs)",
        )
        if writer is not None:
            tb_log_reliability_bias_heatmap_1d(
                writer,
                calib,
                epoch,
                tag=f"{split}/reliability_bias_heatmap",
                title="Reliability Bias % (Model − Obs)",
                args=args
            )
    except Exception as e:
        print(f"[ReliabilityBias] Failed to write heatmap ({split}): {e}")    
    
    if writer is not None:
        try:
            tb_log_reliability_bias_curve(writer, calib, epoch, tag=f"{split}/reliability_bias_pct")
        except Exception as e:
            print(f"[ReliabilityBias] Failed to log TB figure ({split}): {e}")

        per_h = calib.get("per_horizon", {}) or {}
        for h, hc in per_h.items():
            if isinstance(hc, dict) and ("reliability_bias_pct" in hc):
                try:
                    tb_log_reliability_bias_curve(writer, hc, epoch, tag=f"{split}/h{int(h)}_reliability_bias_pct")
                except Exception as e:
                    print(f"[ReliabilityBias] Failed per-horizon TB ({split}, h={h}): {e}")


# ----------------------------------------------------------------------
# Evaluation + Script-2-style metrics
# ----------------------------------------------------------------------

def _compute_roc_from_hist(counts: np.ndarray, true_sums: np.ndarray):
    counts = counts.astype(np.float64)
    pos = true_sums.astype(np.float64)
    neg = counts - pos

    P = pos.sum()
    N = neg.sum()
    if P <= 0 or N <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), float("nan")

    pos_rev = pos[::-1]
    neg_rev = neg[::-1]

    tp_cum = np.cumsum(pos_rev)
    fp_cum = np.cumsum(neg_rev)

    tpr = tp_cum / P
    fpr = fp_cum / N

    fpr = np.concatenate(([0.0], fpr))
    tpr = np.concatenate(([0.0], tpr))

    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def _basic_metrics(tp, fp, fn, tn):
    support = tp + fp + fn + tn
    if support == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": support,
        }
    acc = (tp + tn) / support
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "support": support,
    }


@torch.no_grad()
def evaluate(model, loader, device, criterion, args, calibrator: Optional[ProbCalibrator], use_tqdm: bool):
    model.eval()
    tot_loss, tot_mask = 0.0, 0.0
    tp_total = fp_total = fn_total = tn_total = 0

    num_h = len(args.horizons)
    tp_h = [0] * num_h
    fp_h = [0] * num_h
    fn_h = [0] * num_h
    tn_h = [0] * num_h

    bin_width = float(getattr(args, "reliability_bin_width", 0.005))
    num_bins = max(1, int(math.ceil(1.0 / bin_width)))

    rel_counts = np.zeros((num_h, num_bins), dtype=np.float64)
    rel_pred_sums = np.zeros_like(rel_counts)
    rel_true_sums = np.zeros_like(rel_counts)

    brier_sums = np.zeros(num_h, dtype=np.float64)
    total_counts_per_h = np.zeros(num_h, dtype=np.float64)

    range_min = float(getattr(args, "reliability_bin_min", 0.0))
    range_max = float(getattr(args, "reliability_bin_max", 0.06))

    fine_w = float(getattr(args, "reliability_fine_bin_width", 1e-6))
    fine_bins = max(1, int(math.ceil(1.0 / max(fine_w, 1e-12))))  # bins cover [0,1]



    fine_counts = np.zeros((num_h, fine_bins), dtype=np.float64)
    fine_pred_sums = np.zeros_like(fine_counts)
    fine_true_sums = np.zeros_like(fine_counts)
 
    
    # counts of probs that fall in/out of the reliability-bias *slice* (post-calibration)
    prob_inrange_h = np.zeros(num_h, dtype=np.float64)
    prob_total_h = np.zeros(num_h, dtype=np.float64)
    prob_above_max_h = np.zeros(num_h, dtype=np.float64)   # NEW
    prob_below_min_h = np.zeros(num_h, dtype=np.float64)   # NEW (optional but handy)

    # ✅ NEW: probability distribution stats (post-calibration probs)
    prob_sum_h = np.zeros(num_h, dtype=np.float64)
    prob_sumsq_h = np.zeros(num_h, dtype=np.float64)
    prob_min_h_stat = np.full(num_h, np.inf, dtype=np.float64)
    prob_max_h_stat = np.full(num_h, -np.inf, dtype=np.float64)

    dist_bins = max(1000, int(getattr(args, "prob_stats_bins", 20000)))
    prob_hist_h = np.zeros((num_h, dist_bins), dtype=np.float64)  # used for approximate percentiles


    logloss_sum_overall = 0.0
    logloss_count_overall = 0.0
    logloss_sum_h = np.zeros(num_h, dtype=np.float64)
    logloss_count_h = np.zeros(num_h, dtype=np.float64)

    if calibrator is not None:
        calibrator = calibrator.to(device)

    for batch in TimedDataLoader(loader, desc="eval", use_tqdm=use_tqdm, position=3):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        logits = model(batch)
        loss = criterion(logits, batch["y"], batch["mask"])
        m = batch["mask"].sum().item()
        if m == 0:
            continue
        tot_loss += loss.item() * m
        tot_mask += m

        probs = calibrator.apply_logits(logits) if calibrator is not None else torch.sigmoid(logits)
        targets = batch["y"] > 0.5
        valid = batch["mask"] > 0.5

        preds = probs >= float(args.metrics_threshold)

        pf = preds.reshape(-1)
        tf = targets.reshape(-1)
        vf = valid.reshape(-1)

        tp_total += (pf & tf & vf).sum().item()
        fp_total += (pf & ~tf & vf).sum().item()
        fn_total += (~pf & tf & vf).sum().item()
        tn_total += (~pf & ~tf & vf).sum().item()

        B, K, H, W = logits.shape
        K_eff = min(K, num_h)

        # per-horizon confusion
        for k_idx in range(K_eff):
            pk = preds[:, k_idx].reshape(-1)
            tk = targets[:, k_idx].reshape(-1)
            vk = valid[:, k_idx].reshape(-1)
            tp_h[k_idx] += (pk & tk & vk).sum().item()
            fp_h[k_idx] += (pk & ~tk & vk).sum().item()
            fn_h[k_idx] += (~pk & tk & vk).sum().item()
            tn_h[k_idx] += (~pk & ~tk & vk).sum().item()

        # calibration hist / logloss / brier / prob_range
        for k_idx in range(K_eff):
            v_k = valid[:, k_idx].reshape(-1)
            if v_k.sum().item() == 0:
                continue

            p_flat = probs[:, k_idx].reshape(-1)[v_k].detach().cpu().numpy().astype(np.float32)
            y_flat = targets[:, k_idx].reshape(-1)[v_k].float().detach().cpu().numpy().astype(np.float32)

            # --- NEW: fine-bin reliability (only within [fine_min, fine_max]) ---
            pf = p_flat.astype(np.float64, copy=False)
            yf = y_flat.astype(np.float64, copy=False)

            fi = np.floor(pf / fine_w).astype(np.int64)   # bin in [0,1]
            fi = np.clip(fi, 0, fine_bins - 1)

            fine_counts[k_idx]     += np.bincount(fi, minlength=fine_bins).astype(np.float64)
            fine_pred_sums[k_idx]  += np.bincount(fi, weights=pf, minlength=fine_bins).astype(np.float64)
            fine_true_sums[k_idx]  += np.bincount(fi, weights=yf, minlength=fine_bins).astype(np.float64)



            n_valid = float(p_flat.size)
            if n_valid == 0:
                continue

            # ✅ NEW: stats: min/mean/max/std + histogram for percentiles
            pf64 = p_flat.astype(np.float64, copy=False)
            prob_sum_h[k_idx] += float(pf64.sum())
            prob_sumsq_h[k_idx] += float((pf64 * pf64).sum())

            # pf64 is non-empty here because v_k.sum() > 0 => p_flat.size > 0
            prob_min_h_stat[k_idx] = min(prob_min_h_stat[k_idx], float(pf64.min()))
            prob_max_h_stat[k_idx] = max(prob_max_h_stat[k_idx], float(pf64.max()))

            b = np.floor(pf64 * dist_bins).astype(np.int64)
            b = np.clip(b, 0, dist_bins - 1)
            prob_hist_h[k_idx] += np.bincount(b, minlength=dist_bins).astype(np.float64)



            in_range = (p_flat >= range_min) & (p_flat <= range_max)
            above_max = (p_flat > range_max)   # NEW
            below_min = (p_flat < range_min)   # NEW

            prob_inrange_h[k_idx] += float(in_range.sum())
            prob_above_max_h[k_idx] += float(above_max.sum())   # NEW
            prob_below_min_h[k_idx] += float(below_min.sum())   # NEW
            prob_total_h[k_idx] += n_valid


            total_counts_per_h[k_idx] += n_valid
            brier_sums[k_idx] += float(((p_flat - y_flat) ** 2).sum())

            eps = 1e-7
            p_clip = np.clip(p_flat, eps, 1.0 - eps)
            ll = -(y_flat * np.log(p_clip) + (1.0 - y_flat) * np.log(1.0 - p_clip))
            s_ll = float(ll.sum())
            logloss_sum_overall += s_ll
            logloss_count_overall += n_valid
            logloss_sum_h[k_idx] += s_ll
            logloss_count_h[k_idx] += n_valid

            bin_idx = np.floor(p_flat / bin_width).astype(np.int64)
            bin_idx = np.clip(bin_idx, 0, num_bins - 1)

            rel_counts[k_idx] += np.bincount(bin_idx, minlength=num_bins).astype(np.float64)
            rel_pred_sums[k_idx] += np.bincount(bin_idx, weights=p_flat, minlength=num_bins).astype(np.float64)
            rel_true_sums[k_idx] += np.bincount(bin_idx, weights=y_flat, minlength=num_bins).astype(np.float64)

    # ------------------------------------------------------------------
    # Fine reliability-bias: slice to observed post-calibration min/max
    # ------------------------------------------------------------------

    # You already tracked post-calibration min/max per horizon during eval:
    #   prob_min_h_stat[k_idx], prob_max_h_stat[k_idx]
    finite_mins = prob_min_h_stat[np.isfinite(prob_min_h_stat)]
    finite_maxs = prob_max_h_stat[np.isfinite(prob_max_h_stat)]

    mn_all = float(finite_mins.min()) if finite_mins.size else 0.0
    mx_all = float(finite_maxs.max()) if finite_maxs.size else 1.0

    # convert observed min/max into fine-bin index range
    i0 = int(np.floor(mn_all / fine_w))
    i1 = int(np.ceil(mx_all / fine_w)) + 1

    i0 = max(0, min(i0, fine_bins))
    i1 = max(i0 + 1, min(i1, fine_bins))

    # centers for JUST the observed range
    fine_centers = (np.arange(i0, i1, dtype=np.float64) + 0.5) * fine_w

    # slice fine histograms to just [min,max]
    fine_counts_s    = fine_counts[:, i0:i1]
    fine_pred_sums_s = fine_pred_sums[:, i0:i1]
    fine_true_sums_s = fine_true_sums[:, i0:i1]

    fine_bias_per_h = {}
    fine_min_count = int(getattr(args, "reliability_fine_min_count", 1))

    # per-horizon fine bias (USING SLICED ARRAYS)
    for idx, h in enumerate(args.horizons):
        c = fine_counts_s[idx]
        nz = c > 0

        pred_m = np.full_like(c, np.nan, dtype=np.float64)
        true_m = np.full_like(c, np.nan, dtype=np.float64)

        pred_m[nz] = fine_pred_sums_s[idx][nz] / c[nz]
        true_m[nz] = fine_true_sums_s[idx][nz] / c[nz]

        bias = (pred_m - true_m) * 100.0
        bias[c < fine_min_count] = np.nan

        fine_bias_per_h[int(h)] = {
            "bin_centers": fine_centers.tolist(),
            "bias_pct": bias.tolist(),
            "count": c.tolist(),
        }

    # overall fine bias (sum horizons) (USING SLICED ARRAYS)
    c_all  = fine_counts_s.sum(axis=0)
    ps_all = fine_pred_sums_s.sum(axis=0)
    ts_all = fine_true_sums_s.sum(axis=0)

    nz = c_all > 0
    pred_m = np.full_like(c_all, np.nan, dtype=np.float64)
    true_m = np.full_like(c_all, np.nan, dtype=np.float64)

    pred_m[nz] = ps_all[nz] / c_all[nz]
    true_m[nz] = ts_all[nz] / c_all[nz]

    bias_all = (pred_m - true_m) * 100.0
    bias_all[c_all < fine_min_count] = np.nan

    calibration_fine = {
        "bin_centers": fine_centers.tolist(),
        "bias_pct": bias_all.tolist(),
        "count": c_all.tolist(),
        "per_horizon": fine_bias_per_h,
        "bin_width": fine_w,
        "min": mn_all,   # <-- actual observed min post-calib
        "max": mx_all,   # <-- actual observed max post-calib
    }



    val_loss = float("nan") if tot_mask == 0 else (tot_loss / tot_mask)

    overall = _basic_metrics(tp_total, fp_total, fn_total, tn_total)
    logloss_overall = float("nan") if logloss_count_overall <= 0 else float(
        logloss_sum_overall / max(logloss_count_overall, 1.0)
    )

    per_horizon_metrics: Dict[int, Dict[str, Any]] = {}
    for idx, h in enumerate(args.horizons):
        m = _basic_metrics(tp_h[idx], fp_h[idx], fn_h[idx], tn_h[idx])
        m["logloss"] = float("nan") if logloss_count_h[idx] <= 0 else float(
            logloss_sum_h[idx] / max(logloss_count_h[idx], 1.0)
        )
        per_horizon_metrics[int(h)] = m

    bin_centers = (np.arange(num_bins, dtype=np.float64) + 0.5) * bin_width
    calibration_per_h: Dict[int, Dict[str, Any]] = {}

    b0 = int(math.floor(float(args.reliability_bin_min) / bin_width))
    b1 = int(math.ceil(float(args.reliability_bin_max) / bin_width))
    b0 = max(0, min(b0, num_bins))
    b1 = max(0, min(b1, num_bins))

    for idx, h in enumerate(args.horizons):
        counts = rel_counts[idx]
        true_sums = rel_true_sums[idx]

        if counts.sum() == 0:
            calibration_per_h[int(h)] = {
                "bin_pred": [], "bin_true": [], "bin_count": [],
                "ece": float("nan"), "mce": float("nan"), "brier": float("nan"),
                "roc": {"fpr": [], "tpr": [], "auc": float("nan")},
                "reliability_bias_pct": [], "reliability_bias_count": [], "bin_centers_slice": [],
            }
            per_horizon_metrics[int(h)]["ece"] = float("nan")
            per_horizon_metrics[int(h)]["brier"] = float("nan")
            per_horizon_metrics[int(h)]["roc_auc"] = float("nan")
            continue

        nonzero = counts > 0
        pred_mean = np.zeros_like(counts)
        true_mean = np.zeros_like(counts)
        pred_mean[nonzero] = rel_pred_sums[idx][nonzero] / counts[nonzero]
        true_mean[nonzero] = true_sums[nonzero] / counts[nonzero]

        total = counts.sum()
        gap = np.abs(pred_mean - true_mean)
        ece = float((counts[nonzero] / total * gap[nonzero]).sum())
        mce = float(gap[nonzero].max()) if np.any(nonzero) else float("nan")
        brier = float(brier_sums[idx] / max(total_counts_per_h[idx], 1.0))

        fpr_h, tpr_h, auc_h = _compute_roc_from_hist(counts, true_sums)

        bias_pct = (pred_mean - true_mean) * 100.0
        bias_pct_slice = bias_pct[b0:b1].copy()
        count_slice = counts[b0:b1].copy()

        minc = int(getattr(args, "reliability_min_count", 0))
        if minc > 0:
            bias_pct_slice[count_slice < minc] = np.nan

        calibration_per_h[int(h)] = {
            "bin_pred": pred_mean.tolist(),
            "bin_true": true_mean.tolist(),
            "bin_count": counts.tolist(),
            "ece": ece,
            "mce": mce,
            "brier": brier,
            "roc": {"fpr": fpr_h.tolist(), "tpr": tpr_h.tolist(), "auc": float(auc_h)},
            "reliability_bias_pct": bias_pct_slice.tolist(),
            "reliability_bias_count": count_slice.tolist(),
            "bin_centers_slice": bin_centers[b0:b1].tolist(),
        }

        per_horizon_metrics[int(h)]["ece"] = ece
        per_horizon_metrics[int(h)]["brier"] = brier
        per_horizon_metrics[int(h)]["roc_auc"] = float(auc_h)

    # overall calibration by summing horizons
    overall_counts = rel_counts.sum(axis=0)
    overall_true_sum = rel_true_sums.sum(axis=0)
    overall_pred_sum = rel_pred_sums.sum(axis=0)

    if overall_counts.sum() == 0:
        overall_pred_mean = np.zeros_like(overall_counts)
        overall_true_mean = np.zeros_like(overall_counts)
        ece_overall = float("nan")
        mce_overall = float("nan")
        brier_overall = float("nan")
        fpr_o = np.array([], dtype=np.float64)
        tpr_o = np.array([], dtype=np.float64)
        auc_o = float("nan")
    else:
        nonzero = overall_counts > 0
        overall_pred_mean = np.zeros_like(overall_counts)
        overall_true_mean = np.zeros_like(overall_counts)
        overall_pred_mean[nonzero] = overall_pred_sum[nonzero] / overall_counts[nonzero]
        overall_true_mean[nonzero] = overall_true_sum[nonzero] / overall_counts[nonzero]

        total_o = overall_counts.sum()
        gap = np.abs(overall_pred_mean - overall_true_mean)
        ece_overall = float((overall_counts[nonzero] / total_o * gap[nonzero]).sum())
        mce_overall = float(gap[nonzero].max()) if np.any(nonzero) else float("nan")
        total_pixels = float(total_counts_per_h.sum())
        brier_overall = float(brier_sums.sum() / max(total_pixels, 1.0))

        fpr_o, tpr_o, auc_o = _compute_roc_from_hist(overall_counts, overall_true_sum)

    overall_bias_pct = (overall_pred_mean - overall_true_mean) * 100.0
    overall_bias_pct_slice = overall_bias_pct[b0:b1].copy()
    overall_count_slice = overall_counts[b0:b1].copy()

    minc = int(getattr(args, "reliability_min_count", 0))
    if minc > 0:
        overall_bias_pct_slice[overall_count_slice < minc] = np.nan

    prob_inrange_overall = float(prob_inrange_h.sum())
    prob_above_max_overall = float(prob_above_max_h.sum())   # NEW
    prob_below_min_overall = float(prob_below_min_h.sum())   # NEW
    prob_total_overall = float(prob_total_h.sum())

    prob_frac_overall = float("nan") if prob_total_overall <= 0 else (prob_inrange_overall / prob_total_overall)
    prob_above_frac_overall = float("nan") if prob_total_overall <= 0 else (prob_above_max_overall / prob_total_overall)  # NEW
    prob_below_frac_overall = float("nan") if prob_total_overall <= 0 else (prob_below_min_overall / prob_total_overall)  # NEW

    prob_range_per_h = {}
    for idx, h in enumerate(args.horizons):
        tot = float(prob_total_h[idx])
        cnt = float(prob_inrange_h[idx])
        above = float(prob_above_max_h[idx])   # NEW
        below = float(prob_below_min_h[idx])   # NEW

        frac = float("nan") if tot <= 0 else (cnt / tot)
        above_frac = float("nan") if tot <= 0 else (above / tot)  # NEW
        below_frac = float("nan") if tot <= 0 else (below / tot)  # NEW

        prob_range_per_h[int(h)] = {
            "count": cnt,
            "total": tot,
            "fraction": frac,
            "above_max_count": above,            # NEW
            "above_max_fraction": above_frac,    # NEW
            "below_min_count": below,            # NEW
            "below_min_fraction": below_frac,    # NEW
        }

    # =========================
    # Step 2D: finalize prob_stats
    # =========================
    def _percentiles_from_hist(hist: np.ndarray, qs=(1, 5, 50, 95, 99)):
        total = hist.sum()
        if total <= 0:
            return {f"p{q}": float("nan") for q in qs}
        cdf = np.cumsum(hist) / total
        out = {}
        for q in qs:
            target = q / 100.0
            idx = int(np.searchsorted(cdf, target, side="left"))
            idx = max(0, min(idx, hist.size - 1))
            out[f"p{q}"] = (idx + 0.5) / hist.size  # back to probability in [0,1]
        return out

    prob_stats_per_h: Dict[int, Dict[str, Any]] = {}
    for idx, h in enumerate(args.horizons):
        tot = float(prob_total_h[idx])  # IMPORTANT: same denominator you already maintain
        if tot <= 0:
            d = {"min": float("nan"), "mean": float("nan"), "max": float("nan"), "std": float("nan")}
            d.update(_percentiles_from_hist(prob_hist_h[idx]))
            prob_stats_per_h[int(h)] = d
            continue

        mean = float(prob_sum_h[idx] / tot)
        var = float(prob_sumsq_h[idx] / tot - mean * mean)
        std = float(np.sqrt(max(var, 0.0)))

        mn = float(prob_min_h_stat[idx]) if np.isfinite(prob_min_h_stat[idx]) else float("nan")
        mx = float(prob_max_h_stat[idx]) if np.isfinite(prob_max_h_stat[idx]) else float("nan")

        d = {"min": mn, "mean": mean, "max": mx, "std": std}
        d.update(_percentiles_from_hist(prob_hist_h[idx]))
        prob_stats_per_h[int(h)] = d

    # overall: just combine horizons (same “overall” spirit as your other overall metrics)
    tot_all = float(prob_total_h.sum())
    if tot_all <= 0:
        prob_stats_overall = {"min": float("nan"), "mean": float("nan"), "max": float("nan"), "std": float("nan")}
        prob_stats_overall.update({k: float("nan") for k in ["p1", "p5", "p50", "p95", "p99"]})
    else:
        mean_all = float(prob_sum_h.sum() / tot_all)
        var_all = float(prob_sumsq_h.sum() / tot_all - mean_all * mean_all)
        std_all = float(np.sqrt(max(var_all, 0.0)))

        finite_mins = prob_min_h_stat[np.isfinite(prob_min_h_stat)]
        finite_maxs = prob_max_h_stat[np.isfinite(prob_max_h_stat)]
        mn_all = float(finite_mins.min()) if finite_mins.size else float("nan")
        mx_all = float(finite_maxs.max()) if finite_maxs.size else float("nan")

        hist_all = prob_hist_h.sum(axis=0)
        prob_stats_overall = {"min": mn_all, "mean": mean_all, "max": mx_all, "std": std_all}
        prob_stats_overall.update(_percentiles_from_hist(hist_all))


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
        "logloss": logloss_overall,
        "roc_auc": float(auc_o),
        "per_horizon": per_horizon_metrics,
        "ece": ece_overall,
        "mce": mce_overall,
        "brier": brier_overall,
        "calibration": {
            "bin_centers": bin_centers.tolist(),
            "bin_pred": overall_pred_mean.tolist(),
            "bin_true": overall_true_mean.tolist(),
            "bin_count": overall_counts.tolist(),
            "ece": ece_overall,
            "mce": mce_overall,
            "brier": brier_overall,
            "roc": {"fpr": fpr_o.tolist(), "tpr": tpr_o.tolist(), "auc": float(auc_o)},
            "per_horizon": calibration_per_h,
            "reliability_bias_pct": overall_bias_pct_slice.tolist(),
            "reliability_bias_count": overall_count_slice.tolist(),
            "bin_centers_slice": bin_centers[b0:b1].tolist(),
            "fine": calibration_fine,

        },
        "prob_range": {
            "min": range_min,
            "max": range_max,
            "count": prob_inrange_overall,
            "total": prob_total_overall,
            "fraction": prob_frac_overall,

            # NEW: what doesn't show up in the sliced reliability-bias curve
            "above_max_count": prob_above_max_overall,
            "above_max_fraction": prob_above_frac_overall,
            "below_min_count": prob_below_min_overall,
            "below_min_fraction": prob_below_frac_overall,

            "per_horizon": prob_range_per_h,
        },
        
        "prob_stats": {
            "overall": prob_stats_overall,
            "per_horizon": prob_stats_per_h,
        },

    }

    return val_loss, metrics


def _print_split(epoch: int, split: str, loss: float, metrics: Dict[str, Any], calib_method: str):
    header(f"{split.upper()} (epoch {epoch})")
    print(
        f"[{split.capitalize()}] (calib={calib_method}) loss={loss:.6f}, "
        f"acc={metrics['accuracy']:.4f}, "
        f"prec={metrics['precision']:.4f}, "
        f"rec={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"logloss={metrics.get('logloss', float('nan')):.6f}, "
        f"auc={metrics.get('roc_auc', float('nan')):.4f}, "
        f"ece={metrics.get('ece', float('nan')):.4f}, "
        f"brier={metrics.get('brier', float('nan')):.4f}"
    )

    pr = metrics.get("prob_range", {}) or {}
    if pr:
        print(
            f"[{split.capitalize()}] prob_range [{pr['min']:.6f}, {pr['max']:.6f}]: "
            f"{pr['count']:.0f}/{pr['total']:.0f} ({100.0 * pr['fraction']:.3f}%)"
        )

        # NEW: outside the reliability-bias slice
        if "above_max_count" in pr:
            print(
                f"[{split.capitalize()}] probs > max({pr['max']:.6f}): "
                f"{pr['above_max_count']:.0f}/{pr['total']:.0f} ({100.0 * pr['above_max_fraction']:.3f}%)"
            )
        if "below_min_count" in pr:
            print(
                f"[{split.capitalize()}] probs < min({pr['min']:.6f}): "
                f"{pr['below_min_count']:.0f}/{pr['total']:.0f} ({100.0 * pr['below_min_fraction']:.3f}%)"
            )
        for h, hh in (pr.get("per_horizon", {}) or {}).items():
            line = (
                f"  [{split.capitalize()} prob_range h={h}] "
                f"{hh['count']:.0f}/{hh['total']:.0f} ({100.0 * hh['fraction']:.3f}%)"
            )
            if "above_max_count" in hh:
                line += f" | >max: {hh['above_max_count']:.0f} ({100.0 * hh['above_max_fraction']:.3f}%)"
            if "below_min_count" in hh:
                line += f" | <min: {hh['below_min_count']:.0f} ({100.0 * hh['below_min_fraction']:.3f}%)"
            print(line)

    # =========================
    # Step 2E: print prob_stats (reads metrics["prob_stats"] created in Step 2D)
    # =========================
    ps = (metrics.get("prob_stats", {}) or {}).get("overall", {}) or {}
    if ps:
        print(
            f"[{split.capitalize()}] prob_stats overall: "
            f"min={ps.get('min', float('nan')):.6g}, "
            f"mean={ps.get('mean', float('nan')):.6g}, "
            f"max={ps.get('max', float('nan')):.6g}, "
            f"std={ps.get('std', float('nan')):.6g} | "
            f"p1={ps.get('p1', float('nan')):.6g}, "
            f"p5={ps.get('p5', float('nan')):.6g}, "
            f"p50={ps.get('p50', float('nan')):.6g}, "
            f"p95={ps.get('p95', float('nan')):.6g}, "
            f"p99={ps.get('p99', float('nan')):.6g}"
        )

    per = (metrics.get("prob_stats", {}) or {}).get("per_horizon", {}) or {}
    for h, hh in per.items():
        print(
            f"  [{split.capitalize()} prob_stats h={h}] "
            f"min={hh.get('min', float('nan')):.6g}, "
            f"mean={hh.get('mean', float('nan')):.6g}, "
            f"max={hh.get('max', float('nan')):.6g}, "
            f"std={hh.get('std', float('nan')):.6g} | "
            f"p1={hh.get('p1', float('nan')):.6g}, "
            f"p5={hh.get('p5', float('nan')):.6g}, "
            f"p50={hh.get('p50', float('nan')):.6g}, "
            f"p95={hh.get('p95', float('nan')):.6g}, "
            f"p99={hh.get('p99', float('nan')):.6g}"
        )


    for h, m in (metrics.get("per_horizon", {}) or {}).items():
        print(
            f"  [{split.capitalize()} h={h}] acc={m['accuracy']:.4f}, "
            f"prec={m['precision']:.4f}, rec={m['recall']:.4f}, f1={m['f1']:.4f}, "
            f"logloss={m.get('logloss', float('nan')):.6f}, auc={m.get('roc_auc', float('nan')):.4f}, "
            f"tp={m.get('tp','')}, fp={m.get('fp','')}, fn={m.get('fn','')}, tn={m.get('tn','')}"
        )


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

    header("Building datasets")
    train_ds = JointPeatDataset(mode="train", **common)
    print(f"  • Train patches.............. {len(train_ds)}")

    val_ds = JointPeatDataset(mode="val", **common)
    print(f"  • Val patches................ {len(val_ds)}")

    test_ds = None
    if args.val_frac is not None and (1.0 - args.split - args.val_frac) > 0:
        test_ds = JointPeatDataset(mode="test", **common)
        print(f"  • Test patches............... {len(test_ds)}")

    return train_ds, val_ds, test_ds


def split_train_for_calibration(train_ds, args):
    frac = float(args.calib_frac)
    if frac <= 0.0:
        return train_ds, None

    N = len(train_ds)
    n_cal = max(1, min(int(round(N * frac)), N - 1))

    rng = np.random.RandomState(args.seed + 1337)
    idx = np.arange(N)
    rng.shuffle(idx)

    cal_idx = idx[:n_cal]
    tr_idx = idx[n_cal:]

    header("Calibration split")
    print(f"  • Train (after split)........ {len(tr_idx)}")
    print(f"  • Calib...................... {len(cal_idx)}")

    return Subset(train_ds, tr_idx), Subset(train_ds, cal_idx)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_one_epoch(model, loader, epoch, optimizer, criterion, device, args, monitor, writer, global_step):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    tot_loss, tot_mask = 0.0, 0.0
    last_step = -1
    
    for step, batch in enumerate(TimedDataLoader(loader, desc=f"train[e{epoch}]", use_tqdm=not args.no_tqdm, position=2)):
        last_step = step
        
        if monitor:
            monitor.set(f"Train e{epoch}/{args.epochs} step {step+1}")

        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        with amp_autocast(device):
            logits = model(batch)
            loss_raw = criterion(logits, batch["y"], batch["mask"])
            loss = loss_raw / max(1, int(args.grad_accum))

        loss.backward()

        if (step + 1) % int(args.grad_accum) == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        m = batch["mask"].sum().item()
        tot_loss += loss_raw.item() * m
        tot_mask += m

        if writer is not None:
            writer.add_scalar("train/loss_step", loss_raw.item(), global_step)
        global_step += 1
        
    # ✅ Flush leftover grads if epoch ended mid-accumulation
    if last_step >= 0 and ((last_step + 1) % int(args.grad_accum) != 0):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return (float("nan") if tot_mask == 0 else tot_loss / tot_mask), global_step


# ----------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Per-tile MLP
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--mlp-chunk-tiles", type=int, default=0, help="Chunk tiles to reduce peak memory.")
    p.add_argument("--mlp-norm", choices=["bn", "ln", "none"], default="bn")
    p.add_argument("--mlp-bn-momentum", type=float, default=0.10)
    p.add_argument("--mlp-bn-eps", type=float, default=1e-5)

    # Restored
    p.add_argument("--dropout", type=float, default=0.10, help="MLP dropout probability")
    p.add_argument(
        "--patch-sampling",
        choices=["none", "weighted"],
        default="none",
        help="Sampling strategy for training patches (none=shuffle, weighted=WeightedRandomSampler).",
    )
    p.add_argument(
        "--patch-pos-weight",
        type=float,
        default=5.0,
        help="If --patch-sampling weighted, patches containing any fire get this weight.",
    )
    p.add_argument(
        "--patch-weight-scan-max",
        type=int,
        default=None,
        help="If set, subsample at most this many items when scanning dataset to build weights.",
    )
    p.add_argument("--plot-metrics", action="store_true", help="Save Script-2-style plots + history JSON + ROC curves.")

    # --- Reliability / calibration metrics (match Script 2) ---
    p.add_argument("--reliability-bin-width", type=float, default=0.005,
                   help="Bin width in probability space (e.g. 0.005 = 0.5%).")
    p.add_argument("--reliability-bin-min", type=float, default=0.005)
    p.add_argument("--reliability-bin-max", type=float, default=0.060)
    p.add_argument("--reliability-min-count", type=int, default=50)

    p.add_argument("--reliability-fine-bin-width", type=float, default=1e-4,
               help="Fine bin width for smooth reliability-bias curve (e.g. 1e-6 = 0.0001%).")
    p.add_argument("--reliability-fine-min-count", type=int, default=1,
               help="Min count to keep a fine-bin point (others become NaN).")

    # --- Plotting outputs (match Script 2) ---
    p.add_argument("--plot-file-format", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--plot-dpi", type=int, default=150)

    # Optional: worker stats toggle (Script 2 has this; harmless to expose)
    p.add_argument("--show-worker-stats", action="store_true")

    # Zarr data roots
    p.add_argument("--era5-zarr", required=True)
    p.add_argument("--smap-zarr", required=True)
    p.add_argument("--smap_l4-zarr", required=True)
    p.add_argument("--viirs-zarr", required=True)

    p.add_argument("--era5-array", default="field")
    p.add_argument("--smap-array", default="field")
    p.add_argument("--smap_l4-array", default="field")
    p.add_argument("--viirs-array", default="field")

    # Dataset hyperparameters
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

    # Training
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--mp-context", choices=["auto", "spawn", "fork", "forkserver"], default="auto")
    p.add_argument("--seed", type=int, default=42)

    # Loss / imbalance
    p.add_argument("--loss", choices=["bce", "focal", "focal_tversky"], default="bce")
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--tversky-alpha", type=float, default=0.5)
    p.add_argument("--tversky-beta", type=float, default=0.5)
    p.add_argument("--tversky-gamma", type=float, default=1.5)
    p.add_argument("--enable-pixel-class-weights", action="store_true")
    p.add_argument("--max-fire-class-weight", type=float, default=100.0)

    p.add_argument("--train-loss", choices=["bce", "focal", "focal_tversky"], default=None)
    p.add_argument("--eval-loss", choices=["bce", "focal", "focal_tversky"], default="bce")

    # Metrics
    p.add_argument("--metrics-threshold", type=float, default=0.5)

    # Post-hoc calibration
    p.add_argument("--calibration-method", choices=["none", "platt", "isotonic"], default="none")
    p.add_argument("--calib-frac", type=float, default=0.0)
    p.add_argument("--calib-platt-steps", type=int, default=200)
    p.add_argument("--calib-platt-lr", type=float, default=0.05)
    p.add_argument("--isotonic-bins", type=int, default=400)

    p.add_argument("--prob-stats-bins", type=int, default=20000,
               help="Histogram bins for probability distribution stats (percentiles).")
    
    # Opt
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-tqdm", action="store_true")

    # TensorBoard
    p.add_argument("--logdir", default="runs/peat_mlp")
    p.add_argument("--no-tensorboard", action="store_true")

    # (Kept for backward compatibility; Script-2-style outputs use prob_range instead.)
    p.add_argument(
        "--prob-max-pct",
        type=float,
        default=0.06,
        help="(Deprecated for printing) Probability threshold expressed in percent. 0.06 means 0.06% => prob=0.0006.",
    )

    return p.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    header("Reliability settings sanity check")
    print(f"  • --reliability-bin-width..... {args.reliability_bin_width}")
    print(f"  • --reliability-bin-min....... {args.reliability_bin_min}")
    print(f"  • --reliability-bin-max....... {args.reliability_bin_max}")
    print(f"  • --reliability-min-count..... {args.reliability_min_count}")
    print("  • Note........................ prob_range counts use [bin_min, bin_max] (inclusive).")

    monitor = RAMMonitor(device_id=(device.index or 0) if device.type == "cuda" else 0)
    if not args.no_tqdm:
        monitor.start()

    train_ds, val_ds, test_ds = build_datasets(args)
    train_ds, calib_ds = split_train_for_calibration(train_ds, args)

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty after filtering/splitting.")

    sample = train_ds[0]
    x = sample["x"]
    if x.dim() == 4:  # (T,C,H,W)
        T, C, _, _ = x.shape
        in_channels = T * C
    elif x.dim() == 3:  # (C,H,W)
        in_channels = x.shape[0]
    else:
        raise ValueError(f"Unexpected x shape from dataset: {tuple(x.shape)}")

    model = TileMLP2Layer(
        in_features=in_channels,
        horizons=args.horizons,
        hidden=args.mlp_hidden,
        dropout=float(args.dropout),
        chunk_tiles=args.mlp_chunk_tiles,
        norm=args.mlp_norm,
        bn_momentum=args.mlp_bn_momentum,
        bn_eps=args.mlp_bn_eps,
    ).to(device)

    header("Model")
    tp, tr = count_params(model)
    print(f"  • in_features................. {in_channels}")
    print(f"  • horizons.................... {args.horizons}")
    print(f"  • dropout..................... {args.dropout}")
    print(f"  • params (total/trainable).... {human_int(tp)}/{human_int(tr)}")

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    train_loss_name = args.train_loss or args.loss
    eval_loss_name = args.eval_loss or args.loss
    train_criterion = make_criterion(train_loss_name, args).to(device)
    eval_criterion = make_criterion(eval_loss_name, args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    writer: Optional[SummaryWriter] = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=args.logdir)
        print(f"[TB] Logging to: {args.logdir}")

    metrics_csv_path = init_metrics_csv(args.logdir)

    # loaders
    train_sampler = None
    if args.patch_sampling == "weighted":
        header("Patch sampling")
        print("  • Building WeightedRandomSampler weights...")
        weights = compute_patch_weights(
            train_ds,
            pos_weight=float(args.patch_pos_weight),
            max_items=args.patch_weight_scan_max,
        )
        train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        pos_frac = float((weights.numpy() > 1.0).mean())
        print(f"  • Positive patch fraction (approx).. {100.0*pos_frac:.2f}%")

    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, args=args, sampler=train_sampler)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False, args=args) if val_ds is not None else None
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False, args=args) if test_ds is not None else None
    calib_loader = make_loader(calib_ds, batch_size=args.batch_size, shuffle=False, args=args) if calib_ds is not None else None

    best_val_f1, best_epoch = -1.0, -1
    global_step = 0

    last_val_metrics = None
    final_test_metrics = None

    history: Dict[str, Any] = {
        "train": {"epoch": [], "loss": []},
        "val": {
            "epoch": [],
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "logloss": [],
            "roc_auc": [],
            "tp": [],
            "fp": [],
            "fn": [],
            "tn": [],
            "support": [],
            "per_horizon": {
                int(h): {"epoch": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "logloss": [], "roc_auc": []}
                for h in args.horizons
            },
        },
    }

    for epoch in range(1, int(args.epochs) + 1):
        if monitor:
            monitor.set(f"Train e{epoch}/{args.epochs}")

        train_loss, global_step = train_one_epoch(
            model, train_loader, epoch, optimizer, train_criterion, device, args, monitor, writer, global_step
        )

        history["train"]["epoch"].append(epoch)
        history["train"]["loss"].append(float(train_loss))
        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)

        calibrator = None
        calib_method = "none"
        if calib_loader is not None and args.calibration_method != "none":
            calib_method = args.calibration_method
            if monitor:
                monitor.set(f"Calibrating ({calib_method}) e{epoch}/{args.epochs}")
            calibrator = fit_calibrator(model, calib_loader, device, args)

        if val_loader is not None:
            if monitor:
                monitor.set(f"Eval(val) e{epoch}/{args.epochs}")
            val_loss, val_metrics = evaluate(
                model, val_loader, device, eval_criterion, args, calibrator=calibrator, use_tqdm=not args.no_tqdm
            )
            _print_split(epoch, "val", val_loss, val_metrics, calib_method)
            append_metrics_csv(metrics_csv_path, epoch, "val", val_loss, val_metrics, calib_method)

            save_and_log_calibration("val", val_metrics, epoch, args, writer)

            # history
            vh = history["val"]
            vh["epoch"].append(epoch)
            vh["loss"].append(float(val_loss))
            vh["accuracy"].append(float(val_metrics["accuracy"]))
            vh["precision"].append(float(val_metrics["precision"]))
            vh["recall"].append(float(val_metrics["recall"]))
            vh["f1"].append(float(val_metrics["f1"]))
            vh["logloss"].append(float(val_metrics.get("logloss", float("nan"))))
            vh["roc_auc"].append(float(val_metrics.get("roc_auc", float("nan"))))
            vh["tp"].append(float(val_metrics.get("tp", 0)))
            vh["fp"].append(float(val_metrics.get("fp", 0)))
            vh["fn"].append(float(val_metrics.get("fn", 0)))
            vh["tn"].append(float(val_metrics.get("tn", 0)))
            vh["support"].append(float(val_metrics.get("support", 0)))

            for h, m in (val_metrics.get("per_horizon", {}) or {}).items():
                h_int = int(h)
                ph = vh["per_horizon"].setdefault(
                    h_int,
                    {"epoch": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "logloss": [], "roc_auc": []},
                )
                ph["epoch"].append(epoch)
                ph["accuracy"].append(float(m.get("accuracy", float("nan"))))
                ph["precision"].append(float(m.get("precision", float("nan"))))
                ph["recall"].append(float(m.get("recall", float("nan"))))
                ph["f1"].append(float(m.get("f1", float("nan"))))
                ph["logloss"].append(float(m.get("logloss", float("nan"))))
                ph["roc_auc"].append(float(m.get("roc_auc", float("nan"))))

            # TB (match Script 2 key scalars)
            if writer is not None:
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
                writer.add_scalar("val/precision", val_metrics["precision"], epoch)
                writer.add_scalar("val/recall", val_metrics["recall"], epoch)
                writer.add_scalar("val/f1", val_metrics["f1"], epoch)
                writer.add_scalar("val/logloss", val_metrics.get("logloss", float("nan")), epoch)
                writer.add_scalar("val/roc_auc", val_metrics.get("roc_auc", float("nan")), epoch)
                writer.add_scalar("val/ece", val_metrics.get("ece", float("nan")), epoch)
                writer.add_scalar("val/brier", val_metrics.get("brier", float("nan")), epoch)

            last_val_metrics = val_metrics

            if float(val_metrics.get("f1", -1.0)) > best_val_f1:
                best_val_f1 = float(val_metrics["f1"])
                best_epoch = epoch
                os.makedirs(args.logdir, exist_ok=True)
                best_path = os.path.join(args.logdir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_f1": best_val_f1,
                        "args": vars(args),
                        "calibration_method": calib_method,
                    },
                    best_path,
                )
                print(f"[Model] New best F1={best_val_f1:.4f} at epoch {epoch} -> {best_path}")

        if epoch == int(args.epochs) and test_loader is not None:
            if monitor:
                monitor.set(f"Eval(test) e{epoch}/{args.epochs}")
            test_loss, test_metrics = evaluate(
                model, test_loader, device, eval_criterion, args, calibrator=calibrator, use_tqdm=not args.no_tqdm
            )
            _print_split(epoch, "test", test_loss, test_metrics, calib_method)
            append_metrics_csv(metrics_csv_path, epoch, "test", test_loss, test_metrics, calib_method)

            save_and_log_calibration("test", test_metrics, epoch, args, writer)
            final_test_metrics = test_metrics

    header("Training complete")
    if best_epoch > 0:
        print(f"  • Best val F1.................. {best_val_f1:.4f} (epoch {best_epoch})")
    else:
        print("  • Best val F1.................. n/a")

    if writer is not None:
        writer.flush()
        writer.close()

    if monitor:
        monitor.stop()

    try:
        save_history_json(history, args.logdir)
        if args.plot_metrics:
            plot_metric_history(history, args)
            if last_val_metrics is not None:
                plot_roc_curves_from_metrics(last_val_metrics, args, split="val")
            if final_test_metrics is not None:
                plot_roc_curves_from_metrics(final_test_metrics, args, split="test")
    except Exception as e:
        print(f"[Plots] Failed to generate history JSON or plots: {e}")


if __name__ == "__main__":
    main()
