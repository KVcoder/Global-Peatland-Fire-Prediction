#!/usr/bin/env python3
"""
Spatiotemporal Peat Ignition — Heavily Optimized for Speed & Memory
WITH COMPREHENSIVE REAL-TIME MONITORING

This version adds OPTIONAL post-hoc probability calibration with a separate
calibration split held out from the training set:

  train 1 epoch -> fit calibrator on calib split -> evaluate on val/test using calibrated probs

Calibration methods:
  - Platt scaling: p = sigmoid(a*logit + b), fit per-horizon
  - Isotonic regression: p = iso(sigmoid(logit)) fit per-horizon using hist bins + PAV (memory-safe)

Also adds:
  - Count of probabilities (POST-CALIBRATION) greater than a percent threshold:
      --prob-max-pct 0.06  means 0.06% => prob threshold = 0.0006
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

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from joint_peat_dataset_builder import JointPeatDataset  # latest dataset with coord_as_features + test split


# ----------------------------------------------------------------------
# Small utils
# ----------------------------------------------------------------------

def exists(x):
    return x is not None


def default(val, d):
    return d if val is None else val


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
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


def get_gpu_memory(device_id: int = 0) -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device_id) / (1024 ** 3)
    return 0.0


def print_ram_delta(start_ram: float, label: str) -> float:
    current_ram = get_ram_usage()
    delta = current_ram - start_ram
    print(f"[RAM] {label}: {delta:+.2f} GB (total: {current_ram:.2f} GB)")
    return current_ram


def print_diagnostic_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_diagnostic_item(label: str, value: Any, indent: int = 0):
    prefix = "  " * indent + "• "
    print(f"{prefix}{label:.<40} {value}")


class DiagnosticTimer:
    def __init__(
        self,
        label: str,
        verbose: bool = True,
        track_ram: bool = False,
        track_gpu: bool = False,
        device_index: Optional[int] = None,
    ):
        self.label = label
        self.verbose = verbose
        self.track_ram = track_ram
        self.track_gpu = track_gpu
        self.device_index = 0 if device_index is None else device_index
        self.start_time: Optional[float] = None
        self.start_ram: Optional[float] = None
        self.start_gpu: Optional[float] = None

    def __enter__(self):
        if self.verbose:
            print(f"\n[DIAG] Starting: {self.label}")
        self.start_time = time.time()
        if self.track_ram:
            self.start_ram = get_ram_usage()
        if self.track_gpu and torch.cuda.is_available():
            self.start_gpu = get_gpu_memory(self.device_index)
        return self

    def __exit__(self, *args):
        elapsed = time.time() - (self.start_time or time.time())
        if self.verbose:
            info = f"✓ Completed in {elapsed:.2f}s"
            if self.track_ram and self.start_ram is not None:
                ram_delta = get_ram_usage() - self.start_ram
                info += f" | RAM: {ram_delta:+.2f} GB"
            if self.track_gpu and torch.cuda.is_available():
                gpu0 = 0.0 if self.start_gpu is None else self.start_gpu
                gpu_delta = get_gpu_memory(self.device_index) - gpu0
                info += f" | GPU: {gpu_delta:.2f} GB"
            print(f"[DIAG] {self.label}: {info}")


def amp_autocast(device: torch.device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


# ----------------------------------------------------------------------
# Model: Per-tile MLP (2-layer)
# ----------------------------------------------------------------------

class TileMLP2Layer(nn.Module):
    """
    Treat each pixel/tile independently:
      input: (B, T, C, H, W) or (B, F, H, W)
      flatten time+channels -> (B, F, H, W)
      per pixel: (F) -> hidden -> K logits
      output: (B, K, H, W)
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
        self.norm = str(norm)

        self.fc1 = nn.Linear(self.in_features, self.hidden)

        if self.norm == "bn":
            self.norm1 = nn.BatchNorm1d(self.hidden, momentum=float(bn_momentum), eps=float(bn_eps))
        elif self.norm == "ln":
            self.norm1 = nn.LayerNorm(self.hidden)
        else:
            self.norm1 = nn.Identity()

        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=self.dropout)
        self.fc2 = nn.Linear(self.hidden, self.K)

    def _flatten_time_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        # x expected from JointPeatDataset: per batch often (B,T,C,H,W)
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

        if self.chunk_tiles and self.chunk_tiles > 0 and x2.shape[0] > self.chunk_tiles:
            outs = []
            for i in range(0, x2.shape[0], self.chunk_tiles):
                xi = x2[i : i + self.chunk_tiles]
                hi = self.fc1(xi)
                hi = self.norm1(hi)
                hi = self.act(hi)
                hi = self.drop(hi)
                oi = self.fc2(hi)
                outs.append(oi)
            out = torch.cat(outs, dim=0)
        else:
            h1 = self.fc1(x2)
            h1 = self.norm1(h1)
            h1 = self.act(h1)
            h1 = self.drop(h1)
            out = self.fc2(h1)

        out = out.view(B, H, W, self.K).permute(0, 3, 1, 2).contiguous()
        return out


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------

def make_criterion_from_name(loss_name: str, args) -> nn.Module:
    if loss_name == "bce":
        return MaskedBCEWithLogits(
            enable_class_weights=args.enable_pixel_class_weights,
            max_pos_weight=args.max_fire_class_weight,
        )
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

        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss


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

        eps = 1e-6
        focal_term = (1.0 - p_t).pow(self.gamma)
        loss = -alpha_t * focal_term * torch.log(p_t.clamp(min=eps))

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

        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss


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
        focal_tversky = (1.0 - tversky).pow(self.gamma)
        return focal_tversky.mean()


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


def _choose_mp_context(requested: str) -> Optional[str]:
    if requested != "auto":
        return requested
    if sys.platform.startswith("linux"):
        return "forkserver"
    return "spawn"


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
        if self.method == "none" or self.method is None:
            return torch.sigmoid(logits)

        if self.method == "platt":
            if self.a is None or self.b is None:
                return torch.sigmoid(logits)
            a = self.a.view(1, -1, 1, 1)
            b = self.b.view(1, -1, 1, 1)
            return torch.sigmoid(logits * a + b)

        # isotonic
        p = torch.sigmoid(logits)
        out = torch.empty_like(p)
        for k in range(p.shape[1]):
            x = self.iso_x[k]
            y = self.iso_y[k]
            if x is None or y is None or x.numel() < 2:
                out[:, k] = p[:, k]
            else:
                out[:, k] = _torch_interp1d_monotone(p[:, k], x, y)
        return out


def _torch_interp1d_monotone(p: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    M = x.numel()
    idx = torch.bucketize(p, x)  # [0..M]
    idx0 = (idx - 1).clamp(0, M - 1)
    idx1 = idx.clamp(0, M - 1)

    x0 = x[idx0]
    x1 = x[idx1]
    y0 = y[idx0]
    y1 = y[idx1]

    denom = (x1 - x0).clamp(min=1e-12)
    t = ((p - x0) / denom).clamp(0.0, 1.0)
    out = y0 + t * (y1 - y0)
    return out.clamp(0.0, 1.0)


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


@torch.no_grad()
def fit_calibrator(model, calib_loader, device, args) -> Optional[ProbCalibrator]:
    method = getattr(args, "calibration_method", "none")
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

            steps = max(1, int(args.calib_platt_steps))
            it = iter(calib_loader)

            for _ in range(steps):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(calib_loader)
                    batch = next(it)

                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}

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

        cal.a = a.detach().to("cpu")
        cal.b = b.detach().to("cpu")
        return cal

    if method == "isotonic":
        nb = int(args.isotonic_bins)
        nb = max(16, nb)
        counts = np.zeros((K, nb), dtype=np.float64)
        true_sums = np.zeros((K, nb), dtype=np.float64)

        for batch in calib_loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            logits = model(batch)
            probs = torch.sigmoid(logits)

            targets = (batch["y"] > 0.5)
            valid = (batch["mask"] > 0.5)

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
                cal.iso_x[k] = None
                cal.iso_y[k] = None
                continue

            x = bin_centers[nz]
            y_mean = (true_sums[k][nz] / np.maximum(counts[k][nz], 1e-12))
            w = counts[k][nz]

            y_iso = _pav_weighted(y_mean, w)

            cal.iso_x[k] = torch.tensor(x, dtype=torch.float32).cpu()
            cal.iso_y[k] = torch.tensor(y_iso, dtype=torch.float32).cpu()

        return cal

    return None


# ----------------------------------------------------------------------
# Evaluation + metrics + threshold counting
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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": support,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    criterion,
    args,
    calibrator: Optional[ProbCalibrator] = None,
    use_tqdm: bool = True,
):
    model.eval()

    # IMPORTANT FIX:
    # args.prob_max_pct is in PERCENT units (0.06 means 0.06%),
    # so convert to probability by dividing by 100.
    prob_thresh = float(args.prob_max_pct) / 100.0  # 0.0006 if prob_max_pct=0.06

    tot_loss, tot_mask = 0.0, 0.0
    tp_total = fp_total = fn_total = tn_total = 0

    num_horizons = len(args.horizons)
    tp_h = [0 for _ in range(num_horizons)]
    fp_h = [0 for _ in range(num_horizons)]
    fn_h = [0 for _ in range(num_horizons)]
    tn_h = [0 for _ in range(num_horizons)]

    # Counts of probs > threshold (post-calibration)
    gt_count_h = np.zeros(num_horizons, dtype=np.float64)
    total_prob_h = np.zeros(num_horizons, dtype=np.float64)

    iterator = wrap_loader(loader, desc="eval", use_tqdm=use_tqdm, position=3)
    if calibrator is not None:
        calibrator = calibrator.to(device)

    for batch in iterator:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(batch)
        loss = criterion(logits, batch["y"], batch["mask"])

        msum = batch["mask"].sum().item()
        if msum == 0:
            continue
        tot_loss += loss.item() * msum
        tot_mask += msum

        probs = calibrator.apply_logits(logits) if calibrator is not None else torch.sigmoid(logits)

        targets = batch["y"] > 0.5
        valid = batch["mask"] > 0.5

        preds = probs >= args.metrics_threshold

        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)
        valid_flat = valid.reshape(-1)

        tp_total += (preds_flat & targets_flat & valid_flat).sum().item()
        fp_total += (preds_flat & ~targets_flat & valid_flat).sum().item()
        fn_total += (~preds_flat & targets_flat & valid_flat).sum().item()
        tn_total += (~preds_flat & ~targets_flat & valid_flat).sum().item()

        B, K, H, W = logits.shape
        K_eff = min(K, num_horizons)

        for k_idx in range(K_eff):
            p_k = preds[:, k_idx]
            t_k = targets[:, k_idx]
            v_k = valid[:, k_idx]

            p_flat = p_k.reshape(-1)
            t_flat = t_k.reshape(-1)
            v_flat = v_k.reshape(-1)

            tp_h[k_idx] += (p_flat & t_flat & v_flat).sum().item()
            fp_h[k_idx] += (p_flat & ~t_flat & v_flat).sum().item()
            fn_h[k_idx] += (~p_flat & t_flat & v_flat).sum().item()
            tn_h[k_idx] += (~p_flat & ~t_flat & v_flat).sum().item()

        # --- probability threshold counting (post-calibration) ---
        for k_idx in range(K_eff):
            v = valid[:, k_idx].reshape(-1)
            if v.sum().item() == 0:
                continue
            p = probs[:, k_idx].reshape(-1)[v].detach().cpu().numpy().astype(np.float64)

            total_prob_h[k_idx] += float(p.size)
            gt_count_h[k_idx] += float((p > prob_thresh).sum())

    val_loss = float("nan") if tot_mask == 0 else (tot_loss / tot_mask)

    overall = _compute_basic_metrics(tp_total, fp_total, fn_total, tn_total)

    per_horizon_metrics: Dict[int, Dict[str, float]] = {}
    for idx, h in enumerate(args.horizons):
        per_horizon_metrics[int(h)] = _compute_basic_metrics(tp_h[idx], fp_h[idx], fn_h[idx], tn_h[idx])

    # Package threshold stats
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
# Argument parsing
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # --- Per-tile MLP options ---
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--mlp-chunk-tiles", type=int, default=0,
                   help="If >0, process tiles in chunks of this many to reduce peak memory.")

    # FIX: these were referenced but missing
    p.add_argument("--mlp-norm", choices=["bn", "ln", "none"], default="bn")
    p.add_argument("--mlp-bn-momentum", type=float, default=0.10)
    p.add_argument("--mlp-bn-eps", type=float, default=1e-5)

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

    # --- Training HPs ---
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # --- Loss / imbalance handling ---
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

    # --- Metrics ---
    p.add_argument("--metrics-threshold", type=float, default=0.5)

    # --- Post-hoc calibration ---
    p.add_argument("--calibration-method", choices=["none", "platt", "isotonic"], default="none")
    p.add_argument("--calib-frac", type=float, default=0.0)
    p.add_argument("--calib-platt-steps", type=int, default=200)
    p.add_argument("--calib-platt-lr", type=float, default=0.05)
    p.add_argument("--isotonic-bins", type=int, default=400)

    # --- Optimization flags ---
    p.add_argument("--compile", action="store_true")
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--mp-context", choices=["auto", "spawn", "fork", "forkserver"], default="auto")
    p.add_argument("--no-tqdm", action="store_true")

    # TensorBoard logging
    p.add_argument("--logdir", default="runs/peat_mlp")
    p.add_argument("--no-tensorboard", action="store_true")

    # IMPORTANT FIX:
    # This is a PERCENT input. 0.06 means 0.06% => 0.0006 in probability units.
    p.add_argument(
        "--prob-max-pct",
        type=float,
        default=0.06,
        help="Probability threshold expressed in percent. 0.06 means 0.06% => prob=0.0006. "
             "We count probabilities > (prob_max_pct/100).",
    )

    return p.parse_args()


# ----------------------------------------------------------------------
# Dataset helpers
# ----------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    val_ds = None
    test_ds = None
    val_ds = JointPeatDataset(mode="val", **common)
    print_diagnostic_item("Val patches", len(val_ds), indent=1)

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
# CSV logging
# ----------------------------------------------------------------------

def init_metrics_csv(logdir: str) -> str:
    os.makedirs(logdir, exist_ok=True)
    path = os.path.join(logdir, "metrics_log.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "split", "loss", "acc", "prec", "rec", "f1", "calib", "prob_thresh_pct",
                        "prob_thresh_prob", "gt_count", "gt_total", "gt_frac"])
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
# Training loop
# ----------------------------------------------------------------------

def train_one_epoch(model, train_ds, epoch, optimizer, criterion, device, args, monitor, writer, global_step):
    model.train()
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, args=args)
    iterator = wrap_loader(train_loader, desc=f"train[e{epoch}]", use_tqdm=not args.no_tqdm, position=2)

    optimizer.zero_grad(set_to_none=True)
    total_loss, total_mask = 0.0, 0.0

    for step, batch in enumerate(iterator):
        if monitor is not None:
            monitor.set_status(f"Train e{epoch}/{args.epochs} step {step+1}")

        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        with amp_autocast(device):
            logits = model(batch)
            loss_raw = criterion(logits, batch["y"], batch["mask"])
            loss = loss_raw / max(1, args.grad_accum)

        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        m = batch["mask"].sum().item()
        total_loss += loss_raw.item() * m
        total_mask += m

        if writer is not None:
            writer.add_scalar("train/loss_step", loss_raw.item(), global_step)
        global_step += 1

    avg_loss = float("nan") if total_mask == 0 else (total_loss / total_mask)
    return avg_loss, global_step


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_diagnostic_header("Threshold sanity check")
    prob_thresh = float(args.prob_max_pct) / 100.0
    print_diagnostic_item("--prob-max-pct", f"{args.prob_max_pct}%", indent=1)
    print_diagnostic_item("Converted probability threshold", f"{prob_thresh:.8f}", indent=1)
    print("NOTE: 0.06% => 0.0006. 6% would be 0.06.")

    monitor = RAMMonitor(device_id=device.index if device.type == "cuda" else 0)
    if not args.no_tqdm:
        monitor.start()

    start_ram = get_ram_usage()

    with DiagnosticTimer("Dataset init", track_ram=True):
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
        dropout=0.10,
        chunk_tiles=args.mlp_chunk_tiles,
        norm=args.mlp_norm,
        bn_momentum=args.mlp_bn_momentum,
        bn_eps=args.mlp_bn_eps,
    ).to(device)

    print_diagnostic_header("Model")
    print_diagnostic_item("in_features", in_channels, indent=1)
    print_diagnostic_item("horizons", args.horizons, indent=1)
    tp, tr = count_params(model)
    print_diagnostic_item("Total params", human_int(tp), indent=1)
    print_diagnostic_item("Trainable params", human_int(tr), indent=1)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    train_loss_name = args.train_loss or args.loss
    eval_loss_name = args.eval_loss or args.loss

    train_criterion = make_criterion_from_name(train_loss_name, args).to(device)
    eval_criterion = make_criterion_from_name(eval_loss_name, args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    writer: Optional[SummaryWriter] = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=args.logdir)
        print(f"[TB] Logging to: {args.logdir}")

    metrics_csv_path = init_metrics_csv(args.logdir)

    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False, args=args) if val_ds is not None else None
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False, args=args) if test_ds is not None else None
    calib_loader = make_loader(calib_ds, batch_size=args.batch_size, shuffle=False, args=args) if calib_ds is not None else None

    print_ram_delta(start_ram, "After init")

    best_val_f1 = -1.0
    best_epoch = -1
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if monitor is not None:
            monitor.set_status(f"Train e{epoch}/{args.epochs}")
        train_loss, global_step = train_one_epoch(
            model, train_ds, epoch, optimizer, train_criterion, device, args, monitor, writer, global_step
        )

        calibrator = None
        calib_method = "none"
        if calib_loader is not None and args.calibration_method != "none":
            calib_method = args.calibration_method
            if monitor is not None:
                monitor.set_status(f"Calibrating ({calib_method}) e{epoch}/{args.epochs}")
            calibrator = fit_calibrator(model, calib_loader, device, args)

        if val_loader is not None:
            if monitor is not None:
                monitor.set_status(f"Eval(val) e{epoch}/{args.epochs}")
            val_loss, val_metrics = evaluate(model, val_loader, device, eval_criterion, args, calibrator=calibrator,
                                             use_tqdm=not args.no_tqdm)

            pr = val_metrics.get("prob_gt", {}) or {}
            print_diagnostic_header(f"Validation (epoch {epoch})")
            print(f"[Val] calib={calib_method} loss={val_loss:.6f} acc={val_metrics['accuracy']:.4f} "
                  f"prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}")
            print(f"[Val] P(prob > {pr['threshold_pct']}% = {pr['threshold_prob']:.8f}) : "
                  f"{pr['count']:.0f}/{pr['total']:.0f} ({100.0*pr['fraction']:.4f}%)")
            for h, hh in (pr.get("per_horizon", {}) or {}).items():
                print(f"  [Val h={h}] {hh['count']:.0f}/{hh['total']:.0f} ({100.0*hh['fraction']:.4f}%)")

            append_metrics_csv(metrics_csv_path, epoch, "val", val_loss, val_metrics, calib_method)

            if val_metrics["f1"] > best_val_f1:
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

        if epoch == args.epochs and test_loader is not None:
            if monitor is not None:
                monitor.set_status(f"Eval(test) e{epoch}/{args.epochs}")
            test_loss, test_metrics = evaluate(model, test_loader, device, eval_criterion, args, calibrator=calibrator,
                                               use_tqdm=not args.no_tqdm)
            pr = test_metrics.get("prob_gt", {}) or {}
            print_diagnostic_header(f"Test (epoch {epoch})")
            print(f"[Test] calib={calib_method} loss={test_loss:.6f} acc={test_metrics['accuracy']:.4f} "
                  f"prec={test_metrics['precision']:.4f} rec={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}")
            print(f"[Test] P(prob > {pr['threshold_pct']}% = {pr['threshold_prob']:.8f}) : "
                  f"{pr['count']:.0f}/{pr['total']:.0f} ({100.0*pr['fraction']:.4f}%)")
            for h, hh in (pr.get("per_horizon", {}) or {}).items():
                print(f"  [Test h={h}] {hh['count']:.0f}/{hh['total']:.0f} ({100.0*hh['fraction']:.4f}%)")

            append_metrics_csv(metrics_csv_path, epoch, "test", test_loss, test_metrics, calib_method)

    print_diagnostic_header("Training complete")
    if best_epoch > 0:
        print(f"[Summary] Best val F1={best_val_f1:.4f} at epoch {best_epoch}")
    else:
        print("[Summary] Best epoch undefined (no val?)")

    if writer is not None:
        writer.flush()
        writer.close()

    if monitor is not None:
        monitor.stop()


if __name__ == "__main__":
    main()
