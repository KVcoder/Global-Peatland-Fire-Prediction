#!/usr/bin/env python3
"""
Spatiotemporal Peat Ignition — Heavily Optimized for Speed & Memory
WITH COMPREHENSIVE REAL-TIME MONITORING

This version adds OPTIONAL post-hoc probability calibration with a separate
calibration split held out from the training set:

  train 1 epoch -> fit calibrator on calib split -> evaluate on val/test using calibrated probs

Calibration methods:
  - Platt scaling (logistic recalibration): sigmoid(a*logit + b), fit per-horizon
  - Isotonic regression (monotone): fit per-horizon from probability histograms (memory safe)

Enable via:
  --calib-frac 0.10 --calibration-method platt
or
  --calib-frac 0.10 --calibration-method isotonic
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
    """Return val if not None, else d."""
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


def human_bytes(n: float) -> str:
    """Convert bytes to human readable format (binary units)"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def get_ram_usage() -> float:
    """Get current RAM usage in GiB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


def get_gpu_memory(device_id: int = 0) -> float:
    """Get GPU memory allocated (GiB) as tracked by PyTorch"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device_id) / (1024 ** 3)
    return 0.0


def print_ram_delta(start_ram: float, label: str) -> float:
    """Print RAM usage delta"""
    current_ram = get_ram_usage()
    delta = current_ram - start_ram
    print(f"[RAM] {label}: {delta:+.2f} GB (total: {current_ram:.2f} GB)")
    return current_ram


def print_diagnostic_header(title: str):
    """Print a diagnostic section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_diagnostic_item(label: str, value: Any, indent: int = 0):
    """Print a diagnostic item with consistent formatting"""
    prefix = "  " * indent + "• "
    print(f"{prefix}{label:.<40} {value}")


class DiagnosticTimer:
    """Context manager for timing operations with diagnostics"""

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
    """CPU-safe AMP helper: only enable autocast on CUDA; use bf16 if supported else fp16."""
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


# NEW: storage / SSD detection
def _detect_storage_type(path: str) -> str:
    """
    Best-effort detection of storage type for a given path.

    On Linux:
      - Detects network / parallel FS (nfs, lustre, cifs, etc.).
      - For local block devices, inspects /sys/block/*/queue/rotational:
            0 -> SSD / NVMe
            1 -> spinning HDD
    """
    try:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            return "Unknown (path does not exist yet)"

        if os.name != "posix":
            return "Unknown (non-POSIX OS)"

        device = None
        fstype = None
        best_len = -1
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                dev, mnt, fs = parts[0], parts[1], parts[2]
                if path.startswith(mnt) and len(mnt) > best_len:
                    best_len = len(mnt)
                    device, _, fstype = dev, mnt, fs

        if device is None:
            return "Unknown (no /proc/mounts entry matched)"

        if fstype in ("nfs", "nfs4", "lustre", "cifs", "smb3", "gpfs"):
            return f"Network / parallel FS ({fstype}, device={device})"

        if device.startswith("/dev/"):
            base = os.path.basename(device)
            while base and base[-1].isdigit():
                base = base[:-1]
            base = base.rstrip("p")
            rotational_path = f"/sys/block/{base}/queue/rotational"
            try:
                with open(rotational_path, "r") as rf:
                    val = rf.read().strip()
                if val == "0":
                    return f"Local SSD / NVMe ({device}, fstype={fstype})"
                elif val == "1":
                    return f"Local spinning HDD ({device}, fstype={fstype})"
            except FileNotFoundError:
                pass

        return f"Local filesystem ({fstype}, device={device})"

    except Exception as e:  # noqa: BLE001
        return f"Unknown (error probing storage: {e})"


# ----------------------------------------------------------------------
# UNet baseline with time-as-channels
# ----------------------------------------------------------------------


class ConvBNDrop(nn.Module):
    """
    Conv2d -> BatchNorm2d -> ReLU -> Dropout2d
    repeated twice (a lighter 'DoubleConv' with dropout).
    """
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_drop),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetLiteTimeChannels(nn.Module):
    """
    Smaller UNet (2 down, bottleneck, 2 up) + BatchNorm + Dropout2d.
    Much fewer params than your original 64/128/256/512 stack.
    """

    def __init__(
        self,
        in_channels: int,
        horizons: Sequence[int],
        base_ch: int = 32,
        dropout: float = 0.10,
        t_hist: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.horizons = [int(h) for h in horizons]
        self.k = len(self.horizons)
        self.base_ch = int(base_ch)
        self.dropout = float(dropout)
        self.t_hist = t_hist

        c1 = self.base_ch
        c2 = self.base_ch * 2
        c3 = self.base_ch * 4  # bottleneck width

        # Encoder
        self.enc1 = ConvBNDrop(self.in_channels, c1, p_drop=self.dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBNDrop(c1, c2, p_drop=self.dropout)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottom = ConvBNDrop(c2, c3, p_drop=self.dropout)

        # Decoder
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBNDrop(c2 + c2, c2, p_drop=self.dropout)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBNDrop(c1 + c1, c1, p_drop=self.dropout)

        # Head: per-pixel logits for K horizons
        self.head = nn.Conv2d(c1, self.k, kernel_size=1)

    def _flatten_time_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.view(B, T * C, H, W)
        raise ValueError(f"Expected x with 4 or 5 dims, got {tuple(x.shape)}")

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x = self._flatten_time_if_needed(batch["x"])

        x1 = self.enc1(x)             # (B, c1, H, W)
        x2 = self.enc2(self.pool1(x1))# (B, c2, H/2, W/2)
        x3 = self.bottom(self.pool2(x2))  # (B, c3, H/4, W/4)

        u2 = self.up2(x3)             # (B, c2, H/2, W/2)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.dec2(u2)            # (B, c2, H/2, W/2)

        u1 = self.up1(u2)             # (B, c1, H, W)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.dec1(u1)            # (B, c1, H, W)

        return self.head(u1)          # (B, K, H, W)



# ----------------------------------------------------------------------
# Reliability bias saving/logging helpers
# ----------------------------------------------------------------------

def save_reliability_bias_npz(calib: Dict[str, Any], out_path: str):
    """
    Save overall + per-horizon reliability-bias curves into a single .npz.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data = {}
    data["bin_centers"] = np.asarray(calib.get("bin_centers_slice", []), dtype=np.float64)
    data["bias_pct"] = np.asarray(calib.get("reliability_bias_pct", []), dtype=np.float64)
    data["count"] = np.asarray(calib.get("reliability_bias_count", []), dtype=np.float64)

    per_h = calib.get("per_horizon", {}) or {}
    for h, hc in per_h.items():
        h = int(h)
        data[f"h{h}_bin_centers"] = np.asarray(
            hc.get("bin_centers_slice", calib.get("bin_centers_slice", [])),
            dtype=np.float64,
        )
        data[f"h{h}_bias_pct"] = np.asarray(hc.get("reliability_bias_pct", []), dtype=np.float64)
        data[f"h{h}_count"] = np.asarray(hc.get("reliability_bias_count", []), dtype=np.float64)

    np.savez_compressed(out_path, **data)


def tb_log_reliability_bias_curve(
    writer: SummaryWriter,
    calib: Dict[str, Any],
    epoch: int,
    tag: str = "val/reliability_bias_pct",
):
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
    """
    Save reliability-bias curves to .npz and log an overall curve figure to TensorBoard.
    split: "val" or "test"
    """
    calib = metrics.get("calibration", {}) or {}

    out_npz = os.path.join(args.logdir, f"reliability_bias_{split}_epoch{epoch}.npz")
    try:
        save_reliability_bias_npz(calib, out_npz)
        if args.debug:
            print(f"[ReliabilityBias] Saved: {out_npz}")
    except Exception as e:  # noqa: BLE001
        print(f"[ReliabilityBias] Failed to save NPZ ({split}): {e}")

    if writer is not None:
        try:
            tb_log_reliability_bias_curve(writer, calib, epoch, tag=f"{split}/reliability_bias_pct")
        except Exception as e:  # noqa: BLE001
            print(f"[ReliabilityBias] Failed to log TB figure ({split}): {e}")

        per_h_calib = calib.get("per_horizon", {}) or {}
        for h, hc in per_h_calib.items():
            if isinstance(hc, dict) and ("reliability_bias_pct" in hc):
                try:
                    tb_log_reliability_bias_curve(
                        writer,
                        hc,
                        epoch,
                        tag=f"{split}/h{int(h)}_reliability_bias_pct",
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[ReliabilityBias] Failed per-horizon TB ({split}, h={h}): {e}")


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
        # NOTE: pixel class weights ignored by your implementation (as you already note)
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
        loss = focal_tversky.mean()
        return loss


# ----------------------------------------------------------------------
# Monitoring
# ----------------------------------------------------------------------

class RAMMonitor:
    """Background thread that monitors RAM and GPU usage"""

    def __init__(self, device_id=0, update_interval=0.5):
        self.device_id = device_id
        self.update_interval = update_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.pbar = None
        self.current_ram = 0.0
        self.current_gpu = 0.0
        self.current_status = "Initializing..."
        self.queue_capacity: Optional[int] = None
        self.batches_ready: Optional[int] = None
        self.batches_loading: Optional[int] = None

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

    def set_loader_queue(self, ready: Optional[int], loading: Optional[int], capacity: Optional[int]):
        self.batches_ready = ready
        self.batches_loading = loading
        self.queue_capacity = capacity

    def _monitor_loop(self):
        while self.running:
            self.current_ram = get_ram_usage()
            self.current_gpu = get_gpu_memory(self.device_id) if torch.cuda.is_available() else 0.0

            desc = f"💾 RAM: {self.current_ram:.2f}GB"
            if torch.cuda.is_available():
                desc += f" | 🎮 GPU: {self.current_gpu:.2f}GB"

            if self.queue_capacity:
                if self.batches_ready is not None and self.batches_loading is not None:
                    desc += (
                        f" | 📦 Queue: {self.batches_ready}/{self.queue_capacity} ready, "
                        f"{self.batches_loading} loading"
                    )
                else:
                    desc += f" | 📦 Queue: capacity {self.queue_capacity}"

            desc += f" | {self.current_status}"

            if self.pbar:
                self.pbar.set_description_str(desc)

            time.sleep(self.update_interval)


class TimedDataLoader:
    """Wrapper that tracks batch loading times with detailed metrics"""

    def __init__(
        self,
        loader: DataLoader,
        desc: str = "batch",
        use_tqdm: bool = True,
        show_worker_stats: bool = False,
        position: int = 1,
    ):
        self.loader = loader
        self.desc = desc
        self.use_tqdm = use_tqdm
        self.show_worker_stats = show_worker_stats
        self.position = position
        self.batch_times: deque = deque(maxlen=100)
        self.last_batch_time = 0.0
        self.pbar = None

    def __iter__(self):
        iterator = iter(self.loader)
        if self.use_tqdm:
            try:
                total = len(self.loader)
            except TypeError:
                total = None
            self.pbar = tqdm(
                total=total,
                desc=self.desc,
                leave=True,
                position=self.position,
                bar_format=(
                    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                ),
            )

        while True:
            try:
                t_start = time.time()
                batch = next(iterator)
                t_elapsed = time.time() - t_start
                self.batch_times.append(t_elapsed)
                self.last_batch_time = t_elapsed

                if self.use_tqdm and self.pbar:
                    avg_time = sum(self.batch_times) / len(self.batch_times)
                    postfix = {
                        "⏱️load": f"{t_elapsed*1000:.0f}ms",
                        "avg": f"{avg_time*1000:.0f}ms",
                    }
                    if self.show_worker_stats and len(self.batch_times) >= 10:
                        recent = list(self.batch_times)[-10:]
                        postfix["min"] = f"{min(recent)*1000:.0f}ms"
                        postfix["max"] = f"{max(recent)*1000:.0f}ms"
                    self.pbar.set_postfix(postfix)
                    self.pbar.update(1)

                yield batch

            except StopIteration:
                break

        if self.use_tqdm and self.pbar:
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
    """
    Optimized collate with key integrity check.
    For JointPeatDataset:
        - Tensors ("x", "y", "mask", optional "coords") are stacked.
        - "meta" is kept as a list of dicts.
    """
    if not batch:
        raise RuntimeError("Received empty batch in collate; check dataset filtering and batch_size.")
    keys = batch[0].keys()
    for k in keys:
        if any(k not in b for b in batch):
            raise KeyError(f"Missing key '{k}' in batch elements")

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
            if getattr(args, "verbose_loader", False):
                print(
                    f"[DataLoader] Using multiprocessing context '{ctx}' "
                    f"with {args.workers} workers and prefetch_factor={args.prefetch}."
                )
    return DataLoader(ds, **kw)


def wrap_loader(loader, desc: str, use_tqdm: bool, show_worker_stats: bool = False, position: int = 1):
    return TimedDataLoader(loader, desc=desc, use_tqdm=use_tqdm, show_worker_stats=show_worker_stats, position=position)


# ----------------------------------------------------------------------
# Post-hoc probability calibration (Platt / Isotonic)
# ----------------------------------------------------------------------

class ProbCalibrator:
    """
    Post-hoc probability calibrator applied per-horizon.

    Methods:
      - platt: p = sigmoid(a*logit + b)
      - isotonic: p = iso(sigmoid(logit)) using monotone piecewise-linear interpolation
    """

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
            z = logits * a + b
            return torch.sigmoid(z)

        # isotonic
        p = torch.sigmoid(logits)
        out = torch.empty_like(p)
        for k in range(p.shape[1]):
            x = self.iso_x[k]
            y = self.iso_y[k]
            if x is None or y is None or x.numel() < 2:
                out[:, k] = p[:, k]
                continue
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

    means = []
    weights = []
    sizes = []

    for yi, wi in zip(y, w):
        means.append(float(yi))
        weights.append(float(wi))
        sizes.append(1)

        while len(means) >= 2 and means[-2] > means[-1]:
            w_new = weights[-2] + weights[-1]
            if w_new <= 0:
                m_new = 0.5 * (means[-2] + means[-1])
            else:
                m_new = (means[-2] * weights[-2] + means[-1] * weights[-1]) / w_new
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
        # Re-enable autograd for a,b optimization even though the function is @no_grad
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

                # IMPORTANT: don't build a graph through the model
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

    # isotonic branch unchanged...



# ----------------------------------------------------------------------
# Evaluation, metrics, dataset scan
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
    if (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
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


@torch.no_grad()
def evaluate(model, loader, device, criterion, args, calibrator: Optional[ProbCalibrator] = None,
             use_tqdm: bool = True, show_worker_stats: bool = False):
    model.eval()
    tot_loss, tot_mask = 0.0, 0.0
    tp_total = fp_total = fn_total = tn_total = 0

    num_horizons = len(args.horizons)
    tp_h = [0 for _ in range(num_horizons)]
    fp_h = [0 for _ in range(num_horizons)]
    fn_h = [0 for _ in range(num_horizons)]
    tn_h = [0 for _ in range(num_horizons)]

    bin_width = getattr(args, "reliability_bin_width", 0.005)
    num_bins = max(1, int(math.ceil(1.0 / bin_width)))

    rel_counts = np.zeros((num_horizons, num_bins), dtype=np.float64)
    rel_pred_sums = np.zeros_like(rel_counts)
    rel_true_sums = np.zeros_like(rel_counts)
    brier_sums = np.zeros(num_horizons, dtype=np.float64)
    total_counts_per_h = np.zeros(num_horizons, dtype=np.float64)

    logloss_sum_overall = 0.0
    logloss_count_overall = 0.0
    logloss_sum_h = np.zeros(num_horizons, dtype=np.float64)
    logloss_count_h = np.zeros(num_horizons, dtype=np.float64)

    iterator = wrap_loader(
        loader,
        desc="eval",
        use_tqdm=use_tqdm,
        show_worker_stats=show_worker_stats,
        position=3,
    )
    if calibrator is not None:
        calibrator = calibrator.to(device)

    for batch in iterator:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(batch)
        loss = criterion(logits, batch["y"], batch["mask"])
        m = batch["mask"].sum().item()
        if m == 0:
            continue
        tot_loss += loss.item() * m
        tot_mask += m

        if calibrator is not None:
            probs = calibrator.apply_logits(logits)
        else:
            probs = torch.sigmoid(logits)

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

        for k_idx in range(K_eff):
            p_k = probs[:, k_idx]
            t_k = targets[:, k_idx]
            v_k = valid[:, k_idx]

            mask_flat = v_k.reshape(-1)
            if mask_flat.sum().item() == 0:
                continue

            p_flat_t = p_k.reshape(-1)[mask_flat].detach().cpu().numpy().astype(np.float32)
            y_flat_t = t_k.reshape(-1)[mask_flat].float().detach().cpu().numpy().astype(np.float32)

            n_valid = float(p_flat_t.size)
            if n_valid == 0:
                continue

            total_counts_per_h[k_idx] += n_valid
            brier_sums[k_idx] += float(((p_flat_t - y_flat_t) ** 2).sum())

            eps = 1e-7
            p_clipped = np.clip(p_flat_t, eps, 1.0 - eps)
            logloss_vals = -(
                y_flat_t * np.log(p_clipped)
                + (1.0 - y_flat_t) * np.log(1.0 - p_clipped)
            )
            s_ll = float(logloss_vals.sum())
            logloss_sum_overall += s_ll
            logloss_count_overall += n_valid
            logloss_sum_h[k_idx] += s_ll
            logloss_count_h[k_idx] += n_valid

            bin_idx = np.floor(p_flat_t / bin_width).astype(np.int64)
            bin_idx = np.clip(bin_idx, 0, num_bins - 1)

            counts_batch = np.bincount(bin_idx, minlength=num_bins).astype(np.float64)
            pred_sums_batch = np.bincount(bin_idx, weights=p_flat_t, minlength=num_bins).astype(np.float64)
            true_sums_batch = np.bincount(bin_idx, weights=y_flat_t, minlength=num_bins).astype(np.float64)

            rel_counts[k_idx] += counts_batch
            rel_pred_sums[k_idx] += pred_sums_batch
            rel_true_sums[k_idx] += true_sums_batch

    val_loss = float("nan") if tot_mask == 0 else (tot_loss / tot_mask)

    overall = _compute_basic_metrics(tp_total, fp_total, fn_total, tn_total)
    logloss_overall = float("nan") if logloss_count_overall <= 0 else float(logloss_sum_overall / max(logloss_count_overall, 1.0))

    per_horizon_metrics: Dict[int, Dict[str, float]] = {}
    for idx, h in enumerate(args.horizons):
        m = _compute_basic_metrics(tp_h[idx], fp_h[idx], fn_h[idx], tn_h[idx])
        m["logloss"] = float("nan") if logloss_count_h[idx] <= 0 else float(logloss_sum_h[idx] / max(logloss_count_h[idx], 1.0))
        per_horizon_metrics[h] = m

    bin_centers = (np.arange(num_bins, dtype=np.float64) + 0.5) * bin_width
    calibration_per_horizon: Dict[int, Dict[str, Any]] = {}

    for idx, h in enumerate(args.horizons):
        counts = rel_counts[idx]
        true_sums = rel_true_sums[idx]

        if counts.sum() == 0:
            calibration_per_horizon[h] = {
                "bin_pred": [],
                "bin_true": [],
                "bin_count": [],
                "ece": float("nan"),
                "mce": float("nan"),
                "brier": float("nan"),
                "roc": {"fpr": [], "tpr": [], "auc": float("nan")},
            }
            per_horizon_metrics[h]["ece"] = float("nan")
            per_horizon_metrics[h]["brier"] = float("nan")
            per_horizon_metrics[h]["roc_auc"] = float("nan")
            continue

        nonzero = counts > 0
        pred_mean = np.zeros_like(counts)
        true_mean = np.zeros_like(counts)
        pred_mean[nonzero] = rel_pred_sums[idx][nonzero] / counts[nonzero]
        true_mean[nonzero] = true_sums[nonzero] / counts[nonzero]

        b0 = int(math.floor(args.reliability_bin_min / bin_width))
        b1 = int(math.ceil(args.reliability_bin_max / bin_width))
        b0 = max(0, min(b0, num_bins))
        b1 = max(0, min(b1, num_bins))

        bias_pct = (pred_mean - true_mean) * 100.0
        bias_pct_slice = bias_pct[b0:b1].copy()
        count_slice = counts[b0:b1].copy()

        minc = int(getattr(args, "reliability_min_count", 0))
        if minc > 0:
            bias_pct_slice[count_slice < minc] = np.nan

        total = counts.sum()
        gap = np.abs(pred_mean - true_mean)
        ece = float((counts[nonzero] / total * gap[nonzero]).sum())
        mce = float(gap[nonzero].max()) if np.any(nonzero) else float("nan")
        brier = float(brier_sums[idx] / max(total_counts_per_h[idx], 1.0))

        fpr_h, tpr_h, auc_h = _compute_roc_from_hist(counts, true_sums)

        calibration_per_horizon[h] = {
            "bin_pred": pred_mean.tolist(),
            "bin_true": true_mean.tolist(),
            "bin_count": counts.tolist(),
            "ece": ece,
            "mce": mce,
            "brier": brier,
            "roc": {"fpr": fpr_h.tolist(), "tpr": tpr_h.tolist(), "auc": auc_h},
            "reliability_bias_pct": bias_pct_slice.tolist(),
            "reliability_bias_count": count_slice.tolist(),
            "bin_centers_slice": bin_centers[b0:b1].tolist(),
        }

        per_horizon_metrics[h]["ece"] = ece
        per_horizon_metrics[h]["brier"] = brier
        per_horizon_metrics[h]["roc_auc"] = auc_h

    overall_counts = rel_counts.sum(axis=0)
    if overall_counts.sum() == 0:
        overall_pred_mean = np.zeros_like(overall_counts)
        overall_true_mean = np.zeros_like(overall_counts)
        ece_overall = float("nan")
        mce_overall = float("nan")
        brier_overall = float("nan")
        fpr_overall = np.array([], dtype=np.float64)
        tpr_overall = np.array([], dtype=np.float64)
        auc_overall = float("nan")
    else:
        nonzero = overall_counts > 0
        overall_pred_sum = rel_pred_sums.sum(axis=0)
        overall_true_sum = rel_true_sums.sum(axis=0)

        overall_pred_mean = np.zeros_like(overall_counts)
        overall_true_mean = np.zeros_like(overall_counts)
        overall_pred_mean[nonzero] = overall_pred_sum[nonzero] / overall_counts[nonzero]
        overall_true_mean[nonzero] = overall_true_sum[nonzero] / overall_counts[nonzero]

        total_overall = overall_counts.sum()
        gap = np.abs(overall_pred_mean - overall_true_mean)
        ece_overall = float((overall_counts[nonzero] / total_overall * gap[nonzero]).sum())
        mce_overall = float(gap[nonzero].max()) if np.any(nonzero) else float("nan")
        total_overall_pixels = float(total_counts_per_h.sum())
        brier_overall = float(brier_sums.sum() / max(total_overall_pixels, 1.0))

        fpr_overall, tpr_overall, auc_overall = _compute_roc_from_hist(
            overall_counts, overall_true_sum
        )

    b0 = int(math.floor(args.reliability_bin_min / bin_width))
    b1 = int(math.ceil(args.reliability_bin_max / bin_width))
    b0 = max(0, min(b0, num_bins))
    b1 = max(0, min(b1, num_bins))

    overall_bias_pct = (overall_pred_mean - overall_true_mean) * 100.0
    overall_bias_pct_slice = overall_bias_pct[b0:b1].copy()
    overall_count_slice = overall_counts[b0:b1].copy()

    minc = int(getattr(args, "reliability_min_count", 0))
    if minc > 0:
        overall_bias_pct_slice[overall_count_slice < minc] = np.nan

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
        "roc_auc": auc_overall,
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
            "roc": {"fpr": fpr_overall.tolist(), "tpr": tpr_overall.tolist(), "auc": auc_overall},
            "per_horizon": calibration_per_horizon,
            "reliability_bias_pct": overall_bias_pct_slice.tolist(),
            "reliability_bias_count": overall_count_slice.tolist(),
            "bin_centers_slice": bin_centers[b0:b1].tolist(),
        },
    }

    return val_loss, metrics


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # --- Zarr data roots ---
    p.add_argument("--era5-zarr", required=True)
    p.add_argument("--smap-zarr", required=True)
    p.add_argument("--smap_l4-zarr", required=True, help="New multi-band Zarr store")
    p.add_argument("--viirs-zarr", required=True)

    p.add_argument("--era5-array", default="field")
    p.add_argument("--smap-array", default="field")
    p.add_argument("--smap_l4-array", default="field", help="Array name inside aux-zarr")
    p.add_argument("--viirs-array", default="field")

    # --- Dataset hyperparameters ---
    p.add_argument("--T-hist", type=int, default=30)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 7, 14])
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--stack-time", choices=["separate", "channel"], default="channel")
    p.add_argument("--split", type=float, default=0.9,
                   help="Train fraction. If --val-frac is None, val = 1 - split (2-way split).")
    p.add_argument("--val-frac", type=float, default=None,
                   help="Optional validation fraction for 3-way split.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--normalize-inputs", choices=["none", "per_channel"], default="none")
    p.add_argument("--no-skip-nonpeat", action="store_true",
                   help="Disable peat-based patch filtering (i.e., keep non-peat patches).")
    p.add_argument("--peat-min-fraction", type=float, default=0.01)

    # --- Training HPs ---
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
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
    # --- Loss selection (train vs eval) ---
    p.add_argument("--train-loss", choices=["bce", "focal", "focal_tversky"], default=None,
                help="Loss used for training/backprop. Defaults to --loss if not set.")
    p.add_argument("--eval-loss", choices=["bce", "focal", "focal_tversky"], default="bce",
                help="Loss used only for val/test loss reporting (metrics use probs regardless).")

    
    # --- Patch-level rebalancing & curriculum ---
    p.add_argument("--patch-sampling", choices=["none", "balanced"], default="none")
    p.add_argument("--patch-stats-path", default="patch_fire_flags_train.npy")
    p.add_argument("--patch-stats-batch-size", type=int, default=256)
    p.add_argument("--max-patch-pos-oversample", type=float, default=10.0)

    p.add_argument("--curriculum-epochs", type=int, default=0)
    p.add_argument("--curriculum-start-epoch", type=int, default=1)
    p.add_argument("--curriculum-neg-weight-min", type=float, default=1e-3)
    p.add_argument("--curriculum-neg-weight-max", type=float, default=1.0)

    # --- Metrics ---
    p.add_argument("--metrics-threshold", type=float, default=0.5)

    p.add_argument("--reliability-bin-width", type=float, default=0.005,
                   help="Bin width in probability space (e.g. 0.005 = 0.5%).")
    p.add_argument("--reliability-bin-min", type=float, default=0.005)
    p.add_argument("--reliability-bin-max", type=float, default=0.060)
    p.add_argument("--reliability-min-count", type=int, default=50)

    # --- NEW: Post-hoc calibration ---
    p.add_argument("--calibration-method", choices=["none", "platt", "isotonic"], default="none")
    p.add_argument("--calib-frac", type=float, default=0.0,
                   help="Fraction of TRAIN patches to hold out as calibration set.")
    p.add_argument("--calib-platt-steps", type=int, default=200)
    p.add_argument("--calib-platt-lr", type=float, default=0.05)
    p.add_argument("--isotonic-bins", type=int, default=400,
                   help="Histogram bins used to fit isotonic regression (memory-safe).")

    # --- Optimization flags ---
    p.add_argument("--compile", action="store_true")
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--mp-context", choices=["auto", "spawn", "fork", "forkserver"], default="auto")
    p.add_argument("--show-worker-stats", action="store_true")

    # Verbosity and diagnostics
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument("--log-interval", type=int, default=0)
    p.add_argument("--tqdm-scan", action="store_true")
    p.add_argument("--scan-limit", type=int, default=0)
    p.add_argument("--scan-batch-size", type=int, default=32)
    p.add_argument("--scan-to-device", action="store_true")

    p.add_argument("--debug", action="store_true")
    p.add_argument("--verbose-dataset", action="store_true")
    p.add_argument("--verbose-loader", action="store_true")
    p.add_argument("--verbose-model", action="store_true")
    p.add_argument("--profile-first-epoch", action="store_true")

    p.add_argument("--skip-val-dataset", action="store_true")
    p.add_argument("--quick-test", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sample-one-batch", action="store_true")

    p.add_argument("--limit-train-samples", type=int, default=0)
    p.add_argument("--val-test-only", action="store_true")

    p.add_argument("--pause-after-dataset", action="store_true")
    p.add_argument("--pause-after-model", action="store_true")

    p.add_argument("--sync-every-step", action="store_true")
    p.add_argument("--measure-loader-time", action="store_true")

    # TensorBoard logging
    p.add_argument("--logdir", default="runs/peat_unet")
    p.add_argument("--no-tensorboard", action="store_true")

    # Matplotlib metric plotting
    p.add_argument("--plot-metrics", action="store_true")
    p.add_argument("--plot-file-format", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--plot-dpi", type=int, default=150)

    # --- Model simplification / regularization ---
    p.add_argument("--base-ch", type=int, default=16, help="Base channel width for the UNet")
    p.add_argument("--dropout", type=float, default=0.20, help="Dropout2d probability inside conv blocks")

    
    return p.parse_args()


# ----------------------------------------------------------------------
# Dataset & sampling helpers
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

    if not args.skip_val_dataset:
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


def describe_environment(args, device):
    print_diagnostic_header("Environment")
    print_diagnostic_item("PyTorch", torch.__version__, indent=1)
    print_diagnostic_item("Device", str(device), indent=1)
    if device.type == "cuda":
        print_diagnostic_item("CUDA", torch.version.cuda, indent=1)
        try:
            name = torch.cuda.get_device_name(device)
        except Exception:  # noqa: BLE001
            name = "Unknown"
        print_diagnostic_item("GPU name", name, indent=1)

    print_diagnostic_header("Storage")
    print_diagnostic_item("ERA5", _detect_storage_type(args.era5_zarr), indent=1)
    print_diagnostic_item("SMAP_wtd", _detect_storage_type(args.smap_zarr), indent=1)
    print_diagnostic_item("SMAP_L4", _detect_storage_type(args.smap_l4_zarr), indent=1)
    print_diagnostic_item("VIIRS", _detect_storage_type(args.viirs_zarr), indent=1)


def compute_patch_fire_flags(train_ds, args, device) -> np.ndarray:
    path = args.patch_stats_path
    if os.path.exists(path):
        print_diagnostic_header("Patch fire flags")
        print(f"[PatchStats] Loading existing patch flags from {path}")
        flags = np.load(path)
        if flags.shape[0] != len(train_ds):
            print(
                f"[PatchStats] WARNING: flags length {flags.shape[0]} != len(train_ds) {len(train_ds)}; "
                "truncating / padding with False."
            )
            if flags.shape[0] > len(train_ds):
                flags = flags[: len(train_ds)]
            else:
                extra = np.zeros(len(train_ds) - flags.shape[0], dtype=bool)
                flags = np.concatenate([flags, extra], axis=0)
        print_diagnostic_item("Train patches (flags)", len(flags), indent=1)
        print_diagnostic_item("Fire patches", int(flags.sum()), indent=1)
        return flags

    print_diagnostic_header("Patch fire flags (scan)")
    print("[PatchStats] Computing patch fire flags from scratch...")

    loader = DataLoader(
        train_ds,
        batch_size=args.patch_stats_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        persistent_workers=(args.workers > 0),
    )

    flags = np.zeros(len(train_ds), dtype=np.bool_)
    idx = 0

    iterator = loader
    if args.tqdm_scan and not args.no_tqdm:
        iterator = tqdm(loader, desc="scan_fire_patches", total=len(loader))

    for batch in iterator:
        if args.scan_to_device:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        y = batch["y"]
        m = batch["mask"]
        B = y.shape[0]

        has_fire = ((y > 0.5) & (m > 0.5)).view(B, -1).any(dim=1).cpu().numpy()
        flags[idx : idx + B] = has_fire
        idx += B

    np.save(path, flags)
    print(f"[PatchStats] Saved patch fire flags to {path}")
    print_diagnostic_item("Train patches (flags)", len(flags), indent=1)
    print_diagnostic_item("Fire patches", int(flags.sum()), indent=1)
    return flags


def make_patch_sampler(flags: np.ndarray, args, epoch: int) -> WeightedRandomSampler:
    flags = np.asarray(flags, dtype=np.bool_)
    N = flags.shape[0]
    n_pos = int(flags.sum())
    n_neg = N - n_pos

    if n_pos == 0 or n_neg == 0:
        print("[PatchSampler] Only one class present; using uniform sampling.")
        weights = np.ones(N, dtype=np.float64)
        return WeightedRandomSampler(torch.from_numpy(weights), num_samples=N, replacement=True)

    base_pos = min(args.max_patch_pos_oversample, float(n_neg) / float(max(n_pos, 1)))
    base_neg = 1.0

    neg_scale = 1.0
    if args.curriculum_epochs > 0:
        if epoch < args.curriculum_start_epoch:
            neg_scale = args.curriculum_neg_weight_min
        else:
            cur = max(epoch - args.curriculum_start_epoch, 0)
            cur = min(cur, args.curriculum_epochs)
            if args.curriculum_epochs > 0:
                prog = cur / float(args.curriculum_epochs)
                neg_scale = (
                    args.curriculum_neg_weight_min
                    + prog * (args.curriculum_neg_weight_max - args.curriculum_neg_weight_min)
                )
            else:
                neg_scale = args.curriculum_neg_weight_max

    w_pos = base_pos
    w_neg = base_neg * neg_scale

    weights = np.where(flags, w_pos, w_neg).astype(np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=N, replacement=True)

    print(
        f"[PatchSampler] Epoch {epoch}: n_pos={n_pos}, n_neg={n_neg}, "
        f"w_pos={w_pos:.3f}, w_neg={w_neg:.6f}"
    )
    return sampler


# ----------------------------------------------------------------------
# Logging helpers
# ----------------------------------------------------------------------

def init_metrics_csv(logdir: str) -> str:
    os.makedirs(logdir, exist_ok=True)
    path = os.path.join(logdir, "metrics_log.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
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
                ]
            )
    return path


def append_metrics_csv(path: str, epoch: int, split: str, loss: float, metrics: Dict[str, Any], calib_method: str):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                epoch,
                split,
                loss,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics.get("logloss", ""),
                metrics.get("roc_auc", ""),
                metrics["tp"],
                metrics["fp"],
                metrics["fn"],
                metrics["tn"],
                metrics["support"],
                "",
                calib_method,
            ]
        )

        for h, m in metrics["per_horizon"].items():
            writer.writerow(
                [
                    epoch,
                    f"{split}_h{h}",
                    "",
                    m["accuracy"],
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    m.get("logloss", ""),
                    m.get("roc_auc", ""),
                    m["tp"],
                    m["fp"],
                    m["fn"],
                    m["tn"],
                    m["support"],
                    h,
                    calib_method,
                ]
            )


# ----------------------------------------------------------------------
# History dumping & matplotlib plots (unchanged)
# ----------------------------------------------------------------------

def save_history_json(history: Dict[str, Any], logdir: str) -> str:
    os.makedirs(logdir, exist_ok=True)
    path = os.path.join(logdir, "metrics_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Plots] Saved raw history JSON to {path}")
    return path


def _plot_multi_curves(curves: Sequence[Tuple[Sequence[float], Sequence[float], str]],
                      title: str, xlabel: str, ylabel: str, out_path: str, args):
    fig, ax = plt.subplots()
    has_data = False
    for xs, ys, label in curves:
        if xs is None or ys is None:
            continue
        if len(xs) == 0 or len(ys) == 0:
            continue
        if len(xs) != len(ys):
            n = min(len(xs), len(ys))
            xs = xs[:n]
            ys = ys[:n]
        ax.plot(xs, ys, marker="o", label=label)
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

    train_hist = history.get("train", {})
    val_hist = history.get("val", {})

    train_epochs = train_hist.get("epoch", [])
    train_loss = train_hist.get("loss", [])

    val_epochs = val_hist.get("epoch", [])
    val_loss = val_hist.get("loss", [])
    val_prec = val_hist.get("precision", [])
    val_rec = val_hist.get("recall", [])
    val_f1 = val_hist.get("f1", [])
    val_logloss = val_hist.get("logloss", [])
    val_roc_auc = val_hist.get("roc_auc", [])
    val_tp = val_hist.get("tp", [])
    val_fp = val_hist.get("fp", [])
    val_fn = val_hist.get("fn", [])
    val_tn = val_hist.get("tn", [])

    _plot_multi_curves(
        [(train_epochs, train_loss, "train_loss"), (val_epochs, val_loss, "val_loss")],
        title="Train vs Val Loss",
        xlabel="Epoch",
        ylabel="Loss",
        out_path=os.path.join(logdir, f"loss_curves.{args.plot_file_format}"),
        args=args,
    )

    _plot_multi_curves(
        [(val_epochs, val_prec, "val_precision"), (val_epochs, val_rec, "val_recall"), (val_epochs, val_f1, "val_f1")],
        title="Validation Precision / Recall / F1",
        xlabel="Epoch",
        ylabel="Score",
        out_path=os.path.join(logdir, f"val_core_metrics.{args.plot_file_format}"),
        args=args,
    )

    _plot_multi_curves(
        [(val_epochs, val_tp, "TP"), (val_epochs, val_fp, "FP"), (val_epochs, val_fn, "FN"), (val_epochs, val_tn, "TN")],
        title="Validation Confusion Counts",
        xlabel="Epoch",
        ylabel="Count",
        out_path=os.path.join(logdir, f"val_confusion_counts.{args.plot_file_format}"),
        args=args,
    )

    _plot_multi_curves(
        [(val_epochs, val_logloss, "val_logloss")],
        title="Validation Log Loss",
        xlabel="Epoch",
        ylabel="Log Loss",
        out_path=os.path.join(logdir, f"val_logloss.{args.plot_file_format}"),
        args=args,
    )

    _plot_multi_curves(
        [(val_epochs, val_roc_auc, "val_roc_auc")],
        title="Validation ROC AUC",
        xlabel="Epoch",
        ylabel="AUC",
        out_path=os.path.join(logdir, f"val_roc_auc.{args.plot_file_format}"),
        args=args,
    )

    per_h = val_hist.get("per_horizon", {}) or {}
    for h, h_hist in per_h.items():
        h_epochs = h_hist.get("epoch", [])
        h_prec = h_hist.get("precision", [])
        h_rec = h_hist.get("recall", [])
        h_f1 = h_hist.get("f1", [])
        _plot_multi_curves(
            [(h_epochs, h_prec, f"h={h} precision"), (h_epochs, h_rec, f"h={h} recall"), (h_epochs, h_f1, f"h={h} f1")],
            title=f"Validation Metrics (horizon={h})",
            xlabel="Epoch",
            ylabel="Score",
            out_path=os.path.join(logdir, f"val_h{h}_metrics.{args.plot_file_format}"),
            args=args,
        )


def plot_roc_curves_from_metrics(metrics: Dict[str, Any], args, split: str, epoch: Optional[int] = None):
    calib = metrics.get("calibration", {})
    roc_overall = calib.get("roc", {})
    fpr_overall = roc_overall.get("fpr", [])
    tpr_overall = roc_overall.get("tpr", [])
    auc_overall = roc_overall.get("auc", float("nan"))

    curves = []
    label_overall = "overall"
    if not math.isnan(auc_overall):
        label_overall += f" (AUC={auc_overall:.3f})"
    curves.append((fpr_overall, tpr_overall, label_overall))

    per_h_calib = calib.get("per_horizon", {})
    for h, h_calib in per_h_calib.items():
        roc_h = h_calib.get("roc", {})
        fpr_h = roc_h.get("fpr", [])
        tpr_h = roc_h.get("tpr", [])
        auc_h = roc_h.get("auc", float("nan"))
        label = f"h={h}"
        if not math.isnan(auc_h):
            label += f" (AUC={auc_h:.3f})"
        curves.append((fpr_h, tpr_h, label))

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
# Training loop
# ----------------------------------------------------------------------

def train_one_epoch(
    model,
    train_ds,
    epoch: int,
    optimizer,
    criterion,
    device,
    args,
    monitor: Optional[RAMMonitor],
    writer: Optional[SummaryWriter],
    global_step: int,
    patch_flags: Optional[np.ndarray],
) -> Tuple[float, int]:
    model.train()

    sampler = None
    if args.patch_sampling == "balanced":
        if patch_flags is None:
            raise RuntimeError("patch_flags is None but patch-sampling is 'balanced'")
        sampler = make_patch_sampler(patch_flags, args, epoch=epoch)

    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        args=args,
        sampler=sampler,
    )

    iterator = wrap_loader(
        train_loader,
        desc=f"train[epoch={epoch}]",
        use_tqdm=not args.no_tqdm,
        show_worker_stats=args.show_worker_stats,
        position=2,
    )

    total_loss = 0.0
    total_mask = 0.0
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(iterator):
        if monitor is not None:
            monitor.set_status(f"Train e{epoch}/{args.epochs} step {step + 1}")

        t0 = time.time()
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        t_load = time.time() - t0

        with amp_autocast(device):
            t1 = time.time()
            logits = model(batch)
            t_forward = time.time() - t1

            loss_raw = criterion(logits, batch["y"], batch["mask"])
            loss = loss_raw / max(1, args.grad_accum)

        t2 = time.time()
        loss.backward()
        t_backward = time.time() - t2

        if (step + 1) % args.grad_accum == 0:
            t3 = time.time()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            t_opt = time.time() - t3
        else:
            t_opt = 0.0

        if args.sync_every_step and device.type == "cuda":
            torch.cuda.synchronize()

        m = batch["mask"].sum().item()
        total_loss += loss_raw.item() * m
        total_mask += m

        if writer is not None:
            writer.add_scalar("train/loss_step", loss_raw.item(), global_step)
            writer.add_scalar("train/load_time", t_load, global_step)
            writer.add_scalar("train/forward_time", t_forward, global_step)
            writer.add_scalar("train/backward_time", t_backward, global_step)
            writer.add_scalar("train/optim_time", t_opt, global_step)
            writer.add_scalar("system/ram_gb", get_ram_usage(), global_step)
            if torch.cuda.is_available():
                writer.add_scalar("system/gpu_mem_gb", get_gpu_memory(), global_step)

        global_step += 1

    elapsed = time.time() - start_time
    avg_loss = float("nan") if total_mask == 0 else (total_loss / total_mask)

    n_samples = len(train_ds)
    samples_per_sec = n_samples / max(elapsed, 1e-6)
    print(
        f"[Train] Epoch {epoch} done | loss={avg_loss:.6f} | "
        f"samples={n_samples} | time={elapsed:.1f}s | {samples_per_sec:.1f} samples/s"
    )

    return avg_loss, global_step


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.quick_test:
        print("[Mode] QUICK TEST: overriding some args.")
        args.epochs = 1
        args.batch_size = max(1, min(args.batch_size, 4))
        args.max_samples = 1024
        if args.limit_train_samples <= 0:
            args.limit_train_samples = 512
        else:
            args.limit_train_samples = min(args.limit_train_samples, 512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    describe_environment(args, device)

    monitor = RAMMonitor(device_id=device.index if device.type == "cuda" else 0)
    if not args.no_tqdm:
        monitor.start()

    start_ram = get_ram_usage()

    with DiagnosticTimer("Dataset init", track_ram=True):
        train_ds, val_ds, test_ds = build_datasets(args)

    # NEW: split off calibration subset from TRAIN
    train_ds, calib_ds = split_train_for_calibration(train_ds, args)

    if args.limit_train_samples > 0 and args.limit_train_samples < len(train_ds):
        indices = np.arange(args.limit_train_samples)
        train_ds = Subset(train_ds, indices)
        print_diagnostic_item("Train subset", f"{len(train_ds)} (limit_train_samples)", indent=1)

    if args.pause_after_dataset:
        input("[Pause] Dataset initialized. Press Enter to continue...")

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty after filtering/splitting.")

    sample = train_ds[0]
    x = sample["x"]
    if x.dim() == 4:
        T, C, _, _ = x.shape
        in_channels = T * C
    elif x.dim() == 3:
        in_channels = x.shape[0]
    else:
        raise ValueError(f"Unexpected x shape from dataset: {tuple(x.shape)}")

    model = UNetLiteTimeChannels(in_channels=in_channels, horizons=args.horizons, base_ch=args.base_ch, dropout=args.dropout,t_hist=args.T_hist)
    model.to(device)

    print_diagnostic_item("Sample x shape", tuple(x.shape), indent=1)
    print_diagnostic_item("Computed in_channels", in_channels, indent=1)
    total_params, trainable_params = count_params(model)
    print_diagnostic_header("Model")
    print_diagnostic_item("In channels", in_channels, indent=1)
    print_diagnostic_item("Horizons", args.horizons, indent=1)
    print_diagnostic_item("Total params", human_int(total_params), indent=1)
    print_diagnostic_item("Trainable params", human_int(trainable_params), indent=1)
    if args.verbose_model or args.debug:
        print(model)

    if args.compile and hasattr(torch, "compile"):
        print("[Model] Wrapping with torch.compile()")
        model = torch.compile(model)

    if args.pause_after_model:
        input("[Pause] Model initialized. Press Enter to continue...")

    train_loss_name = args.train_loss or args.loss
    eval_loss_name = args.eval_loss or args.loss

    train_criterion = make_criterion_from_name(train_loss_name, args).to(device)
    eval_criterion  = make_criterion_from_name(eval_loss_name, args).to(device)

    print(f"\n[Loss] Train loss: {train_loss_name}")
    print(f"[Loss] Eval  loss: {eval_loss_name}")
    if train_loss_name == "focal_tversky" and args.enable_pixel_class_weights:
        print("[Loss] NOTE: --enable-pixel-class-weights is ignored for focal_tversky; use --tversky-* instead.")
    if eval_loss_name == "focal_tversky" and args.enable_pixel_class_weights:
        print("[Loss] NOTE: --enable-pixel-class-weights is ignored for focal_tversky; use --tversky-* instead.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    writer: Optional[SummaryWriter] = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=args.logdir)
        print(f"[TB] Logging to: {args.logdir}")

    metrics_csv_path = init_metrics_csv(args.logdir)

    patch_flags = None
    if args.patch_sampling != "none":
        patch_flags = compute_patch_fire_flags(train_ds, args, device)

    print_ram_delta(start_ram, "After model + loss + optimizer + patch stats")

    val_loader = None
    test_loader = None
    calib_loader = None
    if val_ds is not None:
        val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False, args=args)
    if test_ds is not None:
        test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False, args=args)
    if calib_ds is not None:
        calib_loader = make_loader(calib_ds, batch_size=args.batch_size, shuffle=False, args=args)

    if args.sample_one_batch:
        print("[Debug] --sample-one-batch set: grabbing a single batch and exiting.")
        dbg_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, args=args)
        for batch in dbg_loader:
            print("[Debug] Batch keys:", batch.keys())
            print("[Debug] x shape:", batch["x"].shape)
            print("[Debug] y shape:", batch["y"].shape)
            print("[Debug] mask shape:", batch["mask"].shape)
            print("[Debug] meta[0]:", batch["meta"][0])
            break
        if monitor is not None:
            monitor.stop()
        return

    if args.val_test_only:
        print("[Mode] VAL-TEST-ONLY: running eval on val/test then exiting.")
        calibrator = None
        if calib_loader is not None and args.calibration_method != "none":
            calibrator = fit_calibrator(model, calib_loader, device, args)

        if val_loader is not None:
            val_loss, val_metrics = evaluate(model, val_loader, device, eval_criterion, args, calibrator=calibrator,
                                            use_tqdm=not args.no_tqdm, show_worker_stats=args.show_worker_stats)
            print_diagnostic_header("Val-only metrics")
            print(f"[Val] loss={val_loss:.6f}, acc={val_metrics['accuracy']:.4f}, prec={val_metrics['precision']:.4f}, rec={val_metrics['recall']:.4f}, f1={val_metrics['f1']:.4f}")

        if test_loader is not None:
            test_loss, test_metrics = evaluate(model, test_loader, device, eval_criterion, args, calibrator=calibrator,
                                              use_tqdm=not args.no_tqdm, show_worker_stats=args.show_worker_stats)
            print_diagnostic_header("Test-only metrics")
            print(f"[Test] loss={test_loss:.6f}, acc={test_metrics['accuracy']:.4f}, prec={test_metrics['precision']:.4f}, rec={test_metrics['recall']:.4f}, f1={test_metrics['f1']:.4f}")

        if monitor is not None:
            monitor.stop()
        return

    if args.dry_run:
        print("[Mode] DRY-RUN: initialization done; exiting before training.")
        if monitor is not None:
            monitor.stop()
        return

    print_diagnostic_header("Training")

    best_val_f1 = -1.0
    best_epoch = -1
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
                int(h): {
                    "epoch": [],
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                    "f1": [],
                    "logloss": [],
                    "roc_auc": [],
                }
                for h in args.horizons
            },
        },
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, train_ds, epoch, optimizer, train_criterion, device, args, monitor, writer, global_step, patch_flags
        )

        history["train"]["epoch"].append(epoch)
        history["train"]["loss"].append(float(train_loss))
        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)

        # NEW: fit calibrator after training epoch
        calibrator = None
        calib_method = "none"
        if calib_loader is not None and args.calibration_method != "none":
            calib_method = args.calibration_method
            if monitor is not None:
                monitor.set_status(f"Calibrating ({calib_method}) e{epoch}/{args.epochs}")
            calibrator = fit_calibrator(model, calib_loader, device, args)
            if writer is not None and calibrator is not None and calibrator.method == "platt":
                for i, h in enumerate(args.horizons):
                    writer.add_scalar(f"calib/platt_a_h{h}", float(calibrator.a[i].cpu().item()), epoch)
                    writer.add_scalar(f"calib/platt_b_h{h}", float(calibrator.b[i].cpu().item()), epoch)

        if val_loader is not None:
            if monitor is not None:
                monitor.set_status(f"Eval e{epoch}/{args.epochs}")
            val_loss, val_metrics = evaluate(
                model, val_loader, device, eval_criterion, args,
                calibrator=calibrator,
                use_tqdm=not args.no_tqdm,
                show_worker_stats=args.show_worker_stats,
            )

            save_and_log_calibration("val", val_metrics, epoch, args, writer)

            print_diagnostic_header(f"Validation (epoch {epoch})")
            print(
                f"[Val] (calib={calib_method}) loss={val_loss:.6f}, "
                f"acc={val_metrics['accuracy']:.4f}, "
                f"prec={val_metrics['precision']:.4f}, "
                f"rec={val_metrics['recall']:.4f}, "
                f"f1={val_metrics['f1']:.4f}, "
                f"logloss={val_metrics.get('logloss', float('nan')):.6f}, "
                f"auc={val_metrics.get('roc_auc', float('nan')):.4f}, "
                f"ece={val_metrics['ece']:.4f}, "
                f"brier={val_metrics['brier']:.4f}"
            )
            for h, m in val_metrics["per_horizon"].items():
                print(
                    f"  [Val h={h}] acc={m['accuracy']:.4f}, "
                    f"prec={m['precision']:.4f}, rec={m['recall']:.4f}, f1={m['f1']:.4f}, "
                    f"logloss={m.get('logloss', float('nan')):.6f}, auc={m.get('roc_auc', float('nan')):.4f}, "
                    f"tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, tn={m['tn']}"
                )

            append_metrics_csv(metrics_csv_path, epoch, "val", val_loss, val_metrics, calib_method)

            vh = history["val"]
            vh["epoch"].append(epoch)
            vh["loss"].append(float(val_loss))
            vh["accuracy"].append(float(val_metrics["accuracy"]))
            vh["precision"].append(float(val_metrics["precision"]))
            vh["recall"].append(float(val_metrics["recall"]))
            vh["f1"].append(float(val_metrics["f1"]))
            vh["logloss"].append(float(val_metrics.get("logloss", float("nan"))))
            vh["roc_auc"].append(float(val_metrics.get("roc_auc", float("nan"))))
            vh["tp"].append(float(val_metrics["tp"]))
            vh["fp"].append(float(val_metrics["fp"]))
            vh["fn"].append(float(val_metrics["fn"]))
            vh["tn"].append(float(val_metrics["tn"]))
            vh["support"].append(float(val_metrics["support"]))

            for h, m in val_metrics["per_horizon"].items():
                h_int = int(h)
                per_h = vh["per_horizon"].setdefault(
                    h_int,
                    {"epoch": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "logloss": [], "roc_auc": []},
                )
                per_h["epoch"].append(epoch)
                per_h["accuracy"].append(float(m["accuracy"]))
                per_h["precision"].append(float(m["precision"]))
                per_h["recall"].append(float(m["recall"]))
                per_h["f1"].append(float(m["f1"]))
                per_h["logloss"].append(float(m.get("logloss", float("nan"))))
                per_h["roc_auc"].append(float(m.get("roc_auc", float("nan"))))

            if writer is not None:
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
                writer.add_scalar("val/precision", val_metrics["precision"], epoch)
                writer.add_scalar("val/recall", val_metrics["recall"], epoch)
                writer.add_scalar("val/f1", val_metrics["f1"], epoch)
                writer.add_scalar("val/logloss", val_metrics.get("logloss", float("nan")), epoch)
                writer.add_scalar("val/roc_auc", val_metrics.get("roc_auc", float("nan")), epoch)
                writer.add_scalar("val/ece", val_metrics["ece"], epoch)
                writer.add_scalar("val/brier", val_metrics["brier"], epoch)
                for h, m in val_metrics["per_horizon"].items():
                    if "ece" in m:
                        writer.add_scalar(f"val/h{h}_ece", m["ece"], epoch)
                    if "brier" in m:
                        writer.add_scalar(f"val/h{h}_brier", m["brier"], epoch)
                    if "logloss" in m:
                        writer.add_scalar(f"val/h{h}_logloss", m["logloss"], epoch)
                    if "roc_auc" in m:
                        writer.add_scalar(f"val/h{h}_roc_auc", m["roc_auc"], epoch)
                    writer.add_scalar(f"val/h{h}_precision", m["precision"], epoch)
                    writer.add_scalar(f"val/h{h}_recall", m["recall"], epoch)
                    writer.add_scalar(f"val/h{h}_f1", m["f1"], epoch)

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_epoch = epoch
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
                print(f"[Model] New best F1={best_val_f1:.4f} at epoch {epoch} -> saved to {best_path}")
                if writer is not None:
                    writer.add_scalar("val/best_f1", best_val_f1, epoch)

            last_val_metrics = val_metrics

        if epoch == args.epochs and test_loader is not None:
            if monitor is not None:
                monitor.set_status(f"Test after epoch {epoch}")
            test_loss, test_metrics = evaluate(
                model, test_loader, device, eval_criterion, args,
                calibrator=calibrator,
                use_tqdm=not args.no_tqdm,
                show_worker_stats=args.show_worker_stats,
            )
            save_and_log_calibration("test", test_metrics, epoch, args, writer)

            print_diagnostic_header(f"Test (epoch {epoch})")
            print(
                f"[Test] (calib={calib_method}) loss={test_loss:.6f}, "
                f"acc={test_metrics['accuracy']:.4f}, "
                f"prec={test_metrics['precision']:.4f}, "
                f"rec={test_metrics['recall']:.4f}, "
                f"f1={test_metrics['f1']:.4f}, "
                f"logloss={test_metrics.get('logloss', float('nan')):.6f}, "
                f"auc={test_metrics.get('roc_auc', float('nan')):.4f}, "
                f"ece={test_metrics['ece']:.4f}, "
                f"brier={test_metrics['brier']:.4f}"
            )

            append_metrics_csv(metrics_csv_path, epoch, "test", test_loss, test_metrics, calib_method)
            final_test_metrics = test_metrics

    try:
        save_history_json(history, args.logdir)
        if args.plot_metrics:
            plot_metric_history(history, args)
            if last_val_metrics is not None:
                plot_roc_curves_from_metrics(last_val_metrics, args, split="val")
            if final_test_metrics is not None:
                plot_roc_curves_from_metrics(final_test_metrics, args, split="test")
    except Exception as e:  # noqa: BLE001
        print(f"[Plots] Failed to generate history JSON or plots: {e}")

    print_diagnostic_header("Training complete")
    if best_epoch > 0:
        print(f"[Summary] Best val F1={best_val_f1:.4f} at epoch {best_epoch}")
    else:
        print("[Summary] No validation dataset; best epoch undefined.")

    if writer is not None:
        writer.flush()
        writer.close()

    if monitor is not None:
        monitor.stop()


if __name__ == "__main__":
    main()
