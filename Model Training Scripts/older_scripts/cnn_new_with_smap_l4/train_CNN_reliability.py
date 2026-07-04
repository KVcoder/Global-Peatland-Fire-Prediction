#!/usr/bin/env python3
"""
Spatiotemporal Peat Ignition — Heavily Optimized for Speed & Memory
WITH COMPREHENSIVE REAL-TIME MONITORING

Now using JointPeatDataset (ERA5 + SMAP -> VIIRS Zarr-backed patches).

Major features:
- Persistent tqdm bars showing RAM/GPU usage at all times
- Real-time tracking of worker data loading time
- Forward and backward pass timing displayed continuously
- Throughput metrics (samples/sec, updates/sec)
- Detailed bottleneck identification

Data & model:
- JointPeatDataset-based input pipeline (x, y, mask, meta)
- CNN UNet model with time stacked into channels (2D UNet on 32x32 patches)
- Robust NaN detection and hard-stop debugging

Monitoring:
- Storage type detection (SSD vs HDD vs network FS) for Zarr roots
- Explicit multiprocessing context reporting (forkserver/spawn/etc.)
- Per-batch data loading & transfer timing
- DataLoader queue tracker: how many batches are ready in RAM vs currently loading

Metrics & logging:
- Validation metrics after each epoch: accuracy, precision, recall, F1, etc.
- Per-horizon validation metrics (per forecast horizon)
- CSV logging of metrics per epoch (logs/metrics_log.csv inside logdir)
- TensorBoard logging for:
    * Per-step loss, timings, throughput
    * RAM, GPU memory, loader queue stats
    * Epoch losses & validation metrics (overall + per horizon)
    * “Best so far” validation metrics

Patch rebalancing and curriculum:
- Optional patch-level sampler using cached fire/non-fire flags
- Optional PROGRESSIVE curriculum: smoothly ramp in non-fire patches
  over configurable epochs, with configurable start epoch and
  non-fire weights.

Per-pixel class reweighting:
- Dynamic N_bg/N_fire weighting (clipped) for fire pixels for BCE/Focal.

NEW in this edit:
- Added MaskedFocalTverskyLossWithLogits ("focal_tversky" option).
- New CLI flags:
    * --tversky-alpha, --tversky-beta, --tversky-gamma

NEW (this version):
- Matplotlib plotting of training / validation history:
    * Train vs val loss curves
    * Val precision/recall/F1 vs epoch
    * Val confusion counts (TP/FP/FN/TN) vs epoch
    * Per-horizon val precision/recall/F1 vs epoch
- History dumped to metrics_history.json in logdir.
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


class DoubleConv(nn.Module):
    """
    Two Conv2d + BatchNorm + ReLU blocks:
      in_ch -> out_ch -> out_ch
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetTimeChannels(nn.Module):
    """
    2D UNet where *time is stacked into channels*.

    Expects batch["x"] as either:
      - (B, C_in, H, W)    with time already stacked into C_in, or
      - (B, T, C, H, W)    and we flatten to (B, T*C, H, W) on the fly.

    Outputs:
      logits: (B, K, H, W), where K = number of horizons.
    """

    def __init__(self, in_channels: int, horizons: Sequence[int], t_hist: Optional[int] = None):
        super().__init__()
        self.in_channels = int(in_channels)
        self.horizons = [int(h) for h in horizons]
        self.k = len(self.horizons)
        self.t_hist = t_hist  # for logging only; UNet doesn't use it directly

        # Encoder
        self.enc1 = DoubleConv(self.in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottom = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)

        # Head: per-pixel logits for K horizons
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.k, kernel_size=1),
        )

    def _flatten_time_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
          - (B, C_in, H, W)  → returned as-is
          - (B, T, C, H, W)  → reshaped to (B, T*C, H, W)
        """
        if x.dim() == 4:
            return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B, T * C, H, W)
            return x
        raise ValueError(
            f"UNetTimeChannels expected x with 4 or 5 dims, got {tuple(x.shape)}"
        )

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x = batch["x"]
        x = self._flatten_time_if_needed(x)

        # Encoder
        x1 = self.enc1(x)  # (B, 64, H, W)
        x2 = self.enc2(self.pool1(x1))  # (B, 128, H/2, W/2)
        x3 = self.enc3(self.pool2(x2))  # (B, 256, H/4, W/4)
        x4 = self.bottom(self.pool3(x3))  # (B, 512, H/8, W/8)

        # Decoder
        u3 = self.up3(x4)  # (B, 256, H/4, W/4)
        u3 = torch.cat([u3, x3], dim=1)  # (B, 512, H/4, W/4)
        u3 = self.dec3(u3)  # (B, 256, H/4, W/4)

        u2 = self.up2(u3)  # (B, 128, H/2, W/2)
        u2 = torch.cat([u2, x2], dim=1)  # (B, 256, H/2, W/2)
        u2 = self.dec2(u2)  # (B, 128, H/2, W/2)

        u1 = self.up1(u2)  # (B, 64, H, W)
        u1 = torch.cat([u1, x1], dim=1)  # (B, 128, H, W)
        u1 = self.dec1(u1)  # (B, 64, H, W)

        logits = self.head(u1)  # (B, K, H, W)
        return logits


# ----------------------------------------------------------------------
# Loss & monitoring utilities
# ----------------------------------------------------------------------

def save_and_log_calibration(split: str, metrics: Dict[str, Any], epoch: int, args, writer: Optional[SummaryWriter]):
    """
    Save reliability-bias curves to .npz and log an overall curve figure to TensorBoard.
    split: "val" or "test"
    """
    calib = metrics.get("calibration", {}) or {}

    # Save NPZ
    out_npz = os.path.join(args.logdir, f"reliability_bias_{split}_epoch{epoch}.npz")
    try:
        save_reliability_bias_npz(calib, out_npz)
        if args.debug:
            print(f"[ReliabilityBias] Saved: {out_npz}")
    except Exception as e:  # noqa: BLE001
        print(f"[ReliabilityBias] Failed to save NPZ ({split}): {e}")

    # TensorBoard figure (overall curve)
    if writer is not None:
        try:
            tb_log_reliability_bias_curve(writer, calib, epoch, tag=f"{split}/reliability_bias_pct")
        except Exception as e:  # noqa: BLE001
            print(f"[ReliabilityBias] Failed to log TB figure ({split}): {e}")

        # Optional per-horizon figures (only if per-horizon bias exists)
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



class MaskedBCEWithLogits(nn.Module):
    """
    BCE-with-logits + mask, with optional dynamic per-pixel class weights.

    If enable_class_weights:
      - For each batch, compute N_fire and N_background (valid pixels only).
      - Set fire pixels' loss weight to w_fire = min(N_background / N_fire, max_pos_weight).
    """

    def __init__(self, enable_class_weights: bool = False, max_pos_weight: float = 100.0):
        super().__init__()
        self.enable_class_weights = bool(enable_class_weights)
        self.max_pos_weight = float(max_pos_weight)

    def forward(self, logits, targets, mask):
        # logits, targets, mask: (B, K, H, W)
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
                    weights[~(pos | neg)] = 0.0  # everything invalid gets weight 0
                else:
                    weights = torch.ones_like(loss)
            loss = loss * weights

        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss


class MaskedFocalLossWithLogits(nn.Module):
    """
    Mask-aware focal loss for heavily imbalanced binary labels.

    logits, targets, mask: (B, K, H, W)
      - targets: 0/1 floats
      - mask   : 0/1 floats where 1 = valid label

    NEW: optional dynamic per-pixel class weights (same logic as BCE).
    """

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
    """
    Mask-aware Focal Tversky loss for highly imbalanced segmentation.

    logits, targets, mask: (B, K, H, W)
      - targets: 0/1 floats
      - mask   : 0/1 floats where 1 = valid label

    Tversky index (per horizon, soft):
        TP = sum(p * t)
        FP = sum(p * (1 - t))
        FN = sum((1 - p) * t)

        TI = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

    Focal Tversky loss:
        L = mean_horizons[(1 - TI) ^ gamma]

    Notes:
      - This loss *already* handles imbalance via alpha/beta/gamma, so we
        do NOT additionally apply dynamic pixel class weights here.
      - Mask is applied as a multiplicative binary mask before computing TP/FP/FN.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, logits, targets, mask):
        # shapes: (B, K, H, W)
        probs = torch.sigmoid(logits)
        t = (targets > 0.5).float()
        m = (mask > 0.5).float()

        # Apply mask
        probs = probs * m
        t = t * m

        # Sum over batch + spatial dims, leaving horizon dimension
        # (K,) vectors for TP/FP/FN
        tp = (probs * t).sum(dim=(0, 2, 3))
        fp = (probs * (1.0 - t)).sum(dim=(0, 2, 3))
        fn = ((1.0 - probs) * t).sum(dim=(0, 2, 3))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1.0 - tversky).pow(self.gamma)

        loss = focal_tversky.mean()
        return loss


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
    """
    DataLoader factory with optional sampler (used for patch rebalancing / curriculum).
    """
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
    """
    Given per-bin total counts and positive-label sums (same shape arrays),
    compute ROC curve (fpr, tpr) and AUC using a histogram approximation.

    Returns:
        fpr, tpr, auc
        where fpr/tpr are 1D numpy arrays (ascending in FPR).
    """
    counts = counts.astype(np.float64)
    pos = true_sums.astype(np.float64)
    neg = counts - pos

    P = pos.sum()
    N = neg.sum()
    if P <= 0 or N <= 0:
        # ROC is undefined if only one class is present
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), float("nan")

    # Process bins from highest score to lowest (right-to-left)
    pos_rev = pos[::-1]
    neg_rev = neg[::-1]

    tp_cum = np.cumsum(pos_rev)
    fp_cum = np.cumsum(neg_rev)

    tpr = tp_cum / P
    fpr = fp_cum / N

    # prepend origin (0,0)
    fpr = np.concatenate(([0.0], fpr))
    tpr = np.concatenate(([0.0], tpr))

    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


@torch.no_grad()
def evaluate(model, loader, device, criterion, args, use_tqdm: bool = True, show_worker_stats: bool = False):
    """
    Evaluate model on a loader.

    Returns:
        val_loss, metrics

    metrics includes:
        - classification:
            accuracy, precision, recall, f1, tp, fp, fn, tn, support
        - per_horizon[h]: same metrics + ECE/Brier/logloss/roc_auc and per-horizon ROC curve
        - calibration (overall + per horizon) with:
            * bin_centers, bin_pred, bin_true, bin_count
            * ece, mce, brier
            * roc: {fpr, tpr, auc} for overall + each horizon
        - overall:
            * ece, mce, brier
            * logloss (binary cross-entropy averaged over valid pixels)
            * roc_auc (overall AUC from histogram-based ROC)
    """
    model.eval()
    tot_loss, tot_mask = 0.0, 0.0
    tp_total = fp_total = fn_total = tn_total = 0

    num_horizons = len(args.horizons)
    tp_h = [0 for _ in range(num_horizons)]
    fp_h = [0 for _ in range(num_horizons)]
    fn_h = [0 for _ in range(num_horizons)]
    tn_h = [0 for _ in range(num_horizons)]

    # --- Reliability / calibration accumulators ---
    bin_width = getattr(args, "reliability_bin_width", 0.005)
    num_bins = max(1, int(math.ceil(1.0 / bin_width)))

    # shape: (K, num_bins)
    rel_counts = np.zeros((num_horizons, num_bins), dtype=np.float64)
    rel_pred_sums = np.zeros_like(rel_counts)   # sum of predicted probs
    rel_true_sums = np.zeros_like(rel_counts)   # sum of true labels (0/1)
    brier_sums = np.zeros(num_horizons, dtype=np.float64)
    total_counts_per_h = np.zeros(num_horizons, dtype=np.float64)

    # --- Log loss accumulators (overall + per-horizon) ---
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
    for batch in iterator:
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        logits = model(batch)
        loss = criterion(logits, batch["y"], batch["mask"])
        m = batch["mask"].sum().item()
        if m == 0:
            continue
        tot_loss += loss.item() * m
        tot_mask += m

        probs = torch.sigmoid(logits)
        targets = batch["y"] > 0.5
        valid = batch["mask"] > 0.5

        preds = probs >= args.metrics_threshold

        # --- Classification metrics (same as before) ---
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

        # --- Reliability / calibration stats + logloss (probabilities, not hard preds) ---
        for k_idx in range(K_eff):
            p_k = probs[:, k_idx]      # (B, H, W)
            t_k = targets[:, k_idx]    # bool
            v_k = valid[:, k_idx]      # bool

            # Only valid pixels
            mask_flat = v_k.reshape(-1)
            if mask_flat.sum().item() == 0:
                continue

            p_flat_t = p_k.reshape(-1)[mask_flat].detach().cpu().numpy().astype(np.float32)
            y_flat_t = t_k.reshape(-1)[mask_flat].float().detach().cpu().numpy().astype(np.float32)

            n_valid = float(p_flat_t.size)
            if n_valid == 0:
                continue

            total_counts_per_h[k_idx] += n_valid

            # Brier sum for this horizon
            brier_sums[k_idx] += float(((p_flat_t - y_flat_t) ** 2).sum())

            # --- Log loss for this horizon + overall ---
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

            # Bin indices
            bin_idx = np.floor(p_flat_t / bin_width).astype(np.int64)
            bin_idx = np.clip(bin_idx, 0, num_bins - 1)

            # Aggregate into bins
            counts_batch = np.bincount(bin_idx, minlength=num_bins).astype(np.float64)
            pred_sums_batch = np.bincount(bin_idx, weights=p_flat_t, minlength=num_bins).astype(np.float64)
            true_sums_batch = np.bincount(bin_idx, weights=y_flat_t, minlength=num_bins).astype(np.float64)

            rel_counts[k_idx] += counts_batch
            rel_pred_sums[k_idx] += pred_sums_batch
            rel_true_sums[k_idx] += true_sums_batch

    if tot_mask == 0:
        print("[WARN] Evaluation mask sum was zero across all batches; reporting NaN metrics.")
        val_loss = float("nan")
    else:
        val_loss = tot_loss / tot_mask

    # Overall classification metrics
    overall = _compute_basic_metrics(tp_total, fp_total, fn_total, tn_total)

    # Overall logloss
    if logloss_count_overall > 0:
        logloss_overall = float(logloss_sum_overall / max(logloss_count_overall, 1.0))
    else:
        logloss_overall = float("nan")

    # Per-horizon metrics dict
    per_horizon_metrics: Dict[int, Dict[str, float]] = {}
    for idx, h in enumerate(args.horizons):
        m = _compute_basic_metrics(tp_h[idx], fp_h[idx], fn_h[idx], tn_h[idx])
        # attach per-horizon logloss (if available)
        if logloss_count_h[idx] > 0:
            m["logloss"] = float(logloss_sum_h[idx] / max(logloss_count_h[idx], 1.0))
        else:
            m["logloss"] = float("nan")
        per_horizon_metrics[h] = m

    # --- Finalize reliability / calibration + per-horizon ROC / AUC ---
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
                "roc": {
                    "fpr": [],
                    "tpr": [],
                    "auc": float("nan"),
                },
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

        # slice to paper-like range [0.5%, 6.0%] and convert to percent
        b0 = int(math.floor(args.reliability_bin_min / bin_width))
        b1 = int(math.ceil(args.reliability_bin_max / bin_width))
        b0 = max(0, min(b0, num_bins))
        b1 = max(0, min(b1, num_bins))

        bias_pct = (pred_mean - true_mean) * 100.0
        bias_pct_slice = bias_pct[b0:b1].copy()
        count_slice = counts[b0:b1].copy()

        # optionally mask low-count bins
        minc = int(getattr(args, "reliability_min_count", 0))
        if minc > 0:
            bias_pct_slice[count_slice < minc] = np.nan

        

        
        total = counts.sum()
        gap = np.abs(pred_mean - true_mean)
        ece = float((counts[nonzero] / total * gap[nonzero]).sum())
        mce = float(gap[nonzero].max()) if np.any(nonzero) else float("nan")
        brier = float(brier_sums[idx] / max(total_counts_per_h[idx], 1.0))

        # ROC from histogram
        fpr_h, tpr_h, auc_h = _compute_roc_from_hist(counts, true_sums)

        calibration_per_horizon[h] = {
            "bin_pred": pred_mean.tolist(),
            "bin_true": true_mean.tolist(),
            "bin_count": counts.tolist(),
            "ece": ece,
            "mce": mce,
            "brier": brier,
            "roc": {
                "fpr": fpr_h.tolist(),
                "tpr": tpr_h.tolist(),
                "auc": auc_h,
            },
            "reliability_bias_pct": bias_pct_slice.tolist(),
            "reliability_bias_count": count_slice.tolist(),
            "bin_centers_slice": bin_centers[b0:b1].tolist(),
        }

        per_horizon_metrics[h]["ece"] = ece
        per_horizon_metrics[h]["brier"] = brier
        per_horizon_metrics[h]["roc_auc"] = auc_h

    # Overall calibration & ROC
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

        # Overall ROC
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
        # Calibration summaries
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
            "roc": {
                "fpr": fpr_overall.tolist(),
                "tpr": tpr_overall.tolist(),
                "auc": auc_overall,
            },
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
    p.add_argument(
        "--split",
        type=float,
        default=0.9,
        help="Train fraction. If --val-frac is None, val = 1 - split (2-way split).",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help=(
            "Optional validation fraction for 3-way split. "
            "If set, train = split, val = val-frac, test = 1 - split - val-frac."
        ),
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--normalize-inputs", choices=["none", "per_channel"], default="none")
    p.add_argument(
        "--no-skip-nonpeat",
        action="store_true",
        help="Disable peat-based patch filtering (i.e., keep non-peat patches).",
    )
    p.add_argument("--peat-min-fraction", type=float, default=0.01)

    # --- Training HPs ---
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # --- Loss / imbalance handling ---
    p.add_argument(
        "--loss",
        choices=["bce", "focal", "focal_tversky"],
        default="bce",
        help=(
            "Loss function: 'bce' (MaskedBCEWithLogits), "
            "'focal' (MaskedFocalLossWithLogits), or "
            "'focal_tversky' (MaskedFocalTverskyLossWithLogits)."
        ),
    )
    p.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Alpha for focal loss (weight on positive class).",
    )
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma for focal loss (focusing parameter).",
    )
    p.add_argument(
        "--tversky-alpha",
        type=float,
        default=0.5,
        help="Alpha for Tversky index (weight on FP). Higher = penalize FP more.",
    )
    p.add_argument(
        "--tversky-beta",
        type=float,
        default=0.5,
        help="Beta for Tversky index (weight on FN). Higher = penalize FN more.",
    )
    p.add_argument(
        "--tversky-gamma",
        type=float,
        default=1.5,
        help="Gamma for focal Tversky loss (focusing parameter).",
    )
    p.add_argument(
        "--enable-pixel-class-weights",
        action="store_true",
        help=(
            "Reweight fire pixels in the BCE/focal loss by N_background/N_fire (clipped). "
            "NOTE: ignored for focal_tversky; use tversky-{alpha,beta,gamma} instead."
        ),
    )
    p.add_argument(
        "--max-fire-class-weight",
        type=float,
        default=100.0,
        help="Max multiplicative weight for fire pixels in class-weighted loss.",
    )

    # --- Patch-level rebalancing & curriculum ---
    p.add_argument(
        "--patch-sampling",
        choices=["none", "balanced"],
        default="none",
        help="Patch-level sampling for training. 'balanced' oversamples fire patches.",
    )
    p.add_argument(
        "--patch-stats-path",
        default="patch_fire_flags_train.npy",
        help="Path to cache per-patch fire flags (bool array) for training set.",
    )
    p.add_argument(
        "--patch-stats-batch-size",
        type=int,
        default=256,
        help="Batch size when scanning the dataset to compute patch fire stats.",
    )
    p.add_argument(
        "--max-patch-pos-oversample",
        type=float,
        default=10.0,
        help="Upper bound for N_neg/N_pos oversampling factor for fire patches in balanced sampling.",
    )

    # Curriculum controls
    p.add_argument(
        "--curriculum-epochs",
        type=int,
        default=0,
        help="If > 0, use a progressive curriculum over N epochs to ramp in non-fire patches.",
    )
    p.add_argument(
        "--curriculum-start-epoch",
        type=int,
        default=1,
        help="Epoch at which to START ramping in non-fire patches (1-indexed).",
    )
    p.add_argument(
        "--curriculum-neg-weight-min",
        type=float,
        default=1e-3,
        help="Non-fire patch weight at the START of the curriculum.",
    )
    p.add_argument(
        "--curriculum-neg-weight-max",
        type=float,
        default=1.0,
        help="Non-fire patch weight at the END of the curriculum (usually 1.0).",
    )

    # --- Metrics ---
    p.add_argument(
        "--metrics-threshold",
        type=float,
        default=0.5,
        help="Decision threshold on sigmoid(logits) for metrics.",
    )
    
    p.add_argument(
        "--reliability-bin-width",
        type=float,
        default=0.005, # 0.5% bins
        help="Bin width in probability space for calibration & AUC histograms (e.g. 0.005 = 0.5%).",
    )

    p.add_argument("--reliability-bin-min", type=float, default=0.005)  # 0.5%
    p.add_argument("--reliability-bin-max", type=float, default=0.060)  # 6.0%
    p.add_argument("--reliability-min-count", type=int, default=50)     # mask noisy bins

    # --- Optimization flags ---
    p.add_argument("--compile", action="store_true", help="Use torch.compile() for speedup")
    p.add_argument("--prefetch", type=int, default=2, help="Dataloader prefetch factor")
    p.add_argument(
        "--mp-context",
        choices=["auto", "spawn", "fork", "forkserver"],
        default="auto",
        help="Multiprocessing start method for DataLoader workers",
    )

    # Worker diagnostics
    p.add_argument("--show-worker-stats", action="store_true", help="Show detailed worker timing stats")

    # Verbosity and diagnostics
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument("--log-interval", type=int, default=0)
    p.add_argument("--tqdm-scan", action="store_true")
    p.add_argument("--scan-limit", type=int, default=0)
    p.add_argument("--scan-batch-size", type=int, default=32)
    p.add_argument("--scan-to-device", action="store_true", help="Transfer scan batches to device")

    # Diagnostic flags
    p.add_argument("--debug", action="store_true", help="Maximum verbosity everywhere")
    p.add_argument("--verbose-dataset", action="store_true", help="Detailed dataset logging")
    p.add_argument("--verbose-loader", action="store_true", help="Detailed dataloader logging")
    p.add_argument("--verbose-model", action="store_true", help="Detailed model logging")
    p.add_argument("--profile-first-epoch", action="store_true", help="Profile first epoch in detail")

    # Testing and debugging
    p.add_argument(
        "--skip-val-dataset",
        action="store_true",
        help="Skip val (and test) dataset initialization",
    )
    p.add_argument("--quick-test", action="store_true", help="Quick test: 1 epoch, small batches")
    p.add_argument("--dry-run", action="store_true", help="Initialize everything but don't train")
    p.add_argument(
        "--sample-one-batch",
        action="store_true",
        help="Load just 1 batch and exit (debugging shapes)",
    )

    p.add_argument(
        "--limit-train-samples",
        type=int,
        default=0,
        help="(Optional) extra train sample limit",
    )
    p.add_argument(
        "--val-test-only",
        action="store_true",
        help="Run a validation pass only (no training) to test the validation loop, then exit.",
    )

    # Pause points
    p.add_argument("--pause-after-dataset", action="store_true", help="Pause after dataset init")
    p.add_argument("--pause-after-model", action="store_true", help="Pause after model init")

    # New flags
    p.add_argument(
        "--sync-every-step",
        action="store_true",
        help="Force device sync after each phase for exact timings (slower)",
    )
    p.add_argument(
        "--measure-loader-time",
        action="store_true",
        help="(Kept for compatibility; data loading is always timed directly now)",
    )

    # TensorBoard logging
    p.add_argument(
        "--logdir",
        default="runs/peat_unet",
        help="TensorBoard log directory",
    )
    p.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )

    # Matplotlib metric plotting
    p.add_argument(
        "--plot-metrics",
        action="store_true",
        help="After training, save matplotlib plots of training/validation metrics in logdir.",
    )
    p.add_argument(
        "--plot-file-format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="File format for saved metric plots.",
    )
    p.add_argument(
        "--plot-dpi",
        type=int,
        default=150,
        help="DPI for saved metric plots.",
    )

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
    """
    Build train/val/test JointPeatDataset objects.

    Matches the current JointPeatDataset signature:

        JointPeatDataset(
            era5_zarr,
            smap_zarr,
            viirs_zarr,
            era5_array="field",
            smap_array="field",
            viirs_array="field",
            t_hist=30,
            horizons=(1,3,7,14),
            patch=256,
            stride=None,
            time_stack="separate",
            mode="train" | "val" | "test",
            split=0.9,
            val_frac=None,
            seed=42,
            normalize_inputs=None or "per_channel",
            max_samples=None,
            skip_nonpeat_patches=True,
            peat_min_fraction=0.01,
            time_index=None,
            return_coords=False,
            coord_as_features=False,
        )
    """
    # Map CLI args -> JointPeatDataset kwargs
    common = dict(
        era5_zarr=args.era5_zarr,
        smap_zarr=args.smap_zarr,
        extra_zarr=args.smap_l4_zarr,
        viirs_zarr=args.viirs_zarr,       
        era5_array=args.era5_array,
        smap_array=args.smap_array,
        extra_array=args.smap_l4_array,
        viirs_array=args.viirs_array,
        t_hist=args.T_hist,                     # note: t_hist (lowercase) in dataset
        horizons=args.horizons,
        patch=args.patch,                       # dataset expects 'patch', not 'patch_size'
        stride=args.stride,
        time_stack=args.stack_time,             # dataset parameter name: time_stack
        split=args.split,                       # train fraction
        val_frac=args.val_frac,                 # optional explicit val fraction
        seed=args.seed,
        # JointPeatDataset expects None or "per_channel"; we treat "none" as None
        normalize_inputs=(
            None if args.normalize_inputs == "none" else args.normalize_inputs
        ),
        max_samples=args.max_samples,
        skip_nonpeat_patches=not args.no_skip_nonpeat,
        peat_min_fraction=args.peat_min_fraction,
        time_index=None,
        # You were previously forcing coord_as_features=True, keep that behavior:
        coord_as_features=True,
        # We’re not returning coords separately to the model, just using them as features.
        return_coords=False,
    )

    print_diagnostic_header("Building datasets")

    # ---- Train ----
    train_ds = JointPeatDataset(mode="train", **common)
    print_diagnostic_item("Train patches", len(train_ds), indent=1)

    # ---- Val / Test (optional) ----
    val_ds = None
    test_ds = None

    if not args.skip_val_dataset:
        val_ds = JointPeatDataset(mode="val", **common)
        print_diagnostic_item("Val patches", len(val_ds), indent=1)

        # Only create test set if a 3-way split is implied
        if args.val_frac is not None and (1.0 - args.split - args.val_frac) > 0:
            test_ds = JointPeatDataset(mode="test", **common)
            print_diagnostic_item("Test patches", len(test_ds), indent=1)

    return train_ds, val_ds, test_ds


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
    """
    Scan the train dataset once to determine which patches contain any fire pixels.

    Returns:
        flags: np.ndarray of shape (len(train_ds),) with dtype=bool
    """
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
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
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
    """
    Build a WeightedRandomSampler over patches with optional curriculum.

    flags: bool array of shape (N,) where True = fire patch.
    """
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
        # epoch is 1-indexed
        if epoch < args.curriculum_start_epoch:
            neg_scale = args.curriculum_neg_weight_min
        else:
            # ramp epoch in [0, curriculum_epochs]
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
                ]
            )
    return path


def append_metrics_csv(path: str, epoch: int, split: str, loss: float, metrics: Dict[str, Any]):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        # Overall row (horizon="")
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
            ]
        )

        # Per-horizon rows
        for h, m in metrics["per_horizon"].items():
            writer.writerow(
                [
                    epoch,
                    f"{split}_h{h}",
                    "",  # loss left empty per-horizon
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
                ]
            )



# ----------------------------------------------------------------------
# History dumping & matplotlib plots
# ----------------------------------------------------------------------


def save_history_json(history: Dict[str, Any], logdir: str) -> str:
    """Save raw per-epoch history to metrics_history.json in logdir."""
    os.makedirs(logdir, exist_ok=True)
    path = os.path.join(logdir, "metrics_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Plots] Saved raw history JSON to {path}")
    return path


def _plot_multi_curves(
    curves: Sequence[Tuple[Sequence[float], Sequence[float], str]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
    args,
):
    """
    curves: list of (x_values, y_values, label)
    Skips any curve with empty data.
    """
    fig, ax = plt.subplots()
    has_data = False
    for xs, ys, label in curves:
        if xs is None or ys is None:
            continue
        if len(xs) == 0 or len(ys) == 0:
            continue
        if len(xs) != len(ys):
            # Truncate to min length to be safe
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

def save_reliability_bias_npz(calib: Dict[str, Any], out_path: str):
    """
    Save overall + per-horizon reliability-bias curves into a single .npz.

    Expects calib to contain (overall):
      - bin_centers_slice
      - reliability_bias_pct
      - reliability_bias_count

    And optionally per-horizon entries under calib["per_horizon"][h] with:
      - bin_centers_slice (or fall back to overall)
      - reliability_bias_pct
      - reliability_bias_count
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data = {}

    # Overall
    data["bin_centers"] = np.asarray(calib.get("bin_centers_slice", []), dtype=np.float64)
    data["bias_pct"] = np.asarray(calib.get("reliability_bias_pct", []), dtype=np.float64)
    data["count"] = np.asarray(calib.get("reliability_bias_count", []), dtype=np.float64)

    # Per-horizon (optional)
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
    """
    Log a single figure: Reliability Bias % (Model - Obs) vs probability bin center.
    Uses calib["bin_centers_slice"] and calib["reliability_bias_pct"].
    """
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


def plot_metric_history(history: Dict[str, Any], args):
    """
    Generate a bunch of diagnostic plots from the in-memory history dict.

    Files saved into args.logdir:

      - loss_curves.<ext>          : train vs val loss
      - val_core_metrics.<ext>     : val precision / recall / F1
      - val_confusion_counts.<ext> : TP / FP / FN / TN vs epoch
      - val_logloss.<ext>          : val logloss vs epoch
      - val_roc_auc.<ext>          : val ROC AUC vs epoch
      - val_h{h}_metrics.<ext>     : per-horizon val precision / recall / F1
    """
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

    # 1) Train vs val loss
    _plot_multi_curves(
        [
            (train_epochs, train_loss, "train_loss"),
            (val_epochs, val_loss, "val_loss"),
        ],
        title="Train vs Val Loss",
        xlabel="Epoch",
        ylabel="Loss",
        out_path=os.path.join(logdir, f"loss_curves.{args.plot_file_format}"),
        args=args,
    )

    # 2) Val core metrics: precision / recall / F1
    _plot_multi_curves(
        [
            (val_epochs, val_prec, "val_precision"),
            (val_epochs, val_rec, "val_recall"),
            (val_epochs, val_f1, "val_f1"),
        ],
        title="Validation Precision / Recall / F1",
        xlabel="Epoch",
        ylabel="Score",
        out_path=os.path.join(logdir, f"val_core_metrics.{args.plot_file_format}"),
        args=args,
    )

    # 3) Val confusion counts
    _plot_multi_curves(
        [
            (val_epochs, val_tp, "TP"),
            (val_epochs, val_fp, "FP"),
            (val_epochs, val_fn, "FN"),
            (val_epochs, val_tn, "TN"),
        ],
        title="Validation Confusion Counts",
        xlabel="Epoch",
        ylabel="Count",
        out_path=os.path.join(logdir, f"val_confusion_counts.{args.plot_file_format}"),
        args=args,
    )

    # 4) Val logloss
    _plot_multi_curves(
        [
            (val_epochs, val_logloss, "val_logloss"),
        ],
        title="Validation Log Loss",
        xlabel="Epoch",
        ylabel="Log Loss",
        out_path=os.path.join(logdir, f"val_logloss.{args.plot_file_format}"),
        args=args,
    )

    # 5) Val ROC AUC
    _plot_multi_curves(
        [
            (val_epochs, val_roc_auc, "val_roc_auc"),
        ],
        title="Validation ROC AUC",
        xlabel="Epoch",
        ylabel="AUC",
        out_path=os.path.join(logdir, f"val_roc_auc.{args.plot_file_format}"),
        args=args,
    )

    # 6) Per-horizon metrics
    per_h = val_hist.get("per_horizon", {}) or {}
    for h, h_hist in per_h.items():
        h_epochs = h_hist.get("epoch", [])
        h_prec = h_hist.get("precision", [])
        h_rec = h_hist.get("recall", [])
        h_f1 = h_hist.get("f1", [])
        _plot_multi_curves(
            [
                (h_epochs, h_prec, f"h={h} precision"),
                (h_epochs, h_rec, f"h={h} recall"),
                (h_epochs, h_f1, f"h={h} f1"),
            ],
            title=f"Validation Metrics (horizon={h})",
            xlabel="Epoch",
            ylabel="Score",
            out_path=os.path.join(logdir, f"val_h{h}_metrics.{args.plot_file_format}"),
            args=args,
        )


def plot_roc_curves_from_metrics(metrics: Dict[str, Any], args, split: str, epoch: Optional[int] = None):
    """
    Plot ROC curves (overall + per-horizon) using the calibration['roc'] info
    from a metrics dict returned by `evaluate`.
    """
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
    """
    Train for a single epoch.

    Returns:
        avg_loss, new_global_step
    """
    model.train()

    # Optional patch-level sampler (balanced + curriculum)
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
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
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
    if total_mask == 0:
        avg_loss = float("nan")
    else:
        avg_loss = total_loss / total_mask

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

    # RAM monitor
    monitor = RAMMonitor(device_id=device.index if device.type == "cuda" else 0)
    if not args.no_tqdm:
        monitor.start()

    start_ram = get_ram_usage()

    # Datasets
    with DiagnosticTimer("Dataset init", track_ram=True):
        train_ds, val_ds, test_ds = build_datasets(args)

    if args.limit_train_samples > 0 and args.limit_train_samples < len(train_ds):
        indices = np.arange(args.limit_train_samples)
        train_ds = Subset(train_ds, indices)
        print_diagnostic_item(
            "Train subset",
            f"{len(train_ds)} (limit_train_samples)",
            indent=1,
        )

    if args.pause_after_dataset:
        input("[Pause] Dataset initialized. Press Enter to continue...")

    # Model init: infer input channels from a single sample
    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty after filtering/splitting.")

    
    sample = train_ds[0]
    x = sample["x"]
    if x.dim() == 4:
        # (T, C, H, W)
        T, C, _, _ = x.shape
        in_channels = T * C
    elif x.dim() == 3:
        # (C, H, W)
        in_channels = x.shape[0]
    else:
        raise ValueError(f"Unexpected x shape from dataset: {tuple(x.shape)}")

    model = UNetTimeChannels(in_channels=in_channels, horizons=args.horizons, t_hist=args.T_hist)
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

    # Loss function
    if args.loss == "bce":
        criterion = MaskedBCEWithLogits(
            enable_class_weights=args.enable_pixel_class_weights,
            max_pos_weight=args.max_fire_class_weight,
        )
        print("\n[Loss] Using MaskedBCEWithLogits (BCE).")
    elif args.loss == "focal":
        criterion = MaskedFocalLossWithLogits(
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
            enable_class_weights=args.enable_pixel_class_weights,
            max_pos_weight=args.max_fire_class_weight,
        )
        print(
            f"\n[Loss] Using MaskedFocalLossWithLogits (Focal): "
            f"alpha={args.focal_alpha}, gamma={args.focal_gamma}"
        )
    else:  # focal_tversky
        criterion = MaskedFocalTverskyLossWithLogits(
            alpha=args.tversky_alpha,
            beta=args.tversky_beta,
            gamma=args.tversky_gamma,
        )
        print(
            f"\n[Loss] Using MaskedFocalTverskyLossWithLogits (Focal Tversky): "
            f"alpha={args.tversky_alpha}, beta={args.tversky_beta}, gamma={args.tversky_gamma}"
        )
        if args.enable_pixel_class_weights:
            print(
                "[Loss] NOTE: --enable-pixel-class-weights is ignored for focal_tversky; "
                "use --tversky-alpha/--tversky-beta/--tversky-gamma instead."
            )

    criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TensorBoard
    writer: Optional[SummaryWriter] = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=args.logdir)
        print(f"[TB] Logging to: {args.logdir}")

    metrics_csv_path = init_metrics_csv(args.logdir)

    # Optional patch stats for patch-level sampling
    patch_flags = None
    if args.patch_sampling != "none":
        patch_flags = compute_patch_fire_flags(train_ds, args, device)

    print_ram_delta(start_ram, "After model + loss + optimizer + patch stats")

    # Simple loaders for val/test (no curriculum)
    val_loader = None
    test_loader = None
    if val_ds is not None:
        val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False, args=args)
    if test_ds is not None:
        test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False, args=args)

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

    # Optionally just run validation and exit
    if args.val_test_only:
        print("[Mode] VAL-TEST-ONLY: running eval on val/test then exiting.")
        if val_loader is not None:
            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                device,
                criterion,
                args,
                use_tqdm=not args.no_tqdm,
                show_worker_stats=args.show_worker_stats,
            )
            print_diagnostic_header("Val-only metrics")
            print(
                f"[Val] loss={val_loss:.6f}, "
                f"acc={val_metrics['accuracy']:.4f}, "
                f"prec={val_metrics['precision']:.4f}, "
                f"rec={val_metrics['recall']:.4f}, "
                f"f1={val_metrics['f1']:.4f}"
            )
        if test_loader is not None:
            test_loss, test_metrics = evaluate(
                model,
                test_loader,
                device,
                criterion,
                args,
                use_tqdm=not args.no_tqdm,
                show_worker_stats=args.show_worker_stats,
            )
            print_diagnostic_header("Test-only metrics")
            print(
                f"[Test] loss={test_loss:.6f}, "
                f"acc={test_metrics['accuracy']:.4f}, "
                f"prec={test_metrics['precision']:.4f}, "
                f"rec={test_metrics['recall']:.4f}, "
                f"f1={test_metrics['f1']:.4f}"
            )
        if monitor is not None:
            monitor.stop()
        return

    if args.dry_run:
        print("[Mode] DRY-RUN: initialization done; exiting before training.")
        if monitor is not None:
            monitor.stop()
        return

    # ------------------------------------------------------------------
    # Training epochs
    # ------------------------------------------------------------------
    print_diagnostic_header("Training")

    best_val_f1 = -1.0
    best_epoch = -1
    global_step = 0
    last_val_metrics = None
    final_test_metrics = None

    # In-memory history for plotting
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
        # Train
        train_loss, global_step = train_one_epoch(
            model,
            train_ds,
            epoch,
            optimizer,
            criterion,
            device,
            args,
            monitor,
            writer,
            global_step,
            patch_flags,
        )

        # Record train history
        history["train"]["epoch"].append(epoch)
        history["train"]["loss"].append(float(train_loss))

        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)

        # Validation
        if val_loader is not None:
            if monitor is not None:
                monitor.set_status(f"Eval e{epoch}/{args.epochs}")
            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                device,
                criterion,
                args,
                use_tqdm=not args.no_tqdm,
                show_worker_stats=args.show_worker_stats,
            )
        
            save_and_log_calibration("val", val_metrics, epoch, args, writer)
        
            print_diagnostic_header(f"Validation (epoch {epoch})")
            print(
                f"[Val] loss={val_loss:.6f}, "
                f"acc={val_metrics['accuracy']:.4f}, "
                f"prec={val_metrics['precision']:.4f}, "
                f"rec={val_metrics['recall']:.4f}, "
                f"f1={val_metrics['f1']:.4f}, "
                f"ece={val_metrics['ece']:.4f}, "
                f"brier={val_metrics['brier']:.4f}"
            )
            for h, m in val_metrics["per_horizon"].items():
                print(
                    f"  [Val h={h}] acc={m['accuracy']:.4f}, "
                    f"prec={m['precision']:.4f}, rec={m['recall']:.4f}, f1={m['f1']:.4f}, "
                    f"tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, tn={m['tn']}"
                )

            append_metrics_csv(metrics_csv_path, epoch, "val", val_loss, val_metrics)

            # Record val history
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
                    {
                        "epoch": [],
                        "accuracy": [],
                        "precision": [],
                        "recall": [],
                        "f1": [],
                        "logloss": [],
                        "roc_auc": [],
                    },
                )
                per_h["epoch"].append(epoch)
                per_h["accuracy"].append(float(m["accuracy"]))
                per_h["precision"].append(float(m["precision"]))
                per_h["recall"].append(float(m["recall"]))
                per_h["f1"].append(float(m["f1"]))
                per_h["logloss"].append(float(m.get("logloss", float("nan"))))
                per_h["roc_auc"].append(float(m.get("roc_auc", float("nan"))))

            # TensorBoard logging
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

            # Track best F1
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
                    },
                    best_path,
                )
                print(f"[Model] New best F1={best_val_f1:.4f} at epoch {epoch} -> saved to {best_path}")
                if writer is not None:
                    writer.add_scalar("val/best_f1", best_val_f1, epoch)

            # Keep last val metrics around for ROC plots later
            last_val_metrics = val_metrics


        # Optionally test at end of training
        if epoch == args.epochs and test_loader is not None:
            if monitor is not None:
                monitor.set_status(f"Test after epoch {epoch}")
            test_loss, test_metrics = evaluate(
                model,
                test_loader,
                device,
                criterion,
                args,
                use_tqdm=not args.no_tqdm,
                show_worker_stats=args.show_worker_stats,
            )
            save_and_log_calibration("test", test_metrics, epoch, args, writer)

            print_diagnostic_header(f"Test (epoch {epoch})")
            print(
                f"[Test] loss={test_loss:.6f}, "
                f"acc={test_metrics['accuracy']:.4f}, "
                f"prec={test_metrics['precision']:.4f}, "
                f"rec={test_metrics['recall']:.4f}, "
                f"f1={test_metrics['f1']:.4f}, "
                f"logloss={test_metrics.get('logloss', float('nan')):.6f}, "
                f"auc={test_metrics.get('roc_auc', float('nan')):.4f}, "
                f"ece={test_metrics['ece']:.4f}, "
                f"brier={test_metrics['brier']:.4f}"
            )
            for h, m in test_metrics["per_horizon"].items():
                print(
                    f"  [Test h={h}] acc={m['accuracy']:.4f}, "
                    f"prec={m['precision']:.4f}, rec={m['recall']:.4f}, "
                    f"f1={m['f1']:.4f}, auc={m.get('roc_auc', float('nan')):.4f}, "
                    f"tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, tn={m['tn']}"
                )

            append_metrics_csv(metrics_csv_path, epoch, "test", test_loss, test_metrics)
            final_test_metrics = test_metrics

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    # Save history JSON and plots (if requested)
    try:
        save_history_json(history, args.logdir)
        if args.plot_metrics:
            plot_metric_history(history, args)
            # ROC curves for last val and final test (if available)
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
