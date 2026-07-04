#!/usr/bin/env python3
"""
Spatiotemporal Peat Ignition — Heavily Optimized for Speed & Memory
WITH COMPREHENSIVE REAL-TIME MONITORING

Now using JointPeatDataset (ERA5 + SMAP -> VIIRS Zarr-backed patches).

Major features (unchanged):
- Persistent tqdm bars showing RAM/GPU usage at all times
- Real-time tracking of worker data loading time
- Forward and backward pass timing displayed continuously
- Throughput metrics (samples/sec, updates/sec)
- Detailed bottleneck identification

New:
- JointPeatDataset-based input pipeline (x, y, mask, meta)
- Transformer-based PatchForecast model with SH+SIREN positional encoding
- Hard-stop NaN debugger (--nan-debug) to locate non-finite values

This version also includes:
- Storage type detection (SSD vs HDD vs network FS) for Zarr roots
- Explicit multiprocessing context reporting (forkserver/spawn/etc.)
- Per-batch decompression time from dataset (Zarr IO) vs GPU transfer time
- DataLoader queue tracker: how many batches are ready in RAM vs currently loading
- Validation metrics after each epoch: accuracy, precision, recall, F1, etc.
- Optional focal loss for severe class imbalance
- Per-horizon validation metrics (per forecast horizon)
- CSV logging of metrics per epoch (metrics_log.csv, TensorBoard CSV-friendly)
- TensorBoard logging for:
    * Per-step loss, timings, throughput
    * RAM, GPU memory, loader queue stats
    * Epoch losses & validation metrics (overall + per horizon)
    * “Best so far” validation metrics
"""

from __future__ import annotations
import argparse
import contextlib
import math
import os
import sys
import time
import psutil
import threading
from collections import deque
from typing import Sequence, Optional, Dict, Any
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # NEW

from joint_peat_dataset_builder import JointPeatDataset  # <- your new dataset

import warnings
from scipy.special import sph_harm  # where you already import it

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="`scipy.special.sph_harm` is deprecated*",
)


# ----------------------------------------------------------------------
# Small utils
# ----------------------------------------------------------------------

def exists(x): return x is not None
def default(val, d): return d if val is None else val


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def human_int(n: int) -> str:
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


def get_ram_usage():
    """Get current RAM usage in GiB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


def get_gpu_memory(device_id: int = 0):
    """Get GPU memory allocated (GiB) as tracked by PyTorch"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device_id) / (1024 ** 3)
    return 0.0


def print_ram_delta(start_ram, label):
    """Print RAM usage delta"""
    current_ram = get_ram_usage()
    delta = current_ram - start_ram
    print(f"[RAM] {label}: {delta:+.2f} GB (total: {current_ram:.2f} GB)")
    return current_ram


def print_diagnostic_header(title):
    """Print a diagnostic section header"""
    width = 60
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_diagnostic_item(label, value, indent=0):
    """Print a diagnostic item with consistent formatting"""
    prefix = "  " * indent + "• "
    print(f"{prefix}{label:.<40} {value}")


class DiagnosticTimer:
    """Context manager for timing operations with diagnostics"""
    def __init__(self, label, verbose=True, track_ram=False, track_gpu=False, device_index: Optional[int] = None):
        self.label = label
        self.verbose = verbose
        self.track_ram = track_ram
        self.track_gpu = track_gpu
        self.device_index = 0 if device_index is None else device_index
        self.start_time = None
        self.start_ram = None
        self.start_gpu = None

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
        elapsed = time.time() - self.start_time
        if self.verbose:
            info = f"✓ Completed in {elapsed:.2f}s"
            if self.track_ram:
                ram_delta = get_ram_usage() - self.start_ram
                info += f" | RAM: {ram_delta:+.2f} GB"
            if self.track_gpu and torch.cuda.is_available():
                gpu_delta = get_gpu_memory(self.device_index) - (self.start_gpu or 0.0)
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
        mount_point = None
        best_len = -1
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                dev, mnt, fs = parts[0], parts[1], parts[2]
                if path.startswith(mnt) and len(mnt) > best_len:
                    best_len = len(mnt)
                    device, mount_point, fstype = dev, mnt, fs

        if device is None:
            return "Unknown (no /proc/mounts entry matched)"

        if fstype in ("nfs", "nfs4", "lustre", "cifs", "smb3", "gpfs"):
            return f"Network / parallel FS ({fstype}, device={device})"

        if device.startswith("/dev/"):
            base = os.path.basename(device)
            # Strip partition suffix (sda1 -> sda, nvme0n1p1 -> nvme0n1)
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

    except Exception as e:
        return f"Unknown (error probing storage: {e})"


# ----------------------------------------------------------------------
# SIREN + Spherical Harmonics positional encoder + Transformer model
# ----------------------------------------------------------------------

class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Linear):
    """
    One SIREN layer as in Sitzmann et al. (2020).
    Uses sine activation with special initialization.
    """
    def __init__(self, in_features, out_features, w0=1.0, is_first=False, c=6.0, bias=True):
        # IMPORTANT: set these BEFORE calling super().__init__
        self.in_features = in_features
        self.w0 = w0
        self.is_first = is_first
        self.c = c

        # This will call self.reset_parameters() once,
        # which now works because the fields above already exist.
        super().__init__(in_features, out_features, bias=bias)

        # Optional: you can leave this or remove it.
        # It just runs our custom init a second time, which is harmless.
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: U(-1/in, 1/in)
                bound = 1.0 / self.in_features
            else:
                # Subsequent layers: U(-sqrt(c/in)/w0, sqrt(c/in)/w0)
                bound = math.sqrt(self.c / self.in_features) / self.w0
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * F.linear(x, self.weight, self.bias))


class SirenNet(nn.Module):
    """
    Small SIREN MLP: in_features -> out_features
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 128,
        hidden_layers: int = 2,
        w0: float = 30.0,
        w0_initial: float = 30.0,
        c: float = 6.0,
    ):
        super().__init__()

        layers = []
        # First layer
        layers.append(SirenLayer(in_features, hidden_features,
                                 w0=w0_initial, is_first=True, c=c))
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(SirenLayer(hidden_features, hidden_features,
                                     w0=w0, is_first=False, c=c))

        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_features, out_features)

        # Optional: smaller init for last layer
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / w0
            self.final_linear.weight.uniform_(-bound, bound)
            if self.final_linear.bias is not None:
                self.final_linear.bias.zero_()

    def forward(self, x):
        # x: (..., in_features)
        y = self.net(x)
        return self.final_linear(y)


class SphericalHarmonicsSirenPosEncoder(nn.Module):
    """
    Geographic location encoder: PE(lat, lon) -> d_model.

    lat, lon are on the sphere (Earth):
      - lat in radians, range [-pi/2, pi/2]
      - lon in radians, range [-pi, pi]
    """
    def __init__(
        self,
        L_max: int,
        d_model: int,
        siren_hidden: int = 128,
        siren_layers: int = 2,
        use_imag: bool = False,
    ):
        super().__init__()
        # Lazy import so script still imports even if SciPy missing
        try:
            from scipy.special import sph_harm  # noqa: F401
        except Exception as e:
            raise ImportError(
                "SphericalHarmonicsSirenPosEncoder requires SciPy. "
                "Install with `pip install scipy`."
            ) from e

        self.L_max = L_max
        self.use_imag = use_imag

        # Number of SH features: sum_{l=0..L} (2l+1) = (L+1)^2
        sh_dim = (L_max + 1) ** 2
        if use_imag:
            sh_dim *= 2  # real + imag

        self.siren = SirenNet(
            in_features=sh_dim,
            out_features=d_model,
            hidden_features=siren_hidden,
            hidden_layers=siren_layers,
            w0=30.0,
            w0_initial=30.0,
        )

    def forward(self, lat_rad: torch.Tensor, lon_rad: torch.Tensor) -> torch.Tensor:
        """
        lat_rad, lon_rad: (B, H, W) in radians.

        Returns:
            pos_embed: (B, H*W, d_model)
        """
        from scipy.special import sph_harm

        assert lat_rad.shape == lon_rad.shape
        B, H, W = lat_rad.shape
        device = lat_rad.device

        # Move to CPU numpy for SciPy
        lat_np = lat_rad.detach().cpu().numpy()
        lon_np = lon_rad.detach().cpu().numpy()

        # SciPy sph_harm uses sph_harm(m, l, phi, theta):
        #   phi   = azimuth [0, 2π]   (longitude)
        #   theta = colatitude [0, π] (π/2 - latitude)
        phi = lon_np
        theta = 0.5 * math.pi - lat_np

        feats = []
        for l in range(self.L_max + 1):
            for m in range(-l, l + 1):
                Y_lm = sph_harm(m, l, phi, theta)  # complex (B,H,W)
                if self.use_imag:
                    feats.append(Y_lm.real)
                    feats.append(Y_lm.imag)
                else:
                    feats.append(Y_lm.real)

        # (B,H,W,F_sh)
        sh_feats = np.stack(feats, axis=-1).astype("float32")
        # (B, H*W, F_sh)
        sh_feats = torch.from_numpy(sh_feats).to(device=device)
        sh_feats = sh_feats.view(B, H * W, -1)

        # SIREN -> (B, H*W, d_model)
        pos_embed = self.siren(sh_feats)
        return pos_embed


class PatchTransformerSHSirenForecast(nn.Module):
    """
    Transformer-based *temporal* patch forecaster with SH+SIREN geographic encoding.

    - For each spatial location (tile / pixel), we form a sequence of length T_hist
      with C_total features (ERA5+SMAP).
    - The transformer runs along the *time dimension* for each tile.
    - We process tiles in CHUNKS to avoid insane batch sizes and CUDA launch issues.
    """

    def __init__(
        self,
        in_channels_per_timestep: int,
        t_hist: int,
        horizons: Sequence[int],
        d_model: int = 256,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        L_max: int = 10,
        use_sh_siren: bool = True,
        max_tile_batch: int = 4096,   # NEW: max number of tiles per transformer call
    ):
        super().__init__()
        self.in_channels = int(in_channels_per_timestep)   # per timestep
        self.t_hist = int(t_hist)
        self.horizons = [int(h) for h in horizons]
        self.k = len(self.horizons)
        self.d_model = d_model
        self.use_sh_siren = use_sh_siren
        self.max_tile_batch = int(max_tile_batch)

        # Project per-timestep features to d_model
        self.input_proj = nn.Linear(self.in_channels, d_model)

        # Transformer encoder over *time* tokens (sequence length = T_hist)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, T_hist, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Temporal positional encoding (learned) for time steps 0..T_hist-1
        self.time_pos = nn.Parameter(torch.zeros(1, self.t_hist, d_model))
        nn.init.trunc_normal_(self.time_pos, std=0.02)

        # Geographic encoding (one embedding per spatial location)
        if use_sh_siren:
            self.pos_encoder = SphericalHarmonicsSirenPosEncoder(
                L_max=L_max,
                d_model=d_model,
                siren_hidden=128,
                siren_layers=2,
                use_imag=False,
            )
        else:
            self.pos_encoder = None

        # Classification head from sequence summary -> K horizons
        self.head = nn.Linear(d_model, self.k)

    def forward(self, batch: Dict[str, Any]):
        x = batch["x"]  # expected (B, T_hist, C, H, W) with stack-time='separate'
        device = x.device

        if x.dim() == 5:
            B, T, C, H, W = x.shape
            if T != self.t_hist:
                raise ValueError(
                    f"Expected T_hist={self.t_hist} time steps, but got T={T} "
                    f"in x.shape={tuple(x.shape)}"
                )
            if C != self.in_channels:
                raise ValueError(
                    f"in_channels_per_timestep mismatch: model expects {self.in_channels}, got {C}"
                )
        elif x.dim() == 4:
            # Fallback: single timestep (T=1)
            B, C, H, W = x.shape
            T = 1
            if self.t_hist != 1:
                raise ValueError(
                    f"Model configured for T_hist={self.t_hist}, but got 4D x with implicit T=1. "
                    f"Use stack-time='separate' so x is (B, T_hist, C, H, W)."
                )
            x = x.unsqueeze(1)  # (B,1,C,H,W)
        else:
            raise ValueError(
                f"PatchTransformerSHSirenForecast expects x to have 4 or 5 dims, got {tuple(x.shape)}"
            )

        # x: (B, T, C, H, W) -> (B, H, W, T, C)
        x = x.permute(0, 3, 4, 1, 2)  # (B, H, W, T, C)
        N_seq = B * H * W

        # Flatten spatial dims into "tile sequences": (N_seq, T, C)
        seqs = x.reshape(N_seq, T, C)

        # Geographic embedding for each tile (computed once)
        if self.use_sh_siren:
            if "coords" not in batch:
                raise KeyError(
                    "batch['coords'] is required for SH+SIREN positional encoding. "
                    "Expected shape (B, 2, H, W) with [lat_rad, lon_rad]."
                )
            coords = batch["coords"].to(device)  # (B, 2, H, W)
            if coords.shape[0] != B or coords.shape[2] != H or coords.shape[3] != W:
                raise ValueError(
                    f"coords shape {coords.shape} inconsistent with x spatial shape (B={B}, H={H}, W={W})"
                )
            lat = coords[:, 0]  # (B, H, W)
            lon = coords[:, 1]  # (B, H, W)

            # (B, H*W, d_model)
            geo_embed = self.pos_encoder(lat, lon)  # geographic embedding per tile
            # Flatten to (N_seq, d_model)
            geo_embed = geo_embed.view(N_seq, self.d_model)
        else:
            geo_embed = None

        # Process tiles in chunks to avoid huge effective batch sizes
        max_tb = self.max_tile_batch
        reps = []

        for start in range(0, N_seq, max_tb):
            end = min(start + max_tb, N_seq)

            seq_chunk = seqs[start:end]          # (tb, T, C)
            tokens = self.input_proj(seq_chunk)  # (tb, T, d_model)

            # Add temporal positional encoding
            if T != self.t_hist:
                raise ValueError(
                    f"Internal mismatch: T={T}, t_hist={self.t_hist}. This should not happen."
                )
            tokens = tokens + self.time_pos      # broadcast (1, T, d_model) -> (tb, T, d_model)

            # Add geographic embedding (per tile), broadcast along time
            if geo_embed is not None:
                geo_chunk = geo_embed[start:end]          # (tb, d_model)
                geo_chunk = geo_chunk.unsqueeze(1)        # (tb, 1, d_model)
                tokens = tokens + geo_chunk               # (tb, T, d_model)

            # Transformer over time for this chunk
            enc_chunk = self.encoder(tokens)              # (tb, T, d_model)

            # Last time-step representation
            last_repr = enc_chunk[:, -1, :]               # (tb, d_model)
            reps.append(last_repr)

        # Concatenate all tile representations back: (N_seq, d_model)
        reps_all = torch.cat(reps, dim=0)

        # Map to horizons: (N_seq, K)
        logits_seq = self.head(reps_all)

        # Reshape back to (B, K, H, W)
        logits = logits_seq.view(B, H, W, self.k).permute(0, 3, 1, 2)
        return logits



# ----------------------------------------------------------------------
# Loss & monitoring utilities
# ----------------------------------------------------------------------

class MaskedBCEWithLogits(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask):
        # logits, targets, mask: (B, K, H, W)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss


class MaskedFocalLossWithLogits(nn.Module):
    """
    Mask-aware focal loss for heavily imbalanced binary labels.

    logits, targets, mask: (B, K, H, W)
      - targets: 0/1 floats
      - mask   : 0/1 floats where 1 = valid label
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets, mask):
        # Sigmoid probabilities
        probs = torch.sigmoid(logits)
        t = (targets > 0.5).float()

        # p_t = p if y=1 else 1-p
        p_t = probs * t + (1.0 - probs) * (1.0 - t)

        # alpha_t = alpha for positives, 1-alpha for negatives
        alpha_t = self.alpha * t + (1.0 - self.alpha) * (1.0 - t)

        eps = 1e-6
        focal_term = (1.0 - p_t).pow(self.gamma)
        loss = -alpha_t * focal_term * torch.log(p_t.clamp(min=eps))

        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss


class RAMMonitor:
    """Background thread that monitors RAM and GPU usage"""
    def __init__(self, device_id=0, update_interval=0.5):
        self.device_id = device_id
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.pbar = None
        self.current_ram = 0.0
        self.current_gpu = 0.0
        self.current_status = "Initializing..."
        # NEW: DataLoader queue stats
        self.queue_capacity: Optional[int] = None
        self.batches_ready: Optional[int] = None
        self.batches_loading: Optional[int] = None

    def start(self):
        """Start the monitoring thread"""
        self.running = True
        self.pbar = tqdm(total=0, position=0, bar_format='{desc}', leave=True)
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.pbar:
            self.pbar.close()

    def set_status(self, status):
        """Update current activity status"""
        self.current_status = status

    def set_loader_queue(self, ready: Optional[int], loading: Optional[int], capacity: Optional[int]):
        """
        Update DataLoader queue statistics.

        ready   = batches fully prepared and waiting in the DataLoader result queue
        loading = batches with outstanding load/transform work (approximate)
        capacity = theoretical max number of prefetched batches (workers * prefetch_factor)
        """
        self.batches_ready = ready
        self.batches_loading = loading
        self.queue_capacity = capacity

    def _monitor_loop(self):
        """Background loop that updates RAM/GPU stats"""
        while self.running:
            self.current_ram = get_ram_usage()
            self.current_gpu = get_gpu_memory(self.device_id) if torch.cuda.is_available() else 0.0

            desc = f"💾 RAM: {self.current_ram:.2f}GB"
            if torch.cuda.is_available():
                desc += f" | 🎮 GPU: {self.current_gpu:.2f}GB"

            # NEW: DataLoader queue info
            if self.queue_capacity:
                if self.batches_ready is not None and self.batches_loading is not None:
                    desc += f" | 📦 Queue: {self.batches_ready}/{self.queue_capacity} ready, {self.batches_loading} loading"
                else:
                    desc += f" | 📦 Queue: capacity {self.queue_capacity}"

            desc += f" | {self.current_status}"

            if self.pbar:
                self.pbar.set_description_str(desc)

            time.sleep(self.update_interval)


class TimedDataLoader:
    """Wrapper that tracks batch loading times with detailed metrics"""
    def __init__(self, loader, desc="batch", use_tqdm=True, show_worker_stats=False, position=1):
        self.loader = loader
        self.desc = desc
        self.use_tqdm = use_tqdm
        self.show_worker_stats = show_worker_stats
        self.position = position
        self.batch_times = deque(maxlen=100)  # Rolling window
        self.last_batch_time = 0.0

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
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                           '[{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )

        batch_idx = 0
        while True:
            try:
                t_start = time.time()
                batch = next(iterator)
                t_elapsed = time.time() - t_start
                self.batch_times.append(t_elapsed)
                self.last_batch_time = t_elapsed

                batch_idx += 1

                if self.use_tqdm:
                    avg_time = sum(self.batch_times) / len(self.batch_times)
                    postfix = {
                        '⏱️load': f'{t_elapsed*1000:.0f}ms',
                        'avg': f'{avg_time*1000:.0f}ms'
                    }
                    if self.show_worker_stats and len(self.batch_times) >= 10:
                        recent = list(self.batch_times)[-10:]
                        postfix['min'] = f'{min(recent)*1000:.0f}ms'
                        postfix['max'] = f'{max(recent)*1000:.0f}ms'
                    self.pbar.set_postfix(postfix)
                    self.pbar.update(1)

                yield batch

            except StopIteration:
                break

        if self.use_tqdm:
            self.pbar.close()

    def __len__(self):
        return len(self.loader)


def _choose_mp_context(requested: str) -> Optional[str]:
    """Pick a multiprocessing context that plays nicely with CUDA."""
    if requested != "auto":
        return requested
    if sys.platform.startswith("linux"):
        return "forkserver"  # safer with CUDA than 'fork'
    return "spawn"


def collate(batch):
    """
    Optimized collate with pre-allocation & key integrity check.
    For JointPeatDataset:
        - Tensors ("x", "y", "mask") are stacked.
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
            # e.g. meta dicts
            out[k] = vals
    return out


def make_loader(ds, batch_size, shuffle, args):
    """Optimized DataLoader with better worker settings"""
    pin = torch.cuda.is_available()  # pin memory only when CUDA is present
    kw: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=pin,
        collate_fn=collate,
        persistent_workers=(args.workers > 0),
    )
    if args.workers > 0:
        kw["prefetch_factor"] = args.prefetch
        ctx = _choose_mp_context(args.mp_context)
        if ctx:
            kw["multiprocessing_context"] = ctx
            # NEW: make MP context visible
            if getattr(args, "verbose_loader", False):
                print(f"[DataLoader] Using multiprocessing context '{ctx}' "
                      f"with {args.workers} workers and prefetch_factor={args.prefetch}.")
    return DataLoader(ds, **kw)


def wrap_loader(loader, desc: str, use_tqdm: bool, show_worker_stats: bool = False, position: int = 1):
    """Wrap loader with timing diagnostics"""
    return TimedDataLoader(loader, desc=desc, use_tqdm=use_tqdm, show_worker_stats=show_worker_stats, position=position)


# ----------------------------------------------------------------------
# Evaluation, metrics, dataset scan
# ----------------------------------------------------------------------

def _compute_basic_metrics(tp, fp, fn, tn):
    support = tp + fp + fn + tn
    if support == 0:
        return {
            "accuracy": float('nan'),
            "precision": float('nan'),
            "recall": float('nan'),
            "f1": float('nan'),
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


@torch.no_grad()
def evaluate(model, loader, device, criterion, args, use_tqdm: bool = True, show_worker_stats: bool = False):
    """
    Run validation over `loader` and return:
        val_loss, metrics_dict

    metrics_dict contains:
        - accuracy, precision, recall, f1, tp, fp, fn, tn, support (overall)
        - per_horizon: { horizon_value: {accuracy, precision, recall, f1, tp, fp, fn, tn, support} }
    """
    model.eval()
    tot_loss, tot_mask = 0.0, 0.0
    tp_total = fp_total = fn_total = tn_total = 0

    # Per-horizon accumulators
    num_horizons = len(args.horizons)
    tp_h = [0 for _ in range(num_horizons)]
    fp_h = [0 for _ in range(num_horizons)]
    fn_h = [0 for _ in range(num_horizons)]
    tn_h = [0 for _ in range(num_horizons)]

    iterator = wrap_loader(loader, desc="val", use_tqdm=use_tqdm, show_worker_stats=show_worker_stats, position=2)
    for batch in iterator:
        # Move only tensors to device, keep meta on CPU
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        logits = model(batch)
        loss = criterion(logits, batch["y"], batch["mask"])
        m = batch["mask"].sum().item()
        if m == 0:
            continue  # skip all-masked batches
        tot_loss += loss.item() * m
        tot_mask += m

        # ---- Metrics: accuracy, precision, recall, F1 (binary, heavily imbalanced) ----
        probs = torch.sigmoid(logits)
        preds = probs >= args.metrics_threshold
        targets = batch["y"] > 0.5
        valid = batch["mask"] > 0.5

        # Overall stats across all horizons
        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)
        valid_flat  = valid.reshape(-1)

        tp_total += (preds_flat & targets_flat & valid_flat).sum().item()
        fp_total += (preds_flat & ~targets_flat & valid_flat).sum().item()
        fn_total += (~preds_flat & targets_flat & valid_flat).sum().item()
        tn_total += (~preds_flat & ~targets_flat & valid_flat).sum().item()

        # Per-horizon stats
        B, K, H, W = logits.shape
        K_eff = min(K, num_horizons)
        for k_idx in range(K_eff):
            p_k = preds[:, k_idx]    # (B,H,W)
            t_k = targets[:, k_idx]
            v_k = valid[:, k_idx]

            p_flat = p_k.reshape(-1)
            t_flat = t_k.reshape(-1)
            v_flat = v_k.reshape(-1)

            tp_h[k_idx] += (p_flat & t_flat & v_flat).sum().item()
            fp_h[k_idx] += (p_flat & ~t_flat & v_flat).sum().item()
            fn_h[k_idx] += (~p_flat & t_flat & v_flat).sum().item()
            tn_h[k_idx] += (~p_flat & ~t_flat & v_flat).sum().item()

    if tot_mask == 0:
        print("[WARN] Validation mask sum was zero across all batches; reporting NaN metrics.")
        val_loss = float('nan')
    else:
        val_loss = tot_loss / tot_mask

    # Overall metrics
    overall = _compute_basic_metrics(tp_total, fp_total, fn_total, tn_total)

    # Per-horizon metrics mapped by horizon value
    per_horizon_metrics: Dict[int, Dict[str, float]] = {}
    for idx, h in enumerate(args.horizons):
        per_horizon_metrics[h] = _compute_basic_metrics(tp_h[idx], fp_h[idx], fn_h[idx], tn_h[idx])

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
    }

    return val_loss, metrics



def parse_args():
    p = argparse.ArgumentParser()

    # --- Zarr data roots ---
    p.add_argument("--era5-zarr", required=True)
    p.add_argument("--smap-zarr", required=True)
    p.add_argument("--viirs-zarr", required=True)

    p.add_argument("--era5-array", default="field")
    p.add_argument("--smap-array", default="field")
    p.add_argument("--viirs-array", default="field")

    # --- Dataset hyperparameters ---
    p.add_argument("--T-hist", type=int, default=30)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 7, 14])
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--stack-time", choices=["separate", "channel"], default="separate")
    p.add_argument("--split", type=float, default=0.9, help="Train/val split fraction")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--normalize-inputs", choices=["none", "per_channel"], default="none")
    p.add_argument("--no-skip-nonpeat", action="store_true", help="Disable peat-based patch filtering")
    p.add_argument("--peat-min-fraction", type=float, default=0.01)

    # --- Training HPs ---
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # --- Loss / imbalance handling ---
    p.add_argument("--loss", choices=["bce", "focal"], default="bce",
                   help="Loss function: 'bce' (MaskedBCEWithLogits) or 'focal' (MaskedFocalLossWithLogits)")
    p.add_argument("--focal-alpha", type=float, default=0.25,
                   help="Alpha for focal loss (weight on positive class).")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Gamma for focal loss (focusing parameter).")

    # --- Metrics ---
    p.add_argument("--metrics-threshold", type=float, default=0.5,
                   help="Decision threshold on sigmoid(logits) for metrics (precision/recall/F1/accuracy).")

    # --- Optimization flags ---
    p.add_argument("--compile", action="store_true", help="Use torch.compile() for speedup")
    p.add_argument("--prefetch", type=int, default=2, help="Dataloader prefetch factor")
    p.add_argument("--mp-context", choices=["auto", "spawn", "fork", "forkserver"], default="auto",
                   help="Multiprocessing start method for DataLoader workers")

    # NEW: Worker diagnostics
    p.add_argument("--show-worker-stats", action="store_true", help="Show detailed worker timing stats")

    # Verbosity and diagnostics
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument("--log-interval", type=int, default=0)
    p.add_argument("--tqdm-scan", action="store_true")
    p.add_argument("--scan-limit", type=int, default=0)
    p.add_argument("--scan-batch-size", type=int, default=32)
    p.add_argument("--scan-to-device", action="store_true", help="Transfer scan batches to device (default: CPU only)")

    # Diagnostic flags
    p.add_argument("--debug", action="store_true", help="Maximum verbosity everywhere")
    p.add_argument("--verbose-dataset", action="store_true", help="Detailed dataset logging")
    p.add_argument("--verbose-loader", action="store_true", help="Detailed dataloader logging")
    p.add_argument("--verbose-model", action="store_true", help="Detailed model logging")
    p.add_argument("--profile-first-epoch", action="store_true", help="Profile first epoch in detail")

    # Testing and debugging
    p.add_argument("--skip-val-dataset", action="store_true", help="Skip val dataset initialization")
    p.add_argument("--quick-test", action="store_true", help="Quick test: 1 epoch, small batches")
    p.add_argument("--dry-run", action="store_true", help="Initialize everything but don't train")
    p.add_argument("--sample-one-batch", action="store_true", help="Load just 1 batch and exit")
    p.add_argument("--limit-train-samples", type=int, default=0, help="(Optional) extra train sample limit")
    p.add_argument("--val-test-only", action="store_true", help="Run a validation pass only (no training) to test the validation loop, then exit."
    )
    # Pause points
    p.add_argument("--pause-after-dataset", action="store_true", help="Pause after dataset init")
    p.add_argument("--pause-after-model", action="store_true", help="Pause after model init")

    # NEW flags for this revision
    p.add_argument("--sync-every-step", action="store_true",
                   help="Force device sync after each phase for exact timings (slower)")
    p.add_argument("--measure-loader-time", action="store_true",
                   help="(Kept for compatibility; data loading is always timed directly now)")

    # TensorBoard logging
    p.add_argument("--logdir", default="runs/peat_transformer",
                   help="TensorBoard log directory")
    p.add_argument("--no-tensorboard", action="store_true",
                   help="Disable TensorBoard logging")

    return p.parse_args()


def dataset_scan(ds, name, args, device):
    if not args.tqdm_scan:
        return
    n = len(ds)
    if args.scan_limit and args.scan_limit > 0:
        n = min(n, args.scan_limit)
    if n <= 0:
        return
    print(f"\n[DIAG] Scanning {name} dataset ({n} samples)...")
    pin = torch.cuda.is_available()
    kw: Dict[str, Any] = dict(
        batch_size=args.scan_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
        collate_fn=collate,
        persistent_workers=(args.workers > 0),
    )
    if args.workers > 0:
        kw["prefetch_factor"] = args.prefetch
        ctx = _choose_mp_context(args.mp_context)
        if ctx:
            kw["multiprocessing_context"] = ctx
    loader = DataLoader(ds, **kw)
    total_batches = (n + args.scan_batch_size - 1) // args.scan_batch_size
    it = tqdm(loader, total=total_batches, desc=f"scan {name}", leave=False, disable=args.no_tqdm)
    seen = 0
    for batch in it:
        if args.scan_to_device:
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
        # else: keep on CPU to avoid GPU OOM during scan
        seen += batch["y"].shape[0]
        it.set_postfix(samples=min(seen, n))
        if seen >= n:
            break


def diagnose_model_architecture(model, args, train_ds):
    if not args.verbose_model:
        return
    print_diagnostic_header("Model Architecture Details")
    print("\n[DIAG] Testing forward pass with dummy input...")
    try:
        dev = next(model.parameters()).device
        # Dummy batch matching JointPeatDataset layout (stack-time='separate' style)
        dummy_x = torch.randn(2, args.T_hist, train_ds.C_total, args.patch, args.patch, device=dev)
        dummy_coords = torch.zeros(2, 2, args.patch, args.patch, device=dev)
        dummy_batch = {"x": dummy_x, "coords": dummy_coords}
        with torch.no_grad():
            output = model(dummy_batch)
        print_diagnostic_item("Dummy forward pass", f"✓ Output shape: {tuple(output.shape)}")
    except Exception as e:
        print_diagnostic_item("Dummy forward pass", f"✗ Failed: {e}")


def print_env_and_cfg(args, device, train_ds, val_ds, model):
    cuda_ok = torch.cuda.is_available()
    dtype_note = "bfloat16" if (cuda_ok and torch.cuda.is_bf16_supported()) else ("float16" if cuda_ok else "float32")
    p_total, p_train = count_params(model)

    print_diagnostic_header("Environment & Configuration")
    # Environment
    print("\n  Environment:")
    print_diagnostic_item("PyTorch version", torch.__version__, indent=1)
    print_diagnostic_item("Python version", sys.version.split()[0], indent=1)
    print_diagnostic_item("CUDA available", cuda_ok, indent=1)
    if cuda_ok:
        dev_idx = device.index if device.index is not None else 0
        print_diagnostic_item("GPU", torch.cuda.get_device_name(dev_idx), indent=1)
        cap = torch.cuda.get_device_capability(dev_idx)
        print_diagnostic_item("GPU capability", f"{cap[0]}.{cap[1]}", indent=1)
        total_mem = torch.cuda.get_device_properties(dev_idx).total_memory
        print_diagnostic_item("GPU memory", f"{total_mem / (1024**3):.2f} GiB", indent=1)
    print_diagnostic_item("AMP dtype", dtype_note, indent=1)
    print_diagnostic_item("torch.compile", "enabled" if args.compile else "disabled", indent=1)

    # Data
    print("\n  Data:")
    print_diagnostic_item("Train samples", len(train_ds), indent=1)
    print_diagnostic_item("Val samples", len(val_ds) if val_ds else "N/A", indent=1)
    print_diagnostic_item("T_hist", args.T_hist, indent=1)
    print_diagnostic_item("C_total (ERA5+SMAP)", train_ds.C_total, indent=1)
    print_diagnostic_item("Patch size", args.patch, indent=1)
    print_diagnostic_item("Horizons", args.horizons, indent=1)
    print_diagnostic_item("Time stacking", args.stack_time, indent=1)
    print_diagnostic_item("Peat filtering", f"{'on' if not args.no_skip_nonpeat else 'off'} (min frac={args.peat_min_fraction})", indent=1)

    # Model
    print("\n  Model:")
    print_diagnostic_item("Model class", model.__class__.__name__, indent=1)
    print_diagnostic_item("Input channels (per t)", model.in_channels, indent=1)
    print_diagnostic_item("T_hist", model.t_hist, indent=1)
    print_diagnostic_item("Output horizons", len(model.horizons), indent=1)
    print_diagnostic_item("Total params", human_int(p_total), indent=1)
    print_diagnostic_item("Trainable params", human_int(p_train), indent=1)

    # Hyperparameters
    print("\n  Hyperparameters:")
    print_diagnostic_item("Batch size", args.batch_size, indent=1)
    print_diagnostic_item("Grad accumulation", args.grad_accum, indent=1)
    print_diagnostic_item("Effective batch", args.batch_size * args.grad_accum, indent=1)
    print_diagnostic_item("Learning rate", args.lr, indent=1)
    print_diagnostic_item("Epochs", args.epochs, indent=1)
    print_diagnostic_item("Workers", args.workers, indent=1)
    print_diagnostic_item("Prefetch factor", args.prefetch, indent=1)


def test_first_batch(train_loader, device, model, args):
    """Diagnostic: Time and profile the first batch"""
    print_diagnostic_header("First Batch Diagnostic")
    print("[DIAG] Fetching first batch...")
    t_start = time.time()
    try:
        it = iter(train_loader)
        batch = next(it)
    except StopIteration:
        print_diagnostic_item("First batch", "✗ DataLoader was empty (check dataset size and batch_size)")
        return
    t_load = time.time() - t_start
    print_diagnostic_item("Batch load time", f"{t_load:.2f}s")

    # Analyze batch
    print("\n  Batch contents:")
    for k, v in batch.items():
        if torch.is_tensor(v):
            val = f"shape={tuple(v.shape)}, dtype={v.dtype}"
        else:
            val = f"type={type(v)}"
        print_diagnostic_item(k, val, indent=1)

    # Transfer to device
    print("\n[DIAG] Transferring to device...")
    t_start = time.time()
    batch = {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    if device.type == "cuda" and args.sync_every_step:
        torch.cuda.synchronize()
    t_transfer = time.time() - t_start
    print_diagnostic_item("Transfer time", f"{t_transfer:.2f}s")

    # Forward pass
    print("\n[DIAG] Running forward pass...")
    model.train()
    t_start = time.time()
    with amp_autocast(device):
        logits = model(batch)
    if device.type == "cuda" and args.sync_every_step:
        torch.cuda.synchronize()
    t_forward = time.time() - t_start
    print_diagnostic_item("Forward pass time", f"{t_forward:.2f}s")
    print_diagnostic_item("Output shape", tuple(logits.shape))

    # Backward pass
    print("\n[DIAG] Running backward pass...")
    criterion = MaskedBCEWithLogits()
    t_start = time.time()
    with amp_autocast(device):
        loss = criterion(logits, batch["y"], batch["mask"])
    loss.backward()
    if device.type == "cuda" and args.sync_every_step:
        torch.cuda.synchronize()
    t_backward = time.time() - t_start
    print_diagnostic_item("Backward pass time", f"{t_backward:.2f}s")
    print_diagnostic_item("Loss value", f"{loss.item():.4f}")

    # Memory summary
    if device.type == "cuda":
        print("\n  GPU Memory:")
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print_diagnostic_item("Allocated", f"{allocated:.2f} GB", indent=1)
        print_diagnostic_item("Reserved", f"{reserved:.2f} GB", indent=1)

    print("\n  Timing breakdown:")
    total = t_load + t_transfer + t_forward + t_backward
    total = max(total, 1e-9)  # guard
    print_diagnostic_item("Data loading", f"{t_load:.2f}s ({100 * t_load / total:.1f}%)", indent=1)
    print_diagnostic_item("Device transfer", f"{t_transfer:.2f}s ({100 * t_transfer / total:.1f}%)", indent=1)
    print_diagnostic_item("Forward pass", f"{t_forward:.2f}s ({100 * t_forward / total:.1f}%)", indent=1)
    print_diagnostic_item("Backward pass", f"{t_backward:.2f}s ({100 * t_backward / total:.1f}%)", indent=1)
    print_diagnostic_item("Total", f"{total:.2f}s", indent=1)


def diagnose_dataloader(train_loader, args):
    """Diagnostic: Show dataloader configuration"""
    if not args.verbose_loader:
        return
    print_diagnostic_header("DataLoader Configuration")
    print_diagnostic_item("Batch size", args.batch_size)
    print_diagnostic_item("Num workers", args.workers)
    print_diagnostic_item("Pin memory", torch.cuda.is_available())
    print_diagnostic_item("Prefetch factor", args.prefetch if args.workers > 0 else "N/A")
    print_diagnostic_item("Persistent workers", args.workers > 0)
    print_diagnostic_item("Total batches", len(train_loader))


class TrainingStepTimer:
    """Tracks detailed timing for training steps"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.data_times = []
        self.transfer_times = []
        self.forward_times = []
        self.backward_times = []
        self.optim_times = []
        self.total_times = []

    def add(self, data_time, transfer_time, forward_time, backward_time, optim_time):
        self.data_times.append(data_time)
        self.transfer_times.append(transfer_time)
        self.forward_times.append(forward_time)
        self.backward_times.append(backward_time)
        self.optim_times.append(optim_time)
        self.total_times.append(data_time + transfer_time + forward_time + backward_time + optim_time)

    def get_averages(self):
        if not self.total_times:
            return None
        n = len(self.total_times)
        return {
            'data': sum(self.data_times) / n,
            'transfer': sum(self.transfer_times) / n,
            'forward': sum(self.forward_times) / n,
            'backward': sum(self.backward_times) / n,
            'optim': sum(self.optim_times) / n,
            'total': sum(self.total_times) / n,
        }

    def print_summary(self, label="Training"):
        avgs = self.get_averages()
        if avgs is None:
            return
        print_diagnostic_header(f"{label} Step Timing Breakdown")
        total = avgs['total']
        for key in ['data', 'transfer', 'forward', 'backward', 'optim']:
            pct = 100 * avgs[key] / total if total > 0 else 0
            print_diagnostic_item(f"{key.capitalize()}", f"{avgs[key]:.3f}s ({pct:.1f}%)")
        print_diagnostic_item("Total per step", f"{total:.3f}s")
        if total > 0:
            print_diagnostic_item("Steps/sec", f"{1.0/total:.2f}")


def assert_finite_tensor(t: torch.Tensor, name: str, where: str = ""):
    """Raise with helpful info if a tensor contains NaN or Inf."""
    if t is None:
        return
    if not torch.is_tensor(t):
        return

    if not torch.isfinite(t).all():
        bad = ~torch.isfinite(t)
        # Grab the first bad index to help debugging
        idx = torch.nonzero(bad, as_tuple=False)
        idx_str = idx[0].tolist() if idx.numel() > 0 else "unknown"
        # Some quick stats
        t_min = torch.nanmin(t).item()
        t_max = torch.nanmax(t).item()
        print("\n[NaN/Inf DETECTED]")
        print(f"  Tensor: {name}")
        if where:
            print(f"  Where: {where}")
        print(f"  First bad index: {idx_str}")
        print(f"  min={t_min}, max={t_max}")
        raise RuntimeError(f"Non-finite values in tensor '{name}' ({where})")


def make_datasets(args):
    """Create JointPeatDataset train/val splits."""
    print_diagnostic_header("Dataset Initialization")
    print_diagnostic_item("ERA5 Zarr", args.era5_zarr)
    print_diagnostic_item("SMAP Zarr", args.smap_zarr)
    print_diagnostic_item("VIIRS Zarr", args.viirs_zarr)
    print_diagnostic_item("T_hist", args.T_hist)
    print_diagnostic_item("Horizons", args.horizons)
    print_diagnostic_item("Patch size", args.patch)
    print_diagnostic_item("Stride", args.stride)
    print_diagnostic_item("Split (train frac)", args.split)
    print_diagnostic_item("Normalize inputs", args.normalize_inputs)
    print_diagnostic_item("Skip non-peat patches", not args.no_skip_nonpeat)
    print_diagnostic_item("Peat min fraction", args.peat_min_fraction)

    norm_mode = None if args.normalize_inputs == "none" else "per_channel"
    skip_nonpeat = not args.no_skip_nonpeat

    # Train dataset
    print("\n[Dataset] Initializing train dataset...")
    with DiagnosticTimer("Train dataset initialization", track_ram=True):
        train_ds = JointPeatDataset(
            era5_zarr=args.era5_zarr,
            smap_zarr=args.smap_zarr,
            viirs_zarr=args.viirs_zarr,
            era5_array=args.era5_array,
            smap_array=args.smap_array,
            viirs_array=args.viirs_array,
            t_hist=args.T_hist,
            horizons=args.horizons,
            patch=args.patch,
            stride=args.stride,
            time_stack=args.stack_time,
            mode="train",
            split=args.split,
            seed=args.seed,
            normalize_inputs=norm_mode,
            max_samples=args.max_samples if args.max_samples else None,
            skip_nonpeat_patches=skip_nonpeat,
            peat_min_fraction=args.peat_min_fraction,
            time_index=None,
            return_coords=True
        )
    print_diagnostic_item("Train samples", len(train_ds))

    if args.skip_val_dataset:
        print("\n[Dataset] Skipping val dataset (--skip-val-dataset)")
        return train_ds, None

    # Validation dataset; reuse normalization from train if using per_channel
    print("\n[Dataset] Initializing val dataset...")
    with DiagnosticTimer("Val dataset initialization", track_ram=True):
        val_ds = JointPeatDataset(
            era5_zarr=args.era5_zarr,
            smap_zarr=args.smap_zarr,
            viirs_zarr=args.viirs_zarr,
            era5_array=args.era5_array,
            smap_array=args.smap_array,
            viirs_array=args.viirs_array,
            t_hist=args.T_hist,
            horizons=args.horizons,
            patch=args.patch,
            stride=args.stride,
            time_stack=args.stack_time,
            mode="val",
            split=args.split,
            seed=args.seed,
            normalize_inputs=None if norm_mode == "per_channel" else norm_mode,
            max_samples=args.max_samples if args.max_samples else None,
            skip_nonpeat_patches=skip_nonpeat,
            peat_min_fraction=args.peat_min_fraction,
            time_index=None,
            return_coords=True
        )

        if norm_mode == "per_channel":
            mean, std = train_ds.get_normalization()
            val_ds.set_normalization(mean, std)

    print_diagnostic_item("Val samples", len(val_ds))

    if args.pause_after_dataset:
        input("\n[PAUSE] Press Enter to continue after dataset initialization...")

    return train_ds, val_ds


# ----------------------------------------------------------------------
# DataLoader queue stats helper
# ----------------------------------------------------------------------

def get_loader_queue_stats(loader_iter, args):
    """
    Best-effort estimate of DataLoader prefetch queue occupancy.

    Returns:
        ready, loading, capacity
    """
    if args.workers <= 0:
        return 0, 0, 0

    capacity = (args.workers or 0) * (args.prefetch or 0)
    if capacity <= 0:
        return 0, 0, 0

    result_queue = getattr(loader_iter, "_worker_result_queue", None)
    if result_queue is None:
        return None, None, capacity

    qsize = None
    try:
        qsize = result_queue.qsize()
    except (NotImplementedError, AttributeError, OSError):
        qsize = None

    if qsize is None:
        return None, None, capacity

    ready = max(0, int(qsize))
    loading = max(0, capacity - ready)
    return ready, loading, capacity


# ----------------------------------------------------------------------
# CSV logging helper
# ----------------------------------------------------------------------

def log_metrics_to_csv(csv_path: str, epoch: int, train_loss: float, val_loss: float,
                       metrics: Dict[str, Any], horizons: Sequence[int]):
    """
    Append metrics for one epoch to a CSV file. The CSV is shaped so it can be
    imported into TensorBoard's 'Import CSV' plugin or any plotting tool.
    """
    file_exists = os.path.exists(csv_path)

    # Build header
    base_fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "overall_accuracy",
        "overall_precision",
        "overall_recall",
        "overall_f1",
        "overall_support",
        "overall_tp",
        "overall_fp",
        "overall_fn",
        "overall_tn",
    ]
    horizon_fields = []
    for h in horizons:
        prefix = f"h{h}"
        horizon_fields.extend([
            f"{prefix}_accuracy",
            f"{prefix}_precision",
            f"{prefix}_recall",
            f"{prefix}_f1",
            f"{prefix}_support",
            f"{prefix}_tp",
            f"{prefix}_fp",
            f"{prefix}_fn",
            f"{prefix}_tn",
        ])
    fieldnames = base_fields + horizon_fields

    row = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "overall_accuracy": float(metrics.get("accuracy", float('nan'))),
        "overall_precision": float(metrics.get("precision", float('nan'))),
        "overall_recall": float(metrics.get("recall", float('nan'))),
        "overall_f1": float(metrics.get("f1", float('nan'))),
        "overall_support": int(metrics.get("support", 0)),
        "overall_tp": int(metrics.get("tp", 0)),
        "overall_fp": int(metrics.get("fp", 0)),
        "overall_fn": int(metrics.get("fn", 0)),
        "overall_tn": int(metrics.get("tn", 0)),
    }

    per_h = metrics.get("per_horizon", {}) or {}
    for h in horizons:
        prefix = f"h{h}"
        m_h = per_h.get(h, {})
        row[f"{prefix}_accuracy"] = float(m_h.get("accuracy", float('nan')))
        row[f"{prefix}_precision"] = float(m_h.get("precision", float('nan')))
        row[f"{prefix}_recall"] = float(m_h.get("recall", float('nan')))
        row[f"{prefix}_f1"] = float(m_h.get("f1", float('nan')))
        row[f"{prefix}_support"] = int(m_h.get("support", 0))
        row[f"{prefix}_tp"] = int(m_h.get("tp", 0))
        row[f"{prefix}_fp"] = int(m_h.get("fp", 0))
        row[f"{prefix}_fn"] = int(m_h.get("fn", 0))
        row[f"{prefix}_tn"] = int(m_h.get("tn", 0))

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    # TensorBoard writer
    writer: Optional[SummaryWriter] = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=args.logdir)
        print(f"[TensorBoard] Logging to: {args.logdir}")

    # Set verbosity flags if --debug is used
    if args.debug:
        args.verbose_dataset = True
        args.verbose_loader = True
        args.verbose_model = True
        args.profile_first_epoch = True
        args.show_worker_stats = True
        print("\n[DEBUG MODE] All verbosity flags enabled")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_index = device.index if device.index is not None else 0
    use_tqdm = not args.no_tqdm

    # NEW: help track where NaNs appear in backward when debugging
    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    # Track initial RAM usage
    print_diagnostic_header("System Resources")
    initial_ram = get_ram_usage()
    print_diagnostic_item("Initial RAM usage", f"{initial_ram:.2f} GB")
    cpu_count_phys = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
    cpu_count_logical = psutil.cpu_count(logical=True)
    print_diagnostic_item("CPU cores", f"{cpu_count_phys} physical, {cpu_count_logical} logical")

    # NEW: Check storage type for Zarr roots (SSD vs HDD vs network FS)
    print("\n  Storage:")
    print_diagnostic_item("ERA5 store", _detect_storage_type(args.era5_zarr), indent=1)
    print_diagnostic_item("SMAP store", _detect_storage_type(args.smap_zarr), indent=1)
    print_diagnostic_item("VIIRS store", _detect_storage_type(args.viirs_zarr), indent=1)

    # Enable TF32 & matmul precision for faster training on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        print_diagnostic_item("TF32 enabled", "Yes (Ampere+ GPUs)")

    # Datasets - track RAM usage
    print("\n" + "=" * 60)
    print("  DATASET INITIALIZATION")
    print("=" * 60)
    before_data_ram = get_ram_usage()
    train_ds, val_ds = make_datasets(args)
    after_data_ram = print_ram_delta(before_data_ram, "Dataset initialization")

    # Handle quick test mode
    if args.quick_test:
        print("\n[Quick Test Mode] Limiting to 1 epoch")
        args.epochs = 1

    # Optional scan (CPU by default to avoid GPU OOM)
    dataset_scan(train_ds, "train", args, device)
    if val_ds is not None:
        dataset_scan(val_ds, "val", args, device)

    # Optimized dataloaders
    print("\n" + "=" * 60)
    print("  DATALOADER INITIALIZATION")
    print("=" * 60)
    with DiagnosticTimer("Train DataLoader creation", track_ram=True):
        base_train_loader = make_loader(train_ds, args.batch_size, True, args)
    diagnose_dataloader(base_train_loader, args)

    if val_ds is not None:
        with DiagnosticTimer("Val DataLoader creation", track_ram=True):
            val_loader = make_loader(val_ds, args.batch_size, False, args)
    else:
        val_loader = None

    # Model - track RAM usage
    print("\n" + "=" * 60)
    print("  MODEL INITIALIZATION")
    print("=" * 60)
    before_model_ram = get_ram_usage()
    with DiagnosticTimer("Model creation", track_ram=True, track_gpu=True, device_index=device_index):
        in_channels_per_timestep = train_ds.C_total
        model = PatchTransformerSHSirenForecast(
            in_channels_per_timestep=in_channels_per_timestep,
            t_hist=args.T_hist,
            horizons=args.horizons,
            d_model=64,
            num_layers=2,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            L_max=4,          # SH degree; ~8–12 is reasonable
            use_sh_siren=True, # turn off if you want simple learned pos-enc
            max_tile_batch=1024,
        ).to(device)
    after_model_ram = print_ram_delta(before_model_ram, "Model creation")

    # Model diagnostics
    diagnose_model_architecture(model, args, train_ds)
    if args.pause_after_model:
        input("\n[PAUSE] Press Enter to continue after model initialization...")

    # Compile model for speedup (PyTorch 2.0+)
    if args.compile:
        print("\n" + "=" * 60)
        print("  MODEL COMPILATION")
        print("=" * 60)
        try:
            with DiagnosticTimer("torch.compile()", track_ram=True):
                model = torch.compile(model, mode='reduce-overhead')
            print("✓ Model compiled successfully")
        except Exception as e:
            print(f"⚠ torch.compile() failed: {e}")

    # Summary of RAM usage
    total_data_ram = after_data_ram - before_data_ram
    total_model_ram = after_model_ram - before_model_ram
    total_ram = get_ram_usage()
    print_diagnostic_header("RAM Usage Summary")
    print_diagnostic_item("Data loading", f"{total_data_ram:.2f} GB")
    print_diagnostic_item("Model creation", f"{total_model_ram:.2f} GB")
    print_diagnostic_item("Data/Model ratio", f"{(total_data_ram / max(total_model_ram, 0.001)):.2f}x")
    print_diagnostic_item("Total RAM used", f"{total_ram:.2f} GB")

    print_env_and_cfg(args, device, train_ds, val_ds, model)

    # Test first batch if requested
    if args.sample_one_batch:
        test_first_batch(base_train_loader, device, model, args)
        print("\n[SAMPLE-ONE-BATCH] Exiting after first batch test")
        if writer is not None:
            writer.close()
        return

    # Optimizer + AMP scaler (disable scaler on bf16)
    with DiagnosticTimer("Optimizer creation"):
        optim = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-4,
            fused=torch.cuda.is_available()
        )
        use_cuda = (device.type == "cuda")
        use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

        # Use the classic CUDA GradScaler; no device_type kwarg
        try:
            # New-ish location (PyTorch 2.x): torch.amp.GradScaler
            scaler = torch.amp.GradScaler(
                enabled=use_cuda and not use_bf16,
            )
        except TypeError:
            # Fallback for older-style API if needed
            from torch.cuda.amp import GradScaler as CudaGradScaler
            scaler = CudaGradScaler(enabled=use_cuda and not use_bf16)

    # Choose loss function (BCE vs focal)
    if args.loss == "bce":
        criterion = MaskedBCEWithLogits()
        print("\n[Loss] Using MaskedBCEWithLogits (BCE).")
    else:
        criterion = MaskedFocalLossWithLogits(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"\n[Loss] Using MaskedFocalLossWithLogits (Focal): "
              f"alpha={args.focal_alpha}, gamma={args.focal_gamma}")

    if args.dry_run:
        print("\n[DRY-RUN] Exiting without training")
        if writer is not None:
            writer.close()
        return

    # NEW: validation-only test mode
    if args.val_test_only:
        print("\n[VAL-TEST-ONLY] Running a validation pass to test the validation loop, then exiting...")
        if val_loader is None:
            print("[VAL-TEST-ONLY] Error: validation dataset/loader not available. "
                  "Remove --skip-val-dataset if you want to test validation.")
        else:
            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                device,
                criterion,
                args,
                use_tqdm=not args.no_tqdm,
                show_worker_stats=args.show_worker_stats,
            )

            print_diagnostic_header("Validation Test Results (VAL-TEST-ONLY)")
            print_diagnostic_item("Validation loss", f"{val_loss:.6f}", indent=1)
            if val_metrics is not None and not math.isnan(val_metrics["f1"]):
                print_diagnostic_item("Accuracy", f"{val_metrics['accuracy']*100:.2f}%", indent=1)
                print_diagnostic_item("Precision", f"{val_metrics['precision']*100:.2f}%", indent=1)
                print_diagnostic_item("Recall", f"{val_metrics['recall']*100:.2f}%", indent=1)
                print_diagnostic_item("F1-score", f"{val_metrics['f1']*100:.2f}%", indent=1)
                print_diagnostic_item("Support (valid pixels)", int(val_metrics["support"]), indent=1)

        if writer is not None:
            writer.close()
        return

    # TRAINING
    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)


    best_val = float("inf")
    best_metrics = None
    step_timer = TrainingStepTimer()
    global_step = 0  # TensorBoard step counter

    # Start RAM monitor and progress bars only if TQDM enabled
    ram_monitor = None
    if use_tqdm:
        ram_monitor = RAMMonitor(device_id=device_index, update_interval=0.5)
        ram_monitor.start()

    epoch_pbar = tqdm(total=args.epochs, desc="📊 Epochs", position=1, leave=True, disable=not use_tqdm) if use_tqdm else None
    batch_pbar = None  # Will be created per epoch
    timing_pbar = tqdm(total=0, position=3, bar_format='{desc}', leave=True, disable=not use_tqdm) if use_tqdm else None
    throughput_pbar = tqdm(total=0, position=4, bar_format='{desc}', leave=True, disable=not use_tqdm) if use_tqdm else None

    metrics_csv_path = os.path.join("logs", "metrics_log.csv")

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            running_loss, running_mask = 0.0, 0.0
            steps_this_epoch = 0

            # Reset timing for this epoch
            if args.profile_first_epoch and epoch == 1:
                step_timer.reset()

            # Create batch progress bar for this epoch
            if batch_pbar:
                batch_pbar.close()
            if use_tqdm:
                batch_pbar = tqdm(
                    total=len(base_train_loader),
                    desc=f"🔄 Batch (Epoch {epoch})",
                    position=2,
                    leave=True,
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                               '[{elapsed}<{remaining}] {postfix}',
                    disable=not use_tqdm
                )
            else:
                batch_pbar = None

            # Timing accumulators
            total_data_time = 0.0
            total_transfer_time = 0.0
            total_forward_time = 0.0
            total_backward_time = 0.0
            total_optim_time = 0.0
            total_decompress_time = 0.0   # NEW: IO/decompression per step

            # Explicit iterator so we can introspect internal queue
            train_iter = iter(base_train_loader)
            step = 0

            while True:
                try:
                    step += 1
                    steps_this_epoch += 1

                    # --- Data loading (timed directly) ---
                    loader_start = time.time()
                    batch = next(train_iter)
                    data_time = time.time() - loader_start

                    # --- DataLoader queue stats -> RAM monitor ---
                    ready = loading = capacity = None
                    if ram_monitor is not None:
                        ready, loading, capacity = get_loader_queue_stats(train_iter, args)
                        ram_monitor.set_loader_queue(ready, loading, capacity)

                except StopIteration:
                    break

                step_start_time = loader_start

                # --- NEW: aggregate per-sample IO/decompress timings from meta ---
                decompress_time = 0.0
                metas = batch.get("meta", None)
                if isinstance(metas, list) and metas:
                    io_sum = 0.0
                    n_io = 0
                    for m in metas:
                        if isinstance(m, dict) and "io_time_total" in m:
                            try:
                                io_sum += float(m["io_time_total"])
                                n_io += 1
                            except (TypeError, ValueError):
                                continue
                    if n_io > 0:
                        # Average per-sample decompression time (seconds)
                        decompress_time = io_sum / n_io

                # Transfer to device (tensors only)
                transfer_start = time.time()
                batch = {
                    k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                if device.type == "cuda" and args.sync_every_step:
                    torch.cuda.synchronize()
                transfer_time = time.time() - transfer_start

                # --- NEW: check inputs/labels/mask before forward ---
                try:
                    assert_finite_tensor(batch.get("x"), "batch['x']", where=f"epoch={epoch}, step={step}")
                    assert_finite_tensor(batch.get("y"), "batch['y']", where=f"epoch={epoch}, step={step}")
                    assert_finite_tensor(batch.get("mask"), "batch['mask']", where=f"epoch={epoch}, step={step}")
                except RuntimeError as e:
                    print("[ERROR] Non-finite in batch before forward.")
                    if "meta" in batch and isinstance(batch["meta"], list) and batch["meta"]:
                        print("  Example meta[0]:", batch["meta"][0])
                    raise

                # Forward pass
                forward_start = time.time()
                with amp_autocast(device):
                    logits = model(batch)
                    raw_loss = criterion(logits, batch["y"], batch["mask"])
                    loss_div = raw_loss / args.grad_accum
                if device.type == "cuda" and args.sync_every_step:
                    torch.cuda.synchronize()
                forward_time = time.time() - forward_start

                # Backward pass
                backward_start = time.time()
                if scaler.is_enabled():
                    scaler.scale(loss_div).backward()
                else:
                    loss_div.backward()
                if device.type == "cuda" and args.sync_every_step:
                    torch.cuda.synchronize()
                backward_time = time.time() - backward_start

                # Optimizer step
                optim_start = time.time()
                if step % args.grad_accum == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optim)
                        if any(p.grad is not None for p in model.parameters()):
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optim)
                        scaler.update()
                    else:
                        if any(p.grad is not None for p in model.parameters()):
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optim.step()
                    optim.zero_grad(set_to_none=True)
                if device.type == "cuda" and args.sync_every_step:
                    torch.cuda.synchronize()
                optim_time = time.time() - optim_start

                # Calculate step time
                step_total_time = time.time() - step_start_time

                # Accumulate times
                total_data_time += data_time
                total_transfer_time += transfer_time
                total_forward_time += forward_time
                total_backward_time += backward_time
                total_optim_time += optim_time
                total_decompress_time += decompress_time   # NEW

                # Record timing for first epoch profiling
                if args.profile_first_epoch and epoch == 1 and step <= 20:
                    step_timer.add(data_time, transfer_time, forward_time, backward_time, optim_time)

                # Accumulate loss
                mask_sum = batch["mask"].sum().item()
                running_loss += raw_loss.item() * mask_sum
                running_mask += mask_sum

                # Throughput numbers
                avg_step_time = (total_data_time + total_transfer_time + total_forward_time +
                                 total_backward_time + total_optim_time) / max(steps_this_epoch, 1)
                samp_per_sec = (args.batch_size / avg_step_time) if avg_step_time > 0 else 0.0
                updates_per_sec = (1.0 / (avg_step_time * args.grad_accum)) if avg_step_time > 0 else 0.0

                # --------------------------
                # TensorBoard: per-step logs
                # --------------------------
                if writer is not None:
                    # Core training scalars
                    writer.add_scalar("train/step_loss", raw_loss.item(), global_step)
                    writer.add_scalar("train/mask_sum", mask_sum, global_step)

                    # Timings (seconds)
                    writer.add_scalar("timing/data", data_time, global_step)
                    writer.add_scalar("timing/transfer", transfer_time, global_step)
                    writer.add_scalar("timing/forward", forward_time, global_step)
                    writer.add_scalar("timing/backward", backward_time, global_step)
                    writer.add_scalar("timing/optim", optim_time, global_step)
                    writer.add_scalar("timing/decompress", decompress_time, global_step)
                    writer.add_scalar("timing/step_total", step_total_time, global_step)

                    # Throughput
                    writer.add_scalar("throughput/samples_per_sec", samp_per_sec, global_step)
                    writer.add_scalar("throughput/updates_per_sec", updates_per_sec, global_step)

                    # System metrics
                    writer.add_scalar("system/ram_gb", get_ram_usage(), global_step)
                    if torch.cuda.is_available():
                        writer.add_scalar(
                            "system/gpu_gb",
                            get_gpu_memory(device_index),
                            global_step,
                        )

                    # DataLoader queue / worker utilisation
                    if capacity is not None and capacity > 0:
                        writer.add_scalar("loader/queue_capacity", capacity, global_step)
                        writer.add_scalar("loader/queue_ready", ready or 0, global_step)
                        writer.add_scalar("loader/queue_loading", loading or 0, global_step)

                global_step += 1

                # TQDM/UI updates (throttled by --log-interval; 0 => every step)
                should_log = (args.log_interval == 0) or (step % args.log_interval == 0)

                if use_tqdm and should_log:
                    timing_desc = (
                        f"⏱️  Timing: "
                        f"Decomp={decompress_time*1000:.0f}ms | "
                        f"Data={data_time*1000:.0f}ms | "
                        f"Transfer={transfer_time*1000:.0f}ms | "
                        f"▶️Forward={forward_time*1000:.0f}ms | "
                        f"◀️Backward={backward_time*1000:.0f}ms | "
                        f"Optim={optim_time*1000:.0f}ms"
                    )
                    if timing_pbar:
                        timing_pbar.set_description_str(timing_desc)

                    if batch_pbar:
                        batch_pbar.set_postfix({
                            'loss': f'{raw_loss.item():.4f}',
                            'step_ms': f'{step_total_time*1000:.0f}',
                            'samp/s': f'{samp_per_sec:.1f}',
                            'upd/s': f'{updates_per_sec:.2f}',
                        })
                    if throughput_pbar:
                        throughput_pbar.set_description_str(
                            f"📈 Throughput: {samp_per_sec:.1f} samp/s | upd/s: {updates_per_sec:.2f}"
                        )

                if batch_pbar:
                    batch_pbar.update(1)

            # Flush remaining grads if needed
            if steps_this_epoch > 0 and (steps_this_epoch % args.grad_accum) != 0:
                if scaler.is_enabled():
                    scaler.unscale_(optim)
                    if any(p.grad is not None for p in model.parameters()):
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    if any(p.grad is not None for p in model.parameters()):
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                optim.zero_grad(set_to_none=True)

            # Print timing breakdown for first epoch
            if args.profile_first_epoch and epoch == 1:
                print("\n")
                step_timer.print_summary("First Epoch")

            train_loss = running_loss / max(running_mask, 1.0)

            # Clear cache before validation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Validation with timing
            if val_loader is not None:
                val_start = time.time()
                if timing_pbar and use_tqdm:
                    timing_pbar.set_description_str("⏳ Running validation...")
                val_loss, val_metrics = evaluate(
                    model, val_loader, device, criterion, args,
                    use_tqdm=use_tqdm,
                    show_worker_stats=False
                )
                val_time = time.time() - val_start
                if timing_pbar and use_tqdm:
                    timing_pbar.set_description_str(f"✅ Validation complete in {val_time:.1f}s")
            else:
                val_loss, val_metrics = float('nan'), None

            epoch_duration = time.time() - epoch_start

            # Update epoch progress bar (include metrics in postfix if available)
            if epoch_pbar:
                postfix = {
                    'train': f'{train_loss:.4f}',
                    'val': f'{val_loss:.4f}' if val_loader else 'N/A',
                    'time': f'{epoch_duration:.0f}s'
                }
                if val_metrics is not None and not math.isnan(val_metrics["f1"]):
                    postfix['F1'] = f'{val_metrics["f1"]:.3f}'
                    postfix['P'] = f'{val_metrics["precision"]:.3f}'
                    postfix['R'] = f'{val_metrics["recall"]:.3f}'
                epoch_pbar.set_postfix(postfix)
                epoch_pbar.update(1)

            # Show epoch timing breakdown
            avg_data = total_data_time / max(steps_this_epoch, 1)
            avg_transfer = total_transfer_time / max(steps_this_epoch, 1)
            avg_forward = total_forward_time / max(steps_this_epoch, 1)
            avg_backward = total_backward_time / max(steps_this_epoch, 1)
            avg_optim = total_optim_time / max(steps_this_epoch, 1)
            avg_decompress = total_decompress_time / max(steps_this_epoch, 1)
            total_avg = max(1e-9, avg_data + avg_transfer + avg_forward + avg_backward + avg_optim)

            # --------------------------
            # TensorBoard: epoch-level logs
            # --------------------------
            if writer is not None:
                writer.add_scalar("epoch/train_loss", train_loss, epoch)
                writer.add_scalar("epoch/val_loss", val_loss, epoch if not math.isnan(val_loss) else epoch)

                # Average timing breakdown for this epoch
                writer.add_scalar("epoch_timing/avg_data", avg_data, epoch)
                writer.add_scalar("epoch_timing/avg_transfer", avg_transfer, epoch)
                writer.add_scalar("epoch_timing/avg_forward", avg_forward, epoch)
                writer.add_scalar("epoch_timing/avg_backward", avg_backward, epoch)
                writer.add_scalar("epoch_timing/avg_optim", avg_optim, epoch)
                writer.add_scalar("epoch_timing/avg_decompress", avg_decompress, epoch)

                # Constant-ish loader info
                writer.add_scalar("loader/num_workers", args.workers, epoch)
                writer.add_scalar("loader/prefetch_factor", args.prefetch, epoch)

                # Validation metrics (overall + per horizon)
                if val_metrics is not None and not math.isnan(val_metrics["f1"]):
                    writer.add_scalar("metrics/overall_accuracy", val_metrics["accuracy"], epoch)
                    writer.add_scalar("metrics/overall_precision", val_metrics["precision"], epoch)
                    writer.add_scalar("metrics/overall_recall", val_metrics["recall"], epoch)
                    writer.add_scalar("metrics/overall_f1", val_metrics["f1"], epoch)
                    writer.add_scalar("metrics/overall_support", val_metrics["support"], epoch)

                    per_h = val_metrics.get("per_horizon", {}) or {}
                    for h in args.horizons:
                        mh = per_h.get(h, None)
                        if mh is None:
                            continue
                        prefix = f"h{h}"
                        writer.add_scalar(f"metrics/{prefix}_accuracy", mh["accuracy"], epoch)
                        writer.add_scalar(f"metrics/{prefix}_precision", mh["precision"], epoch)
                        writer.add_scalar(f"metrics/{prefix}_recall", mh["recall"], epoch)
                        writer.add_scalar(f"metrics/{prefix}_f1", mh["f1"], epoch)
                        writer.add_scalar(f"metrics/{prefix}_support", mh["support"], epoch)

            timing_summary = (
                f"📊 Epoch {epoch} Summary: "
                f"Data={avg_data*1000:.0f}ms ({100*avg_data/total_avg:.0f}%) | "
                f"Transfer={avg_transfer*1000:.0f}ms ({100*avg_transfer/total_avg:.0f}%) | "
                f"Forward={avg_forward*1000:.0f}ms ({100*avg_forward/total_avg:.0f}%) | "
                f"Backward={avg_backward*1000:.0f}ms ({100*avg_backward/total_avg:.0f}%) | "
                f"Optim={avg_optim*1000:.0f}ms ({100*avg_optim/total_avg:.0f}%)"
            )
            print(f"\n{timing_summary}")

            # NEW: explicit decompression stats
            print_diagnostic_item(
                "Avg decompression (dataset IO)",
                f"{avg_decompress*1000:.0f}ms per step",
                indent=1
            )

            # NEW: validation metrics summary
            if val_metrics is not None:
                print_diagnostic_header(f"Validation Metrics (Epoch {epoch})")
                print_diagnostic_item("Accuracy", f"{val_metrics['accuracy']*100:.2f}%", indent=1)
                print_diagnostic_item("Precision", f"{val_metrics['precision']*100:.2f}%", indent=1)
                print_diagnostic_item("Recall", f"{val_metrics['recall']*100:.2f}%", indent=1)
                print_diagnostic_item("F1-score", f"{val_metrics['f1']*100:.2f}%", indent=1)
                print_diagnostic_item("TP / FP / FN / TN",
                                      f"{int(val_metrics['tp'])} / {int(val_metrics['fp'])} / "
                                      f"{int(val_metrics['fn'])} / {int(val_metrics['tn'])}",
                                      indent=1)
                print_diagnostic_item("Support (valid pixels)", int(val_metrics["support"]), indent=1)

                # Per-horizon metrics
                per_h = val_metrics.get("per_horizon", {}) or {}
                for h in args.horizons:
                    mh = per_h.get(h, None)
                    if mh is None:
                        continue
                    print_diagnostic_item(
                        f"Horizon {h}d - Acc/Prec/Rec/F1",
                        f"{mh['accuracy']*100:.2f}% / "
                        f"{mh['precision']*100:.2f}% / "
                        f"{mh['recall']*100:.2f}% / "
                        f"{mh['f1']*100:.2f}%",
                        indent=2
                    )
                    print_diagnostic_item(
                        f"Horizon {h}d - TP/FP/FN/TN",
                        f"{int(mh['tp'])} / {int(mh['fp'])} / {int(mh['fn'])} / {int(mh['tn'])}",
                        indent=2
                    )

                # Log metrics to CSV
                log_metrics_to_csv(
                    metrics_csv_path,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    metrics=val_metrics,
                    horizons=args.horizons
                )

            # Show resource usage periodically
            if args.debug and epoch % max(1, max(args.epochs // 5, 1)) == 0:
                current_ram = get_ram_usage()
                line = f"[DEBUG] Epoch {epoch} RAM: {current_ram:.2f} GB"
                if torch.cuda.is_available():
                    gpu_alloc = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
                    gpu_res = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
                    line += f" | GPU alloc/res: {gpu_alloc:.2f}/{gpu_res:.2f} GB"
                print(line)

            # Save best checkpoint (based on validation loss, keep metrics)
            if val_loader is not None and val_loss < best_val:
                best_val = val_loss
                best_metrics = val_metrics
                os.makedirs("checkpoints", exist_ok=True)
                model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
                if timing_pbar and use_tqdm:
                    timing_pbar.set_description_str("💾 Saving best checkpoint...")
                state = {
                    "model": model_to_save.state_dict(),
                    "args": vars(args),
                    "val_loss": val_loss,
                    "val_metrics": best_metrics,
                }
                if scaler.is_enabled():
                    state["scaler"] = scaler.state_dict()
                torch.save(state, os.path.join("checkpoints", "best.pt"))
                if timing_pbar and use_tqdm:
                    timing_pbar.set_description_str(f"✅ Saved best model (val_loss={best_val:.4f})")
                print(f"[checkpoint] new best: {best_val:.4f}")

                # TensorBoard: best metrics snapshot
                if writer is not None and best_metrics is not None:
                    writer.add_scalar("best/val_loss", best_val, epoch)
                    writer.add_scalar("best/f1", best_metrics["f1"], epoch)
                    writer.add_scalar("best/precision", best_metrics["precision"], epoch)
                    writer.add_scalar("best/recall", best_metrics["recall"], epoch)
                    writer.add_scalar("best/accuracy", best_metrics["accuracy"], epoch)

    finally:
        # Clean up progress bars
        if ram_monitor:
            ram_monitor.stop()
        if batch_pbar:
            batch_pbar.close()
        if epoch_pbar:
            epoch_pbar.close()
        if timing_pbar:
            timing_pbar.close()
        if throughput_pbar:
            throughput_pbar.close()
        if writer is not None:
            writer.close()

    # Final summary
    print_diagnostic_header("Training Complete")
    if val_loader is not None and best_val < float('inf'):
        print_diagnostic_item("Best validation loss", f"{best_val:.4f}")
        if best_metrics is not None:
            print_diagnostic_item("Best val F1", f"{best_metrics['f1']*100:.2f}%")
            print_diagnostic_item("Best val Precision", f"{best_metrics['precision']*100:.2f}%")
            print_diagnostic_item("Best val Recall", f"{best_metrics['recall']*100:.2f}%")
            print_diagnostic_item("Best val Accuracy", f"{best_metrics['accuracy']*100:.2f}%")
            per_h = best_metrics.get("per_horizon", {}) or {}
            for h in args.horizons:
                mh = per_h.get(h, None)
                if mh is None:
                    continue
                print_diagnostic_item(
                    f"Best Horizon {h}d F1",
                    f"{mh['f1']*100:.2f}%",
                    indent=1
                )
    print_diagnostic_item("Total epochs", args.epochs)
    print_diagnostic_item("Final RAM usage", f"{get_ram_usage():.2f} GB")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        reserv = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
        print_diagnostic_item("Final GPU mem (alloc/res)", f"{alloc:.2f}/{reserv:.2f} GB")


if __name__ == "__main__":
    main()
