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
- Simple PatchForecastCNN model that operates on patches
- Hard-stop NaN debugger (--nan-debug) to locate non-finite values

This version also includes:
- Storage type detection (SSD vs HDD vs network FS) for Zarr roots
- Explicit multiprocessing context reporting (forkserver/spawn/etc.)
- Per-batch decompression time from dataset (Zarr IO) vs GPU transfer time
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from joint_peat_dataset_builder import JointPeatDataset  # <- your new dataset


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
                info += f" | GPU: {gpu_delta:+.2f} GB"
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
# Simple patch-based CNN for JointPeatDataset
# ----------------------------------------------------------------------

class PatchForecastCNN(nn.Module):
    """
    Simple 2D CNN for patch forecasting.

    Expects:
        batch["x"]:
            - (B, T_hist, C_total, H, W)   if stack-time='separate'
            - (B, T_hist*C_total, H, W)    if stack-time='channel'

    Outputs:
        logits: (B, K, H, W) where K = len(horizons)
    """
    def __init__(self, in_channels: int, horizons: Sequence[int]):
        super().__init__()
        self.in_channels = int(in_channels)
        self.horizons = list(int(h) for h in horizons)
        k = len(self.horizons)

        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, k, kernel_size=1),
        )

    def forward(self, batch: Dict[str, Any]):
        x = batch["x"]
        if x.dim() == 5:
            # (B, T_hist, C_total, H, W) -> (B, T_hist*C_total, H, W)
            B, T, C, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B, C * T, H, W)
        elif x.dim() == 4:
            # Already (B, C_in, H, W)
            pass
        else:
            raise ValueError(f"PatchForecastCNN expects x to have 4 or 5 dims, got shape {tuple(x.shape)}")
        return self.net(x)


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

    def _monitor_loop(self):
        """Background loop that updates RAM/GPU stats"""
        while self.running:
            self.current_ram = get_ram_usage()
            self.current_gpu = get_gpu_memory(self.device_id) if torch.cuda.is_available() else 0.0

            desc = f"💾 RAM: {self.current_ram:.2f}GB"
            if torch.cuda.is_available():
                desc += f" | 🎮 GPU: {self.current_gpu:.2f}GB"
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
# Evaluation & dataset scan
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, criterion, use_tqdm: bool = True, show_worker_stats: bool = False):
    model.eval()
    tot_loss, tot_mask = 0.0, 0.0
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
    if tot_mask == 0:
        print("[WARN] Validation mask sum was zero across all batches; reporting NaN.")
        return float('nan')
    return tot_loss / tot_mask


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

    # Pause points
    p.add_argument("--pause-after-dataset", action="store_true", help="Pause after dataset init")
    p.add_argument("--pause-after-model", action="store_true", help="Pause after model init")

    # NEW flags for this revision
    p.add_argument("--sync-every-step", action="store_true",
                   help="Force device sync after each phase for exact timings (slower)")
    p.add_argument("--measure-loader-time", action="store_true",
                   help="Measure true loader time using TimedDataLoader during training (slower)")

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
        dummy_batch = {"x": dummy_x}
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
    print_diagnostic_item("Input channels", model.in_channels, indent=1)
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

    # Train dataset (optionally with normalization estimation)
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
        )

        if norm_mode == "per_channel":
            mean, std = train_ds.get_normalization()
            val_ds.set_normalization(mean, std)

    print_diagnostic_item("Val samples", len(val_ds))

    if args.pause_after_dataset:
        input("\n[PAUSE] Press Enter to continue after dataset initialization...")

    return train_ds, val_ds


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()

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
        if args.measure_loader_time:
            train_loader = wrap_loader(
                base_train_loader,
                desc="train loader",
                use_tqdm=False,
                show_worker_stats=args.show_worker_stats,
                position=5
            )
        else:
            train_loader = base_train_loader

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
        in_channels = train_ds.C_total * args.T_hist
        model = PatchForecastCNN(
            in_channels=in_channels,
            horizons=args.horizons,
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

    criterion = MaskedBCEWithLogits()

    if args.dry_run:
        print("\n[DRY-RUN] Exiting without training")
        return

    # TRAINING
    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)

    best_val = float("inf")
    step_timer = TrainingStepTimer()

    # Start RAM monitor and progress bars only if TQDM enabled
    ram_monitor = None
    if use_tqdm:
        ram_monitor = RAMMonitor(device_id=device_index, update_interval=0.5)
        ram_monitor.start()

    epoch_pbar = tqdm(total=args.epochs, desc="📊 Epochs", position=1, leave=True, disable=not use_tqdm) if use_tqdm else None
    batch_pbar = None  # Will be created per epoch
    timing_pbar = tqdm(total=0, position=3, bar_format='{desc}', leave=True, disable=not use_tqdm) if use_tqdm else None
    throughput_pbar = tqdm(total=0, position=4, bar_format='{desc}', leave=True, disable=not use_tqdm) if use_tqdm else None

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

            # Choose the iterable for this epoch (wrapped iff measuring loader time)
            if args.measure_loader_time:
                epoch_iterable = wrap_loader(
                    base_train_loader,
                    desc=f"train e{epoch}",
                    use_tqdm=False,
                    show_worker_stats=args.show_worker_stats,
                    position=5
                )
            else:
                epoch_iterable = base_train_loader

            for step, batch in enumerate(epoch_iterable, 1):
                step_start_time = time.time()
                steps_this_epoch += 1

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
                transfer_time = time.time() - transfer_start

                # --- NEW: check inputs/labels/mask before forward ---
                try:
                    assert_finite_tensor(batch.get("x"), "batch['x']", where=f"epoch={epoch}, step={step}")
                    assert_finite_tensor(batch.get("y"), "batch['y']", where=f"epoch={epoch}, step={step}")
                    assert_finite_tensor(batch.get("mask"), "batch['mask']", where=f"epoch={epoch}, step={step}")
                except RuntimeError as e:
                    print("[ERROR] Non-finite in batch before forward.")
                    # Optionally dump some meta info for one sample
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

                # Data time: either use precise loader timing (if enabled) or estimate as remainder
                if args.measure_loader_time and isinstance(epoch_iterable, TimedDataLoader):
                    data_time = float(epoch_iterable.last_batch_time)
                else:
                    data_time = max(0.0, step_total_time - transfer_time - forward_time - backward_time - optim_time)

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
                                 total_backward_time + total_optim_time) / steps_this_epoch
                samp_per_sec = (args.batch_size / avg_step_time) if avg_step_time > 0 else 0.0
                updates_per_sec = (1.0 / (avg_step_time * args.grad_accum)) if avg_step_time > 0 else 0.0

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

            # Validation with timing (reuse timing_pbar slot)
            if val_loader is not None:
                val_start = time.time()
                if timing_pbar and use_tqdm:
                    timing_pbar.set_description_str("⏳ Running validation...")
                val_loss = evaluate(
                    model, val_loader, device, criterion,
                    use_tqdm=use_tqdm,
                    show_worker_stats=False
                )
                val_time = time.time() - val_start
                if timing_pbar and use_tqdm:
                    timing_pbar.set_description_str(f"✅ Validation complete in {val_time:.1f}s")
            else:
                val_loss = float('nan')

            epoch_duration = time.time() - epoch_start

            # Update epoch progress bar
            if epoch_pbar:
                epoch_pbar.set_postfix({
                    'train': f'{train_loss:.4f}',
                    'val': f'{val_loss:.4f}' if val_loader else 'N/A',
                    'time': f'{epoch_duration:.0f}s'
                })
                epoch_pbar.update(1)

            # Show epoch timing breakdown
            avg_data = total_data_time / max(steps_this_epoch, 1)
            avg_transfer = total_transfer_time / max(steps_this_epoch, 1)
            avg_forward = total_forward_time / max(steps_this_epoch, 1)
            avg_backward = total_backward_time / max(steps_this_epoch, 1)
            avg_optim = total_optim_time / max(steps_this_epoch, 1)
            avg_decompress = total_decompress_time / max(steps_this_epoch, 1)
            total_avg = max(1e-9, avg_data + avg_transfer + avg_forward + avg_backward + avg_optim)

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

            # Show resource usage periodically
            if args.debug and epoch % max(1, max(args.epochs // 5, 1)) == 0:
                current_ram = get_ram_usage()
                line = f"[DEBUG] Epoch {epoch} RAM: {current_ram:.2f} GB"
                if torch.cuda.is_available():
                    gpu_alloc = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
                    gpu_res = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
                    line += f" | GPU alloc/res: {gpu_alloc:.2f}/{gpu_res:.2f} GB"
                print(line)

            # Save best checkpoint
            if val_loader is not None and val_loss < best_val:
                best_val = val_loss
                os.makedirs("checkpoints", exist_ok=True)
                model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
                if timing_pbar and use_tqdm:
                    timing_pbar.set_description_str("💾 Saving best checkpoint...")
                state = {
                    "model": model_to_save.state_dict(),
                    "args": vars(args),
                    "val_loss": val_loss,
                }
                if scaler.is_enabled():
                    state["scaler"] = scaler.state_dict()
                torch.save(state, os.path.join("checkpoints", "best.pt"))
                if timing_pbar and use_tqdm:
                    timing_pbar.set_description_str(f"✅ Saved best model (val_loss={best_val:.4f})")
                print(f"[checkpoint] new best: {best_val:.4f}")

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

    # Final summary
    print_diagnostic_header("Training Complete")
    if val_loader is not None and best_val < float('inf'):
        print_diagnostic_item("Best validation loss", f"{best_val:.4f}")
    print_diagnostic_item("Total epochs", args.epochs)
    print_diagnostic_item("Final RAM usage", f"{get_ram_usage():.2f} GB")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        reserv = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
        print_diagnostic_item("Final GPU mem (alloc/res)", f"{alloc:.2f}/{reserv:.2f} GB")


if __name__ == "__main__":
    main()
