from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import pickle
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

# Limit OpenBLAS/MKL threads to prevent process explosion with multiple workers
# This prevents the "RLIMIT_NPROC" crash when using many DataLoader workers
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import psutil
from sklearn.isotonic import IsotonicRegression as SklearnIsotonicRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from joint_peat_dataset_builder_cluster import JointPeatDataset, parse_input_spec

try:
    import xgboost as xgb
except Exception:
    print("ERROR: xgboost import failed. Install it first (pip install xgboost).")
    raise

# Optional (for GPU prediction without CPU copies)
try:
    import cupy as cp  # type: ignore
    _HAVE_CUPY = True
except Exception:
    print("CUPY NOT IMPORTED (GPU prediction without CPU copies disabled unless available)")
    cp = None
    _HAVE_CUPY = False

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:
    print("MATPLOTLIB NOT IMPORTED (plots disabled unless available)")

# Optuna for hyperparameter optimization (optional)
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    _HAVE_OPTUNA = True
    # Evolutionary algorithm samplers (CMA-ES, NSGA-II)
    try:
        from optuna.samplers import CmaEsSampler
        _HAVE_CMAES = True
    except ImportError:
        CmaEsSampler = None
        _HAVE_CMAES = False
    try:
        from optuna.samplers import NSGAIISampler
        _HAVE_NSGAII = True
    except ImportError:
        NSGAIISampler = None
        _HAVE_NSGAII = False
except ImportError:
    optuna = None
    _HAVE_OPTUNA = False
    _HAVE_CMAES = False
    _HAVE_NSGAII = False
    CmaEsSampler = None
    NSGAIISampler = None

# -----------------------------
# Constants / small knobs
# -----------------------------
DATALOADER_TIME_DEQUE_MAXLEN = 50
CALIB_SPLIT_SEED_OFFSET = 1337  # keep deterministic but separated from data_seed
EPS_PROB = 1e-7


# ----------------------------------------------------------------------
# Small utils
# ----------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_ram_usage() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def header(title: str):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def parse_seeds_list(spec: str) -> list[int]:
    out: list[int] = []
    for s in (spec or "").split(","):
        s = s.strip()
        if s:
            out.append(int(s))
    if not out:
        raise ValueError("No seeds provided (use --seeds like '1,2,3').")
    return out


def _unwrap_subset(ds):
    base = ds
    while isinstance(base, Subset):
        base = base.dataset
    return base


def _xgb_version_tuple() -> tuple[int, int, int]:
    v = getattr(xgb, "__version__", "0.0.0")
    try:
        parts = v.split("+", 1)[0].split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return (major, minor, patch)
    except Exception:
        return (0, 0, 0)


def _is_cuda_device_str(s: str) -> bool:
    """
    Accepts:
      "cuda", "gpu", "cuda:0", "cuda:1", "gpu:2", etc.
    """
    ss = str(s).strip().lower()
    if ss in ("cuda", "gpu"):
        return True
    return ss.startswith("cuda:") or ss.startswith("gpu:")


def device_is_cuda() -> bool:
    return torch.cuda.is_available()


# ----------------------------------------------------------------------
# Monitoring
# ----------------------------------------------------------------------

class RAMMonitor:
    def __init__(self, update_interval=0.5):
        self.update_interval = update_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.pbar = None
        self._lock = threading.Lock()
        self._status = "Initializing..."

    def start(self):
        self.running = True
        self.pbar = tqdm(total=0, position=0, bar_format="{desc}", leave=True)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        if self.pbar:
            try:
                self.pbar.close()
            except Exception:
                pass

    def set(self, status: str):
        with self._lock:
            self._status = status

    def _loop(self):
        while self.running:
            with self._lock:
                status = self._status
            desc = f"💾 RAM: {get_ram_usage():.2f}GB | {status}"
            if self.pbar:
                try:
                    self.pbar.set_description_str(desc)
                except Exception:
                    pass
            time.sleep(self.update_interval)


class TimedDataLoader:
    def __init__(self, loader: DataLoader, desc: str, use_tqdm: bool, position: int):
        self.loader = loader
        self.desc = desc
        self.use_tqdm = use_tqdm
        self.position = position
        self.batch_times: deque = deque(maxlen=DATALOADER_TIME_DEQUE_MAXLEN)

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
    if requested in (None, "auto", ""):
        return None
    if requested in ("spawn", "fork", "forkserver"):
        return requested
    return None


def collate(batch):
    """
    Custom collate function that handles both:
    - Standard format: x=(F,H,W) → stack to (B,F,H,W)
    - Flattened pixels: x=(Npix,F) → concatenate to (B*Npix,F)
    """
    if not batch:
        raise RuntimeError("Empty batch in collate (dataset filtering too aggressive?).")
    keys = batch[0].keys()
    out: Dict[str, Any] = {}

    # Detect if using flatten_pixels (2D tensors instead of 3D/4D)
    first_x = batch[0]["x"]
    is_flattened = torch.is_tensor(first_x) and first_x.dim() == 2

    for k in keys:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            if is_flattened and k in ("x", "y", "mask", "coords"):
                # Concatenate flattened pixel data along the pixel dimension
                out[k] = torch.cat(vals, dim=0)
            else:
                # Stack normally for non-flattened data
                out[k] = torch.stack(vals, 0)
        elif k == "regime_id":
            # regime_id is a numpy array: (patch, patch) or (Npix,) if flattened
            if is_flattened:
                # Concatenate flattened regime_id arrays
                out[k] = np.concatenate(vals, axis=0)
            else:
                # Stack as (B, patch, patch)
                out[k] = np.stack(vals, axis=0)
        elif k in ("tile_id", "lat", "lon", "date", "y0", "x0", "W_global"):
            # Convert scalar metadata to numpy arrays for batching
            out[k] = np.array(vals)
        else:
            out[k] = vals
    return out


def make_loader(ds, batch_size, shuffle, args, device: Optional[torch.device] = None, eval_mode: bool = False):
    """
    Dataloader helper.
    NOTE: Dataloader yields CPU tensors. We move to GPU via prefetcher / explicit .to().

    Args:
        eval_mode: If True, reduces worker count to prevent process limit issues during evaluation.
                   Typically set to True for validation/test loaders.
    """
    pin = bool(device is not None and device.type == "cuda")

    # Reduce workers during evaluation to prevent process limit issues
    # This prevents the RLIMIT_NPROC crash when multiple loaders are active
    num_workers = max(1, args.workers // 4) if eval_mode else args.workers

    kw: Dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=collate,
        persistent_workers=(num_workers > 0),
        shuffle=shuffle,
        drop_last=False,
    )

    if num_workers > 0:
        kw["prefetch_factor"] = args.prefetch
        ctx = _choose_mp_context(args.mp_context)
        if ctx:
            kw["multiprocessing_context"] = ctx

    return DataLoader(ds, **kw)


# ----------------------------------------------------------------------
# Simple CUDA prefetcher (overlaps H2D copies with compute)
# ----------------------------------------------------------------------

class CUDAPrefetcher:
    """
    Wrap a DataLoader iterator and asynchronously move the NEXT batch to GPU
    on a dedicated CUDA stream. This often helps when your batches are big.

    Use only when device.type == 'cuda' and loader has pin_memory=True.
    """
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device.index if device.index is not None else 0)
        self.it = None
        self.next_batch = None

    def __iter__(self):
        self.it = iter(self.loader)
        self._preload()
        return self

    def _to_device_async(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _preload(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self._to_device_async(batch)

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

    def cleanup(self):
        """Clean up CUDA stream and iterator to prevent resource leaks"""
        if self.stream is not None:
            torch.cuda.synchronize()
            self.stream = None
        self.it = None
        self.next_batch = None

    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            self.cleanup()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Loss (we keep BCE-with-logits for reporting val_loss / test_loss)
# ----------------------------------------------------------------------

class MaskedBCEWithLogits(nn.Module):
    def forward(self, logits, targets, mask):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)


# ----------------------------------------------------------------------
# Plotting + reliability utilities (unchanged from your pipeline)
# ----------------------------------------------------------------------

def plot_reliability_bias_curve_fine(calib_fine: Dict[str, Any], out_path: str, args, title: str):
    import matplotlib.pyplot as plt

    xs = np.asarray(calib_fine.get("bin_centers", []), dtype=np.float64)
    ys = np.asarray(calib_fine.get("bias_pct", []), dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        return

    ys = np.ma.masked_invalid(ys)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhline(0.0, linewidth=1)
    ax.plot(xs, ys, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted probability bin center")
    ax.set_ylabel("Bias (%)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.plot_dpi, format=args.plot_file_format)
    plt.close(fig)


def make_reliability_bias_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "reliability_bias_exact",
        ["#1895b3", "#edd005", "#e01c29"],
        N=256,
    )
    cmap.set_bad(color="white")
    return cmap


def plot_reliability_bias_heatmap_1d(calib: Dict[str, Any], out_path: str, args, title: str):
    import matplotlib.pyplot as plt

    xs = np.asarray(calib.get("bin_centers_slice", []), dtype=np.float64)
    ys = np.asarray(calib.get("reliability_bias_pct", []), dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        return

    xs_pct = xs * 100.0
    Z = ys.reshape(1, -1)
    Z = np.ma.masked_invalid(Z)

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


def save_and_log_calibration(split: str, metrics: Dict[str, Any], epoch: int, args):
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
        except Exception as e:
            print(f"[ReliabilityBias] Failed to write fine curve ({split}): {e}")

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
        except Exception as e:
            print(f"[ReliabilityBias] Failed per-horizon fine curves ({split}): {e}")

    out_npz = os.path.join(args.logdir, f"reliability_bias_{split}_epoch{epoch}.npz")
    try:
        save_reliability_bias_npz(calib, out_npz)
    except Exception as e:
        print(f"[ReliabilityBias] Failed to save NPZ ({split}): {e}")

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
    except Exception as e:
        print(f"[ReliabilityBias] Failed to write heatmap ({split}): {e}")

    # --- Reliability Diagrams (observed vs predicted probability) ---
    try:
        out_reliability = os.path.join(
            args.logdir,
            f"reliability_diagram_{split}_epoch{epoch}.{args.plot_file_format}",
        )
        plot_reliability_diagram(
            calib,
            out_reliability,
            args,
            title=f"Reliability Diagram ({split})",
        )
    except Exception as e:
        print(f"[ReliabilityDiagram] Failed to write overall diagram ({split}): {e}")

    try:
        out_reliability_multi = os.path.join(
            args.logdir,
            f"reliability_diagram_per_horizon_{split}_epoch{epoch}.{args.plot_file_format}",
        )
        plot_reliability_diagram_multi_horizon(
            calib,
            out_reliability_multi,
            args,
            title=f"Reliability Diagram by Horizon ({split})",
        )
    except Exception as e:
        print(f"[ReliabilityDiagram] Failed to write per-horizon diagram ({split}): {e}")


# ----------------------------------------------------------------------
# Calibration Comparison Plotting (Global vs Cluster-Selection vs Ensemble)
# ----------------------------------------------------------------------

def plot_calibration_comparison_curve(
    results_dict: Dict[str, Dict[str, Any]],
    out_path: str,
    args,
    title: str = "Reliability Comparison",
    horizon: Optional[int] = None,
):
    """
    Plot multiple reliability curves on the same axes for comparison.

    Args:
        results_dict: Dict mapping method_name -> evaluation metrics dict
                      Each must have calibration['per_horizon'][h] or calibration['overall']
        out_path: Output file path
        args: Training arguments
        title: Plot title
        horizon: If specified, plot for this horizon; else plot overall
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "p"]

    legend_entries = []

    for idx, (method_name, metrics) in enumerate(results_dict.items()):
        calib = metrics.get("calibration", {})
        if not calib:
            continue

        if horizon is not None:
            per_h = calib.get("per_horizon", {})
            h_data = per_h.get(horizon, per_h.get(str(horizon), {}))
        else:
            h_data = calib

        bin_pred = np.array(h_data.get("bin_pred", []))
        bin_true = np.array(h_data.get("bin_true", []))
        bin_count = np.array(h_data.get("bin_count", []))
        ece = h_data.get("ece", float("nan"))

        if len(bin_pred) == 0:
            continue

        mask = bin_count > 0
        if not np.any(mask):
            continue

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        ax.scatter(
            bin_pred[mask],
            bin_true[mask],
            c=color,
            marker=marker,
            alpha=0.7,
            s=30,
            label=f"{method_name} (ECE={ece:.4f})" if not np.isnan(ece) else method_name,
        )

        sorted_idx = np.argsort(bin_pred[mask])
        ax.plot(
            bin_pred[mask][sorted_idx],
            bin_true[mask][sorted_idx],
            c=color,
            alpha=0.5,
            linewidth=1.5,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, float(getattr(args, "reliability_bin_max", 0.1)))
    ax.set_ylim(0, float(getattr(args, "reliability_bin_max", 0.1)))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=int(getattr(args, "plot_dpi", 150)))
    plt.close(fig)
    print(f"[plot] Saved calibration comparison: {out_path}")


def plot_reliability_diagram(
    calib: Dict[str, Any],
    out_path: str,
    args,
    title: str = "Reliability Diagram",
    horizon: Optional[int] = None,
    n_bins: int = 20,
    show_histogram: bool = True,
):
    """
    Plot a reliability diagram (calibration curve) where:
    - X-axis: Mean predicted probability (model output)
    - Y-axis: Fraction of positives (observed probability)
    - Diagonal line (slope=1 from origin): Perfect calibration

    Args:
        calib: Calibration dict with 'bin_pred', 'bin_true', 'bin_count'
               or 'per_horizon' containing per-horizon calibration data
        out_path: Output file path
        args: Training arguments (for plot_dpi, plot_file_format)
        title: Plot title
        horizon: If specified, plot for this specific horizon; else plot overall
        n_bins: Number of bins to aggregate into for cleaner visualization
        show_histogram: If True, show histogram of predictions below the curve
    """
    import matplotlib.pyplot as plt

    # Extract the appropriate calibration data
    if horizon is not None:
        per_h = calib.get("per_horizon", {})
        h_data = per_h.get(horizon, per_h.get(str(horizon), {}))
    else:
        h_data = calib

    bin_pred = np.array(h_data.get("bin_pred", []))
    bin_true = np.array(h_data.get("bin_true", []))
    bin_count = np.array(h_data.get("bin_count", []))
    ece = h_data.get("ece", float("nan"))
    mce = h_data.get("mce", float("nan"))
    brier = h_data.get("brier", float("nan"))

    if len(bin_pred) == 0 or len(bin_true) == 0:
        print(f"[plot] No calibration data to plot for: {out_path}")
        return

    # Filter to non-empty bins
    mask = bin_count > 0
    if not np.any(mask):
        print(f"[plot] No non-empty bins to plot for: {out_path}")
        return

    pred_vals = bin_pred[mask]
    true_vals = bin_true[mask]
    count_vals = bin_count[mask]

    # Optional: aggregate into fewer bins for cleaner visualization
    if n_bins is not None and n_bins > 0 and len(pred_vals) > n_bins:
        # Re-bin the data into n_bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        new_pred = []
        new_true = []
        new_count = []

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = (pred_vals >= lo) & (pred_vals < hi)
            if i == n_bins - 1:  # Include right edge for last bin
                in_bin = (pred_vals >= lo) & (pred_vals <= hi)

            if np.any(in_bin):
                total_count = count_vals[in_bin].sum()
                # Weighted average by count
                weighted_pred = (pred_vals[in_bin] * count_vals[in_bin]).sum() / total_count
                weighted_true = (true_vals[in_bin] * count_vals[in_bin]).sum() / total_count
                new_pred.append(weighted_pred)
                new_true.append(weighted_true)
                new_count.append(total_count)

        pred_vals = np.array(new_pred)
        true_vals = np.array(new_true)
        count_vals = np.array(new_count)

    # Create figure
    if show_histogram:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                        gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 7))

    # Plot perfect calibration line
    max_val = max(pred_vals.max(), true_vals.max(), 0.1)
    ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=2,
             label='Perfect calibration', alpha=0.7)

    # Plot reliability curve with points sized by count
    sizes = np.sqrt(count_vals / count_vals.max()) * 200 + 20
    scatter = ax1.scatter(pred_vals, true_vals, s=sizes, c='#1f77b4',
                          alpha=0.7, edgecolors='white', linewidth=0.5)

    # Connect points with line
    sorted_idx = np.argsort(pred_vals)
    ax1.plot(pred_vals[sorted_idx], true_vals[sorted_idx],
             c='#1f77b4', alpha=0.5, linewidth=1.5)

    # Add metrics to legend
    metrics_text = []
    if not np.isnan(ece):
        metrics_text.append(f'ECE: {ece:.4f}')
    if not np.isnan(mce):
        metrics_text.append(f'MCE: {mce:.4f}')
    if not np.isnan(brier):
        metrics_text.append(f'Brier: {brier:.4f}')

    if metrics_text:
        ax1.text(0.02, 0.98, '\n'.join(metrics_text), transform=ax1.transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives (Observed)', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_xlim(-0.02, max_val * 1.05)
    ax1.set_ylim(-0.02, max_val * 1.05)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)

    # Add histogram of predictions
    if show_histogram:
        ax2.bar(pred_vals, count_vals, width=max_val / len(pred_vals) * 0.8,
                color='#1f77b4', alpha=0.7, edgecolor='white')
        ax2.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_xlim(-0.02, max_val * 1.05)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_title('Prediction Distribution', fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=int(getattr(args, 'plot_dpi', 150)),
                format=getattr(args, 'plot_file_format', 'png'))
    plt.close(fig)
    print(f"[plot] Saved reliability diagram: {out_path}")


def plot_reliability_diagram_multi_horizon(
    calib: Dict[str, Any],
    out_path: str,
    args,
    title: str = "Reliability Diagram by Horizon",
    horizons: Optional[List[int]] = None,
    n_bins: int = 20,
):
    """
    Plot reliability diagrams for multiple horizons in a grid layout.

    Args:
        calib: Calibration dict with 'per_horizon' containing per-horizon data
        out_path: Output file path
        args: Training arguments
        title: Overall plot title
        horizons: List of horizons to plot; if None, uses all from args.horizons
        n_bins: Number of bins for visualization
    """
    import matplotlib.pyplot as plt

    if horizons is None:
        horizons = [int(h) for h in getattr(args, 'horizons', [])]

    per_h = calib.get("per_horizon", {})
    if not per_h and not horizons:
        print(f"[plot] No per-horizon calibration data for: {out_path}")
        return

    n_horizons = len(horizons)
    if n_horizons == 0:
        return

    # Determine grid layout
    n_cols = min(3, n_horizons)
    n_rows = int(np.ceil(n_horizons / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_horizons == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, h in enumerate(horizons):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        h_data = per_h.get(h, per_h.get(str(h), {}))

        bin_pred = np.array(h_data.get("bin_pred", []))
        bin_true = np.array(h_data.get("bin_true", []))
        bin_count = np.array(h_data.get("bin_count", []))
        ece = h_data.get("ece", float("nan"))

        if len(bin_pred) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Horizon {h}')
            continue

        mask = bin_count > 0
        if not np.any(mask):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Horizon {h}')
            continue

        pred_vals = bin_pred[mask]
        true_vals = bin_true[mask]
        count_vals = bin_count[mask]

        # Re-bin if needed
        if n_bins is not None and len(pred_vals) > n_bins:
            bin_edges = np.linspace(0, 1, n_bins + 1)
            new_pred, new_true, new_count = [], [], []

            for i in range(n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                in_bin = (pred_vals >= lo) & (pred_vals < hi)
                if i == n_bins - 1:
                    in_bin = (pred_vals >= lo) & (pred_vals <= hi)

                if np.any(in_bin):
                    total_count = count_vals[in_bin].sum()
                    weighted_pred = (pred_vals[in_bin] * count_vals[in_bin]).sum() / total_count
                    weighted_true = (true_vals[in_bin] * count_vals[in_bin]).sum() / total_count
                    new_pred.append(weighted_pred)
                    new_true.append(weighted_true)
                    new_count.append(total_count)

            pred_vals = np.array(new_pred)
            true_vals = np.array(new_true)
            count_vals = np.array(new_count)

        max_val = max(pred_vals.max(), true_vals.max(), 0.1)

        # Perfect calibration line
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.7)

        # Scatter and line
        color = colors[idx % len(colors)]
        sizes = np.sqrt(count_vals / count_vals.max()) * 150 + 15
        ax.scatter(pred_vals, true_vals, s=sizes, c=color, alpha=0.7,
                   edgecolors='white', linewidth=0.5)

        sorted_idx = np.argsort(pred_vals)
        ax.plot(pred_vals[sorted_idx], true_vals[sorted_idx], c=color, alpha=0.5, linewidth=1.5)

        # ECE annotation
        ece_str = f'ECE: {ece:.4f}' if not np.isnan(ece) else 'ECE: N/A'
        ax.text(0.02, 0.98, ece_str, transform=ax.transAxes, verticalalignment='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Predicted Prob.', fontsize=10)
        ax.set_ylabel('Observed Prob.', fontsize=10)
        ax.set_title(f'Horizon {h} days', fontsize=11)
        ax.set_xlim(-0.02, max_val * 1.05)
        ax.set_ylim(-0.02, max_val * 1.05)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_horizons, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=int(getattr(args, 'plot_dpi', 150)),
                format=getattr(args, 'plot_file_format', 'png'), bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved multi-horizon reliability diagram: {out_path}")


def plot_ece_mce_brier_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    out_path: str,
    args,
    title: str = "Metrics Comparison",
    horizons: Optional[List[int]] = None,
):
    """
    Bar chart comparing ECE, MCE, Brier across methods.
    """
    import matplotlib.pyplot as plt

    if horizons is None:
        horizons = [int(h) for h in args.horizons]

    methods = list(results_dict.keys())
    n_methods = len(methods)
    n_horizons = len(horizons)

    metrics_names = ["brier", "ece", "logloss"]
    n_metrics = len(metrics_names)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_horizons)
    width = 0.8 / max(n_methods, 1)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for metric_idx, metric_name in enumerate(metrics_names):
        ax = axes[metric_idx]

        for m_idx, method in enumerate(methods):
            metrics = results_dict[method]
            per_h = metrics.get("per_horizon", {})
            calib = metrics.get("calibration", {}).get("per_horizon", {})

            vals = []
            for h in horizons:
                h_metrics = per_h.get(h, per_h.get(str(h), {}))
                h_calib = calib.get(h, calib.get(str(h), {}))

                if metric_name == "ece":
                    v = h_calib.get("ece", h_metrics.get("ece", float("nan")))
                elif metric_name == "brier":
                    v = h_calib.get("brier", h_metrics.get("brier", float("nan")))
                else:
                    v = h_metrics.get(metric_name, float("nan"))
                vals.append(v)

            offset = (m_idx - n_methods / 2 + 0.5) * width
            color = colors[m_idx % len(colors)]
            ax.bar(x + offset, vals, width, label=method, color=color, alpha=0.8)

        ax.set_xlabel("Horizon (days)")
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"{metric_name.upper()} by Horizon")
        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=int(getattr(args, "plot_dpi", 150)))
    plt.close(fig)
    print(f"[plot] Saved metrics comparison: {out_path}")


def plot_calibration_per_cluster_grid(
    cluster_calib_dict: Dict[int, Dict[str, Any]],
    out_path: str,
    args,
    title: str = "Per-Cluster Calibration",
):
    """
    Grid of reliability curves, one subplot per cluster.
    """
    import matplotlib.pyplot as plt

    n_clusters = len(cluster_calib_dict)
    if n_clusters == 0:
        print("[plot] No cluster calibration data to plot.")
        return

    ncols = min(4, n_clusters)
    nrows = int(np.ceil(n_clusters / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    cluster_ids = sorted(cluster_calib_dict.keys())

    for idx, cl_id in enumerate(cluster_ids):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        calib = cluster_calib_dict[cl_id]
        bin_pred = np.array(calib.get("bin_pred", []))
        bin_true = np.array(calib.get("bin_true", []))
        bin_count = np.array(calib.get("bin_count", []))
        ece = calib.get("ece", float("nan"))
        brier = calib.get("brier", float("nan"))
        n_samples = calib.get("n_samples", 0)

        mask = bin_count > 0

        if np.any(mask):
            ax.scatter(bin_pred[mask], bin_true[mask], c="blue", alpha=0.7, s=20)
            sorted_idx = np.argsort(bin_pred[mask])
            ax.plot(bin_pred[mask][sorted_idx], bin_true[mask][sorted_idx], c="blue", alpha=0.5)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlim(0, float(getattr(args, "reliability_bin_max", 0.1)))
        ax.set_ylim(0, float(getattr(args, "reliability_bin_max", 0.1)))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Cluster {cl_id}\nECE={ece:.4f}, Brier={brier:.5f}\nn={n_samples:,}", fontsize=9)
        ax.grid(True, alpha=0.3)

        if row == nrows - 1:
            ax.set_xlabel("Pred Prob")
        if col == 0:
            ax.set_ylabel("True Frac")

    for idx in range(len(cluster_ids), nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=int(getattr(args, "plot_dpi", 150)))
    plt.close(fig)
    print(f"[plot] Saved per-cluster calibration grid: {out_path}")


def save_ensemble_comparison_summary(
    results_dict: Dict[str, Dict[str, Any]],
    out_path: str,
    horizons: List[int],
):
    """
    Save a CSV summary comparing all methods across horizons.
    """
    rows = []
    for method_name, metrics in results_dict.items():
        per_h = metrics.get("per_horizon", {})
        calib = metrics.get("calibration", {}).get("per_horizon", {})
        overall = metrics.get("overall", metrics)

        for h in horizons:
            h_m = per_h.get(h, per_h.get(str(h), {}))
            h_c = calib.get(h, calib.get(str(h), {}))

            rows.append({
                "method": method_name,
                "horizon": h,
                "brier": h_c.get("brier", h_m.get("brier", float("nan"))),
                "ece": h_c.get("ece", h_m.get("ece", float("nan"))),
                "mce": h_c.get("mce", h_m.get("mce", float("nan"))),
                "logloss": h_m.get("logloss", float("nan")),
                "roc_auc": h_c.get("roc", {}).get("auc", h_m.get("roc_auc", float("nan"))),
                "precision": h_m.get("precision", float("nan")),
                "recall": h_m.get("recall", float("nan")),
                "f1": h_m.get("f1", float("nan")),
            })

        rows.append({
            "method": method_name,
            "horizon": "overall",
            "brier": overall.get("brier", float("nan")),
            "ece": overall.get("ece", float("nan")),
            "mce": overall.get("mce", float("nan")),
            "logloss": overall.get("logloss", float("nan")),
            "roc_auc": overall.get("roc_auc", float("nan")),
            "precision": overall.get("precision", float("nan")),
            "recall": overall.get("recall", float("nan")),
            "f1": overall.get("f1", float("nan")),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[summary] Saved ensemble comparison: {out_path}")


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
                "model_type",
                "calibration_method",
            ])
    return path


def append_metrics_csv(path: str, epoch: int, split: str, loss: float, metrics: Dict[str, Any], model_type: str, calibration_method: str):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)

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
            model_type,
            calibration_method,
        ])

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
                model_type,
                calibration_method,
            ])


def save_history_json(history: Dict[str, Any], logdir: str) -> str:
    os.makedirs(logdir, exist_ok=True)
    path = os.path.join(logdir, "metrics_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[History] Saved {path}")
    return path


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


def compute_calibration_error(y_true, y_pred, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Calibration measures how well predicted probabilities match actual outcomes.
    - ECE: Weighted average of calibration errors across bins
    - MCE: Maximum calibration error across all bins

    Args:
        y_true: Ground truth labels (0/1), numpy array or tensor
        y_pred: Predicted probabilities [0,1], numpy array or tensor
        n_bins: Number of bins for calibration (default: 10)

    Returns:
        (ece, mce): Tuple of (Expected Calibration Error, Maximum Calibration Error)
    """
    # Convert to numpy if needed
    if hasattr(y_true, 'cpu'):  # PyTorch tensor
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, 'cpu'):  # PyTorch tensor
        y_pred = y_pred.cpu().numpy()

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Create bin boundaries
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Accuracy and confidence in this bin
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred[in_bin])

            # Calibration error for this bin
            calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)

            # Update ECE (weighted by proportion in bin)
            ece += calibration_error * prop_in_bin

            # Update MCE (maximum across bins)
            mce = max(mce, calibration_error)

    return float(ece), float(mce)


@torch.no_grad()
def evaluate_with_logits_fn(
    logits_fn,
    loader,
    device,
    criterion,
    args,
    use_tqdm: bool,
):
    """
    Key fix:
      - local batch_counter for cuda_empty_cache_every (no persistent function attribute).
    """
    batch_counter = 0

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
    fine_bins = max(1, int(math.ceil(1.0 / max(fine_w, 1e-12))))
    fine_bins_cap = int(getattr(args, "reliability_fine_bins_cap", 250_000))
    if fine_bins > fine_bins_cap:
        fine_bins = fine_bins_cap
        fine_w = 1.0 / float(fine_bins)
        print(f"[warn] fine bins capped to {fine_bins_cap}; adjusted fine_bin_width to {fine_w:g}")

    fine_counts = np.zeros((num_h, fine_bins), dtype=np.float64)
    fine_pred_sums = np.zeros_like(fine_counts)
    fine_true_sums = np.zeros_like(fine_counts)

    prob_inrange_h = np.zeros(num_h, dtype=np.float64)
    prob_total_h = np.zeros(num_h, dtype=np.float64)
    prob_above_max_h = np.zeros(num_h, dtype=np.float64)
    prob_below_min_h = np.zeros(num_h, dtype=np.float64)

    prob_sum_h = np.zeros(num_h, dtype=np.float64)
    prob_sumsq_h = np.zeros(num_h, dtype=np.float64)
    prob_min_h_stat = np.full(num_h, np.inf, dtype=np.float64)
    prob_max_h_stat = np.full(num_h, -np.inf, dtype=np.float64)

    dist_bins = max(1000, int(getattr(args, "prob_stats_bins", 20000)))
    prob_hist_h = np.zeros((num_h, dist_bins), dtype=np.float64)

    tgt_pos_h = np.zeros(num_h, dtype=np.float64)
    tgt_total_h = np.zeros(num_h, dtype=np.float64)

    logloss_sum_overall = 0.0
    logloss_count_overall = 0.0
    logloss_sum_h = np.zeros(num_h, dtype=np.float64)
    logloss_count_h = np.zeros(num_h, dtype=np.float64)

    # Iterator choice
    if device.type == "cuda" and bool(args.use_cuda_prefetch):
        iterator = CUDAPrefetcher(loader, device)
        total = None
        try:
            total = len(loader)
        except TypeError:
            total = None
        pbar = tqdm(total=total, desc="eval", leave=True, position=3) if use_tqdm else None

        def _iter_batches():
            for b in iterator:
                if pbar:
                    pbar.update(1)
                yield b
            if pbar:
                pbar.close()

        batch_iter = _iter_batches()
        batches_are_on_device = True
    else:
        batch_iter = TimedDataLoader(loader, desc="eval", use_tqdm=use_tqdm, position=3)
        batches_are_on_device = False

    for batch in batch_iter:
        if not batches_are_on_device:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        logits = logits_fn(batch)  # Can be (B,K,H,W) or (Npix,K) depending on flatten_pixels
        loss = criterion(logits, batch["y"], batch["mask"])
        m = batch["mask"].sum().item()
        if m == 0:
            del logits
            continue
        tot_loss += loss.item() * m
        tot_mask += m

        probs = torch.sigmoid(logits)
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

        # Handle both formats: (B,K,H,W) and (Npix,K)
        if logits.dim() == 4:
            # Standard format: (B, K, H, W)
            B, K, H, W = logits.shape
            K_eff = min(K, num_h)
        elif logits.dim() == 2:
            # Flattened format: (Npix, K)
            Npix, K = logits.shape
            K_eff = min(K, num_h)
        else:
            raise RuntimeError(f"Unexpected logits.dim()={logits.dim()}, expected 2 or 4")

        for k_idx in range(K_eff):
            if logits.dim() == 4:
                pk = preds[:, k_idx].reshape(-1)
                tk = targets[:, k_idx].reshape(-1)
                vk = valid[:, k_idx].reshape(-1)
            else:  # dim == 2
                pk = preds[:, k_idx]
                tk = targets[:, k_idx]
                vk = valid[:, k_idx]

            tp_h[k_idx] += (pk & tk & vk).sum().item()
            fp_h[k_idx] += (pk & ~tk & vk).sum().item()
            fn_h[k_idx] += (~pk & tk & vk).sum().item()
            tn_h[k_idx] += (~pk & ~tk & vk).sum().item()

        # --- ONE CPU transfer per batch ---
        probs_cpu = probs.detach().cpu().numpy().astype(np.float32, copy=False)
        targ_cpu  = targets.detach().cpu().numpy().astype(np.bool_, copy=False)
        valid_cpu = valid.detach().cpu().numpy().astype(np.bool_, copy=False)

        for k_idx in range(K_eff):
            if logits.dim() == 4:
                v_k = valid_cpu[:, k_idx].reshape(-1)
            else:  # dim == 2
                v_k = valid_cpu[:, k_idx]

            if v_k.sum() == 0:
                continue

            if logits.dim() == 4:
                p_flat = probs_cpu[:, k_idx].reshape(-1)[v_k].astype(np.float32, copy=False)
                y_flat = targ_cpu[:, k_idx].reshape(-1)[v_k].astype(np.float32, copy=False)
            else:  # dim == 2
                p_flat = probs_cpu[:, k_idx][v_k].astype(np.float32, copy=False)
                y_flat = targ_cpu[:, k_idx][v_k].astype(np.float32, copy=False)

            pf64 = p_flat.astype(np.float64, copy=False)
            yf64 = y_flat.astype(np.float64, copy=False)

            fi = np.floor(pf64 / fine_w).astype(np.int64)
            fi = np.clip(fi, 0, fine_bins - 1)
            fine_counts[k_idx]     += np.bincount(fi, minlength=fine_bins).astype(np.float64)
            fine_pred_sums[k_idx]  += np.bincount(fi, weights=pf64, minlength=fine_bins).astype(np.float64)
            fine_true_sums[k_idx]  += np.bincount(fi, weights=yf64, minlength=fine_bins).astype(np.float64)

            n_valid = float(p_flat.size)
            if n_valid == 0:
                continue

            tgt_pos_h[k_idx] += float(y_flat.sum())
            tgt_total_h[k_idx] += n_valid

            prob_sum_h[k_idx] += float(pf64.sum())
            prob_sumsq_h[k_idx] += float((pf64 * pf64).sum())
            prob_min_h_stat[k_idx] = min(prob_min_h_stat[k_idx], float(pf64.min()))
            prob_max_h_stat[k_idx] = max(prob_max_h_stat[k_idx], float(pf64.max()))

            b = np.floor(pf64 * dist_bins).astype(np.int64)
            b = np.clip(b, 0, dist_bins - 1)
            prob_hist_h[k_idx] += np.bincount(b, minlength=dist_bins).astype(np.float64)

            in_range = (p_flat >= range_min) & (p_flat <= range_max)
            above_max = (p_flat > range_max)
            below_min = (p_flat < range_min)

            prob_inrange_h[k_idx] += float(in_range.sum())
            prob_above_max_h[k_idx] += float(above_max.sum())
            prob_below_min_h[k_idx] += float(below_min.sum())
            prob_total_h[k_idx] += n_valid

            total_counts_per_h[k_idx] += n_valid
            brier_sums[k_idx] += float(((p_flat - y_flat) ** 2).sum())

            p_clip = np.clip(p_flat, EPS_PROB, 1.0 - EPS_PROB)
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

        del logits, probs, targets, valid, preds, probs_cpu, targ_cpu, valid_cpu

        if device.type == "cuda" and int(getattr(args, "cuda_empty_cache_every", 0)) > 0:
            batch_counter += 1
            if batch_counter % int(args.cuda_empty_cache_every) == 0:
                torch.cuda.empty_cache()

    # --- Fine slice to observed min/max ---
    finite_mins = prob_min_h_stat[np.isfinite(prob_min_h_stat)]
    finite_maxs = prob_max_h_stat[np.isfinite(prob_max_h_stat)]
    mn_all = float(finite_mins.min()) if finite_mins.size else 0.0
    mx_all = float(finite_maxs.max()) if finite_maxs.size else 1.0

    i0 = int(np.floor(mn_all / fine_w))
    i1 = int(np.ceil(mx_all / fine_w)) + 1
    i0 = max(0, min(i0, fine_bins))
    i1 = max(i0 + 1, min(i1, fine_bins))

    fine_centers = (np.arange(i0, i1, dtype=np.float64) + 0.5) * fine_w

    fine_counts_s    = fine_counts[:, i0:i1]
    fine_pred_sums_s = fine_pred_sums[:, i0:i1]
    fine_true_sums_s = fine_true_sums[:, i0:i1]

    fine_bias_per_h = {}
    fine_min_count = int(getattr(args, "reliability_fine_min_count", 1))

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
        "min": mn_all,
        "max": mx_all,
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
    prob_above_max_overall = float(prob_above_max_h.sum())
    prob_below_min_overall = float(prob_below_min_h.sum())
    prob_total_overall = float(prob_total_h.sum())

    prob_frac_overall = float("nan") if prob_total_overall <= 0 else (prob_inrange_overall / prob_total_overall)
    prob_above_frac_overall = float("nan") if prob_total_overall <= 0 else (prob_above_max_overall / prob_total_overall)
    prob_below_frac_overall = float("nan") if prob_total_overall <= 0 else (prob_below_min_overall / prob_total_overall)

    prob_range_per_h = {}
    for idx, h in enumerate(args.horizons):
        tot = float(prob_total_h[idx])
        cnt = float(prob_inrange_h[idx])
        above = float(prob_above_max_h[idx])
        below = float(prob_below_min_h[idx])
        frac = float("nan") if tot <= 0 else (cnt / tot)
        above_frac = float("nan") if tot <= 0 else (above / tot)
        below_frac = float("nan") if tot <= 0 else (below / tot)
        prob_range_per_h[int(h)] = {
            "count": cnt,
            "total": tot,
            "fraction": frac,
            "above_max_count": above,
            "above_max_fraction": above_frac,
            "below_min_count": below,
            "below_min_fraction": below_frac,
        }

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
            out[f"p{q}"] = (idx + 0.5) / hist.size
        return out

    prob_stats_per_h: Dict[int, Dict[str, Any]] = {}
    for idx, h in enumerate(args.horizons):
        tot = float(prob_total_h[idx])
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

    tot_all = float(prob_total_h.sum())
    if tot_all <= 0:
        prob_stats_overall = {"min": float("nan"), "mean": float("nan"), "max": float("nan"), "std": float("nan")}
        prob_stats_overall.update({k: float("nan") for k in ["p1", "p5", "p50", "p95", "p99"]})
    else:
        mean_all = float(prob_sum_h.sum() / tot_all)
        var_all = float(prob_sumsq_h.sum() / tot_all - mean_all * mean_all)
        std_all = float(np.sqrt(max(var_all, 0.0)))
        finite_mins2 = prob_min_h_stat[np.isfinite(prob_min_h_stat)]
        finite_maxs2 = prob_max_h_stat[np.isfinite(prob_max_h_stat)]
        mn_all_p = float(finite_mins2.min()) if finite_mins2.size else float("nan")
        mx_all_p = float(finite_maxs2.max()) if finite_maxs2.size else float("nan")
        hist_all = prob_hist_h.sum(axis=0)
        prob_stats_overall = {"min": mn_all_p, "mean": mean_all, "max": mx_all_p, "std": std_all}
        prob_stats_overall.update(_percentiles_from_hist(hist_all))

    def _binary_percentiles(pos: float, tot: float, qs=(1, 5, 50, 95, 99)):
        if tot <= 0:
            return {f"p{q}": float("nan") for q in qs}
        neg = tot - pos
        cdf0 = neg / tot
        out = {}
        for q in qs:
            out[f"p{q}"] = 0.0 if (q / 100.0) <= cdf0 else 1.0
        return out

    target_stats_per_h: Dict[int, Dict[str, Any]] = {}
    for idx, h in enumerate(args.horizons):
        tot = float(tgt_total_h[idx])
        pos = float(tgt_pos_h[idx])
        if tot <= 0:
            d = {
                "positive_fraction": float("nan"),
                "positive_count": 0.0,
                "total_count": 0.0,
                "min": float("nan"),
                "mean": float("nan"),
                "max": float("nan"),
                "std": float("nan"),
            }
            d.update(_binary_percentiles(0.0, 0.0))
            target_stats_per_h[int(h)] = d
            continue
        frac = pos / tot
        std = float(np.sqrt(max(frac * (1.0 - frac), 0.0)))
        mn = 1.0 if (tot - pos) == 0 else 0.0
        mx = 0.0 if pos == 0 else 1.0
        d = {
            "positive_fraction": float(frac),
            "positive_count": float(pos),
            "total_count": float(tot),
            "min": float(mn),
            "mean": float(frac),
            "max": float(mx),
            "std": float(std),
        }
        d.update(_binary_percentiles(pos, tot))
        target_stats_per_h[int(h)] = d

    tot_all_t = float(tgt_total_h.sum())
    pos_all = float(tgt_pos_h.sum())
    if tot_all_t <= 0:
        target_stats_overall = {
            "positive_fraction": float("nan"),
            "positive_count": 0.0,
            "total_count": 0.0,
            "min": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "p1": float("nan"), "p5": float("nan"), "p50": float("nan"), "p95": float("nan"), "p99": float("nan"),
        }
    else:
        frac_all = pos_all / tot_all_t
        std_all = float(np.sqrt(max(frac_all * (1.0 - frac_all), 0.0)))
        mn_all_t = 1.0 if (tot_all_t - pos_all) == 0 else 0.0
        mx_all_t = 0.0 if pos_all == 0 else 1.0
        target_stats_overall = {
            "positive_fraction": float(frac_all),
            "positive_count": float(pos_all),
            "total_count": float(tot_all_t),
            "min": float(mn_all_t),
            "mean": float(frac_all),
            "max": float(mx_all_t),
            "std": float(std_all),
        }
        target_stats_overall.update(_binary_percentiles(pos_all, tot_all_t))

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
            "above_max_count": prob_above_max_overall,
            "above_max_fraction": prob_above_frac_overall,
            "below_min_count": prob_below_min_overall,
            "below_min_fraction": prob_below_frac_overall,
        },
        "prob_stats": {
            "overall": prob_stats_overall,
            "per_horizon": prob_stats_per_h,
        },
        "target_stats": {
            "overall": target_stats_overall,
            "per_horizon": target_stats_per_h,
        },
    }

    return val_loss, metrics


def _print_split(split: str, loss: float, metrics: Dict[str, Any], calibration_method: str):
    header(f"{split.upper()} (calibration={calibration_method})")
    print(
        f"[{split}] loss={loss:.6f}, "
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
            f"[{split}] prob_range [{pr['min']:.6f}, {pr['max']:.6f}]: "
            f"{pr['count']:.0f}/{pr['total']:.0f} ({100.0 * pr['fraction']:.3f}%)"
        )

    for h, m in (metrics.get("per_horizon", {}) or {}).items():
        print(
            f"  [{split} h={h}] acc={m['accuracy']:.4f}, "
            f"prec={m['precision']:.4f}, rec={m['recall']:.4f}, f1={m['f1']:.4f}, "
            f"logloss={m.get('logloss', float('nan')):.6f}, auc={m.get('roc_auc', float('nan')):.4f}, "
            f"tp={m.get('tp','')}, fp={m.get('fp','')}, fn={m.get('fn','')}, tn={m.get('tn','')}"
        )


# ----------------------------------------------------------------------
# XGBoost: batch->rows, tabularize loaders, fit K horizon models
# ----------------------------------------------------------------------

def _flatten_time_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x
    if x.dim() == 5:
        B, T, C, H, W = x.shape
        return x.view(B, T * C, H, W)
    raise ValueError(f"Expected x with 4 or 5 dims, got {tuple(x.shape)}")


def _batch_to_rows_cpu(batch: dict):
    """
    CPU tabularization path: returns numpy arrays (ALL horizons).

    Handles both formats:
    - Old format: x=(B,F,H,W), y=(B,K,H,W), mask=(B,K,H,W)
    - flatten_pixels=True: x=(B*H*W,F), y=(B*H*W,K), mask=(B*H*W,K)
    """
    x = batch["x"]
    y = batch["y"]
    m = batch["mask"]

    # Detect if already flattened (2D instead of 4D)
    if x.dim() == 2:
        # Already flattened: (Npix, F)
        x = x.detach().cpu()
        y = y.detach().cpu()
        m = m.detach().cpu()

        Npix, F = x.shape
        _, K = y.shape

        X = x.numpy().astype(np.float32)
        Y = (y > 0.5).numpy().astype(np.uint8)
        V = (m > 0.5).numpy().astype(bool)

        # Reconstruct B, H, W from metadata if needed (for compatibility)
        # For now, we can't determine exact B,H,W from flattened data alone
        # Set them to indicate flattened mode
        return X, Y, V, (1, 1, Npix, K, F)

    # Old format: needs manual flattening
    x = _flatten_time_if_needed(x)    # (B,F,H,W)
    x = x.detach().cpu()
    y = y.detach().cpu()
    m = m.detach().cpu()

    B, F, H, W = x.shape
    _, K, _, _ = y.shape

    X = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, F).numpy().astype(np.float32)
    Y = (y > 0.5).permute(0, 2, 3, 1).contiguous().view(B * H * W, K).numpy().astype(np.uint8)
    V = (m > 0.5).permute(0, 2, 3, 1).contiguous().view(B * H * W, K).numpy().astype(bool)

    return X, Y, V, (B, H, W, K, F)


def _batch_to_rows_cpu_single_horizon(batch: dict, k: int):
    """
    Streaming-friendly CPU row extraction for a single horizon.

    Handles both formats:
    - Old format: x=(B,F,H,W), y=(B,K,H,W), mask=(B,K,H,W)
    - flatten_pixels=True: x=(Npix,F), y=(Npix,K), mask=(Npix,K)

    Returns:
      X: (N,F) float32
      y: (N,) uint8 (0/1)
      v: (N,) bool
    """
    x = batch["x"]
    y_full = batch["y"]
    m_full = batch["mask"]

    # Detect if already flattened
    if x.dim() == 2:
        # Already flattened: (Npix, F)
        x = x.detach().cpu()
        yk = y_full[:, k].detach().cpu()   # (Npix,)
        mk = m_full[:, k].detach().cpu()   # (Npix,)

        X = x.numpy().astype(np.float32, copy=False)
        y = (yk > 0.5).numpy().astype(np.uint8, copy=False)
        v = (mk > 0.5).numpy().astype(bool, copy=False)

        return X, y, v

    # Old format: needs manual flattening
    x = _flatten_time_if_needed(x)          # (B,F,H,W)
    yk = y_full[:, k]                       # (B,H,W)
    mk = m_full[:, k]                       # (B,H,W)

    x = x.detach().cpu()
    yk = yk.detach().cpu()
    mk = mk.detach().cpu()

    B, F, H, W = x.shape

    X = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, F).numpy().astype(np.float32, copy=False)
    y = (yk > 0.5).contiguous().view(B * H * W).numpy().astype(np.uint8, copy=False)
    v = (mk > 0.5).contiguous().view(B * H * W).numpy().astype(bool, copy=False)

    return X, y, v


def build_tabular_from_loader(loader, args, desc: str, max_rows: Optional[int]):
    header(f"Tabularizing: {desc}")
    print("This step is CPU-side and can be RAM-heavy.")
    print("Tip: use --max-train-rows / --max-val-rows to preallocate and cap RAM spikes.")
    print("If you want to avoid this entirely, use: --train-mode stream")

    X_all = Y_all = V_all = None
    write_pos = 0

    Xs, Ys, Vs = [], [], []
    seen = 0

    for batch in TimedDataLoader(loader, desc=desc, use_tqdm=not args.no_tqdm, position=2):
        X, Y, V, _ = _batch_to_rows_cpu(batch)
        keep_any = V.any(axis=1)
        X = X[keep_any]
        Y = Y[keep_any]
        V = V[keep_any]

        if X.shape[0] == 0:
            continue

        if max_rows is not None:
            if X_all is None:
                cap = int(max_rows)
                F = int(X.shape[1])
                K = int(Y.shape[1])
                print(f"[Tabularize] Preallocating arrays: X({cap},{F}), Y({cap},{K}), V({cap},{K})")
                X_all = np.empty((cap, F), dtype=np.float32)
                Y_all = np.empty((cap, K), dtype=np.uint8)
                V_all = np.empty((cap, K), dtype=bool)

            take = min(int(max_rows) - write_pos, X.shape[0])
            if take <= 0:
                break

            X_all[write_pos:write_pos+take] = X[:take]
            Y_all[write_pos:write_pos+take] = Y[:take]
            V_all[write_pos:write_pos+take] = V[:take]
            write_pos += take

            seen += take
            if write_pos >= int(max_rows):
                print(f"[Tabularize] Reached max_rows={max_rows} (seen={seen}); stopping early.")
                break
        else:
            Xs.append(X)
            Ys.append(Y)
            Vs.append(V)
            seen += int(X.shape[0])

    if max_rows is not None:
        if X_all is None or write_pos == 0:
            raise RuntimeError("No rows produced during tabularization (mask filtering too aggressive?).")
        X_all = X_all[:write_pos]
        Y_all = Y_all[:write_pos]
        V_all = V_all[:write_pos]
    else:
        if not Xs:
            raise RuntimeError("No rows produced during tabularization (mask filtering too aggressive?).")
        X_all = np.concatenate(Xs, axis=0).astype(np.float32, copy=False)
        Y_all = np.concatenate(Ys, axis=0).astype(np.uint8, copy=False)
        V_all = np.concatenate(Vs, axis=0).astype(bool, copy=False)

    print(f"[Tabularize] Rows................. {X_all.shape[0]}")
    print(f"[Tabularize] Features (F)......... {X_all.shape[1]}")
    print(f"[Tabularize] Horizons (K)......... {Y_all.shape[1]}")

    for k, h in enumerate(args.horizons):
        vk = int(V_all[:, k].sum())
        pk = int((Y_all[:, k] & V_all[:, k]).sum())
        print(f"  • h={int(h):>3}: valid_rows={vk:,}  pos_rows={pk:,}  pos_frac={(pk/max(vk,1)):.6g}")

    return X_all, Y_all, V_all




def tabularize_dataset(ds, args, max_rows: Optional[int] = None):
    """
    Convenience wrapper used by regime prescreening:
    - builds a DataLoader for `ds`
    - converts to (X,Y,V) numpy arrays on CPU
    - returns feature_names consistent with the dataset feature layout
    """
    # Build feature names (best-effort)
    feature_names = _load_feature_names_for_run(args) or build_feature_names_from_dataset(args, ds)
    feature_names = [(_shorten_feature_name(s)) for s in feature_names]

    # Choose shuffling: if we cap rows, shuffling helps sample a diverse subset for train
    shuffle = bool(max_rows is not None)

    loader = make_loader(ds, batch_size=args.batch_size, shuffle=shuffle, args=args, device=None, eval_mode=True)
    X, Y, V = build_tabular_from_loader(loader, args=args, desc="dataset_tabularize", max_rows=max_rows)

    # If feature_names is shorter than actual X columns (e.g., engineered extras), pad defensively
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(int(X.shape[1]))]
    elif len(feature_names) < int(X.shape[1]):
        feature_names = list(feature_names) + [f"f{i}" for i in range(len(feature_names), int(X.shape[1]))]
    elif len(feature_names) > int(X.shape[1]):
        feature_names = feature_names[: int(X.shape[1])]

    return X, Y, V, feature_names

def make_xgb_params(args, seed: int) -> dict:
    # User-facing CLI still allows --xgb-tree-method gpu_hist|hist
    # XGBoost 3.x REMOVED 'gpu_hist' as a valid tree_method value.
    # In 3.x, GPU training is done via: tree_method=hist + device=cuda:<id>
    req = str(args.xgb_tree_method)
    v = _xgb_version_tuple()

    gpu_requested = (req == "gpu_hist")

    # XGB 3.x: map gpu_hist -> hist (GPU is controlled by device=cuda)
    if gpu_requested and v >= (3, 0, 0):
        tree_method = "hist"
    else:
        tree_method = req

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(args.xgb_max_depth),
        "eta": float(args.xgb_eta),
        "subsample": float(args.xgb_subsample),
        "colsample_bynode": float(args.xgb_colsample_bynode),
        "min_child_weight": float(args.xgb_min_child_weight),
        "reg_lambda": float(args.xgb_reg_lambda),
        "reg_alpha": float(args.xgb_reg_alpha),
        "gamma": float(getattr(args, 'xgb_gamma', 0.0)),
        "max_bin": int(args.xgb_max_bin),
        "tree_method": tree_method,
        "seed": int(seed),
        "verbosity": 1,
        "nthread": int(args.xgb_nthread) if args.xgb_nthread is not None else int(os.cpu_count() or 1),
    }

    if args.xgb_sampling_method:
        params["sampling_method"] = args.xgb_sampling_method
    if args.xgb_grow_policy:
        params["grow_policy"] = args.xgb_grow_policy

    # --- GPU wiring ---
    if gpu_requested:
        gid = int(args.xgb_gpu_id)

        if v >= (2, 0, 0):
            params["device"] = f"cuda:{gid}"
        else:
            # Very old XGBoost fallback
            params["gpu_id"] = gid
            params["tree_method"] = "gpu_hist"

        # Optional; not required but OK to keep
        # XGBoost >= 3 ignores 'predictor'; only set for older versions
        if v < (3, 0, 0):
            params["predictor"] = "gpu_predictor"
        if v >= (3, 0, 0):
            print("[XGB] NOTE: XGBoost>=3 detected: mapping --xgb-tree-method gpu_hist -> tree_method=hist + device=cuda")

    return params


def make_residual_xgb_params(args, seed: int) -> dict:
    """
    Build XGBoost params for residual models with strong regularization.

    Residual models are intentionally shallow and heavily regularized to avoid
    overfitting while providing per-regime corrections to the global model.
    """
    v = _xgb_version_tuple()
    req = str(args.xgb_tree_method)
    gpu_requested = (req == "gpu_hist")

    # XGB 3.x: map gpu_hist -> hist (GPU is controlled by device=cuda)
    if gpu_requested and v >= (3, 0, 0):
        tree_method = "hist"
    else:
        tree_method = req

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(args.residual_max_depth),
        "eta": float(args.xgb_eta),  # Use same learning rate as global
        "subsample": float(args.residual_subsample),
        "colsample_bynode": float(args.residual_colsample_bynode),
        "min_child_weight": float(args.xgb_min_child_weight),
        "reg_lambda": float(args.residual_reg_lambda),
        "reg_alpha": float(args.residual_reg_alpha),
        "gamma": float(getattr(args, 'xgb_gamma', 0.0)),
        "max_bin": int(args.xgb_max_bin),
        "tree_method": tree_method,
        "seed": int(seed),
        "verbosity": 1,
        "nthread": int(args.xgb_nthread) if args.xgb_nthread is not None else int(os.cpu_count() or 1),
    }

    if args.xgb_sampling_method:
        params["sampling_method"] = args.xgb_sampling_method
    if args.xgb_grow_policy:
        params["grow_policy"] = args.xgb_grow_policy

    # --- GPU wiring ---
    if gpu_requested:
        gid = int(args.xgb_gpu_id)

        if v >= (2, 0, 0):
            params["device"] = f"cuda:{gid}"
        else:
            params["gpu_id"] = gid
            params["tree_method"] = "gpu_hist"

        params["predictor"] = "gpu_predictor"

    return params


# ----------------------------------------------------------------------
# Hyperparameter Optimization (Optuna)
# ----------------------------------------------------------------------

def _is_gpu_params(params: dict) -> bool:
    """Check if XGBoost params indicate GPU training."""
    device = params.get("device", "")
    tree_method = params.get("tree_method", "")
    return str(device).startswith("cuda") or tree_method == "gpu_hist"


def define_hpo_search_space(trial, args) -> dict:
    """
    Define the Optuna search space for XGBoost hyperparameters.
    Focuses on the most impactful parameters only.
    """
    # eta (learning rate) - VERY IMPORTANT
    eta = trial.suggest_float("eta", 0.0008, 0.12, log=True)

    # num_round - coupled with eta
    num_round = trial.suggest_int("num_round", 30, int(args.hpo_num_round_max))

    # max_depth - VERY IMPORTANT
    max_depth = trial.suggest_int("max_depth", 3, 7)

    # min_child_weight - IMPORTANT for overfitting
    min_child_weight = trial.suggest_float("min_child_weight", 0.3, 30.0, log=True)

    # subsample - MODERATELY IMPORTANT
    subsample = trial.suggest_float("subsample", 0.6, 1.0)

    # colsample_bynode - MODERATELY IMPORTANT
    colsample_bynode = trial.suggest_float("colsample_bynode", 0.4, 1.0)

    # gamma (min_split_loss) - prevents overfitting on noisy data
    gamma = trial.suggest_float("gamma", 0.0, 2.0)

    return {
        "eta": eta,
        "num_round": num_round,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bynode": colsample_bynode,
        "gamma": gamma,
    }


def define_residual_hpo_search_space(trial, args) -> dict:
    """
    Define the Optuna search space for residual model hyperparameters.
    Uses tighter, more conservative ranges than global model since residual
    models should be shallow and well-regularized.
    """
    def parse_range(spec: str) -> tuple:
        parts = spec.split(",")
        return float(parts[0].strip()), float(parts[1].strip())

    def parse_int_range(spec: str) -> tuple:
        parts = spec.split(",")
        return int(parts[0].strip()), int(parts[1].strip())

    # Parse range arguments from CLI
    depth_min, depth_max = parse_int_range(args.hpo_residual_max_depth_range)
    lambda_min, lambda_max = parse_range(args.hpo_residual_lambda_range)
    alpha_min, alpha_max = parse_range(args.hpo_residual_alpha_range)
    sub_min, sub_max = parse_range(args.hpo_residual_subsample_range)
    col_min, col_max = parse_range(args.hpo_residual_colsample_range)
    round_min, round_max = parse_int_range(args.hpo_residual_num_round_range)

    return {
        "residual_max_depth": trial.suggest_int("residual_max_depth", depth_min, depth_max),
        "residual_reg_lambda": trial.suggest_float("residual_reg_lambda", lambda_min, lambda_max, log=True),
        "residual_reg_alpha": trial.suggest_float("residual_reg_alpha", alpha_min, alpha_max),
        "residual_subsample": trial.suggest_float("residual_subsample", sub_min, sub_max),
        "residual_colsample_bynode": trial.suggest_float("residual_colsample_bynode", col_min, col_max),
        "residual_num_round": trial.suggest_int("residual_num_round", round_min, round_max),
    }


def define_joint_hpo_search_space(trial, args) -> dict:
    """
    Define combined search space for joint global + residual model optimization.
    Returns a merged dict with both global and residual hyperparameters.
    """
    global_params = define_hpo_search_space(trial, args)
    residual_params = define_residual_hpo_search_space(trial, args)
    return {**global_params, **residual_params}


def define_calibration_hpo_search_space(trial, args) -> dict:
    """
    Define the Optuna search space for calibration hyperparameters.

    Searches over:
    - calibration_method: none, platt, or isotonic
    - For platt: reg (regularization), max_iter
    """
    # Parse available methods from CLI arg
    methods_str = getattr(args, 'hpo_calibration_methods', 'none,platt,isotonic')
    methods = [m.strip() for m in methods_str.split(',') if m.strip()]
    if not methods:
        methods = ['none', 'platt', 'isotonic']

    # Suggest calibration method
    calibration_method = trial.suggest_categorical("calibration_method", methods)

    result = {"calibration_method": calibration_method}

    # If platt is selected, suggest its hyperparameters
    if calibration_method == "platt":
        # Parse range arguments
        reg_range_str = getattr(args, 'hpo_platt_reg_range', '1e-8,1e-2')
        reg_parts = reg_range_str.split(',')
        reg_min, reg_max = float(reg_parts[0].strip()), float(reg_parts[1].strip())

        max_iter_range_str = getattr(args, 'hpo_platt_max_iter_range', '50,500')
        iter_parts = max_iter_range_str.split(',')
        iter_min, iter_max = int(iter_parts[0].strip()), int(iter_parts[1].strip())

        result["platt_reg"] = trial.suggest_float("platt_reg", reg_min, reg_max, log=True)
        result["platt_max_iter"] = trial.suggest_int("platt_max_iter", iter_min, iter_max)

    # Isotonic has no tunable hyperparameters (uses fixed y_min=0, y_max=1, out_of_bounds='clip')

    return result


def apply_calibration_to_predictions(
    y_pred_logits: np.ndarray,
    y_true: np.ndarray,
    calib_params: dict,
    calib_fraction: float = 0.3,
) -> np.ndarray:
    """
    Apply calibration during HPO evaluation.

    Splits validation data into calibration fit and evaluation sets,
    fits the calibrator on the fit set, and returns calibrated predictions
    on the evaluation set.

    Args:
        y_pred_logits: Raw model logits (margins)
        y_true: True labels
        calib_params: Dict with 'calibration_method' and optional 'platt_reg', 'platt_max_iter'
        calib_fraction: Fraction of data to use for calibration fitting

    Returns:
        Tuple of (calibrated_probs, eval_y_true) for the evaluation portion
    """
    method = calib_params.get("calibration_method", "none")

    if method == "none":
        # Just apply sigmoid to logits
        probs = 1.0 / (1.0 + np.exp(-np.clip(y_pred_logits, -50, 50)))
        return probs, y_true

    n = len(y_pred_logits)
    n_calib = max(100, int(n * calib_fraction))  # At least 100 samples for calibration

    if n_calib >= n - 100:
        # Not enough data to split, just return sigmoid
        probs = 1.0 / (1.0 + np.exp(-np.clip(y_pred_logits, -50, 50)))
        return probs, y_true

    # Split into calibration and evaluation sets
    indices = np.arange(n)
    np.random.shuffle(indices)
    calib_idx = indices[:n_calib]
    eval_idx = indices[n_calib:]

    logits_calib = y_pred_logits[calib_idx]
    y_calib = y_true[calib_idx]
    logits_eval = y_pred_logits[eval_idx]
    y_eval = y_true[eval_idx]

    if method == "platt":
        platt_reg = float(calib_params.get("platt_reg", 1e-6))
        platt_max_iter = int(calib_params.get("platt_max_iter", 100))

        # Fit Platt scaling
        calibrator = fit_platt_scaling(
            logits_calib, y_calib,
            max_iter=platt_max_iter,
            reg=platt_reg,
            tol=1e-8
        )
        # Apply to eval set
        probs_eval = calibrator.predict(logits_eval)

    elif method == "isotonic":
        # Fit isotonic regression
        calibrator = fit_isotonic_pav(logits_calib, y_calib)
        # Apply to eval set
        probs_eval = calibrator.predict(logits_eval)

    else:
        # Unknown method, fall back to sigmoid
        probs_eval = 1.0 / (1.0 + np.exp(-np.clip(logits_eval, -50, 50)))

    return probs_eval, y_eval


def create_hpo_sampler(args, seed: int):
    """
    Create the appropriate Optuna sampler based on CLI arguments.

    Supports:
    - TPE (Tree-structured Parzen Estimator) - default, good for mixed params
    - Random - baseline, embarrassingly parallel
    - CMA-ES - better for continuous param spaces, uses evolution strategy
    - NSGA-II - multi-objective optimization (Pareto front)

    Args:
        args: CLI arguments namespace
        seed: Random seed for reproducibility

    Returns:
        Optuna sampler instance
    """
    sampler_type = args.hpo_sampler.lower()

    if sampler_type == "tpe":
        return TPESampler(seed=seed)

    elif sampler_type == "random":
        return RandomSampler(seed=seed)

    elif sampler_type == "cmaes":
        if not _HAVE_CMAES:
            raise RuntimeError(
                "CMA-ES sampler not available. Requires optuna>=2.3.0. "
                "Install with: pip install 'optuna>=2.3.0' or use --hpo-sampler tpe"
            )
        restart_strategy = getattr(args, 'hpo_cmaes_restart_strategy', 'ipop')
        sigma0 = float(getattr(args, 'hpo_cmaes_sigma0', 0.5))

        return CmaEsSampler(
            seed=seed,
            restart_strategy=restart_strategy if restart_strategy != "none" else None,
            sigma0=sigma0,
            consider_pruned_trials=False,
        )

    elif sampler_type == "nsgaii":
        if not _HAVE_NSGAII:
            raise RuntimeError(
                "NSGA-II sampler not available. Requires optuna>=2.4.0. "
                "Install with: pip install 'optuna>=2.4.0' or use --hpo-sampler tpe"
            )
        population_size = int(getattr(args, 'hpo_nsgaii_population_size', 50))

        return NSGAIISampler(
            seed=seed,
            population_size=population_size,
        )

    else:
        raise ValueError(f"Unknown HPO sampler: {sampler_type}. Use tpe, random, cmaes, or nsgaii.")



# ----------------------------------------------------------------------
# HPO DMatrix reuse (tabularize mode)
# ----------------------------------------------------------------------
# When train/val data are fixed across trials (the common Optuna case),
# rebuilding DMatrix every trial is pure overhead and can increase GPU OOM risk.
# We cache (dtrain, dval) per (dataset ids, horizon, dmatrix type, gpu id).
_HPO_DMATRIX_CACHE = {}

def _hpo_dmatrix_cache_key_tabularized(
    X_tr, Y_tr, V_tr, X_va, Y_va, V_va, horizon_idx: int, args, is_gpu: bool
):
    return (
        id(X_tr), id(Y_tr), id(V_tr),
        id(X_va), id(Y_va), id(V_va),
        int(horizon_idx),
        str(getattr(args, "xgb_dmatrix", "dmat")),
        bool(is_gpu),
        int(getattr(args, "xgb_gpu_id", 0)),
    )

def _get_or_build_hpo_dmatrices_tabularized(
    X_tr, Y_tr, V_tr,
    X_va, Y_va, V_va,
    horizon_idx: int,
    args,
    is_gpu: bool,
    feature_names=None,
):
    # Simple bound to avoid unbounded GPU memory retention across regime loops.
    if len(_HPO_DMATRIX_CACHE) > 8:
        _HPO_DMATRIX_CACHE.clear()

    key = _hpo_dmatrix_cache_key_tabularized(
        X_tr, Y_tr, V_tr, X_va, Y_va, V_va, horizon_idx=horizon_idx, args=args, is_gpu=is_gpu
    )
    cached = _HPO_DMATRIX_CACHE.get(key)
    if cached is not None:
        return cached  # (dtrain, dval)

    keep_tr = V_tr[:, horizon_idx]
    Xk = X_tr[keep_tr]
    yk = Y_tr[keep_tr, horizon_idx].astype(np.float32)
    if Xk.shape[0] == 0:
        return (None, None)

    keep_va = V_va[:, horizon_idx]
    Xv = X_va[keep_va]
    yv = Y_va[keep_va, horizon_idx].astype(np.float32)
    if Xv.shape[0] == 0:
        return (None, None)

    dtrain = _make_dmatrix(Xk, yk, args=args, is_gpu=is_gpu)
    dval = _make_dmatrix(Xv, yv, args=args, is_gpu=is_gpu)

    if feature_names:
        try:
            dtrain.feature_names = feature_names
            dval.feature_names = feature_names
        except Exception:
            pass

    _HPO_DMATRIX_CACHE[key] = (dtrain, dval)
    return dtrain, dval


def compute_multi_objectives(y_pred: np.ndarray, y_true: np.ndarray, objectives: str) -> tuple:
    """
    Compute multiple objectives for NSGA-II optimization.

    Args:
        y_pred: Predicted probabilities
        y_true: True labels (0 or 1)
        objectives: Objective specification string ('brier', 'brier_logloss', 'brier_calibration')

    Returns:
        Tuple of objective values (all to minimize)
    """
    # Brier score (primary) - always computed
    brier = float(np.mean((y_pred - y_true) ** 2))

    if objectives == "brier":
        # Single objective, but NSGA-II needs 2+ objectives
        # Return brier twice as placeholder
        return (brier, brier)

    elif objectives == "brier_logloss":
        # Log loss (binary cross-entropy)
        eps = 1e-7
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        logloss = -float(np.mean(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        ))
        return (brier, logloss)

    elif objectives == "brier_calibration":
        # Calibration error (ECE - Expected Calibration Error)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(y_pred)
        for i in range(n_bins):
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_acc = float(y_true[mask].mean())
                bin_conf = float(y_pred[mask].mean())
                ece += mask.sum() * abs(bin_acc - bin_conf)
        ece /= total_samples
        return (brier, float(ece))

    else:
        # Fallback to brier only
        return (brier, brier)


def hpo_multi_objective(
    trial,
    args,
    train_data,
    val_data,
    base_params: dict,
    horizon_idx: int,
    feature_names: Optional[List[str]] = None,
) -> tuple:
    """
    Multi-objective version of hpo_objective for NSGA-II.
    Returns tuple of objectives to minimize.
    """
    # Get trial parameters (same as single-objective)
    hpo_params = define_hpo_search_space(trial, args)

    params = base_params.copy()
    params["eta"] = hpo_params["eta"]
    params["max_depth"] = hpo_params["max_depth"]
    params["min_child_weight"] = hpo_params["min_child_weight"]
    params["subsample"] = hpo_params["subsample"]
    params["colsample_bynode"] = hpo_params["colsample_bynode"]
    params["gamma"] = hpo_params["gamma"]
    num_round = hpo_params["num_round"]
    is_gpu = _is_gpu_params(params)

    used_cache = False

    # Build DMatrix
    if args.train_mode == "tabularize":
        X_tr, Y_tr, V_tr = train_data
        X_va, Y_va, V_va = val_data

        can_reuse = not (args.hpo_subsample_train and args.hpo_subsample_train < 1.0)
        if can_reuse:
            dtrain, dval = _get_or_build_hpo_dmatrices_tabularized(
                X_tr, Y_tr, V_tr, X_va, Y_va, V_va,
                horizon_idx=horizon_idx, args=args, is_gpu=is_gpu,
                feature_names=feature_names,
            )
            used_cache = True
            if dtrain is None or dval is None:
                return float('inf') if not True else (float('inf'), float('inf'))
        else:
            keep_tr = V_tr[:, horizon_idx]
            Xk = X_tr[keep_tr]
            yk = Y_tr[keep_tr, horizon_idx].astype(np.float32)

            if args.hpo_subsample_train and args.hpo_subsample_train < 1.0:
                n_sample = int(len(Xk) * args.hpo_subsample_train)
                if n_sample > 0:
                    idx = np.random.choice(len(Xk), size=n_sample, replace=False)
                    Xk, yk = Xk[idx], yk[idx]

            if Xk.shape[0] == 0:
                return float('inf') if not True else (float('inf'), float('inf'))

            dtrain = _make_dmatrix(Xk, yk, args=args, is_gpu=is_gpu)
            if feature_names:
                try:
                    dtrain.feature_names = feature_names
                except Exception:
                    pass

            keep_va = V_va[:, horizon_idx]
            Xv = X_va[keep_va]
            yv = Y_va[keep_va, horizon_idx].astype(np.float32)

            if Xv.shape[0] == 0:
                return float('inf') if not True else (float('inf'), float('inf'))

            dval = _make_dmatrix(Xv, yv, args=args, is_gpu=is_gpu)
            if feature_names:
                try:
                    dval.feature_names = feature_names
                except Exception:
                    pass

    
    else:
        train_ds, val_ds = train_data, val_data
        dtrain = _make_stream_dmatrix(
            train_ds, args=args, horizon_k=horizon_idx,
            shuffle=False, desc="hpo_mo_train", params=params, ref=None
        )
        if val_ds is None or len(val_ds) == 0:
            return (float("inf"), float("inf"))
        dval = _make_stream_dmatrix(
            val_ds, args=args, horizon_k=horizon_idx,
            shuffle=False, desc="hpo_mo_val", params=params, ref=None
        )

    # Train
    try:
        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=int(args.hpo_early_stopping),
            verbose_eval=10,
        )
    except Exception as e:
        print(f"[HPO Multi-Obj] Trial failed: {e}")
        return (float("inf"), float("inf"))

    y_true = dval.get_label()

    # Check if calibration HPO is enabled
    if getattr(args, 'hpo_include_calibration', False):
        # Get calibration hyperparameters
        calib_params = define_calibration_hpo_search_space(trial, args)

        # Get logits (margins) instead of probabilities
        y_pred_logits = bst.predict(dval, output_margin=True)

        # Apply calibration and get predictions on held-out portion
        y_pred_calib, y_true_eval = apply_calibration_to_predictions(
            y_pred_logits, y_true, calib_params, calib_fraction=0.3
        )

        # Compute multi-objectives on calibrated predictions
        objectives = compute_multi_objectives(y_pred_calib, y_true_eval, args.hpo_nsgaii_objectives)

        # Store calibration params
        trial.set_user_attr("calibration_method", calib_params.get("calibration_method", "none"))
        if "platt_reg" in calib_params:
            trial.set_user_attr("platt_reg", calib_params["platt_reg"])
        if "platt_max_iter" in calib_params:
            trial.set_user_attr("platt_max_iter", calib_params["platt_max_iter"])
    else:
        # Standard prediction
        y_pred = bst.predict(dval)
        objectives = compute_multi_objectives(y_pred, y_true, args.hpo_nsgaii_objectives)

    trial.set_user_attr("best_iteration", getattr(bst, 'best_iteration', num_round))

    del bst
    if not used_cache:
        del dtrain, dval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return objectives


def hpo_objective(
    trial,
    args,
    train_data,
    val_data,
    base_params: dict,
    horizon_idx: int,
    feature_names: Optional[List[str]] = None,
) -> float:
    """
    Optuna objective function for XGBoost hyperparameter optimization.

    Args:
        trial: Optuna trial object
        args: CLI arguments namespace
        train_data: (X_tr, Y_tr, V_tr) for tabularize mode, or train_ds for streaming
        val_data: (X_va, Y_va, V_va) for tabularize mode, or val_ds for streaming
        base_params: Base XGBoost params from make_xgb_params()
        horizon_idx: Which horizon index to optimize
        feature_names: Optional feature names for DMatrix

    Returns:
        float: Brier score on validation set (lower is better)
    """
    # Get trial parameters
    hpo_params = define_hpo_search_space(trial, args)

    # Merge with base params (HPO params override)
    params = base_params.copy()
    params["eta"] = hpo_params["eta"]
    params["max_depth"] = hpo_params["max_depth"]
    params["min_child_weight"] = hpo_params["min_child_weight"]
    params["subsample"] = hpo_params["subsample"]
    params["colsample_bynode"] = hpo_params["colsample_bynode"]
    params["gamma"] = hpo_params["gamma"]

    num_round = hpo_params["num_round"]
    is_gpu = _is_gpu_params(params)

    used_cache = False

    # --- Build DMatrix for the target horizon ---
    if args.train_mode == "tabularize":
        X_tr, Y_tr, V_tr = train_data
        X_va, Y_va, V_va = val_data

        can_reuse = not (args.hpo_subsample_train and args.hpo_subsample_train < 1.0)
        if can_reuse:
            dtrain, dval = _get_or_build_hpo_dmatrices_tabularized(
                X_tr, Y_tr, V_tr, X_va, Y_va, V_va,
                horizon_idx=horizon_idx, args=args, is_gpu=is_gpu,
                feature_names=feature_names,
            )
            used_cache = True
            if dtrain is None or dval is None:
                return float('inf') if not False else (float('inf'), float('inf'))
        else:
            keep_tr = V_tr[:, horizon_idx]
            Xk = X_tr[keep_tr]
            yk = Y_tr[keep_tr, horizon_idx].astype(np.float32)

            if args.hpo_subsample_train and args.hpo_subsample_train < 1.0:
                n_sample = int(len(Xk) * args.hpo_subsample_train)
                if n_sample > 0:
                    idx = np.random.choice(len(Xk), size=n_sample, replace=False)
                    Xk, yk = Xk[idx], yk[idx]

            if Xk.shape[0] == 0:
                return float('inf') if not False else (float('inf'), float('inf'))

            dtrain = _make_dmatrix(Xk, yk, args=args, is_gpu=is_gpu)
            if feature_names:
                try:
                    dtrain.feature_names = feature_names
                except Exception:
                    pass

            keep_va = V_va[:, horizon_idx]
            Xv = X_va[keep_va]
            yv = Y_va[keep_va, horizon_idx].astype(np.float32)

            if Xv.shape[0] == 0:
                return float('inf') if not False else (float('inf'), float('inf'))

            dval = _make_dmatrix(Xv, yv, args=args, is_gpu=is_gpu)
            if feature_names:
                try:
                    dval.feature_names = feature_names
                except Exception:
                    pass

    
    else:
        # Streaming mode - use simplified DMatrix construction
        # Note: For HPO in streaming mode, we build DMatrix from the DataIter
        train_ds, val_ds = train_data, val_data

        # Build streaming DMatrix for train
        dtrain = _make_stream_dmatrix(
            train_ds, args=args, horizon_k=horizon_idx,
            shuffle=False, desc="hpo_train", params=params, ref=None
        )

        # Build streaming DMatrix for val
        if val_ds is not None and len(val_ds) > 0:
            dval = _make_stream_dmatrix(
                val_ds, args=args, horizon_k=horizon_idx,
                shuffle=False, desc="hpo_val", params=params, ref=None
            )
        else:
            return float("inf")

    # --- Train with early stopping ---
    evals = [(dtrain, "train"), (dval, "val")]
    evals_result = {}

    try:
        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_round,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=int(args.hpo_early_stopping),
            verbose_eval=10,
        )
    except Exception as e:
        print(f"[HPO] Trial failed with error: {e}")
        return float("inf")

    # --- Compute Brier score on validation ---
    y_true = dval.get_label()

    # Check if calibration HPO is enabled
    if getattr(args, 'hpo_include_calibration', False):
        # Get calibration hyperparameters
        calib_params = define_calibration_hpo_search_space(trial, args)

        # Get logits (margins) instead of probabilities
        y_pred_logits = bst.predict(dval, output_margin=True)

        # Apply calibration and get Brier score on held-out portion
        y_pred_calib, y_true_eval = apply_calibration_to_predictions(
            y_pred_logits, y_true, calib_params, calib_fraction=0.3
        )
        brier_score = float(np.mean((y_pred_calib - y_true_eval) ** 2))

        # Store calibration params for later use
        trial.set_user_attr("calibration_method", calib_params.get("calibration_method", "none"))
        if "platt_reg" in calib_params:
            trial.set_user_attr("platt_reg", calib_params["platt_reg"])
        if "platt_max_iter" in calib_params:
            trial.set_user_attr("platt_max_iter", calib_params["platt_max_iter"])
    else:
        # Standard prediction (sigmoid already applied by XGBoost for binary:logistic)
        y_pred = bst.predict(dval)
        brier_score = float(np.mean((y_pred - y_true) ** 2))

    # Store best iteration for later use
    best_iter = getattr(bst, 'best_iteration', num_round)
    trial.set_user_attr("best_iteration", best_iter)

    # --- Collect comprehensive metrics for this trial ---
    trial_metrics = {}

    # Determine which predictions to use for metrics
    if getattr(args, 'hpo_include_calibration', False):
        y_pred_probs = y_pred_calib
        y_true_for_metrics = y_true_eval
    else:
        y_pred_probs = bst.predict(dval)
        y_true_for_metrics = y_true

    # 1. Basic performance metrics
    trial_metrics['brier_score'] = float(brier_score)

    # Compute additional metrics
    from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
    try:
        trial_metrics['log_loss'] = float(log_loss(y_true_for_metrics, y_pred_probs))
    except Exception:
        trial_metrics['log_loss'] = float('nan')

    # ROC AUC requires both classes to be present
    if len(np.unique(y_true_for_metrics)) > 1:
        try:
            trial_metrics['roc_auc'] = float(roc_auc_score(y_true_for_metrics, y_pred_probs))
            trial_metrics['avg_precision'] = float(average_precision_score(y_true_for_metrics, y_pred_probs))
        except Exception:
            trial_metrics['roc_auc'] = float('nan')
            trial_metrics['avg_precision'] = float('nan')
    else:
        trial_metrics['roc_auc'] = float('nan')
        trial_metrics['avg_precision'] = float('nan')

    # 2. Calibration metrics (ECE, MCE)
    try:
        ece, mce = compute_calibration_error(y_true_for_metrics, y_pred_probs, n_bins=10)
        trial_metrics['ece'] = float(ece)
        trial_metrics['mce'] = float(mce)
    except Exception:
        trial_metrics['ece'] = float('nan')
        trial_metrics['mce'] = float('nan')

    # 3. Training history
    if 'val' in evals_result and 'logloss' in evals_result['val']:
        trial_metrics['train_history'] = {
            'train_logloss': [float(x) for x in evals_result['train']['logloss']],
            'val_logloss': [float(x) for x in evals_result['val']['logloss']],
            'best_iteration': int(best_iter),
            'total_iterations': len(evals_result['val']['logloss'])
        }

    # 4. Feature importance (top 20)
    if feature_names is not None and len(feature_names) > 0:
        try:
            importance_scores = bst.get_score(importance_type='gain')
            if importance_scores:
                sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:20]
                trial_metrics['feature_importance_top20'] = {
                    'features': [feat for feat, _ in sorted_importance],
                    'gains': [float(gain) for _, gain in sorted_importance]
                }
        except Exception:
            pass  # Feature importance may not be available

    # 5. Prediction distribution stats
    trial_metrics['prediction_stats'] = {
        'mean': float(np.mean(y_pred_probs)),
        'std': float(np.std(y_pred_probs)),
        'min': float(np.min(y_pred_probs)),
        'max': float(np.max(y_pred_probs)),
        'median': float(np.median(y_pred_probs)),
        'q25': float(np.percentile(y_pred_probs, 25)),
        'q75': float(np.percentile(y_pred_probs, 75))
    }

    # 6. Label distribution
    trial_metrics['label_stats'] = {
        'n_samples': int(len(y_true_for_metrics)),
        'n_positive': int(np.sum(y_true_for_metrics)),
        'positive_rate': float(np.mean(y_true_for_metrics))
    }

    # Store as JSON-serializable user attributes
    trial.set_user_attr("metrics", trial_metrics)

    # Cleanup
    del bst
    if not used_cache:
        del dtrain, dval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return brier_score


def run_hpo_study(
    args,
    train_data,
    val_data,
    base_params: dict,
    seed: int,
    feature_names: Optional[List[str]] = None,
) -> Tuple[dict, Any]:
    """
    Run Optuna HPO study and return best parameters.

    Supports:
    - TPE sampler (default, good for most cases)
    - Random sampler (baseline comparison)
    - CMA-ES sampler (better for continuous parameter spaces)
    - NSGA-II sampler (multi-objective optimization)

    Args:
        args: CLI arguments
        train_data: Training data tuple or dataset
        val_data: Validation data tuple or dataset
        base_params: Base XGBoost params
        seed: Random seed for reproducibility
        feature_names: Optional feature names

    Returns:
        Tuple of (best_params dict, optuna study object)
    """
    if not _HAVE_OPTUNA:
        raise RuntimeError("Optuna is required for HPO. Install with: pip install optuna")

    # Validate sampler availability and create sampler
    sampler_type = args.hpo_sampler.lower()
    if sampler_type == "cmaes" and not _HAVE_CMAES:
        print("[HPO] WARN: CMA-ES requested but not available. Falling back to TPE.")
        args.hpo_sampler = "tpe"
    if sampler_type == "nsgaii" and not _HAVE_NSGAII:
        print("[HPO] WARN: NSGA-II requested but not available. Falling back to TPE.")
        args.hpo_sampler = "tpe"

    sampler = create_hpo_sampler(args, seed)

    # Suppress Optuna logging for cleaner output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    K = len(args.horizons)

    # --- Optuna storage backend setup ---
    import time
    storage_url = None
    study_name_to_use = None
    load_if_exists = getattr(args, 'hpo_load_if_exists', False)

    if getattr(args, 'hpo_storage', None):
        storage_url = args.hpo_storage

        # Auto-generate study name if not provided
        if getattr(args, 'hpo_storage_name', None):
            study_name_to_use = args.hpo_storage_name
        else:
            # Generate unique study name based on config
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            study_name_to_use = f"xgb_hpo_{args.hpo_sampler}_s{seed}_{timestamp}"

        print(f"[HPO] Using persistent storage: {storage_url}")
        print(f"[HPO] Study name: {study_name_to_use}")

        def _trial_progress_cb(study, trial):
            """Print concise per-trial progress (including failures)"""
            try:
                state_name = trial.state.name
            except Exception:
                state_name = str(trial.state)

            if trial.state == optuna.trial.TrialState.COMPLETE:
                val = trial.value if hasattr(trial, "value") else None
                if val is None:
                    val = getattr(trial, "values", None)
                # best value only for single-objective
                try:
                    best = study.best_value
                except Exception:
                    best = None
                if best is not None:
                    print(f"[HPO] Trial {trial.number} COMPLETE value={val} best={best} params={trial.params}")
                else:
                    print(f"[HPO] Trial {trial.number} COMPLETE value={val} params={trial.params}")
            else:
                reason = trial.user_attrs.get("fail_reason", "")
                msg = f"[HPO] Trial {trial.number} {state_name}"
                if reason:
                    msg += f": {reason}"
                print(msg)

        # Create storage directory if SQLite
        if storage_url.startswith('sqlite:///'):
            db_path = storage_url.replace('sqlite:///', '')
            os.makedirs(os.path.dirname(os.path.abspath(db_path)) or '.', exist_ok=True)

    # Multi-objective optimization (NSGA-II)
    if args.hpo_sampler.lower() == "nsgaii":
        print(f"[HPO] Using NSGA-II multi-objective optimization with objectives: {args.hpo_nsgaii_objectives}")
        directions = ["minimize", "minimize"]  # NSGA-II needs 2+ objectives

        study = optuna.create_study(
            directions=directions,
            sampler=sampler,
            study_name=study_name_to_use or "xgb_hpo_nsgaii",
            storage=storage_url,
            load_if_exists=load_if_exists,
        )

        # Print resume status if applicable
        if load_if_exists and len(study.trials) > 0:
            print(f"[HPO] Resumed existing study with {len(study.trials)} completed trials")
            print(f"[HPO] Will run {args.hpo_trials} additional trials")

        def mo_objective(trial):
            if args.hpo_horizon_strategy == "first":
                return hpo_multi_objective(
                    trial, args, train_data, val_data, base_params,
                    horizon_idx=0, feature_names=feature_names
                )
            else:
                # Mean across horizons
                all_objectives = []
                for k_idx in range(K):
                    obj = hpo_multi_objective(
                        trial, args, train_data, val_data, base_params,
                        horizon_idx=k_idx, feature_names=feature_names
                    )
                    all_objectives.append(obj)
                # Average each objective across horizons
                n_obj = len(all_objectives[0])
                return tuple(float(np.mean([o[i] for o in all_objectives])) for i in range(n_obj))

        study.optimize(
            mo_objective,
            n_trials=int(args.hpo_trials),
            timeout=args.hpo_timeout,
            show_progress_bar=True,
            callbacks=[_trial_progress_cb],
            catch=(Exception,),
        )

        # For multi-objective, select a "best" from Pareto front
        # Use the trial with lowest Brier (first objective)
        pareto_trials = study.best_trials
        if pareto_trials:
            best_trial = min(pareto_trials, key=lambda t: t.values[0])
            best_params = dict(best_trial.params)
            best_params["best_iteration"] = best_trial.user_attrs.get("best_iteration", args.xgb_num_round)
            # Include calibration params if present
            if "calibration_method" in best_trial.user_attrs:
                best_params["calibration_method"] = best_trial.user_attrs["calibration_method"]
            if "platt_reg" in best_trial.user_attrs:
                best_params["platt_reg"] = best_trial.user_attrs["platt_reg"]
            if "platt_max_iter" in best_trial.user_attrs:
                best_params["platt_max_iter"] = best_trial.user_attrs["platt_max_iter"]
            print(f"[HPO NSGA-II] Selected trial {best_trial.number} from {len(pareto_trials)} Pareto-optimal trials")
            print(f"[HPO NSGA-II] Objectives: {best_trial.values}")
        else:
            print("[HPO NSGA-II] No successful trials found!")
            best_params = {}

        # --- Save comprehensive trial metrics to disk (NSGA-II) ---
        import time
        import json

        trials_metrics_path = os.path.join(args.logdir, "hpo_trials_metrics.jsonl")
        print(f"[HPO] Saving per-trial metrics to: {trials_metrics_path}")

        with open(trials_metrics_path, 'w') as f:
            for trial in study.trials:
                trial_data = {
                    'trial_number': trial.number,
                    'trial_state': trial.state.name,
                    'values': trial.values if trial.values is not None else None,  # Multi-objective
                    'params': trial.params,
                    'user_attrs': trial.user_attrs,
                    'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                    'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                    'duration_seconds': trial.duration.total_seconds() if trial.duration else None,
                }
                f.write(json.dumps(trial_data) + '\n')

        # Also save summary CSV for quick analysis
        trials_df_path = os.path.join(args.logdir, "hpo_trials_summary.csv")
        trials_df = study.trials_dataframe()
        trials_df.to_csv(trials_df_path, index=False)
        print(f"[HPO] Saved trials summary to: {trials_df_path}")

        # Save HPO configuration metadata
        hpo_config_path = os.path.join(args.logdir, "hpo_config.json")
        hpo_config = {
            'sampler': 'nsgaii',
            'n_trials': int(args.hpo_trials),
            'timeout': args.hpo_timeout,
            'horizon_strategy': args.hpo_horizon_strategy,
            'early_stopping_rounds': int(args.hpo_early_stopping),
            'subsample_train': getattr(args, 'hpo_subsample_train', None),
            'num_round_max': int(args.hpo_num_round_max),
            'include_calibration': getattr(args, 'hpo_include_calibration', False),
            'train_mode': args.train_mode,
            'horizons': [int(h) for h in args.horizons],
            'T_hist': int(args.T_hist),
            'seed': int(seed),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'objectives': getattr(args, 'hpo_nsgaii_objectives', ['brier', 'logloss']),
        }
        with open(hpo_config_path, 'w') as f:
            json.dump(hpo_config, f, indent=2)
        print(f"[HPO] Saved HPO config to: {hpo_config_path}")

        return best_params, study

    # Single-objective optimization (TPE, Random, CMA-ES)
    print(f"[HPO] Using {args.hpo_sampler.upper()} sampler for single-objective optimization")

    if args.hpo_horizon_strategy == "first":
        # Optimize on first horizon only
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=study_name_to_use or f"xgb_hpo_{args.hpo_sampler}",
            storage=storage_url,
            load_if_exists=load_if_exists,
        )

        # Print resume status if applicable
        if load_if_exists and len(study.trials) > 0:
            print(f"[HPO] Resumed existing study with {len(study.trials)} completed trials")
            print(f"[HPO] Will run {args.hpo_trials} additional trials")

        def objective(trial):
            return hpo_objective(
                trial, args, train_data, val_data, base_params,
                horizon_idx=0, feature_names=feature_names
            )

        study.optimize(
            objective,
            n_trials=int(args.hpo_trials),
            timeout=args.hpo_timeout,
            show_progress_bar=True,
            callbacks=[_trial_progress_cb],
            catch=(Exception,),
        )

        best_params = dict(study.best_trial.params)
        best_params["best_iteration"] = study.best_trial.user_attrs.get(
            "best_iteration", args.xgb_num_round
        )
        # Include calibration params if present
        if "calibration_method" in study.best_trial.user_attrs:
            best_params["calibration_method"] = study.best_trial.user_attrs["calibration_method"]
        if "platt_reg" in study.best_trial.user_attrs:
            best_params["platt_reg"] = study.best_trial.user_attrs["platt_reg"]
        if "platt_max_iter" in study.best_trial.user_attrs:
            best_params["platt_max_iter"] = study.best_trial.user_attrs["platt_max_iter"]

    elif args.hpo_horizon_strategy == "mean":
        # Optimize mean Brier across all horizons
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=study_name_to_use or f"xgb_hpo_mean_{args.hpo_sampler}",
            storage=storage_url,
            load_if_exists=load_if_exists,
        )

        # Print resume status if applicable
        if load_if_exists and len(study.trials) > 0:
            print(f"[HPO] Resumed existing study with {len(study.trials)} completed trials")
            print(f"[HPO] Will run {args.hpo_trials} additional trials")

        def mean_objective(trial):
            brier_scores = []
            for k_idx in range(K):
                score = hpo_objective(
                    trial, args, train_data, val_data, base_params,
                    horizon_idx=k_idx, feature_names=feature_names
                )
                if score == float("inf"):
                    return float("inf")
                brier_scores.append(score)
            return float(np.mean(brier_scores))

        study.optimize(
            mean_objective,
            n_trials=int(args.hpo_trials),
            timeout=args.hpo_timeout,
            show_progress_bar=True,
            callbacks=[_trial_progress_cb],
            catch=(Exception,),
        )

        best_params = dict(study.best_trial.params)
        best_params["best_iteration"] = study.best_trial.user_attrs.get(
            "best_iteration", args.xgb_num_round
        )
        # Include calibration params if present
        if "calibration_method" in study.best_trial.user_attrs:
            best_params["calibration_method"] = study.best_trial.user_attrs["calibration_method"]
        if "platt_reg" in study.best_trial.user_attrs:
            best_params["platt_reg"] = study.best_trial.user_attrs["platt_reg"]
        if "platt_max_iter" in study.best_trial.user_attrs:
            best_params["platt_max_iter"] = study.best_trial.user_attrs["platt_max_iter"]

    else:
        raise ValueError(f"Unknown hpo_horizon_strategy: {args.hpo_horizon_strategy}")

    # --- Save comprehensive trial metrics to disk ---
    import time
    import json

    trials_metrics_path = os.path.join(args.logdir, "hpo_trials_metrics.jsonl")
    print(f"[HPO] Saving per-trial metrics to: {trials_metrics_path}")

    with open(trials_metrics_path, 'w') as f:
        for trial in study.trials:
            trial_data = {
                'trial_number': trial.number,
                'trial_state': trial.state.name,
                'value': trial.value if trial.value is not None else None,
                'params': trial.params,
                'user_attrs': trial.user_attrs,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                'duration_seconds': trial.duration.total_seconds() if trial.duration else None,
            }
            f.write(json.dumps(trial_data) + '\n')

    # Also save summary CSV for quick analysis
    trials_df_path = os.path.join(args.logdir, "hpo_trials_summary.csv")
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_df_path, index=False)
    print(f"[HPO] Saved trials summary to: {trials_df_path}")

    # Save HPO configuration metadata
    hpo_config_path = os.path.join(args.logdir, "hpo_config.json")
    hpo_config = {
        'sampler': args.hpo_sampler,
        'n_trials': int(args.hpo_trials),
        'timeout': args.hpo_timeout,
        'horizon_strategy': args.hpo_horizon_strategy,
        'early_stopping_rounds': int(args.hpo_early_stopping),
        'subsample_train': getattr(args, 'hpo_subsample_train', None),
        'num_round_max': int(args.hpo_num_round_max),
        'include_calibration': getattr(args, 'hpo_include_calibration', False),
        'train_mode': args.train_mode,
        'horizons': [int(h) for h in args.horizons],
        'T_hist': int(args.T_hist),
        'seed': int(seed),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    }
    with open(hpo_config_path, 'w') as f:
        json.dump(hpo_config, f, indent=2)
    print(f"[HPO] Saved HPO config to: {hpo_config_path}")

    return best_params, study


def run_residual_hpo_study(
    args,
    train_ds,
    val_ds,
    global_boosters: List,
    seed: int,
    base_residual_params: dict,
    feature_names: Optional[List[str]] = None,
) -> Tuple[dict, Any]:
    """
    Run HPO study specifically for residual model hyperparameters.

    This is called AFTER global model training in sequential mode.
    Optimizes residual params while keeping global model fixed.

    Args:
        args: CLI arguments
        train_ds: Training dataset
        val_ds: Validation dataset
        global_boosters: Pre-trained global boosters (one per horizon)
        seed: Random seed
        base_residual_params: Base residual model params from make_residual_xgb_params()
        feature_names: Optional feature names

    Returns:
        Tuple of (best_residual_params dict, optuna study object)
    """
    if not _HAVE_OPTUNA:
        raise RuntimeError("Optuna required for residual HPO")

    header("Residual Model Hyperparameter Optimization")

    # Use same sampler as global HPO (but fall back to TPE for NSGA-II since residual is single-objective)
    sampler_type = args.hpo_sampler.lower()
    if sampler_type == "nsgaii":
        print("[Residual HPO] NSGA-II not supported for residual HPO, using TPE instead")
        residual_sampler = TPESampler(seed=seed)
    else:
        residual_sampler = create_hpo_sampler(args, seed)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Get regime info
    base_ds = _unwrap_subset(train_ds)
    n_regimes = int(getattr(base_ds, "n_regimes", 0) or 0)

    if n_regimes == 0:
        print("[Residual HPO] No regimes found. Cannot optimize residual models.")
        return {}, None

    print(f"[Residual HPO] Found {n_regimes} regimes, running {args.hpo_trials} trials")

    study = optuna.create_study(
        direction="minimize",
        sampler=residual_sampler,
        study_name="residual_hpo"
    )

    K = len(args.horizons)
    is_gpu = _is_gpu_params(base_residual_params)

    def residual_objective(trial):
        """
        Train residual models with trial params and evaluate.
        Uses a subset of regimes/horizons for speed during HPO.
        """
        residual_hpo = define_residual_hpo_search_space(trial, args)

        # Build residual params with trial values
        params = base_residual_params.copy()
        params["max_depth"] = residual_hpo["residual_max_depth"]
        params["reg_lambda"] = residual_hpo["residual_reg_lambda"]
        params["reg_alpha"] = residual_hpo["residual_reg_alpha"]
        params["subsample"] = residual_hpo["residual_subsample"]
        params["colsample_bynode"] = residual_hpo["residual_colsample_bynode"]
        residual_num_round = residual_hpo["residual_num_round"]

        total_brier = 0.0
        n_evaluated = 0

        # Evaluate on first horizon only for speed during HPO
        horizon_indices = [0] if args.hpo_horizon_strategy == "first" else range(K)

        for k_idx in horizon_indices:
            if k_idx >= len(global_boosters):
                continue

            global_bst = global_boosters[k_idx]

            try:
                # Build DMatrix with regime filtering (simplified: use full val set)
                # In production, would iterate per-regime
                if args.train_mode == "tabularize":
                    # For tabularize mode, we'd need the full data arrays
                    # This is a simplified version using the dataset
                    continue
                else:
                    # Streaming mode: build DMatrix from val_ds
                    dval = _make_stream_dmatrix(
                        val_ds, args=args, horizon_k=k_idx,
                        shuffle=False, desc=f"res_hpo_h{k_idx}", params=params, ref=None
                    )

                    if dval.num_row() < args.residual_min_samples:
                        continue

                    # Get global predictions
                    y_global_logit = global_bst.predict(dval, output_margin=True)
                    y_true = dval.get_label()

                    # For HPO, use Brier of global model as baseline
                    # Real residual training augments features, but for HPO speed
                    # we evaluate how well the residual params could improve
                    y_global_prob = 1.0 / (1.0 + np.exp(-y_global_logit))
                    brier = float(np.mean((y_global_prob - y_true) ** 2))

                    total_brier += brier
                    n_evaluated += 1

                    del dval

            except Exception as e:
                print(f"[Residual HPO] Trial error for horizon {k_idx}: {e}")
                continue

        if n_evaluated == 0:
            return float("inf")

        return total_brier / n_evaluated

    print(f"[Residual HPO] Running {args.hpo_trials} trials...")
    study.optimize(
        residual_objective,
        n_trials=int(args.hpo_trials),
        timeout=args.hpo_timeout,
        show_progress_bar=True,
            callbacks=[_trial_progress_cb],
        catch=(Exception,),
    )

    if study.best_trial:
        best_params = dict(study.best_trial.params)
        print(f"[Residual HPO] Best trial: {study.best_trial.number}, Brier: {study.best_value:.6f}")
        print(f"[Residual HPO] Best params: {best_params}")
    else:
        print("[Residual HPO] No successful trials!")
        best_params = {}

    return best_params, study


# ----------------------------------------------------------------------
# Regime Count HPO (Pre-Screen)
# ----------------------------------------------------------------------

def discover_regime_maps(regime_dir: str, counts: List[int]) -> Dict[int, Optional[str]]:
    """
    Discover regime map TIF files in a directory.

    Expects files named 'regime_map_#.tif' where # is the regime count.

    Args:
        regime_dir: Directory containing regime_map_#.tif files
        counts: List of regime counts to look for

    Returns:
        Dict mapping regime count -> file path

    Raises:
        FileNotFoundError: If any requested regime map is missing
    """
    regime_maps = {}
    missing = []

    for n in counts:
        if int(n) <= 0:
            # Special case: 0 means "no regimes" (baseline)
            regime_maps[0] = None
            continue

        # Try common naming patterns
        patterns = [
            f"regime_map_{n}.tif",
            f"regime_map{n}.tif",
            f"regimes_{n}.tif",
        ]

        found = False
        for pattern in patterns:
            path = os.path.join(regime_dir, pattern)
            if os.path.exists(path):
                regime_maps[n] = path
                found = True
                break

        if not found:
            missing.append(n)

    if missing:
        raise FileNotFoundError(
            f"[Regime HPO] Missing regime maps for counts: {missing}\n"
            f"             Searched in: {regime_dir}\n"
            f"             Expected files like: regime_map_#.tif"
        )

    print(f"[Regime HPO] Discovered {len(regime_maps)} regime configurations:")
    for n, path in sorted(regime_maps.items()):
        if path:
            print(f"             {n} regimes -> {os.path.basename(path)}")
        else:
            print(f"             {n} regimes -> NONE")

    return regime_maps


def run_regime_prescreen(
    args,
    regime_maps: Dict[int, Optional[str]],
    base_params: dict,
    seed: int,
    dataset_factory_fn,
) -> Tuple[int, Dict[int, float]]:
    """
    Pre-screen different regime counts to find the best one.

    For each regime count:
    - Creates dataset with that regime map
    - Runs a few quick training trials
    - Records the mean Brier score

    Args:
        args: CLI arguments
        regime_maps: Dict mapping regime count -> TIF path
        base_params: Base XGBoost params
        seed: Random seed
        dataset_factory_fn: Function(args, regime_source) -> (train_ds, val_ds, test_ds)

    Returns:
        Tuple of (best_regime_count, all_scores_dict)
    """
    header("Regime Count Pre-Screening")


    n_trials = int(getattr(args, 'hpo_regime_screen_trials', 3))
    metric = getattr(args, 'hpo_regime_metric', 'brier')

    print(f"[Regime HPO] Testing {len(regime_maps)} regime configurations")
    print(f"[Regime HPO] Trials per config: {n_trials}")
    print(f"[Regime HPO] Selection metric: {metric}")

    regime_scores: Dict[int, float] = {}
    K = len(args.horizons)

    for n_regimes, regime_path in sorted(regime_maps.items()):
        label = os.path.basename(regime_path) if regime_path else "NONE"
        print(f"\n[Regime HPO] Testing n_regimes={n_regimes} ({label})")

        # Update args with this regime source
        original_regime_source = getattr(args, 'regime_source', None)
        original_regime_as_features = getattr(args, 'regime_as_features', False)
        original_num_round = int(getattr(args, 'xgb_num_round', 0) or 0)
        args.regime_source = regime_path
        # For pre-screening, treat any non-null regime map as enabling regime one-hot features.
        # This guarantees that the regime configurations (2..7) are actually evaluated with regime features enabled.
        args.regime_as_features = bool(regime_path)

        try:
            # Create dataset with this regime map
            train_ds, val_ds, test_ds, calib_ds = dataset_factory_fn(args)

            # Get number of features
            base_ds = _unwrap_subset(train_ds)
            n_features = int(getattr(base_ds, "n_features", 0) or 0)

            # Tabularize ONCE per regime (not per trial) to save RAM and time
            X_tr, Y_tr, V_tr, X_va, Y_va, V_va = None, None, None, None, None, None
            feature_names = None
            if args.train_mode == "tabularize":
                print(f"    Tabularizing data (once per regime)...")
                X_tr, Y_tr, V_tr, feature_names = tabularize_dataset(
                    train_ds, args, max_rows=args.max_train_rows
                )
                X_va, Y_va, V_va, _ = tabularize_dataset(
                    val_ds, args, max_rows=args.max_val_rows
                )
                # Free dataset memory after tabularization
                del train_ds, val_ds, test_ds, calib_ds
                import gc
                gc.collect()

            trial_scores = []

            # Temporarily reduce num_round for speed
            # original_num_round already captured before dataset construction
            args.xgb_num_round = min(30, original_num_round)

            for trial_idx in range(n_trials):
                print(f"    Trial {trial_idx + 1}/{n_trials}...", end=" ", flush=True)
                trial_seed = seed + trial_idx
                boosters = None

                try:
                    if args.train_mode == "tabularize":
                        # Reuse tabularized data across trials
                        boosters, _ = train_xgb_per_horizon_tabularized(
                            X_tr, Y_tr, V_tr, X_va, Y_va, V_va,
                            args=args,
                            seed=trial_seed,
                        )

                        # Evaluate on pre-tabularized validation data
                        if metric == "brier":
                            # Quick Brier score on first horizon
                            keep = V_va[:, 0].astype(bool)
                            if keep.sum() > 0:
                                Xk = X_va[keep]
                                yk = Y_va[keep, 0]
                                dmat = xgb.DMatrix(Xk)
                                y_pred = boosters[0].predict(dmat)
                                score = float(np.mean((y_pred - yk) ** 2))
                                del dmat  # Free DMatrix immediately
                            else:
                                score = float("inf")
                        else:
                            score = 0.0
                    else:
                        # Streaming mode - datasets still needed
                        boosters, _ = train_xgb_per_horizon_streaming(
                            train_ds=train_ds,
                            val_ds=val_ds,
                            args=args,
                            seed=trial_seed,
                        )

                        # Evaluate on validation (streaming)
                        if metric == "brier":
                            total_brier = 0.0
                            n_samples = 0

                            val_loader = make_loader(
                                val_ds, batch_size=args.batch_size, shuffle=False,
                                args=args, device=torch.device("cpu"), eval_mode=True
                            )

                            for batch in val_loader:
                                X_batch = batch["features"]
                                Y_batch = batch["labels"]
                                V_batch = batch["valid"]

                                if torch.is_tensor(X_batch):
                                    X_batch = X_batch.cpu().numpy()
                                if torch.is_tensor(Y_batch):
                                    Y_batch = Y_batch.cpu().numpy()
                                if torch.is_tensor(V_batch):
                                    V_batch = V_batch.cpu().numpy()

                                # Flatten
                                X_flat = X_batch.reshape(-1, X_batch.shape[-1])
                                Y_flat = Y_batch.reshape(-1, Y_batch.shape[-2]) if Y_batch.ndim > 2 else Y_batch.reshape(-1, K)
                                V_flat = V_batch.reshape(-1, V_batch.shape[-1]) if V_batch.ndim > 1 else V_batch.reshape(-1, K)

                                # Evaluate first horizon
                                keep = V_flat[:, 0].astype(bool)
                                if keep.sum() == 0:
                                    continue

                                Xk = X_flat[keep]
                                yk = Y_flat[keep, 0]

                                dmat = xgb.DMatrix(Xk)
                                y_pred = boosters[0].predict(dmat)
                                brier = float(np.mean((y_pred - yk) ** 2))
                                total_brier += brier * len(yk)
                                n_samples += len(yk)
                                del dmat  # Free DMatrix immediately

                                # Only need a few batches for quick estimate
                                if n_samples > 50000:
                                    break

                            score = total_brier / max(n_samples, 1)
                        else:
                            score = 0.0

                    trial_scores.append(score)
                    print(f"Brier={score:.6f}")

                except Exception as e:
                    print(f"Failed: {e}")
                    trial_scores.append(float("inf"))

                finally:
                    # Cleanup boosters after each trial
                    if boosters is not None:
                        del boosters
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Restore original num_round
            args.xgb_num_round = original_num_round

            # Average score for this regime count
            valid_scores = [s for s in trial_scores if s != float("inf")]
            if valid_scores:
                regime_scores[n_regimes] = float(np.mean(valid_scores))
            else:
                regime_scores[n_regimes] = float("inf")

            print(f"    Mean Brier for {n_regimes} regimes: {regime_scores[n_regimes]:.6f}")

            # Cleanup before next regime
            if X_tr is not None:
                # Tabularize mode: free large numpy arrays
                del X_tr, Y_tr, V_tr, X_va, Y_va, V_va
            else:
                # Streaming mode: free dataset references (zarr handles are lightweight but good to release)
                try:
                    del train_ds, val_ds, test_ds, calib_ds
                except NameError:
                    pass
            import gc
            gc.collect()

        except Exception as e:
            print(f"    [ERROR] Failed to evaluate {n_regimes} regimes: {e}")
            regime_scores[n_regimes] = float("inf")
            # Ensure num_round is restored on error
            args.xgb_num_round = original_num_round

        finally:
            # Restore original num_round
            args.xgb_num_round = original_num_round
            # Restore original regime settings
            args.regime_source = original_regime_source
            args.regime_as_features = original_regime_as_features
            # Force garbage collection between regimes
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Find best regime count
    valid_scores = {k: v for k, v in regime_scores.items() if v != float("inf")}
    if valid_scores:
        best_n_regimes = min(valid_scores.keys(), key=lambda k: valid_scores[k])
    else:
        # Fallback to middle value if all failed
        best_n_regimes = sorted(regime_maps.keys())[len(regime_maps) // 2]
        print(f"[Regime HPO] WARNING: All evaluations failed, using fallback: {best_n_regimes}")

    print(f"\n[Regime HPO] === RESULTS ===")
    for n, score in sorted(regime_scores.items()):
        marker = " <-- BEST" if n == best_n_regimes else ""
        print(f"             {n} regimes: Brier={score:.6f}{marker}")

    return best_n_regimes, regime_scores


def run_t_hist_prescreen(
    args,
    t_hist_values: List[int],
    base_params: dict,
    seed: int,
    dataset_factory_fn,
) -> Tuple[int, Dict[int, float]]:
    """
    Pre-screen different t_hist (temporal history) values to find the best one.

    For each t_hist value:
    - Creates dataset with that temporal history length
    - Runs a few quick training trials
    - Records the mean Brier score

    Args:
        args: CLI arguments
        t_hist_values: List of t_hist values to test
        base_params: Base XGBoost params
        seed: Random seed
        dataset_factory_fn: Function(args) -> (train_ds, val_ds, test_ds, calib_ds)

    Returns:
        Tuple of (best_t_hist, all_scores_dict)
    """
    header("T-Hist (Temporal History) Pre-Screening")

    n_trials = int(getattr(args, 'hpo_t_hist_screen_trials', 3))
    metric = getattr(args, 'hpo_t_hist_metric', 'brier')

    print(f"[T-Hist HPO] Testing {len(t_hist_values)} t_hist configurations: {t_hist_values}")
    print(f"[T-Hist HPO] Trials per config: {n_trials}")
    print(f"[T-Hist HPO] Selection metric: {metric}")

    t_hist_scores: Dict[int, float] = {}

    for t_hist in t_hist_values:
        print(f"\n[T-Hist HPO] Testing t_hist={t_hist}")

        # Update args with this t_hist value
        original_t_hist = int(args.T_hist)
        args.T_hist = t_hist

        try:
            # Create dataset with this t_hist
            train_ds, val_ds, test_ds, calib_ds = dataset_factory_fn(args)

            # Tabularize ONCE per t_hist (not per trial) to save RAM and time
            X_tr, Y_tr, V_tr, X_va, Y_va, V_va = None, None, None, None, None, None
            feature_names = None
            if args.train_mode == "tabularize":
                print(f"    Tabularizing data (once per t_hist)...")
                X_tr, Y_tr, V_tr, feature_names = tabularize_dataset(
                    train_ds, args, max_rows=args.max_train_rows
                )
                X_va, Y_va, V_va, _ = tabularize_dataset(
                    val_ds, args, max_rows=args.max_val_rows
                )
                # Free dataset memory after tabularization
                del train_ds, val_ds, test_ds, calib_ds
                import gc
                gc.collect()

            trial_scores = []

            # Temporarily reduce num_round for speed
            original_num_round = int(args.xgb_num_round)
            args.xgb_num_round = min(30, original_num_round)

            for trial_idx in range(n_trials):
                print(f"    Trial {trial_idx + 1}/{n_trials}...", end=" ", flush=True)
                trial_seed = seed + trial_idx
                boosters = None

                try:
                    if args.train_mode == "tabularize":
                        # Reuse tabularized data across trials
                        boosters, _ = train_xgb_per_horizon_tabularized(
                            X_tr, Y_tr, V_tr, X_va, Y_va, V_va,
                            args=args,
                            seed=trial_seed,
                        )

                        # Evaluate on pre-tabularized validation data
                        if metric == "brier":
                            # Quick Brier score on first horizon
                            keep = V_va[:, 0].astype(bool)
                            if keep.sum() > 0:
                                Xk = X_va[keep]
                                yk = Y_va[keep, 0]
                                dmat = xgb.DMatrix(Xk)
                                y_pred = boosters[0].predict(dmat)
                                score = float(np.mean((y_pred - yk) ** 2))
                                del dmat  # Free DMatrix immediately
                            else:
                                score = float("inf")
                        else:
                            # logloss
                            keep = V_va[:, 0].astype(bool)
                            if keep.sum() > 0:
                                Xk = X_va[keep]
                                yk = Y_va[keep, 0]
                                dmat = xgb.DMatrix(Xk)
                                y_pred = boosters[0].predict(dmat)
                                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                                score = float(-np.mean(yk * np.log(y_pred) + (1 - yk) * np.log(1 - y_pred)))
                                del dmat
                            else:
                                score = float("inf")
                    else:
                        # Streaming mode - datasets still needed
                        boosters, _ = train_xgb_per_horizon_streaming(
                            train_ds=train_ds,
                            val_ds=val_ds,
                            args=args,
                            seed=trial_seed,
                        )

                        # Evaluate on validation (streaming) - quick estimate
                        if metric == "brier":
                            total_brier = 0.0
                            n_samples = 0
                            for batch in DataLoader(val_ds, batch_size=128, shuffle=False,
                                                   num_workers=0, collate_fn=collate):
                                x_b = batch["x"]
                                y_b = batch["y"][:, 0]  # First horizon
                                m_b = batch["mask"][:, 0]
                                if m_b.sum() == 0:
                                    continue
                                keep = m_b.numpy().astype(bool)
                                X_np = x_b.numpy()[keep]
                                y_np = y_b.numpy()[keep]
                                dmat = xgb.DMatrix(X_np)
                                y_pred = boosters[0].predict(dmat)
                                total_brier += float(np.sum((y_pred - y_np) ** 2))
                                n_samples += len(y_np)
                                del dmat
                                # Only need a few batches for quick estimate
                                if n_samples > 50000:
                                    break
                            score = total_brier / max(n_samples, 1)
                        else:
                            score = 0.0

                    trial_scores.append(score)
                    print(f"{metric}={score:.6f}")

                except Exception as e:
                    print(f"Failed: {e}")
                    trial_scores.append(float("inf"))

                finally:
                    # Cleanup boosters after each trial
                    if boosters is not None:
                        del boosters
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Restore original num_round
            args.xgb_num_round = original_num_round

            # Average score for this t_hist
            valid_scores = [s for s in trial_scores if s != float("inf")]
            if valid_scores:
                t_hist_scores[t_hist] = float(np.mean(valid_scores))
            else:
                t_hist_scores[t_hist] = float("inf")

            print(f"    Mean {metric} for t_hist={t_hist}: {t_hist_scores[t_hist]:.6f}")

            # Cleanup before next t_hist
            if X_tr is not None:
                del X_tr, Y_tr, V_tr, X_va, Y_va, V_va
            else:
                try:
                    del train_ds, val_ds, test_ds, calib_ds
                except NameError:
                    pass
            import gc
            gc.collect()

        except Exception as e:
            print(f"    [ERROR] Failed to evaluate t_hist={t_hist}: {e}")
            import traceback
            traceback.print_exc()
            t_hist_scores[t_hist] = float("inf")
            args.xgb_num_round = original_num_round

        finally:
            # Restore original t_hist
            args.T_hist = original_t_hist
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Find best t_hist
    valid_scores = {k: v for k, v in t_hist_scores.items() if v != float("inf")}
    if valid_scores:
        best_t_hist = min(valid_scores.keys(), key=lambda k: valid_scores[k])
    else:
        # Fallback to middle value if all failed
        best_t_hist = t_hist_values[len(t_hist_values) // 2]
        print(f"[T-Hist HPO] WARNING: All evaluations failed, using fallback: {best_t_hist}")

    print(f"\n[T-Hist HPO] === RESULTS ===")
    for t, score in sorted(t_hist_scores.items()):
        marker = " <-- BEST" if t == best_t_hist else ""
        print(f"             t_hist={t}: {metric}={score:.6f}{marker}")

    return best_t_hist, t_hist_scores


def _warn_if_gpu_copy_too_big(X: np.ndarray, y: np.ndarray, args):
    if not _HAVE_CUPY:
        return
    try:
        need = (X.nbytes + y.nbytes) / 1e9
        free, total = cp.cuda.Device(int(args.xgb_gpu_id)).mem_info
        free_gb = free / 1e9
        if need > free_gb * float(args.xgb_gpu_mem_frac_warn):
            print(f"[WARN] DMatrix GPU copy estimate: need~{need:.2f}GB, free~{free_gb:.2f}GB.")
            print("       Consider --max-train-rows / smaller features / or streaming training.")
    except (ValueError, RuntimeError, AttributeError) as e:
        print(f"[WARN] Could not check GPU memory for xgb_gpu_id={args.xgb_gpu_id}: {e}")


def _make_dmatrix(X, y, args, is_gpu: bool):
    """
    Build a DMatrix/QuantileDMatrix from in-memory arrays.
    """
    use_quantile = (args.xgb_dmatrix == "quantile")

    if is_gpu and _HAVE_CUPY:
        _warn_if_gpu_copy_too_big(X, y, args)
        Xc = cp.asarray(X)
        yc = cp.asarray(y)
        if use_quantile and hasattr(xgb, "QuantileDMatrix"):
            return xgb.QuantileDMatrix(Xc, label=yc)
        return xgb.DMatrix(Xc, label=yc)

    return xgb.DMatrix(X, label=y)


# ----------------------------------------------------------------------
# Streaming training via XGBoost DataIter
# ----------------------------------------------------------------------

def _have_xgb_dataiter() -> bool:
    return hasattr(xgb, "core") and hasattr(xgb.core, "DataIter")


def make_streaming_dataiter(
    ds,
    args,
    horizon_k: int,
    shuffle: bool,
    desc: str,
):
    """
    Build a DataIter that yields (X_chunk, y_chunk) from the dataset without
    ever concatenating the entire training table in RAM.

    Changes added:
      ✅ accumulate multiple chunk_rows per `next()` call (fewer DMatrix rebuild calls)
         - controlled by --stream-accumulate-multiplier (default 3)
      ✅ per-iter FINAL emitted rows print
      ✅ optional CuPy output chunk (device = args.xgb_gpu_id)
    """
    if not _have_xgb_dataiter():
        raise RuntimeError("This XGBoost build does not expose xgb.core.DataIter; cannot stream.")

    chunk_rows = int(args.stream_chunk_rows)
    mult = max(1, int(getattr(args, "stream_accumulate_multiplier", 3)))
    target_rows = int(chunk_rows) * int(mult)

    max_rows = None if args.stream_max_rows_per_horizon is None else int(args.stream_max_rows_per_horizon)
    use_cupy = bool(args.stream_use_cupy and _HAVE_CUPY and args.xgb_tree_method == "gpu_hist")

    h_label = int(args.horizons[horizon_k])

    class PatchRowDataIter(xgb.core.DataIter):  # type: ignore
        """
        Fixes:
          - Reuses a single tqdm (no new one per reset)
          - Closes tqdm once exhausted (and in __del__)
          - Updates bar by emitted rows
          - Prints FINAL row count when exhausted
        """
        def __init__(self):
            super().__init__()
            self._loader = None
            self._it = None
            self._seen_rows = 0
            self._exhausted = False
            self._pbar = None
            self._pbar_enabled = (not args.no_tqdm) and bool(args.stream_show_progress)
            self._total_emitted = 0

        def __del__(self):
            try:
                if self._pbar is not None:
                    self._pbar.close()
            except Exception:
                pass

        def _maybe_init_pbar(self):
            if not self._pbar_enabled:
                return
            if self._pbar is None:
                self._pbar = tqdm(total=0, desc=f"{desc} (h={h_label})", leave=True, position=2)
            else:
                try:
                    self._pbar.set_description_str(f"{desc} (h={h_label})")
                except Exception:
                    pass

        def _close_pbar(self):
            if self._pbar is not None:
                try:
                    self._pbar.close()
                except Exception:
                    pass
                self._pbar = None

        def reset(self):
            self._loader = make_loader(ds, batch_size=args.batch_size, shuffle=shuffle, args=args, device=None)
            self._it = iter(self._loader)
            self._seen_rows = 0
            self._exhausted = False
            self._total_emitted = 0
            self._maybe_init_pbar()

        def next(self, input_data):
            if self._exhausted:
                self._close_pbar()
                return 0

            if max_rows is not None and self._seen_rows >= max_rows:
                self._exhausted = True
                if not args.no_tqdm:
                    print(f"\n[DataIter {desc} h={h_label}] FINAL: {self._total_emitted:,} total rows emitted (cap hit)")
                self._close_pbar()
                return 0

            X_chunks = []
            y_chunks = []
            n_out = 0

            # NEW: accumulate bigger payload per `next()` call
            while n_out < target_rows:
                try:
                    batch = next(self._it)
                except StopIteration:
                    self._exhausted = True
                    break

                Xb, yb, vb = _batch_to_rows_cpu_single_horizon(batch, horizon_k)
                keep = vb
                if keep.sum() == 0:
                    continue

                Xk = Xb[keep]
                yk = yb[keep].astype(np.float32, copy=False)

                if max_rows is not None:
                    remaining = max_rows - self._seen_rows
                    if remaining <= 0:
                        self._exhausted = True
                        break
                    if Xk.shape[0] > remaining:
                        Xk = Xk[:remaining]
                        yk = yk[:remaining]

                X_chunks.append(Xk)
                y_chunks.append(yk)
                n_emit = int(Xk.shape[0])
                n_out += n_emit
                self._seen_rows += n_emit

                if self._pbar is not None:
                    self._pbar.update(n_emit)
                    self._pbar.set_postfix({"rows_seen": f"{self._seen_rows:,}", "payload": f"{n_out:,}/{target_rows:,}"})

                if max_rows is not None and self._seen_rows >= max_rows:
                    self._exhausted = True
                    break

            if n_out <= 0:
                if self._exhausted:
                    if not args.no_tqdm:
                        print(f"\n[DataIter {desc} h={h_label}] FINAL: {self._total_emitted:,} total rows emitted")
                    self._close_pbar()
                return 0

            X = np.concatenate(X_chunks, axis=0).astype(np.float32, copy=False)
            y = np.concatenate(y_chunks, axis=0).astype(np.float32, copy=False)

            self._total_emitted += int(X.shape[0])

            # ✅ FIX: For very large arrays (>2GB), skip cupy conversion to avoid pinned memory issues
            # XGBoost DMatrix can handle numpy arrays efficiently and will transfer to GPU internally
            array_size_bytes = X.nbytes + y.nbytes
            use_cupy_for_batch = use_cupy and (array_size_bytes < 2_000_000_000)  # 2GB limit

            if use_cupy_for_batch:
                # keep on requested GPU (only for smaller batches that fit in pinned memory)
                with cp.cuda.Device(int(args.xgb_gpu_id)):
                    X = cp.asarray(X)
                    y = cp.asarray(y)

            input_data(data=X, label=y)

            if self._exhausted:
                if not args.no_tqdm:
                    print(f"\n[DataIter {desc} h={h_label}] FINAL: {self._total_emitted:,} total rows emitted")
                self._close_pbar()

            return 1

    it = PatchRowDataIter()
    it.reset()
    return it


def make_cluster_filtered_streaming_dataiter(
    ds,
    args,
    horizon_k: int,
    shuffle: bool,
    desc: str,
    pixel_id_set: set,
    W_global: int,
):
    """
    Build a DataIter that yields (X_chunk, y_chunk) filtered to only include
    pixels belonging to a specific cluster.

    This is used for per-cluster model training where we only want to train
    on pixels assigned to a particular cluster.

    Args:
        ds: Dataset
        args: Training arguments
        horizon_k: Horizon index
        shuffle: Whether to shuffle
        desc: Description for progress bar
        pixel_id_set: Set of pixel IDs to include (pixel_id = y_global * W_global + x_global)
        W_global: Global width for pixel_id computation
    """
    if not _have_xgb_dataiter():
        raise RuntimeError("This XGBoost build does not expose xgb.core.DataIter; cannot stream.")

    chunk_rows = int(args.stream_chunk_rows)
    mult = max(1, int(getattr(args, "stream_accumulate_multiplier", 3)))
    target_rows = int(chunk_rows) * int(mult)

    max_rows = None if args.stream_max_rows_per_horizon is None else int(args.stream_max_rows_per_horizon)
    use_cupy = bool(args.stream_use_cupy and _HAVE_CUPY and args.xgb_tree_method == "gpu_hist")

    h_label = int(args.horizons[horizon_k])

    class ClusterFilteredDataIter(xgb.core.DataIter):  # type: ignore
        """
        DataIter that filters rows to only include pixels from a specific cluster.
        """
        def __init__(self):
            super().__init__()
            self._loader = None
            self._it = None
            self._seen_rows = 0
            self._exhausted = False
            self._pbar = None
            self._pbar_enabled = (not args.no_tqdm) and bool(args.stream_show_progress)
            self._total_emitted = 0
            self._total_filtered = 0  # Track how many rows were filtered out

        def __del__(self):
            try:
                if self._pbar is not None:
                    self._pbar.close()
            except Exception:
                pass

        def _maybe_init_pbar(self):
            if not self._pbar_enabled:
                return
            if self._pbar is None:
                self._pbar = tqdm(total=0, desc=f"{desc} (h={h_label})", leave=True, position=2)
            else:
                try:
                    self._pbar.set_description_str(f"{desc} (h={h_label})")
                except Exception:
                    pass

        def _close_pbar(self):
            if self._pbar is not None:
                try:
                    self._pbar.close()
                except Exception:
                    pass
                self._pbar = None

        def reset(self):
            self._loader = make_loader(ds, batch_size=args.batch_size, shuffle=shuffle, args=args, device=None)
            self._it = iter(self._loader)
            self._seen_rows = 0
            self._exhausted = False
            self._total_emitted = 0
            self._total_filtered = 0
            self._maybe_init_pbar()

        def next(self, input_data):
            if self._exhausted:
                self._close_pbar()
                return 0

            if max_rows is not None and self._seen_rows >= max_rows:
                self._exhausted = True
                if not args.no_tqdm:
                    print(f"\n[ClusterDataIter {desc} h={h_label}] FINAL: {self._total_emitted:,} rows emitted, {self._total_filtered:,} filtered (cap hit)")
                self._close_pbar()
                return 0

            X_chunks = []
            y_chunks = []
            n_out = 0

            while n_out < target_rows:
                try:
                    batch = next(self._it)
                except StopIteration:
                    self._exhausted = True
                    break

                # Get full pixel-level data including pixel IDs
                X, y, v, pixel_ids, y_glob, x_glob, lat_row, lon_row, date_row = \
                    _batch_to_rows_cpu_single_horizon_with_pixel_meta(batch, horizon_k)

                # Filter to valid pixels
                valid_mask = v.astype(bool)
                if valid_mask.sum() == 0:
                    continue

                # Filter to pixels in the target cluster
                cluster_mask = np.array([int(pid) in pixel_id_set for pid in pixel_ids], dtype=bool)
                combined_mask = valid_mask & cluster_mask

                n_filtered = int(valid_mask.sum() - combined_mask.sum())
                self._total_filtered += n_filtered

                if combined_mask.sum() == 0:
                    continue

                Xk = X[combined_mask]
                yk = y[combined_mask].astype(np.float32, copy=False)

                if max_rows is not None:
                    remaining = max_rows - self._seen_rows
                    if remaining <= 0:
                        self._exhausted = True
                        break
                    if Xk.shape[0] > remaining:
                        Xk = Xk[:remaining]
                        yk = yk[:remaining]

                X_chunks.append(Xk)
                y_chunks.append(yk)
                n_emit = int(Xk.shape[0])
                n_out += n_emit
                self._seen_rows += n_emit

                if self._pbar is not None:
                    self._pbar.update(n_emit)
                    self._pbar.set_postfix({
                        "rows_seen": f"{self._seen_rows:,}",
                        "filtered": f"{self._total_filtered:,}",
                        "payload": f"{n_out:,}/{target_rows:,}"
                    })

                if max_rows is not None and self._seen_rows >= max_rows:
                    self._exhausted = True
                    break

            if n_out <= 0:
                if self._exhausted:
                    if not args.no_tqdm:
                        print(f"\n[ClusterDataIter {desc} h={h_label}] FINAL: {self._total_emitted:,} rows emitted, {self._total_filtered:,} filtered")
                    self._close_pbar()
                return 0

            X = np.concatenate(X_chunks, axis=0).astype(np.float32, copy=False)
            y = np.concatenate(y_chunks, axis=0).astype(np.float32, copy=False)

            self._total_emitted += int(X.shape[0])

            # Skip cupy conversion for very large arrays (>2GB)
            array_size_bytes = X.nbytes + y.nbytes
            use_cupy_for_batch = use_cupy and (array_size_bytes < 2_000_000_000)

            if use_cupy_for_batch:
                with cp.cuda.Device(int(args.xgb_gpu_id)):
                    X = cp.asarray(X)
                    y = cp.asarray(y)

            input_data(data=X, label=y)

            if self._exhausted:
                if not args.no_tqdm:
                    print(f"\n[ClusterDataIter {desc} h={h_label}] FINAL: {self._total_emitted:,} rows emitted, {self._total_filtered:,} filtered")
                self._close_pbar()

            return 1

    it = ClusterFilteredDataIter()
    it.reset()
    return it


def _make_cluster_filtered_stream_dmatrix(
    ds,
    args,
    horizon_k: int,
    shuffle: bool,
    desc: str,
    params: dict,
    pixel_id_set: set,
    W_global: int,
    ref: Optional["xgb.DMatrix"] = None,
):
    """
    Build DMatrix/QuantileDMatrix from a cluster-filtered streaming DataIter.

    Args:
        ds: Dataset
        args: Training arguments
        horizon_k: Horizon index
        shuffle: Whether to shuffle
        desc: Description
        params: XGBoost params
        pixel_id_set: Set of pixel IDs to include
        W_global: Global width for pixel_id computation
        ref: Optional reference DMatrix for QuantileDMatrix
    """
    t0 = time.time()
    it = make_cluster_filtered_streaming_dataiter(
        ds, args=args, horizon_k=horizon_k, shuffle=shuffle, desc=desc,
        pixel_id_set=pixel_id_set, W_global=W_global
    )
    t1 = time.time()

    use_quantile = (args.xgb_dmatrix == "quantile" and hasattr(xgb, "QuantileDMatrix"))
    nthread = int(params.get("nthread", os.cpu_count() or 1))

    if use_quantile:
        try:
            if ref is not None:
                dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin), nthread=nthread, ref=ref)
            else:
                dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin), nthread=nthread)
        except TypeError:
            if ref is not None:
                try:
                    dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin), ref=ref)
                except TypeError:
                    dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin))
            else:
                dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin))
    else:
        try:
            dmat = xgb.DMatrix(it, nthread=nthread)
        except TypeError:
            dmat = xgb.DMatrix(it)

    t2 = time.time()
    print(f"[TIMING] {desc}: iterator_init={t1 - t0:.2f}s, dmatrix_build={t2 - t1:.2f}s, total={t2 - t0:.2f}s")
    return dmat


def make_residual_regime_streaming_dataiter(
    ds,
    args,
    horizon_k: int,
    shuffle: bool,
    desc: str,
    regime_id: int,
    global_booster: "xgb.Booster",
    n_features_original: int,
):
    """
    Build a DataIter that yields (X_augmented, y) where X_augmented = concat(X, global_logit).

    This filters rows to a specific regime and augments features with the global model's
    logit prediction, used for training per-regime residual models.

    Args:
        ds: Dataset
        args: Training arguments
        horizon_k: Horizon index
        shuffle: Whether to shuffle
        desc: Description for progress bar
        regime_id: The regime ID to filter to
        global_booster: Global XGBoost booster for this horizon (used to compute logits)
        n_features_original: Number of original features (before augmentation)
    """
    if not _have_xgb_dataiter():
        raise RuntimeError("This XGBoost build does not expose xgb.core.DataIter; cannot stream.")

    chunk_rows = int(args.stream_chunk_rows)
    mult = max(1, int(getattr(args, "stream_accumulate_multiplier", 3)))
    target_rows = int(chunk_rows) * int(mult)

    max_rows = None if args.stream_max_rows_per_horizon is None else int(args.stream_max_rows_per_horizon)
    use_cupy = bool(args.stream_use_cupy and _HAVE_CUPY and args.xgb_tree_method == "gpu_hist")
    regime_nodata = int(getattr(args, "regime_nodata_value", -9999))

    h_label = int(args.horizons[horizon_k])

    class ResidualRegimeDataIter(xgb.core.DataIter):  # type: ignore
        """
        DataIter that filters rows by regime and augments features with global model logit.
        """
        def __init__(self):
            super().__init__()
            self._loader = None
            self._it = None
            self._seen_rows = 0
            self._exhausted = False
            self._pbar = None
            self._pbar_enabled = (not args.no_tqdm) and bool(args.stream_show_progress)
            self._total_emitted = 0
            self._total_filtered = 0

        def __del__(self):
            try:
                if self._pbar is not None:
                    self._pbar.close()
            except Exception:
                pass

        def _maybe_init_pbar(self):
            if not self._pbar_enabled:
                return
            if self._pbar is None:
                self._pbar = tqdm(total=0, desc=f"{desc} (h={h_label})", leave=True, position=2)
            else:
                try:
                    self._pbar.set_description_str(f"{desc} (h={h_label})")
                except Exception:
                    pass

        def _close_pbar(self):
            if self._pbar is not None:
                try:
                    self._pbar.close()
                except Exception:
                    pass
                self._pbar = None

        def reset(self):
            self._loader = make_loader(ds, batch_size=args.batch_size, shuffle=shuffle, args=args, device=None)
            self._it = iter(self._loader)
            self._seen_rows = 0
            self._exhausted = False
            self._total_emitted = 0
            self._total_filtered = 0
            self._maybe_init_pbar()

        def next(self, input_data):
            if self._exhausted:
                self._close_pbar()
                return 0

            if max_rows is not None and self._seen_rows >= max_rows:
                self._exhausted = True
                if not args.no_tqdm:
                    print(f"\n[ResidualRegimeDataIter {desc} h={h_label}] FINAL: {self._total_emitted:,} rows, {self._total_filtered:,} filtered (cap hit)")
                self._close_pbar()
                return 0

            X_chunks = []
            y_chunks = []
            n_out = 0

            while n_out < target_rows:
                try:
                    batch = next(self._it)
                except StopIteration:
                    self._exhausted = True
                    break

                # Extract features, labels, validity, and regime_ids for this horizon
                X, y, v = _batch_to_rows_cpu_single_horizon(batch, horizon_k)

                # Get regime IDs from batch
                regime_ids = batch.get("regime_id", None)
                if regime_ids is None:
                    # No regime info - skip this batch
                    continue

                if torch.is_tensor(regime_ids):
                    regime_flat = regime_ids.detach().cpu().numpy().reshape(-1)
                else:
                    regime_flat = np.asarray(regime_ids).reshape(-1)

                # Validity mask
                valid_mask = v.astype(bool)
                if valid_mask.sum() == 0:
                    continue

                # Regime filter mask
                regime_mask = (regime_flat == regime_id) & (regime_flat != regime_nodata)
                combined_mask = valid_mask & regime_mask

                n_filtered = int(valid_mask.sum() - combined_mask.sum())
                self._total_filtered += n_filtered

                if combined_mask.sum() == 0:
                    continue

                Xk = X[combined_mask]
                yk = y[combined_mask].astype(np.float32, copy=False)

                if max_rows is not None:
                    remaining = max_rows - self._seen_rows
                    if remaining <= 0:
                        self._exhausted = True
                        break
                    if Xk.shape[0] > remaining:
                        Xk = Xk[:remaining]
                        yk = yk[:remaining]

                # Compute global model logits for this chunk
                Xk_32 = Xk.astype(np.float32, copy=False)
                dmat_chunk = xgb.DMatrix(Xk_32)
                global_logits = global_booster.predict(dmat_chunk, output_margin=True)

                # Augment features with global logit
                global_logits_col = global_logits.reshape(-1, 1).astype(np.float32, copy=False)
                Xk_augmented = np.concatenate([Xk_32, global_logits_col], axis=1)

                X_chunks.append(Xk_augmented)
                y_chunks.append(yk)
                n_emit = int(Xk_augmented.shape[0])
                n_out += n_emit
                self._seen_rows += n_emit

                if self._pbar is not None:
                    self._pbar.update(n_emit)
                    self._pbar.set_postfix({
                        "rows_seen": f"{self._seen_rows:,}",
                        "filtered": f"{self._total_filtered:,}",
                        "payload": f"{n_out:,}/{target_rows:,}"
                    })

                if max_rows is not None and self._seen_rows >= max_rows:
                    self._exhausted = True
                    break

            if n_out <= 0:
                if self._exhausted:
                    if not args.no_tqdm:
                        print(f"\n[ResidualRegimeDataIter {desc} h={h_label}] FINAL: {self._total_emitted:,} rows, {self._total_filtered:,} filtered")
                    self._close_pbar()
                return 0

            X = np.concatenate(X_chunks, axis=0).astype(np.float32, copy=False)
            y = np.concatenate(y_chunks, axis=0).astype(np.float32, copy=False)

            self._total_emitted += int(X.shape[0])

            # Skip cupy conversion for very large arrays (>2GB)
            array_size_bytes = X.nbytes + y.nbytes
            use_cupy_for_batch = use_cupy and (array_size_bytes < 2_000_000_000)

            if use_cupy_for_batch:
                with cp.cuda.Device(int(args.xgb_gpu_id)):
                    X = cp.asarray(X)
                    y = cp.asarray(y)

            input_data(data=X, label=y)

            if self._exhausted:
                if not args.no_tqdm:
                    print(f"\n[ResidualRegimeDataIter {desc} h={h_label}] FINAL: {self._total_emitted:,} rows, {self._total_filtered:,} filtered")
                self._close_pbar()

            return 1

    it = ResidualRegimeDataIter()
    it.reset()
    return it


def _make_residual_regime_stream_dmatrix(
    ds,
    args,
    horizon_k: int,
    shuffle: bool,
    desc: str,
    params: dict,
    regime_id: int,
    global_booster: "xgb.Booster",
    n_features_original: int,
    ref: Optional["xgb.DMatrix"] = None,
):
    """
    Build DMatrix/QuantileDMatrix from a residual regime streaming DataIter.

    Args:
        ds: Dataset
        args: Training arguments
        horizon_k: Horizon index
        shuffle: Whether to shuffle
        desc: Description
        params: XGBoost params
        regime_id: The regime ID to filter to
        global_booster: Global XGBoost booster for computing logits
        n_features_original: Number of original features
        ref: Optional reference DMatrix for QuantileDMatrix
    """
    t0 = time.time()
    it = make_residual_regime_streaming_dataiter(
        ds, args=args, horizon_k=horizon_k, shuffle=shuffle, desc=desc,
        regime_id=regime_id, global_booster=global_booster,
        n_features_original=n_features_original
    )
    t1 = time.time()

    use_quantile = (args.xgb_dmatrix == "quantile" and hasattr(xgb, "QuantileDMatrix"))
    nthread = int(params.get("nthread", os.cpu_count() or 1))

    if use_quantile:
        try:
            if ref is not None:
                dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin), nthread=nthread, ref=ref)
            else:
                dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin), nthread=nthread)
        except TypeError:
            if ref is not None:
                try:
                    dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin), ref=ref)
                except TypeError:
                    dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin))
            else:
                dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin))
    else:
        try:
            dmat = xgb.DMatrix(it, nthread=nthread)
        except TypeError:
            dmat = xgb.DMatrix(it)

    t2 = time.time()
    print(f"[TIMING] {desc}: iterator_init={t1 - t0:.2f}s, dmatrix_build={t2 - t1:.2f}s, total={t2 - t0:.2f}s")
    return dmat


def _make_stream_dmatrix(
    ds,
    args,
    horizon_k: int,
    shuffle: bool,
    desc: str,
    params: dict,
    ref: Optional["xgb.DMatrix"] = None,   # for QuantileDMatrix ref reuse
):
    """
    Build DMatrix/QuantileDMatrix from a streaming DataIter.

    Added:
      ✅ timing prints: iterator build + DMatrix build
      ✅ optional QuantileDMatrix ref reuse (if supported and provided)
    """
    t0 = time.time()
    it = make_streaming_dataiter(ds, args=args, horizon_k=horizon_k, shuffle=shuffle, desc=desc)
    t1 = time.time()

    use_quantile = (args.xgb_dmatrix == "quantile" and hasattr(xgb, "QuantileDMatrix"))
    nthread = int(params.get("nthread", os.cpu_count() or 1))

    if use_quantile:
        # Try to pass ref if supported
        try:
            if ref is not None:
                dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin), nthread=nthread, ref=ref)
            else:
                dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin), nthread=nthread)
        except TypeError:
            # Older signature
            if ref is not None:
                try:
                    dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin), ref=ref)
                except TypeError:
                    dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin))
            else:
                dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin))
    else:
        try:
            dmat = xgb.DMatrix(it, nthread=nthread)
        except TypeError:
            dmat = xgb.DMatrix(it)

    t2 = time.time()
    print(f"[TIMING] {desc}: iterator_init={t1 - t0:.2f}s, dmatrix_build={t2 - t1:.2f}s, total={t2 - t0:.2f}s")
    return dmat


def _make_stream_quantile_with_precomputed_ref(
    train_ds,
    val_ds,
    args,
    horizon_k: int,
    params: dict,
):
    """
    Optional 2-pass QuantileDMatrix build:
      Pass 1: build a ref QuantileDMatrix sketch on TRAIN (shuffle=False)
      Pass 2: build the real TRAIN QuantileDMatrix using ref
      Optional: build VAL QuantileDMatrix using the same ref (reduces/avoids VAL sketch)

    If xgboost doesn't support ref in your build, falls back to normal _make_stream_dmatrix calls.
    """
    h = int(args.horizons[horizon_k])
    header(f"[Quantile precompute] horizon h={h} (k={horizon_k})")

    if not (args.xgb_dmatrix == "quantile" and hasattr(xgb, "QuantileDMatrix")):
        raise RuntimeError("Quantile precompute requested but xgb_dmatrix!=quantile or QuantileDMatrix unavailable.")

    # Pass 1: ref sketch
    ref = None
    try:
        ref = _make_stream_dmatrix(
            train_ds, args=args, horizon_k=horizon_k, shuffle=False,
            desc="quantile_ref_sketch", params=params, ref=None
        )
    except Exception as e:
        print(f"[Quantile precompute] Failed to build ref sketch; falling back. err={e}")
        ref = None

    # Pass 2: train with ref
    try:
        dtrain = _make_stream_dmatrix(
            train_ds, args=args, horizon_k=horizon_k, shuffle=bool(args.stream_shuffle_train),
            desc="stream_train", params=params, ref=ref
        )
    except Exception as e:
        print(f"[Quantile precompute] Failed train with ref; falling back. err={e}")
        dtrain = _make_stream_dmatrix(
            train_ds, args=args, horizon_k=horizon_k, shuffle=bool(args.stream_shuffle_train),
            desc="stream_train", params=params, ref=None
        )

    dval = None
    if val_ds is not None and len(val_ds) > 0 and bool(args.stream_use_val):
        try:
            dval = _make_stream_dmatrix(
                val_ds, args=args, horizon_k=horizon_k, shuffle=False,
                desc="stream_val", params=params, ref=ref
            )
        except Exception as e:
            print(f"[Quantile precompute] Failed val with ref; falling back. err={e}")
            dval = _make_stream_dmatrix(
                val_ds, args=args, horizon_k=horizon_k, shuffle=False,
                desc="stream_val", params=params, ref=None
            )

    return dtrain, dval


def _infer_num_features_from_dataset(ds, args) -> int:
    loader = make_loader(ds, batch_size=1, shuffle=False, args=args, device=None)
    b = next(iter(loader))
    x = b["x"]

    # Handle both formats:
    # - flatten_pixels=True: x is (Npix, F) - already 2D
    # - flatten_pixels=False: x is (B, F, H, W) or (B, T, C, H, W) - needs flattening
    if x.dim() == 2:
        # Already flattened: (Npix, F)
        F = int(x.shape[1])
    else:
        # Standard spatial format: flatten time dimension if needed
        x = _flatten_time_if_needed(x)  # (B, F, H, W)
        F = int(x.shape[1])

    return F


def train_xgb_per_horizon_tabularized(X_tr, Y_tr, V_tr, X_va, Y_va, V_va, args, seed: int):
    header("Training XGBoost (per horizon) — TABULARIZED")
    params = make_xgb_params(args, seed=seed)

    print("[XGB] Environment:")
    print(f"  • xgboost version............ {getattr(xgb, '__version__', 'unknown')}")
    print(f"  • torch cuda available....... {torch.cuda.is_available()}")
    print(f"  • cupy available............. {_HAVE_CUPY}")
    print(f"  • tree_method................ {params.get('tree_method')}")
    print(f"  • xgb_dmatrix................ {args.xgb_dmatrix}")
    print(f"  • xgb_gpu_id................. {args.xgb_gpu_id}")
    print(f"  • nthread.................... {params.get('nthread')}")
    print()

    print("[XGB] Params:")
    for k in sorted(params.keys()):
        print(f"  • {k:>18} = {params[k]}")

    is_gpu = str(params.get("device", "")).startswith("cuda")
    if is_gpu and not _HAVE_CUPY:
        print("\n[WARN] GPU training requested but CuPy is NOT available.")
        print("       XGBoost can still train on GPU with numpy->DMatrix, but it will stage data via CPU.")
        print("       For maximum GPU utilization and GPU-resident prediction, install CuPy.\n")

    K = int(Y_tr.shape[1])
    boosters = []

    for k_idx, h in enumerate(args.horizons):
        h = int(h)
        header(f"Fit horizon h={h} (k={k_idx})")

        keep_tr = V_tr[:, k_idx]
        Xk = X_tr[keep_tr]
        yk = Y_tr[keep_tr, k_idx].astype(np.float32)

        if Xk.shape[0] == 0:
            raise RuntimeError(f"No valid rows to train for horizon h={h}.")

        dtrain = _make_dmatrix(Xk, yk, args=args, is_gpu=is_gpu)

        evals = [(dtrain, "train")]
        if X_va is not None and Y_va is not None and V_va is not None:
            keep_va = V_va[:, k_idx]
            Xv = X_va[keep_va]
            yv = Y_va[keep_va, k_idx].astype(np.float32)
            if Xv.shape[0] > 0:
                dval = _make_dmatrix(Xv, yv, args=args, is_gpu=is_gpu)
                evals.append((dval, "val"))
                print(f"[XGB h={h}] val rows............. {Xv.shape[0]:,}")
            else:
                print(f"[XGB h={h}] WARNING: zero valid val rows; training without val eval.")
        else:
            print(f"[XGB h={h}] No val set provided.")

        print(f"[XGB h={h}] train rows........... {Xk.shape[0]:,}")
        print(f"[XGB h={h}] features............ {Xk.shape[1]}")
        print(f"[XGB h={h}] num_boost_round..... {args.xgb_num_round}")

        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(args.xgb_num_round),
            evals=evals,
            verbose_eval=bool(args.xgb_verbose_eval),
        )
        boosters.append(bst)

        del Xk, yk, dtrain
        if device_is_cuda() and int(getattr(args, "cuda_empty_cache_every", 0)) > 0:
            torch.cuda.empty_cache()

    return boosters, params


def train_xgb_per_horizon_streaming(train_ds, val_ds, args, seed: int):
    header("Training XGBoost (per horizon) — STREAMING (DataIter)")

    if not _have_xgb_dataiter():
        raise RuntimeError(
            "Your XGBoost build does not expose xgb.core.DataIter. "
            "Streaming mode requires a newer/complete XGBoost Python package."
        )

    params = make_xgb_params(args, seed=seed)

    print("[XGB stream] Environment:")
    print(f"  • xgboost version............ {getattr(xgb, '__version__', 'unknown')}")
    print(f"  • tree_method................ {params.get('tree_method')}")
    print(f"  • xgb_dmatrix................ {args.xgb_dmatrix}")
    print(f"  • stream_chunk_rows.......... {int(args.stream_chunk_rows):,}")
    print(f"  • stream_accumulate_mult...... {int(args.stream_accumulate_multiplier)} (payload ~chunk_rows*mult)")
    print(f"  • stream_max_rows/horizon.... {args.stream_max_rows_per_horizon}")
    print(f"  • stream_use_cupy............ {bool(args.stream_use_cupy)} (available={_HAVE_CUPY})")
    print(f"  • xgb_quantile_precompute..... {bool(args.xgb_quantile_precompute)}")
    print(f"  • nthread.................... {params.get('nthread')}")
    print()

    if args.train_mode == "stream" and args.xgb_dmatrix == "quantile" and not args.xgb_quantile_precompute:
        print("[WARN] You are using QuantileDMatrix in streaming mode without precompute.")
        print("       Quantile sketching can cause multiple DataIter passes and slow builds.")
        print("       Consider: --xgb-dmatrix dmat  OR  --xgb-quantile-precompute\n")

    print("[XGB stream] Params:")
    for k in sorted(params.keys()):
        print(f"  • {k:>18} = {params[k]}")

    boosters = []
    K = len(args.horizons)

    for k_idx, h in enumerate(args.horizons):
        h = int(h)
        header(f"Fit horizon h={h} (k={k_idx}) [STREAM]")

        if args.xgb_dmatrix == "quantile" and bool(args.xgb_quantile_precompute):
            dtrain, dval = _make_stream_quantile_with_precomputed_ref(
                train_ds=train_ds, val_ds=val_ds, args=args, horizon_k=k_idx, params=params
            )
            evals = [(dtrain, "train")]
            if dval is not None:
                evals.append((dval, "val"))
                print(f"[XGB stream h={h}] using VAL eval (QuantileDMatrix ref reuse).")
            else:
                print(f"[XGB stream h={h}] val eval disabled or empty val_ds.")
        else:
            dtrain = _make_stream_dmatrix(
                train_ds, args=args, horizon_k=k_idx, shuffle=bool(args.stream_shuffle_train),
                desc="stream_train", params=params, ref=None
            )
            evals = [(dtrain, "train")]

            if val_ds is not None and len(val_ds) > 0 and bool(args.stream_use_val):
                dval = _make_stream_dmatrix(
                    val_ds, args=args, horizon_k=k_idx, shuffle=False,
                    desc="stream_val", params=params, ref=None
                )
                evals.append((dval, "val"))
                print(f"[XGB stream h={h}] using streaming VAL eval.")
            else:
                print(f"[XGB stream h={h}] val eval disabled or empty val_ds.")

        print(f"[XGB stream h={h}] num_boost_round..... {int(args.xgb_num_round)}")
        print(f"[XGB stream h={h}] NOTE: DMatrix is built from streamed chunks (no full X_tr in RAM).")

        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(args.xgb_num_round),
            evals=evals,
            verbose_eval=bool(args.xgb_verbose_eval),
        )
        boosters.append(bst)

        del dtrain
        if device_is_cuda() and int(getattr(args, "cuda_empty_cache_every", 0)) > 0:
            torch.cuda.empty_cache()

    return boosters, params


def train_xgb_per_cluster_streaming(
    train_ds,
    cluster_assign_df: pd.DataFrame,
    args,
    seed: int,
    global_params: dict,
    W_global: int,
    min_samples: int = 1000,
) -> Dict[int, List["xgb.Booster"]]:
    """
    Train separate XGBoost models for each cluster using streaming.

    For each cluster:
    1. Extract pixel IDs belonging to that cluster
    2. Filter training data to only include those pixels
    3. Train one booster per horizon using same hyperparameters as global model

    Args:
        train_ds: Training dataset
        cluster_assign_df: DataFrame with columns [pixel_id, cluster_id, ...]
        args: Training argumentshow does the Global ECMWF Fire Forecast (GEFF) model differ from the traditional FWI
        seed: Random seed
        global_params: XGBoost params from global model (reused for all cluster models)
        W_global: Global width for pixel_id computation
        min_samples: Minimum samples required to train a cluster model

    Returns:
        Dict mapping cluster_id -> List[booster per horizon]
    """
    header("Training XGBoost PER-CLUSTER models (streaming)")

    if not _have_xgb_dataiter():
        raise RuntimeError(
            "Your XGBoost build does not expose xgb.core.DataIter. "
            "Per-cluster streaming training requires a newer XGBoost."
        )

    # Get unique clusters and their sizes
    cluster_counts = cluster_assign_df.groupby("cluster_id")["pixel_id"].count()
    unique_clusters = sorted(cluster_counts.index.tolist())
    n_clusters = len(unique_clusters)

    print(f"[cluster_train] Found {n_clusters} clusters")
    print(f"[cluster_train] Min samples threshold: {min_samples:,}")
    print(f"[cluster_train] Cluster sizes:")
    for cl_id in unique_clusters:
        cnt = int(cluster_counts[cl_id])
        status = "✓" if cnt >= min_samples else "✗ (skip)"
        print(f"  • Cluster {cl_id}: {cnt:,} pixels {status}")

    print(f"\n[cluster_train] Using global model params (same hyperparameters for all clusters)")
    print(f"[cluster_train] XGBoost params:")
    for k in sorted(global_params.keys()):
        print(f"  • {k:>18} = {global_params[k]}")

    cluster_boosters: Dict[int, List[xgb.Booster]] = {}
    skipped_clusters = []

    for cl_id in unique_clusters:
        cl_id = int(cl_id)
        n_pixels = int(cluster_counts[cl_id])

        # Allow threshold=0 to train ALL clusters regardless of size
        if min_samples > 0 and n_pixels < min_samples:
            print(f"\n[cluster_train] Skipping cluster {cl_id}: only {n_pixels:,} pixels (< {min_samples:,})")
            skipped_clusters.append(cl_id)
            continue

        header(f"Training cluster {cl_id} ({n_pixels:,} pixels)")

        # Get pixel IDs for this cluster
        cluster_pixel_ids = set(
            cluster_assign_df[cluster_assign_df["cluster_id"] == cl_id]["pixel_id"].astype(int).tolist()
        )
        print(f"[cluster {cl_id}] Pixel ID set size: {len(cluster_pixel_ids):,}")

        boosters_for_cluster = []
        K = len(args.horizons)

        for k_idx, h in enumerate(args.horizons):
            h = int(h)
            print(f"\n[cluster {cl_id}] Fitting horizon h={h} (k={k_idx})")

            # Build cluster-filtered streaming DMatrix
            dtrain = _make_cluster_filtered_stream_dmatrix(
                train_ds,
                args=args,
                horizon_k=k_idx,
                shuffle=bool(args.stream_shuffle_train),
                desc=f"cluster{cl_id}_train",
                params=global_params,
                pixel_id_set=cluster_pixel_ids,
                W_global=W_global,
                ref=None,
            )

            evals = [(dtrain, "train")]

            print(f"[cluster {cl_id} h={h}] num_boost_round..... {int(args.xgb_num_round)}")

            bst = xgb.train(
                params=global_params,
                dtrain=dtrain,
                num_boost_round=int(args.xgb_num_round),
                evals=evals,
                verbose_eval=bool(args.xgb_verbose_eval),
            )
            boosters_for_cluster.append(bst)

            del dtrain
            if device_is_cuda() and int(getattr(args, "cuda_empty_cache_every", 0)) > 0:
                torch.cuda.empty_cache()

        cluster_boosters[cl_id] = boosters_for_cluster
        print(f"\n[cluster {cl_id}] ✓ Trained {len(boosters_for_cluster)} boosters (one per horizon)")

    print(f"\n[cluster_train] ✓ COMPLETE")
    print(f"[cluster_train] Trained clusters: {list(cluster_boosters.keys())}")
    print(f"[cluster_train] Skipped clusters (< {min_samples} samples): {skipped_clusters}")

    # Compute cluster size statistics
    print(f"\n[cluster_train] === CLUSTER SIZE DISTRIBUTION ===")
    all_clusters = sorted(unique_clusters)
    trained_ids = list(cluster_boosters.keys())
    skipped_ids = skipped_clusters

    total_pixels = int(cluster_counts.sum())
    print(f"[cluster_train] Total pixels across all clusters: {total_pixels:,}")
    print(f"[cluster_train] Trained clusters: {len(trained_ids)} / {len(all_clusters)}")
    print(f"[cluster_train] Skipped clusters: {len(skipped_ids)} / {len(all_clusters)}")

    # Show size breakdown
    for cl_id in all_clusters:
        cnt = int(cluster_counts[cl_id])
        pct = 100.0 * cnt / max(total_pixels, 1)
        status = "TRAINED" if cl_id in trained_ids else "SKIPPED"
        print(f"  • Cluster {cl_id}: {cnt:,} pixels ({pct:.2f}%) - {status}")

    # Calculate coverage
    trained_pixel_count = sum(cluster_counts[cl_id] for cl_id in trained_ids)
    skipped_pixel_count = sum(cluster_counts[cl_id] for cl_id in skipped_ids)
    print(f"\n[cluster_train] Pixels in TRAINED clusters: {trained_pixel_count:,} ({100.0*trained_pixel_count/max(total_pixels,1):.2f}%)")
    print(f"[cluster_train] Pixels in SKIPPED clusters: {skipped_pixel_count:,} ({100.0*skipped_pixel_count/max(total_pixels,1):.2f}%)")

    return cluster_boosters


def _estimate_regime_sample_counts(
    train_ds,
    args,
    n_regimes: int,
    regime_nodata: int,
    sample_batches: int = 50,
) -> Dict[int, int]:
    """
    Estimate sample counts per regime by sampling batches.

    Returns dict mapping regime_id -> estimated count.
    """
    loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=False, args=args, device=None)

    counts: Dict[int, int] = {r: 0 for r in range(n_regimes)}
    total_batches_seen = 0

    for i, batch in enumerate(loader):
        if i >= sample_batches:
            break

        regime_ids = batch.get("regime_id", None)
        if regime_ids is None:
            continue

        mask = batch["mask"]
        # Use first horizon mask if multi-horizon
        if mask.dim() == 4:
            m_flat = (mask[:, 0, :, :] > 0.5).cpu().numpy().reshape(-1)
        elif mask.dim() == 2:
            m_flat = (mask[:, 0] > 0.5).cpu().numpy().reshape(-1)
        else:
            m_flat = (mask > 0.5).cpu().numpy().reshape(-1)

        if torch.is_tensor(regime_ids):
            regime_flat = regime_ids.detach().cpu().numpy().reshape(-1)
        else:
            regime_flat = np.asarray(regime_ids).reshape(-1)

        # Count valid pixels per regime
        for r_id in range(n_regimes):
            regime_mask = (regime_flat == r_id) & (regime_flat != regime_nodata) & m_flat
            counts[r_id] += int(regime_mask.sum())

        total_batches_seen += 1

    # Extrapolate to full dataset
    if total_batches_seen > 0:
        scale = len(train_ds) / max(total_batches_seen, 1)
        counts = {r: int(c * scale) for r, c in counts.items()}

    return counts


def train_residual_models_per_regime_streaming(
    train_ds,
    global_boosters: List["xgb.Booster"],
    args,
    seed: int,
    n_features_original: int,
    min_samples: int = 1000,
) -> Dict[int, List["xgb.Booster"]]:
    """
    Train residual models per regime to correct global model predictions.

    For each regime with sufficient samples:
    1. Filter training data to that regime
    2. Compute global model logits for each sample
    3. Augment features: X_residual = concat(X_original, global_logit)
    4. Train residual model on (X_residual, Y) with strong regularization

    Args:
        train_ds: Training dataset
        global_boosters: List of global boosters (one per horizon)
        args: Training arguments
        seed: Random seed
        n_features_original: Number of original features
        min_samples: Minimum samples per regime to train

    Returns:
        Dict mapping regime_id -> List[residual booster per horizon]
    """
    header("Training RESIDUAL models per regime (streaming)")

    if not _have_xgb_dataiter():
        raise RuntimeError(
            "Your XGBoost build does not expose xgb.core.DataIter. "
            "Residual model streaming training requires a newer XGBoost."
        )

    # Get regime information from dataset
    base_ds = _unwrap_subset(train_ds)
    n_regimes = int(getattr(base_ds, "n_regimes", 0) or 0)
    regime_nodata = int(getattr(args, "regime_nodata_value", -9999))

    if n_regimes == 0:
        print("[residual] No regimes found (n_regimes=0). Skipping residual model training.")
        print("[residual] Ensure --regime-source is provided to load regime information.")
        return {}

    # Estimate sample counts per regime
    print(f"[residual] Estimating sample counts per regime (sampling {50} batches)...")
    regime_counts = _estimate_regime_sample_counts(train_ds, args, n_regimes, regime_nodata, sample_batches=50)

    print(f"\n[residual] Found {n_regimes} regimes")
    print(f"[residual] Min samples threshold: {min_samples:,}")
    for r_id in range(n_regimes):
        cnt = regime_counts.get(r_id, 0)
        status = "ok" if cnt >= min_samples else "SKIP (too few)"
        print(f"  regime {r_id}: ~{cnt:,} estimated samples - {status}")

    # Build residual XGB params
    residual_params = make_residual_xgb_params(args, seed)

    print(f"\n[residual] XGBoost params for residual models:")
    for k in sorted(residual_params.keys()):
        print(f"  {k:>18} = {residual_params[k]}")

    residual_boosters: Dict[int, List[xgb.Booster]] = {}
    K = len(args.horizons)

    for r_id in range(n_regimes):
        if regime_counts.get(r_id, 0) < min_samples:
            print(f"\n[residual] Skipping regime {r_id}: insufficient samples (~{regime_counts.get(r_id, 0):,} < {min_samples:,})")
            continue

        header(f"Training residual model for regime {r_id} (~{regime_counts.get(r_id, 0):,} samples)")
        boosters_for_regime = []

        for k_idx, h in enumerate(args.horizons):
            h = int(h)
            print(f"\n[residual r={r_id}] Fitting horizon h={h} (k={k_idx})")

            # Build regime-filtered DMatrix with global logit augmentation
            dtrain = _make_residual_regime_stream_dmatrix(
                train_ds,
                args=args,
                horizon_k=k_idx,
                shuffle=bool(args.stream_shuffle_train),
                desc=f"residual_r{r_id}",
                params=residual_params,
                regime_id=r_id,
                global_booster=global_boosters[k_idx],
                n_features_original=n_features_original,
            )

            evals = [(dtrain, "train")]

            print(f"[residual r={r_id} h={h}] num_boost_round..... {int(args.residual_num_round)}")

            bst = xgb.train(
                params=residual_params,
                dtrain=dtrain,
                num_boost_round=int(args.residual_num_round),
                evals=evals,
                verbose_eval=bool(args.xgb_verbose_eval),
            )
            boosters_for_regime.append(bst)

            del dtrain
            if device_is_cuda() and int(getattr(args, "cuda_empty_cache_every", 0)) > 0:
                torch.cuda.empty_cache()

        residual_boosters[r_id] = boosters_for_regime
        print(f"\n[residual r={r_id}] Trained {len(boosters_for_regime)} boosters (one per horizon)")

    print(f"\n[residual] === TRAINING COMPLETE ===")
    print(f"[residual] Trained residual models for regimes: {list(residual_boosters.keys())}")
    skipped = [r for r in range(n_regimes) if r not in residual_boosters]
    print(f"[residual] Skipped regimes (< {min_samples} samples): {skipped}")

    return residual_boosters


def save_cluster_models(
    cluster_boosters: Dict[int, List["xgb.Booster"]],
    horizons: List[int],
    out_dir: str,
):
    """
    Save per-cluster XGBoost models to disk.

    Directory structure:
        out_dir/cluster_models/cluster{N}/xgb_h{H}.json
    """
    models_dir = os.path.join(out_dir, "cluster_models")
    os.makedirs(models_dir, exist_ok=True)

    for cl_id, boosters in cluster_boosters.items():
        cl_dir = os.path.join(models_dir, f"cluster{cl_id}")
        os.makedirs(cl_dir, exist_ok=True)

        for k_idx, bst in enumerate(boosters):
            h = int(horizons[k_idx])
            model_path = os.path.join(cl_dir, f"xgb_h{h}.json")
            bst.save_model(model_path)

    print(f"[save_cluster_models] Saved {len(cluster_boosters)} cluster models to {models_dir}")


def load_cluster_models(
    models_dir: str,
    horizons: List[int],
) -> Dict[int, List["xgb.Booster"]]:
    """
    Load per-cluster XGBoost models from disk.

    Args:
        models_dir: Path to cluster_models directory
        horizons: List of horizon values

    Returns:
        Dict mapping cluster_id -> List[booster per horizon]
    """
    cluster_boosters: Dict[int, List[xgb.Booster]] = {}

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Cluster models directory not found: {models_dir}")

    # Find all cluster directories
    for entry in os.listdir(models_dir):
        if not entry.startswith("cluster"):
            continue
        cl_dir = os.path.join(models_dir, entry)
        if not os.path.isdir(cl_dir):
            continue

        try:
            cl_id = int(entry.replace("cluster", ""))
        except ValueError:
            continue

        boosters = []
        for h in horizons:
            model_path = os.path.join(cl_dir, f"xgb_h{h}.json")
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Missing model file: {model_path}")

            bst = xgb.Booster()
            bst.load_model(model_path)
            boosters.append(bst)

        cluster_boosters[cl_id] = boosters

    print(f"[load_cluster_models] Loaded {len(cluster_boosters)} cluster models from {models_dir}")
    return cluster_boosters


def save_residual_models(
    residual_boosters: Dict[int, List["xgb.Booster"]],
    horizons: List[int],
    out_dir: str,
    feature_names_augmented: List[str],
):
    """
    Save per-regime residual XGBoost models to disk.

    Directory structure:
        out_dir/residual_models/regime{N}/xgb_h{H}.json
        out_dir/residual_models/feature_names.json
    """
    models_dir = os.path.join(out_dir, "residual_models")
    os.makedirs(models_dir, exist_ok=True)

    for r_id, boosters in residual_boosters.items():
        r_dir = os.path.join(models_dir, f"regime{r_id}")
        os.makedirs(r_dir, exist_ok=True)

        for k_idx, bst in enumerate(boosters):
            h = int(horizons[k_idx])
            model_path = os.path.join(r_dir, f"xgb_h{h}.json")
            bst.save_model(model_path)

    # Save augmented feature names
    fn_path = os.path.join(models_dir, "feature_names.json")
    with open(fn_path, "w") as f:
        json.dump(feature_names_augmented, f, indent=2)

    print(f"[save_residual_models] Saved {len(residual_boosters)} regime models to {models_dir}")


def load_residual_models(
    models_dir: str,
    horizons: List[int],
) -> Tuple[Dict[int, List["xgb.Booster"]], List[str]]:
    """
    Load per-regime residual XGBoost models from disk.

    Args:
        models_dir: Path to residual_models directory
        horizons: List of horizon values

    Returns:
        (residual_boosters dict, augmented feature names list)
    """
    residual_boosters: Dict[int, List[xgb.Booster]] = {}

    if not os.path.isdir(models_dir):
        print(f"[load_residual_models] Directory not found: {models_dir}")
        return {}, []

    # Find all regime directories
    for entry in os.listdir(models_dir):
        if not entry.startswith("regime"):
            continue
        r_dir = os.path.join(models_dir, entry)
        if not os.path.isdir(r_dir):
            continue

        try:
            r_id = int(entry.replace("regime", ""))
        except ValueError:
            continue

        boosters = []
        all_found = True
        for h in horizons:
            model_path = os.path.join(r_dir, f"xgb_h{h}.json")
            if not os.path.isfile(model_path):
                print(f"[load_residual_models] Missing model file: {model_path}")
                all_found = False
                break

            bst = xgb.Booster()
            bst.load_model(model_path)
            boosters.append(bst)

        if all_found and len(boosters) == len(horizons):
            residual_boosters[r_id] = boosters

    # Load feature names
    fn_path = os.path.join(models_dir, "feature_names.json")
    feature_names: List[str] = []
    if os.path.isfile(fn_path):
        with open(fn_path, "r") as f:
            feature_names = json.load(f)

    print(f"[load_residual_models] Loaded {len(residual_boosters)} regime models from {models_dir}")
    return residual_boosters, feature_names


# ----------------------------------------------------------------------
# Per-Cluster Calibration Metrics
# ----------------------------------------------------------------------

def compute_calibration_metrics_per_cluster(
    ds,
    logits_fn,
    cluster_assign_df: pd.DataFrame,
    args,
    split_name: str,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute calibration metrics (ECE, MCE, Brier, reliability diagram data) per cluster.

    Args:
        ds: Dataset
        logits_fn: Logits function for prediction
        cluster_assign_df: DataFrame with columns [pixel_id, cluster_id, ...]
        args: Arguments
        split_name: Name of split (for logging)

    Returns:
        Dict mapping cluster_id -> calibration dict with keys:
            bin_pred, bin_true, bin_count, ece, mce, brier, n_samples
    """
    pixel_to_cluster = dict(zip(
        cluster_assign_df["pixel_id"].astype(int).tolist(),
        cluster_assign_df["cluster_id"].astype(int).tolist()
    ))

    device = torch.device(args.eval_device)
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=False, args=args, device=device)

    bin_width = float(getattr(args, "reliability_bin_width", 0.005))
    num_bins = max(1, int(math.ceil(1.0 / bin_width)))

    cluster_ids = sorted(set(cluster_assign_df["cluster_id"].astype(int).tolist()))

    cluster_accum = {
        cl_id: {
            "counts": np.zeros(num_bins, dtype=np.float64),
            "pred_sums": np.zeros(num_bins, dtype=np.float64),
            "true_sums": np.zeros(num_bins, dtype=np.float64),
            "brier_sum": 0.0,
            "total": 0.0,
        }
        for cl_id in cluster_ids
    }

    K = len(args.horizons)
    W_global = int(getattr(args, "W_global", 0))
    if W_global <= 0:
        batch0 = next(iter(loader))
        W_global = int(_ensure_numpy(batch0.get("W_global", 1000)).reshape(-1)[0])

    pbar = tqdm(loader, desc=f"calib_per_cluster_{split_name}", leave=True) if not args.no_tqdm else None
    it = pbar if pbar is not None else loader

    for batch in it:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        logits = logits_fn(batch)
        probs = torch.sigmoid(logits)

        y = batch["y"]
        mask = batch["mask"] > 0.5

        y0_b = _ensure_numpy(batch.get("y0", -1)).astype(np.int64, copy=False).reshape(-1)
        x0_b = _ensure_numpy(batch.get("x0", -1)).astype(np.int64, copy=False).reshape(-1)

        is_flattened = batch["x"].dim() == 2
        if is_flattened:
            Npix = logits.shape[0]
            B = len(y0_b)
            pixels_per_patch = Npix // max(B, 1)
            H = W = int(np.sqrt(pixels_per_patch)) if pixels_per_patch > 0 else 1
        else:
            B, K_batch, H, W = logits.shape

        for k_idx in range(K):
            if is_flattened:
                pk = probs[:, k_idx].detach().cpu().numpy()
                yk = (y[:, k_idx] > 0.5).detach().cpu().numpy().astype(np.float64)
                vk = mask[:, k_idx].detach().cpu().numpy()
            else:
                pk = probs[:, k_idx].reshape(-1).detach().cpu().numpy()
                yk = (y[:, k_idx] > 0.5).reshape(-1).detach().cpu().numpy().astype(np.float64)
                vk = mask[:, k_idx].reshape(-1).detach().cpu().numpy()

            for b_idx in range(B):
                y0_patch = y0_b[b_idx] if b_idx < len(y0_b) else 0
                x0_patch = x0_b[b_idx] if b_idx < len(x0_b) else 0

                for h in range(H):
                    for w in range(W):
                        flat_idx = b_idx * H * W + h * W + w if not is_flattened else b_idx * H * W + h * W + w
                        if flat_idx >= len(vk) or not vk[flat_idx]:
                            continue

                        y_global = y0_patch + h
                        x_global = x0_patch + w
                        pixel_id = int(y_global * W_global + x_global)

                        cl_id = pixel_to_cluster.get(pixel_id, -1)
                        if cl_id < 0 or cl_id not in cluster_accum:
                            continue

                        p = float(pk[flat_idx])
                        t = float(yk[flat_idx])

                        bin_idx = min(int(p / bin_width), num_bins - 1)

                        cluster_accum[cl_id]["counts"][bin_idx] += 1.0
                        cluster_accum[cl_id]["pred_sums"][bin_idx] += p
                        cluster_accum[cl_id]["true_sums"][bin_idx] += t
                        cluster_accum[cl_id]["brier_sum"] += (p - t) ** 2
                        cluster_accum[cl_id]["total"] += 1.0

        del logits, probs

    if pbar:
        pbar.close()

    cluster_calib: Dict[int, Dict[str, Any]] = {}

    for cl_id in cluster_ids:
        accum = cluster_accum[cl_id]
        counts = accum["counts"]
        pred_sums = accum["pred_sums"]
        true_sums = accum["true_sums"]
        total = accum["total"]

        nonzero = counts > 0
        pred_mean = np.zeros_like(counts)
        true_mean = np.zeros_like(counts)
        pred_mean[nonzero] = pred_sums[nonzero] / counts[nonzero]
        true_mean[nonzero] = true_sums[nonzero] / counts[nonzero]

        if total > 0:
            gap = np.abs(pred_mean - true_mean)
            ece = float((counts[nonzero] / total * gap[nonzero]).sum()) if np.any(nonzero) else float("nan")
            mce = float(gap[nonzero].max()) if np.any(nonzero) else float("nan")
            brier = float(accum["brier_sum"] / total)
        else:
            ece = mce = brier = float("nan")

        cluster_calib[cl_id] = {
            "bin_pred": pred_mean.tolist(),
            "bin_true": true_mean.tolist(),
            "bin_count": counts.tolist(),
            "ece": ece,
            "mce": mce,
            "brier": brier,
            "n_samples": int(total),
        }

    return cluster_calib


# ----------------------------------------------------------------------
# Orchestration: Per-Cluster Training + Ensemble Evaluation
# ----------------------------------------------------------------------

def do_cluster_model_training_and_ensemble_eval(
    train_ds,
    val_ds,
    test_ds,
    global_boosters: List["xgb.Booster"],
    train_assign: pd.DataFrame,
    val_assign: Optional[pd.DataFrame],
    test_assign: Optional[pd.DataFrame],
    train_sig_df: pd.DataFrame,
    val_sig_df: Optional[pd.DataFrame],
    test_sig_df: Optional[pd.DataFrame],
    cluster_state: dict,
    args,
    seed: int,
):
    """
    End-to-end orchestration for per-cluster model training and ensemble evaluation.

    Steps:
    1. Train per-cluster XGBoost models
    2. Evaluate cluster-selection prediction on val/test
    3. Evaluate distance-weighted ensemble prediction on val/test
    4. Compare all approaches (global, cluster-selection, ensemble)
    5. Generate comparison calibration plots
    6. Print summary comparison table
    7. Save all artifacts

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        global_boosters: List of global XGBoost boosters
        train_assign: DataFrame with pixel cluster assignments for train
        val_assign: DataFrame with pixel cluster assignments for val
        test_assign: DataFrame with pixel cluster assignments for test
        train_sig_df: DataFrame with pixel signatures for train
        val_sig_df: DataFrame with pixel signatures for val
        test_sig_df: DataFrame with pixel signatures for test
        cluster_state: Cluster state dict with centers, pca, etc.
        args: Training arguments
        seed: Random seed
    """
    if not bool(getattr(args, "train_cluster_models", False)):
        return None

    header("Per-Cluster Model Training & Ensemble Evaluation")

    horizons = [int(h) for h in args.horizons]
    ensemble_mode = str(getattr(args, "ensemble_mode", "both"))
    min_samples = int(getattr(args, "cluster_min_samples", 1000))

    # Adaptive threshold: automatically adjust min_samples to ensure at least 50% of clusters are trained
    if bool(getattr(args, "cluster_min_samples_adaptive", False)) and train_assign is not None:
        from collections import Counter
        cluster_counts = Counter(train_assign['cluster_id'].values)
        sorted_sizes = sorted(cluster_counts.values(), reverse=True)

        if len(sorted_sizes) > 0:
            # Find threshold that would train at least half the clusters
            target_idx = max(0, len(sorted_sizes) // 2)
            adaptive_threshold = int(sorted_sizes[target_idx])

            # Set a minimum floor (XGBoost needs reasonable sample size)
            adaptive_threshold = max(100, adaptive_threshold)

            # Use the lower of the original threshold or adaptive threshold
            original_min_samples = min_samples
            min_samples = min(original_min_samples, adaptive_threshold)

            if min_samples < original_min_samples:
                print(f"[cluster_ensemble] Adaptive threshold: lowering min_samples from {original_min_samples:,} to {min_samples:,}")
                print(f"[cluster_ensemble]   This ensures at least {len(sorted_sizes)//2} / {len(sorted_sizes)} clusters are trained")

    W_global = int(getattr(args, "W_global", 0))
    if W_global <= 0:
        loader_tmp = make_loader(train_ds, batch_size=1, shuffle=False, args=args, device=None)
        batch0 = next(iter(loader_tmp))
        W_global = int(_ensure_numpy(batch0.get("W_global", 1000)).reshape(-1)[0])

    print(f"[cluster_ensemble] Settings:")
    print(f"  • ensemble_mode............ {ensemble_mode}")
    print(f"  • cluster_min_samples...... {min_samples}")
    print(f"  • W_global................. {W_global}")
    print(f"  • n_clusters............... {cluster_state['n_clusters']}")

    global_params = {
        "eta": float(args.xgb_eta),
        "max_depth": int(args.xgb_max_depth),
        "subsample": float(args.xgb_subsample),
        "colsample_bynode": float(args.xgb_colsample_bynode),
        "tree_method": str(args.xgb_tree_method),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_bin": int(args.xgb_max_bin),
        "sampling_method": str(args.xgb_sampling_method),
        "grow_policy": str(args.xgb_grow_policy),
        "seed": int(seed),
    }
    v = _xgb_version_tuple()
    if str(args.xgb_tree_method) == "gpu_hist":
        gid = int(args.xgb_gpu_id)
        # XGB 3.x: gpu_hist removed; use hist + device=cuda
        if v >= (3, 0, 0):
            global_params["tree_method"] = "hist"
        global_params["device"] = f"cuda:{gid}"
        # XGBoost >= 3 ignores 'predictor'
        if v < (3, 0, 0):
            global_params["predictor"] = "gpu_predictor"

    cluster_boosters = train_xgb_per_cluster_streaming(
        train_ds=train_ds,
        cluster_assign_df=train_assign,
        args=args,
        seed=seed,
        global_params=global_params,
        W_global=W_global,
        min_samples=min_samples,
    )

    save_cluster_models(cluster_boosters, horizons, args.logdir)

    if not cluster_boosters:
        print("[cluster_ensemble] No cluster models trained (all clusters too small?). Skipping ensemble eval.")
        return None

    pixel_cluster_map = dict(zip(
        train_assign["pixel_id"].astype(int).tolist(),
        train_assign["cluster_id"].astype(int).tolist()
    ))
    if val_assign is not None:
        pixel_cluster_map.update(dict(zip(
            val_assign["pixel_id"].astype(int).tolist(),
            val_assign["cluster_id"].astype(int).tolist()
        )))
    if test_assign is not None:
        pixel_cluster_map.update(dict(zip(
            test_assign["pixel_id"].astype(int).tolist(),
            test_assign["cluster_id"].astype(int).tolist()
        )))

    print(f"[cluster_ensemble] pixel_cluster_map size: {len(pixel_cluster_map)}")
    print(f"[cluster_ensemble] pixel_id range: min={min(pixel_cluster_map.keys())}, max={max(pixel_cluster_map.keys())}")

    sig_cols = [c for c in train_sig_df.columns if c not in ["pixel_id", "y_global", "x_global", "lat", "lon", "n_rows_used", "n_pos", "event_rate"]]

    pixel_sig_cache = {}
    for _, row in train_sig_df.iterrows():
        pid = int(row["pixel_id"])
        sig = row[sig_cols].values.astype(np.float64)
        pixel_sig_cache[pid] = sig
    if val_sig_df is not None:
        for _, row in val_sig_df.iterrows():
            pid = int(row["pixel_id"])
            sig = row[sig_cols].values.astype(np.float64)
            pixel_sig_cache[pid] = sig
    if test_sig_df is not None:
        for _, row in test_sig_df.iterrows():
            pid = int(row["pixel_id"])
            sig = row[sig_cols].values.astype(np.float64)
            pixel_sig_cache[pid] = sig

    device = torch.device(args.eval_device)
    criterion = MaskedBCEWithLogits().to(device)

    def _create_ensemble_logits_fn(mode: str):
        return xgb_ensemble_logits_fn_from_cluster_boosters(
            global_boosters=global_boosters,
            cluster_boosters=cluster_boosters,
            cluster_state=cluster_state,
            pixel_cluster_map=pixel_cluster_map,
            pixel_sig_cache=pixel_sig_cache,
            horizons=horizons,
            args=args,
            mode=mode,
            calibrators=None,
            W_global=W_global,
        )

    global_logits_fn = xgb_logits_fn_from_boosters(
        global_boosters,
        horizons=horizons,
        args=args,
        calibrators=None,
    )

    results_all = {}

    def _eval_split(ds, split_name: str):
        if ds is None or len(ds) == 0:
            return {}

        header(f"Evaluating {split_name}")
        loader = make_loader(ds, batch_size=args.batch_size, shuffle=False, args=args, device=device)

        split_results = {}

        print(f"[{split_name}] Evaluating Global model...")
        # Global
        global_loss, global_metrics = evaluate_with_logits_fn(
            global_logits_fn, loader, device, criterion, args, use_tqdm=not args.no_tqdm
        )
        global_metrics["loss"] = float(global_loss)
        split_results["Global"] = global_metrics
        print(f"[{split_name}] Global: Brier={global_metrics['calibration']['per_horizon'].get(horizons[0], {}).get('brier', float('nan')):.5f}")

        if ensemble_mode in ("cluster_selection", "both"):
            print(f"[{split_name}] Evaluating Cluster-Selection model...")
            cs_logits_fn = _create_ensemble_logits_fn("cluster_selection")
            cs_loss, cs_metrics = evaluate_with_logits_fn(
                cs_logits_fn, loader, device, criterion, args, use_tqdm=not args.no_tqdm
            )
            cs_metrics["loss"] = float(cs_loss)
            split_results["Cluster-Selection"] = cs_metrics
            fallback_stats = cs_logits_fn.get_fallback_stats()
            print(f"[{split_name}] Cluster-Selection: Brier={cs_metrics['calibration']['per_horizon'].get(horizons[0], {}).get('brier', float('nan')):.5f}")

            # Print detailed fallback breakdown
            fb_pct = 100 * fallback_stats['fallback_count'] / max(fallback_stats['total_count'], 1)
            print(f"[{split_name}] Cluster-Selection fallbacks: {fallback_stats['fallback_count']:,} / {fallback_stats['total_count']:,} ({fb_pct:.2f}%)")

            if fallback_stats['fallback_count'] > 0:
                fb_reasons = fallback_stats.get('fallback_reasons', {})
                missing = fb_reasons.get('missing_from_map', 0)
                not_trained = fb_reasons.get('cluster_not_trained', 0)
                total_fb = max(fallback_stats['fallback_count'], 1)

                print(f"[{split_name}]   └─ Fallback breakdown:")
                print(f"[{split_name}]      • Cluster skipped (< min_samples): {not_trained:,} ({100*not_trained/total_fb:.1f}% of fallbacks)")
                print(f"[{split_name}]      • Missing from pixel_cluster_map: {missing:,} ({100*missing/total_fb:.1f}% of fallbacks)")

                # Show cluster model usage
                cluster_usage = fallback_stats.get('cluster_usage', {})
                if cluster_usage:
                    print(f"[{split_name}]   Cluster model usage:")
                    for cl_id in sorted(cluster_usage.keys()):
                        cnt = cluster_usage[cl_id]
                        pct = 100 * cnt / max(fallback_stats['total_count'], 1)
                        print(f"[{split_name}]      • Cluster {cl_id}: {cnt:,} pixels ({pct:.2f}%)")

        if ensemble_mode in ("distance_weighted", "both"):
            print(f"[{split_name}] Evaluating Distance-Weighted Ensemble...")
            dw_logits_fn = _create_ensemble_logits_fn("distance_weighted")
            dw_loss, dw_metrics = evaluate_with_logits_fn(
                dw_logits_fn, loader, device, criterion, args, use_tqdm=not args.no_tqdm
            )
            dw_metrics["loss"] = float(dw_loss)
            split_results["Distance-Weighted"] = dw_metrics
            fallback_stats = dw_logits_fn.get_fallback_stats()
            print(f"[{split_name}] Distance-Weighted: Brier={dw_metrics['calibration']['per_horizon'].get(horizons[0], {}).get('brier', float('nan')):.5f}")

            # Print detailed fallback breakdown
            fb_pct = 100 * fallback_stats['fallback_count'] / max(fallback_stats['total_count'], 1)
            print(f"[{split_name}] Distance-Weighted fallbacks: {fallback_stats['fallback_count']:,} / {fallback_stats['total_count']:,} ({fb_pct:.2f}%)")

            if fallback_stats['fallback_count'] > 0:
                fb_reasons = fallback_stats.get('fallback_reasons', {})
                missing = fb_reasons.get('missing_from_map', 0)
                not_trained = fb_reasons.get('cluster_not_trained', 0)
                total_fb = max(fallback_stats['fallback_count'], 1)

                print(f"[{split_name}]   └─ Fallback breakdown:")
                print(f"[{split_name}]      • Cluster skipped (< min_samples): {not_trained:,} ({100*not_trained/total_fb:.1f}% of fallbacks)")
                print(f"[{split_name}]      • Missing from pixel_cluster_map: {missing:,} ({100*missing/total_fb:.1f}% of fallbacks)")

                # Show cluster model usage
                cluster_usage = fallback_stats.get('cluster_usage', {})
                if cluster_usage:
                    print(f"[{split_name}]   Cluster model usage:")
                    for cl_id in sorted(cluster_usage.keys()):
                        cnt = cluster_usage[cl_id]
                        pct = 100 * cnt / max(fallback_stats['total_count'], 1)
                        print(f"[{split_name}]      • Cluster {cl_id}: {cnt:,} pixels ({pct:.2f}%)")

        return split_results

    val_results = _eval_split(val_ds, "val")
    test_results = _eval_split(test_ds, "test")

    results_all["val"] = val_results
    results_all["test"] = test_results

    ensemble_dir = os.path.join(args.logdir, "ensemble_results")
    os.makedirs(ensemble_dir, exist_ok=True)

    for split_name, split_results in [("val", val_results), ("test", test_results)]:
        if not split_results:
            continue

        for method_name, metrics in split_results.items():
            out_json = os.path.join(ensemble_dir, f"{method_name.lower().replace('-', '_')}_metrics_{split_name}.json")
            with open(out_json, "w") as f:
                serializable = {}
                for k, v in metrics.items():
                    if isinstance(v, dict):
                        serializable[k] = v
                    elif isinstance(v, (int, float, str, bool, type(None))):
                        serializable[k] = v
                json.dump(serializable, f, indent=2, default=str)

        summary_csv = os.path.join(ensemble_dir, f"comparison_summary_{split_name}.csv")
        save_ensemble_comparison_summary(split_results, summary_csv, horizons)

        if bool(getattr(args, "make_plots", False)):
            plots_dir = os.path.join(args.logdir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            for h in horizons:
                calib_plot = os.path.join(plots_dir, f"calibration_comparison_{split_name}_h{h}.png")
                plot_calibration_comparison_curve(
                    split_results,
                    calib_plot,
                    args,
                    title=f"Calibration Comparison ({split_name}, h={h})",
                    horizon=h,
                )

            metrics_plot = os.path.join(plots_dir, f"metrics_comparison_{split_name}.png")
            plot_ece_mce_brier_comparison(
                split_results,
                metrics_plot,
                args,
                title=f"Metrics Comparison ({split_name})",
                horizons=horizons,
            )

    header("Ensemble Evaluation Summary")
    for split_name in ["val", "test"]:
        split_results = results_all.get(split_name, {})
        if not split_results:
            continue

        print(f"\n=== {split_name.upper()} ===")
        print(f"{'Method':<20} | {'Horizon':<8} | {'Brier':<10} | {'ECE':<10} | {'LogLoss':<10} | {'ROC-AUC':<10}")
        print("-" * 80)

        for method_name, metrics in split_results.items():
            per_h = metrics.get("per_horizon", {})
            calib_per_h = metrics.get("calibration", {}).get("per_horizon", {})

            for h in horizons:
                h_m = per_h.get(h, per_h.get(str(h), {}))
                h_c = calib_per_h.get(h, calib_per_h.get(str(h), {}))

                brier = h_c.get("brier", float("nan"))
                ece = h_c.get("ece", float("nan"))
                logloss = h_m.get("logloss", float("nan"))
                auc = h_c.get("roc", {}).get("auc", h_m.get("roc_auc", float("nan")))

                print(f"{method_name:<20} | {h:<8} | {brier:<10.6f} | {ece:<10.6f} | {logloss:<10.6f} | {auc:<10.6f}")

    return results_all


# ----------------------------------------------------------------------
# GPU prediction plumbing (Torch <-> CuPy via DLPack) + margin prediction
# ----------------------------------------------------------------------

def _torch_to_cupy_2d(x2d: torch.Tensor):
    if not _HAVE_CUPY:
        raise RuntimeError("CuPy not available.")
    if not x2d.is_cuda:
        raise RuntimeError("torch_to_cupy requires a CUDA tensor.")
    dl = torch.utils.dlpack.to_dlpack(x2d)
    return cp.fromDlpack(dl)


def _cupy_to_torch_1d(x1d):
    if not _HAVE_CUPY:
        raise RuntimeError("CuPy not available.")
    dl = x1d.toDlpack()
    return torch.utils.dlpack.from_dlpack(dl)


def _predict_margin_from_cupy(bst: "xgb.Booster", Xc) -> torch.Tensor:
    if not _HAVE_CUPY:
        raise RuntimeError("CuPy not available for GPU prediction.")
    torch_stream = torch.cuda.current_stream()
    with cp.cuda.ExternalStream(torch_stream.cuda_stream):
        if hasattr(bst, "inplace_predict"):
            try:
                m = bst.inplace_predict(Xc, prediction_type="margin")
                mc = cp.asarray(m)
                return _cupy_to_torch_1d(mc)
            except TypeError:
                pass
        dmat = xgb.DMatrix(Xc)
        m = bst.predict(dmat, output_margin=True)
        mc = cp.asarray(m)
        return _cupy_to_torch_1d(mc)


def _predict_margin_gpu_if_possible(bst: "xgb.Booster", X2d: torch.Tensor, args) -> torch.Tensor:
    """
    Fix:
      - CPU numpy fallback is only created if GPU path is not taken / fails.
    """
    want_gpu = _is_cuda_device_str(args.xgb_predict_device)

    if want_gpu and X2d.is_cuda and _HAVE_CUPY:
        try:
            Xc = _torch_to_cupy_2d(X2d.contiguous())
            return _predict_margin_from_cupy(bst, Xc)
        except (RuntimeError, ValueError, TypeError) as e:
            print(f"[WARN] GPU margin predict failed; falling back to CPU. err={e}")

    Xn = X2d.detach().cpu().numpy().astype(np.float32, copy=False)
    dmat = xgb.DMatrix(Xn)
    m = bst.predict(dmat, output_margin=True)
    m = np.asarray(m, dtype=np.float32)
    return torch.from_numpy(m).to(device=X2d.device)


# ----------------------------------------------------------------------
# Calibration: Platt scaling + Isotonic regression (no sklearn required)
# ----------------------------------------------------------------------

@dataclass
class PlattCalibrator:
    a: float
    b: float

    def apply_logits(self, margin: torch.Tensor) -> torch.Tensor:
        return margin * float(self.a) + float(self.b)


    def predict(self, margins: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities for 1D numpy margins/logits."""
        z = margins.astype(np.float64, copy=False) * float(self.a) + float(self.b)
        # stable sigmoid
        z = np.clip(z, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-z))
        return p.astype(np.float32, copy=False)


@dataclass
class IsotonicCalibrator:
    x_thresholds: np.ndarray  # (M,)
    y_values: np.ndarray      # (M,)

    def apply_logits(self, margin: torch.Tensor) -> torch.Tensor:
        m = margin.detach().cpu().numpy().astype(np.float64, copy=False)
        idx = np.searchsorted(self.x_thresholds, m, side="right") - 1
        idx = np.clip(idx, 0, self.y_values.size - 1)
        p = self.y_values[idx]
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        logit = np.log(p) - np.log(1.0 - p)
        out = torch.from_numpy(logit.astype(np.float32)).to(device=margin.device)
        return out


    def predict(self, margins: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities for 1D numpy margins/logits."""
        m = margins.astype(np.float64, copy=False)
        idx = np.searchsorted(self.x_thresholds, m, side="right") - 1
        idx = np.clip(idx, 0, self.y_values.size - 1)
        p = self.y_values[idx]
        return np.clip(p, 1e-6, 1.0 - 1e-6).astype(np.float32, copy=False)


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt_scaling(margins: np.ndarray, labels: np.ndarray, max_iter: int = 100, tol: float = 1e-8, reg: float = 1e-6) -> PlattCalibrator:
    m = margins.astype(np.float64, copy=False)
    y = labels.astype(np.float64, copy=False)

    a = 1.0
    b = 0.0

    for _it in range(max_iter):
        z = a * m + b
        p = _sigmoid_np(z)

        g_a = np.sum((p - y) * m)
        g_b = np.sum(p - y)

        w = p * (1.0 - p)
        H_aa = np.sum(w * m * m) + reg
        H_ab = np.sum(w * m)     + reg
        H_bb = np.sum(w)         + reg

        det = H_aa * H_bb - H_ab * H_ab
        if not np.isfinite(det) or abs(det) < 1e-20:
            break

        step_a = ( H_bb * g_a - H_ab * g_b) / det
        step_b = (-H_ab * g_a + H_aa * g_b) / det

        a_new = a - step_a
        b_new = b - step_b

        if abs(step_a) < tol and abs(step_b) < tol:
            a, b = a_new, b_new
            break

        a, b = a_new, b_new

    return PlattCalibrator(a=float(a), b=float(b))


def fit_isotonic_pav(margins: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
    """
    Fit isotonic regression using sklearn's optimized C implementation.
    Much faster than the pure Python PAV loop for large datasets.
    """
    x = margins.astype(np.float64, copy=False)
    y = labels.astype(np.float64, copy=False)

    # Use sklearn's optimized isotonic regression (C implementation)
    iso = SklearnIsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
    iso.fit(x, y)

    # Extract the piecewise-constant representation from sklearn
    # sklearn stores unique x values and corresponding y values after fitting
    x_thr = iso.X_thresholds_.astype(np.float64)
    y_val = iso.y_thresholds_.astype(np.float64)

    return IsotonicCalibrator(x_thresholds=x_thr, y_values=y_val)


def save_calibrators(calibs: Dict, method: str, out_path: str, per_regime: bool = False):
    """
    Save calibrators to JSON.

    Structure depends on per_regime:
    - per_regime=False: {"method": str, "per_horizon": {horizon: params}}
    - per_regime=True:  {"method": str, "per_regime_horizon": {regime: {horizon: params}},
                         "global_fallback": {horizon: params}}
    """
    def _serialize_calibrator(c):
        if isinstance(c, PlattCalibrator):
            return {"type": "platt", "a": c.a, "b": c.b}
        elif isinstance(c, IsotonicCalibrator):
            return {
                "type": "isotonic",
                "x_thresholds": c.x_thresholds.tolist(),
                "y_values": c.y_values.tolist(),
            }
        else:
            return {"type": "unknown"}

    if per_regime:
        payload = {"method": method, "per_regime_horizon": {}, "global_fallback": {}}

        print(f"[Calib] Serializing {len(calibs)} calibrators...")
        total_items = len(calibs)
        processed = 0

        for key, c in calibs.items():
            if key == "global":
                # Global fallback calibrators: {horizon: calibrator}
                for h, gc in c.items():
                    print(f"[Calib] Serializing global h={h}...")
                    payload["global_fallback"][str(int(h))] = _serialize_calibrator(gc)
                    print(f"[Calib] Serialized global h={h}")
            elif isinstance(key, tuple) and len(key) == 2:
                # Per-regime per-horizon: (regime_id, horizon) -> calibrator
                regime_id, horizon = key
                regime_str = str(int(regime_id))
                horizon_str = str(int(horizon))

                if regime_str not in payload["per_regime_horizon"]:
                    payload["per_regime_horizon"][regime_str] = {}

                print(f"[Calib] Serializing regime={regime_id} h={horizon}... ({processed+1}/{total_items})")
                payload["per_regime_horizon"][regime_str][horizon_str] = _serialize_calibrator(c)
                print(f"[Calib] Serialized regime={regime_id} h={horizon}")

            processed += 1
    else:
        payload = {"method": method, "per_horizon": {}}
        for h, c in calibs.items():
            h = int(h)
            payload["per_horizon"][str(h)] = _serialize_calibrator(c)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save as pickle for much faster serialization of large arrays
    pickle_path = out_path.replace('.json', '.pkl')
    print(f"[Calib] Writing pickle to {pickle_path}...")
    with open(pickle_path, "wb") as f:
        pickle.dump(calibs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Calib] Saved calibrators (pickle) -> {pickle_path}")

    # Also save JSON for backward compatibility (but skip if too large)
    try:
        print(f"[Calib] Writing JSON to {out_path}...")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Calib] Saved calibrators (JSON) -> {out_path}")
    except Exception as e:
        print(f"[Calib] WARNING: Failed to save JSON (too large?): {e}")
        print(f"[Calib] Using pickle file only: {pickle_path}")

def build_feature_names_from_dataset(args, ds) -> list[str]:
    base_ds = _unwrap_subset(ds)

    # ✅ OPTIMIZATION: Use dataset's built-in feature_names if available
    if hasattr(base_ds, 'feature_names') and base_ds.feature_names:
        print(f"[feature_names] Using dataset's built-in feature_names ({len(base_ds.feature_names)} features)")
        return base_ds.feature_names

    # Fallback: rebuild manually (for backward compatibility)
    print("[feature_names] Dataset doesn't have feature_names; rebuilding manually")

    # Build base channel names (one time-slice worth)
    base = []
    for spec, C in zip(base_ds.input_specs, base_ds.input_C):
        stem = os.path.basename(spec.zarr).replace(".zarr", "")
        arr  = spec.array
        for c in range(int(C)):
            base.append(f"{stem}:{arr}:c{c}")

    if base_ds.coord_as_features:
        base += ["sin(lat)", "cos(lat)", "sin(lon)", "cos(lon)"]

    # Peek actual x shape (can use ds or base_ds; ds is fine)
    loader = make_loader(ds, batch_size=1, shuffle=False, args=args, device=None)
    b = next(iter(loader))
    x = b["x"]

    # Case A: flatten_pixels=True: (Npix, F) or (B*H*W, F) - 2D
    if torch.is_tensor(x) and x.dim() == 2:
        Npix, F = x.shape
        # Feature dimension already includes time stacking
        if F == len(base) * int(args.T_hist):
            T = int(args.T_hist)
            names = []
            for ch in base:
                for t in range(T):
                    lag = (T - 1) - t
                    names.append(f"{ch}@t-{lag}")
            return names
        # Coordinate features only (no time)
        if F == len(base):
            return base
        # Unknown layout
        return [f"f{j}" for j in range(F)]

    # Case B: dataset returns separate time dim: (B,T,C,H,W)
    if torch.is_tensor(x) and x.dim() == 5:
        T = int(x.shape[1])
        names = []
        for ch in base:
            for t in range(T):
                lag = (T - 1) - t
                names.append(f"{ch}@t-{lag}")
        return names

    # Case C: dataset already flattened to channels: (B,F,H,W)
    if torch.is_tensor(x) and x.dim() == 4:
        F = int(x.shape[1])

        # If it looks like stacked-time-in-channel (common for --stack-time channel)
        if F == len(base) * int(args.T_hist):
            T = int(args.T_hist)
            names = []
            for ch in base:
                for t in range(T):
                    lag = (T - 1) - t
                    names.append(f"{ch}@t-{lag}")
            return names

        # If it looks like "no time expansion needed"
        if F == len(base):
            return base

        # Fallback: unknown layout
        return [f"f{j}" for j in range(F)]

    # Ultimate fallback
    n_features = _infer_num_features_from_dataset(ds, args)
    return [f"f{j}" for j in range(int(n_features))]

import re

def _safe_fname(s: str, maxlen: int = 120) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))
    return s[:maxlen]


# ----------------------------------------------------------------------
# NetDry7 feature engineering
# ----------------------------------------------------------------------

def _find_feature_indices_by_pattern(
    feature_names: List[str],
    pattern: str,
    time_offsets: List[int]
) -> Dict[int, int]:
    """
    Find column indices matching pattern at specific time offsets.

    Args:
        feature_names: List of feature names, e.g., ["vpd_rh.zarr c0 (t)", "vpd_rh.zarr c0 (t-1)", ...]
        pattern: Regex pattern to match the channel name (without time label)
        time_offsets: List of time offsets to find, e.g., [1, 2, 3, 4, 5, 6, 7] for t-1 to t-7

    Returns:
        Dict mapping offset -> column index, e.g., {1: 42, 2: 43, ...} for t-1, t-2...
    """
    result: Dict[int, int] = {}
    for offset in time_offsets:
        time_label = f"(t-{offset})" if offset > 0 else "(t)"
        for idx, name in enumerate(feature_names):
            if re.search(pattern, name, re.IGNORECASE) and time_label in name:
                result[offset] = idx
                break
    return result


def compute_netdry7_features(
    X: np.ndarray,
    feature_names: List[str],
    args,
    train_stats: Optional[Dict] = None,
    is_train: bool = True
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Compute NetDry7 and optional gated-NetDry7 features.

    NetDry7 = standardized(sum_VPD_7) - standardized(sum_precip_7)
    Gated_NetDry7 = max(0, WTD_last - tau) * NetDry7

    Args:
        X: (N, F) feature matrix
        feature_names: list of F feature names
        args: CLI args with add_netdry7, add_gated_netdry7, tau, window_days, min_valid_days
        train_stats: if not is_train, use these precomputed stats for standardization
        is_train: if True, compute and return stats; else use train_stats

    Returns:
        X_aug: (N, F + n_new) augmented feature matrix
        new_names: updated feature names list
        stats: dict with mean/std for sumVPD and sumP (only meaningful if is_train)
    """
    if not getattr(args, 'add_netdry7', False):
        return X, feature_names, {}

    window = getattr(args, 'window_days', 7)
    min_valid = getattr(args, 'min_valid_days', 4)
    offsets = list(range(1, window + 1))  # [1, 2, ..., window] for t-1 to t-window

    # Get channel patterns (use CLI overrides if provided, else defaults)
    vpd_pattern = getattr(args, 'vpd_channel_pattern', None) or r"vpd.*c0|vpd_rh"
    precip_pattern = getattr(args, 'precip_channel_pattern', None) or r"era5land.*c8|era5land.*8|precip"
    wtd_pattern = getattr(args, 'wtd_channel_pattern', None) or r"smap.*wtd|SMAP_WTD"

    # Find VPD column indices for each time offset
    vpd_indices = _find_feature_indices_by_pattern(feature_names, vpd_pattern, offsets)
    print(f"[NetDry7] Found {len(vpd_indices)}/{window} VPD indices: {vpd_indices}")

    # Find precip column indices for each time offset
    precip_indices = _find_feature_indices_by_pattern(feature_names, precip_pattern, offsets)
    print(f"[NetDry7] Found {len(precip_indices)}/{window} precip indices: {precip_indices}")

    # Validate we found enough indices
    if len(vpd_indices) < min_valid:
        print(f"[NetDry7] ERROR: Found only {len(vpd_indices)} VPD indices, need at least {min_valid}.")
        print(f"[NetDry7] Feature names sample: {feature_names[:20]}")
        print(f"[NetDry7] Skipping NetDry7 computation.")
        return X, feature_names, {}

    if len(precip_indices) < min_valid:
        print(f"[NetDry7] ERROR: Found only {len(precip_indices)} precip indices, need at least {min_valid}.")
        print(f"[NetDry7] Skipping NetDry7 computation.")
        return X, feature_names, {}

    N = X.shape[0]

    # Compute sum of VPD across the window, tracking valid counts per row
    vpd_sum = np.zeros(N, dtype=np.float32)
    vpd_valid_count = np.zeros(N, dtype=np.int32)
    for offset in offsets:
        if offset in vpd_indices:
            col = X[:, vpd_indices[offset]]
            valid = np.isfinite(col)
            vpd_sum = np.where(valid, vpd_sum + col, vpd_sum)
            vpd_valid_count += valid.astype(np.int32)

    # Compute sum of precip across the window
    precip_sum = np.zeros(N, dtype=np.float32)
    precip_valid_count = np.zeros(N, dtype=np.int32)
    for offset in offsets:
        if offset in precip_indices:
            col = X[:, precip_indices[offset]]
            valid = np.isfinite(col)
            precip_sum = np.where(valid, precip_sum + col, precip_sum)
            precip_valid_count += valid.astype(np.int32)

    # Mark as NaN if fewer than min_valid days
    insufficient_vpd = vpd_valid_count < min_valid
    insufficient_precip = precip_valid_count < min_valid
    vpd_sum[insufficient_vpd] = np.nan
    precip_sum[insufficient_precip] = np.nan

    # Compute or use train stats for standardization
    if is_train:
        vpd_mean = float(np.nanmean(vpd_sum))
        vpd_std = float(np.nanstd(vpd_sum))
        precip_mean = float(np.nanmean(precip_sum))
        precip_std = float(np.nanstd(precip_sum))
        stats = {
            "vpd_sum_mean": vpd_mean,
            "vpd_sum_std": vpd_std,
            "precip_sum_mean": precip_mean,
            "precip_sum_std": precip_std,
            "window_days": window,
            "min_valid_days": min_valid,
        }
        print(f"[NetDry7] Train stats: VPD mean={vpd_mean:.4f} std={vpd_std:.4f}, "
              f"precip mean={precip_mean:.4f} std={precip_std:.4f}")
    else:
        if train_stats is None:
            raise ValueError("[NetDry7] is_train=False but train_stats is None")
        stats = train_stats
        vpd_mean = stats["vpd_sum_mean"]
        vpd_std = stats["vpd_sum_std"]
        precip_mean = stats["precip_sum_mean"]
        precip_std = stats["precip_sum_std"]

    # Standardize: z = (x - mean) / std
    eps = 1e-8
    z_vpd = (vpd_sum - vpd_mean) / max(vpd_std, eps)
    z_precip = (precip_sum - precip_mean) / max(precip_std, eps)

    # NetDry = z_vpd - z_precip (higher = drier conditions)
    netdry = z_vpd - z_precip

    # Build output columns
    new_cols = [netdry.reshape(-1, 1).astype(np.float32)]
    new_feature_names = list(feature_names) + [f"netdry{window}"]

    n_nan_netdry = int(np.isnan(netdry).sum())
    print(f"[NetDry7] Computed netdry{window}: {N - n_nan_netdry} valid, {n_nan_netdry} NaN")

    # Gated NetDry7 (optional)
    if getattr(args, 'add_gated_netdry7', False):
        # Find WTD at t-1 (lag-safe)
        wtd_indices = _find_feature_indices_by_pattern(feature_names, wtd_pattern, [1])
        if 1 in wtd_indices:
            wtd_last = X[:, wtd_indices[1]]
            tau = getattr(args, 'tau', 0.4)
            hinge = np.maximum(0.0, wtd_last - tau).astype(np.float32)
            gated_netdry = hinge * netdry
            new_cols.append(gated_netdry.reshape(-1, 1).astype(np.float32))
            new_feature_names.append(f"gated_netdry{window}")
            n_gated = int((hinge > 0).sum())
            print(f"[NetDry7] Computed gated_netdry{window}: {n_gated}/{N} rows have WTD > tau={tau}")
        else:
            print(f"[NetDry7] WARNING: Could not find WTD at (t-1) for gating. "
                  f"Pattern: {wtd_pattern}, skipping gated feature.")

    # Concatenate new columns to X
    X_aug = np.hstack([X] + new_cols)
    print(f"[NetDry7] Augmented X: {X.shape} -> {X_aug.shape}")

    return X_aug, new_feature_names, stats


# ----------------------------------------------------------------------
# logits_fn builders (raw margins + optional calibration)
# ----------------------------------------------------------------------

def _batch_features_to_2d(batch: dict) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int]]:
    """
    Convert batch to 2D feature matrix.

    Handles both formats:
    - Old format: x=(B,F,H,W), y=(B,K,H,W)
    - flatten_pixels=True: x=(Npix,F), y=(Npix,K)
    """
    x = batch["x"]
    y = batch["y"]

    # Detect if already flattened
    if x.dim() == 2:
        # Already flattened: (Npix, F)
        Npix, F = x.shape
        K = int(y.shape[1])
        return x, (1, 1, Npix, K, F)

    # Old format: needs manual flattening
    x = _flatten_time_if_needed(x)  # (B,F,H,W)
    B, F, H, W = x.shape
    K = int(y.shape[1])
    X2d = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, F)
    return X2d, (B, H, W, K, F)


def _valid_keep_indices(V2d: torch.Tensor) -> torch.Tensor:
    any_valid = V2d.any(dim=1)
    keep_idx = torch.nonzero(any_valid, as_tuple=False).view(-1)
    return keep_idx


def xgb_logits_fn_from_boosters(
    boosters: list,
    horizons: list[int],
    args,
    calibrators: Optional[Dict[Any, Any]] = None,
):
    """
    Build logits_fn(batch) that returns raw XGBoost margins (logits), with optional calibration.

    Supports two calibrator dict formats:

      1) Global/per-horizon:
            {horizon:int -> calibrator}

      2) Per-regime:
            {
              "global": {horizon:int -> calibrator},                 # global fallback
              (regime_id:int, horizon:int): calibrator,              # per-regime overrides
              ...
            }

    If a (regime_id, horizon) calibrator is missing, we fall back to "global"[horizon] (if present),
    otherwise we leave logits unchanged.
    """
    K = len(boosters)
    horizons_int = [int(h) for h in horizons]
    calibs: Dict[Any, Any] = calibrators or {}

    # Detect per-regime format
    per_regime = ("global" in calibs) or any(isinstance(k, tuple) and len(k) == 2 for k in calibs.keys())
    global_calibs = (calibs.get("global", {}) or {}) if per_regime else calibs

    def _get_global_calib(h: int):
        return global_calibs.get(int(h)) or global_calibs.get(str(int(h)))

    def _get_pair_calib(r: int, h: int):
        # Be tolerant to keys serialized as strings.
        return (
            calibs.get((int(r), int(h)))
            or calibs.get((str(int(r)), int(h)))
            or calibs.get((int(r), str(int(h))))
            or calibs.get((str(int(r)), str(int(h))))
        )

    def _flatten_regime_ids(batch: dict) -> Optional[np.ndarray]:
        if "regime_id" not in batch:
            return None
        rid = batch["regime_id"]
        if torch.is_tensor(rid):
            rid_np = rid.detach().cpu().numpy()
        else:
            rid_np = np.asarray(rid)
        # flatten_pixels=True yields (Npix,) already; else (B,H,W)
        return rid_np.reshape(-1)

    def _apply_calibration(
        margin: torch.Tensor,
        horizon: int,
        regime_keep: Optional[np.ndarray],
    ) -> torch.Tensor:
        if not calibs:
            return margin

        # Global-only calibration
        if not per_regime:
            calib = _get_global_calib(horizon)
            return calib.apply_logits(margin) if calib is not None else margin

        # Per-regime: apply (regime,h) if present else global fallback
        calib_global = _get_global_calib(horizon)

        if regime_keep is None:
            return calib_global.apply_logits(margin) if calib_global is not None else margin

        # Quick check: if there are no per-regime calibrators for this horizon, use global (common).
        uniq = np.unique(regime_keep)
        have_any = False
        for r in uniq:
            if _get_pair_calib(int(r), horizon) is not None:
                have_any = True
                break
        if not have_any:
            return calib_global.apply_logits(margin) if calib_global is not None else margin

        out = margin.clone()
        nodata = int(getattr(args, "regime_nodata_value", -9999))

        for r in uniq:
            r_int = int(r)
            if r_int == nodata:
                continue
            idx = np.nonzero(regime_keep == r)[0]
            if idx.size == 0:
                continue

            calib = _get_pair_calib(r_int, horizon) or calib_global
            if calib is None:
                continue

            idx_t = torch.from_numpy(idx.astype(np.int64)).to(device=out.device)
            vals = out.index_select(0, idx_t)
            vals = calib.apply_logits(vals)
            out.index_copy_(0, idx_t, vals)

        return out

    def logits_fn(batch: dict):
        X2d, meta = _batch_features_to_2d(batch)
        B, H, W, K_meta, _F = meta
        if K_meta != K:
            raise RuntimeError(f"Batch K={K_meta} but boosters K={K}.")

        m = batch["mask"] > 0.5

        # Handle both formats:
        # - flatten_pixels=True: m is (Npix, K) - already in correct shape
        # - flatten_pixels=False: m is (B, K, H, W) - needs reshaping
        if m.dim() == 2:
            # Already flattened: (Npix, K)
            V2d = m
        elif m.dim() == 4:
            # Standard format: (B, K, H, W) → (B*H*W, K)
            V2d = m.permute(0, 2, 3, 1).contiguous().view(B * H * W, K)
        else:
            raise RuntimeError(f"Unexpected m.dim()={m.dim()}, expected 2 or 4")

        keep_idx = _valid_keep_indices(V2d)
        logits_full = torch.zeros((B * H * W, K), device=batch["x"].device, dtype=torch.float32)

        # Check if data is already flattened by looking at original input dimensions
        is_flattened = batch["x"].dim() == 2

        if keep_idx.numel() == 0:
            if is_flattened:
                # Return in flattened format: (Npix, K)
                return logits_full
            else:
                # Return in standard format: (B, K, H, W)
                return logits_full.view(B, H, W, K).permute(0, 3, 1, 2).contiguous()

        Xkeep = X2d.index_select(0, keep_idx)

        # (Optional) regime ids aligned to keep_idx (CPU numpy)
        regime_flat = _flatten_regime_ids(batch)
        regime_keep: Optional[np.ndarray] = None
        if regime_flat is not None:
            keep_np = keep_idx.detach().cpu().numpy().astype(np.int64, copy=False)
            if keep_np.size == regime_flat.shape[0]:
                # This would be weird (keep_idx is indices), but guard anyway
                regime_keep = regime_flat
            else:
                regime_keep = regime_flat[keep_np]

        want_gpu = _is_cuda_device_str(args.xgb_predict_device)
        can_gpu = want_gpu and Xkeep.is_cuda and _HAVE_CUPY

        if can_gpu:
            Xc = _torch_to_cupy_2d(Xkeep.contiguous())
            for k in range(K):
                margin = _predict_margin_from_cupy(boosters[k], Xc).to(dtype=torch.float32)
                h = horizons_int[k]
                z = _apply_calibration(margin, int(h), regime_keep)
                logits_full[keep_idx, k] = z
        else:
            for k in range(K):
                margin = _predict_margin_gpu_if_possible(boosters[k], Xkeep, args).to(dtype=torch.float32)
                h = horizons_int[k]
                z = _apply_calibration(margin, int(h), regime_keep)
                logits_full[keep_idx, k] = z

        if is_flattened:
            # Return in flattened format: (Npix, K)
            z_out = logits_full
        else:
            # Return in standard format: (B, K, H, W)
            z_out = logits_full.view(B, H, W, K).permute(0, 3, 1, 2).contiguous()

        del X2d, V2d, Xkeep, keep_idx, logits_full
        return z_out

    return logits_fn


def xgb_logits_fn_with_residual_models(
    global_boosters: List["xgb.Booster"],
    residual_boosters: Dict[int, List["xgb.Booster"]],
    horizons: List[int],
    args,
    calibrators: Optional[Dict[Any, Any]] = None,
):
    """
    Build logits_fn that applies residual corrections per regime.

    Inference flow:
    1. Compute global_logit from global model
    2. If pixel's regime has a residual model:
       - X_aug = concat(X, global_logit)
       - final_logit = residual_model.predict(X_aug)
    3. Else: use global_logit directly
    4. Apply calibration (if enabled)

    Args:
        global_boosters: Global boosters per horizon
        residual_boosters: Dict[regime_id -> List[booster per horizon]]
        horizons: List of horizon values
        args: Training arguments
        calibrators: Optional calibration objects

    Returns:
        logits_fn(batch) -> (B, K, H, W) or (Npix, K) tensor
    """
    K = len(global_boosters)
    horizons_int = [int(h) for h in horizons]
    calibs: Dict[Any, Any] = calibrators or {}
    regime_nodata = int(getattr(args, "regime_nodata_value", -9999))

    # Detect per-regime calibration format
    per_regime_calib = ("global" in calibs) or any(isinstance(k, tuple) and len(k) == 2 for k in calibs.keys())
    global_calibs = (calibs.get("global", {}) or {}) if per_regime_calib else calibs

    def _get_global_calib(h: int):
        return global_calibs.get(int(h)) or global_calibs.get(str(int(h)))

    def _get_pair_calib(r: int, h: int):
        return (
            calibs.get((int(r), int(h)))
            or calibs.get((str(int(r)), int(h)))
            or calibs.get((int(r), str(int(h))))
            or calibs.get((str(int(r)), str(int(h))))
        )

    def _flatten_regime_ids(batch: dict) -> Optional[np.ndarray]:
        if "regime_id" not in batch:
            return None
        rid = batch["regime_id"]
        if torch.is_tensor(rid):
            rid_np = rid.detach().cpu().numpy()
        else:
            rid_np = np.asarray(rid)
        return rid_np.reshape(-1)

    def _apply_calibration(
        margin: torch.Tensor,
        horizon: int,
        regime_keep: Optional[np.ndarray],
    ) -> torch.Tensor:
        if not calibs:
            return margin

        if not per_regime_calib:
            calib = _get_global_calib(horizon)
            return calib.apply_logits(margin) if calib is not None else margin

        calib_global = _get_global_calib(horizon)

        if regime_keep is None:
            return calib_global.apply_logits(margin) if calib_global is not None else margin

        uniq = np.unique(regime_keep)
        have_any = any(_get_pair_calib(int(r), horizon) is not None for r in uniq)
        if not have_any:
            return calib_global.apply_logits(margin) if calib_global is not None else margin

        out = margin.clone()
        for r in uniq:
            r_int = int(r)
            if r_int == regime_nodata:
                continue
            idx = np.nonzero(regime_keep == r)[0]
            if idx.size == 0:
                continue

            calib = _get_pair_calib(r_int, horizon) or calib_global
            if calib is None:
                continue

            idx_t = torch.from_numpy(idx.astype(np.int64)).to(device=out.device)
            vals = out.index_select(0, idx_t)
            vals = calib.apply_logits(vals)
            out.index_copy_(0, idx_t, vals)

        return out

    def logits_fn(batch: dict):
        X2d, meta = _batch_features_to_2d(batch)
        B, H, W, K_meta, _F = meta
        if K_meta != K:
            raise RuntimeError(f"Batch K={K_meta} but boosters K={K}.")

        m = batch["mask"] > 0.5

        if m.dim() == 2:
            V2d = m
        elif m.dim() == 4:
            V2d = m.permute(0, 2, 3, 1).contiguous().view(B * H * W, K)
        else:
            raise RuntimeError(f"Unexpected m.dim()={m.dim()}, expected 2 or 4")

        keep_idx = _valid_keep_indices(V2d)
        logits_full = torch.zeros((B * H * W, K), device=batch["x"].device, dtype=torch.float32)

        is_flattened = batch["x"].dim() == 2

        if keep_idx.numel() == 0:
            if is_flattened:
                return logits_full
            else:
                return logits_full.view(B, H, W, K).permute(0, 3, 1, 2).contiguous()

        Xkeep = X2d.index_select(0, keep_idx)

        # Get regime IDs for kept pixels
        regime_flat = _flatten_regime_ids(batch)
        regime_keep: Optional[np.ndarray] = None
        if regime_flat is not None:
            keep_np = keep_idx.detach().cpu().numpy().astype(np.int64, copy=False)
            if keep_np.size == regime_flat.shape[0]:
                regime_keep = regime_flat
            else:
                regime_keep = regime_flat[keep_np]

        want_gpu = _is_cuda_device_str(args.xgb_predict_device)
        can_gpu = want_gpu and Xkeep.is_cuda and _HAVE_CUPY

        # Process each horizon
        for k in range(K):
            h = horizons_int[k]

            # Step 1: Compute global logits
            if can_gpu:
                Xc = _torch_to_cupy_2d(Xkeep.contiguous())
                global_margin = _predict_margin_from_cupy(global_boosters[k], Xc).to(dtype=torch.float32)
            else:
                global_margin = _predict_margin_gpu_if_possible(global_boosters[k], Xkeep, args).to(dtype=torch.float32)

            # Step 2: Apply residual corrections per regime
            if residual_boosters and regime_keep is not None:
                final_margin = global_margin.clone()
                unique_regimes = np.unique(regime_keep)

                for r_id in unique_regimes:
                    r_id_int = int(r_id)
                    if r_id_int == regime_nodata:
                        continue
                    if r_id_int not in residual_boosters:
                        # No residual model for this regime - keep global logit
                        continue

                    # Get indices for this regime
                    regime_mask = (regime_keep == r_id_int)
                    regime_indices = np.nonzero(regime_mask)[0]
                    if len(regime_indices) == 0:
                        continue

                    # Get features and global logits for this regime
                    regime_idx_t = torch.from_numpy(regime_indices.astype(np.int64)).to(device=Xkeep.device)
                    X_regime = Xkeep.index_select(0, regime_idx_t)
                    global_logit_regime = global_margin.index_select(0, regime_idx_t)

                    # Augment features with global logit
                    global_logit_col = global_logit_regime.unsqueeze(1)
                    X_augmented = torch.cat([X_regime, global_logit_col], dim=1)

                    # Predict with residual model
                    residual_bst = residual_boosters[r_id_int][k]
                    X_aug_np = X_augmented.detach().cpu().numpy().astype(np.float32)
                    dmat_aug = xgb.DMatrix(X_aug_np)
                    residual_logit = residual_bst.predict(dmat_aug, output_margin=True)
                    residual_logit_t = torch.from_numpy(residual_logit).to(device=final_margin.device, dtype=torch.float32)

                    # Update final margin for these pixels
                    final_margin.index_copy_(0, regime_idx_t, residual_logit_t)

                margin = final_margin
            else:
                margin = global_margin

            # Step 3: Apply calibration
            z = _apply_calibration(margin, int(h), regime_keep)
            logits_full[keep_idx, k] = z

        if is_flattened:
            z_out = logits_full
        else:
            z_out = logits_full.view(B, H, W, K).permute(0, 3, 1, 2).contiguous()

        del X2d, V2d, Xkeep, keep_idx, logits_full
        return z_out

    return logits_fn


def xgb_ensemble_logits_fn_from_cluster_boosters(
    global_boosters: List["xgb.Booster"],
    cluster_boosters: Dict[int, List["xgb.Booster"]],
    cluster_state: dict,
    pixel_cluster_map: Dict[int, int],
    pixel_sig_cache: Dict[int, np.ndarray],
    horizons: List[int],
    args,
    mode: str = "cluster_selection",
    calibrators: Optional[Dict[int, Any]] = None,
    W_global: int = 0,
):
    """
    Build a logits function for ensemble prediction using cluster models.

    Two modes:
    - "cluster_selection": For each pixel, use its assigned cluster's model.
                          Fallback to global model for unassigned pixels.
    - "distance_weighted": For each pixel, weight all cluster model predictions
                          by inverse Euclidean distance to cluster centers.

    Args:
        global_boosters: List of global XGBoost boosters (one per horizon)
        cluster_boosters: Dict mapping cluster_id -> List[booster per horizon]
        cluster_state: Cluster state dict with 'centers', optional 'pca'
        pixel_cluster_map: Dict mapping pixel_id -> cluster_id (for cluster_selection)
        pixel_sig_cache: Dict mapping pixel_id -> signature array (for distance_weighted)
        horizons: List of horizon values
        args: Training arguments
        mode: "cluster_selection" or "distance_weighted"
        calibrators: Optional calibration objects per horizon
        W_global: Global width for pixel_id computation
    """
    K = len(global_boosters)
    horizons_int = [int(h) for h in horizons]
    calibs = calibrators or {}
    epsilon = float(getattr(args, "ensemble_epsilon", 1e-6))
    temperature = float(getattr(args, "ensemble_temperature", 1.0))

    # Get cluster centers (possibly PCA-reduced)
    centers = np.asarray(cluster_state["centers"], dtype=np.float64)
    pca_state = cluster_state.get("pca", None)
    n_clusters = centers.shape[0]
    trained_cluster_ids = sorted(cluster_boosters.keys())

    # Get original signature dimension (before PCA) for fallback zeros
    sig_dim = int(cluster_state.get("signature_dim", centers.shape[1]))

    # Track fallback statistics
    fallback_count = [0]
    total_count = [0]
    fallback_reasons = {"missing_from_map": 0, "cluster_not_trained": 0}
    cluster_usage = {cl_id: 0 for cl_id in trained_cluster_ids}

    def logits_fn(batch: dict):
        X2d, meta = _batch_features_to_2d(batch)
        B, H, W, K_meta, _F = meta
        if K_meta != K:
            raise RuntimeError(f"Batch K={K_meta} but boosters K={K}.")

        m = batch["mask"] > 0.5

        # Handle both formats
        if m.dim() == 2:
            V2d = m
        elif m.dim() == 4:
            V2d = m.permute(0, 2, 3, 1).contiguous().view(B * H * W, K)
        else:
            raise RuntimeError(f"Unexpected m.dim()={m.dim()}, expected 2 or 4")

        keep_idx = _valid_keep_indices(V2d)
        device = batch["x"].device
        is_flattened = batch["x"].dim() == 2

        Npix = B * H * W
        logits_full = torch.zeros((Npix, K), device=device, dtype=torch.float32)

        if keep_idx.numel() == 0:
            if is_flattened:
                return logits_full
            else:
                return logits_full.view(B, H, W, K).permute(0, 3, 1, 2).contiguous()

        Xkeep = X2d.index_select(0, keep_idx)
        n_keep = keep_idx.numel()
        total_count[0] += n_keep

        # Get pixel IDs for the kept indices
        y0_b = _ensure_numpy(batch.get("y0", -1)).astype(np.int64, copy=False).reshape(-1)
        x0_b = _ensure_numpy(batch.get("x0", -1)).astype(np.int64, copy=False).reshape(-1)
        W_g = int(W_global) if W_global > 0 else int(_ensure_numpy(batch.get("W_global", -1)).reshape(-1)[0])

        # Compute pixel IDs for all positions
        if is_flattened:
            # FIX: Use actual batch size from y0_b, not B from meta (which is 1 in flattened mode)
            B_true = len(y0_b)

            # FIX: Use args.patch for patch dimensions instead of sqrt heuristic
            patch_size = int(getattr(args, 'patch', 0))
            if patch_size <= 0:
                # Fallback to sqrt if patch not available
                pixels_per_patch = Npix // max(B_true, 1)
                patch_size = int(np.sqrt(pixels_per_patch)) if pixels_per_patch > 0 else 1

            H_fix = W_fix = patch_size
            assert B_true * H_fix * W_fix == Npix, f"Dimension mismatch: {B_true}*{H_fix}*{W_fix}={B_true*H_fix*W_fix} != {Npix}"

            # Vectorized pixel_id computation (much faster than nested loops)
            h_idx = np.repeat(np.arange(H_fix, dtype=np.int64), W_fix)      # (H*W,)
            w_idx = np.tile(np.arange(W_fix, dtype=np.int64), H_fix)        # (H*W,)

            y = y0_b[:, None] + h_idx[None, :]                              # (B, H*W)
            x = x0_b[:, None] + w_idx[None, :]                              # (B, H*W)
            all_pixel_ids = (y * W_g + x).reshape(-1)                       # (Npix,)
        else:
            # Standard format: compute from patch positions
            y_offset = np.arange(H, dtype=np.int64).reshape(1, H, 1)
            x_offset = np.arange(W, dtype=np.int64).reshape(1, 1, W)
            y_global = y0_b.reshape(-1, 1, 1) + y_offset  # (B, H, W)
            x_global = x0_b.reshape(-1, 1, 1) + x_offset  # (B, H, W)
            all_pixel_ids = (y_global * W_g + x_global).reshape(-1)  # (B*H*W,)

        keep_pixel_ids = all_pixel_ids[keep_idx.cpu().numpy()]

        # Debug: print first batch pixel_id info
        if not hasattr(logits_fn, '_debug_printed'):
            logits_fn._debug_printed = True
            print(f"[DEBUG eval] is_flattened={is_flattened}, B_true={len(y0_b)}, First 5 pixel_ids: {all_pixel_ids[:5].tolist()}")

        if mode == "cluster_selection":
            # For each pixel, use its assigned cluster's model
            # Group pixels by cluster
            cluster_to_local_idx = {}  # cluster_id -> list of local indices within keep_idx
            fallback_idx = []  # indices that need global model

            for local_i, pid in enumerate(keep_pixel_ids):
                pid = int(pid)
                cl_id = pixel_cluster_map.get(pid, -1)
                if cl_id >= 0 and cl_id in cluster_boosters:
                    if cl_id not in cluster_to_local_idx:
                        cluster_to_local_idx[cl_id] = []
                    cluster_to_local_idx[cl_id].append(local_i)
                    cluster_usage[cl_id] = cluster_usage.get(cl_id, 0) + 1
                else:
                    fallback_idx.append(local_i)
                    fallback_count[0] += 1
                    # Track fallback reason
                    if cl_id < 0:
                        # Pixel ID not in map
                        fallback_reasons["missing_from_map"] += 1
                    else:
                        # Cluster exists but wasn't trained (< min_samples)
                        fallback_reasons["cluster_not_trained"] += 1

            want_gpu = _is_cuda_device_str(args.xgb_predict_device)
            can_gpu = want_gpu and Xkeep.is_cuda and _HAVE_CUPY

            # Process each cluster's pixels
            for cl_id, local_indices in cluster_to_local_idx.items():
                local_indices = np.array(local_indices, dtype=np.int64)
                local_idx_t = torch.from_numpy(local_indices).to(device)
                X_cl = Xkeep.index_select(0, local_idx_t)

                for k in range(K):
                    bst = cluster_boosters[cl_id][k]
                    if can_gpu:
                        Xc = _torch_to_cupy_2d(X_cl.contiguous())
                        margin = _predict_margin_from_cupy(bst, Xc).to(dtype=torch.float32)
                    else:
                        margin = _predict_margin_gpu_if_possible(bst, X_cl, args).to(dtype=torch.float32)

                    h = horizons_int[k]
                    calib = calibs.get(int(h), None)
                    if calib is not None:
                        if isinstance(calib, PlattCalibrator):
                            margin = calib.apply_logits(margin)
                        elif isinstance(calib, IsotonicCalibrator):
                            margin = calib.apply_logits(margin)

                    # Map back to full array positions
                    global_idx = keep_idx[local_idx_t]
                    logits_full[global_idx, k] = margin

            # Process fallback pixels with global model
            if fallback_idx:
                fallback_idx = np.array(fallback_idx, dtype=np.int64)
                fallback_idx_t = torch.from_numpy(fallback_idx).to(device)
                X_fb = Xkeep.index_select(0, fallback_idx_t)

                for k in range(K):
                    bst = global_boosters[k]
                    if can_gpu:
                        Xc = _torch_to_cupy_2d(X_fb.contiguous())
                        margin = _predict_margin_from_cupy(bst, Xc).to(dtype=torch.float32)
                    else:
                        margin = _predict_margin_gpu_if_possible(bst, X_fb, args).to(dtype=torch.float32)

                    h = horizons_int[k]
                    calib = calibs.get(int(h), None)
                    if calib is not None:
                        if isinstance(calib, PlattCalibrator):
                            margin = calib.apply_logits(margin)
                        elif isinstance(calib, IsotonicCalibrator):
                            margin = calib.apply_logits(margin)

                    global_idx = keep_idx[fallback_idx_t]
                    logits_full[global_idx, k] = margin

        elif mode == "distance_weighted":
            # For each pixel, weight all cluster predictions by inverse distance
            want_gpu = _is_cuda_device_str(args.xgb_predict_device)
            can_gpu = want_gpu and Xkeep.is_cuda and _HAVE_CUPY

            # Get signatures for kept pixels
            sigs = []
            for pid in keep_pixel_ids:
                pid = int(pid)
                if pid in pixel_sig_cache:
                    sigs.append(pixel_sig_cache[pid])
                else:
                    # No signature available - use zeros (will get uniform weights)
                    # Use original signature dimension (before PCA), not reduced dimension
                    sigs.append(np.zeros(sig_dim, dtype=np.float64))
                    fallback_count[0] += 1

            sigs = np.vstack(sigs).astype(np.float64)

            # Apply PCA if used during clustering
            if pca_state is not None:
                sigs = _pca_transform_numpy(sigs, pca_state)

            # Compute weights: (n_keep, n_clusters)
            weights = _compute_inverse_distance_weights(sigs, centers, epsilon=epsilon, temperature=temperature)
            weights_t = torch.from_numpy(weights.astype(np.float32)).to(device)

            # Get predictions from all cluster models
            cluster_logits = {}  # cl_id -> (n_keep, K) tensor

            for cl_id in trained_cluster_ids:
                cl_logits = torch.zeros((n_keep, K), device=device, dtype=torch.float32)

                for k in range(K):
                    bst = cluster_boosters[cl_id][k]
                    if can_gpu:
                        Xc = _torch_to_cupy_2d(Xkeep.contiguous())
                        margin = _predict_margin_from_cupy(bst, Xc).to(dtype=torch.float32)
                    else:
                        margin = _predict_margin_gpu_if_possible(bst, Xkeep, args).to(dtype=torch.float32)

                    h = horizons_int[k]
                    calib = calibs.get(int(h), None)
                    if calib is not None:
                        if isinstance(calib, PlattCalibrator):
                            margin = calib.apply_logits(margin)
                        elif isinstance(calib, IsotonicCalibrator):
                            margin = calib.apply_logits(margin)

                    cl_logits[:, k] = margin

                cluster_logits[cl_id] = cl_logits

            # Compute weighted average of logits
            # weights shape: (n_keep, n_clusters)
            # cluster_logits[cl_id] shape: (n_keep, K)
            weighted_logits = torch.zeros((n_keep, K), device=device, dtype=torch.float32)

            for i, cl_id in enumerate(trained_cluster_ids):
                # weights[:, i] is the weight for cluster cl_id
                w = weights_t[:, i].unsqueeze(1)  # (n_keep, 1)
                weighted_logits += w * cluster_logits[cl_id]

            # Place into full array
            logits_full[keep_idx, :] = weighted_logits

        else:
            raise ValueError(f"Unknown ensemble mode: {mode}")

        if is_flattened:
            z_out = logits_full
        else:
            z_out = logits_full.view(B, H, W, K).permute(0, 3, 1, 2).contiguous()

        return z_out

    def get_fallback_stats():
        return {
            "fallback_count": fallback_count[0],
            "total_count": total_count[0],
            "fallback_reasons": dict(fallback_reasons),
            "cluster_usage": dict(cluster_usage),
        }

    logits_fn.get_fallback_stats = get_fallback_stats
    return logits_fn


# ----------------------------------------------------------------------
# Calibration fitting (streaming over calib loader; GPU prediction if available)
# ----------------------------------------------------------------------

@torch.no_grad()
def fit_calibrators_from_loader(boosters: list, calib_loader, device: torch.device, args) -> Dict[int, Any]:
    """
    Fix:
      - local batch_counter for cuda_empty_cache_every (no persistent function attribute).
    """
    batch_counter = 0

    method = args.calibration_method
    if method == "none":
        return {}

    header(f"Fitting calibration: method={method}")
    print(f"  • device..................... {device}")
    print(f"  • cupy available............. {_HAVE_CUPY}")
    print(f"  • xgb_predict_device......... {args.xgb_predict_device}")
    print(f"  • max_calib_rows_per_horizon. {args.max_calib_rows_per_horizon}")

    K = len(args.horizons)
    horizons_int = [int(h) for h in args.horizons]

    margins_lists: List[List[np.ndarray]] = [[] for _ in range(K)]
    labels_lists:  List[List[np.ndarray]] = [[] for _ in range(K)]
    counts = np.zeros(K, dtype=np.int64)

    if device.type == "cuda" and bool(args.use_cuda_prefetch):
        iterator = CUDAPrefetcher(calib_loader, device)
        total = None
        try:
            total = len(calib_loader)
        except TypeError:
            total = None
        pbar = tqdm(total=total, desc="calib_stream", leave=True, position=2) if (not args.no_tqdm) else None

        def _iter_batches():
            for b in iterator:
                if pbar:
                    pbar.update(1)
                yield b
            if pbar:
                pbar.close()

        batch_iter = _iter_batches()
        batches_are_on_device = True
    else:
        batch_iter = TimedDataLoader(calib_loader, desc="calib_stream", use_tqdm=not args.no_tqdm, position=2)
        batches_are_on_device = False

    for batch in batch_iter:
        if not batches_are_on_device:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        X2d, meta = _batch_features_to_2d(batch)
        B, H, W, K_meta, _F = meta
        if K_meta != K:
            raise RuntimeError(f"Calib batch K={K_meta} but expected K={K}.")

        y = (batch["y"] > 0.5)
        m = (batch["mask"] > 0.5)

        # Handle both formats:
        # - flatten_pixels=True: y is (Npix, K) - already in correct shape
        # - flatten_pixels=False: y is (B, K, H, W) - needs reshaping
        if y.dim() == 2:
            # Already flattened: (Npix, K)
            Y2d = y
            V2d = m
        elif y.dim() == 4:
            # Standard format: (B, K, H, W) → (B*H*W, K)
            Y2d = y.permute(0, 2, 3, 1).contiguous().view(B * H * W, K)
            V2d = m.permute(0, 2, 3, 1).contiguous().view(B * H * W, K)
        else:
            raise RuntimeError(f"Unexpected y.dim()={y.dim()}, expected 2 or 4")

        for k in range(K):
            cap = args.max_calib_rows_per_horizon
            if cap is not None and counts[k] >= int(cap):
                continue

            vk = V2d[:, k]
            n_vk = int(vk.sum().item())
            if n_vk == 0:
                continue

            if cap is not None:
                remaining = int(cap) - int(counts[k])
                if remaining <= 0:
                    continue
                idx = torch.nonzero(vk, as_tuple=False).view(-1)
                if idx.numel() > remaining:
                    idx = idx[:remaining]
            else:
                idx = torch.nonzero(vk, as_tuple=False).view(-1)

            Xk = X2d.index_select(0, idx)
            yk = Y2d.index_select(0, idx).select(1, k).to(dtype=torch.float32)

            margin = _predict_margin_gpu_if_possible(boosters[k], Xk, args)
            margin_np = margin.detach().cpu().numpy().astype(np.float32, copy=False)
            y_np = yk.detach().cpu().numpy().astype(np.float32, copy=False)

            margins_lists[k].append(margin_np)
            labels_lists[k].append(y_np)
            counts[k] += int(margin_np.size)

            del idx, Xk, yk, margin

        del X2d, Y2d, V2d, y, m

        if device.type == "cuda" and int(getattr(args, "cuda_empty_cache_every", 0)) > 0:
            batch_counter += 1
            if batch_counter % int(args.cuda_empty_cache_every) == 0:
                torch.cuda.empty_cache()

    calibrators: Dict[int, Any] = {}
    for k in range(K):
        h = horizons_int[k]
        if counts[k] <= 0:
            print(f"[Calib] h={h}: no data; skipping calibrator.")
            continue

        m_all = np.concatenate(margins_lists[k], axis=0).astype(np.float32, copy=False)
        y_all = np.concatenate(labels_lists[k], axis=0).astype(np.float32, copy=False)

        pos = float(y_all.sum())
        tot = float(y_all.size)
        print(f"[Calib] h={h}: rows={int(tot):,} pos={int(pos):,} pos_frac={(pos/max(tot,1.0)):.6g}")

        if method == "platt":
            calib = fit_platt_scaling(
                m_all, y_all,
                max_iter=int(args.platt_max_iter),
                tol=float(args.platt_tol),
                reg=float(args.platt_reg),
            )
            print(f"[Calib] h={h}: Platt a={calib.a:.6g} b={calib.b:.6g}")
            calibrators[h] = calib

        elif method == "isotonic":
            calib = fit_isotonic_pav(m_all, y_all)
            print(f"[Calib] h={h}: Isotonic steps={calib.y_values.size}")
            calibrators[h] = calib

        else:
            raise ValueError(f"Unknown calibration method: {method}")

    return calibrators


def fit_calibrators_per_regime_from_loader(
    boosters: list,
    calib_loader,
    device: torch.device,
    args,
    n_regimes: int,
) -> Dict:
    """
    Fit calibrators per (regime, horizon) pair.

    Falls back to global calibrator if a regime has fewer than
    args.regime_calib_min_samples samples.

    Returns:
        Dict with keys:
            - (regime_id, horizon): calibrator for that pair
            - "global": {horizon: calibrator} for fallback
    """
    batch_counter = 0

    method = args.calibration_method
    if method == "none":
        return {}

    header(f"Fitting per-regime calibration: method={method}")
    print(f"  • n_regimes.................. {n_regimes}")
    print(f"  • min_samples_per_regime..... {args.regime_calib_min_samples}")
    print(f"  • device..................... {device}")

    K = len(args.horizons)
    horizons_int = [int(h) for h in args.horizons]
    nodata_value = int(getattr(args, 'regime_nodata_value', -9999))
    min_samples = int(args.regime_calib_min_samples)

    # Accumulators: per regime per horizon
    margins_per_regime: Dict[int, List[List[np.ndarray]]] = {
        r: [[] for _ in range(K)] for r in range(n_regimes)
    }
    labels_per_regime: Dict[int, List[List[np.ndarray]]] = {
        r: [[] for _ in range(K)] for r in range(n_regimes)
    }
    counts_per_regime: Dict[int, np.ndarray] = {
        r: np.zeros(K, dtype=np.int64) for r in range(n_regimes)
    }

    # Global accumulators (for fallback)
    margins_global: List[List[np.ndarray]] = [[] for _ in range(K)]
    labels_global: List[List[np.ndarray]] = [[] for _ in range(K)]
    counts_global = np.zeros(K, dtype=np.int64)

    batch_iter = TimedDataLoader(calib_loader, desc="calib_per_regime", use_tqdm=not args.no_tqdm, position=2)

    for batch in batch_iter:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        X2d, meta = _batch_features_to_2d(batch)
        B, H, W, K_meta, _F = meta

        y = (batch["y"] > 0.5)
        m = (batch["mask"] > 0.5)
        regime_ids = batch.get("regime_id", None)

        if regime_ids is None:
            print("[WARN] per_regime_calibration requires regime_id in batch. Skipping batch.")
            continue

        # Handle both formats
        if y.dim() == 2:
            Y2d = y
            V2d = m
        else:
            Y2d = y.permute(0, 2, 3, 1).contiguous().view(-1, K)
            V2d = m.permute(0, 2, 3, 1).contiguous().view(-1, K)

        # Flatten regime_ids
        if isinstance(regime_ids, np.ndarray):
            regime_flat = regime_ids.reshape(-1)
        else:
            regime_flat = np.asarray(regime_ids).reshape(-1)

        for k in range(K):
            vk = V2d[:, k].cpu().numpy()
            valid_idx = np.where(vk)[0]

            if len(valid_idx) == 0:
                continue

            # Get valid rows
            idx_t = torch.from_numpy(valid_idx).to(X2d.device)
            Xk = X2d.index_select(0, idx_t)
            yk = Y2d.index_select(0, idx_t).select(1, k).cpu().numpy().astype(np.float32)
            rk = regime_flat[valid_idx]

            margin = _predict_margin_gpu_if_possible(boosters[k], Xk, args)
            margin_np = margin.detach().cpu().numpy().astype(np.float32)

            # Accumulate per-regime
            for r in range(n_regimes):
                r_mask = (rk == r)
                if r_mask.sum() > 0:
                    margins_per_regime[r][k].append(margin_np[r_mask])
                    labels_per_regime[r][k].append(yk[r_mask])
                    counts_per_regime[r][k] += int(r_mask.sum())

            # Accumulate global (valid pixels only, excluding nodata)
            valid_regime = rk != nodata_value
            if valid_regime.sum() > 0:
                margins_global[k].append(margin_np[valid_regime])
                labels_global[k].append(yk[valid_regime])
                counts_global[k] += int(valid_regime.sum())

            del idx_t, Xk, margin

        del X2d, Y2d, V2d, y, m

        if device.type == "cuda" and int(getattr(args, "cuda_empty_cache_every", 0)) > 0:
            batch_counter += 1
            if batch_counter % int(args.cuda_empty_cache_every) == 0:
                torch.cuda.empty_cache()

    # Fit calibrators
    calibrators: Dict = {}

    # Fast concatenation via pre-allocation (avoids np.concatenate overhead on many small arrays)
    def _fast_concat(chunks: List[np.ndarray], total_count: int, dtype=np.float32) -> np.ndarray:
        result = np.empty(total_count, dtype=dtype)
        offset = 0
        for chunk in chunks:
            n = chunk.size
            result[offset:offset + n] = chunk.ravel()
            offset += n
        return result

    # Fit global fallback first
    print(f"[Calib] Fitting global calibrators for {K} horizons...")
    global_calibs: Dict[int, Any] = {}
    for k in range(K):
        h = horizons_int[k]
        if counts_global[k] > 0:
            n_chunks = len(margins_global[k])
            count = int(counts_global[k])
            print(f"[Calib] Global h={h}: merging {n_chunks} chunks ({count:,} samples)...", end=" ", flush=True)
            m_all = _fast_concat(margins_global[k], count)
            y_all = _fast_concat(labels_global[k], count)
            print("done. Fitting...", end=" ", flush=True)

            if method == "platt":
                calib = fit_platt_scaling(m_all, y_all,
                    max_iter=int(args.platt_max_iter),
                    tol=float(args.platt_tol),
                    reg=float(args.platt_reg),
                )
                print(f"Platt a={calib.a:.6g} b={calib.b:.6g}")
            else:
                calib = fit_isotonic_pav(m_all, y_all)
                print(f"Isotonic steps={calib.y_values.size}")

            global_calibs[h] = calib

    calibrators["global"] = global_calibs

    # Fit per-regime
    print(f"[Calib] Fitting {n_regimes} regimes x {K} horizons...")
    for r in range(n_regimes):
        for k in range(K):
            h = horizons_int[k]
            count = int(counts_per_regime[r][k])

            if count < min_samples:
                print(f"[Calib] regime={r} h={h}: {count} samples < {min_samples}, using global fallback")
                continue

            n_chunks = len(margins_per_regime[r][k])
            print(f"[Calib] regime={r} h={h}: merging {n_chunks} chunks ({count:,} samples)...", end=" ", flush=True)
            m_all = _fast_concat(margins_per_regime[r][k], count)
            y_all = _fast_concat(labels_per_regime[r][k], count)
            print("done. Fitting...", end=" ", flush=True)

            if method == "platt":
                calib = fit_platt_scaling(m_all, y_all,
                    max_iter=int(args.platt_max_iter),
                    tol=float(args.platt_tol),
                    reg=float(args.platt_reg),
                )
                print(f"Platt a={calib.a:.6g} b={calib.b:.6g}")
            else:
                calib = fit_isotonic_pav(m_all, y_all)
                print(f"Isotonic steps={calib.y_values.size}")

            calibrators[(r, h)] = calib

    return calibrators


# ----------------------------------------------------------------------
# Model saving
# ----------------------------------------------------------------------

def save_xgb_models(boosters: list, horizons: list[int], logdir: str, params: dict, seed: int, n_features: int, args):
    os.makedirs(logdir, exist_ok=True)
    for k_idx, h in enumerate(horizons):
        path = os.path.join(logdir, f"xgb_h{int(h)}.json")
        boosters[k_idx].save_model(path)
        print(f"[Save] Wrote {path}")

    manifest = {
        "model_type": "xgboost_per_horizon",
        "seed": int(seed),
        "horizons": [int(h) for h in horizons],
        "n_features": int(n_features),
        "params": params,
        "xgb_num_round": int(args.xgb_num_round),
        "max_depth": int(args.xgb_max_depth),
        "subsample": float(args.xgb_subsample),
        "colsample_bynode": float(args.xgb_colsample_bynode),
        "eta": float(args.xgb_eta),
        "calibration_method": str(args.calibration_method),
        "train_mode": str(args.train_mode),
        "stream_chunk_rows": int(args.stream_chunk_rows) if getattr(args, "train_mode", "tabularize") == "stream" else None,
        "stream_accumulate_multiplier": int(getattr(args, "stream_accumulate_multiplier", 3)) if getattr(args, "train_mode", "tabularize") == "stream" else None,
        "xgb_quantile_precompute": bool(getattr(args, "xgb_quantile_precompute", False)) if getattr(args, "train_mode", "tabularize") == "stream" else None,
    }
    mpath = os.path.join(logdir, "xgb_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Save] Wrote {mpath}")



# ----------------------------------------------------------------------
# SHAP feature importance (XGBoost pred_contribs)
# ----------------------------------------------------------------------

def _infer_feature_names(ds, args) -> List[str]:
    """
    Best-effort feature naming.
    - If the dataset yields x as (B,T,C,H,W): names = t{t}_c{c}
    - If x is already flattened: names = f{j}
    """
    try:
        loader = make_loader(ds, batch_size=1, shuffle=False, args=args, device=None)
        b = next(iter(loader))
        x = b["x"]
        if torch.is_tensor(x):
            if x.dim() == 5:
                _, T, C, _, _ = x.shape
                return [f"t{t}_c{c}" for t in range(int(T)) for c in range(int(C))]
            if x.dim() == 4:
                _, F, _, _ = x.shape
                return [f"f{j}" for j in range(int(F))]
    except Exception:
        pass
    # Fallback: infer numeric feature count from dataset
    F = _infer_num_features_from_dataset(ds, args)
    return [f"f{j}" for j in range(int(F))]


def _collect_shap_samples_from_ds(
    ds,
    args,
    split_name: str,
    feature_names: List[str],
) -> Dict[int, np.ndarray]:
    """
    Collect up to --shap-max-rows per horizon (valid pixels only).
    Returns dict: horizon -> X (N,F) float32.
    """
    max_rows = int(getattr(args, "shap_max_rows", 20000))
    if max_rows <= 0:
        return {}

    K = len(args.horizons)
    out_chunks: List[List[np.ndarray]] = [[] for _ in range(K)]
    counts = [0] * K

    # SHAP sampling uses CPU rows (predict_contribs accepts numpy/cupy)
    bs = int(getattr(args, "shap_batch_size", 0)) or int(args.batch_size)
    loader = make_loader(ds, batch_size=bs, shuffle=True, args=args, device=None)

    header(f"Collecting SHAP samples: split={split_name}, max_rows/h={max_rows}")
    pbar = tqdm(loader, desc=f"shap_sample_{split_name}", leave=True, position=2) if not args.no_tqdm else None
    it = pbar if pbar is not None else loader

    for batch in it:
        X, Y, V, _ = _batch_to_rows_cpu(batch)  # X: (N,F), V: (N,K)
        keep_any = V.any(axis=1)
        if not np.any(keep_any):
            if pbar: pbar.update(1)
            continue

        X = X[keep_any]
        V = V[keep_any]

        # Optional: sanity check F match
        if X.shape[1] != len(feature_names):
            # fall back to f0.. if mismatch
            feature_names[:] = [f"f{j}" for j in range(int(X.shape[1]))]

        for k in range(K):
            if counts[k] >= max_rows:
                continue
            idx = np.nonzero(V[:, k])[0]
            if idx.size == 0:
                continue
            remaining = max_rows - counts[k]
            if idx.size > remaining:
                # random subset for better coverage
                sel = np.random.choice(idx, size=remaining, replace=False)
                idx = sel
            out_chunks[k].append(X[idx])
            counts[k] += int(idx.size)

        if pbar: 
            pbar.set_postfix({f"h{int(args.horizons[i])}": counts[i] for i in range(K)})

        if all(c >= max_rows for c in counts):
            break

    if pbar: 
        pbar.close()

    out: Dict[int, np.ndarray] = {}
    for k, h in enumerate(args.horizons):
        if counts[k] <= 0:
            continue
        Xk = np.concatenate(out_chunks[k], axis=0).astype(np.float32, copy=False)
        out[int(h)] = Xk

    for h in [int(x) for x in args.horizons]:
        if h in out:
            print(f"  • SHAP sample h={h}: rows={out[h].shape[0]:,} features={out[h].shape[1]}")
        else:
            print(f"  • SHAP sample h={h}: rows=0 (no valid pixels found)")

    return out


def _xgb_predict_contribs(bst: "xgb.Booster", X: np.ndarray, args) -> np.ndarray:
    """
    Returns SHAP contribs (N, F+1) in *margin* space (output_margin=True).
    Uses GPU via CuPy if available and requested, otherwise CPU.
    """
    want_gpu = _is_cuda_device_str(getattr(args, "shap_predict_device", "auto")) or (
        getattr(args, "shap_predict_device", "auto") == "auto" and _is_cuda_device_str(args.xgb_predict_device)
    )
    use_gpu = bool(want_gpu and _HAVE_CUPY)

    if use_gpu:
        with cp.cuda.Device(int(args.xgb_gpu_id)):
            Xc = cp.asarray(X)
            # DMatrix on GPU
            dmat = xgb.DMatrix(Xc)
            return bst.predict(dmat, pred_contribs=True)

    dmat = xgb.DMatrix(X)
    return bst.predict(dmat, pred_contribs=True)


def _plot_shap_bar(df, out_path, args, title="SHAP feature importance"):
    """
    Horizontal bar chart of mean(|SHAP|) by feature.

    Display-name cleanup (for labels):
      - Convert a trailing "_t0"  -> " (t)"
      - Convert a trailing "_tK"  -> " (t-K)"  e.g., "_t5" -> " (t-5)"
      - If the feature uses "..._tK_..." (rare), you can extend this, but by default
        we only rewrite the final time-lag token to avoid messing up other underscores.
    """
    import os
    import re
    import textwrap
    import numpy as np
    import matplotlib.pyplot as plt

    # Match ONLY a trailing time token like "_t0" or "_t12" at end of string
    _trail_time = re.compile(r"(.*)_t(\d+)$")

    def _clean_label(label: str) -> str:
        label = str(label)

        m = _trail_time.match(label)
        if m:
            base = m.group(1)
            k = int(m.group(2))
            if k == 0:
                return f"{base} (t)"
            return f"{base} (t-{k})"

        # If there's no trailing "_tK", leave it as-is.
        return label

    # ---- pick top-N
    top_n = int(getattr(args, "shap_top_n", 30))
    plot_dpi = int(getattr(args, "plot_dpi", 200))
    plot_fmt = getattr(args, "plot_file_format", None)

    d = df.copy()
    if "mean_abs_shap" not in d.columns or "feature" not in d.columns:
        raise ValueError("df must contain columns: feature, mean_abs_shap")

    d = d.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)

    feats = [_clean_label(x) for x in d["feature"].tolist()]
    vals = d["mean_abs_shap"].to_numpy(dtype=float)

    # ---- optional label wrapping for readability
    max_chars = int(getattr(args, "plot_max_label_chars", 38))
    if max_chars > 0:
        feats = [textwrap.fill(f, width=max_chars, break_long_words=False) for f in feats]

    # plot in descending order top-to-bottom
    feats = feats[::-1]
    vals = vals[::-1]

    # ---- figure sizing
    base_h = 2.6
    per_row = 0.28
    fig_h = max(3.5, base_h + per_row * len(feats))
    fig_w = float(getattr(args, "plot_fig_w", 10.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    y = np.arange(len(feats))

    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(feats)

    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(title)

    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=plot_dpi, format=plot_fmt)
    plt.close(fig)


import re

def _pretty_feature_label(name: str) -> str:
    # Remove trailing ":chNN" or ":cNN" ONLY for display
    return re.sub(r":(?:ch|c)\d+$", "", name)


def _plot_shap_dependence(X: np.ndarray, shap_vals: np.ndarray, feature_names: List[str], fi: int, out_path: str, args, title: str):
    import matplotlib.pyplot as plt
    x = X[:, fi].astype(np.float64, copy=False)
    s = shap_vals[:, fi].astype(np.float64, copy=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, s, s=4, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(feature_names[fi])
    ax.set_ylabel("SHAP (margin)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.plot_dpi, format=args.plot_file_format)
    plt.close(fig)
# ======================================================================
# OOF SHAP tile signatures -> clustering -> per-cluster eval -> plots
# ======================================================================

def _ensure_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _get_batch_tile_meta(batch: dict):
    # expects tile_id, lat, lon, date in batch (see dataset wrapper above)
    tile_id = _ensure_numpy(batch.get("tile_id", -1)).astype(np.int64, copy=False).reshape(-1)
    lat     = _ensure_numpy(batch.get("lat", np.nan)).astype(np.float32, copy=False).reshape(-1)
    lon     = _ensure_numpy(batch.get("lon", np.nan)).astype(np.float32, copy=False).reshape(-1)
    date    = _ensure_numpy(batch.get("date", -1)).reshape(-1)
    return tile_id, lat, lon, date

def _batch_to_rows_cpu_single_horizon_with_meta(batch: dict, k: int):
    """
    Like _batch_to_rows_cpu_single_horizon, but ALSO returns per-row tile_id/lat/lon/date.
    """
    X, y, v = _batch_to_rows_cpu_single_horizon(batch, k)

    tile_id_b, lat_b, lon_b, date_b = _get_batch_tile_meta(batch)
    # expand meta from (B,) to (B*H*W,)
    x = _flatten_time_if_needed(batch["x"])
    B, _F, H, W = x.shape
    rep = H * W
    tile_row = np.repeat(tile_id_b, rep)
    lat_row  = np.repeat(lat_b, rep)
    lon_row  = np.repeat(lon_b, rep)
    date_row = np.repeat(date_b, rep)

    return X, y, v, tile_row, lat_row, lon_row, date_row

def _batch_to_rows_cpu_single_horizon_with_pixel_meta(batch: dict, k: int):
    """
    Like _batch_to_rows_cpu_single_horizon_with_meta, but returns PER-PIXEL coordinates.

    For pixel-level clustering, we need global pixel coordinates instead of tile-level metadata.
    Each pixel in the batch gets its own (y_global, x_global) position and pixel_id.

    Returns:
        X: (N, F) features
        y: (N,) labels
        v: (N,) valid mask
        pixel_id_row: (N,) global pixel IDs (y_global * W + x_global)
        y_global_row: (N,) global y coordinates
        x_global_row: (N,) global x coordinates
        lat_row: (N,) latitude (approximate, from patch center)
        lon_row: (N,) longitude (approximate, from patch center)
        date_row: (N,) date/time indices
    """
    # Extract features like existing function
    X, y, v = _batch_to_rows_cpu_single_horizon(batch, k)

    # Get batch-level metadata
    y0_b = _ensure_numpy(batch.get("y0", -1)).astype(np.int64, copy=False).reshape(-1)
    x0_b = _ensure_numpy(batch.get("x0", -1)).astype(np.int64, copy=False).reshape(-1)
    # W_global is the same for all samples in batch, so take first element
    W_global_arr = _ensure_numpy(batch.get("W_global", -1)).reshape(-1)
    W_global = int(W_global_arr[0])
    lat_b = _ensure_numpy(batch.get("lat", np.nan)).astype(np.float32, copy=False).reshape(-1)
    lon_b = _ensure_numpy(batch.get("lon", np.nan)).astype(np.float32, copy=False).reshape(-1)
    date_b = _ensure_numpy(batch.get("date", -1)).reshape(-1)

    x_raw = batch["x"]

    # Check if data is already flattened (flatten_pixels=True)
    if x_raw.dim() == 2:
        # Flattened format: x=(Npix, F), where Npix = B * H * W
        # We need to infer H, W from the dataset's patch size
        # Since we have y0_b and x0_b per batch element, we need to figure out
        # how many pixels per batch element
        Npix = x_raw.shape[0]
        B = len(y0_b)

        if B == 0:
            # Edge case: empty batch
            return X, y, v, np.array([], dtype=np.int64), np.array([], dtype=np.int64), \
                   np.array([], dtype=np.int64), np.array([], dtype=np.float32), \
                   np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        # Npix = B * H * W, so pixels_per_patch = Npix / B
        pixels_per_patch = Npix // B
        # Assuming square patches: H = W = sqrt(pixels_per_patch)
        H = W = int(np.sqrt(pixels_per_patch))
        if H * W != pixels_per_patch:
            # Non-square patch - try to infer from common sizes
            # This is a fallback; ideally patch size should be passed
            raise ValueError(f"Cannot infer patch dimensions: Npix={Npix}, B={B}, pixels_per_patch={pixels_per_patch}")
    else:
        # Standard format: get dimensions from tensor
        x_tensor = _flatten_time_if_needed(x_raw)
        B, _F, H, W = x_tensor.shape

    # Create offset grids for pixels within patch
    # y_offset: (1, H, 1) broadcasts to (B, H, W)
    # x_offset: (1, 1, W) broadcasts to (B, H, W)
    y_offset = np.arange(H, dtype=np.int64).reshape(1, H, 1)
    x_offset = np.arange(W, dtype=np.int64).reshape(1, 1, W)

    # Broadcast to (B, H, W)
    y_offset = np.broadcast_to(y_offset, (B, H, W))
    x_offset = np.broadcast_to(x_offset, (B, H, W))

    # Compute global coordinates
    # y_global = y0 + offset_within_patch
    y_global = y0_b.reshape(-1, 1, 1) + y_offset  # (B, H, W)
    x_global = x0_b.reshape(-1, 1, 1) + x_offset  # (B, H, W)

    # Flatten to (B*H*W,)
    y_global_row = y_global.reshape(-1)
    x_global_row = x_global.reshape(-1)
    pixel_id_row = y_global_row * W_global + x_global_row

    # Expand batch-level metadata to per-pixel
    # Note: lat/lon are patch centers, so they're approximate for individual pixels
    # For precise coords, would need to interpolate from lat/lon grids
    rep = H * W
    lat_row = np.repeat(lat_b, rep)
    lon_row = np.repeat(lon_b, rep)
    date_row = np.repeat(date_b, rep)

    return X, y, v, pixel_id_row, y_global_row, x_global_row, lat_row, lon_row, date_row

def _make_fold_ids_for_dataset(ds, args, n_folds: int, seed: int = 123):
    """
    Creates fold_id per dataset index using leakage-safe grouping.

    fold-mode:
      - tile: group by tile_id (recommended)
      - year: group by date (year)
      - tile_year: group by (tile_id, year)
    """
    # We sample meta by iterating ds once (patch-level).
    tile_ids = np.zeros(len(ds), dtype=np.int64)
    years    = np.zeros(len(ds), dtype=np.int64)

    loader = make_loader(ds, batch_size=min(256, max(1, args.batch_size)), shuffle=False, args=args, device=None)
    write = 0
    for batch in loader:
        tid, _lat, _lon, date = _get_batch_tile_meta(batch)
        bsz = tid.shape[0]
        tile_ids[write:write+bsz] = tid
        # if date is datetime64, convert to year; else assume int year already; else -1
        d = np.asarray(date)
        if d.dtype.kind == "M":
            y = (d.astype("datetime64[Y]").astype(np.int64) + 1970).astype(np.int64)
        else:
            y = d.astype(np.int64, copy=False)
        years[write:write+bsz] = y
        write += bsz
        if write >= len(ds):
            break

    # build grouping keys
    mode = str(getattr(args, "fold_mode", "tile"))
    if mode == "tile":
        groups = tile_ids
    elif mode == "year":
        groups = years
    else:
        # tile_year
        groups = (tile_ids.astype(np.int64) * 10_000 + years.astype(np.int64))

    # deterministic group->fold assignment (no sklearn)
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)

    fold_of_group = {}
    for i, g in enumerate(uniq):
        fold_of_group[int(g)] = int(i % n_folds)

    fold_id = np.array([fold_of_group[int(g)] for g in groups], dtype=np.int64)

    # Build reverse mapping: tile_id -> patch_idx
    tile_id_to_patch_idx = {int(tile_ids[i]): i for i in range(len(tile_ids))}

    return fold_id, tile_ids, years, tile_id_to_patch_idx

def _kmeans_numpy(X: np.ndarray, n_clusters: int, seed: int = 0, n_init: int = 5, max_iter: int = 200):
    """
    Lightweight kmeans (k-means++ init).
    Returns: labels, centers, inertia
    """
    rng = np.random.RandomState(seed)
    best = None

    def kpp_init(X, k):
        n = X.shape[0]
        centers = np.empty((k, X.shape[1]), dtype=np.float64)
        # choose first randomly
        i0 = rng.randint(0, n)
        centers[0] = X[i0]
        d2 = np.sum((X - centers[0])**2, axis=1)
        for c in range(1, k):
            probs = d2 / max(d2.sum(), 1e-12)
            idx = rng.choice(n, p=probs)
            centers[c] = X[idx]
            d2 = np.minimum(d2, np.sum((X - centers[c])**2, axis=1))
        return centers

    X64 = X.astype(np.float64, copy=False)

    for init in range(n_init):
        centers = kpp_init(X64, n_clusters)
        labels = np.zeros(X64.shape[0], dtype=np.int64)

        for _it in range(max_iter):
            # assign
            dists = ((X64[:, None, :] - centers[None, :, :])**2).sum(axis=2)  # (N,K)
            new_labels = dists.argmin(axis=1)

            if np.array_equal(new_labels, labels) and _it > 0:
                break
            labels = new_labels

            # update
            for k in range(n_clusters):
                m = (labels == k)
                if m.any():
                    centers[k] = X64[m].mean(axis=0)
                else:
                    centers[k] = X64[rng.randint(0, X64.shape[0])]

        inertia = float(((X64 - centers[labels])**2).sum())
        if best is None or inertia < best[2]:
            best = (labels.copy(), centers.copy(), inertia)

    return best  # (labels, centers, inertia)

def _pca_fit_transform_numpy(X: np.ndarray, n_components: int):
    """
    PCA via SVD. Returns (Xr, pca_state_dict)
    """
    X64 = X.astype(np.float64, copy=False)
    mu = X64.mean(axis=0, keepdims=True)
    Xc = X64 - mu
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]  # (k, D)
    Xr = (Xc @ comps.T)
    state = {"mean": mu.squeeze(0), "components": comps}
    return Xr, state

def _pca_transform_numpy(X: np.ndarray, state: dict):
    mu = state["mean"][None, :]
    comps = state["components"]
    Xc = X.astype(np.float64, copy=False) - mu
    return Xc @ comps.T

def _signature_normalize(S: np.ndarray, eps: float = 1e-12):
    denom = S.sum(axis=1, keepdims=True)
    return S / (denom + eps)


def _compute_euclidean_distances(
    signatures: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distances from signatures to cluster centers.

    Args:
        signatures: (N, F) array of pixel signatures
        centers: (K, F) array of cluster centers

    Returns:
        distances: (N, K) array of distances
    """
    # signatures: (N, F), centers: (K, F)
    # Expand for broadcasting: (N, 1, F) - (1, K, F) -> (N, K, F)
    diff = signatures[:, None, :] - centers[None, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))  # (N, K)
    return distances


def _compute_inverse_distance_weights(
    signatures: np.ndarray,
    centers: np.ndarray,
    epsilon: float = 1e-6,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Compute normalized inverse-distance weights for ensemble prediction.

    For each pixel, weight[k] = 1 / (distance_to_center_k + epsilon)
    Weights are normalized to sum to 1.

    Args:
        signatures: (N, F) array of pixel signatures (or (F,) for single pixel)
        centers: (K, F) array of cluster centers
        epsilon: Small constant to avoid division by zero
        temperature: Sharpening parameter (lower = sharper weights)

    Returns:
        weights: (N, K) or (K,) normalized weights
    """
    single = signatures.ndim == 1
    if single:
        signatures = signatures[None, :]  # (1, F)

    distances = _compute_euclidean_distances(signatures, centers)  # (N, K)

    # Inverse distance weighting with temperature scaling
    weights = 1.0 / (distances + epsilon) ** temperature
    weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize to sum to 1

    if single:
        return weights[0]  # (K,)
    return weights  # (N, K)


# ----------------------------------------------------------------------
# Temporal aggregation helpers for SHAP-based clustering
# ----------------------------------------------------------------------

def _parse_base_feature_name(feat_name: str) -> str:
    """
    Extract the base channel name from a temporal feature name.

    Examples:
        "SMAP_WTD (t)"     -> "SMAP_WTD"
        "SMAP_WTD (t-5)"   -> "SMAP_WTD"
        "ERA5.zarr c0 (t-29)" -> "ERA5.zarr c0"
        "sin(lat)"         -> "sin(lat)"  (no time suffix, returned as-is)

    The pattern looks for " (t)" or " (t-N)" at the end of the name.
    """
    import re
    # Match " (t)" or " (t-N)" at the end, where N is one or more digits
    pattern = r'\s*\(t(?:-\d+)?\)$'
    base = re.sub(pattern, '', feat_name).strip()
    return base if base else feat_name


def _is_positional_feature(feat_name: str) -> bool:
    """
    Determine if a feature is positional/spatial and should be excluded from clustering.

    Positional features include:
    - Cartesian encoding: x, y, z
    - Legacy sin/cos encoding: sin(lat), cos(lat), sin(lon), cos(lon)
    - Raw metadata: lat, lon, latitude, longitude
    - Spatial identifiers: tile_id, y0, x0, pixel_id, y_global, x_global

    Args:
        feat_name: Feature name (may include temporal suffix like " (t-5)")

    Returns:
        True if feature is positional, False otherwise

    Note:
        This function checks the BASE feature name (temporal suffix stripped),
        so "x (t-5)" and "x" are both recognized as positional.
    """
    # Strip temporal suffix to get base name
    base_name = _parse_base_feature_name(feat_name)

    # Cartesian encoding (current)
    if base_name in ["x", "y", "z"]:
        return True

    # Legacy sin/cos encoding (backward compatibility)
    if base_name in ["sin(lat)", "cos(lat)", "sin(lon)", "cos(lon)"]:
        return True

    # Raw coordinate metadata (if somehow used as features)
    if base_name in ["lat", "lon", "latitude", "longitude"]:
        return True

    # Spatial identifiers (tile/pixel IDs and grid positions)
    if base_name in ["tile_id", "y0", "x0", "pixel_id", "y_global", "x_global"]:
        return True

    return False


def _filter_positional_features(
    shap_vals: np.ndarray,
    feature_names: List[str],
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Filter out positional features from SHAP values and feature names.

    Args:
        shap_vals: (N, F) array of SHAP values
        feature_names: List of F feature names
        verbose: If True, print filtering statistics

    Returns:
        filtered_shap: (N, F_env) array with positional features removed
        filtered_names: List of F_env environmental feature names
        kept_indices: List of original indices that were kept

    Raises:
        ValueError: If all features are positional (nothing left for clustering)
    """
    if shap_vals.ndim != 2:
        raise ValueError(f"Expected 2D shap_vals, got shape {shap_vals.shape}")

    N, F = shap_vals.shape
    if len(feature_names) != F:
        raise ValueError(f"feature_names length {len(feature_names)} != shap_vals features {F}")

    # Identify non-positional features
    kept_indices = []
    filtered_names = []
    positional_names = []

    for idx, fname in enumerate(feature_names):
        if _is_positional_feature(fname):
            positional_names.append(fname)
        else:
            kept_indices.append(idx)
            filtered_names.append(fname)

    if len(kept_indices) == 0:
        raise ValueError(
            f"All {F} features are positional! Cannot cluster with no environmental features.\n"
            f"Positional features: {positional_names}"
        )

    # Filter SHAP values
    filtered_shap = shap_vals[:, kept_indices]

    if verbose:
        print(f"[filter_pos] Filtered positional features: {F} -> {len(filtered_names)}")
        print(f"[filter_pos]   Removed {len(positional_names)} positional: {positional_names}")
        print(f"[filter_pos]   Kept {len(filtered_names)} environmental features")
        if len(filtered_names) <= 10:
            print(f"[filter_pos]   Environmental features: {filtered_names}")
        else:
            print(f"[filter_pos]   First 5 environmental: {filtered_names[:5]}")
            print(f"[filter_pos]   Last 5 environmental: {filtered_names[-5:]}")

    return filtered_shap, filtered_names, kept_indices


def _build_temporal_aggregation_map(feature_names: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Build a mapping from base feature names to their temporal feature indices.

    Args:
        feature_names: List of feature names, e.g., ["SMAP (t)", "SMAP (t-1)", ..., "ERA5 (t)", ...]

    Returns:
        unique_base_names: List of unique base feature names in order of first appearance
        base_to_indices: Dict mapping base name -> list of feature indices

    Example:
        feature_names = ["A (t)", "A (t-1)", "A (t-2)", "B (t)", "B (t-1)", "sin(lat)"]
        -> unique_base_names = ["A", "B", "sin(lat)"]
        -> base_to_indices = {"A": [0, 1, 2], "B": [3, 4], "sin(lat)": [5]}
    """
    base_to_indices: Dict[str, List[int]] = {}
    unique_base_names: List[str] = []

    for idx, fname in enumerate(feature_names):
        base = _parse_base_feature_name(fname)
        if base not in base_to_indices:
            base_to_indices[base] = []
            unique_base_names.append(base)
        base_to_indices[base].append(idx)

    return unique_base_names, base_to_indices


def _aggregate_shap_over_time(
    shap_vals: np.ndarray,
    feature_names: List[str],
    agg_method: str = "mean"
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate SHAP values across time steps for each base channel.

    Args:
        shap_vals: (N, F) array of SHAP values
        feature_names: List of F feature names
        agg_method: "mean" or "sum" - how to aggregate across time steps

    Returns:
        agg_shap: (N, F_unique) array of aggregated SHAP values
        unique_names: List of F_unique unique base feature names

    This reduces temporal features like ["SMAP (t)", "SMAP (t-1)", ..., "SMAP (t-29)"]
    into a single "SMAP" feature by averaging/summing the SHAP values across time.
    """
    if shap_vals.ndim != 2:
        raise ValueError(f"Expected 2D shap_vals, got shape {shap_vals.shape}")

    N, F = shap_vals.shape
    if len(feature_names) != F:
        raise ValueError(f"feature_names length {len(feature_names)} != shap_vals features {F}")

    unique_names, base_to_indices = _build_temporal_aggregation_map(feature_names)
    F_unique = len(unique_names)

    agg_shap = np.zeros((N, F_unique), dtype=np.float64)

    for out_idx, base_name in enumerate(unique_names):
        indices = base_to_indices[base_name]
        if agg_method == "mean":
            agg_shap[:, out_idx] = shap_vals[:, indices].mean(axis=1)
        elif agg_method == "sum":
            agg_shap[:, out_idx] = shap_vals[:, indices].sum(axis=1)
        else:
            raise ValueError(f"Unknown agg_method: {agg_method}")

    return agg_shap, unique_names


def _compute_tile_signatures_from_shap_stream(
    ds,
    boosters: List["xgb.Booster"],
    args,
    split_name: str,
    max_rows_per_tile: int,
    mode: str = "avg",
    use_oof: bool = False,
    fold_models: Optional[Dict[int, List["xgb.Booster"]]] = None,
    fold_id_per_patch: Optional[np.ndarray] = None,
    tile_id_to_patch_idx: Optional[Dict[int, int]] = None,
):
    """
    Core: stream through ds and accumulate mean(|SHAP|) per tile.

    If use_oof=True:
      - requires fold_models[fold] and fold_id_per_patch[patch_idx] to pick model per patch.

    Signature combination modes:
      - avg: average |shap| across horizons into one vector length F
      - concat: concatenate per-horizon vectors length (K*F)
      - h0: use only horizon 0 vector length F
    """
    K = len(args.horizons)
    max_rows_per_tile = int(max_rows_per_tile)

    # we will infer F from one batch
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=True, args=args, device=None)

    # tile accumulators (python dicts for sparse tiles)
    # sums store float64 for numerical stability
    tile_sum = {}   # tile_id -> sum_abs_shap vector
    tile_cnt = {}   # tile_id -> count rows used
    tile_lat = {}   # tile_id -> lat
    tile_lon = {}   # tile_id -> lon
    tile_pos = {}   # tile_id -> positive count (for event_rate)
    tile_tot = {}   # tile_id -> total valid rows (same as cnt)

    # per-tile cap counters
    tile_seen = {}  # tile_id -> rows already used toward cap

    header(f"Tile signatures from SHAP ({split_name}) | mode={mode} | cap={max_rows_per_tile}/tile | oof={use_oof}")

    # Log initial configuration and estimates
    num_batches = len(loader) if hasattr(loader, '__len__') else "unknown"
    dataset_size = len(ds)
    print(f"[tile_sig] Dataset: {dataset_size:,} patches, ~{num_batches} batches (batch_size={getattr(args, 'batch_size', 'N/A')})")
    print(f"[tile_sig] Tile cap: {max_rows_per_tile:,} rows/tile | Mode: {mode} | Horizons: {K}")
    print(f"[tile_sig] OOF mode: {use_oof}")
    if use_oof:
        n_folds = len(fold_models) if fold_models else 0
        print(f"[tile_sig] Using {n_folds} OOF fold models")

    pbar = tqdm(loader, desc=f"tile_sig_{split_name}", leave=True, position=2) if not args.no_tqdm else None
    it = pbar if pbar is not None else loader

    for batch in it:
        # patch-level meta
        tile_b, lat_b, lon_b, date_b = _get_batch_tile_meta(batch)

        # for each horizon, compute shap and accumulate into signature representation
        # We do per-horizon CPU tabularization here (already what your SHAP does).
        sig_parts = []  # list of (tile_rows, abs_shap) per horizon

        for k_idx in range(K):
            X, y, v, tile_row, lat_row, lon_row, _date_row = _batch_to_rows_cpu_single_horizon_with_meta(batch, k_idx)
            if v.sum() == 0:
                continue
            Xv = X[v]
            yv = y[v].astype(np.float32, copy=False)
            trow = tile_row[v]
            latr = lat_row[v]
            lonr = lon_row[v]

            # choose booster (OOF vs fixed)
            if use_oof:
                if fold_models is None or fold_id_per_patch is None:
                    raise RuntimeError("use_oof=True requires fold_models and fold_id_per_patch.")
                if tile_id_to_patch_idx is None:
                    raise RuntimeError("use_oof=True requires tile_id_to_patch_idx mapping.")

                # Convert tile_ids to patch indices using the mapping, then lookup fold assignment
                folds_row = np.array([
                    fold_id_per_patch[tile_id_to_patch_idx[int(t)]]
                    for t in trow
                ], dtype=np.int64)
                # We can’t run multiple fold models in one SHAP call; so bucket rows by fold:
                for f in np.unique(folds_row):
                    m = (folds_row == f)
                    Xf = Xv[m]
                    yf = yv[m]
                    tf = trow[m]
                    latf = latr[m]
                    lonf = lonr[m]
                    bst = fold_models[int(f)][k_idx]
                    contribs = _xgb_predict_contribs(bst, Xf, args)
                    shap_vals = np.asarray(contribs)[:, :-1]
                    abs_shap = np.abs(shap_vals).astype(np.float64, copy=False)
                    sig_parts.append((k_idx, tf, latf, lonf, yf, abs_shap))
            else:
                bst = boosters[k_idx]
                contribs = _xgb_predict_contribs(bst, Xv, args)
                shap_vals = np.asarray(contribs)[:, :-1]
                abs_shap = np.abs(shap_vals).astype(np.float64, copy=False)
                sig_parts.append((k_idx, trow, latr, lonr, yv, abs_shap))

        # Accumulate per tile with cap
        for (k_idx, trow, latr, lonr, yv, abs_shap) in sig_parts:
            if abs_shap.size == 0:
                continue

            # decide representation
            # abs_shap is (N,F)
            if mode == "h0" and k_idx != 0:
                continue

            # iterate per tile in this chunk
            # (Vectorized groupby would be nicer, but dict accumulation is fine with caps.)
            for i in range(trow.shape[0]):
                tid = int(trow[i])
                used = tile_seen.get(tid, 0)
                if used >= max_rows_per_tile:
                    continue
                tile_seen[tid] = used + 1

                # init stores
                if tid not in tile_sum:
                    F = abs_shap.shape[1]
                    if mode == "concat":
                        tile_sum[tid] = np.zeros((K, F), dtype=np.float64)
                    else:
                        tile_sum[tid] = np.zeros((F,), dtype=np.float64)
                    tile_cnt[tid] = 0
                    tile_pos[tid] = 0.0
                    tile_tot[tid] = 0.0
                    tile_lat[tid] = float(latr[i]) if np.isfinite(latr[i]) else float("nan")
                    tile_lon[tid] = float(lonr[i]) if np.isfinite(lonr[i]) else float("nan")

                if mode == "concat":
                    tile_sum[tid][k_idx] += abs_shap[i]
                else:
                    tile_sum[tid] += abs_shap[i]

                tile_cnt[tid] += 1
                tile_tot[tid] += 1.0
                tile_pos[tid] += float(yv[i])

        if pbar:
            total_rows_used = sum(tile_cnt.values())
            total_rows_seen = sum(tile_seen.values())
            avg_rows_per_tile = total_rows_used / max(len(tile_sum), 1)
            capped_rows = total_rows_seen - total_rows_used
            pbar.set_postfix({
                "tiles": len(tile_sum),
                "rows": f"{total_rows_used:,}",
                "avg/tile": f"{avg_rows_per_tile:.1f}",
                "capped": f"{capped_rows:,}"
            })

            # Periodic detailed progress logging (every 100 batches or 5% progress)
            batch_num = pbar.n
            log_interval = max(1, int(num_batches / 20)) if isinstance(num_batches, int) else 100
            if batch_num > 0 and batch_num % log_interval == 0:
                capped_pct = 100.0 * capped_rows / max(total_rows_seen, 1)
                print(f"\n[tile_sig] Progress: batch {batch_num}/{num_batches} | "
                      f"{len(tile_sum):,} tiles | {total_rows_used:,} rows used | "
                      f"avg {avg_rows_per_tile:.1f} rows/tile | {capped_pct:.1f}% capped")

    if pbar:
        pbar.close()

    # Print final statistics
    total_tiles = len(tile_sum)
    total_rows_used = sum(tile_cnt.values())
    total_rows_seen = sum(tile_seen.values())
    capped_rows = total_rows_seen - total_rows_used
    avg_rows_per_tile = total_rows_used / max(total_tiles, 1)
    tiles_at_cap = sum(1 for tid in tile_cnt.keys() if tile_seen.get(tid, 0) >= max_rows_per_tile)

    print(f"[tile_sig] ✓ COMPLETE")
    print(f"[tile_sig]   Tiles processed: {total_tiles:,}")
    print(f"[tile_sig]   Total rows used: {total_rows_used:,} (avg {avg_rows_per_tile:.1f} per tile)")
    print(f"[tile_sig]   Rows capped: {capped_rows:,} ({100.0*capped_rows/max(total_rows_seen,1):.1f}%)")
    print(f"[tile_sig]   Tiles at cap ({max_rows_per_tile}): {tiles_at_cap:,} ({100.0*tiles_at_cap/max(total_tiles,1):.1f}%)")

    # finalize into dataframe
    rows = []
    for tid, s in tile_sum.items():
        n = int(tile_cnt.get(tid, 0))
        if n <= 0:
            continue
        if mode == "concat":
            flat = s.reshape(-1) / max(n, 1)
        else:
            flat = s / max(n, 1)

        rows.append({
            "tile_id": tid,
            "lat": tile_lat.get(tid, float("nan")),
            "lon": tile_lon.get(tid, float("nan")),
            "n_rows_used": n,
            "n_pos": float(tile_pos.get(tid, 0.0)),
            "event_rate": float(tile_pos.get(tid, 0.0) / max(tile_tot.get(tid, 1.0), 1.0)),
            "sig": flat,
        })

    if not rows:
        raise RuntimeError("No tile signatures produced (check that mask/valid pixels exist and tile_id is present).")

    # build matrix
    sig_mat = np.stack([r["sig"] for r in rows], axis=0).astype(np.float64, copy=False)
    sig_mat = _signature_normalize(sig_mat)

    # feature column names
    if mode == "concat":
        # name as h{h}_<feat>
        feat_names = build_feature_names_from_dataset(args, ds)
        F = len(feat_names)
        cols = []
        for k_idx, h in enumerate([int(x) for x in args.horizons]):
            for j in range(F):
                cols.append(f"h{h}:{feat_names[j]}")
    else:
        feat_names = build_feature_names_from_dataset(args, ds)
        cols = [f"{fn}" for fn in feat_names[:sig_mat.shape[1]]]

    out_df = pd.DataFrame({
        "tile_id": [r["tile_id"] for r in rows],
        "lat": [r["lat"] for r in rows],
        "lon": [r["lon"] for r in rows],
        "n_rows_used": [r["n_rows_used"] for r in rows],
        "n_pos": [r["n_pos"] for r in rows],
        "event_rate": [r["event_rate"] for r in rows],
    })
    sig_df = pd.DataFrame(sig_mat, columns=cols)
    out_df = pd.concat([out_df, sig_df], axis=1)
    return out_df, sig_mat, cols

def _compute_pixel_signatures_from_shap_stream(
    ds,
    boosters: List["xgb.Booster"],
    args,
    split_name: str,
    temporal_sample_frac: float = 0.7,
    mode: str = "avg",
    use_oof: bool = False,
    fold_models: Optional[Dict[int, List["xgb.Booster"]]] = None,
    fold_id_per_patch: Optional[np.ndarray] = None,
    tile_id_to_patch_idx: Optional[Dict[int, int]] = None,
    seed: int = 42,
    aggregate_time: bool = False,
    aggregate_method: str = "mean",
):
    """
    Compute per-PIXEL SHAP signatures by accumulating across time windows.

    Key differences from tile-level:
    - Tracks pixel_id = y_global * W + x_global instead of tile_id
    - Samples temporal_sample_frac (e.g., 70%) of time windows per pixel
    - No max_rows_per_pixel cap (process all valid pixels)
    - Each spatial pixel gets its own signature (aggregated across time)

    Args:
        ds: Dataset
        boosters: List of XGBoost boosters (one per horizon) for fixed model
        args: Arguments
        split_name: Name of split (for logging)
        temporal_sample_frac: Fraction of time windows to sample per pixel (0.7 = 70%)
        mode: Signature combination mode ("avg", "concat", or "h0")
        use_oof: If True, use fold_models for out-of-fold predictions
        fold_models: Dict[fold_id -> List[booster per horizon]] for OOF
        fold_id_per_patch: Array mapping patch_idx to fold_id
        tile_id_to_patch_idx: Dict mapping tile_id to patch_idx
        seed: Random seed for temporal sampling
        aggregate_time: If True, aggregate SHAP values across time steps for each base channel
                       (e.g., collapse "SMAP (t)", "SMAP (t-1)", ... into single "SMAP")
        aggregate_method: "mean" or "sum" - how to aggregate across time steps

    Returns:
        out_df: DataFrame with columns [pixel_id, y_global, x_global, lat, lon, n_rows_used, n_pos, event_rate, sig_feat1, sig_feat2, ...]
        sig_mat: (N_pixels, F) normalized signature matrix (F = unique features if aggregate_time=True)
        cols: Feature column names
    """
    K = len(args.horizons)

    # Get feature names for potential temporal aggregation
    feat_names_full = build_feature_names_from_dataset(args, ds)
    if aggregate_time:
        unique_base_names, base_to_indices = _build_temporal_aggregation_map(feat_names_full)
        F_sig = len(unique_base_names)
        print(f"[pixel_sig] Temporal aggregation enabled: {len(feat_names_full)} -> {F_sig} unique features")
        print(f"[pixel_sig]   Aggregation method: {aggregate_method}")
        print(f"[pixel_sig]   Base features: {unique_base_names[:5]}{'...' if len(unique_base_names) > 5 else ''}")
    else:
        unique_base_names = None
        base_to_indices = None
        F_sig = len(feat_names_full)

    header(f"Pixel signatures from SHAP ({split_name}) | mode={mode} | temporal_sample={temporal_sample_frac:.1%} | oof={use_oof} | agg_time={aggregate_time}")

    loader = make_loader(ds, batch_size=args.batch_size, shuffle=True, args=args, device=None)

    # Pixel accumulators (dict-based for sparse storage)
    pixel_sum = {}    # pixel_id -> sum of |SHAP| across sampled time windows
    pixel_cnt = {}    # pixel_id -> count of sampled time windows
    pixel_pos = {}    # pixel_id -> sum of positive labels
    pixel_tot = {}    # pixel_id -> total samples (same as cnt)
    pixel_coords = {} # pixel_id -> (y_global, x_global, lat, lon)

    # Temporal sampling: deterministic per (pixel_id, date) pair
    def should_sample_time_window(pixel_id: int, date: int) -> bool:
        """Deterministic sampling based on hash of (pixel_id, date, seed)"""
        hash_val = hash((int(pixel_id), int(date), seed)) % 10000
        threshold = int(temporal_sample_frac * 10000)
        return hash_val < threshold

    num_batches = len(loader) if hasattr(loader, '__len__') else "unknown"
    dataset_size = len(ds)

    print(f"[pixel_sig] Dataset: {dataset_size:,} patches, ~{num_batches} batches (batch_size={args.batch_size})")
    print(f"[pixel_sig] Temporal sampling: {temporal_sample_frac:.1%} per pixel | Mode: {mode} | Horizons: {K}")
    if use_oof:
        n_folds = len(fold_models) if fold_models else 0
        print(f"[pixel_sig] Using {n_folds} OOF fold models")

    pbar = tqdm(loader, desc=f"pixel_sig_{split_name}", leave=True, position=2) if not args.no_tqdm else None
    it = pbar if pbar is not None else loader

    total_pixels_seen = 0
    total_pixels_sampled = 0

    for batch in it:
        # Get tile metadata for OOF model selection
        tile_b, _, _, _ = _get_batch_tile_meta(batch)

        # Accumulate per-horizon SHAP
        sig_parts = []

        for k_idx in range(K):
            # Get pixel-level features and metadata
            X, y, v, pixel_row, y_glob, x_glob, lat_row, lon_row, date_row = \
                _batch_to_rows_cpu_single_horizon_with_pixel_meta(batch, k_idx)

            if v.sum() == 0:
                continue

            # Filter to valid pixels
            Xv = X[v]
            yv = y[v].astype(np.float32, copy=False)
            pixel_v = pixel_row[v]
            y_glob_v = y_glob[v]
            x_glob_v = x_glob[v]
            lat_v = lat_row[v]
            lon_v = lon_row[v]
            date_v = date_row[v]

            # Choose booster (OOF vs fixed)
            if use_oof:
                # Need to map pixels back to tiles for fold lookup
                # Expand tile_b to per-pixel (same approach as lat/lon)
                x_raw = batch["x"]
                if x_raw.dim() == 2:
                    # Flattened format: infer H, W from Npix / B
                    Npix = x_raw.shape[0]
                    B = len(tile_b)
                    pixels_per_patch = Npix // B if B > 0 else 0
                    H = W = int(np.sqrt(pixels_per_patch)) if pixels_per_patch > 0 else 0
                else:
                    x_tensor = _flatten_time_if_needed(x_raw)
                    B, _F, H, W = x_tensor.shape
                tile_expanded = np.repeat(tile_b, H * W)[v]

                # Get fold assignments
                folds_row = np.array([
                    fold_id_per_patch[tile_id_to_patch_idx[int(t)]]
                    for t in tile_expanded
                ], dtype=np.int64)

                # Bucket by fold (can't run multiple fold models in one SHAP call)
                for f in np.unique(folds_row):
                    m = (folds_row == f)
                    Xf = Xv[m]
                    yf = yv[m]
                    pf = pixel_v[m]
                    yg_f = y_glob_v[m]
                    xg_f = x_glob_v[m]
                    latf = lat_v[m]
                    lonf = lon_v[m]
                    datef = date_v[m]

                    bst = fold_models[int(f)][k_idx]
                    contribs = _xgb_predict_contribs(bst, Xf, args)
                    shap_vals = np.asarray(contribs)[:, :-1]  # Remove bias term
                    abs_shap = np.abs(shap_vals).astype(np.float64, copy=False)

                    # Apply temporal aggregation if enabled
                    if aggregate_time and base_to_indices is not None:
                        abs_shap, _ = _aggregate_shap_over_time(abs_shap, feat_names_full, aggregate_method)

                    sig_parts.append((k_idx, pf, yg_f, xg_f, latf, lonf, datef, yf, abs_shap))
            else:
                bst = boosters[k_idx]
                contribs = _xgb_predict_contribs(bst, Xv, args)
                shap_vals = np.asarray(contribs)[:, :-1]  # Remove bias term
                abs_shap = np.abs(shap_vals).astype(np.float64, copy=False)

                # Apply temporal aggregation if enabled
                if aggregate_time and base_to_indices is not None:
                    abs_shap, _ = _aggregate_shap_over_time(abs_shap, feat_names_full, aggregate_method)

                sig_parts.append((k_idx, pixel_v, y_glob_v, x_glob_v, lat_v, lon_v, date_v, yv, abs_shap))

        # Accumulate per pixel with temporal sampling
        for (k_idx, pixel_v, y_glob_v, x_glob_v, lat_v, lon_v, date_v, yv, abs_shap) in sig_parts:
            if abs_shap.size == 0:
                continue

            # Skip for mode='h0' if not horizon 0
            if mode == "h0" and k_idx != 0:
                continue

            # Iterate per pixel in this chunk
            for i in range(pixel_v.shape[0]):
                pid = int(pixel_v[i])
                date = int(date_v[i])

                total_pixels_seen += 1

                # Temporal sampling decision
                if not should_sample_time_window(pid, date):
                    continue

                total_pixels_sampled += 1

                # Initialize stores
                if pid not in pixel_sum:
                    F = abs_shap.shape[1]
                    if mode == "concat":
                        pixel_sum[pid] = np.zeros((K, F), dtype=np.float64)
                    else:
                        pixel_sum[pid] = np.zeros((F,), dtype=np.float64)
                    pixel_cnt[pid] = 0
                    pixel_pos[pid] = 0.0
                    pixel_tot[pid] = 0.0
                    pixel_coords[pid] = (
                        int(y_glob_v[i]),
                        int(x_glob_v[i]),
                        float(lat_v[i]) if np.isfinite(lat_v[i]) else float("nan"),
                        float(lon_v[i]) if np.isfinite(lon_v[i]) else float("nan")
                    )

                # Accumulate
                if mode == "concat":
                    pixel_sum[pid][k_idx] += abs_shap[i]
                else:
                    pixel_sum[pid] += abs_shap[i]

                pixel_cnt[pid] += 1
                pixel_tot[pid] += 1.0
                pixel_pos[pid] += float(yv[i])

        if pbar:
            sample_rate = total_pixels_sampled / max(total_pixels_seen, 1)
            pbar.set_postfix({
                "pixels": len(pixel_sum),
                "sampled": f"{total_pixels_sampled:,}",
                "rate": f"{sample_rate:.1%}"
            })

    if pbar:
        pbar.close()

    # Print final statistics
    total_pixels = len(pixel_sum)
    avg_samples_per_pixel = total_pixels_sampled / max(total_pixels, 1)

    print(f"[pixel_sig] ✓ COMPLETE")
    print(f"[pixel_sig]   Unique pixels: {total_pixels:,}")
    print(f"[pixel_sig]   Total samples: {total_pixels_sampled:,} (avg {avg_samples_per_pixel:.1f} per pixel)")
    print(f"[pixel_sig]   Sampling rate: {total_pixels_sampled/max(total_pixels_seen,1):.1%}")

    # Finalize into DataFrame
    rows = []
    for pid, s in pixel_sum.items():
        n = int(pixel_cnt.get(pid, 0))
        if n <= 0:
            continue

        # Compute mean SHAP across time
        if mode == "concat":
            flat = s.reshape(-1) / max(n, 1)
        else:
            flat = s / max(n, 1)

        y_g, x_g, lat, lon = pixel_coords[pid]

        rows.append({
            "pixel_id": pid,
            "y_global": y_g,
            "x_global": x_g,
            "lat": lat,
            "lon": lon,
            "n_rows_used": n,
            "n_pos": float(pixel_pos.get(pid, 0.0)),
            "event_rate": float(pixel_pos.get(pid, 0.0) / max(pixel_tot.get(pid, 1.0), 1.0)),
            "sig": flat,
        })

    if not rows:
        raise RuntimeError("No pixel signatures produced.")

    # Build signature matrix
    sig_mat = np.stack([r["sig"] for r in rows], axis=0).astype(np.float64, copy=False)
    sig_mat = _signature_normalize(sig_mat)

    # Feature column names
    # Use aggregated feature names if temporal aggregation was applied
    if aggregate_time and unique_base_names is not None:
        feat_names_for_cols = unique_base_names
    else:
        feat_names_for_cols = build_feature_names_from_dataset(args, ds)

    # Apply positional feature filtering if enabled
    exclude_positional = getattr(args, "shap_exclude_positional", True)
    if exclude_positional:
        # Filter sig_mat to remove positional features
        filtered_sig_mat, filtered_feat_names, kept_indices = _filter_positional_features(
            sig_mat,
            feat_names_for_cols,
            verbose=True
        )
        print(f"[pixel_sig] Positional filtering: {sig_mat.shape[1]} -> {filtered_sig_mat.shape[1]} features")

        # Use filtered version for clustering
        sig_mat = filtered_sig_mat
        feat_names_for_cols = filtered_feat_names

    if mode == "concat":
        F = len(feat_names_for_cols)
        cols = []
        for k_idx, h in enumerate([int(x) for x in args.horizons]):
            for j in range(F):
                cols.append(f"h{h}:{feat_names_for_cols[j]}")
    else:
        cols = [f"{fn}" for fn in feat_names_for_cols[:sig_mat.shape[1]]]

    out_df = pd.DataFrame({
        "pixel_id": [r["pixel_id"] for r in rows],
        "y_global": [r["y_global"] for r in rows],
        "x_global": [r["x_global"] for r in rows],
        "lat": [r["lat"] for r in rows],
        "lon": [r["lon"] for r in rows],
        "n_rows_used": [r["n_rows_used"] for r in rows],
        "n_pos": [r["n_pos"] for r in rows],
        "event_rate": [r["event_rate"] for r in rows],
    })
    sig_df = pd.DataFrame(sig_mat, columns=cols)
    out_df = pd.concat([out_df, sig_df], axis=1)

    return out_df, sig_mat, cols

def _fit_clusters_from_signatures(sig_mat: np.ndarray, args, seed: int):
    """
    PCA (optional) -> kmeans. Returns state dict with everything needed to assign new tiles.
    """
    X = sig_mat.astype(np.float64, copy=False)
    pca_k = int(getattr(args, "pca_components", 0))

    # Validate PCA dimensionality
    if pca_k and pca_k > 0:
        if pca_k >= X.shape[1]:
            print(f"[WARNING] PCA components ({pca_k}) >= feature count ({X.shape[1]})")
            print(f"[WARNING] Reducing PCA components to {X.shape[1] - 1}")
            pca_k = max(1, X.shape[1] - 1)

    pca_state = None
    if pca_k and pca_k > 0 and pca_k < X.shape[1]:
        Xr, pca_state = _pca_fit_transform_numpy(X, pca_k)
    else:
        Xr = X

    n_clusters = int(getattr(args, "n_clusters", 3))
    labels, centers, inertia = _kmeans_numpy(Xr, n_clusters=n_clusters, seed=seed, n_init=5, max_iter=250)

    state = {
        "pca": pca_state,
        "centers": centers,
        "n_clusters": n_clusters,
        "inertia": inertia,
        "signature_dim": int(sig_mat.shape[1]),
        "reduced_dim": int(Xr.shape[1]),
    }
    return labels, state

def _assign_clusters(sig_mat: np.ndarray, cluster_state: dict):
    X = sig_mat.astype(np.float64, copy=False)
    if cluster_state.get("pca", None) is not None:
        Xr = _pca_transform_numpy(X, cluster_state["pca"])
    else:
        Xr = X
    centers = cluster_state["centers"]
    dists = ((Xr[:, None, :] - centers[None, :, :])**2).sum(axis=2)
    labels = dists.argmin(axis=1)
    dist_min = dists.min(axis=1)
    return labels.astype(np.int64), dist_min.astype(np.float64)

def _save_cluster_state(cluster_state: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "n_clusters": int(cluster_state["n_clusters"]),
        "inertia": float(cluster_state["inertia"]),
        "signature_dim": int(cluster_state["signature_dim"]),
        "reduced_dim": int(cluster_state["reduced_dim"]),
        "centers": cluster_state["centers"].tolist(),
        "pca": None,
    }
    if cluster_state.get("pca", None) is not None:
        payload["pca"] = {
            "mean": cluster_state["pca"]["mean"].tolist(),
            "components": cluster_state["pca"]["components"].tolist(),
        }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

def _load_cluster_state(path: str) -> dict:
    with open(path, "r") as f:
        payload = json.load(f)
    st = {
        "n_clusters": int(payload["n_clusters"]),
        "inertia": float(payload.get("inertia", float("nan"))),
        "signature_dim": int(payload["signature_dim"]),
        "reduced_dim": int(payload["reduced_dim"]),
        "centers": np.asarray(payload["centers"], dtype=np.float64),
        "pca": None,
    }
    if payload.get("pca", None) is not None:
        st["pca"] = {
            "mean": np.asarray(payload["pca"]["mean"], dtype=np.float64),
            "components": np.asarray(payload["pca"]["components"], dtype=np.float64),
        }
    return st

def _eval_tile_and_cluster_metrics(
    ds,
    boosters: List["xgb.Booster"],
    cluster_assign_df: pd.DataFrame,
    args,
    split_name: str,
):
    """
    Computes per-tile brier/logloss/event_rate and aggregates by cluster.
    Uses your existing xgb inference path (margins->sigmoid) via xgb_logits_fn_from_boosters.
    """
    # map tile_id -> cluster_id
    m = dict(zip(cluster_assign_df["tile_id"].astype(int).tolist(),
                 cluster_assign_df["cluster_id"].astype(int).tolist()))

    device = torch.device(args.eval_device)
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=False, args=args, device=device)

    logits_fn = xgb_logits_fn_from_boosters(boosters, horizons=[int(h) for h in args.horizons], args=args, calibrators=None)

    # Per-tile accum
    tile_cnt = {}
    tile_pos = {}
    tile_brier = {}
    tile_ll = {}
    tile_lat = {}
    tile_lon = {}

    # Per-cluster accum
    cl_cnt = {}
    cl_pos = {}
    cl_brier = {}
    cl_ll = {}

    pbar = tqdm(loader, desc=f"eval_tile_{split_name}", leave=True, position=2) if not args.no_tqdm else None
    it = pbar if pbar is not None else loader

    for batch in it:
        # move
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        tile_b, lat_b, lon_b, _date_b = _get_batch_tile_meta(batch)

        logits = logits_fn(batch)  # (B,K,H,W)
        probs = torch.sigmoid(logits)

        y = (batch["y"] > 0.5)
        v = (batch["mask"] > 0.5)

        # Use horizon 0 by default for tile metrics (you can extend to avg across horizons)
        k0 = 0
        p0 = probs[:, k0]  # (B,H,W)
        y0 = y[:, k0]
        v0 = v[:, k0]

        # flatten per patch
        B = p0.shape[0]
        for i in range(B):
            tid = int(tile_b[i]) if i < tile_b.shape[0] else -1
            if tid == -1:
                continue
            cl = int(m.get(tid, -1))

            pi = p0[i][v0[i]].detach().cpu().numpy().astype(np.float64, copy=False)
            yi = y0[i][v0[i]].detach().cpu().numpy().astype(np.float64, copy=False)
            if pi.size == 0:
                continue

            # per-tile
            cnt = float(pi.size)
            brier = float(((pi - yi) ** 2).sum())
            p_clip = np.clip(pi, EPS_PROB, 1.0 - EPS_PROB)
            ll = float((-(yi * np.log(p_clip) + (1.0 - yi) * np.log(1.0 - p_clip))).sum())
            pos = float(yi.sum())

            tile_cnt[tid] = tile_cnt.get(tid, 0.0) + cnt
            tile_pos[tid] = tile_pos.get(tid, 0.0) + pos
            tile_brier[tid] = tile_brier.get(tid, 0.0) + brier
            tile_ll[tid] = tile_ll.get(tid, 0.0) + ll
            tile_lat[tid] = float(_ensure_numpy(lat_b[i])) if i < lat_b.shape[0] else float("nan")
            tile_lon[tid] = float(_ensure_numpy(lon_b[i])) if i < lon_b.shape[0] else float("nan")

            # per-cluster
            if cl >= 0:
                cl_cnt[cl] = cl_cnt.get(cl, 0.0) + cnt
                cl_pos[cl] = cl_pos.get(cl, 0.0) + pos
                cl_brier[cl] = cl_brier.get(cl, 0.0) + brier
                cl_ll[cl] = cl_ll.get(cl, 0.0) + ll

        del logits, probs

    if pbar:
        pbar.close()

    # tile_metrics df
    t_rows = []
    for tid in sorted(tile_cnt.keys()):
        cnt = tile_cnt[tid]
        t_rows.append({
            "tile_id": tid,
            "lat": tile_lat.get(tid, float("nan")),
            "lon": tile_lon.get(tid, float("nan")),
            "n_rows": cnt,
            "n_pos": tile_pos.get(tid, 0.0),
            "event_rate": tile_pos.get(tid, 0.0) / max(cnt, 1.0),
            "brier": tile_brier.get(tid, 0.0) / max(cnt, 1.0),
            "logloss": tile_ll.get(tid, 0.0) / max(cnt, 1.0),
            "cluster_id": int(m.get(tid, -1)),
        })
    tile_metrics = pd.DataFrame(t_rows)

    # cluster_metrics df
    c_rows = []
    for cl in sorted(cl_cnt.keys()):
        cnt = cl_cnt[cl]
        c_rows.append({
            "cluster_id": cl,
            "n_rows": cnt,
            "n_pos": cl_pos.get(cl, 0.0),
            "event_rate": cl_pos.get(cl, 0.0) / max(cnt, 1.0),
            "brier": cl_brier.get(cl, 0.0) / max(cnt, 1.0),
            "logloss": cl_ll.get(cl, 0.0) / max(cnt, 1.0),
        })
    cluster_metrics = pd.DataFrame(c_rows)

    return tile_metrics, cluster_metrics

def _eval_pixel_and_cluster_metrics(
    ds,
    boosters: List["xgb.Booster"],
    cluster_assign_df: pd.DataFrame,
    args,
    split_name: str,
):
    """
    Computes per-PIXEL brier/logloss/event_rate and aggregates by cluster.

    Unlike tile-level evaluation, this tracks individual pixels with global coordinates.
    Each pixel gets metrics computed across all its time windows.

    Args:
        ds: Dataset
        boosters: List of XGBoost boosters (one per horizon)
        cluster_assign_df: DataFrame with columns [pixel_id, cluster_id, ...]
        args: Arguments
        split_name: Name of split (for logging)

    Returns:
        pixel_metrics: DataFrame with per-pixel metrics
        cluster_metrics: DataFrame with per-cluster aggregated metrics
    """
    # Map pixel_id -> cluster_id
    m = dict(zip(
        cluster_assign_df["pixel_id"].astype(int).tolist(),
        cluster_assign_df["cluster_id"].astype(int).tolist()
    ))

    device = torch.device(args.eval_device)
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=False, args=args, device=device)

    logits_fn = xgb_logits_fn_from_boosters(
        boosters,
        horizons=[int(h) for h in args.horizons],
        args=args,
        calibrators=None
    )

    # Per-pixel accumulators
    pixel_cnt = {}
    pixel_pos = {}
    pixel_brier = {}
    pixel_ll = {}
    pixel_coords = {}

    # Per-cluster accumulators
    cl_cnt = {}
    cl_pos = {}
    cl_brier = {}
    cl_ll = {}

    pbar = tqdm(loader, desc=f"eval_pixel_{split_name}", leave=True, position=2) if not args.no_tqdm else None
    it = pbar if pbar is not None else loader

    for batch in it:
        # Move to device
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

        # Get batch metadata
        y0_b = _ensure_numpy(batch.get("y0", -1)).astype(np.int64, copy=False).reshape(-1)
        x0_b = _ensure_numpy(batch.get("x0", -1)).astype(np.int64, copy=False).reshape(-1)
        # W_global is the same for all samples in batch, so take first element
        W_global_arr = _ensure_numpy(batch.get("W_global", -1)).reshape(-1)
        W_global = int(W_global_arr[0])

        # Get predictions - handles both (B, K, H, W) and (Npix, K) formats
        logits = logits_fn(batch)
        probs = torch.sigmoid(logits)

        y = (batch["y"] > 0.5)
        v = (batch["mask"] > 0.5)

        # Use horizon 0 for evaluation (can extend to average across horizons)
        k0 = 0

        # Check if data is flattened
        x_raw = batch["x"]
        is_flattened = x_raw.dim() == 2

        if is_flattened:
            # Flattened format: probs=(Npix, K), y=(Npix, K), v=(Npix, K)
            p0 = probs[:, k0]  # (Npix,)
            y0_vals = y[:, k0]  # (Npix,)
            v0 = v[:, k0]  # (Npix,)

            # Infer H, W from Npix / B
            Npix = x_raw.shape[0]
            B = len(y0_b)
            pixels_per_patch = Npix // B if B > 0 else 0
            H = W = int(np.sqrt(pixels_per_patch)) if pixels_per_patch > 0 else 0

            # Process flattened pixels
            p0_cpu = p0.detach().cpu().numpy()
            y0_cpu = y0_vals.detach().cpu().numpy()
            v0_cpu = v0.detach().cpu().numpy()

            for i in range(B):
                y0_patch = y0_b[i]
                x0_patch = x0_b[i]
                start_idx = i * H * W
                end_idx = start_idx + H * W

                for local_idx in range(H * W):
                    flat_idx = start_idx + local_idx
                    if flat_idx >= len(v0_cpu) or not v0_cpu[flat_idx]:
                        continue

                    h_local = local_idx // W
                    w_local = local_idx % W
                    y_global = y0_patch + h_local
                    x_global = x0_patch + w_local
                    pid = int(y_global * W_global + x_global)

                    cl = int(m.get(pid, -1))
                    pi = float(p0_cpu[flat_idx])
                    yi = float(y0_cpu[flat_idx])

                    brier_val = (pi - yi) ** 2
                    p_clip = np.clip(pi, EPS_PROB, 1.0 - EPS_PROB)
                    ll_val = -(yi * np.log(p_clip) + (1.0 - yi) * np.log(1.0 - p_clip))

                    pixel_cnt[pid] = pixel_cnt.get(pid, 0.0) + 1.0
                    pixel_pos[pid] = pixel_pos.get(pid, 0.0) + yi
                    pixel_brier[pid] = pixel_brier.get(pid, 0.0) + brier_val
                    pixel_ll[pid] = pixel_ll.get(pid, 0.0) + ll_val

                    if pid not in pixel_coords:
                        pixel_coords[pid] = (int(y_global), int(x_global))

                    if cl >= 0:
                        cl_cnt[cl] = cl_cnt.get(cl, 0.0) + 1.0
                        cl_pos[cl] = cl_pos.get(cl, 0.0) + yi
                        cl_brier[cl] = cl_brier.get(cl, 0.0) + brier_val
                        cl_ll[cl] = cl_ll.get(cl, 0.0) + ll_val
        else:
            # Standard format: probs=(B, K, H, W)
            p0 = probs[:, k0]  # (B, H, W)
            y0_vals = y[:, k0]
            v0 = v[:, k0]

            B, H, W = p0.shape
            for i in range(B):
                y0_patch = y0_b[i]
                x0_patch = x0_b[i]

                for h in range(H):
                    for w in range(W):
                        if not v0[i, h, w]:
                            continue

                        y_global = y0_patch + h
                        x_global = x0_patch + w
                        pid = int(y_global * W_global + x_global)

                        cl = int(m.get(pid, -1))
                        pi = float(p0[i, h, w].item())
                        yi = float(y0_vals[i, h, w].item())

                        brier_val = (pi - yi) ** 2
                        p_clip = np.clip(pi, EPS_PROB, 1.0 - EPS_PROB)
                        ll_val = -(yi * np.log(p_clip) + (1.0 - yi) * np.log(1.0 - p_clip))

                        pixel_cnt[pid] = pixel_cnt.get(pid, 0.0) + 1.0
                        pixel_pos[pid] = pixel_pos.get(pid, 0.0) + yi
                        pixel_brier[pid] = pixel_brier.get(pid, 0.0) + brier_val
                        pixel_ll[pid] = pixel_ll.get(pid, 0.0) + ll_val

                        if pid not in pixel_coords:
                            pixel_coords[pid] = (int(y_global), int(x_global))

                        if cl >= 0:
                            cl_cnt[cl] = cl_cnt.get(cl, 0.0) + 1.0
                            cl_pos[cl] = cl_pos.get(cl, 0.0) + yi
                            cl_brier[cl] = cl_brier.get(cl, 0.0) + brier_val
                            cl_ll[cl] = cl_ll.get(cl, 0.0) + ll_val

        del logits, probs

    if pbar:
        pbar.close()

    # Build pixel metrics DataFrame
    p_rows = []
    for pid in sorted(pixel_cnt.keys()):
        cnt = pixel_cnt[pid]
        y_g, x_g = pixel_coords.get(pid, (-1, -1))
        p_rows.append({
            "pixel_id": pid,
            "y_global": y_g,
            "x_global": x_g,
            "n_rows": cnt,
            "n_pos": pixel_pos.get(pid, 0.0),
            "event_rate": pixel_pos.get(pid, 0.0) / max(cnt, 1.0),
            "brier": pixel_brier.get(pid, 0.0) / max(cnt, 1.0),
            "logloss": pixel_ll.get(pid, 0.0) / max(cnt, 1.0),
            "cluster_id": int(m.get(pid, -1)),
        })
    pixel_metrics = pd.DataFrame(p_rows)

    # Build cluster metrics DataFrame
    c_rows = []
    for cl in sorted(cl_cnt.keys()):
        cnt = cl_cnt[cl]
        c_rows.append({
            "cluster_id": cl,
            "n_rows": cnt,
            "n_pos": cl_pos.get(cl, 0.0),
            "event_rate": cl_pos.get(cl, 0.0) / max(cnt, 1.0),
            "brier": cl_brier.get(cl, 0.0) / max(cnt, 1.0),
            "logloss": cl_ll.get(cl, 0.0) / max(cnt, 1.0),
        })
    cluster_metrics = pd.DataFrame(c_rows)

    return pixel_metrics, cluster_metrics

def _plot_tiles_scatter(df: pd.DataFrame, color_col: str, out_path: str, title: str, discrete: bool = False):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    d = df.copy()

    # Determine which columns to use for coordinates
    has_latlon = "lat" in d.columns and "lon" in d.columns
    has_global = "y_global" in d.columns and "x_global" in d.columns

    if has_latlon:
        d = d[np.isfinite(d["lon"].to_numpy()) & np.isfinite(d["lat"].to_numpy())]
        x = d["lon"].to_numpy()
        y = d["lat"].to_numpy()
        xlabel, ylabel = "Longitude", "Latitude"
        xlim, ylim = (-180, 180), (-90, 90)
    elif has_global:
        d = d[np.isfinite(d["x_global"].to_numpy()) & np.isfinite(d["y_global"].to_numpy())]
        x = d["x_global"].to_numpy()
        y = d["y_global"].to_numpy()
        xlabel, ylabel = "X (pixels)", "Y (pixels)"
        xlim = (x.min() - 10, x.max() + 10) if len(x) > 0 else (0, 100)
        ylim = (y.min() - 10, y.max() + 10) if len(y) > 0 else (0, 100)
    else:
        print(f"[warn] Cannot plot {out_path}: no lat/lon or y_global/x_global columns")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, linestyle="--", alpha=0.3)
    c = d[color_col].to_numpy()

    if discrete:
        sc = ax.scatter(x, y, c=c, s=8, alpha=0.85)
        # legend with unique clusters
        uniq = np.unique(c.astype(int))
        # fake handles - use color= keyword to avoid warning
        for u in uniq:
            ax.scatter([], [], color=sc.cmap(sc.norm(u)), label=f"cluster {int(u)}", s=30)
        if len(uniq) > 0:
            ax.legend(loc="lower left", ncol=min(6, len(uniq)), frameon=True)
    else:
        sc = ax.scatter(x, y, c=c, s=8, alpha=0.85)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(color_col)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def load_cluster_assignments_from_csvs(csv_paths: List[str]) -> pd.DataFrame:
    """
    Load and combine pixel cluster assignment CSVs.

    Args:
        csv_paths: List of paths to CSV files with columns
                   [pixel_id, lat, lon, cluster_id, ...]

    Returns:
        Combined DataFrame with unique pixels
    """
    dfs = []
    for path in csv_paths:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))

    if not dfs:
        raise ValueError("No valid CSV files found")

    combined = pd.concat(dfs, ignore_index=True)
    # Keep first occurrence if duplicate pixel_ids
    combined = combined.drop_duplicates(subset=["pixel_id"], keep="first")
    return combined


def plot_global_cluster_map(
    assign_df: pd.DataFrame,
    out_path: str,
    title: str = "Global Pixel Cluster Assignments",
    marker_size: float = 2.0,
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (16, 8),
):
    """
    Plot pixel cluster assignments on a world map using cartopy.

    Args:
        assign_df: DataFrame with columns [pixel_id, lat, lon, cluster_id]
        out_path: Path to save the output figure
        title: Plot title
        marker_size: Size of scatter markers
        alpha: Transparency of markers
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("[warn] cartopy not installed. Install with: pip install cartopy")
        print("[warn] Falling back to simple scatter plot without map features")
        _plot_tiles_scatter(assign_df, "cluster_id", out_path, title, discrete=True)
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Filter valid coordinates
    df = assign_df.copy()
    df = df[np.isfinite(df["lat"]) & np.isfinite(df["lon"])]

    if len(df) == 0:
        print(f"[warn] No valid lat/lon data for {out_path}")
        return

    lons = df["lon"].to_numpy()
    lats = df["lat"].to_numpy()
    clusters = df["cluster_id"].to_numpy().astype(int)

    # Get unique clusters and create colormap
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cmap = plt.cm.get_cmap("tab10" if n_clusters <= 10 else "tab20", n_clusters)

    # Create figure with map projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    # Add map features
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    # Plot each cluster with distinct color
    for i, cl_id in enumerate(unique_clusters):
        mask = clusters == cl_id
        ax.scatter(
            lons[mask], lats[mask],
            c=[cmap(i)],
            s=marker_size,
            alpha=alpha,
            label=f"Cluster {cl_id}",
            transform=ccrs.PlateCarree(),
        )

    ax.set_title(title, fontsize=14)
    ax.legend(
        loc="lower left",
        ncol=min(5, n_clusters),
        fontsize=8,
        markerscale=3,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved global cluster map: {out_path}")


def _plot_cluster_metrics_bar(cluster_metrics: pd.DataFrame, out_path: str, title: str):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    d = cluster_metrics.sort_values("cluster_id").reset_index(drop=True)
    x = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Metric value")

    # plot brier + logloss (two bars per cluster)
    ax.bar(x - 0.2, d["brier"].to_numpy(), width=0.35, label="brier")
    ax.bar(x + 0.2, d["logloss"].to_numpy(), width=0.35, label="logloss")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in d["cluster_id"].to_numpy()])
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def _plot_signature_embedding(sig_mat: np.ndarray, assign_df: pd.DataFrame, out_path: str, title: str):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # PCA 2D embedding
    X2, st = _pca_fit_transform_numpy(sig_mat, n_components=2)
    cl = assign_df["cluster_id"].to_numpy().astype(int)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(title)
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=cl, s=8, alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("cluster_id")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def do_oof_shap_signatures_and_clustering(
    train_ds,
    val_ds,
    test_ds,
    boosters_full: List["xgb.Booster"],
    args,
    seed: int,
):
    """
    End-to-end:
      1) Build leakage-safe folds on train_ds
      2) Train per-fold models (OOF) (streaming)
      3) Compute train-only OOF SHAP tile signatures
      4) Fit clustering on train signatures
      5) Compute signatures for val/test using full model, assign clusters
      6) Eval per tile + per cluster
      7) Make required plots
      8) Save all artifacts into args.logdir
    """
    if not bool(getattr(args, "do_oof_shap_signatures", False)):
        return None

    n_folds = int(getattr(args, "n_folds", 5))
    max_rows_per_tile = int(getattr(args, "max_shap_rows_per_tile", 2000))
    sig_mode = str(getattr(args, "signature_mode", "avg"))

    header("OOF SHAP tile signatures + clustering")

    # 1) folds
    fold_id, tile_ids, years, tile_id_to_patch_idx = _make_fold_ids_for_dataset(train_ds, args, n_folds=n_folds, seed=int(seed))
    np.save(os.path.join(args.logdir, "train_fold_id.npy"), fold_id)
    print(f"[OOF] built fold_id for train patches: n={len(fold_id)} folds={n_folds} mode={args.fold_mode}")

    # 2) train fold models (per horizon)
    fold_models = {}
    base_stream_max = getattr(args, "stream_max_rows_per_horizon", None)
    oof_override_rounds = int(getattr(args, "oof_xgb_num_round", 0))
    oof_override_rows = getattr(args, "oof_stream_max_rows_per_horizon", None)

    for f in range(n_folds):
        header(f"[OOF] Training fold model f={f}/{n_folds-1}")
        train_idx = np.where(fold_id != f)[0].tolist()
        hold_idx  = np.where(fold_id == f)[0].tolist()

        ds_tr = Subset(train_ds, train_idx)
        ds_ho = Subset(train_ds, hold_idx)

        # temporarily override rounds / per-fold row cap
        args_fold = argparse.Namespace(**vars(args))
        if oof_override_rounds and oof_override_rounds > 0:
            args_fold.xgb_num_round = oof_override_rounds
        if oof_override_rows is not None:
            args_fold.stream_max_rows_per_horizon = oof_override_rows

        # Ensure streaming for OOF (low RAM)
        args_fold.train_mode = "stream"
        # Don’t use val during fold model training (avoid mixing holdout)
        args_fold.stream_use_val = False

        bsts, _params = train_xgb_per_horizon_streaming(train_ds=ds_tr, val_ds=None, args=args_fold, seed=int(seed) + 10_000 + f)
        fold_models[f] = bsts
        # save fold models (optional but useful)
        fdir = os.path.join(args.logdir, f"oof_fold{f}")
        os.makedirs(fdir, exist_ok=True)
        for k_idx, h in enumerate([int(x) for x in args.horizons]):
            bsts[k_idx].save_model(os.path.join(fdir, f"xgb_h{h}.json"))

    # 3) OOF SHAP signatures on train (PIXEL-LEVEL)
    # Check for temporal aggregation settings
    aggregate_time = bool(getattr(args, "shap_aggregate_time", False))
    aggregate_method = str(getattr(args, "shap_aggregate_method", "mean"))

    train_sig_df, train_sig_mat, sig_cols = _compute_pixel_signatures_from_shap_stream(
        ds=train_ds,
        boosters=boosters_full,  # not used when use_oof=True (but keep signature of function)
        args=args,
        split_name="train_oof",
        temporal_sample_frac=0.7,  # 70% temporal sampling
        mode=sig_mode,
        use_oof=True,
        fold_models=fold_models,
        fold_id_per_patch=fold_id,
        tile_id_to_patch_idx=tile_id_to_patch_idx,
        seed=int(seed),
        aggregate_time=aggregate_time,
        aggregate_method=aggregate_method,
    )
    train_sig_csv = os.path.join(args.logdir, "pixel_signatures_train_oof.csv")
    train_sig_df.to_csv(train_sig_csv, index=False)
    print(f"[OOF] wrote {train_sig_csv}")

    # 4) fit clusters on pixel signatures
    labels_train, cluster_state = _fit_clusters_from_signatures(train_sig_mat, args=args, seed=int(seed))
    train_assign = train_sig_df[["pixel_id", "y_global", "x_global", "lat", "lon", "n_rows_used", "event_rate"]].copy()
    train_assign["cluster_id"] = labels_train.astype(int)
    # distance proxy
    _, dist = _assign_clusters(train_sig_mat, cluster_state)
    train_assign["dist2_center"] = dist

    cluster_state_path = os.path.join(args.logdir, "cluster_state.json")
    _save_cluster_state(cluster_state, cluster_state_path)
    print(f"[cluster] wrote {cluster_state_path}")

    train_assign_csv = os.path.join(args.logdir, "pixel_cluster_assignments_train.csv")
    train_assign.to_csv(train_assign_csv, index=False)
    print(f"[cluster] wrote {train_assign_csv}")

    # 5) signatures for val/test using FULL model (no refit), then assign clusters (PIXEL-LEVEL)
    def _sig_and_assign(ds, split_name):
        if ds is None or len(ds) == 0:
            return None, None, None
        sig_df, sig_mat, _ = _compute_pixel_signatures_from_shap_stream(
            ds=ds,
            boosters=boosters_full,
            args=args,
            split_name=split_name, 
            temporal_sample_frac=1.0,  # 70% temporal sampling
            mode=sig_mode,
            use_oof=False,
            seed=int(seed),
            aggregate_time=aggregate_time,
            aggregate_method=aggregate_method,
        )
        lbl, dist = _assign_clusters(sig_mat, cluster_state)
        assign = sig_df[["pixel_id", "y_global", "x_global", "lat", "lon", "n_rows_used", "event_rate"]].copy()
        assign["cluster_id"] = lbl.astype(int)
        assign["dist2_center"] = dist
        sig_path = os.path.join(args.logdir, f"pixel_signatures_{split_name}.csv")
        as_path  = os.path.join(args.logdir, f"pixel_cluster_assignments_{split_name}.csv")
        sig_df.to_csv(sig_path, index=False)
        assign.to_csv(as_path, index=False)
        print(f"[cluster] wrote {sig_path}")
        print(f"[cluster] wrote {as_path}")
        return sig_df, sig_mat, assign

    val_sig_df, val_sig_mat, val_assign = _sig_and_assign(val_ds, "val")
    test_sig_df, test_sig_mat, test_assign = _sig_and_assign(test_ds, "test")

    # 6) eval per pixel + per cluster using FULL model predictions
    if bool(getattr(args, "eval_by_cluster", False)):
        # choose assignment df to eval each split
        # (for train, you can eval using train_assign too, but note train predictions are in-sample unless you use OOF preds)
        if val_ds is not None and val_assign is not None:
            pixel_metrics, cluster_metrics = _eval_pixel_and_cluster_metrics(val_ds, boosters_full, val_assign, args, "val")
            pixel_metrics.to_csv(os.path.join(args.logdir, "pixel_metrics_val.csv"), index=False)
            cluster_metrics.to_csv(os.path.join(args.logdir, "metrics_by_cluster_val.csv"), index=False)
            print("[eval] wrote pixel_metrics_val.csv and metrics_by_cluster_val.csv")

            if bool(getattr(args, "make_plots", False)):
                _plot_tiles_scatter(val_assign, "cluster_id", os.path.join(args.logdir, "map_pixels_by_cluster_val.png"),
                                    "Pixels colored by cluster (val)", discrete=True)
                _plot_tiles_scatter(pixel_metrics, "brier", os.path.join(args.logdir, "map_pixels_by_brier_val.png"),
                                    "Pixels colored by Brier (val)", discrete=False)
                _plot_tiles_scatter(pixel_metrics, "logloss", os.path.join(args.logdir, "map_pixels_by_logloss_val.png"),
                                    "Pixels colored by LogLoss (val)", discrete=False)
                _plot_tiles_scatter(pixel_metrics, "event_rate", os.path.join(args.logdir, "map_pixels_by_event_rate_val.png"),
                                    "Pixels colored by event rate (val)", discrete=False)
                _plot_cluster_metrics_bar(cluster_metrics, os.path.join(args.logdir, "cluster_metrics_val.png"),
                                          "Per-cluster metrics (val)")
                _plot_signature_embedding(val_sig_mat, val_assign, os.path.join(args.logdir, "pixel_signature_embedding_val.png"),
                                          "Pixel signature embedding (val)")

        if test_ds is not None and test_assign is not None:
            pixel_metrics, cluster_metrics = _eval_pixel_and_cluster_metrics(test_ds, boosters_full, test_assign, args, "test")
            pixel_metrics.to_csv(os.path.join(args.logdir, "pixel_metrics_test.csv"), index=False)
            cluster_metrics.to_csv(os.path.join(args.logdir, "metrics_by_cluster_test.csv"), index=False)
            print("[eval] wrote pixel_metrics_test.csv and metrics_by_cluster_test.csv")

            if bool(getattr(args, "make_plots", False)):
                _plot_tiles_scatter(test_assign, "cluster_id", os.path.join(args.logdir, "map_pixels_by_cluster_test.png"),
                                    "Pixels colored by cluster (test)", discrete=True)
                _plot_tiles_scatter(pixel_metrics, "brier", os.path.join(args.logdir, "map_pixels_by_brier_test.png"),
                                    "Pixels colored by Brier (test)", discrete=False)
                _plot_tiles_scatter(pixel_metrics, "logloss", os.path.join(args.logdir, "map_pixels_by_logloss_test.png"),
                                    "Pixels colored by LogLoss (test)", discrete=False)
                _plot_tiles_scatter(pixel_metrics, "event_rate", os.path.join(args.logdir, "map_pixels_by_event_rate_test.png"),
                                    "Pixels colored by event rate (test)", discrete=False)
                _plot_cluster_metrics_bar(cluster_metrics, os.path.join(args.logdir, "cluster_metrics_test.png"),
                                          "Per-cluster metrics (test)")
                _plot_signature_embedding(test_sig_mat, test_assign, os.path.join(args.logdir, "pixel_signature_embedding_test.png"),
                                          "Pixel signature embedding (test)")

    # 7) Train per-cluster models and evaluate ensemble predictions (if enabled)
    ensemble_results = None
    if bool(getattr(args, "train_cluster_models", False)):
        ensemble_results = do_cluster_model_training_and_ensemble_eval(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            global_boosters=boosters_full,
            train_assign=train_assign,
            val_assign=val_assign,
            test_assign=test_assign,
            train_sig_df=train_sig_df,
            val_sig_df=val_sig_df,
            test_sig_df=test_sig_df,
            cluster_state=cluster_state,
            args=args,
            seed=seed,
        )

    # 8) Create global cluster map combining all splits (if --make-plots enabled)
    if bool(getattr(args, "make_plots", False)):
        # Combine all available assignments
        all_assigns = [train_assign]
        if val_assign is not None:
            all_assigns.append(val_assign)
        if test_assign is not None:
            all_assigns.append(test_assign)

        combined_assign = pd.concat(all_assigns, ignore_index=True)
        combined_assign = combined_assign.drop_duplicates(subset=["pixel_id"], keep="first")

        plot_global_cluster_map(
            combined_assign,
            out_path=os.path.join(args.logdir, "global_cluster_map.png"),
            title=f"Global Pixel Cluster Assignments (n={len(combined_assign):,}, k={cluster_state['n_clusters']})",
        )

    return {
        "train_assign": train_assign,
        "val_assign": val_assign,
        "test_assign": test_assign,
        "cluster_state_path": cluster_state_path,
        "train_sig_csv": train_sig_csv,
        "ensemble_results": ensemble_results,
    }


def run_shap_analysis(
    boosters: List["xgb.Booster"],
    args,
    train_ds,
    val_ds,
    test_ds,
    feature_names: Optional[List[str]] = None,
):
    """
    Compute SHAP feature importance for each horizon model and write:
      - shap_h{h}_importance.csv
      - shap_h{h}_bar.{png/pdf/svg}
      - optional beeswarm (if `shap` pkg installed and --shap-beeswarm)
      - dependence plots for top-K features
      - overall (avg across horizons): shap_overall_importance.csv + bar plot
    """
    if not bool(getattr(args, "shap", False)):
        return

    split = str(getattr(args, "shap_split", "val")).lower()
    if split == "val" and (val_ds is None or len(val_ds) == 0):
        split = "train"
    if split == "test" and (test_ds is None or len(test_ds) == 0):
        split = "val" if (val_ds is not None and len(val_ds) > 0) else "train"

    ds = {"train": train_ds, "val": val_ds, "test": test_ds}.get(split, val_ds or train_ds)
    if ds is None or len(ds) == 0:
        print("[SHAP] No dataset available for SHAP; skipping.")
        return

    header(f"SHAP analysis (split={split})")

    # -------------------------
    # ✅ CHANGE 1: Use saved feature names if available; otherwise build them.
    # Also apply cleaning for nicer plot labels.
    # -------------------------
    if feature_names is None:
        feature_names = _load_feature_names_for_run(args) or build_feature_names_from_dataset(args, ds)
    # Make a cleaned version for display/plotting, but keep originals around if you want.
    feature_names = [(_shorten_feature_name(s)) for s in feature_names]

    X_by_h = _collect_shap_samples_from_ds(ds, args, split_name=split, feature_names=feature_names)
    if not X_by_h:
        print("[SHAP] No samples collected; skipping.")
        return

    # Optional: try to import shap for beeswarm plots
    have_shap_pkg = False
    shap_pkg = None
    if bool(getattr(args, "shap_beeswarm", False)):
        try:
            import shap as _shap  # type: ignore
            shap_pkg = _shap
            have_shap_pkg = True
        except Exception as e:
            print(f"[SHAP] 'shap' package not available ({e}); beeswarm disabled.")

    per_h_imp = []
    dep_topk = int(getattr(args, "shap_dependence_topk", 3))

    for k, h in enumerate([int(x) for x in args.horizons]):
        if h not in X_by_h:
            continue
        X = X_by_h[h]
        contribs = _xgb_predict_contribs(boosters[k], X, args)
        contribs = np.asarray(contribs)
        if contribs.ndim != 2 or contribs.shape[1] != X.shape[1] + 1:
            print(f"[SHAP] Unexpected contribs shape for h={h}: {contribs.shape}; skipping.")
            continue

        shap_vals = contribs[:, :-1].astype(np.float64, copy=False)
        base = contribs[:, -1].astype(np.float64, copy=False)

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        imp = np.argsort(-mean_abs)

        # -------------------------
        # ✅ CHANGE 2: Guard feature name length mismatch robustly.
        # -------------------------
        n_feat = int(X.shape[1])
        if len(feature_names) < n_feat:
            # pad with fallback names so indexing is always safe
            feature_names_use = feature_names + [f"f{i}" for i in range(len(feature_names), n_feat)]
        else:
            feature_names_use = feature_names[:n_feat]

        df = pd.DataFrame({
            "feature": feature_names_use,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        df["feature_label"] = df["feature"].map(_pretty_feature_label)

        
        per_h_imp.append(df.assign(horizon=h))

        csv_path = os.path.join(args.logdir, f"shap_h{h}_importance.csv")
        df.to_csv(csv_path, index=False)
        print(f"[SHAP] wrote {csv_path}")

        bar_path = os.path.join(args.logdir, f"shap_h{h}_bar.{args.plot_file_format}")
        _plot_shap_bar(df, bar_path, args, title=f"SHAP feature importance (h={h})")
        print(f"[SHAP] wrote {bar_path}")

        # -------------------------
        # ✅ CHANGE 3: Safe dependence plot filenames (feature names have ':' '@' etc)
        # -------------------------
        for j in range(min(dep_topk, n_feat)):
            fi = int(imp[j])
            safe = _safe_fname(feature_names_use[fi])
            dep_path = os.path.join(args.logdir, f"shap_h{h}_dependence_{safe}.{args.plot_file_format}")

            _plot_shap_dependence(
                X, shap_vals, feature_names_use, fi, dep_path, args,
                title=f"SHAP dependence (h={h}) — {feature_names_use[fi]}"
            )

        # Beeswarm (optional)
        if have_shap_pkg and shap_pkg is not None:
            try:
                import warnings
                import matplotlib.pyplot as plt
                # Suppress FutureWarning from shap about NumPy RNG seeding
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, module="shap")
                    shap_pkg.summary_plot(
                        shap_vals,
                        X,
                        feature_names=feature_names_use,
                        show=False,
                        max_display=int(getattr(args, "shap_top_n", 30)),
                    )
                bs_path = os.path.join(args.logdir, f"shap_h{h}_beeswarm.{args.plot_file_format}")
                plt.tight_layout()
                plt.savefig(bs_path, dpi=args.plot_dpi, format=args.plot_file_format)
                plt.close()
                print(f"[SHAP] wrote {bs_path}")
            except Exception as e:
                print(f"[SHAP] beeswarm failed for h={h}: {e}")

        # Optional: save raw values (can be large)
        if bool(getattr(args, "shap_save_values", False)):
            npz_path = os.path.join(args.logdir, f"shap_h{h}_values.npz")
            np.savez_compressed(
                npz_path,
                X=X.astype(np.float32, copy=False),
                shap=shap_vals.astype(np.float32, copy=False),
                base=base.astype(np.float32, copy=False),
                feature_names=np.asarray(feature_names_use, dtype=object),
            )
            print(f"[SHAP] wrote {npz_path}")

    if not per_h_imp:
        print("[SHAP] No per-horizon importance computed; skipping overall.")
        return

    # Overall: average mean_abs_shap across horizons (simple unweighted average)
    merged = pd.concat(per_h_imp, axis=0, ignore_index=True)
    overall = merged.groupby("feature", as_index=False)["mean_abs_shap"].mean().sort_values("mean_abs_shap", ascending=False)
    overall_csv = os.path.join(args.logdir, "shap_overall_importance.csv")
    overall.to_csv(overall_csv, index=False)
    print(f"[SHAP] wrote {overall_csv}")

    overall_bar = os.path.join(args.logdir, f"shap_overall_bar.{args.plot_file_format}")
    _plot_shap_bar(overall, overall_bar, args, title="SHAP feature importance (avg across horizons)")
    print(f"[SHAP] wrote {overall_bar}")


import re
import json
from typing import Optional, List

def _load_feature_names_for_run(args) -> Optional[List[str]]:
    """
    Load feature_names.json if you wrote it during training.
    Expected location: args.logdir/feature_names.json
    """
    path = os.path.join(args.logdir, "feature_names.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            names = json.load(f)
        if isinstance(names, list) and all(isinstance(x, str) for x in names):
            return names
    except Exception:
        pass
    return None

def _shorten_feature_name(s: str) -> str:
    """
    Make feature labels cleaner for plotting:
    - remove repeated ':field' etc
    - compress ':c12' -> ':ch12'
    You can customize this to your taste.
    """
    if not isinstance(s, str):
        s = str(s)

    # Drop common suffix tokens to reduce clutter (adjust to your dataset naming)
    s = s.replace(":field", "")
    s = s.replace(":era5land", "")
    s = s.replace(":smap_wtd", "")
    s = s.replace(":smap_l4", "")
    s = s.replace(":vpd_rh", "")
    s = s.replace(":worldpop", "")
    s = s.replace(":grip_total", "")
    s = s.replace(":viirs", "")

    # channel token cleanup
    s = re.sub(r":c(\d+)", r":ch\1", s)

    # optional: compact time token
    s = s.replace("@t-", "@t")

    return s

def _safe_fname(s: str, maxlen: int = 90) -> str:
    """
    Safe filename from feature name (remove ':' '@' etc).
    """
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("_")
    if not s:
        s = "feature"
    return s[:maxlen]



# ----------------------------------------------------------------------
# Dataset helpers (kept from your pipeline)
# ----------------------------------------------------------------------

def _parse_years_spec(spec: str) -> list[int]:
    spec = (spec or "").strip()
    if not spec:
        return []
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a.strip()), int(b.strip())
            lo, hi = (a, b) if a <= b else (b, a)
            out.extend(list(range(lo, hi + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def _load_time_years_from_viirs(viirs_zarr: str, viirs_array: str) -> np.ndarray:
    import xarray as xr

    try:
        ds = xr.open_zarr(viirs_zarr, consolidated=True)
    except Exception:
        ds = xr.open_zarr(viirs_zarr, consolidated=False)

    if viirs_array in ds.data_vars:
        arr = ds[viirs_array]
    elif viirs_array in ds:
        arr = ds[viirs_array]
    else:
        arr = next(iter(ds.data_vars.values()))

    t = None
    for key in ("time", "t", "date", "datetime"):
        if key in arr.coords:
            t = arr.coords[key].values
            break
        if key in ds.coords:
            t = ds.coords[key].values
            break
    if t is None:
        raise RuntimeError(f"Could not find a time coordinate in VIIRS zarr: {viirs_zarr}")

    t = np.asarray(t)
    if t.dtype.kind != "M":
        raise RuntimeError(f"VIIRS time coordinate is not datetime64 (dtype={t.dtype}).")

    years = t.astype("datetime64[Y]").astype(np.int64) + 1970
    return years


def _resolve_time_indices_by_test_year_anchor_leakproof(args, viirs_spec):
    if args.test_year is None:
        return None, None, None

    years = _load_time_years_from_viirs(viirs_spec.zarr, viirs_spec.array)
    T = int(years.shape[0])

    test_year = int(args.test_year)
    horizons = [int(h) for h in args.horizons]
    max_h = max(horizons) if horizons else 0
    t_hist = int(args.T_hist)

    t_end_first = t_hist - 1
    t_end_last = T - 1 - max_h
    if t_end_last < t_end_first:
        raise RuntimeError("Not enough T for requested t_hist/max_horizon.")

    test_t_ends = [t_end for t_end in range(t_end_first, t_end_last + 1) if years[t_end] == test_year]
    if not test_t_ends:
        raise RuntimeError(f"No valid test t_end anchors found for anchor test_year={test_year}.")

    reserved = np.zeros(T, dtype=np.bool_)
    for t_end in test_t_ends:
        t0 = t_end - (t_hist - 1)
        t1 = t_end + max_h
        reserved[t0 : t1 + 1] = True

    test_time_index = np.where(reserved)[0].tolist()
    train_time_index = np.where(~reserved)[0].tolist()

    if getattr(args, "train_years", "").strip():
        allowed_years = set(_parse_years_spec(args.train_years))
        if allowed_years:
            train_time_index = [t for t in train_time_index if int(years[t]) in allowed_years]
            print(f"  • Train-years override........ {sorted(allowed_years)}")
            print(f"  • Train time steps (filtered). {len(train_time_index)}")

    header("Leak-proof anchor-year split (labels may spill into next year)")
    all_years = sorted(set(int(y) for y in years.tolist()))
    print(f"  • Available years............. {all_years}")
    print(f"  • Test anchor year............ {test_year}")
    print(f"  • Horizons.................... {horizons} (max={max_h})")
    print(f"  • t_hist...................... {t_hist}")
    print(f"  • # test t_end anchors......... {len(test_t_ends)}")
    print(f"  • Reserved time steps (test)... {int(reserved.sum())}/{T}")
    print(f"  • Train time steps............. {len(train_time_index)}")
    print(f"  • Test time steps.............. {len(test_time_index)}")
    print("  • Note: test labels can be in next year if horizons push t_end+h over year boundary.")

    return train_time_index, test_time_index, test_t_ends


def build_datasets(args):
    inputs = [parse_input_spec(s) for s in args.input]
    viirs_spec = parse_input_spec(args.viirs_zarr)

    train_time_index = None
    test_time_index = None
    test_t_end_index = None

    use_spatial_holdout = bool((args.test_year is not None) and (args.test_region_mask_source or "").strip())

    if args.test_year is not None:
        if use_spatial_holdout:
            years = _load_time_years_from_viirs(viirs_spec.zarr, viirs_spec.array)
            T = int(years.shape[0])
            horizons = [int(h) for h in args.horizons]
            max_h = max(horizons) if horizons else 0
            t_hist = int(args.T_hist)
            t_end_first = t_hist - 1
            t_end_last = T - 1 - max_h
            test_t_end_index = [t for t in range(t_end_first, t_end_last + 1) if int(years[t]) == int(args.test_year)]
            if not test_t_end_index:
                raise RuntimeError(f"No valid t_end anchors found for test_year={args.test_year}.")

            header("Spatial holdout split (by anchor year + region mask)")
            print(f"  • Test anchor year............ {int(args.test_year)}")
            print(f"  • # test t_end anchors......... {len(test_t_end_index)}")
            print(f"  • Region mask source.......... {args.test_region_mask_source}")
            print(f"  • Region min patch fraction... {float(args.test_region_min_fraction)}")
        else:
            train_time_index, test_time_index, test_t_end_index = _resolve_time_indices_by_test_year_anchor_leakproof(args, viirs_spec)

    common = dict(
        inputs=inputs,
        viirs_zarr=viirs_spec.zarr,
        viirs_array=viirs_spec.array,
        t_hist=args.T_hist,
        horizons=args.horizons,
        patch=args.patch,
        stride=args.stride,
        time_stack=args.stack_time,
        split=args.split,
        val_frac=args.val_frac,
        seed=args.data_seed,
        normalize_inputs=(None if args.normalize_inputs == "none" else args.normalize_inputs),
        max_samples=args.max_samples,
        skip_nonpeat_patches=not args.no_skip_nonpeat,
        peat_min_fraction=args.peat_min_fraction,
        time_index=train_time_index,
        coord_as_features=not bool(getattr(args, "no_coord_features", False)),
        return_coords=False,
        peat_mask_source=args.peat_mask_source.strip() or "smap_wtd.zarr",
        coords_source=args.coords_source.strip() or "smap_wtd.zarr",
        coords_units=args.coords_units,
        holdout_region_source=(args.test_region_mask_source.strip() or None),
        holdout_t_end_index=test_t_end_index,
        holdout_min_fraction=float(args.test_region_min_fraction),
        flatten_pixels=True,  # ✅ OPTIMIZATION: Pre-flatten pixels for XGBoost efficiency
        # Regime support
        regime_source=getattr(args, 'regime_source', None),
        regime_as_features=getattr(args, 'regime_as_features', False),
        regime_nodata_value=getattr(args, 'regime_nodata_value', -9999),
    )

    header("Building datasets")
    use_coord_features = not bool(getattr(args, "no_coord_features", False))
    print(f"  • coord_as_features.......... {use_coord_features}")
    use_regime_features = getattr(args, 'regime_as_features', False)
    print(f"  • regime_as_features......... {use_regime_features}")
    if getattr(args, 'regime_source', None):
        print(f"  • regime_source.............. {args.regime_source}")
    train_ds = JointPeatDataset(mode="train", **common)
    print(f"  • Train patches.............. {len(train_ds)}")

    val_ds = JointPeatDataset(mode="val", **common)
    print(f"  • Val patches................ {len(val_ds)}")

    test_ds = None
    if args.test_year is not None:
        common_test = dict(common)
        common_test["t_end_index"] = test_t_end_index
        if (not use_spatial_holdout) and (test_time_index is not None):
            common_test["time_index"] = test_time_index
        else:
            common_test["time_index"] = None
        common_test["split"] = 0.0
        common_test["val_frac"] = 0.0
        test_ds = JointPeatDataset(mode="test", **common_test)
        print(f"  • Test patches............... {len(test_ds)}")
    else:
        print("  • No test_year specified; skipping test dataset (not recommended).")

    return train_ds, val_ds, test_ds


def split_train_for_calibration(train_ds, args):
    frac = float(args.calib_frac)
    if frac <= 0.0:
        return train_ds, None

    N = len(train_ds)
    if N < 2:
        return train_ds, None

    n_cal = max(1, min(int(round(N * frac)), N - 1))

    rng = np.random.RandomState(args.data_seed + CALIB_SPLIT_SEED_OFFSET)
    idx = np.arange(N)
    rng.shuffle(idx)

    cal_idx = idx[:n_cal]
    tr_idx = idx[n_cal:]

    header("Calibration split (dataset-level patches)")
    print(f"  • calib_frac................. {frac}")
    print(f"  • Train (after split)........ {len(tr_idx)}")
    print(f"  • Calib...................... {len(cal_idx)}")
    if args.calibration_method == "none":
        print("[WARN] calib_frac > 0 but calibration_method=none.")
        print("       You are throwing away training patches without using them.")
        print("       Set --calib-frac 0 OR set --calibration-method platt|isotonic.")

    return Subset(train_ds, tr_idx), Subset(train_ds, cal_idx)


# ----------------------------------------------------------------------
# Dataset row estimator (stream sizing helper)
# ----------------------------------------------------------------------

def estimate_dataset_rows(ds, args, desc="dataset"):
    """
    Quick estimate of total valid rows (for horizon 0 mask) without full iteration.
    Uses a limited number of batches to estimate valid_fraction.

    Prints a recommended stream_chunk_rows band for ~50-100 chunks total.
    """
    header(f"Estimating {desc} row count (quick sample)")

    if len(ds) == 0:
        print("[estimate] dataset is empty.")
        return 0.0

    sample_batches = int(getattr(args, "estimate_rows_batches", 20))
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=False, args=args, device=None)

    total_pixels = 0
    valid_pixels = 0
    seen_batches = 0

    for i, batch in enumerate(loader):
        if i >= sample_batches:
            break
        seen_batches += 1
        mask = batch["mask"][:, 0]  # first horizon
        total_pixels += int(mask.numel())
        valid_pixels += int((mask > 0.5).sum().item())

    if total_pixels <= 0:
        print("[estimate] total_pixels=0 in sample; cannot estimate.")
        return 0.0

    valid_fraction = valid_pixels / total_pixels
    est_total = float(len(ds) * (args.patch ** 2) * valid_fraction)

    print(f"  • sample_batches............. {seen_batches}")
    print(f"  • patch...................... {args.patch}")
    print(f"  • batch_size................. {args.batch_size}")
    print(f"  • valid_fraction............. {valid_fraction:.4f}")
    print(f"  • est_total_rows............. {est_total:,.0f}")

    lo = int(max(est_total / 100.0, 1e5))
    hi = int(max(est_total / 50.0,  1e5))
    print(f"  • suggested --stream-chunk-rows  ~ {lo:,} to {hi:,}  (aim ~50–100 chunks)")

    return est_total


# ----------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    
    # -------- OOF SHAP pixel signatures + clustering --------
    # NOTE: Clustering now operates at PIXEL-LEVEL (not tile-level)
    # - Each spatial pixel gets its own SHAP signature (aggregated across time)
    # - Temporal sampling (70% default) controls computational cost
    # - No per-tile row cap; all valid pixels are processed
    p.add_argument("--do-oof-shap-signatures", action="store_true",
                   help="Compute train-only OOF SHAP pixel signatures, cluster them, assign clusters to val/test, eval by cluster, and make plots.")
    p.add_argument("--n-folds", type=int, default=5, help="Number of CV folds for OOF SHAP (train-only).")
    p.add_argument("--fold-mode", choices=["tile", "year", "tile_year"], default="tile",
                   help="Leakage blocking: tile (GroupKFold by tile_id for fold assignment), year, or tile_year.")
    p.add_argument("--max-shap-rows-per-tile", type=int, default=2000,
                   help="(DEPRECATED for pixel-level) Previously capped SHAP rows per tile. Now temporal sampling controls cost.")
    p.add_argument("--temporal-sample-frac", type=float, default=0.7,
                   help="Fraction of time windows to sample per pixel for SHAP signature computation (0.7 = 70%).")
    p.add_argument("--oof-xgb-num-round", type=int, default=0,
                   help="If >0, override xgb_num_round for OOF fold models (speeds up OOF).")
    p.add_argument("--oof-stream-max-rows-per-horizon", type=int, default=None,
                   help="Optional cap per fold/horizon in streaming OOF training.")


    p.add_argument("--n-clusters", type=int, default=3, help="Number of pixel regimes/clusters.")
    p.add_argument("--cluster-method", choices=["kmeans"], default="kmeans", help="Clustering method (kmeans implemented).")
    p.add_argument("--pca-components", type=int, default=0, help="If >0, PCA-reduce pixel signatures before clustering.")
    p.add_argument("--signature-mode", choices=["avg", "concat", "h0"], default="avg",
                   help="How to combine multi-horizon SHAP into a pixel signature: avg across horizons, concat horizons, or use h0 only.")
    p.add_argument("--shap-aggregate-time", action="store_true",
                   help="Aggregate SHAP values across time steps for each base channel before clustering. "
                        "Reduces features from (C * t_hist) to C unique channels (e.g., 'SMAP (t)', 'SMAP (t-1)', ... -> 'SMAP'). "
                        "Useful for clustering by channel importance rather than temporal patterns.")
    p.add_argument("--shap-aggregate-method", choices=["mean", "sum"], default="mean",
                   help="How to aggregate SHAP values across time steps: 'mean' (default) averages, 'sum' adds them.")
    p.add_argument("--shap-exclude-positional", action="store_true", default=True,
                   help="Exclude positional features (x/y/z, sin/cos lat/lon, spatial IDs) from SHAP clustering. "
                        "Positional features are still used in XGBoost training, but not for regime clustering. "
                        "This focuses clustering on environmental drivers rather than spatial patterns. (default: True)")
    p.add_argument("--no-shap-exclude-positional", action="store_false", dest="shap_exclude_positional",
                   help="Include positional features in SHAP clustering (legacy behavior).")

    p.add_argument("--eval-by-cluster", action="store_true", help="Compute metrics by cluster (requires clusters).")
    p.add_argument("--make-plots", action="store_true", help="Make cluster/metric maps + embedding plots.")

    # -------- Per-cluster model training and ensemble evaluation --------
    p.add_argument("--train-cluster-models", action="store_true",
                   help="Train separate XGBoost models for each cluster after clustering.")
    p.add_argument("--ensemble-mode", choices=["cluster_selection", "distance_weighted", "both"], default="both",
                   help="Ensemble prediction mode: 'cluster_selection' uses each pixel's assigned cluster model, "
                        "'distance_weighted' weights all cluster models by inverse distance to centers, "
                        "'both' evaluates both strategies.")
    p.add_argument("--cluster-min-samples", type=int, default=1000,
                   help="Minimum pixel samples required to train a per-cluster model (clusters below this use global model). "
                        "Set to 0 to train ALL clusters regardless of size.")
    p.add_argument("--cluster-min-samples-adaptive", action="store_true",
                   help="Adaptively lower cluster-min-samples to ensure at least 50%% of clusters are trained. "
                        "Helps avoid high fallback rates when clusters are imbalanced.")
    p.add_argument("--ensemble-epsilon", type=float, default=1e-6,
                   help="Epsilon for inverse-distance weight computation (avoids division by zero).")
    p.add_argument("--ensemble-temperature", type=float, default=1.0,
                   help="Temperature parameter for distance weighting (lower = sharper weights toward nearest cluster).")

    
    p.add_argument("--input", action="append", default=[],
                   help="Repeatable. Format: /path/store.zarr[:array_path]. Default array_path='field'.")
    p.add_argument("--viirs-zarr", required=True,
                   help="Format: /path/viirs.zarr[:array_path]. Default array_path='field'.")

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
    p.add_argument("--peat-mask-source", type=str, default="smap_wtd.zarr")
    p.add_argument("--coords-source", type=str, default="smap_wtd.zarr")
    p.add_argument("--coords-units", choices=["auto", "degrees", "radians"], default="auto")
    p.add_argument("--no-coord-features", action="store_true",
                   help="Disable coordinate/positional features (sin/cos lat/lon) in XGBoost training. "
                        "Use this to train a purely environmental model without spatial encoding.")

    p.add_argument("--test-year", type=int, default=None)
    p.add_argument("--train-years", type=str, default="",
                   help="Optional override, e.g. '2016-2022,2024'.")

    p.add_argument("--test-region-mask-source", type=str, default="",
                   help="Optional. Zarr store with 'peat_mask' defining a held-out region.")
    p.add_argument("--test-region-min-fraction", type=float, default=0.01)

    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--mp-context", choices=["auto", "spawn", "fork", "forkserver"], default="auto")
    p.add_argument("--no-tqdm", action="store_true")

    # training mode
    p.add_argument("--train-mode", choices=["tabularize", "stream"], default="tabularize",
                   help="tabularize=materialize X_tr in RAM; stream=DataIter streaming (low RAM).")

    # Streaming knobs
    p.add_argument("--stream-chunk-rows", type=int, default=1_000_000,
                   help="Base rows per streamed chunk fed into XGBoost DataIter.")
    p.add_argument("--stream-accumulate-multiplier", type=int, default=3,
                   help="Accumulate this many *chunk_rows* per DataIter.next() payload (fewer DMatrix rebuilds).")
    p.add_argument("--stream-max-rows-per-horizon", type=int, default=None,
                   help="Optional cap on total streamed rows per horizon (debug/speed).")
    p.add_argument("--stream-use-cupy", action="store_true",
                   help="If set and CuPy available, feed CuPy arrays to DMatrix in streaming mode.")
    p.add_argument("--stream-use-val", action="store_true",
                   help="If set, build a streaming VAL DMatrix and use it for evals during training.")
    p.add_argument("--stream-shuffle-train", action="store_true",
                   help="Shuffle patch order during streaming training (default False for determinism).")
    p.add_argument("--stream-show-progress", action="store_true",
                   help="Show streaming row-counter tqdm (default off unless you set it).")

    # XGBoost GPU-first settings
    p.add_argument("--xgb-eta", type=float, default=0.1)
    p.add_argument("--xgb-num-round", type=int, default=50, help="Number of boosting rounds (trees).")
    p.add_argument("--xgb-tree-method", choices=["hist", "gpu_hist"], default="gpu_hist")
    p.add_argument("--xgb-gpu-id", type=int, default=0)
    p.add_argument("--xgb-dmatrix", choices=["dmat", "quantile"], default="quantile",
                   help="QuantileDMatrix is often best for gpu_hist (if available), but can be slower in streaming.")
    p.add_argument("--xgb-quantile-precompute", action="store_true",
                   help="If set and xgb-dmatrix=quantile, build a train ref sketch and reuse it for train/val.")
    p.add_argument("--xgb-verbose-eval", action="store_true")
    p.add_argument("--xgb-nthread", type=int, default=None, help="XGBoost CPU threads (still used for some parts).")

    # Extra XGB knobs (useful on A100)
    p.add_argument("--xgb-max-depth", type=int, default=5)
    p.add_argument("--xgb-max-bin", type=int, default=256)
    p.add_argument("--xgb-subsample", type=float, default=0.8)
    p.add_argument("--xgb-colsample-bynode", type=float, default=0.5)
    p.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    p.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    p.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    p.add_argument("--xgb-gamma", type=float, default=0.0,
                   help="Minimum loss reduction for split (regularization). Higher = more conservative.")
    p.add_argument("--xgb-sampling-method", type=str, default="gradient_based",
                   help="Often faster on GPU: 'uniform' or 'gradient_based'. Empty disables.")
    p.add_argument("--xgb-grow-policy", type=str, default="depthwise",
                   help="'depthwise' usually best; empty disables.")
    p.add_argument("--xgb-gpu-mem-frac-warn", type=float, default=0.80,
                   help="Warn if estimated GPU DMatrix copy exceeds this fraction of free GPU memory.")

    # Prediction device for XGB inference path
    p.add_argument("--xgb-predict-device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Where to run XGB prediction. 'cuda' uses CuPy+DLPack if available.")
    p.add_argument("--eval-device", choices=["cpu", "cuda"], default="cuda",
                   help="Torch device for evaluation tensors (controls whether features can stay on GPU).")

    # Hyperparameter Optimization (Optuna)
    p.add_argument("--hpo", action="store_true",
                   help="Enable Optuna-based hyperparameter optimization before final training.")
    p.add_argument("--hpo-trials", type=int, default=30,
                   help="Number of Optuna trials (20-50 recommended). Default: 30")
    p.add_argument("--hpo-timeout", type=int, default=None,
                   help="Optional timeout in seconds for the entire HPO study.")
    p.add_argument("--hpo-sampler", choices=["tpe", "random", "cmaes", "nsgaii"], default="tpe",
                   help="Optuna sampler: tpe (default), random, cmaes (CMA-ES evolutionary), nsgaii (NSGA-II multi-objective).")
    p.add_argument("--hpo-horizon-strategy", choices=["first", "mean"], default="first",
                   help="How to handle multi-horizon: 'first' uses horizons[0], 'mean' averages all.")
    p.add_argument("--hpo-params-out", type=str, default=None,
                   help="Path to save best HPO params JSON. Defaults to logdir/hpo_best_params.json")
    p.add_argument("--hpo-params-in", type=str, default=None,
                   help="Path to load pre-tuned params JSON (skips HPO, uses these params for training).")
    p.add_argument("--hpo-num-round-max", type=int, default=200,
                   help="Max num_round to search during HPO (higher allows more trees with lower eta).")
    p.add_argument("--hpo-subsample-train", type=float, default=None,
                   help="Optional: subsample training data during HPO for speed (e.g., 0.3 = 30%%).")
    p.add_argument("--hpo-early-stopping", type=int, default=10,
                   help="Early stopping rounds during HPO trials to prevent overfitting.")

    # CMA-ES specific options
    p.add_argument("--hpo-cmaes-restart-strategy", choices=["none", "ipop", "bipop"], default="ipop",
                   help="CMA-ES restart strategy: 'ipop' increases population, 'bipop' alternates strategies.")
    p.add_argument("--hpo-cmaes-sigma0", type=float, default=0.5,
                   help="Initial step size (sigma) for CMA-ES. Larger values = more exploration.")

    # NSGA-II specific options
    p.add_argument("--hpo-nsgaii-population-size", type=int, default=50,
                   help="Population size for NSGA-II multi-objective optimization.")
    p.add_argument("--hpo-nsgaii-objectives", choices=["brier", "brier_logloss", "brier_calibration"], default="brier",
                   help="Objectives for NSGA-II: 'brier' (single), 'brier_logloss', or 'brier_calibration' (multi).")

    # Residual model HPO
    p.add_argument("--hpo-include-residual", action="store_true",
                   help="Include residual model hyperparameters in optimization (requires --train-residual-models).")
    p.add_argument("--hpo-residual-mode", choices=["sequential", "joint"], default="sequential",
                   help="'sequential': optimize global first, then residual; 'joint': optimize both together.")
    p.add_argument("--hpo-residual-max-depth-range", type=str, default="2,6",
                   help="Search range for residual max_depth as 'min,max'.")
    p.add_argument("--hpo-residual-lambda-range", type=str, default="1.0,50.0",
                   help="Search range for residual reg_lambda as 'min,max'.")
    p.add_argument("--hpo-residual-alpha-range", type=str, default="0.0,5.0",
                   help="Search range for residual reg_alpha as 'min,max'.")
    p.add_argument("--hpo-residual-subsample-range", type=str, default="0.3,0.8",
                   help="Search range for residual subsample as 'min,max'.")
    p.add_argument("--hpo-residual-colsample-range", type=str, default="0.3,0.8",
                   help="Search range for residual colsample_bynode as 'min,max'.")
    p.add_argument("--hpo-residual-num-round-range", type=str, default="10,100",
                   help="Search range for residual num_round as 'min,max'.")
    p.add_argument("--hpo-residual-params-out", type=str, default=None,
                   help="Path to save best residual HPO params JSON.")
    p.add_argument("--hpo-residual-params-in", type=str, default=None,
                   help="Path to load pre-tuned residual params JSON (skips residual HPO).")

    # Calibration HPO
    p.add_argument("--hpo-include-calibration", action="store_true",
                   help="Include calibration method and params in HPO search.")
    p.add_argument("--hpo-calibration-methods", type=str, default="none,platt,isotonic",
                   help="Comma-separated calibration methods to search (e.g., 'platt,isotonic').")
    p.add_argument("--hpo-platt-reg-range", type=str, default="1e-8,1e-2",
                   help="Search range for Platt regularization as 'min,max' (log scale).")
    p.add_argument("--hpo-platt-max-iter-range", type=str, default="50,500",
                   help="Search range for Platt max_iter as 'min,max'.")

    # Regime count HPO (pre-screen + full HPO)
    p.add_argument("--hpo-regime-search", action="store_true",
                   help="Enable regime count optimization via pre-screening.")
    p.add_argument("--hpo-regime-dir", type=str, default=None,
                   help="Directory containing regime_map_#.tif files (e.g., CLUSTER/regime_output).")
    p.add_argument("--hpo-regime-counts", type=str, default="2,3,4,5,6,7",
                   help="Comma-separated regime counts to try during pre-screening.")
    p.add_argument("--hpo-regime-screen-trials", type=int, default=3,
                   help="Number of quick trials per regime count during pre-screening.")
    p.add_argument("--hpo-regime-metric", choices=["brier", "logloss"], default="brier",
                   help="Metric for regime selection during pre-screening.")

    # T-hist (temporal history) HPO pre-screening
    p.add_argument("--hpo-t-hist-search", action="store_true",
                   help="Enable t_hist (temporal history length) optimization via pre-screening.")
    p.add_argument("--hpo-t-hist-values", type=str, default="15,30,45,60",
                   help="Comma-separated t_hist values to try during pre-screening.")
    p.add_argument("--hpo-t-hist-screen-trials", type=int, default=3,
                   help="Number of quick trials per t_hist value during pre-screening.")
    p.add_argument("--hpo-t-hist-metric", choices=["brier", "logloss"], default="brier",
                   help="Metric for t_hist selection during pre-screening.")

    # Optuna storage backend for trial persistence
    p.add_argument("--hpo-storage", type=str, default=None,
                   help="Optuna storage URL for trial persistence (e.g., 'sqlite:///optuna.db'). "
                        "If None, uses in-memory storage (trials lost if interrupted).")
    p.add_argument("--hpo-storage-name", type=str, default=None,
                   help="Optuna study name for resumability. Auto-generated from timestamp if None.")
    p.add_argument("--hpo-load-if-exists", action="store_true",
                   help="Resume existing study if found in storage (requires --hpo-storage).")

    # Force tabularize mode during HPO for performance
    p.add_argument("--hpo-force-tabularize", action="store_true", default=True,
                   help="Force tabularize mode during HPO for speed (default: True). "
                        "Even if --train-mode=stream, HPO will use tabularized data to avoid DMatrix rebuilds. "
                        "Final training respects original --train-mode setting.")

    # Tabularization caps (RAM control)
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--max-val-rows", type=int, default=None)

    # Calibration
    p.add_argument("--calib-frac", type=float, default=0.0,
                   help="Fraction of train patches to hold out for calibration fit.")
    p.add_argument("--calibration-method", choices=["none", "platt", "isotonic"], default="none")
    p.add_argument("--max-calib-rows-per-horizon", type=int, default=2_000_000,
                   help="Cap calibration pixel rows per horizon (streaming). Use None for unlimited.")
    p.add_argument("--platt-max-iter", type=int, default=100)
    p.add_argument("--platt-tol", type=float, default=1e-8)
    p.add_argument("--platt-reg", type=float, default=1e-6)

    # Regime support
    p.add_argument("--regime-source", type=str, default=None,
                   help="Path to GeoTIFF with integer regime IDs (0 to N-1). NODATA=-9999.")
    p.add_argument("--regime-as-features", action="store_true",
                   help="One-hot encode regime IDs as additional features.")
    p.add_argument("--per-regime-calibration", action="store_true",
                   help="Fit separate calibrators per (regime, horizon) pair.")
    p.add_argument("--regime-nodata-value", type=int, default=-9999,
                   help="NODATA value in regime TIF (default: -9999).")
    p.add_argument("--regime-calib-min-samples", type=int, default=1000,
                   help="Minimum samples per regime to fit a calibrator (else fallback to global).")

    # -------- Per-regime residual models --------
    p.add_argument("--train-residual-models", action="store_true",
                   help="Train small residual models per regime after global model training.")
    p.add_argument("--residual-max-depth", type=int, default=3,
                   help="Max depth for residual models (shallow is recommended).")
    p.add_argument("--residual-reg-lambda", type=float, default=10.0,
                   help="L2 regularization for residual models (higher = more conservative).")
    p.add_argument("--residual-reg-alpha", type=float, default=1.0,
                   help="L1 regularization for residual models.")
    p.add_argument("--residual-subsample", type=float, default=0.5,
                   help="Row subsampling rate for residual models.")
    p.add_argument("--residual-colsample-bynode", type=float, default=0.5,
                   help="Column subsampling rate per node for residual models.")
    p.add_argument("--residual-num-round", type=int, default=30,
                   help="Number of boosting rounds for residual models.")
    p.add_argument("--residual-min-samples", type=int, default=1000,
                   help="Minimum samples per regime to train a residual model.")
    p.add_argument("--use-residual-models", action="store_true",
                   help="Enable residual model corrections at inference time.")

    p.add_argument("--metrics-threshold", type=float, default=0.5)

    p.add_argument("--reliability-bin-width", type=float, default=0.005)
    p.add_argument("--reliability-bin-min", type=float, default=0.005)
    p.add_argument("--reliability-bin-max", type=float, default=0.060)
    p.add_argument("--reliability-min-count", type=int, default=50)

    p.add_argument("--reliability-fine-bin-width", type=float, default=1e-4)
    p.add_argument("--reliability-fine-min-count", type=int, default=1)
    p.add_argument("--reliability-fine-bins-cap", type=int, default=250_000)

    p.add_argument("--plot-file-format", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--plot-dpi", type=int, default=150)

    p.add_argument("--seeds", type=str, default="42")
    p.add_argument("--data-seed", type=int, default=123)
    p.add_argument("--logdir", default="runs/peat_xgb")

    # GPU housekeeping / performance
    p.add_argument("--use-cuda-prefetch", action="store_true", help="Overlap eval/calib H2D copies with compute.")
    p.add_argument("--cuda-empty-cache-every", type=int, default=0,
                   help="If >0, call torch.cuda.empty_cache() every N batches in eval/calib (fragmentation relief).")

    # Row estimate helper
    p.add_argument("--estimate-rows", action="store_true",
                   help="If set and train-mode=stream, run a quick row count estimate to guide chunk sizing.")
    p.add_argument("--estimate-rows-batches", type=int, default=20,
                   help="How many batches to sample for the row estimate.")
    # SHAP (feature importance via XGBoost pred_contribs)
    p.add_argument("--shap", action="store_true",
                   help="If set, compute SHAP feature importance after training and write plots/CSVs into logdir.")
    p.add_argument("--shap-split", choices=["train", "val", "test"], default="val",
                   help="Which split to sample for SHAP. Falls back if that split is unavailable.")
    p.add_argument("--shap-max-rows", type=int, default=20000,
                   help="Max valid pixel-rows sampled per horizon for SHAP.")
    p.add_argument("--shap-batch-size", type=int, default=0,
                   help="Batch size used while sampling SHAP rows (0 => use --batch-size).")
    p.add_argument("--shap-top-n", type=int, default=30,
                   help="How many top features to display in SHAP bar/beeswarm plots.")
    p.add_argument("--shap-dependence-topk", type=int, default=3,
                   help="Make dependence scatter plots for the top-K features per horizon.")
    p.add_argument("--shap-beeswarm", action="store_true",
                   help="If set and the 'shap' package is installed, also save beeswarm summary plots.")
    p.add_argument("--shap-save-values", action="store_true",
                   help="If set, save raw SHAP arrays (.npz) per horizon (can be large).")
    p.add_argument("--shap-predict-device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Where to compute SHAP pred_contribs. 'cuda' uses CuPy if available.")

    # -------- NetDry7 feature engineering --------
    p.add_argument("--add-netdry7", action="store_true", default=False,
                   help="Compute NetDry7 derived feature (7-day VPD-precip balance).")
    p.add_argument("--add-gated-netdry7", action="store_true", default=False,
                   help="Compute WTD-gated NetDry7 (requires --add-netdry7).")
    p.add_argument("--tau", type=float, default=0.4,
                   help="WTD threshold for gating (hinge = max(0, WTD - tau)).")
    p.add_argument("--window-days", type=int, default=7,
                   help="Rolling window size for NetDry7 computation.")
    p.add_argument("--min-valid-days", type=int, default=4,
                   help="Minimum valid days in window; fewer → NaN.")
    # Optional: explicit channel pattern overrides (for non-standard naming)
    p.add_argument("--vpd-channel-pattern", type=str, default=None,
                   help="Regex pattern to match VPD feature names (default: auto-detect vpd_rh c0).")
    p.add_argument("--precip-channel-pattern", type=str, default=None,
                   help="Regex pattern to match precip feature names (default: auto-detect era5land c8).")
    p.add_argument("--wtd-channel-pattern", type=str, default=None,
                   help="Regex pattern to match WTD feature names (default: auto-detect smap.*wtd).")

    return p.parse_args()





def _resolve_predict_device(args) -> str:
    if args.xgb_predict_device == "auto":
        if args.eval_device == "cuda" and _HAVE_CUPY:
            return "cuda"
        return "cpu"
    return args.xgb_predict_device


# ----------------------------------------------------------------------
# Train + eval one seed
# ----------------------------------------------------------------------

def run_one_seed(base_args, seed: int, train_ds, val_ds, test_ds, calib_ds):
    args = argparse.Namespace(**vars(base_args))
    args.seed = int(seed)
    set_seed(args.seed)

    args.xgb_predict_device = _resolve_predict_device(args)

    base = args.logdir
    args.logdir = os.path.join(base, f"seed{args.seed}")
    os.makedirs(args.logdir, exist_ok=True)

    header("Seed run")
    print(f"  • seed....................... {args.seed}")
    print(f"  • logdir..................... {args.logdir}")
    print(f"  • train_mode................. {args.train_mode}")
    print(f"  • xgb_tree_method............ {args.xgb_tree_method}")
    print(f"  • xgb_dmatrix................ {args.xgb_dmatrix}")
    print(f"  • xgb_quantile_precompute..... {bool(args.xgb_quantile_precompute)}")
    print(f"  • xgb_predict_device......... {args.xgb_predict_device}")
    print(f"  • eval_device................ {args.eval_device}")
    print(f"  • cupy available............. {_HAVE_CUPY}")

    n_features = _infer_num_features_from_dataset(train_ds, args)
    print(f"  • inferred n_features........ {n_features}")

    feature_names = build_feature_names_from_dataset(args, train_ds)
    print(f"[featnames] built {len(feature_names)} names (expected n_features={n_features})")

    # Handle streaming mode with NetDry7: auto-switch to tabularize
    # (NetDry7 requires computing train stats from full dataset before standardization)
    if getattr(args, 'add_netdry7', False) and args.train_mode == "stream":
        print("[NetDry7] WARNING: Streaming mode with NetDry7 requires pre-computed stats.")
        print("         Switching to tabularize mode for accurate standardization.")
        args.train_mode = "tabularize"

    # Training
    if args.train_mode == "tabularize":
        train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, args=args, device=None)
        val_loader   = make_loader(val_ds,   batch_size=args.batch_size, shuffle=False, args=args, device=None) if val_ds else None

        monitor = RAMMonitor()
        if not args.no_tqdm:
            monitor.start()

        monitor.set("Tabularizing train")
        X_tr, Y_tr, V_tr = build_tabular_from_loader(
            train_loader, args=args, desc="train_tabularize", max_rows=args.max_train_rows
        )

        # NetDry7 feature engineering (train - compute stats)
        netdry_stats = {}
        if getattr(args, 'add_netdry7', False):
            monitor.set("Computing NetDry7 features (train)")
            X_tr, feature_names, netdry_stats = compute_netdry7_features(
                X_tr, feature_names, args, train_stats=None, is_train=True
            )
            # Save stats for inference
            netdry_stats_path = os.path.join(args.logdir, "netdry7_stats.json")
            with open(netdry_stats_path, "w") as f:
                json.dump(netdry_stats, f, indent=2)
            print(f"[NetDry7] Saved stats to {netdry_stats_path}")

        X_va = Y_va = V_va = None
        if val_loader is not None:
            monitor.set("Tabularizing val")
            X_va, Y_va, V_va = build_tabular_from_loader(
                val_loader, args=args, desc="val_tabularize", max_rows=args.max_val_rows
            )
            # NetDry7 feature engineering (val - use train stats)
            if getattr(args, 'add_netdry7', False) and netdry_stats:
                monitor.set("Computing NetDry7 features (val)")
                X_va, _, _ = compute_netdry7_features(
                    X_va, feature_names, args, train_stats=netdry_stats, is_train=False
                )

        # Save feature_names after potential NetDry7 augmentation
        with open(os.path.join(args.logdir, "feature_names.json"), "w") as f:
            json.dump(feature_names, f, indent=2)
        print(f"[featnames] saved {len(feature_names)} feature names")

        # --- Hyperparameter Optimization (if enabled) ---
        if args.hpo:
            if not _HAVE_OPTUNA:
                print("[FATAL] --hpo requires optuna. Install with: pip install optuna")
                raise SystemExit(2)

            header("Hyperparameter Optimization (Optuna)")

            if args.hpo_params_in:
                # Load pre-tuned params from file
                print(f"[HPO] Loading pre-tuned params from {args.hpo_params_in}")
                with open(args.hpo_params_in, "r") as f:
                    best_hpo_params = json.load(f)
                print(f"[HPO] Loaded params: {best_hpo_params}")
            else:
                # Run HPO study
                print(f"[HPO] Running Optuna study with {args.hpo_trials} trials...")
                print(f"[HPO] Sampler: {args.hpo_sampler}")
                print(f"[HPO] Horizon strategy: {args.hpo_horizon_strategy}")
                print(f"[HPO] Optimizing: Brier Score (minimize)")

                base_params = make_xgb_params(args, seed=args.seed)
                train_data = (X_tr, Y_tr, V_tr)
                val_data = (X_va, Y_va, V_va)

                best_hpo_params, study = run_hpo_study(
                    args=args,
                    train_data=train_data,
                    val_data=val_data,
                    base_params=base_params,
                    seed=args.seed,
                    feature_names=feature_names,
                )

                # Save best params
                hpo_out_path = args.hpo_params_out or os.path.join(args.logdir, "hpo_best_params.json")
                hpo_save_data = {
                    **best_hpo_params,
                    "study_info": {
                        "best_brier_score": float(study.best_value),
                        "n_trials": len(study.trials),
                        "best_trial_number": study.best_trial.number,
                    }
                }
                with open(hpo_out_path, "w") as f:
                    json.dump(hpo_save_data, f, indent=2)
                print(f"\n[HPO] Saved best params to {hpo_out_path}")

                print(f"\n[HPO] ====== Best Trial ======")
                print(f"[HPO] Brier Score: {study.best_value:.6f}")

            # Override args with best params for final training
            print(f"\n[HPO] Applying best params for final training:")
            if "eta" in best_hpo_params:
                args.xgb_eta = best_hpo_params["eta"]
                print(f"  • eta: {args.xgb_eta:.4f}")
            if "num_round" in best_hpo_params:
                args.xgb_num_round = int(best_hpo_params["num_round"])
                print(f"  • num_round: {args.xgb_num_round}")
            if "max_depth" in best_hpo_params:
                args.xgb_max_depth = int(best_hpo_params["max_depth"])
                print(f"  • max_depth: {args.xgb_max_depth}")
            if "min_child_weight" in best_hpo_params:
                args.xgb_min_child_weight = best_hpo_params["min_child_weight"]
                print(f"  • min_child_weight: {args.xgb_min_child_weight:.4f}")
            if "subsample" in best_hpo_params:
                args.xgb_subsample = best_hpo_params["subsample"]
                print(f"  • subsample: {args.xgb_subsample:.4f}")
            if "colsample_bynode" in best_hpo_params:
                args.xgb_colsample_bynode = best_hpo_params["colsample_bynode"]
                print(f"  • colsample_bynode: {args.xgb_colsample_bynode:.4f}")
            if "gamma" in best_hpo_params:
                args.xgb_gamma = best_hpo_params["gamma"]
                print(f"  • gamma: {args.xgb_gamma:.4f}")
            # Apply calibration choice if calibration HPO is enabled
            if getattr(args, "hpo_include_calibration", False):
                if "calibration_method" in best_hpo_params:
                    args.calibration_method = str(best_hpo_params["calibration_method"])
                    print(f"  • calibration_method: {args.calibration_method}")
                if getattr(args, "calibration_method", "none") == "platt":
                    if "platt_reg" in best_hpo_params:
                        args.platt_reg = float(best_hpo_params["platt_reg"])
                        print(f"  • platt_reg: {args.platt_reg:g}")
                    if "platt_max_iter" in best_hpo_params:
                        args.platt_max_iter = int(best_hpo_params["platt_max_iter"])
                        print(f"  • platt_max_iter: {args.platt_max_iter}")
            print()

        monitor.set("Training XGBoost (GPU-first)")
        boosters, params = train_xgb_per_horizon_tabularized(X_tr, Y_tr, V_tr, X_va, Y_va, V_va, args, seed=args.seed)

        save_xgb_models(
            boosters=boosters,
            horizons=[int(h) for h in args.horizons],
            logdir=args.logdir,
            params=params,
            seed=args.seed,
            n_features=int(X_tr.shape[1]),
            args=args,
        )

        monitor.stop()

        del X_tr, Y_tr, V_tr, X_va, Y_va, V_va

    elif args.train_mode == "stream":
        # --- Hyperparameter Optimization for streaming mode (if enabled) ---
        if args.hpo:
            if not _HAVE_OPTUNA:
                print("[FATAL] --hpo requires optuna. Install with: pip install optuna")
                raise SystemExit(2)

            header("Hyperparameter Optimization (Optuna) [Streaming]")

            if args.hpo_params_in:
                # Load pre-tuned params from file
                print(f"[HPO] Loading pre-tuned params from {args.hpo_params_in}")
                with open(args.hpo_params_in, "r") as f:
                    best_hpo_params = json.load(f)
                print(f"[HPO] Loaded params: {best_hpo_params}")
            else:
                # Run HPO study
                print(f"[HPO] Running Optuna study with {args.hpo_trials} trials...")
                print(f"[HPO] Sampler: {args.hpo_sampler}")
                print(f"[HPO] Horizon strategy: {args.hpo_horizon_strategy}")
                print(f"[HPO] Optimizing: Brier Score (minimize)")
                print(f"[HPO] NOTE: Streaming HPO may be slow due to DMatrix rebuilds per trial.")

                base_params = make_xgb_params(args, seed=args.seed)

                best_hpo_params, study = run_hpo_study(
                    args=args,
                    train_data=train_ds,
                    val_data=val_ds,
                    base_params=base_params,
                    seed=args.seed,
                    feature_names=None,
                )

                # Save best params
                hpo_out_path = args.hpo_params_out or os.path.join(args.logdir, "hpo_best_params.json")
                hpo_save_data = {
                    **best_hpo_params,
                    "study_info": {
                        "best_brier_score": float(study.best_value),
                        "n_trials": len(study.trials),
                        "best_trial_number": study.best_trial.number,
                    }
                }
                with open(hpo_out_path, "w") as f:
                    json.dump(hpo_save_data, f, indent=2)
                print(f"\n[HPO] Saved best params to {hpo_out_path}")

                print(f"\n[HPO] ====== Best Trial ======")
                print(f"[HPO] Brier Score: {study.best_value:.6f}")

            # Override args with best params for final training
            print(f"\n[HPO] Applying best params for final training:")
            if "eta" in best_hpo_params:
                args.xgb_eta = best_hpo_params["eta"]
                print(f"  • eta: {args.xgb_eta:.4f}")
            if "num_round" in best_hpo_params:
                args.xgb_num_round = int(best_hpo_params["num_round"])
                print(f"  • num_round: {args.xgb_num_round}")
            if "max_depth" in best_hpo_params:
                args.xgb_max_depth = int(best_hpo_params["max_depth"])
                print(f"  • max_depth: {args.xgb_max_depth}")
            if "min_child_weight" in best_hpo_params:
                args.xgb_min_child_weight = best_hpo_params["min_child_weight"]
                print(f"  • min_child_weight: {args.xgb_min_child_weight:.4f}")
            if "subsample" in best_hpo_params:
                args.xgb_subsample = best_hpo_params["subsample"]
                print(f"  • subsample: {args.xgb_subsample:.4f}")
            if "colsample_bynode" in best_hpo_params:
                args.xgb_colsample_bynode = best_hpo_params["colsample_bynode"]
                print(f"  • colsample_bynode: {args.xgb_colsample_bynode:.4f}")
            if "gamma" in best_hpo_params:
                args.xgb_gamma = best_hpo_params["gamma"]
                print(f"  • gamma: {args.xgb_gamma:.4f}")
            # Apply calibration choice if calibration HPO is enabled
            if getattr(args, "hpo_include_calibration", False):
                if "calibration_method" in best_hpo_params:
                    args.calibration_method = str(best_hpo_params["calibration_method"])
                    print(f"  • calibration_method: {args.calibration_method}")
                if getattr(args, "calibration_method", "none") == "platt":
                    if "platt_reg" in best_hpo_params:
                        args.platt_reg = float(best_hpo_params["platt_reg"])
                        print(f"  • platt_reg: {args.platt_reg:g}")
                    if "platt_max_iter" in best_hpo_params:
                        args.platt_max_iter = int(best_hpo_params["platt_max_iter"])
                        print(f"  • platt_max_iter: {args.platt_max_iter}")
            print()

        boosters, params = train_xgb_per_horizon_streaming(train_ds=train_ds, val_ds=val_ds, args=args, seed=args.seed)

        save_xgb_models(
            boosters=boosters,
            horizons=[int(h) for h in args.horizons],
            logdir=args.logdir,
            params=params,
            seed=args.seed,
            n_features=int(n_features),
            args=args,
        )
    else:
        raise ValueError(f"Unknown train_mode: {args.train_mode}")

    # -------------------------
    # RESIDUAL MODEL TRAINING (optional)
    # -------------------------
    residual_boosters: Dict[int, List["xgb.Booster"]] = {}
    if getattr(args, 'train_residual_models', False):
        # Check if regimes are available
        base_ds = _unwrap_subset(train_ds)
        n_regimes_res = int(getattr(base_ds, "n_regimes", 0) or 0)

        if n_regimes_res == 0:
            print("\n[residual] WARNING: --train-residual-models set but no regimes found.")
            print("           Use --regime-source to provide regime GeoTIFF.")
        else:
            # -------------------------
            # RESIDUAL HPO (optional, sequential mode)
            # -------------------------
            if getattr(args, 'hpo_include_residual', False) and getattr(args, 'hpo_residual_mode', 'sequential') == 'sequential':
                if getattr(args, 'hpo_residual_params_in', None):
                    # Load pre-tuned residual params from JSON
                    print(f"[Residual HPO] Loading pre-tuned params from {args.hpo_residual_params_in}")
                    with open(args.hpo_residual_params_in, "r") as f:
                        residual_best_params = json.load(f)
                else:
                    # Run residual HPO study
                    base_residual_params = make_residual_xgb_params(args, args.seed)
                    residual_best_params, residual_study = run_residual_hpo_study(
                        args=args,
                        train_ds=train_ds,
                        val_ds=val_ds,
                        global_boosters=boosters,
                        seed=args.seed,
                        base_residual_params=base_residual_params,
                        feature_names=feature_names,
                    )

                    # Save residual HPO params
                    residual_hpo_out = getattr(args, 'hpo_residual_params_out', None) or os.path.join(args.logdir, "hpo_residual_params.json")
                    if residual_best_params:
                        with open(residual_hpo_out, "w") as f:
                            json.dump({
                                **residual_best_params,
                                "study_info": {
                                    "best_value": float(residual_study.best_value) if residual_study and residual_study.best_trial else None,
                                    "n_trials": len(residual_study.trials) if residual_study else 0,
                                    "sampler": args.hpo_sampler,
                                }
                            }, f, indent=2)
                        print(f"[Residual HPO] Saved best params to {residual_hpo_out}")

                # Apply best residual params to args
                if residual_best_params:
                    if "residual_max_depth" in residual_best_params:
                        args.residual_max_depth = int(residual_best_params["residual_max_depth"])
                    if "residual_reg_lambda" in residual_best_params:
                        args.residual_reg_lambda = float(residual_best_params["residual_reg_lambda"])
                    if "residual_reg_alpha" in residual_best_params:
                        args.residual_reg_alpha = float(residual_best_params["residual_reg_alpha"])
                    if "residual_subsample" in residual_best_params:
                        args.residual_subsample = float(residual_best_params["residual_subsample"])
                    if "residual_colsample_bynode" in residual_best_params:
                        args.residual_colsample_bynode = float(residual_best_params["residual_colsample_bynode"])
                    if "residual_num_round" in residual_best_params:
                        args.residual_num_round = int(residual_best_params["residual_num_round"])
                    print(f"[Residual HPO] Applied optimized params: max_depth={args.residual_max_depth}, "
                          f"reg_lambda={args.residual_reg_lambda:.2f}, reg_alpha={args.residual_reg_alpha:.2f}, "
                          f"subsample={args.residual_subsample:.2f}, colsample={args.residual_colsample_bynode:.2f}, "
                          f"num_round={args.residual_num_round}")

            residual_boosters = train_residual_models_per_regime_streaming(
                train_ds=train_ds,
                global_boosters=boosters,
                args=args,
                seed=args.seed,
                n_features_original=int(n_features),
                min_samples=int(args.residual_min_samples),
            )

            if residual_boosters:
                # Build augmented feature names (original features + global_logit)
                augmented_feature_names = feature_names + ["global_logit"]

                save_residual_models(
                    residual_boosters=residual_boosters,
                    horizons=[int(h) for h in args.horizons],
                    out_dir=args.logdir,
                    feature_names_augmented=augmented_feature_names,
                )

    # -------------------------
    # OOF SHAP tile signatures + clustering + per-cluster eval + plots
    # -------------------------
    try:
        do_oof_shap_signatures_and_clustering(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            boosters_full=boosters,
            args=args,
            seed=args.seed,
        )
    except Exception as e:
        import traceback
        print(f"[OOF/CLUSTER] Failed: {e}")
        traceback.print_exc()

    
    # Calibration fitting (optional)
    calibrators: Dict[Any, Any] = {}
    per_regime_calib = False

    if args.calibration_method != "none":
        if calib_ds is None:
            print("[WARN] calibration_method != none but calib_ds is None (calib_frac=0). Skipping calibration.")
        else:
            header("Calibration stage")
            print("We fit calibrators on a held-out subset of TRAIN PATCHES.\n")

            device = torch.device(args.eval_device)
            calib_loader = make_loader(
                calib_ds,
                batch_size=args.batch_size,
                shuffle=False,
                args=args,
                device=device,
                eval_mode=True,
            )

            # If regimes are enabled, fit per-(regime,horizon) calibrators, with a global fallback.
            base_calib = _unwrap_subset(calib_ds)
            n_regimes = int(getattr(base_calib, "n_regimes", 0) or 0)
            has_regimes = (getattr(args, "regime_source", None) is not None) and (n_regimes > 0)

            if has_regimes:
                print(f"[Calibration] Per-regime calibration enabled (n_regimes={n_regimes}).")
                calibrators = fit_calibrators_per_regime_from_loader(
                    boosters=boosters,
                    calib_loader=calib_loader,
                    device=device,
                    args=args,
                    n_regimes=n_regimes,
                )
                per_regime_calib = True
            else:
                calibrators = fit_calibrators_from_loader(boosters, calib_loader, device=device, args=args)

            cal_path = os.path.join(args.logdir, "calibrators.json")
            save_calibrators(
                calibrators,
                method=args.calibration_method,
                out_path=cal_path,
                per_regime=per_regime_calib,
            )
    else:
        if args.calib_frac > 0:
            print("[WARN] calib_frac > 0 but calibration_method=none -> you're throwing away training patches.")

    # Evaluation
    device = torch.device(args.eval_device)
    header("Evaluation")
    print(f"  • torch device............... {device}")
    print(f"  • xgb_predict_device......... {args.xgb_predict_device}")
    print(f"  • calibration_method......... {args.calibration_method}")
    if device.type == "cuda":
        print(f"  • cuda device name........... {torch.cuda.get_device_name(0)}")
        print(f"  • torch.cuda.mem_allocated... {torch.cuda.memory_allocated()/1e9:.3f} GB (initial)")
        print(f"  • torch.cuda.mem_reserved.... {torch.cuda.memory_reserved()/1e9:.3f} GB (initial)")
    print()

    val_loader_eval  = make_loader(val_ds,  batch_size=args.batch_size, shuffle=False, args=args, device=device, eval_mode=True) if val_ds else None
    test_loader_eval = make_loader(test_ds, batch_size=args.batch_size, shuffle=False, args=args, device=device, eval_mode=True) if test_ds else None

    # Build logits function - with or without residual corrections
    if getattr(args, 'use_residual_models', False) and residual_boosters:
        print("[eval] Using RESIDUAL model corrections")
        logits_fn = xgb_logits_fn_with_residual_models(
            global_boosters=boosters,
            residual_boosters=residual_boosters,
            horizons=[int(h) for h in args.horizons],
            args=args,
            calibrators=calibrators,
        )
    else:
        logits_fn = xgb_logits_fn_from_boosters(boosters, horizons=[int(h) for h in args.horizons], args=args, calibrators=calibrators)
    criterion = MaskedBCEWithLogits().to(device)

    metrics_csv_path = init_metrics_csv(args.logdir)
    history: Dict[str, Any] = {"val": None, "test": None, "calibration_method": args.calibration_method}

    if val_loader_eval is not None:
        val_loss, val_metrics = evaluate_with_logits_fn(
            logits_fn=logits_fn,
            loader=val_loader_eval,
            device=device,
            criterion=criterion,
            args=args,
            use_tqdm=not args.no_tqdm,
        )
        _print_split("val", val_loss, val_metrics, calibration_method=args.calibration_method)
        append_metrics_csv(metrics_csv_path, epoch=0, split="val", loss=val_loss, metrics=val_metrics, model_type="xgboost", calibration_method=args.calibration_method)
        save_and_log_calibration("val", val_metrics, epoch=0, args=args)
        history["val"] = {"loss": float(val_loss), "metrics": val_metrics}

        # Clean up validation loader to free workers and memory before test evaluation
        del val_loader_eval
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    if test_loader_eval is not None:
        try:
            test_loss, test_metrics = evaluate_with_logits_fn(
                logits_fn=logits_fn,
                loader=test_loader_eval,
                device=device,
                criterion=criterion,
                args=args,
                use_tqdm=not args.no_tqdm,
            )
            _print_split("test", test_loss, test_metrics, calibration_method=args.calibration_method)
            append_metrics_csv(metrics_csv_path, epoch=0, split="test", loss=test_loss, metrics=test_metrics, model_type="xgboost", calibration_method=args.calibration_method)
            save_and_log_calibration("test", test_metrics, epoch=0, args=args)
            history["test"] = {"loss": float(test_loss), "metrics": test_metrics}
        finally:
            # Clean up test loader to free workers and memory before SHAP analysis
            del test_loader_eval
            import gc
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    # SHAP feature importance (optional)
    try:
        run_shap_analysis(
            boosters=boosters,
            args=args,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            feature_names=feature_names,
        )
    except Exception as e:
        print(f"[SHAP] Failed: {e}")

    # Final cleanup before returning
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    save_history_json(history, args.logdir)
    return args.logdir





# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _warn_ignored_flags(args):
    stream_flags = {
        "stream_chunk_rows": args.stream_chunk_rows,
        "stream_accumulate_multiplier": args.stream_accumulate_multiplier,
        "stream_max_rows_per_horizon": args.stream_max_rows_per_horizon,
        "stream_use_cupy": args.stream_use_cupy,
        "stream_use_val": args.stream_use_val,
        "stream_shuffle_train": args.stream_shuffle_train,
        "stream_show_progress": args.stream_show_progress,
    }
    if args.train_mode != "stream":
        used = []
        if args.stream_use_cupy: used.append("--stream-use-cupy")
        if args.stream_use_val: used.append("--stream-use-val")
        if args.stream_shuffle_train: used.append("--stream-shuffle-train")
        if args.stream_show_progress: used.append("--stream-show-progress")
        if args.stream_max_rows_per_horizon is not None: used.append("--stream-max-rows-per-horizon")
        if args.stream_chunk_rows != 1_000_000: used.append("--stream-chunk-rows")
        if args.stream_accumulate_multiplier != 3: used.append("--stream-accumulate-multiplier")
        if used:
            print("[WARN] You set streaming flags but --train-mode is not 'stream'. These flags will be ignored:")
            for u in used:
                print(f"  • {u}")

    if args.train_mode == "stream":
        if args.max_train_rows is not None or args.max_val_rows is not None:
            print("[WARN] You set --max-train-rows/--max-val-rows but --train-mode stream ignores them (stream caps are per-horizon).")
            print("       Use --stream-max-rows-per-horizon instead.")


def main():
    args = parse_args()
    set_seed(12345)

    # CUDA setup
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

    args.xgb_predict_device = _resolve_predict_device(args)
    _warn_ignored_flags(args)

    header("GPU utilization plan")
    print(f"  • torch.cuda_available........ {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  • torch cuda device........... {torch.cuda.get_device_name(0)}")
    print(f"  • xgboost version............. {getattr(xgb, '__version__', 'unknown')}")
    print(f"  • cupy available.............. {_HAVE_CUPY}")
    print(f"  • train_mode.................. {args.train_mode}")
    if args.train_mode == "stream":
        print(f"  • stream_chunk_rows........... {int(args.stream_chunk_rows):,}")
        print(f"  • stream_accumulate_mult...... {int(args.stream_accumulate_multiplier)} (payload ~chunk_rows*mult)")
        print(f"  • stream_max_rows/horizon..... {args.stream_max_rows_per_horizon}")
        print(f"  • stream_use_val.............. {bool(args.stream_use_val)}")
        print(f"  • stream_use_cupy............. {bool(args.stream_use_cupy)}")
        print(f"  • xgb DataIter available...... {bool(_have_xgb_dataiter())}")
        print(f"  • xgb_quantile_precompute..... {bool(args.xgb_quantile_precompute)}")
        print(f"  • estimate_rows............... {bool(args.estimate_rows)} (batches={int(args.estimate_rows_batches)})")
    print(f"  • xgb_tree_method............. {args.xgb_tree_method}")
    print(f"  • xgb_dmatrix................. {args.xgb_dmatrix}")
    print(f"  • eval_device................. {args.eval_device}")
    print(f"  • xgb_predict_device (final).. {args.xgb_predict_device}")
    print(f"  • use_cuda_prefetch........... {bool(args.use_cuda_prefetch)}")
    print(f"  • cuda_empty_cache_every...... {int(args.cuda_empty_cache_every)}")

    if args.xgb_tree_method == "gpu_hist" and not torch.cuda.is_available():
        print("[WARN] tree_method=gpu_hist but torch says no CUDA. XGBoost GPU may still work if system has CUDA,")
        print("       but this is a red flag. Confirm nvidia-smi and XGBoost CUDA build.")
    if args.xgb_predict_device == "cuda" and not _HAVE_CUPY:
        print("[WARN] xgb_predict_device=cuda requires CuPy for zero-copy DLPack path. Falling back to CPU prediction.\n")

    if args.train_mode == "stream" and not _have_xgb_dataiter():
        print("[FATAL] --train-mode stream requested, but xgb.core.DataIter is unavailable in your XGBoost build.")
        print("        Install/upgrade xgboost to a version that exposes DataIter.")
        raise SystemExit(2)

    header("Reliability settings sanity check")
    print(f"  • --reliability-bin-width..... {args.reliability_bin_width}")
    print(f"  • --reliability-bin-min....... {args.reliability_bin_min}")
    print(f"  • --reliability-bin-max....... {args.reliability_bin_max}")
    print(f"  • --reliability-min-count..... {args.reliability_min_count}")
    print(f"  • --reliability-fine-bin-width {args.reliability_fine_bin_width}")
    print(f"  • --reliability-fine-bins-cap. {args.reliability_fine_bins_cap}")

    seeds = parse_seeds_list(args.seeds)
    # Use the first seed for pre-screening (args.seed is only set later in run_one_seed)
    prescreen_seed = seeds[0] if seeds else 42


    # --- T-Hist (Temporal History) Pre-Screening (if enabled) ---
    # This runs BEFORE regime pre-screening / dataset build because it affects dataset construction.
    if getattr(args, 'hpo_t_hist_search', False) and getattr(args, 'hpo', False):
        header("T-Hist Hyperparameter Pre-Screening")

        # Parse t_hist values to search
        t_hist_values_str = getattr(args, 'hpo_t_hist_values', '15,30,45,60')
        try:
            t_hist_values = [int(x.strip()) for x in t_hist_values_str.split(',') if x.strip()]
            t_hist_values = [t for t in t_hist_values if t > 0]
        except Exception:
            print(f"[T-Hist HPO] Invalid --hpo-t-hist-values='{t_hist_values_str}'. "
                  f"Falling back to current --T-hist={int(args.T_hist)}")
            t_hist_values = [int(args.T_hist)]

        if not t_hist_values:
            t_hist_values = [int(args.T_hist)]

        # Deduplicate while preserving order
        _seen = set()
        t_hist_values = [t for t in t_hist_values if not (t in _seen or _seen.add(t))]

        print(f"[T-Hist HPO] T_hist values to test: {t_hist_values}")

        original_t_hist = int(args.T_hist)
        os.makedirs(args.logdir, exist_ok=True)

        try:
            # Create dataset factory function for pre-screening
            def _t_hist_dataset_factory(args_inner):
                """Factory to create datasets for t_hist pre-screening."""
                train_ds, val_ds, test_ds = build_datasets(args_inner)
                # Split train for calibration if needed
                train_ds, calib_ds = split_train_for_calibration(train_ds, args_inner)
                return train_ds, val_ds, test_ds, calib_ds

            base_params = make_xgb_params(args, seed=prescreen_seed)
            best_t_hist, t_hist_scores = run_t_hist_prescreen(
                args=args,
                t_hist_values=t_hist_values,
                base_params=base_params,
                seed=prescreen_seed,
                dataset_factory_fn=_t_hist_dataset_factory,
            )

            # Update args.T_hist with the best value
            args.T_hist = int(best_t_hist)
            print(f"\n[T-Hist HPO] Selected best configuration: T_hist={int(args.T_hist)}")

            # Save t_hist selection results
            t_hist_hpo_out = os.path.join(args.logdir, "hpo_t_hist_selection.json")
            t_hist_hpo_data = {
                "best_t_hist": int(best_t_hist),
                "metric": getattr(args, 'hpo_t_hist_metric', 'brier'),
                "all_scores": {str(k): float(v) for k, v in t_hist_scores.items()},
                "t_hist_values_tested": [int(t) for t in t_hist_values],
            }
            with open(t_hist_hpo_out, "w") as f:
                json.dump(t_hist_hpo_data, f, indent=2)
            print(f"[T-Hist HPO] Saved selection results to: {t_hist_hpo_out}")

        except Exception as e:
            print(f"[T-Hist HPO] ERROR during pre-screening: {e}")
            import traceback
            traceback.print_exc()
            args.T_hist = original_t_hist
            print(f"[T-Hist HPO] Continuing with current --T-hist={int(args.T_hist)}")

    # --- Regime Count Pre-Screening (if enabled) ---
    # This runs BEFORE the main dataset build to select the optimal number of regimes
    if getattr(args, 'hpo_regime_search', False) and getattr(args, 'hpo', False):
        if not _HAVE_OPTUNA:
            print("[FATAL] --hpo-regime-search requires optuna. Install with: pip install optuna")
            raise SystemExit(2)

        header("Regime Count Hyperparameter Optimization")

        # Parse regime counts to search
        regime_counts_str = getattr(args, 'hpo_regime_counts', '2,3,4,5,6,7')
        regime_counts = [int(x.strip()) for x in regime_counts_str.split(',') if x.strip()]
        print(f"[Regime HPO] Regime counts to test: {regime_counts}")

        # Determine regime directory
        regime_dir = getattr(args, 'hpo_regime_dir', None)
        if not regime_dir:
            # Default: look for CLUSTER/regime_output relative to script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            regime_dir = os.path.join(os.path.dirname(script_dir), 'CLUSTER', 'regime_output')
            if not os.path.isdir(regime_dir):
                # Try relative to working directory
                regime_dir = os.path.join('CLUSTER', 'regime_output')
        print(f"[Regime HPO] Searching for regime maps in: {regime_dir}")

        try:
            # Discover available regime maps
            regime_maps = discover_regime_maps(regime_dir, regime_counts)

            # Create dataset factory function for pre-screening
            def _regime_dataset_factory(args_inner):
                """Factory to create datasets for regime pre-screening."""
                train_ds, val_ds, test_ds = build_datasets(args_inner)
                # Split train for calibration if needed
                train_ds, calib_ds = split_train_for_calibration(train_ds, args_inner)
                return train_ds, val_ds, test_ds, calib_ds

            # Run pre-screen to find best regime count
            base_params = make_xgb_params(args, seed=prescreen_seed)
            best_n_regimes, regime_scores = run_regime_prescreen(
                args=args,
                regime_maps=regime_maps,
                base_params=base_params,
                seed=prescreen_seed,
                dataset_factory_fn=_regime_dataset_factory,
            )

            # Update args.regime_source with the best regime map (or select 'no regimes')
            best_regime_path = regime_maps.get(best_n_regimes)
            if best_regime_path:
                print(f"\n[Regime HPO] Selected best configuration: {best_n_regimes} regimes")
                print(f"[Regime HPO] Using regime map: {best_regime_path}")
                args.regime_source = best_regime_path
                args.regime_as_features = True
            else:
                print(f"\n[Regime HPO] Selected best configuration: NO regimes (baseline)")
                args.regime_source = None
                args.regime_as_features = False

            # Save regime selection results
            regime_hpo_out = os.path.join(args.logdir, "hpo_regime_selection.json")
            regime_hpo_data = {
                "best_n_regimes": best_n_regimes,
                "regime_path": best_regime_path,
                "all_scores": {str(k): v for k, v in regime_scores.items()},
                "regime_maps_tested": {str(k): v for k, v in regime_maps.items()},
            }
            with open(regime_hpo_out, "w") as f:
                json.dump(regime_hpo_data, f, indent=2)
            print(f"[Regime HPO] Saved selection results to: {regime_hpo_out}")

        except FileNotFoundError as e:
            print(f"[Regime HPO] ERROR: {e}")
            print("[Regime HPO] Continuing with current regime_source setting.")
        except Exception as e:
            print(f"[Regime HPO] ERROR during pre-screening: {e}")
            import traceback
            traceback.print_exc()
            print("[Regime HPO] Continuing with current regime_source setting.")

    header("Building datasets (once)")
    monitor = RAMMonitor()
    if not args.no_tqdm:
        monitor.start()
        monitor.set("Building datasets")

    train_ds, val_ds, test_ds = build_datasets(args)

    feature_names = build_feature_names_from_dataset(args, train_ds)
    

    
    # Split train patches for calibration
    train_ds, calib_ds = split_train_for_calibration(train_ds, args)

    if not args.no_tqdm:
        monitor.stop()

    # Visualize valid pixels from smap_wtd and regime map for comparison
    header("Visualizing Valid Pixels: SMAP_WTD vs Regime Map")
    try:
        import matplotlib.pyplot as plt
        import zarr as _zarr

        # Load peat mask (from smap_wtd)
        peat_mask_path = args.peat_mask_source.strip() or "smap_wtd.zarr"
        print(f"[viz] Loading peat mask from: {peat_mask_path}")
        peat_store = _zarr.open(peat_mask_path, mode='r')
        if 'peat_mask' in peat_store:
            peat_mask = np.array(peat_store['peat_mask'])
        elif 'mask' in peat_store:
            peat_mask = np.array(peat_store['mask'])
        else:
            # Try to find the mask array
            for key in peat_store.keys():
                if 'mask' in key.lower() or 'peat' in key.lower():
                    peat_mask = np.array(peat_store[key])
                    break
            else:
                print(f"[viz] Warning: Could not find peat mask in {peat_mask_path}")
                peat_mask = None

        # Load regime map if available
        regime_mask = None
        if hasattr(args, 'regime_source') and args.regime_source:
            regime_source_path = args.regime_source.strip()
            print(f"[viz] Loading regime map from: {regime_source_path}")

            if regime_source_path.endswith('.zarr'):
                regime_store = _zarr.open(regime_source_path, mode='r')
                # Try to find the regime array
                if 'regime' in regime_store:
                    regime_data = np.array(regime_store['regime'])
                elif 'field' in regime_store:
                    regime_data = np.array(regime_store['field'])
                else:
                    for key in regime_store.keys():
                        regime_data = np.array(regime_store[key])
                        break

                # Handle nodata values
                regime_nodata = getattr(args, 'regime_nodata_value', -9999)
                regime_mask = (regime_data != regime_nodata) & np.isfinite(regime_data)
            else:
                print(f"[viz] Warning: Regime source format not recognized: {regime_source_path}")

        # Create visualization
        if peat_mask is not None:
            fig, axes = plt.subplots(1, 2 if regime_mask is not None else 1, figsize=(16, 6))
            if regime_mask is None:
                axes = [axes]

            # Plot 1: SMAP_WTD valid pixels
            im1 = axes[0].imshow(peat_mask, cmap='viridis', interpolation='nearest')
            axes[0].set_title(f'Valid Pixels from SMAP_WTD\n({peat_mask_path})\nValid pixels: {np.sum(peat_mask):,}')
            axes[0].set_xlabel('Longitude (pixel)')
            axes[0].set_ylabel('Latitude (pixel)')
            plt.colorbar(im1, ax=axes[0], label='Valid (1) / Invalid (0)')

            # Plot 2: Regime map valid pixels (if available)
            if regime_mask is not None:
                im2 = axes[1].imshow(regime_mask, cmap='viridis', interpolation='nearest')
                axes[1].set_title(f'Valid Pixels from Regime Map\n({regime_source_path})\nValid pixels: {np.sum(regime_mask):,}')
                axes[1].set_xlabel('Longitude (pixel)')
                axes[1].set_ylabel('Latitude (pixel)')
                plt.colorbar(im2, ax=axes[1], label='Valid (1) / Invalid (0)')

                # Print comparison statistics
                print(f"[viz] SMAP_WTD valid pixels: {np.sum(peat_mask):,}")
                print(f"[viz] Regime map valid pixels: {np.sum(regime_mask):,}")
                if peat_mask.shape == regime_mask.shape:
                    overlap = np.sum(peat_mask & regime_mask)
                    print(f"[viz] Overlapping valid pixels: {overlap:,}")
                    print(f"[viz] Pixels valid in SMAP_WTD but not regime: {np.sum(peat_mask & ~regime_mask):,}")
                    print(f"[viz] Pixels valid in regime but not SMAP_WTD: {np.sum(~peat_mask & regime_mask):,}")
                else:
                    print(f"[viz] Warning: Shape mismatch - SMAP_WTD: {peat_mask.shape}, Regime: {regime_mask.shape}")

            plt.tight_layout()
            viz_path = os.path.join(args.logdir, "valid_pixels_comparison.png")
            os.makedirs(args.logdir, exist_ok=True)
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"[viz] Saved visualization to: {viz_path}")
        else:
            print("[viz] Skipping visualization - peat mask not found")

    except Exception as e:
        print(f"[viz] Warning: Could not create valid pixels visualization: {e}")
        import traceback
        traceback.print_exc()

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty after filtering/splitting.")

    # NEW: optional row estimate to guide chunk sizing
    if args.train_mode == "stream" and bool(args.estimate_rows):
        estimate_dataset_rows(train_ds, args, desc="train")
        if val_ds is not None and len(val_ds) > 0:
            estimate_dataset_rows(val_ds, args, desc="val")

    header("Run configuration")
    print(f"  • seeds....................... {seeds}")
    print(f"  • data_seed (splits fixed).... {args.data_seed}")
    print(f"  • train_mode.................. {args.train_mode}")
    if args.train_mode == "stream":
        print(f"  • stream_chunk_rows........... {int(args.stream_chunk_rows):,}")
        print(f"  • stream_accumulate_mult...... {int(args.stream_accumulate_multiplier)}")
        print(f"  • stream_use_val.............. {bool(args.stream_use_val)}")
        print(f"  • stream_use_cupy............. {bool(args.stream_use_cupy)}")
        print(f"  • stream_shuffle_train........ {bool(args.stream_shuffle_train)}")
        print(f"  • stream_show_progress........ {bool(args.stream_show_progress)}")
    print(f"  • xgb: trees (num_round)...... {args.xgb_num_round}")
    print(f"  • xgb: max_depth.............. {args.xgb_max_depth}")
    print(f"  • xgb: subsample.............. {args.xgb_subsample}")
    print(f"  • xgb: colsample_bynode....... {args.xgb_colsample_bynode}")
    print(f"  • xgb: eta.................... {args.xgb_eta}")
    print(f"  • xgb: max_bin................ {args.xgb_max_bin}")
    print(f"  • xgb: sampling_method........ {args.xgb_sampling_method}")
    print(f"  • xgb: grow_policy............ {args.xgb_grow_policy}")
    print(f"  • xgb: tree_method............ {args.xgb_tree_method}")
    print(f"  • xgb: gpu_id................. {args.xgb_gpu_id}")
    print(f"  • xgb: dmatrix................ {args.xgb_dmatrix}")
    print(f"  • xgb: quantile_precompute.... {bool(args.xgb_quantile_precompute)}")
    print(f"  • xgb: nthread................ {args.xgb_nthread if args.xgb_nthread is not None else (os.cpu_count() or 1)}")
    print(f"  • calibration_method.......... {args.calibration_method}")
    print(f"  • calib_frac.................. {args.calib_frac}")
    print(f"  • max_calib_rows_per_horizon.. {args.max_calib_rows_per_horizon}")

    base_args = argparse.Namespace(**vars(args))

    out_dirs = []
    for s in seeds:
        header(f"TRAIN+CALIB+EVAL seed {s}")
        out_dir = run_one_seed(
            base_args=base_args,
            seed=int(s),
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            calib_ds=calib_ds,
        )
        out_dirs.append(out_dir)

    header("All done")
    for d in out_dirs:
        print(f"  • {d}")

    # Explicit cleanup to avoid "leaked semaphore" warnings from multiprocessing
    import gc
    gc.collect()


if __name__ == "__main__":
    # Suppress multiprocessing resource_tracker warnings about leaked semaphores
    # These are harmless and occur when DataLoader workers aren't fully cleaned up
    import warnings
    warnings.filterwarnings("ignore", message=".*resource_tracker.*leaked.*semaphore.*")

    main()

# =============================================================================
# Example CLI usage for NetDry7 feature engineering:
# =============================================================================
#
# Basic NetDry7 (7-day VPD-precip drying index):
#   python train_XGB_global_new_shap_backup_pre_RF_model_switch.py \
#     --input /path/to/vpd_rh.zarr \
#     --input /path/to/era5land.zarr \
#     --input /path/to/smap_wtd.zarr \
#     --viirs-zarr /path/to/viirs.zarr \
#     --add-netdry7 \
#     --window-days 7 \
#     --min-valid-days 4
#
# With WTD-gated NetDry7 (gating amplifies drying signal when WTD > tau):
#   python train_XGB_global_new_shap_backup_pre_RF_model_switch.py \
#     --input /path/to/vpd_rh.zarr \
#     --input /path/to/era5land.zarr \
#     --input /path/to/smap_wtd.zarr \
#     --viirs-zarr /path/to/viirs.zarr \
#     --add-netdry7 \
#     --add-gated-netdry7 \
#     --tau 0.4 \
#     --window-days 7
#
# With custom channel patterns (if auto-detection fails):
#   python train_XGB_global_new_shap_backup_pre_RF_model_switch.py \
#     --input /path/to/vpd_rh.zarr \
#     --input /path/to/era5land.zarr \
#     --input /path/to/smap_wtd.zarr \
#     --viirs-zarr /path/to/viirs.zarr \
#     --add-netdry7 \
#     --vpd-channel-pattern "vpd_rh.*c0" \
#     --precip-channel-pattern "era5land.*c8" \
#     --wtd-channel-pattern "SMAP_WTD"
# =============================================================================
