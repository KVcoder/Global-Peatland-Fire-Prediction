#!/usr/bin/env python3
"""
Spatiotemporal Peat Ignition — XGBoost (per-pixel, multi-horizon) + GPU-first + Platt/Isotonic calibration
FULL SCRIPT (updated with GPU-first fixes + STREAMING TRAINING + bugfixes)

Fixes applied (matching the issues you listed)
----------------------------------------------
✅ (1) Streaming DataIter progress-bar lifecycle:
    - No longer creates “leaky” tqdm objects across repeated reset() calls.
    - Properly closes tqdm when the iterator is exhausted (and in __del__ as backup).
    - Progress bar now actually advances with .update(rows_emitted).

✅ (2) torch.cuda.empty_cache() “counter”:
    - Removed function-attribute counters entirely.
    - Each eval/calib run maintains its own local batch_counter.

✅ (3) Calibration counter issue:
    - Same as (2): local counter per run, no shared static state.

✅ (7) _is_cuda_device_str:
    - Now matches cuda:1/cuda:2/... and gpu:1/... etc.

✅ (8) RAMMonitor thread-safety:
    - Protects self.status with a lock (and pbar updates).

✅ (9) _warn_if_gpu_copy_too_big no longer swallows everything silently:
    - Uses narrower exceptions and prints a warning with details.

✅ (15) Warn about ignored streaming flags when not in stream mode (and vice versa).

Other small robustness tweaks
-----------------------------
- Streaming DataIter reset() no longer recreates tqdm; it reuses one instance.
- Streaming DataIter always uses args.stream_show_progress (no “missing defaults to True”).
- CPU fallback arrays in _predict_margin_gpu_if_possible are only created in the fallback path.
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
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from joint_peat_dataset_builder import JointPeatDataset, parse_input_spec

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
    cp = None
    _HAVE_CUPY = False

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:
    print("MATPLOTLIB NOT IMPORTED (plots disabled unless available)")

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
    if not batch:
        raise RuntimeError("Empty batch in collate (dataset filtering too aggressive?).")
    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals, 0) if torch.is_tensor(vals[0]) else vals
    return out


def make_loader(ds, batch_size, shuffle, args, device: Optional[torch.device] = None):
    """
    Dataloader helper.
    NOTE: Dataloader yields CPU tensors. We move to GPU via prefetcher / explicit .to().
    """
    pin = bool(device is not None and device.type == "cuda")
    kw: Dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=pin,
        collate_fn=collate,
        persistent_workers=(args.workers > 0),
        shuffle=shuffle,
        drop_last=False,
    )

    if args.workers > 0:
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

        logits = logits_fn(batch)  # (B,K,H,W) on device
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

        B, K, H, W = logits.shape
        K_eff = min(K, num_h)

        for k_idx in range(K_eff):
            pk = preds[:, k_idx].reshape(-1)
            tk = targets[:, k_idx].reshape(-1)
            vk = valid[:, k_idx].reshape(-1)
            tp_h[k_idx] += (pk & tk & vk).sum().item()
            fp_h[k_idx] += (pk & ~tk & vk).sum().item()
            fn_h[k_idx] += (~pk & tk & vk).sum().item()
            tn_h[k_idx] += (~pk & ~tk & vk).sum().item()

        # --- ONE CPU transfer per batch ---
        probs_cpu = probs.detach().cpu().numpy().astype(np.float32, copy=False)      # (B,K,H,W)
        targ_cpu  = targets.detach().cpu().numpy().astype(np.bool_, copy=False)     # (B,K,H,W)
        valid_cpu = valid.detach().cpu().numpy().astype(np.bool_, copy=False)       # (B,K,H,W)

        for k_idx in range(K_eff):
            v_k = valid_cpu[:, k_idx].reshape(-1)
            if v_k.sum() == 0:
                continue

            p_flat = probs_cpu[:, k_idx].reshape(-1)[v_k].astype(np.float32, copy=False)
            y_flat = targ_cpu[:, k_idx].reshape(-1)[v_k].astype(np.float32, copy=False)

            # float64 only where needed (binning is sensitive); keep as float32 otherwise
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
    """
    x = _flatten_time_if_needed(batch["x"])    # (B,F,H,W)
    y = batch["y"]                             # (B,K,H,W)
    m = batch["mask"]                          # (B,K,H,W)

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

    Returns:
      X: (N,F) float32
      y: (N,) uint8 (0/1)
      v: (N,) bool
    """
    x = _flatten_time_if_needed(batch["x"])          # (B,F,H,W)
    yk = batch["y"][:, k]                           # (B,H,W)
    mk = batch["mask"][:, k]                        # (B,H,W)

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
        # NOTE: this is O(total_rows) but can spike RAM (copies). If you need disk-backed, we can add memmap.
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


def make_xgb_params(args, seed: int) -> dict:
    """
    GPU-first:
      - default tree_method = gpu_hist
      - sets device/gpu_id depending on xgboost version
      - exposes extra GPU-friendly knobs
    """
    tree_method = args.xgb_tree_method

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

    if tree_method == "gpu_hist":
        v = _xgb_version_tuple()
        params["gpu_id"] = int(args.xgb_gpu_id)
        if v >= (2, 0, 0):
            params["device"] = f"cuda:{int(args.xgb_gpu_id)}"
        params["predictor"] = "gpu_predictor"

    return params


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

    - default: NumPy chunks (reliable)
    - optional: --stream-use-cupy to feed CuPy directly (if available)
    """
    if not _have_xgb_dataiter():
        raise RuntimeError("This XGBoost build does not expose xgb.core.DataIter; cannot stream.")

    chunk_rows = int(args.stream_chunk_rows)
    max_rows = None if args.stream_max_rows_per_horizon is None else int(args.stream_max_rows_per_horizon)
    use_cupy = bool(args.stream_use_cupy and _HAVE_CUPY and args.xgb_tree_method == "gpu_hist")

    h_label = int(args.horizons[horizon_k])

    class PatchRowDataIter(xgb.core.DataIter):  # type: ignore
        """
        Fixes:
          - Reuses a single tqdm (no new one per reset)
          - Closes tqdm once exhausted (and in __del__)
          - Actually updates the bar position by emitted rows
        """
        def __init__(self):
            super().__init__()
            self._loader = None
            self._it = None
            self._seen_rows = 0
            self._exhausted = False
            self._pbar = None
            self._pbar_enabled = (not args.no_tqdm) and bool(args.stream_show_progress)

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
                # total unknown: use dynamic; show rows via update()
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
            # Called multiple times (esp. QuantileDMatrix sketching).
            # Do NOT recreate tqdm every time; reuse one.
            self._loader = make_loader(ds, batch_size=args.batch_size, shuffle=shuffle, args=args, device=None)
            self._it = iter(self._loader)
            self._seen_rows = 0
            self._exhausted = False
            self._maybe_init_pbar()

        def next(self, input_data):
            if self._exhausted:
                self._close_pbar()
                return 0

            if max_rows is not None and self._seen_rows >= max_rows:
                self._exhausted = True
                self._close_pbar()
                return 0

            X_chunks = []
            y_chunks = []
            n_out = 0

            while n_out < chunk_rows:
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
                    self._pbar.set_postfix({"rows_seen": f"{self._seen_rows:,}"})

                if max_rows is not None and self._seen_rows >= max_rows:
                    self._exhausted = True
                    break

            if n_out <= 0:
                if self._exhausted:
                    self._close_pbar()
                return 0

            X = np.concatenate(X_chunks, axis=0).astype(np.float32, copy=False)
            y = np.concatenate(y_chunks, axis=0).astype(np.float32, copy=False)

            if use_cupy:
                X = cp.asarray(X)
                y = cp.asarray(y)

            input_data(data=X, label=y)

            if self._exhausted:
                self._close_pbar()

            return 1

    it = PatchRowDataIter()
    it.reset()
    return it


def _make_stream_dmatrix(ds, args, horizon_k: int, shuffle: bool, desc: str, params: dict):
    """
    Build DMatrix/QuantileDMatrix from a streaming DataIter.
    """
    it = make_streaming_dataiter(ds, args=args, horizon_k=horizon_k, shuffle=shuffle, desc=desc)

    use_quantile = (args.xgb_dmatrix == "quantile" and hasattr(xgb, "QuantileDMatrix"))
    nthread = int(params.get("nthread", os.cpu_count() or 1))

    if use_quantile:
        try:
            dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin), nthread=nthread)
        except TypeError:
            dmat = xgb.QuantileDMatrix(it, max_bin=int(args.xgb_max_bin))
    else:
        try:
            dmat = xgb.DMatrix(it, nthread=nthread)
        except TypeError:
            dmat = xgb.DMatrix(it)

    return dmat


def _infer_num_features_from_dataset(ds, args) -> int:
    loader = make_loader(ds, batch_size=1, shuffle=False, args=args, device=None)
    b = next(iter(loader))
    x = _flatten_time_if_needed(b["x"])
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

    is_gpu = (params.get("tree_method") == "gpu_hist")
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
    print(f"  • stream_max_rows/horizon.... {args.stream_max_rows_per_horizon}")
    print(f"  • stream_use_cupy............ {bool(args.stream_use_cupy)} (available={_HAVE_CUPY})")
    print(f"  • nthread.................... {params.get('nthread')}")
    print()

    print("[XGB stream] Params:")
    for k in sorted(params.keys()):
        print(f"  • {k:>18} = {params[k]}")

    boosters = []
    K = len(args.horizons)

    for k_idx, h in enumerate(args.horizons):
        h = int(h)
        header(f"Fit horizon h={h} (k={k_idx}) [STREAM]")

        dtrain = _make_stream_dmatrix(
            train_ds, args=args, horizon_k=k_idx, shuffle=bool(args.stream_shuffle_train),
            desc="stream_train", params=params
        )

        evals = [(dtrain, "train")]

        if val_ds is not None and len(val_ds) > 0 and bool(args.stream_use_val):
            dval = _make_stream_dmatrix(
                val_ds, args=args, horizon_k=k_idx, shuffle=False,
                desc="stream_val", params=params
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
    x = margins.astype(np.float64, copy=False)
    y = labels.astype(np.float64, copy=False)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    block_starts = []
    block_ends = []
    block_sumy = []
    block_w = []

    for i in range(x.size):
        block_starts.append(i)
        block_ends.append(i)
        block_sumy.append(float(y[i]))
        block_w.append(1.0)

        while len(block_sumy) >= 2:
            m1 = block_sumy[-2] / max(block_w[-2], 1e-12)
            m2 = block_sumy[-1] / max(block_w[-1], 1e-12)
            if m1 <= m2:
                break
            s = block_starts[-2]
            e = block_ends[-1]
            sy = block_sumy[-2] + block_sumy[-1]
            ww = block_w[-2] + block_w[-1]
            block_starts = block_starts[:-2] + [s]
            block_ends   = block_ends[:-2]   + [e]
            block_sumy   = block_sumy[:-2]   + [sy]
            block_w      = block_w[:-2]      + [ww]

    M = len(block_sumy)
    x_thr = np.empty(M, dtype=np.float64)
    y_val = np.empty(M, dtype=np.float64)

    for j in range(M):
        e = block_ends[j]
        x_thr[j] = x[e]
        y_val[j] = block_sumy[j] / max(block_w[j], 1e-12)

    return IsotonicCalibrator(x_thresholds=x_thr, y_values=y_val)


def save_calibrators(calibs: Dict[int, Any], method: str, out_path: str):
    payload = {"method": method, "per_horizon": {}}
    for h, c in calibs.items():
        h = int(h)
        if isinstance(c, PlattCalibrator):
            payload["per_horizon"][str(h)] = {"type": "platt", "a": c.a, "b": c.b}
        elif isinstance(c, IsotonicCalibrator):
            payload["per_horizon"][str(h)] = {
                "type": "isotonic",
                "x_thresholds": c.x_thresholds.tolist(),
                "y_values": c.y_values.tolist(),
            }
        else:
            payload["per_horizon"][str(h)] = {"type": "unknown"}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Calib] Saved calibrators -> {out_path}")


# ----------------------------------------------------------------------
# logits_fn builders (raw margins + optional calibration)
# ----------------------------------------------------------------------

def _batch_features_to_2d(batch: dict) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int]]:
    x = _flatten_time_if_needed(batch["x"])  # (B,F,H,W)
    y = batch["y"]                           # (B,K,H,W)
    B, F, H, W = x.shape
    K = int(y.shape[1])
    X2d = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, F)
    return X2d, (B, H, W, K, F)


def _valid_keep_indices(V2d: torch.Tensor) -> torch.Tensor:
    any_valid = V2d.any(dim=1)
    keep_idx = torch.nonzero(any_valid, as_tuple=False).view(-1)
    return keep_idx


def xgb_logits_fn_from_boosters(boosters: list, horizons: list[int], args, calibrators: Optional[Dict[int, Any]] = None):
    K = len(boosters)
    horizons_int = [int(h) for h in horizons]
    calibs = calibrators or {}

    def logits_fn(batch: dict):
        X2d, meta = _batch_features_to_2d(batch)
        B, H, W, K_meta, _F = meta
        if K_meta != K:
            raise RuntimeError(f"Batch K={K_meta} but boosters K={K}.")

        m = batch["mask"] > 0.5
        V2d = m.permute(0, 2, 3, 1).contiguous().view(B * H * W, K)

        keep_idx = _valid_keep_indices(V2d)
        logits_full = torch.zeros((B * H * W, K), device=batch["x"].device, dtype=torch.float32)

        if keep_idx.numel() == 0:
            return logits_full.view(B, H, W, K).permute(0, 3, 1, 2).contiguous()

        Xkeep = X2d.index_select(0, keep_idx)

        want_gpu = _is_cuda_device_str(args.xgb_predict_device)
        can_gpu = want_gpu and Xkeep.is_cuda and _HAVE_CUPY

        if can_gpu:
            Xc = _torch_to_cupy_2d(Xkeep.contiguous())
            for k in range(K):
                margin = _predict_margin_from_cupy(boosters[k], Xc).to(dtype=torch.float32)
                h = horizons_int[k]
                calib = calibs.get(int(h), None)
                if calib is None:
                    z = margin
                elif isinstance(calib, PlattCalibrator):
                    z = calib.apply_logits(margin)
                elif isinstance(calib, IsotonicCalibrator):
                    z = calib.apply_logits(margin)
                else:
                    z = margin
                logits_full[keep_idx, k] = z
        else:
            for k in range(K):
                margin = _predict_margin_gpu_if_possible(boosters[k], Xkeep, args).to(dtype=torch.float32)
                h = horizons_int[k]
                calib = calibs.get(int(h), None)
                if calib is None:
                    z = margin
                elif isinstance(calib, PlattCalibrator):
                    z = calib.apply_logits(margin)
                elif isinstance(calib, IsotonicCalibrator):
                    z = calib.apply_logits(margin)
                else:
                    z = margin
                logits_full[keep_idx, k] = z

        z4 = logits_full.view(B, H, W, K).permute(0, 3, 1, 2).contiguous()

        del X2d, V2d, Xkeep, keep_idx, logits_full
        return z4

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

        Y2d = y.permute(0, 2, 3, 1).contiguous().view(B * H * W, K)
        V2d = m.permute(0, 2, 3, 1).contiguous().view(B * H * W, K)

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
    }
    mpath = os.path.join(logdir, "xgb_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Save] Wrote {mpath}")


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
        coord_as_features=True,
        return_coords=False,
        peat_mask_source=args.peat_mask_source.strip() or "smap_wtd.zarr",
        coords_source=args.coords_source.strip() or "smap_wtd.zarr",
        coords_units=args.coords_units,
        holdout_region_source=(args.test_region_mask_source.strip() or None),
        holdout_t_end_index=test_t_end_index,
        holdout_min_fraction=float(args.test_region_min_fraction),
    )

    header("Building datasets")
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
# Args
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

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
                   help="Rows per streamed chunk fed into XGBoost DataIter.")
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
                   help="QuantileDMatrix is often best for gpu_hist (if available).")
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
    print(f"  • xgb_predict_device......... {args.xgb_predict_device}")
    print(f"  • eval_device................ {args.eval_device}")
    print(f"  • cupy available............. {_HAVE_CUPY}")

    n_features = _infer_num_features_from_dataset(train_ds, args)
    print(f"  • inferred n_features........ {n_features}")

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

        X_va = Y_va = V_va = None
        if val_loader is not None:
            monitor.set("Tabularizing val")
            X_va, Y_va, V_va = build_tabular_from_loader(
                val_loader, args=args, desc="val_tabularize", max_rows=args.max_val_rows
            )

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

    # Calibration fitting (optional)
    calibrators: Dict[int, Any] = {}
    if args.calibration_method != "none":
        if calib_ds is None:
            print("[WARN] calibration_method != none but calib_ds is None (calib_frac=0). Skipping calibration.")
        else:
            header("Calibration stage")
            print("We fit calibrators on a held-out subset of TRAIN PATCHES.\n")

            device = torch.device(args.eval_device)
            calib_loader = make_loader(calib_ds, batch_size=args.batch_size, shuffle=False, args=args, device=device)

            calibrators = fit_calibrators_from_loader(boosters, calib_loader, device=device, args=args)

            cal_path = os.path.join(args.logdir, "calibrators.json")
            save_calibrators(calibrators, method=args.calibration_method, out_path=cal_path)
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

    val_loader_eval  = make_loader(val_ds,  batch_size=args.batch_size, shuffle=False, args=args, device=device) if val_ds else None
    test_loader_eval = make_loader(test_ds, batch_size=args.batch_size, shuffle=False, args=args, device=device) if test_ds else None

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

    if test_loader_eval is not None:
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

    save_history_json(history, args.logdir)
    return args.logdir


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _warn_ignored_flags(args):
    # streaming flags ignored in tabularize mode
    stream_flags = {
        "stream_chunk_rows": args.stream_chunk_rows,
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
        if used:
            print("[WARN] You set streaming flags but --train-mode is not 'stream'. These flags will be ignored:")
            for u in used:
                print(f"  • {u}")

    if args.train_mode == "stream":
        # tabularization caps ignored
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
        print(f"  • stream_max_rows/horizon..... {args.stream_max_rows_per_horizon}")
        print(f"  • stream_use_val.............. {bool(args.stream_use_val)}")
        print(f"  • stream_use_cupy............. {bool(args.stream_use_cupy)}")
        print(f"  • xgb DataIter available...... {bool(_have_xgb_dataiter())}")
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

    header("Building datasets (once)")
    monitor = RAMMonitor()
    if not args.no_tqdm:
        monitor.start()
        monitor.set("Building datasets")

    train_ds, val_ds, test_ds = build_datasets(args)

    # Split train patches for calibration
    train_ds, calib_ds = split_train_for_calibration(train_ds, args)

    if not args.no_tqdm:
        monitor.stop()

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty after filtering/splitting.")

    header("Run configuration")
    print(f"  • seeds....................... {seeds}")
    print(f"  • data_seed (splits fixed).... {args.data_seed}")
    print(f"  • train_mode.................. {args.train_mode}")
    if args.train_mode == "stream":
        print(f"  • stream_chunk_rows........... {int(args.stream_chunk_rows):,}")
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


if __name__ == "__main__":
    main()
