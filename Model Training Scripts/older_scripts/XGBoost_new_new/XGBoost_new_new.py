"""
Spatiotemporal Peat Ignition — Random Forest per-tile model + optional post-hoc calibration

Replaces the per-tile MLP with per-horizon Random Forest binary classifiers.
Each pixel (tile) is treated independently:
  features: time×channels (flattened) + any extra features the dataset already provides
  label: fire (0/1) for each horizon

Calibration (optional; per-horizon):
  - Platt: p' = sigmoid(a * logit(p) + b)
  - Isotonic: p' = iso(p) using histogram bins + PAV (memory-safe)

Script-2-style reliability artifacts:
  - calibration histograms + ECE/MCE/Brier + ROC-from-hist
  - reliability bias slice for probs in [--reliability-bin-min, --reliability-bin-max]
  - saves NPZ + TensorBoard figures/scalars (+ optional saved plot files)

NOTE:
  Script-2 reliability slice defaults to max=0.060 (6%).
  If you meant 0.06% use: --reliability-bin-max 0.0006 (and reduce bin width).
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

from sklearn.ensemble import RandomForestClassifier
import joblib

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

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
class ProbCalibratorRF:
    method: str
    horizons: List[int]
    a: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    iso_x: Optional[List[Optional[np.ndarray]]] = None
    iso_y: Optional[List[Optional[np.ndarray]]] = None

    @property
    def K(self) -> int:
        return len(self.horizons)

    def apply_k(self, p: np.ndarray, k: int) -> np.ndarray:
        """Apply calibrator for a single horizon k to 1D probs."""
        if self.method == "none" or self.method is None:
            return p

        p = np.asarray(p, dtype=np.float64)
        p = np.clip(p, 1e-8, 1.0 - 1e-8)

        if self.method == "platt":
            if self.a is None or self.b is None:
                return p
            ak = float(self.a[k])
            bk = float(self.b[k])
            logit = np.log(p / (1.0 - p))
            z = logit * ak + bk
            out = 1.0 / (1.0 + np.exp(-z))
            return np.clip(out, 0.0, 1.0)

        if self.method == "isotonic":
            if self.iso_x is None or self.iso_y is None:
                return p
            x = self.iso_x[k]
            y = self.iso_y[k]
            if x is None or y is None or x.size < 2:
                return p
            out = np.interp(p, x, y)
            return np.clip(out, 0.0, 1.0)

        return p

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """probs: (N,K) in [0,1]"""
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2:
            raise ValueError(f"Expected probs (N,K), got {probs.shape}")
        out = np.empty_like(probs)
        for k in range(probs.shape[1]):
            out[:, k] = self.apply_k(probs[:, k], k)
        return out


@torch.no_grad()
def fit_calibrator_rf(
    models: List[RandomForestClassifier],
    calib_loader,
    args,
    monitor: Optional[RAMMonitor] = None
) -> Optional[ProbCalibratorRF]:
    method = getattr(args, "calibration_method", "none")
    if method == "none" or calib_loader is None:
        return None

    K = len(args.horizons)
    cal = ProbCalibratorRF(method=method, horizons=[int(h) for h in args.horizons], iso_x=[None]*K, iso_y=[None]*K)

    if method == "platt":
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

            losses = []
            for k in range(K):
                valid = m[:, k]  # (B,H,W)
                if valid.sum().item() == 0:
                    continue

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

            for k in range(K):
                valid = m[:, k]
                if valid.sum().item() == 0:
                    continue

                idx = torch.nonzero(valid, as_tuple=False)
                if idx.shape[0] == 0:
                    continue

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
# Metrics helpers (Script-2-style)
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


def _compute_roc_from_hist(counts: np.ndarray, true_sums: np.ndarray):
    counts = counts.astype(np.float64)
    pos = true_sums.astype(np.float64)
    neg = counts - pos
    P = pos.sum()
    N = neg.sum()
    if P <= 0 or N <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), float("nan")

    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    tpr = tp / P
    fpr = fp / N
    fpr = np.concatenate(([0.0], fpr))
    tpr = np.concatenate(([0.0], tpr))
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def save_reliability_bias_npz(split: str, metrics: Dict[str, Any], epoch: int, args) -> str:
    os.makedirs(args.logdir, exist_ok=True)
    out_path = os.path.join(args.logdir, f"reliability_bias_{split}_epoch{epoch}.npz")

    cal = metrics.get("calibration", {}) or {}
    per_h = (cal.get("per_horizon", {}) or {})

    # Overall arrays
    def _arr(key, default=None):
        v = cal.get(key, default)
        if v is None:
            return np.array([], dtype=np.float64)
        return np.asarray(v, dtype=np.float64)

    payload: Dict[str, Any] = {
        "epoch": np.array([epoch], dtype=np.int64),
        "split": np.array([split], dtype=object),

        "bin_centers": _arr("bin_centers"),
        "bin_pred": _arr("bin_pred"),
        "bin_true": _arr("bin_true"),
        "bin_count": _arr("bin_count"),

        "reliability_bias_pct": _arr("reliability_bias_pct"),
        "reliability_bias_count": _arr("reliability_bias_count"),
        "bin_centers_slice": _arr("bin_centers_slice"),

        "ece": np.array([cal.get("ece", float("nan"))], dtype=np.float64),
        "mce": np.array([cal.get("mce", float("nan"))], dtype=np.float64),
        "brier": np.array([cal.get("brier", float("nan"))], dtype=np.float64),
        "roc_auc": np.array([((cal.get("roc", {}) or {}).get("auc", float("nan")))], dtype=np.float64),

        # also store whole calibration dict as JSON (handy for exact reconstruction)
        "calibration_json": np.array([json.dumps(cal)], dtype=object),
    }

    # Per-horizon arrays (keyed by horizon)
    for h_str, ch in per_h.items():
        try:
            h = int(h_str)
        except Exception:
            continue
        ch = ch or {}
        roc = ch.get("roc", {}) or {}

        payload[f"h{h}_bin_pred"] = np.asarray(ch.get("bin_pred", []), dtype=np.float64)
        payload[f"h{h}_bin_true"] = np.asarray(ch.get("bin_true", []), dtype=np.float64)
        payload[f"h{h}_bin_count"] = np.asarray(ch.get("bin_count", []), dtype=np.float64)
        payload[f"h{h}_ece"] = np.array([ch.get("ece", float("nan"))], dtype=np.float64)
        payload[f"h{h}_mce"] = np.array([ch.get("mce", float("nan"))], dtype=np.float64)
        payload[f"h{h}_brier"] = np.array([ch.get("brier", float("nan"))], dtype=np.float64)
        payload[f"h{h}_roc_auc"] = np.array([roc.get("auc", float("nan"))], dtype=np.float64)

        payload[f"h{h}_reliability_bias_pct"] = np.asarray(ch.get("reliability_bias_pct", []), dtype=np.float64)
        payload[f"h{h}_reliability_bias_count"] = np.asarray(ch.get("reliability_bias_count", []), dtype=np.float64)
        payload[f"h{h}_bin_centers_slice"] = np.asarray(ch.get("bin_centers_slice", []), dtype=np.float64)

        payload[f"h{h}_roc_fpr"] = np.asarray(roc.get("fpr", []), dtype=np.float64)
        payload[f"h{h}_roc_tpr"] = np.asarray(roc.get("tpr", []), dtype=np.float64)

    np.savez_compressed(out_path, **payload)
    return out_path


def tb_log_reliability_bias_curve(
    writer: SummaryWriter,
    split: str,
    metrics: Dict[str, Any],
    epoch: int,
    args,
):
    cal = metrics.get("calibration", {}) or {}
    if not cal:
        return

    # overall calibration curve
    bin_pred = np.asarray(cal.get("bin_pred", []), dtype=np.float64)
    bin_true = np.asarray(cal.get("bin_true", []), dtype=np.float64)
    bin_count = np.asarray(cal.get("bin_count", []), dtype=np.float64)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--")
    if bin_count.size > 0:
        nz = bin_count > 0
        ax.plot(bin_pred[nz], bin_true[nz], marker="o", linewidth=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(f"{split} calibration (overall)")
    fig.tight_layout()
    writer.add_figure(f"{split}/calibration_curve", fig, global_step=epoch, close=True)

    # overall reliability bias slice
    centers = np.asarray(cal.get("bin_centers_slice", []), dtype=np.float64)
    bias = np.asarray(cal.get("reliability_bias_pct", []), dtype=np.float64)
    count = np.asarray(cal.get("reliability_bias_count", []), dtype=np.float64)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    if centers.size > 0 and bias.size == centers.size:
        ax2.axhline(0.0, linestyle="--")
        ax2.plot(centers, bias, marker="o", linewidth=1)
    ax2.set_xlabel("Predicted probability (slice)")
    ax2.set_ylabel("Bias (pred - true) [%]")
    ax2.set_title(f"{split} reliability bias slice (overall)")
    fig2.tight_layout()
    writer.add_figure(f"{split}/reliability_bias_slice", fig2, global_step=epoch, close=True)


def save_and_log_calibration(
    split: str,
    metrics: Dict[str, Any],
    epoch: int,
    args,
    writer: Optional[SummaryWriter] = None,
):
    # NPZ always
    npz_path = save_reliability_bias_npz(split, metrics, epoch, args)
    print(f"[{split}] Saved reliability NPZ: {npz_path}")

    # TB scalars + figures
    if writer is not None:
        writer.add_scalar(f"{split}/logloss", float(metrics.get("logloss", float("nan"))), epoch)
        writer.add_scalar(f"{split}/roc_auc", float(metrics.get("roc_auc", float("nan"))), epoch)
        writer.add_scalar(f"{split}/ece", float(metrics.get("ece", float("nan"))), epoch)
        writer.add_scalar(f"{split}/mce", float(metrics.get("mce", float("nan"))), epoch)
        writer.add_scalar(f"{split}/brier", float(metrics.get("brier", float("nan"))), epoch)

        # per-horizon scalars
        per_h = metrics.get("per_horizon", {}) or {}
        for h, mh in per_h.items():
            try:
                h_int = int(h)
            except Exception:
                continue
            writer.add_scalar(f"{split}/h{h_int}_logloss", float(mh.get("logloss", float("nan"))), epoch)
            writer.add_scalar(f"{split}/h{h_int}_roc_auc", float(mh.get("roc_auc", float("nan"))), epoch)
            writer.add_scalar(f"{split}/h{h_int}_ece", float(mh.get("ece", float("nan"))), epoch)
            writer.add_scalar(f"{split}/h{h_int}_brier", float(mh.get("brier", float("nan"))), epoch)

        tb_log_reliability_bias_curve(writer, split, metrics, epoch, args)

    # Optional save plots to disk
    if getattr(args, "plot_metrics", False):
        fmt = getattr(args, "plot_file_format", "png")
        dpi = int(getattr(args, "plot_dpi", 150))
        plot_dir = os.path.join(args.logdir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        cal = metrics.get("calibration", {}) or {}
        bin_pred = np.asarray(cal.get("bin_pred", []), dtype=np.float64)
        bin_true = np.asarray(cal.get("bin_true", []), dtype=np.float64)
        bin_count = np.asarray(cal.get("bin_count", []), dtype=np.float64)
        nz = bin_count > 0

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--")
        if bin_pred.size > 0:
            ax.plot(bin_pred[nz], bin_true[nz], marker="o", linewidth=1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical frequency")
        ax.set_title(f"{split} calibration (overall)")
        fig.tight_layout()
        out = os.path.join(plot_dir, f"calibration_{split}_epoch{epoch}.{fmt}")
        fig.savefig(out, dpi=dpi)
        plt.close(fig)

        centers = np.asarray(cal.get("bin_centers_slice", []), dtype=np.float64)
        bias = np.asarray(cal.get("reliability_bias_pct", []), dtype=np.float64)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.axhline(0.0, linestyle="--")
        if centers.size > 0 and bias.size == centers.size:
            ax2.plot(centers, bias, marker="o", linewidth=1)
        ax2.set_xlabel("Predicted probability (slice)")
        ax2.set_ylabel("Bias (pred - true) [%]")
        ax2.set_title(f"{split} reliability bias slice (overall)")
        fig2.tight_layout()
        out2 = os.path.join(plot_dir, f"reliability_bias_{split}_epoch{epoch}.{fmt}")
        fig2.savefig(out2, dpi=dpi)
        plt.close(fig2)


# ----------------------------------------------------------------------
# Evaluation (REPLACED with Script-2-style reliability)
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate_rf(models, loader, args, calibrator=None, use_tqdm=True, monitor=None):
    K = len(args.horizons)

    # Script-2-style reliability bins
    bin_width = float(args.reliability_bin_width)
    num_bins = max(1, int(math.ceil(1.0 / bin_width)))
    bin_centers = (np.arange(num_bins, dtype=np.float64) + 0.5) * bin_width

    rmin = float(args.reliability_bin_min)
    rmax = float(args.reliability_bin_max)
    minc = int(args.reliability_min_count)

    rel_counts = np.zeros((K, num_bins), dtype=np.float64)
    rel_pred_sums = np.zeros_like(rel_counts)
    rel_true_sums = np.zeros_like(rel_counts)

    brier_sums = np.zeros(K, dtype=np.float64)
    logloss_sum_h = np.zeros(K, dtype=np.float64)
    logloss_cnt_h = np.zeros(K, dtype=np.float64)

    prob_total_h = np.zeros(K, dtype=np.float64)
    prob_inrange_h = np.zeros(K, dtype=np.float64)
    prob_above_h = np.zeros(K, dtype=np.float64)
    prob_below_h = np.zeros(K, dtype=np.float64)

    tp_total = fp_total = fn_total = tn_total = 0
    tp_h = [0]*K
    fp_h = [0]*K
    fn_h = [0]*K
    tn_h = [0]*K

    iterator = wrap_loader(loader, desc="eval", use_tqdm=use_tqdm, position=3)
    for batch in iterator:
        if monitor is not None:
            monitor.set_status("Evaluating")

        x = _flatten_time_if_needed(batch["x"])  # (B,F,H,W)
        y = (batch["y"] > 0.5)
        m = (batch["mask"] > 0.5)

        for k in range(K):
            valid = m[:, k]
            if valid.sum().item() == 0:
                continue

            idx = torch.nonzero(valid, as_tuple=False)
            if idx.numel() == 0:
                continue

            if args.eval_max_tiles > 0 and idx.shape[0] > args.eval_max_tiles:
                idx = idx[torch.randperm(idx.shape[0])[: args.eval_max_tiles]]

            bs, ys, xs = idx[:, 0], idx[:, 1], idx[:, 2]

            feat = x[bs, :, ys, xs].contiguous().cpu().numpy().astype(np.float32)
            t = y[bs, k, ys, xs].contiguous().cpu().numpy().astype(np.float64)

            p = models[k].predict_proba(feat)[:, 1].astype(np.float64)
            p = np.clip(p, 0.0, 1.0)

            # post-hoc calibration (per-horizon)
            if calibrator is not None:
                p2 = calibrator.apply_k(p, k)
            else:
                p2 = p
            p2 = np.clip(p2, 0.0, 1.0)

            # confusion
            pred = (p2 >= float(args.metrics_threshold))
            tt = (t > 0.5)
            tp = int((pred & tt).sum())
            fp = int((pred & ~tt).sum())
            fn = int((~pred & tt).sum())
            tn = int((~pred & ~tt).sum())

            tp_total += tp
            fp_total += fp
            fn_total += fn
            tn_total += tn
            tp_h[k] += tp
            fp_h[k] += fp
            fn_h[k] += fn
            tn_h[k] += tn

            # prob_range counts (Script 2-style slice)
            n = float(p2.size)
            prob_total_h[k] += n
            prob_inrange_h[k] += float(((p2 >= rmin) & (p2 <= rmax)).sum())
            prob_above_h[k] += float((p2 > rmax).sum())
            prob_below_h[k] += float((p2 < rmin).sum())

            # brier + logloss
            brier_sums[k] += float(((p2 - t) ** 2).sum())
            eps = 1e-7
            pc = np.clip(p2, eps, 1.0 - eps)
            ll = -(t * np.log(pc) + (1.0 - t) * np.log(1.0 - pc))
            logloss_sum_h[k] += float(ll.sum())
            logloss_cnt_h[k] += n

            # reliability hist
            bin_idx = np.floor(p2 / bin_width).astype(np.int64)
            bin_idx = np.clip(bin_idx, 0, num_bins - 1)

            rel_counts[k] += np.bincount(bin_idx, minlength=num_bins).astype(np.float64)
            rel_pred_sums[k] += np.bincount(bin_idx, weights=p2, minlength=num_bins).astype(np.float64)
            rel_true_sums[k] += np.bincount(bin_idx, weights=t, minlength=num_bins).astype(np.float64)

    # per-horizon metrics + per-horizon calibration dict
    per_h = {}
    calib_per_h = {}

    b0 = int(math.floor(rmin / bin_width))
    b1 = int(math.ceil(rmax / bin_width))
    b0 = max(0, min(b0, num_bins))
    b1 = max(0, min(b1, num_bins))

    for k, h in enumerate(args.horizons):
        mh = _compute_basic_metrics(tp_h[k], fp_h[k], fn_h[k], tn_h[k])

        counts = rel_counts[k]
        if counts.sum() > 0:
            nz = counts > 0
            pred_mean = np.zeros_like(counts)
            true_mean = np.zeros_like(counts)
            pred_mean[nz] = rel_pred_sums[k][nz] / counts[nz]
            true_mean[nz] = rel_true_sums[k][nz] / counts[nz]

            gap = np.abs(pred_mean - true_mean)
            total = counts.sum()
            ece = float((counts[nz] / total * gap[nz]).sum())
            mce = float(gap[nz].max()) if np.any(nz) else float("nan")
            brier = float(brier_sums[k] / max(prob_total_h[k], 1.0))
            fpr, tpr, auc = _compute_roc_from_hist(counts, rel_true_sums[k])

            bias_pct = (pred_mean - true_mean) * 100.0
            bias_slice = bias_pct[b0:b1].copy()
            count_slice = counts[b0:b1].copy()
            if minc > 0:
                bias_slice[count_slice < minc] = np.nan

            calib_per_h[int(h)] = {
                "bin_pred": pred_mean.tolist(),
                "bin_true": true_mean.tolist(),
                "bin_count": counts.tolist(),
                "ece": ece,
                "mce": mce,
                "brier": brier,
                "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc)},
                "reliability_bias_pct": bias_slice.tolist(),
                "reliability_bias_count": count_slice.tolist(),
                "bin_centers_slice": bin_centers[b0:b1].tolist(),
            }

            mh["logloss"] = float(logloss_sum_h[k] / max(logloss_cnt_h[k], 1.0))
            mh["roc_auc"] = float(auc)
            mh["ece"] = float(ece)
            mh["brier"] = float(brier)
        else:
            mh["logloss"] = float("nan")
            mh["roc_auc"] = float("nan")
            mh["ece"] = float("nan")
            mh["brier"] = float("nan")
            calib_per_h[int(h)] = {
                "bin_pred": [], "bin_true": [], "bin_count": [],
                "ece": float("nan"), "mce": float("nan"), "brier": float("nan"),
                "roc": {"fpr": [], "tpr": [], "auc": float("nan")},
                "reliability_bias_pct": [], "reliability_bias_count": [], "bin_centers_slice": [],
            }

        per_h[int(h)] = mh

    # overall calibration by summing histograms
    overall_counts = rel_counts.sum(axis=0)
    overall_pred_sum = rel_pred_sums.sum(axis=0)
    overall_true_sum = rel_true_sums.sum(axis=0)

    if overall_counts.sum() > 0:
        nz = overall_counts > 0
        pred_mean = np.zeros_like(overall_counts)
        true_mean = np.zeros_like(overall_counts)
        pred_mean[nz] = overall_pred_sum[nz] / overall_counts[nz]
        true_mean[nz] = overall_true_sum[nz] / overall_counts[nz]

        gap = np.abs(pred_mean - true_mean)
        total = overall_counts.sum()
        ece_o = float((overall_counts[nz] / total * gap[nz]).sum())
        mce_o = float(gap[nz].max()) if np.any(nz) else float("nan")
        brier_o = float(brier_sums.sum() / max(prob_total_h.sum(), 1.0))
        fpr_o, tpr_o, auc_o = _compute_roc_from_hist(overall_counts, overall_true_sum)

        bias_pct = (pred_mean - true_mean) * 100.0
        bias_slice = bias_pct[b0:b1].copy()
        count_slice = overall_counts[b0:b1].copy()
        if minc > 0:
            bias_slice[count_slice < minc] = np.nan
    else:
        pred_mean = np.zeros_like(overall_counts)
        true_mean = np.zeros_like(overall_counts)
        ece_o = mce_o = brier_o = auc_o = float("nan")
        fpr_o = tpr_o = np.array([], dtype=np.float64)
        bias_slice = np.array([], dtype=np.float64)
        count_slice = np.array([], dtype=np.float64)

    # Script-2-style prob_range summary
    prob_total = float(prob_total_h.sum())
    inrange = float(prob_inrange_h.sum())
    above = float(prob_above_h.sum())
    below = float(prob_below_h.sum())

    prob_range_per_h = {}
    for k, h in enumerate(args.horizons):
        tot = float(prob_total_h[k])
        prob_range_per_h[int(h)] = {
            "count": float(prob_inrange_h[k]),
            "total": tot,
            "fraction": (float("nan") if tot <= 0 else float(prob_inrange_h[k] / tot)),
            "above_max_count": float(prob_above_h[k]),
            "above_max_fraction": (float("nan") if tot <= 0 else float(prob_above_h[k] / tot)),
            "below_min_count": float(prob_below_h[k]),
            "below_min_fraction": (float("nan") if tot <= 0 else float(prob_below_h[k] / tot)),
        }

    overall = _compute_basic_metrics(tp_total, fp_total, fn_total, tn_total)

    metrics = {
        **overall,
        "per_horizon": per_h,
        "logloss": (float("nan") if logloss_cnt_h.sum() <= 0 else float(logloss_sum_h.sum() / max(logloss_cnt_h.sum(), 1.0))),
        "roc_auc": float(auc_o),
        "ece": float(ece_o),
        "mce": float(mce_o),
        "brier": float(brier_o),
        "prob_range": {
            "min": rmin,
            "max": rmax,
            "count": inrange,
            "total": prob_total,
            "fraction": (float("nan") if prob_total <= 0 else inrange / prob_total),
            "above_max_count": above,
            "above_max_fraction": (float("nan") if prob_total <= 0 else above / prob_total),
            "below_min_count": below,
            "below_min_fraction": (float("nan") if prob_total <= 0 else below / prob_total),
            "per_horizon": prob_range_per_h,
        },
        "calibration": {
            "bin_centers": bin_centers.tolist(),
            "bin_pred": pred_mean.tolist(),
            "bin_true": true_mean.tolist(),
            "bin_count": overall_counts.tolist(),
            "ece": float(ece_o),
            "mce": float(mce_o),
            "brier": float(brier_o),
            "roc": {"fpr": fpr_o.tolist(), "tpr": tpr_o.tolist(), "auc": float(auc_o)},
            "per_horizon": calib_per_h,
            "reliability_bias_pct": bias_slice.tolist(),
            "reliability_bias_count": count_slice.tolist(),
            "bin_centers_slice": bin_centers[b0:b1].tolist(),
        },
    }

    val_loss = metrics["logloss"]
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
            if args.rf_max_train_tiles > 0 and n_collected[k] >= args.rf_max_train_tiles:
                continue

            for b in range(B):
                valid_b = m[b, k]  # (H,W)
                nv = int(valid_b.sum().item())
                if nv <= 0:
                    continue

                if args.rf_tiles_per_patch > 0:
                    n_take = min(args.rf_tiles_per_patch, nv)
                else:
                    frac = float(args.rf_sample_frac)
                    n_take = int(max(1, round(nv * frac)))

                if args.rf_max_train_tiles > 0:
                    remaining = args.rf_max_train_tiles - n_collected[k]
                    if remaining <= 0:
                        break
                    n_take = min(n_take, remaining)

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

                lab = y[b, k, ys, xs].contiguous().cpu().numpy().astype(np.int64)

                X_list[k].append(feat)
                y_list[k].append(lab)
                n_collected[k] += int(lab.size)

        if args.rf_max_train_tiles > 0 and all(n >= args.rf_max_train_tiles for n in n_collected):
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
                "epoch", "split", "loss",
                "logloss", "roc_auc", "ece", "mce", "brier",
                "acc", "prec", "rec", "f1",
                "calib",
                "prob_range_min", "prob_range_max",
                "inrange_count", "total_count", "inrange_frac",
                "above_max_count", "above_max_frac",
                "below_min_count", "below_min_frac",
            ])
    return path


def append_metrics_csv(path: str, epoch: int, split: str, loss: float, metrics: Dict[str, Any], calib_method: str):
    pr = metrics.get("prob_range", {}) or {}
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            epoch, split, loss,
            metrics.get("logloss", float("nan")),
            metrics.get("roc_auc", float("nan")),
            metrics.get("ece", float("nan")),
            metrics.get("mce", float("nan")),
            metrics.get("brier", float("nan")),
            metrics.get("accuracy", float("nan")),
            metrics.get("precision", float("nan")),
            metrics.get("recall", float("nan")),
            metrics.get("f1", float("nan")),
            calib_method,
            pr.get("min", float("nan")),
            pr.get("max", float("nan")),
            pr.get("count", float("nan")),
            pr.get("total", float("nan")),
            pr.get("fraction", float("nan")),
            pr.get("above_max_count", float("nan")),
            pr.get("above_max_fraction", float("nan")),
            pr.get("below_min_count", float("nan")),
            pr.get("below_min_fraction", float("nan")),
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

    # --- Random Forest training sampling ---
    p.add_argument("--rf-sample-frac", type=float, default=0.02,
                   help="Fraction of valid pixels per patch to include in training samples (if tiles_per_patch=0).")
    p.add_argument("--rf-tiles-per-patch", type=int, default=0,
                   help="If >0, sample this many valid pixels per patch (overrides rf_sample_frac).")
    p.add_argument("--rf-max-train-tiles", type=int, default=300_000,
                   help="Max training samples PER HORIZON (caps memory).")

    # --- Random Forest hyperparameters ---
    p.add_argument("--rf-n-estimators", type=int, default=400,
                   help="Number of trees in the forest (default: 400)")
    p.add_argument("--rf-max-depth", type=int, default=20,
                   help="Max tree depth; 0=unlimited (default: 20)")
    p.add_argument("--rf-max-features", default="sqrt",
                   help="Features per split: int/float/'sqrt'/'log2'/None (default: 'sqrt')")
    p.add_argument("--rf-max-samples", type=float, default=0.8,
                   help="Fraction of samples to draw per tree (default: 0.8)")
    p.add_argument("--rf-min-samples-leaf", type=int, default=1,
                   help="Min samples required at a leaf node (default: 1)")
    p.add_argument("--rf-min-impurity-decrease", type=float, default=0.0,
                   help="Min impurity decrease required for split (default: 0.0)")
    p.add_argument("--rf-n-jobs", type=int, default=-1,
                   help="Number of parallel jobs; -1=all CPU cores (default: -1)")
    p.add_argument("--rf-verbosity", type=int, default=1,
                   help="Verbosity level: 0=silent, 1=progress, 2+=debug (default: 1)")

    # Imbalance handling
    p.add_argument("--rf-use-class-weight", action="store_true",
                   help="If set, uses class_weight='balanced' to auto-weight classes.")
    p.add_argument("--rf-use-sample-weight", action="store_true",
                   help="If set, uses manual sample weighting (n_neg/n_pos) for positive class.")

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

    # logdir + saving
    p.add_argument("--logdir", default="runs/peat_rf")

    # reliability / calibration curve bins (match Script 2)
    p.add_argument("--reliability-bin-width", type=float, default=0.005)
    p.add_argument("--reliability-bin-min", type=float, default=0.005)
    p.add_argument("--reliability-bin-max", type=float, default=0.060)
    p.add_argument("--reliability-min-count", type=int, default=50)

    # plots + tensorboard
    p.add_argument("--plot-metrics", action="store_true")
    p.add_argument("--plot-file-format", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--plot-dpi", type=int, default=150)
    p.add_argument("--no-tensorboard", action="store_true")

    return p.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.logdir, exist_ok=True)

    print_diagnostic_header("Reliability slice sanity check")
    print_diagnostic_item("--reliability-bin-width", args.reliability_bin_width, indent=1)
    print_diagnostic_item("--reliability-bin-min", args.reliability_bin_min, indent=1)
    print_diagnostic_item("--reliability-bin-max", args.reliability_bin_max, indent=1)
    print("NOTE: max=0.060 means 6%. If you meant 0.06%, use 0.0006 and reduce bin width.")

    monitor = RAMMonitor(device_id=0)
    if not args.no_tqdm:
        monitor.start()

    writer = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=args.logdir)
        print(f"[TB] Logging to: {args.logdir}")

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
        print_diagnostic_item("Random Forest per-horizon classifiers", "binary classification", indent=1)
        print_diagnostic_item("in_features", in_features, indent=1)
        print_diagnostic_item("horizons", args.horizons, indent=1)

        print_diagnostic_header("Sampling training tiles")
        if monitor is not None:
            monitor.set_status("Sampling train tiles")

        train_samples = collect_tabular_samples(train_ds, args, split_name="train", monitor=monitor)

        # Train per horizon
        models: List[RandomForestClassifier] = []
        for k, h in enumerate(args.horizons):
            Xk = train_samples.X[k]
            yk = train_samples.y[k]
            if yk.size == 0:
                raise RuntimeError(f"No training samples collected for horizon {h}. Increase rf_sample_frac or relax filtering.")

            # Build Random Forest parameters
            params = dict(
                n_estimators=int(args.rf_n_estimators),
                max_depth=int(args.rf_max_depth) if args.rf_max_depth > 0 else None,
                max_features=args.rf_max_features,
                max_samples=float(args.rf_max_samples) if args.rf_max_samples < 1.0 else None,
                min_samples_leaf=int(args.rf_min_samples_leaf),
                min_impurity_decrease=float(args.rf_min_impurity_decrease),
                n_jobs=int(args.rf_n_jobs),
                random_state=args.seed,
                verbose=int(args.rf_verbosity),
                bootstrap=True,
            )

            # Handle class imbalance
            sample_weight = None
            if args.rf_use_class_weight:
                params["class_weight"] = "balanced"
            elif args.rf_use_sample_weight:
                n_pos = float((yk == 1).sum())
                n_neg = float((yk == 0).sum())
                if n_pos > 0:
                    weight_ratio = n_neg / max(n_pos, 1.0)
                    sample_weight = np.ones(len(yk), dtype=np.float32)
                    sample_weight[yk == 1] = weight_ratio

            print_diagnostic_header(f"Training RF (horizon={h})")
            print_diagnostic_item("Samples", human_int(int(yk.size)), indent=1)
            if params.get("class_weight") == "balanced":
                print_diagnostic_item("class_weight", "balanced", indent=1)
            elif sample_weight is not None:
                print_diagnostic_item("sample_weight (pos)", f"{weight_ratio:.2f}", indent=1)

            if monitor is not None:
                monitor.set_status(f"Training RF h={h}")

            clf = RandomForestClassifier(**params)
            if sample_weight is not None:
                clf.fit(Xk, yk, sample_weight=sample_weight)
            else:
                clf.fit(Xk, yk)
            models.append(clf)

        # Optional calibration
        calib_loader = make_loader(calib_ds, batch_size=args.batch_size, shuffle=False, args=args) if calib_ds is not None else None
        calibrator = None
        calib_method = "none"
        if calib_loader is not None and args.calibration_method != "none":
            calib_method = args.calibration_method
            print_diagnostic_header(f"Calibration ({calib_method})")
            calibrator = fit_calibrator_rf(models, calib_loader, args, monitor=monitor)

        # Eval
        metrics_csv_path = init_metrics_csv(args.logdir)

        val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False, args=args) if val_ds is not None else None
        test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False, args=args) if test_ds is not None else None

        if val_loader is not None:
            if monitor is not None:
                monitor.set_status("Eval(val)")
            val_loss, val_metrics = evaluate_rf(models, val_loader, args, calibrator=calibrator, use_tqdm=not args.no_tqdm, monitor=monitor)

            save_and_log_calibration("val", val_metrics, epoch=1, args=args, writer=writer)

            pr = val_metrics.get("prob_range", {}) or {}
            print_diagnostic_header("Validation")
            print(
                f"[Val] calib={calib_method} logloss={val_metrics.get('logloss', float('nan')):.6f} "
                f"auc={val_metrics.get('roc_auc', float('nan')):.4f} ece={val_metrics.get('ece', float('nan')):.4f} "
                f"brier={val_metrics.get('brier', float('nan')):.6f} "
                f"acc={val_metrics['accuracy']:.4f} prec={val_metrics['precision']:.4f} "
                f"rec={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}"
            )
            print(
                f"[Val] P(prob in [{pr.get('min')},{pr.get('max')}]) : "
                f"{pr.get('count', 0):.0f}/{pr.get('total', 0):.0f} ({100.0*pr.get('fraction', 0.0):.4f}%) | "
                f"above: {pr.get('above_max_count', 0):.0f} ({100.0*pr.get('above_max_fraction', 0.0):.4f}%) | "
                f"below: {pr.get('below_min_count', 0):.0f} ({100.0*pr.get('below_min_fraction', 0.0):.4f}%)"
            )
            for h, hh in (pr.get("per_horizon", {}) or {}).items():
                print(
                    f"  [Val h={h}] in: {hh['count']:.0f}/{hh['total']:.0f} ({100.0*hh['fraction']:.4f}%) | "
                    f"above: {hh['above_max_count']:.0f} ({100.0*hh['above_max_fraction']:.4f}%) | "
                    f"below: {hh['below_min_count']:.0f} ({100.0*hh['below_min_fraction']:.4f}%)"
                )

            append_metrics_csv(metrics_csv_path, 1, "val", val_loss, val_metrics, calib_method)

        if test_loader is not None:
            if monitor is not None:
                monitor.set_status("Eval(test)")
            test_loss, test_metrics = evaluate_rf(models, test_loader, args, calibrator=calibrator, use_tqdm=not args.no_tqdm, monitor=monitor)

            save_and_log_calibration("test", test_metrics, epoch=1, args=args, writer=writer)

            pr = test_metrics.get("prob_range", {}) or {}
            print_diagnostic_header("Test")
            print(
                f"[Test] calib={calib_method} logloss={test_metrics.get('logloss', float('nan')):.6f} "
                f"auc={test_metrics.get('roc_auc', float('nan')):.4f} ece={test_metrics.get('ece', float('nan')):.4f} "
                f"brier={test_metrics.get('brier', float('nan')):.6f} "
                f"acc={test_metrics['accuracy']:.4f} prec={test_metrics['precision']:.4f} "
                f"rec={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}"
            )
            print(
                f"[Test] P(prob in [{pr.get('min')},{pr.get('max')}]) : "
                f"{pr.get('count', 0):.0f}/{pr.get('total', 0):.0f} ({100.0*pr.get('fraction', 0.0):.4f}%) | "
                f"above: {pr.get('above_max_count', 0):.0f} ({100.0*pr.get('above_max_fraction', 0.0):.4f}%) | "
                f"below: {pr.get('below_min_count', 0):.0f} ({100.0*pr.get('below_min_fraction', 0.0):.4f}%)"
            )
            for h, hh in (pr.get("per_horizon", {}) or {}).items():
                print(
                    f"  [Test h={h}] in: {hh['count']:.0f}/{hh['total']:.0f} ({100.0*hh['fraction']:.4f}%) | "
                    f"above: {hh['above_max_count']:.0f} ({100.0*hh['above_max_fraction']:.4f}%) | "
                    f"below: {hh['below_min_count']:.0f} ({100.0*hh['below_min_fraction']:.4f}%)"
                )

            append_metrics_csv(metrics_csv_path, 1, "test", test_loss, test_metrics, calib_method)

        # Save models + calibrator
        os.makedirs(args.logdir, exist_ok=True)
        for k, h in enumerate(args.horizons):
            path = os.path.join(args.logdir, f"rf_h{h}.joblib")
            joblib.dump(models[k], path)
            print(f"[Save] Random Forest h={h} → {path}")

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

    if writer is not None:
        writer.flush()
        writer.close()

    if monitor is not None:
        monitor.stop()


if __name__ == "__main__":
    main()

