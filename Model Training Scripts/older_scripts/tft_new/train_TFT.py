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
- Simple LSTM-based model over time per pixel
  (per-pixel LSTM sequence model producing (B, K, H, W) logits)

Monitoring:
- Storage type detection (SSD vs HDD vs network FS) for Zarr roots
- Explicit multiprocessing context reporting (forkserver/spawn/etc.)
- Per-batch data loading & transfer timing
- DataLoader queue tracker: how many batches are ready in RAM vs currently loading

Metrics & logging:
- Validation metrics after each epoch: accuracy, precision, recall, F1, etc.
- Per-horizon validation metrics (per forecast horizon)
- CSV logging of metrics per epoch (logs/metrics_log.csv)
- TensorBoard logging for:
    * Per-step loss, timings, throughput
    * RAM, GPU memory, loader queue stats
    * Epoch losses & validation metrics (overall + per horizon)
    * “Best so far” validation metrics

NEW in this revision:
- JointPeatDataset is configured with coord_as_features=True:
    * x now includes 4 extra channels built from sin/cos(lat, lon),
      repeated across time steps.
- Optional 3-way split (train / val / test):
    * Use --split for train fraction.
    * Optional --val-frac for explicit validation fraction.
    * Test fraction = 1 - split - val_frac (if positive).
- Training loop bug fixed so each epoch uses ALL batches
- Validation/summary now prints per-horizon metrics.
- Per-horizon metrics now also report TP / FP / FN / TN counts.
- Patch rebalancing and curriculum:
    * Optional patch-level sampler using cached fire/non-fire flags
    * Optional PROGRESSIVE curriculum: smoothly ramp in non-fire patches
      over configurable epochs, with configurable start epoch and
      non-fire weights.
- Per-pixel class reweighting in the loss:
    * Dynamic N_bg/N_fire weighting (clipped) for fire pixels.
- NEW: LSTMSpatiotemporal model instead of CNN UNet
- NEW: True SMOTE-driven patch oversampling (--use-smote):
    * Build a low-dimensional descriptor for each patch (global means).
    * Run imblearn.SMOTE once on descriptors + patch labels.
    * Map synthetic SMOTE points to nearest real patches and turn that
      into extra sampling weight for those patches (no synthetic tensors
      are stored).
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
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from joint_peat_dataset_builder import JointPeatDataset  # latest dataset with coord_as_features + test split

# Optional dependencies for true SMOTE-based sampling
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    NearestNeighbors = None


# ----------------------------------------------------------------------
# Small utils
# ----------------------------------------------------------------------

def exists(x):
    return x is not None


def default(val, d):
    """Return val if not None, else d."""
    return d if val is None else val


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

    def __enter__(selfself):
        if selfself.verbose:
            print(f"\n[DIAG] Starting: {selfself.label}")
        selfself.start_time = time.time()
        if selfself.track_ram:
            selfself.start_ram = get_ram_usage()
        if selfself.track_gpu and torch.cuda.is_available():
            selfself.start_gpu = get_gpu_memory(selfself.device_index)
        return selfself

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

    except Exception as e:
        return f"Unknown (error probing storage: {e})"


# ----------------------------------------------------------------------
# TFT building blocks
# ----------------------------------------------------------------------

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) from the TFT paper, simplified for
    continuous inputs and optional context.

    GRN(x, c) = LayerNorm(residual + GLU( W2( ELU( W1([x, c]) ) ) ))
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim) if output_dim is not None else int(input_dim)
        self.context_dim = int(context_dim) if context_dim is not None else None

        in_linear_dim = self.input_dim + (self.context_dim or 0)
        self.linear1 = nn.Linear(in_linear_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        # Gated Linear Unit acting on the transformed signal
        self.gate_linear = nn.Linear(self.output_dim, self.output_dim)
        self.gate_sigmoid = nn.Sigmoid()

        # Optional skip projection if dimensions differ
        self.skip_linear = (
            nn.Linear(self.input_dim, self.output_dim)
            if self.input_dim != self.output_dim
            else None
        )
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:       (..., input_dim)
        context: (..., context_dim) or None
        """
        residual = x

        if context is not None:
            # Broadcast context along the leading dimensions if needed
            if context.dim() < x.dim():
                # e.g. x: (B,T,D), context: (B,Dc) -> (B,T,Dc)
                for _ in range(x.dim() - context.dim()):
                    context = context.unsqueeze(1)
            x_in = torch.cat([x, context], dim=-1)
        else:
            x_in = x

        h = self.linear1(x_in)
        h = self.elu(h)
        h = self.dropout(h)
        h = self.linear2(h)  # (..., output_dim)

        # Gated Linear Unit
        gate = self.gate_sigmoid(self.gate_linear(h))
        h = h * gate

        if self.skip_linear is not None:
            residual = self.skip_linear(residual)

        out = self.layer_norm(residual + h)
        return out


class TemporalVariableSelection(nn.Module):
    """
    Memory-friendly variable selection network for continuous covariates.

    Given x of shape (B, T, C), we:
      - Use a GRN to produce per-variable logits
      - Softmax over C to get selection weights
      - Multiply weights * x (feature-wise)
      - Linearly project back to hidden_dim

    This preserves the spirit of TFT's variable selection (per-variable
    softmax and gating) without materializing (B,T,C,hidden_dim).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.grn = GatedResidualNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.input_dim,  # logits per variable
            context_dim=context_dim,
            dropout=dropout,
        )
        self.proj = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ):
        """
        x:       (B, T, C)
        context: (B, context_dim) or None

        Returns:
          rep:      (B, T, hidden_dim)  - reweighted temporal features
          weights:  (B, T, C)           - per-variable selection weights
        """
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)

        if context is not None:
            # Broadcast context across time
            c = context
            if c.dim() == 2:  # (B, Dc)
                c = c.unsqueeze(1).expand(B, T, c.shape[-1])  # (B,T,Dc)
            c_flat = c.reshape(B * T, -1)
        else:
            c_flat = None

        logits = self.grn(x_flat, context=c_flat)  # (B*T, C)
        weights = F.softmax(logits, dim=-1).reshape(B, T, C)

        x_weighted = x * weights  # (B, T, C)
        rep = self.proj(x_weighted)  # (B, T, hidden_dim)

        return rep, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable multi-head attention as in TFT:
    - Shared value projection across heads
    - Different Q/K per head
    - Heads aggregated additively then projected.

    Inputs:
      x: (B, T, d_model)

    Returns:
      out:        (B, T, d_model)
      attn_mean:  (B, num_heads, T, T)  (for interpretability/debugging)
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model > 0, "d_model must be positive"
        assert num_heads > 0, "num_heads must be positive"

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dropout = nn.Dropout(dropout)

        # Shared value projection
        self.v_linear = nn.Linear(self.d_model, self.d_model)

        # Per-head Q/K projections
        self.q_linears = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_model) for _ in range(self.num_heads)]
        )
        self.k_linears = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_model) for _ in range(self.num_heads)]
        )

        # Output projection
        self.out_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x:    (B, T, d_model)
        mask: (B, T, T) where 0 = masked, 1 = keep (optional)
        """
        B, T, D = x.shape
        V = self.v_linear(x)  # (B, T, D)

        head_outputs = []
        attn_weights_all = []

        scale = 1.0 / math.sqrt(D)

        for q_lin, k_lin in zip(self.q_linears, self.k_linears):
            Q = q_lin(x)  # (B, T, D)
            K = k_lin(x)  # (B, T, D)

            scores = torch.bmm(Q, K.transpose(1, 2)) * scale  # (B, T, T)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn_weights_all.append(attn.unsqueeze(1))  # (B,1,T,T)

            H = torch.bmm(attn, V)  # (B, T, D)  using shared V
            head_outputs.append(H)

        # Interpretability: average attention across heads
        attn_weights_all = torch.cat(attn_weights_all, dim=1)  # (B, H, T, T)
        H_mean = torch.stack(head_outputs, dim=0).mean(dim=0)  # (B, T, D)
        out = self.out_linear(H_mean)  # (B, T, D)

        return out, attn_weights_all


class TemporalFusionTransformerSpatiotemporal(nn.Module):
    """
    Spatiotemporal TFT-like model:

    - Takes x as (B, T, C, H, W) or (B, T*C, H, W) depending on stack-time.
    - For each pixel (h, w), builds a time series of length T with C features.
    - Applies:
        * Temporal variable selection (GRN + softmax over features)
        * LSTM encoder over time
        * Interpretable multi-head self-attention over encoded sequence
        * Gated residual network on top of attention output
        * Linear multi-horizon head producing K logits per pixel
    - Returns logits of shape (B, K, H, W) to match your current pipeline.
    """
    def __init__(
        self,
        feature_dim: int,
        horizons: Sequence[int],
        t_hist: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.horizons = [int(h) for h in horizons]
        self.k = len(self.horizons)
        self.t_hist = int(t_hist)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)

        # Variable selection over temporal covariates
        self.vsn = TemporalVariableSelection(
            input_dim=self.feature_dim,
            hidden_dim=self.hidden_size,
            context_dim=None,  # can be extended with static context later
            dropout=dropout,
        )

        # Local temporal processing: LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Gated residual on LSTM outputs
        self.post_lstm_grn = GatedResidualNetwork(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.hidden_size,
            context_dim=None,
            dropout=dropout,
        )

        # Long-range temporal dependencies: interpretable attention
        self.attn = InterpretableMultiHeadAttention(
            d_model=self.hidden_size,
            num_heads=self.num_heads,
            dropout=dropout,
        )
        self.post_attn_grn = GatedResidualNetwork(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.hidden_size,
            context_dim=None,
            dropout=dropout,
        )

        # Output head: GRN + linear K-horizon logits
        self.output_grn = GatedResidualNetwork(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.hidden_size,
            context_dim=None,
            dropout=dropout,
        )
        self.output_linear = nn.Linear(self.hidden_size, self.k)

        # For compatibility with previous logging
        self.in_channels = self.feature_dim * self.t_hist

    def _to_B_T_C(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize x to shape (B, T, C, H, W), consistent with LSTMSpatiotemporal.
        """
        if x.dim() == 5:
            # Assume (B, T, C, H, W)
            B, T, C, H, W = x.shape
            return x.view(B, T, C, H, W)
        elif x.dim() == 4:
            # Assume (B, T*C, H, W) with time stacked into channels
            B, TC, H, W = x.shape
            if self.feature_dim <= 0 or self.t_hist <= 0:
                raise ValueError(
                    f"TFT: invalid feature_dim={self.feature_dim} or t_hist={self.t_hist}"
                )
            if TC % self.feature_dim != 0:
                raise ValueError(
                    f"Cannot reshape x of shape {tuple(x.shape)} into (B,T,C,H,W) "
                    f"with feature_dim={self.feature_dim}, t_hist={self.t_hist}. "
                    f"Got channels={TC}, expected multiple of feature_dim."
                )
            T = TC // self.feature_dim
            if T != self.t_hist:
                # Trust runtime data more than config
                self.t_hist = T
            x = x.view(B, T, self.feature_dim, H, W)
            return x
        else:
            raise ValueError(
                f"TFT expected x with 4 or 5 dims, got {tuple(x.shape)}"
            )

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x = batch["x"]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)

        # (B, T, C, H, W)
        x = self._to_B_T_C(x)
        B, T, C, H, W = x.shape

        # Flatten spatial dims -> treat each pixel as its own time series
        # x_seq: (B*H*W, T, C)
        x_seq = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)

        # 1) Variable selection over temporal covariates
        # rep_seq: (B*H*W, T, hidden_size)
        # vsn_weights: (B*H*W, T, C)  (we could log this for interpretability later)
        rep_seq, vsn_weights = self.vsn(x_seq, context=None)

        # 2) LSTM encoder over time
        lstm_out, _ = self.lstm(rep_seq)  # (B*H*W, T, hidden_size)

        # 3) Post-LSTM gated residual
        lstm_out = self.post_lstm_grn(lstm_out)

        # 4) Interpretable self-attention over time
        attn_out, attn_weights = self.attn(lstm_out)  # (B*H*W, T, hidden_size), (B*H*W, H, T, T)

        # 5) Post-attention gated residual
        attn_out = self.post_attn_grn(attn_out)

        # 6) Aggregate temporal representation -> horizon logits
        #    Here we use the last time step's representation; you could also try
        #    mean pooling or a small GRN over all time steps.
        final_rep = attn_out[:, -1, :]  # (B*H*W, hidden_size)
        final_rep = self.output_grn(final_rep)
        logits_flat = self.output_linear(final_rep)  # (B*H*W, K)

        # 7) Reshape back to (B, K, H, W)
        logits = logits_flat.view(B, H, W, self.k).permute(0, 3, 1, 2).contiguous()
        return logits



# ----------------------------------------------------------------------
# Loss & monitoring utilities
# ----------------------------------------------------------------------

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
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

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
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 enable_class_weights: bool = False, max_pos_weight: float = 100.0):
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
        self.queue_capacity: Optional[int] = None
        self.batches_ready: Optional[int] = None
        self.batches_loading: Optional[int] = None

    def start(self):
        self.running = True
        self.pbar = tqdm(total=0, position=0, bar_format='{desc}', leave=True)
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.pbar:
            self.pbar.close()

    def set_status(self, status):
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
        self.batch_times = deque(maxlen=100)
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

        while True:
            try:
                t_start = time.time()
                batch = next(iterator)
                t_elapsed = time.time() - t_start
                self.batch_times.append(t_elapsed)
                self.last_batch_time = t_elapsed

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
    DataLoader factory with optional sampler (used for patch rebalancing / curriculum / SMOTE-based weights).
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
                print(f"[DataLoader] Using multiprocessing context '{ctx}' "
                      f"with {args.workers} workers and prefetch_factor={args.prefetch}.")
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
    model.eval()
    tot_loss, tot_mask = 0.0, 0.0
    tp_total = fp_total = fn_total = tn_total = 0

    num_horizons = len(args.horizons)
    tp_h = [0 for _ in range(num_horizons)]
    fp_h = [0 for _ in range(num_horizons)]
    fn_h = [0 for _ in range(num_horizons)]
    tn_h = [0 for _ in range(num_horizons)]

    # Put validation/test progress bar on its own line (position=3)
    iterator = wrap_loader(loader, desc="eval", use_tqdm=use_tqdm, show_worker_stats=show_worker_stats, position=3)
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
        preds = probs >= args.metrics_threshold
        targets = batch["y"] > 0.5
        valid = batch["mask"] > 0.5

        if 'debug_stats_done' not in locals():
            debug_mask = valid
            p_valid = probs[debug_mask]
            t_valid = targets[debug_mask]

            print("\n[DEBUG] On first eval batch:")
            print("  mean prob:          ", p_valid.mean().item())
            print("  frac prob > 0.3:    ", (p_valid > 0.3).float().mean().item())
            print("  frac prob > 0.5:    ", (p_valid > 0.5).float().mean().item())
            print("  frac prob > 0.7:    ", (p_valid > 0.7).float().mean().item())
            print("  label fire fraction:", t_valid.float().mean().item())

            debug_stats_done = True
        
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

    if tot_mask == 0:
        print("[WARN] Evaluation mask sum was zero across all batches; reporting NaN metrics.")
        val_loss = float('nan')
    else:
        val_loss = tot_loss / tot_mask

    overall = _compute_basic_metrics(tp_total, fp_total, fn_total, tn_total)

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
    p.add_argument("--stack-time", choices=["separate", "channel"], default="channel")
    p.add_argument("--split", type=float, default=0.9,
                   help="Train fraction. If --val-frac is None, val = 1 - split (2-way split).")
    p.add_argument("--val-frac", type=float, default=None,
                   help="Optional validation fraction for 3-way split. "
                        "If set, train = split, val = val-frac, test = 1 - split - val-frac.")
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

    # --- LSTM model hyperparameters ---
    p.add_argument("--lstm-hidden", type=int, default=64,
                   help="Hidden size for the LSTM model.")
    p.add_argument("--lstm-layers", type=int, default=1,
                   help="Number of LSTM layers.")
    p.add_argument("--lstm-bidirectional", action="store_true",
                   help="Use a bidirectional LSTM over time.")

    # --- Loss / imbalance handling ---
    p.add_argument("--loss", choices=["bce", "focal"], default="bce",
                   help="Loss function: 'bce' (MaskedBCEWithLogits) or 'focal' (MaskedFocalLossWithLogits)")
    p.add_argument("--focal-alpha", type=float, default=0.25,
                   help="Alpha for focal loss (weight on positive class).")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Gamma for focal loss (focusing parameter).")
    p.add_argument("--enable-pixel-class-weights", action="store_true",
                   help="Reweight fire pixels in the loss by N_background/N_fire (clipped).")
    p.add_argument("--max-fire-class-weight", type=float, default=100.0,
                   help="Max multiplicative weight for fire pixels in class-weighted loss.")

    # --- Patch-level rebalancing & curriculum ---
    p.add_argument("--patch-sampling", choices=["none", "balanced"], default="none",
                   help="Patch-level sampling for training. 'balanced' oversamples fire patches.")
    p.add_argument("--patch-stats-path", default="patch_fire_flags_train.npy",
                   help="Path to cache per-patch fire flags (bool array) for training set.")
    p.add_argument("--patch-stats-batch-size", type=int, default=256,
                   help="Batch size when scanning the dataset to compute patch fire stats.")
    p.add_argument("--max-patch-pos-oversample", type=float, default=10.0,
                   help="Upper bound for N_neg/N_pos oversampling factor for fire patches in balanced sampling.")

    # Curriculum controls
    p.add_argument("--curriculum-epochs", type=int, default=0,
                   help="If > 0, use a progressive curriculum over N epochs to ramp in non-fire patches.")
    p.add_argument("--curriculum-start-epoch", type=int, default=1,
                   help="Epoch at which to START ramping in non-fire patches (1-indexed).")
    p.add_argument("--curriculum-neg-weight-min", type=float, default=1e-3,
                   help="Non-fire patch weight at the START of the curriculum.")
    p.add_argument("--curriculum-neg-weight-max", type=float, default=1.0,
                   help="Non-fire patch weight at the END of the curriculum (usually 1.0).")

    # True SMOTE-based oversampling toggle
    p.add_argument("--use-smote", action="store_true",
                   help="Enable true SMOTE at the PATCH level: we build a low-dimensional "
                        "descriptor (global means) for each patch, run imblearn.SMOTE once "
                        "on those descriptors + patch labels, then convert each synthetic "
                        "SMOTE sample into extra sampling weight for its nearest real patch.")
    p.add_argument("--smote-k-neighbors", type=int, default=5,
                   help="k_neighbors parameter for imblearn.SMOTE.")
    p.add_argument("--smote-ratio", type=float, default=1.0,
                   help="Desired minority/majority ratio after SMOTE. 1.0 = balanced.")
    p.add_argument("--smote-cache-path", default="smote_patch_weights.npy",
                   help="Where to cache per-patch SMOTE weights so we only run SMOTE once.")

    # --- Metrics ---
    p.add_argument("--metrics-threshold", type=float, default=0.5,
                   help="Decision threshold on sigmoid(logits) for metrics.")

    # --- Optimization flags ---
    p.add_argument("--compile", action="store_true", help="Use torch.compile() for speedup")
    p.add_argument("--prefetch", type=int, default=2, help="Dataloader prefetch factor")
    p.add_argument("--mp-context", choices=["auto", "spawn", "fork", "forkserver"], default="auto",
                   help="Multiprocessing start method for DataLoader workers")

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
    p.add_argument("--skip-val-dataset", action="store_true", help="Skip val (and test) dataset initialization")
    p.add_argument("--quick-test", action="store_true", help="Quick test: 1 epoch, small batches")
    p.add_argument("--dry-run", action="store_true", help="Initialize everything but don't train")
    p.add_argument("--sample-one-batch", action="store_true", help="Load just 1 batch and exit")
    p.add_argument(
        "--limit-train-samples",
        type=int,
        default=0,
        help="(Optional) extra train sample limit"
    )
    p.add_argument(
        "--val-test-only",
        action="store_true",
        help="Run a validation pass only (no training) to test the validation loop, then exit."
    )

    # Pause points
    p.add_argument("--pause-after-dataset", action="store_true", help="Pause after dataset init")
    p.add_argument("--pause-after-model", action="store_true", help="Pause after model init")

    # New flags
    p.add_argument("--sync-every-step", action="store_true",
                   help="Force device sync after each phase for exact timings (slower)")
    p.add_argument("--measure-loader-time", action="store_true",
                   help="(Kept for compatibility; data loading is always timed directly now)")

    # TensorBoard logging
    p.add_argument("--logdir", default="runs/peat_unet",
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
        seen += batch["y"].shape[0]
        it.set_postfix(samples=min(seen, n))
        if seen >= n:
            break


# NEW: compute / cache per-patch fire flags for training set
def compute_or_load_patch_fire_flags(train_ds, args) -> np.ndarray:
    """
    Returns a boolean numpy array of shape (len(train_ds),)
    indicating whether each training patch contains at least one fire pixel
    (any horizon, valid mask).

    Results are cached to args.patch_stats_path (.npy). Additionally, a JSON
    list of positive indices is written next to it (for offline analysis).
    """
    n = len(train_ds)
    path = args.patch_stats_path

    if path and os.path.exists(path):
        try:
            flags = np.load(path)
            if flags.shape[0] == n:
                flags = flags.astype(np.bool_)
                print_diagnostic_header("Patch Fire Stats (cached)")
                print_diagnostic_item("Loaded from", path, indent=1)
                print_diagnostic_item("Num train patches", n, indent=1)
                print_diagnostic_item("Patches with fire", int(flags.sum()), indent=1)
                return flags
            else:
                print(f"[PatchStats] Cached file {path} has length {flags.shape[0]}, expected {n}; recomputing.")
        except Exception as e:
            print(f"[PatchStats] Failed to load cached stats from {path}: {e}; recomputing.")

    print_diagnostic_header("Patch Fire Stats (compute)")
    print_diagnostic_item("Num train patches", n, indent=1)
    print_diagnostic_item("Batch size", args.patch_stats_batch_size, indent=1)

    pin = False
    kw: Dict[str, Any] = dict(
        batch_size=args.patch_stats_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
        collate_fn=collate,
        persistent_workers=False,
    )
    if args.workers > 0:
        kw["prefetch_factor"] = args.prefetch
        ctx = _choose_mp_context(args.mp_context)
        if ctx:
            kw["multiprocessing_context"] = ctx

    loader = DataLoader(train_ds, **kw)
    flags_list = []
    total_seen = 0
    it = tqdm(loader, desc="patch-fire-scan", leave=False, disable=args.no_tqdm)
    for batch in it:
        y = batch["y"]      # (B, K, H, W)
        mask = batch["mask"]
        fire = ((y > 0.5) & (mask > 0.5)).any(dim=(1, 2, 3))  # (B,)
        flags_list.append(fire.cpu().numpy().astype(np.bool_))
        total_seen += y.shape[0]
        it.set_postfix(seen=total_seen)

    if not flags_list:
        flags = np.zeros(n, dtype=np.bool_)
    else:
        flags = np.concatenate(flags_list, axis=0)
        if flags.shape[0] > n:
            flags = flags[:n]
        elif flags.shape[0] < n:
            pad = np.zeros(n - flags.shape[0], dtype=np.bool_)
            flags = np.concatenate([flags, pad], axis=0)

    n_pos = int(flags.sum())
    print_diagnostic_item("Patches with fire", f"{n_pos} / {n}", indent=1)

    if path:
        try:
            np.save(path, flags)
            print_diagnostic_item("Saved flags to", path, indent=1)
            pos_indices = np.nonzero(flags)[0].tolist()
            json_path = os.path.splitext(path)[0] + "_pos_indices.json"
            with open(json_path, "w") as f:
                json.dump(pos_indices, f)
            print_diagnostic_item("Saved pos indices to", json_path, indent=1)
        except Exception as e:
            print(f"[PatchStats] Warning: failed to save patch stats: {e}")

    return flags


# NEW: compute / cache per-patch SMOTE-based weights (true SMOTE on patch descriptors)
def compute_or_load_smote_patch_weights(train_ds, patch_fire_flags: np.ndarray, args) -> Optional[np.ndarray]:
    """
    Use imblearn.SMOTE on low-dimensional per-patch descriptors to derive
    extra sampling weight for patches.

    Steps:
      1. Build a descriptor f_i for each patch i by taking the global
         mean over spatial dims of x, flattened across time/features.
      2. Use patch_fire_flags as binary labels (0/1).
      3. Run SMOTE on (F, y).
      4. For each synthetic SMOTE sample, find its nearest real patch
         in descriptor space using sklearn.NearestNeighbors.
      5. For each real patch j, set weight_j = 1 + (#synthetic samples
         whose nearest neighbor is j).
      6. Cache weights to args.smote_cache_path.
    """
    if not args.use_smote:
        return None

    if SMOTE is None or NearestNeighbors is None:
        print("\n[SMOTE] imblearn or sklearn not available; disabling --use-smote.")
        print("[SMOTE] Install with 'pip install imbalanced-learn scikit-learn' to enable.")
        return None

    n = len(train_ds)
    path = args.smote_cache_path

    if path and os.path.exists(path):
        try:
            w = np.load(path)
            if w.shape[0] == n:
                w = w.astype(np.float64)
                print_diagnostic_header("SMOTE Patch Weights (cached)")
                print_diagnostic_item("Loaded from", path, indent=1)
                print_diagnostic_item("Num train patches", n, indent=1)
                print_diagnostic_item("Min weight", f"{w.min():.3f}", indent=1)
                print_diagnostic_item("Max weight", f"{w.max():.3f}", indent=1)
                return w
            else:
                print(f"[SMOTE] Cached weights {path} have length {w.shape[0]}, expected {n}; recomputing.")
        except Exception as e:
            print(f"[SMOTE] Failed to load cached weights from {path}: {e}; recomputing.")

    print_diagnostic_header("SMOTE Patch Weights (compute)")
    print_diagnostic_item("Train patches", n, indent=1)

    y = patch_fire_flags.astype(int)
    n_pos = int(y.sum())
    n_neg = int(n - n_pos)
    print_diagnostic_item("Minority (fire) patches", n_pos, indent=1)
    print_diagnostic_item("Majority (non-fire) patches", n_neg, indent=1)

    if n_pos == 0 or n_neg == 0:
        print("[SMOTE] Dataset is single-class; skipping SMOTE.")
        return None

    # Build descriptors F in a single streaming pass (no full patch flatten)
    pin = False
    kw: Dict[str, Any] = dict(
        batch_size=args.patch_stats_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
        collate_fn=collate,
        persistent_workers=False,
    )
    if args.workers > 0:
        kw["prefetch_factor"] = args.prefetch
        ctx = _choose_mp_context(args.mp_context)
        if ctx:
            kw["multiprocessing_context"] = ctx

    loader = DataLoader(train_ds, **kw)
    feats_list = []
    total_seen = 0
    it = tqdm(loader, desc="smote-descriptor-scan", leave=False, disable=args.no_tqdm)
    for batch in it:
        x = batch["x"]  # (B, ..., H, W)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(torch.float32)
        if x.dim() < 4:
            raise ValueError(f"[SMOTE] Expected x with spatial dims, got shape={tuple(x.shape)}")
        # Global mean over spatial dims, keep time/feature structure
        x_mean = x.mean(dim=(-2, -1))          # (B, *non-spatial-dims)
        f = x_mean.view(x_mean.shape[0], -1)   # (B, D)
        feats_list.append(f.cpu().numpy())
        total_seen += f.shape[0]
        it.set_postfix(seen=total_seen)
        if total_seen >= n:
            break

    if not feats_list:
        print("[SMOTE] No features collected; skipping SMOTE.")
        return None

    F = np.concatenate(feats_list, axis=0)
    if F.shape[0] > n:
        F = F[:n]
    elif F.shape[0] < n:
        print(f"[SMOTE] Warning: collected {F.shape[0]} descriptors, expected {n}; "
              f"truncating labels to {F.shape[0]}.")
        y = y[:F.shape[0]]
        n = F.shape[0]

    print_diagnostic_item("Descriptor dim", F.shape[1], indent=1)

    # Run SMOTE on descriptors
    sampling_strategy = float(args.smote_ratio) if args.smote_ratio > 0 else "auto"
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=args.smote_k_neighbors,
        random_state=args.seed,
    )
    print_diagnostic_item("SMOTE sampling_strategy", sampling_strategy, indent=1)
    print_diagnostic_item("SMOTE k_neighbors", args.smote_k_neighbors, indent=1)

    F_res, y_res = smote.fit_resample(F, y)
    N_res = F_res.shape[0]
    N_synth = N_res - n
    n_pos_res = int((y_res == 1).sum())
    n_neg_res = int((y_res == 0).sum())
    print_diagnostic_item("Resampled total", N_res, indent=1)
    print_diagnostic_item("Synthetic samples", N_synth, indent=1)
    print_diagnostic_item("Resampled minority patches", n_pos_res, indent=1)
    print_diagnostic_item("Resampled majority patches", n_neg_res, indent=1)

    # Map synthetic SMOTE descriptors back to nearest real patch index
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nn.fit(F)
    counts = np.zeros(n, dtype=np.int64)

    if N_synth > 0:
        F_synth = F_res[n:]   # synthetic part
        _, idxs = nn.kneighbors(F_synth, return_distance=True)
        idxs = idxs.reshape(-1)
        binc = np.bincount(idxs, minlength=n)
        counts[:len(binc)] += binc

    weights = 1.0 + counts.astype(np.float64)
    print_diagnostic_item("Min SMOTE weight", f"{weights.min():.3f}", indent=1)
    print_diagnostic_item("Max SMOTE weight", f"{weights.max():.3f}", indent=1)
    print_diagnostic_item("Mean SMOTE weight", f"{weights.mean():.3f}", indent=1)

    if path:
        try:
            np.save(path, weights)
            print_diagnostic_item("Saved SMOTE weights to", path, indent=1)
        except Exception as e:
            print(f"[SMOTE] Warning: failed to save SMOTE weights: {e}")

    return weights


# NEW: build sampler from patch fire flags + curriculum settings + SMOTE-based weights
def build_train_sampler(
    patch_fire_flags: Optional[np.ndarray],
    args,
    epoch: int,
    smote_patch_weights: Optional[np.ndarray] = None,
):
    """
    Build a WeightedRandomSampler for training according to:
      - a PROGRESSIVE curriculum that gradually mixes in non-fire patches
      - optional patch-level balanced sampling
      - optional SMOTE-based per-patch weights (from compute_or_load_smote_patch_weights)

    The final per-patch weight is approximately:

        w_i(epoch) = curriculum_weight(epoch, fire/neg) * balanced_weight(fire/neg)
                     * smote_weight_i

    Any of the three mechanisms can be disabled.
    """
    if patch_fire_flags is None:
        return None

    n = int(patch_fire_flags.shape[0])
    pos_mask = patch_fire_flags.astype(bool)
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    if n_pos == 0 or n_neg == 0:
        print("[PatchSampling] Dataset is single-class; skipping sampler.")
        return None

    weights = np.ones(n, dtype=np.float64)

    base_ratio = n_neg / max(float(n_pos), 1.0)
    pos_weight_bal = min(base_ratio, args.max_patch_pos_oversample)

    N = int(max(args.curriculum_epochs, 0))
    S = int(max(args.curriculum_start_epoch, 1))
    end_epoch = S + N - 1

    # No curriculum
    if N <= 0:
        if args.patch_sampling == "balanced":
            weights[pos_mask] *= pos_weight_bal
            print(
                f"[PatchSampling] Balanced sampling (no curriculum): "
                f"pos_weight≈{pos_weight_bal:.1f}, n_pos={n_pos}, n_neg={n_neg}."
            )
        # else: weights stay all ones

    else:
        # With curriculum
        if epoch < S:
            # Pre-curriculum fire-only phase
            weights[neg_mask] = 0.0
            weights[pos_mask] *= pos_weight_bal
            expected_fire_frac = 1.0
            print(
                f"[Curriculum] Epoch {epoch}: pre-curriculum fire-only phase "
                f"(S={S}, N={N}). pos_weight≈{pos_weight_bal:.1f}, "
                f"expected fire frac≈{expected_fire_frac:.3f}"
            )

        elif epoch <= end_epoch:
            # Curriculum ramp
            if N == 1:
                progress = 1.0
            else:
                progress = float(epoch - S) / float(max(N - 1, 1))
            progress = float(np.clip(progress, 0.0, 1.0))

            neg_min = float(max(args.curriculum_neg_weight_min, 0.0))
            neg_max = float(max(args.curriculum_neg_weight_max, neg_min))
            neg_weight = neg_min + progress * (neg_max - neg_min)

            weights[pos_mask] *= pos_weight_bal
            weights[neg_mask] *= neg_weight

            num = pos_weight_bal * n_pos
            den = max(pos_weight_bal * n_pos + neg_weight * n_neg, 1e-9)
            expected_fire_frac = num / den
            print(
                f"[Curriculum] Epoch {epoch}/{end_epoch} "
                f"(S={S}, N={N}, progress={progress:.2f}): "
                f"pos_weight≈{pos_weight_bal:.1f}, neg_weight≈{neg_weight:.3f}, "
                f"expected fire frac≈{expected_fire_frac:.3f}"
            )

        else:
            # Post-curriculum behaviour
            if args.patch_sampling == "balanced":
                weights[pos_mask] *= pos_weight_bal
                print(
                    f"[PatchSampling] Balanced sampler (post-curriculum): "
                    f"pos_weight≈{pos_weight_bal:.1f}, n_pos={n_pos}, n_neg={n_neg}."
                )
            # else: weights all ones

    # Apply SMOTE-based per-patch weights if available
    if smote_patch_weights is not None:
        if smote_patch_weights.shape[0] != n:
            print(
                f"[SMOTE] Warning: smote_patch_weights length={smote_patch_weights.shape[0]} "
                f"!= n_patches={n}; ignoring SMOTE weights."
            )
        else:
            weights *= smote_patch_weights
            print("[SMOTE] Applied SMOTE-based per-patch weights on top of curriculum/balanced weights.")

    if not np.isfinite(weights).all():
        raise RuntimeError("[PatchSampling] Non-finite values detected in sampler weights.")

    if weights.sum() <= 0:
        print("[PatchSampling] All weights are zero; skipping sampler.")
        return None

    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=n,
        replacement=True,
    )
    return sampler


def diagnose_model_architecture(model, args, train_ds):
    if not args.verbose_model:
        return
    print_diagnostic_header("Model Architecture Details")
    print("\n[DIAG] Testing forward pass with a real sample...")
    try:
        dev = next(model.parameters()).device
        sample = train_ds[0]
        x = sample["x"]
        if torch.is_tensor(x):
            if x.dim() == 3:
                x = x.unsqueeze(0)  # (1, C, H, W)
            elif x.dim() == 4:
                x = x.unsqueeze(0)  # (1, T, C, H, W) or (1, C, H, W)
        else:
            x = torch.as_tensor(x).unsqueeze(0)
        dummy_batch = {"x": x.to(dev)}
        if "coords" in sample and torch.is_tensor(sample["coords"]):
            coords = sample["coords"]
            if coords.dim() == 3:
                coords = coords.unsqueeze(0)
            dummy_batch["coords"] = coords.to(dev)

        with torch.no_grad():
            output = model(dummy_batch)
        print_diagnostic_item("Dummy forward pass", f"✓ Output shape: {tuple(output.shape)}")
    except Exception as e:
        print_diagnostic_item("Dummy forward pass", f"✗ Failed: {e}")


def print_env_and_cfg(args, device, train_ds, val_ds, test_ds, model):
    cuda_ok = torch.cuda.is_available()
    dtype_note = "bfloat16" if (cuda_ok and torch.cuda.is_bf16_supported()) else ("float16" if cuda_ok else "float32")
    p_total, p_train = count_params(model)

    print_diagnostic_header("Environment & Configuration")
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

    print("\n  Data:")
    print_diagnostic_item("Train samples", len(train_ds), indent=1)
    print_diagnostic_item("Val samples", len(val_ds) if val_ds else "N/A", indent=1)
    print_diagnostic_item("Test samples", len(test_ds) if test_ds else "N/A", indent=1)
    print_diagnostic_item("T_hist (arg)", args.T_hist, indent=1)
    print_diagnostic_item("Input feature channels (incl. coord sin/cos)", train_ds.C_total, indent=1)
    print_diagnostic_item("Patch size", args.patch, indent=1)
    print_diagnostic_item("Horizons", args.horizons, indent=1)
    print_diagnostic_item("Time stacking", args.stack_time, indent=1)
    print_diagnostic_item("Peat filtering", f"{'on' if not args.no_skip_nonpeat else 'off'} (min frac={args.peat_min_fraction})", indent=1)

    print("\n  Model:")
    print_diagnostic_item("Model class", model.__class__.__name__, indent=1)
    if hasattr(model, "in_channels"):
        print_diagnostic_item("Input channels (total)", model.in_channels, indent=1)
    if hasattr(model, "feature_dim"):
        print_diagnostic_item("Per-time feature dim", model.feature_dim, indent=1)
    if hasattr(model, "t_hist") and model.t_hist is not None:
        print_diagnostic_item("T_hist (inferred)", model.t_hist, indent=1)
    print_diagnostic_item("Output horizons", len(args.horizons), indent=1)
    print_diagnostic_item("Total params", human_int(p_total), indent=1)
    print_diagnostic_item("Trainable params", human_int(p_train), indent=1)

    print("\n  Hyperparameters:")
    print_diagnostic_item("Batch size", args.batch_size, indent=1)
    print_diagnostic_item("Grad accumulation", args.grad_accum, indent=1)
    print_diagnostic_item("Effective batch", args.batch_size * args.grad_accum, indent=1)
    print_diagnostic_item("Learning rate", args.lr, indent=1)
    print_diagnostic_item("Epochs", args.epochs, indent=1)
    print_diagnostic_item("Workers", args.workers, indent=1)
    print_diagnostic_item("Prefetch factor", args.prefetch, indent=1)
    print_diagnostic_item("Patch sampling", args.patch_sampling, indent=1)
    print_diagnostic_item("Curriculum epochs", args.curriculum_epochs, indent=1)
    print_diagnostic_item("Curriculum start epoch", args.curriculum_start_epoch, indent=1)
    print_diagnostic_item("Curriculum neg_weight_min", args.curriculum_neg_weight_min, indent=1)
    print_diagnostic_item("Curriculum neg_weight_max", args.curriculum_neg_weight_max, indent=1)
    print_diagnostic_item("Pixel class weights", "on" if args.enable_pixel_class_weights else "off", indent=1)
    print_diagnostic_item("SMOTE patch oversampling", "on" if args.use_smote else "off", indent=1)


def test_first_batch(train_loader, device, model, args):
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

    print("\n  Batch contents:")
    for k, v in batch.items():
        if torch.is_tensor(v):
            val = f"shape={tuple(v.shape)}, dtype={v.dtype}"
        else:
            val = f"type={type(v)}"
        print_diagnostic_item(k, val, indent=1)

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

    if device.type == "cuda":
        print("\n  GPU Memory:")
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print_diagnostic_item("Allocated", f"{allocated:.2f} GB", indent=1)
        print_diagnostic_item("Reserved", f"{reserved:.2f} GB", indent=1)

    print("\n  Timing breakdown:")
    total = t_load + t_transfer + t_forward + t_backward
    total = max(total, 1e-9)
    print_diagnostic_item("Data loading", f"{t_load:.2f}s ({100 * t_load / total:.1f}%)", indent=1)
    print_diagnostic_item("Device transfer", f"{t_transfer:.2f}s ({100 * t_transfer / total:.1f}%)", indent=1)
    print_diagnostic_item("Forward pass", f"{t_forward:.2f}s ({100 * t_forward / total:.1f}%)", indent=1)
    print_diagnostic_item("Backward pass", f"{t_backward:.2f}s ({100 * t_backward / total:.1f}%)", indent=1)
    print_diagnostic_item("Total", f"{total:.2f}s", indent=1)


def diagnose_dataloader(train_loader, args):
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
    if t is None or not torch.is_tensor(t):
        return

    if not torch.isfinite(t).all():
        bad = ~torch.isfinite(t)
        idx = torch.nonzero(bad, as_tuple=False)
        idx_str = idx[0].tolist() if idx.numel() > 0 else "unknown"
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
    print_diagnostic_header("Dataset Initialization")
    print_diagnostic_item("ERA5 Zarr", args.era5_zarr)
    print_diagnostic_item("SMAP Zarr", args.smap_zarr)
    print_diagnostic_item("VIIRS Zarr", args.viirs_zarr)
    print_diagnostic_item("T_hist", args.T_hist)
    print_diagnostic_item("Horizons", args.horizons)
    print_diagnostic_item("Patch size", args.patch)
    print_diagnostic_item("Stride", args.stride)
    print_diagnostic_item("Split (train frac)", args.split)
    print_diagnostic_item("Val frac (explicit)", args.val_frac if args.val_frac is not None else "None")
    if args.val_frac is not None:
        test_frac = 1.0 - args.split - args.val_frac
        print_diagnostic_item("Implied test frac", f"{max(test_frac, 0.0):.3f}")
    print_diagnostic_item("Normalize inputs", args.normalize_inputs)
    print_diagnostic_item("Skip non-peat patches", not args.no_skip_nonpeat)
    print_diagnostic_item("Peat min fraction", args.peat_min_fraction)

    norm_mode = None if args.normalize_inputs == "none" else "per_channel"
    skip_nonpeat = not args.no_skip_nonpeat

    if args.skip_val_dataset:
        print("\n[Dataset] NOTE: --skip-val-dataset also skips test dataset.")
        create_test = False
    else:
        create_test = args.val_frac is not None and (1.0 - args.split - args.val_frac) > 0.0

    # For LSTM baseline:
    #  - Append sin/cos(lat,lon) as features (coord_as_features=True)
    #  - Do NOT return coords tensor (return_coords=False)
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
            val_frac=args.val_frac,
            seed=args.seed,
            normalize_inputs=norm_mode,
            max_samples=args.max_samples if args.max_samples else None,
            skip_nonpeat_patches=skip_nonpeat,
            peat_min_fraction=args.peat_min_fraction,
            time_index=None,
            return_coords=False,
            coord_as_features=True,
        )
    print_diagnostic_item("Train samples", len(train_ds))

    if args.skip_val_dataset:
        print("\n[Dataset] Skipping val/test datasets (--skip-val-dataset)")
        return train_ds, None, None

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
            val_frac=args.val_frac,
            seed=args.seed,
            normalize_inputs=None if norm_mode == "per_channel" else norm_mode,
            max_samples=args.max_samples if args.max_samples else None,
            skip_nonpeat_patches=skip_nonpeat,
            peat_min_fraction=args.peat_min_fraction,
            time_index=None,
            return_coords=False,
            coord_as_features=True,
        )

    print_diagnostic_item("Val samples", len(val_ds))

    test_ds = None
    if create_test:
        print("\n[Dataset] Initializing test dataset...")
        with DiagnosticTimer("Test dataset initialization", track_ram=True):
            test_ds = JointPeatDataset(
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
                mode="test",
                split=args.split,
                val_frac=args.val_frac,
                seed=args.seed,
                normalize_inputs=None if norm_mode == "per_channel" else norm_mode,
                max_samples=args.max_samples if args.max_samples else None,
                skip_nonpeat_patches=skip_nonpeat,
                peat_min_fraction=args.peat_min_fraction,
                time_index=None,
                return_coords=False,
                coord_as_features=True,
            )
        print_diagnostic_item("Test samples", len(test_ds))

    # Share normalization stats if per_channel
    if norm_mode == "per_channel":
        mean, std = train_ds.get_normalization()
        val_ds.set_normalization(mean, std)
        if test_ds is not None:
            test_ds.set_normalization(mean, std)

    if args.pause_after_dataset:
        input("\n[PAUSE] Press Enter to continue after dataset initialization...")

    return train_ds, val_ds, test_ds


def get_loader_queue_stats(loader_iter, args):
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


def log_metrics_to_csv(csv_path: str, epoch: int, train_loss: float, val_loss: float,
                       metrics: Dict[str, Any], horizons: Sequence[int]):
    file_exists = os.path.exists(csv_path)

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

    writer: Optional[SummaryWriter] = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=args.logdir)
        print(f"[TensorBoard] Logging to: {args.logdir}")

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

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    print_diagnostic_header("System Resources")
    initial_ram = get_ram_usage()
    print_diagnostic_item("Initial RAM usage", f"{initial_ram:.2f} GB")
    cpu_count_phys = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
    cpu_count_logical = psutil.cpu_count(logical=True)
    print_diagnostic_item("CPU cores", f"{cpu_count_phys} physical, {cpu_count_logical} logical")

    print("\n  Storage:")
    print_diagnostic_item("ERA5 store", _detect_storage_type(args.era5_zarr), indent=1)
    print_diagnostic_item("SMAP store", _detect_storage_type(args.smap_zarr), indent=1)
    print_diagnostic_item("VIIRS store", _detect_storage_type(args.viirs_zarr), indent=1)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        print_diagnostic_item("TF32 enabled", "Yes (Ampere+ GPUs)")

    print("\n" + "=" * 60)
    print("  DATASET INITIALIZATION")
    print("=" * 60)
    before_data_ram = get_ram_usage()
    train_ds, val_ds, test_ds = make_datasets(args)
    after_data_ram = print_ram_delta(before_data_ram, "Dataset initialization")

    if args.quick_test:
        print("\n[Quick Test Mode] Limiting to 1 epoch")
        args.epochs = 1

    dataset_scan(train_ds, "train", args, device)
    if val_ds is not None:
        dataset_scan(val_ds, "val", args, device)
    if test_ds is not None:
        dataset_scan(test_ds, "test", args, device)

    # NEW: compute/cached per-patch fire flags if needed for sampling, curriculum, or SMOTE-based oversampling
    patch_fire_flags = None
    if args.patch_sampling != "none" or args.curriculum_epochs > 0 or args.use_smote:
        patch_fire_flags = compute_or_load_patch_fire_flags(train_ds, args)

    # NEW: compute/cached SMOTE-based per-patch weights (true SMOTE over descriptors)
    smote_patch_weights = None
    if args.use_smote and patch_fire_flags is not None:
        smote_patch_weights = compute_or_load_smote_patch_weights(train_ds, patch_fire_flags, args)

    print("\n" + "=" * 60)
    print("  DATALOADER INITIALIZATION")
    print("=" * 60)
    with DiagnosticTimer("Train DataLoader creation", track_ram=True):
        base_train_loader = make_loader(train_ds, args.batch_size, True, args, sampler=None)
    diagnose_dataloader(base_train_loader, args)

    if val_ds is not None:
        with DiagnosticTimer("Val DataLoader creation", track_ram=True):
            val_loader = make_loader(val_ds, args.batch_size, False, args, sampler=None)
    else:
        val_loader = None

    if test_ds is not None:
        with DiagnosticTimer("Test DataLoader creation", track_ram=True):
            test_loader = make_loader(test_ds, args.batch_size, False, args, sampler=None)
    else:
        test_loader = None

    print("\n" + "=" * 60)
    print("  MODEL INITIALIZATION (LSTMSpatiotemporal)")
    print("=" * 60)
    before_model_ram = get_ram_usage()
    with DiagnosticTimer("Model creation", track_ram=True, track_gpu=True, device_index=device_index):
        # Infer per-time feature dimension and T_hist from a real sample
        sample = train_ds[0]
        x_sample = sample["x"]
        if not torch.is_tensor(x_sample):
            x_sample = torch.as_tensor(x_sample)

        feature_dim = getattr(train_ds, "C_total", None)
        t_hist_inferred = args.T_hist

        if x_sample.dim() == 4 and args.stack_time == "separate":
            # (T, C, H, W)
            T_s, C_s, H_s, W_s = x_sample.shape
            t_hist_inferred = T_s
            if feature_dim is None:
                feature_dim = C_s
        elif x_sample.dim() == 4 and args.stack_time == "channel":
            # (C_in, H, W) where C_in = T * C_total (for a single patch)
            C_in, H_s, W_s = x_sample.shape
            if feature_dim is not None and feature_dim > 0:
                t_hist_inferred = max(1, C_in // feature_dim)
            else:
                # Fallback: assume args.T_hist is correct and infer feature_dim
                t_hist_inferred = args.T_hist
                feature_dim = max(1, C_in // max(1, t_hist_inferred))
        elif x_sample.dim() == 3:
            # (C_in, H, W) with channel-stacked time
            C_in, H_s, W_s = x_sample.shape
            if feature_dim is not None:
                t_hist_inferred = max(1, C_in // feature_dim)
            else:
                t_hist_inferred = args.T_hist
                feature_dim = max(1, C_in // max(1, t_hist_inferred))
        else:
            raise ValueError(
                f"Unexpected x_sample.shape={tuple(x_sample.shape)} from dataset; "
                f"expected 3D (C,H,W) or 4D (T,C,H,W) / (C,H,W)."
            )

        if feature_dim is None or feature_dim <= 0:
            raise ValueError(
                "Could not infer per-time feature dimension from dataset. "
                "Please ensure JointPeatDataset exposes C_total."
            )

        model = TemporalFusionTransformerSpatiotemporal(
            feature_dim=feature_dim,
            horizons=args.horizons,
            t_hist=t_hist_inferred,
            hidden_size=args.lstm_hidden,
            num_layers=args.lstm_layers,
            num_heads=4,
            dropout=0.1,
        ).to(device)
    after_model_ram = print_ram_delta(before_model_ram, "Model creation")

    diagnose_model_architecture(model, args, train_ds)
    if args.pause_after_model:
        input("\n[PAUSE] Press Enter to continue after model initialization...")

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

    total_data_ram = after_data_ram - before_data_ram
    total_model_ram = after_model_ram - before_model_ram
    total_ram = get_ram_usage()
    print_diagnostic_header("RAM Usage Summary")
    print_diagnostic_item("Data loading", f"{total_data_ram:.2f} GB")
    print_diagnostic_item("Model creation", f"{total_model_ram:.2f} GB")
    print_diagnostic_item("Data/Model ratio", f"{(total_data_ram / max(total_model_ram, 0.001)):.2f}x")
    print_diagnostic_item("Total RAM used", f"{total_ram:.2f} GB")

    print_env_and_cfg(args, device, train_ds, val_ds, test_ds, model)

    if args.sample_one_batch:
        test_first_batch(base_train_loader, device, model, args)
        print("\n[SAMPLE-ONE-BATCH] Exiting after first batch test")
        if writer is not None:
            writer.close()
        return

    with DiagnosticTimer("Optimizer creation"):
        optim_kwargs = dict(lr=args.lr, weight_decay=1e-4)
        if torch.cuda.is_available():
            try:
                optim = torch.optim.AdamW(model.parameters(), fused=True, **optim_kwargs)
            except TypeError:
                optim = torch.optim.AdamW(model.parameters(), **optim_kwargs)
        else:
            optim = torch.optim.AdamW(model.parameters(), **optim_kwargs)

        use_cuda = (device.type == "cuda")
        use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

        try:
            # PyTorch 2.x style
            scaler = torch.amp.GradScaler(
                enabled=use_cuda and not use_bf16,
            )
        except (AttributeError, TypeError):
            # PyTorch 1.x style fallback
            try:
                from torch.cuda.amp import GradScaler as CudaGradScaler
            except ImportError:
                # Last-resort: dummy scaler that just passes things through
                class _DummyScaler:
                    def __init__(self, enabled=True): self.enabled = enabled
                    def scale(self, x): return x
                    def step(self, opt): opt.step()
                    def update(self): pass
                    def state_dict(self): return {}
                    def load_state_dict(self, state): pass
                CudaGradScaler = _DummyScaler
            scaler = CudaGradScaler(enabled=use_cuda and not use_bf16)

    if args.loss == "bce":
        criterion = MaskedBCEWithLogits(
            enable_class_weights=args.enable_pixel_class_weights,
            max_pos_weight=args.max_fire_class_weight,
        )
        print("\n[Loss] Using MaskedBCEWithLogits (BCE).")
    else:
        criterion = MaskedFocalLossWithLogits(
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
            enable_class_weights=args.enable_pixel_class_weights,
            max_pos_weight=args.max_fire_class_weight,
        )
        print(f"\n[Loss] Using MaskedFocalLossWithLogits (Focal): "
              f"alpha={args.focal_alpha}, gamma={args.focal_gamma}")

    if args.dry_run:
        print("\n[DRY-RUN] Exiting without training")
        if writer is not None:
            writer.close()
        return

    # VAL-TEST-ONLY: run one validation pass and exit
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
                # Overall
                print_diagnostic_item("Overall accuracy",  f"{val_metrics['accuracy']*100:.2f}%", indent=1)
                print_diagnostic_item("Overall precision", f"{val_metrics['precision']*100:.2f}%", indent=1)
                print_diagnostic_item("Overall recall",    f"{val_metrics['recall']*100:.2f}%", indent=1)
                print_diagnostic_item("Overall F1-score",  f"{val_metrics['f1']*100:.2f}%", indent=1)
                print_diagnostic_item("Overall support (valid pixels)", int(val_metrics["support"]), indent=1)
                print_diagnostic_item("Overall TP", int(val_metrics["tp"]), indent=1)
                print_diagnostic_item("Overall FP", int(val_metrics["fp"]), indent=1)
                print_diagnostic_item("Overall FN", int(val_metrics["fn"]), indent=1)
                print_diagnostic_item("Overall TN", int(val_metrics["tn"]), indent=1)

                # Per-horizon breakdown
                per_h = val_metrics.get("per_horizon") or {}
                for h in args.horizons:
                    m_h = per_h.get(h)
                    if not m_h:
                        continue
                    print(f"\n    Horizon +{h}:")
                    print_diagnostic_item("Accuracy",  f"{m_h['accuracy']*100:.2f}%", indent=3)
                    print_diagnostic_item("Precision", f"{m_h['precision']*100:.2f}%", indent=3)
                    print_diagnostic_item("Recall",    f"{m_h['recall']*100:.2f}%", indent=3)
                    print_diagnostic_item("F1-score",  f"{m_h['f1']*100:.2f}%", indent=3)
                    print_diagnostic_item("Support (valid pixels)", int(m_h["support"]), indent=3)
                    print_diagnostic_item("TP", int(m_h["tp"]), indent=3)
                    print_diagnostic_item("FP", int(m_h["fp"]), indent=3)
                    print_diagnostic_item("FN", int(m_h["fn"]), indent=3)
                    print_diagnostic_item("TN", int(m_h["tn"]), indent=3)

        if writer is not None:
            writer.close()
        return

    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)

    best_val = float("inf")
    best_metrics = None
    best_epoch = None
    step_timer = TrainingStepTimer()
    global_step = 0

    ram_monitor = None
    if use_tqdm:
        ram_monitor = RAMMonitor(device_id=device_index, update_interval=0.5)
        ram_monitor.start()

    epoch_pbar = tqdm(total=args.epochs, desc="📊 Epochs", position=1, leave=True, disable=not use_tqdm) if use_tqdm else None
    timing_pbar = tqdm(total=0, position=4, bar_format='{desc}', leave=True, disable=not use_tqdm) if use_tqdm else None
    throughput_pbar = tqdm(total=0, position=5, bar_format='{desc}', leave=True, disable=not use_tqdm) if use_tqdm else None

    metrics_csv_path = os.path.join("logs", "metrics_log.csv")
    os.makedirs("checkpoints", exist_ok=True)

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            step_timer.reset()

            if epoch_pbar:
                epoch_pbar.set_postfix_str(f"epoch={epoch}")

            # Build epoch-specific train loader (for curriculum / patch sampling / SMOTE-based oversampling)
            if args.patch_sampling != "none" or args.curriculum_epochs > 0 or args.use_smote:
                sampler = build_train_sampler(patch_fire_flags, args, epoch, smote_patch_weights)
                train_loader = make_loader(train_ds, args.batch_size, True, args, sampler=sampler)
            else:
                train_loader = base_train_loader

            # Training loop
            train_loss_accum = 0.0
            train_mask_accum = 0.0
            num_samples = 0
            num_updates = 0

            timed_loader = wrap_loader(
                train_loader,
                desc=f"train-epoch-{epoch}",
                use_tqdm=False,
                show_worker_stats=args.show_worker_stats,
                position=2,
            )

            total_batches = len(train_loader)
            batch_pbar = tqdm(
                total=total_batches,
                desc=f"🌀 Train {epoch}/{args.epochs}",
                position=2,
                leave=True,
                disable=not use_tqdm,
            )

            optim.zero_grad(set_to_none=True)
            seen_samples = 0

            for batch_idx, batch in enumerate(timed_loader):
                data_time = timed_loader.last_batch_time

                if ram_monitor:
                    ram_monitor.set_loader_queue(None, None, None)
                    ram_monitor.set_status(
                        f"Train ep {epoch}/{args.epochs}, step {batch_idx+1}/{total_batches}"
                    )

                # Transfer to device
                t_transfer_start = time.time()
                batch = {
                    k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                if device.type == "cuda" and args.sync_every_step:
                    torch.cuda.synchronize()
                transfer_time = time.time() - t_transfer_start

                # Forward + loss
                t_forward_start = time.time()
                with amp_autocast(device):
                    logits = model(batch)
                    loss = criterion(logits, batch["y"], batch["mask"])
                if device.type == "cuda" and args.sync_every_step:
                    torch.cuda.synchronize()
                forward_time = time.time() - t_forward_start

                m = batch["mask"].sum().item()
                if m > 0:
                    train_loss_accum += loss.item() * m
                    train_mask_accum += m

                loss_scaled = loss / max(1, args.grad_accum)

                # Backward
                t_backward_start = time.time()
                scaler.scale(loss_scaled).backward()
                if device.type == "cuda" and args.sync_every_step:
                    torch.cuda.synchronize()
                backward_time = time.time() - t_backward_start

                # Optimizer step
                t_optim_start = time.time()
                do_step = ((batch_idx + 1) % args.grad_accum == 0)
                if do_step:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    num_updates += 1
                if device.type == "cuda" and args.sync_every_step:
                    torch.cuda.synchronize()
                optim_time = time.time() - t_optim_start

                step_timer.add(data_time, transfer_time, forward_time, backward_time, optim_time)

                B = batch["x"].shape[0]
                num_samples += B
                seen_samples += B
                global_step += 1

                if batch_pbar:
                    batch_pbar.update(1)
                    batch_pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        data_ms=f"{data_time*1000:.0f}",
                        fwd_ms=f"{forward_time*1000:.0f}",
                        bwd_ms=f"{backward_time*1000:.0f}",
                    )

                # TensorBoard logging (per step)
                if writer is not None and (args.log_interval <= 0 or global_step % args.log_interval == 0):
                    writer.add_scalar("train/loss_step", loss.item(), global_step)
                    writer.add_scalar("train/mask_sum_step", m, global_step)
                    writer.add_scalar("train/timing/data", data_time, global_step)
                    writer.add_scalar("train/timing/transfer", transfer_time, global_step)
                    writer.add_scalar("train/timing/forward", forward_time, global_step)
                    writer.add_scalar("train/timing/backward", backward_time, global_step)
                    writer.add_scalar("train/timing/optim", optim_time, global_step)
                    if ram_monitor is not None:
                        writer.add_scalar("system/ram_gb", ram_monitor.current_ram, global_step)
                        writer.add_scalar("system/gpu_gb", ram_monitor.current_gpu, global_step or 1)

                # Optional train sample limit
                if args.limit_train_samples > 0 and seen_samples >= args.limit_train_samples:
                    if batch_pbar:
                        batch_pbar.set_postfix_str("limit-train-samples reached")
                    break

            if batch_pbar:
                batch_pbar.close()

            # Epoch-level stats
            if train_mask_accum > 0:
                train_loss = train_loss_accum / train_mask_accum
            else:
                train_loss = float("nan")

            epoch_time = time.time() - epoch_start
            steps_in_epoch = max(1, num_updates if num_updates > 0 else len(train_loader))
            samples_per_sec = num_samples / max(epoch_time, 1e-9)
            updates_per_sec = steps_in_epoch / max(epoch_time, 1e-9)

            if timing_pbar:
                avgs = step_timer.get_averages() or {}
                timing_pbar.set_description_str(
                    "⏱️ data={:.3f}s, xfer={:.3f}s, fwd={:.3f}s, bwd={:.3f}s, opt={:.3f}s".format(
                        avgs.get("data", 0.0),
                        avgs.get("transfer", 0.0),
                        avgs.get("forward", 0.0),
                        avgs.get("backward", 0.0),
                        avgs.get("optim", 0.0),
                    )
                )

            if throughput_pbar:
                throughput_pbar.set_description_str(
                    f"🚀 throughput: {samples_per_sec:.1f} samples/s, {updates_per_sec:.2f} updates/s"
                )

            if writer is not None:
                writer.add_scalar("train/loss_epoch", train_loss, epoch)
                writer.add_scalar("train/samples_per_sec", samples_per_sec, epoch)
                writer.add_scalar("train/updates_per_sec", updates_per_sec, epoch)
                writer.add_scalar("train/epoch_time_sec", epoch_time, epoch)

            # Validation
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
            else:
                val_loss, val_metrics = float("nan"), None

            print_diagnostic_header(f"Epoch {epoch} Summary")
            print_diagnostic_item("Train loss", f"{train_loss:.6f}", indent=1)
            print_diagnostic_item("Val loss", f"{val_loss:.6f}", indent=1)
            print_diagnostic_item("Epoch time", f"{epoch_time:.2f}s", indent=1)
            print_diagnostic_item("Samples/sec", f"{samples_per_sec:.1f}", indent=1)
            print_diagnostic_item("Updates/sec", f"{updates_per_sec:.2f}", indent=1)

            if val_metrics is not None and not math.isnan(val_metrics["f1"]):
                # Overall metrics
                print_diagnostic_item("Val accuracy",  f"{val_metrics['accuracy']*100:.2f}%", indent=1)
                print_diagnostic_item("Val precision", f"{val_metrics['precision']*100:.2f}%", indent=1)
                print_diagnostic_item("Val recall",    f"{val_metrics['recall']*100:.2f}%", indent=1)
                print_diagnostic_item("Val F1",        f"{val_metrics['f1']*100:.2f}%", indent=1)
                print_diagnostic_item("Val support",   int(val_metrics["support"]), indent=1)
                print_diagnostic_item("Val TP",        int(val_metrics["tp"]), indent=1)
                print_diagnostic_item("Val FP",        int(val_metrics["fp"]), indent=1)
                print_diagnostic_item("Val FN",        int(val_metrics["fn"]), indent=1)
                print_diagnostic_item("Val TN",        int(val_metrics["tn"]), indent=1)

                # Per-horizon stats
                per_h = val_metrics.get("per_horizon") or {}
                for h in args.horizons:
                    m_h = per_h.get(h)
                    if not m_h:
                        continue
                    print(f"\n    Horizon +{h}:")
                    print_diagnostic_item("Accuracy",  f"{m_h['accuracy']*100:.2f}%", indent=3)
                    print_diagnostic_item("Precision", f"{m_h['precision']*100:.2f}%", indent=3)
                    print_diagnostic_item("Recall",    f"{m_h['recall']*100:.2f}%", indent=3)
                    print_diagnostic_item("F1-score",  f"{m_h['f1']*100:.2f}%", indent=3)
                    print_diagnostic_item("Support (valid pixels)", int(m_h["support"]), indent=3)
                    print_diagnostic_item("TP", int(m_h["tp"]), indent=3)
                    print_diagnostic_item("FP", int(m_h["fp"]), indent=3)
                    print_diagnostic_item("FN", int(m_h["fn"]), indent=3)
                    print_diagnostic_item("TN", int(m_h["tn"]), indent=3)

            # Log to TensorBoard (validation)
            if writer is not None:
                writer.add_scalar("val/loss_epoch", val_loss, epoch)
                if val_metrics is not None:
                    writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
                    writer.add_scalar("val/precision", val_metrics["precision"], epoch)
                    writer.add_scalar("val/recall", val_metrics["recall"], epoch)
                    writer.add_scalar("val/f1", val_metrics["f1"], epoch)
                    writer.add_scalar("val/support", val_metrics["support"], epoch)
                    writer.add_scalar("val/tp", val_metrics["tp"], epoch)
                    writer.add_scalar("val/fp", val_metrics["fp"], epoch)
                    writer.add_scalar("val/fn", val_metrics["fn"], epoch)
                    writer.add_scalar("val/tn", val_metrics["tn"], epoch)
                    for h, m_h in (val_metrics.get("per_horizon") or {}).items():
                        writer.add_scalar(f"val/h{h}_f1", m_h.get("f1", float("nan")), epoch)
                        writer.add_scalar(f"val/h{h}_tp", m_h.get("tp", 0), epoch)
                        writer.add_scalar(f"val/h{h}_fp", m_h.get("fp", 0), epoch)
                        writer.add_scalar(f"val/h{h}_fn", m_h.get("fn", 0), epoch)
                        writer.add_scalar(f"val/h{h}_tn", m_h.get("tn", 0), epoch)

            # Save metrics to CSV
            if val_metrics is None:
                empty_metrics = {
                    "accuracy": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 0,
                    "support": 0,
                    "per_horizon": {},
                }
                log_metrics_to_csv(metrics_csv_path, epoch, train_loss, val_loss, empty_metrics, args.horizons)
            else:
                log_metrics_to_csv(metrics_csv_path, epoch, train_loss, val_loss, val_metrics, args.horizons)

            # Save checkpoints: per epoch + best
            ckpt_epoch_path = os.path.join("checkpoints", f"epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                ckpt_epoch_path,
            )
            print(f"[Checkpoint] Saved epoch checkpoint to {ckpt_epoch_path}")

            if val_loss < best_val:
                best_val = val_loss
                best_metrics = val_metrics
                best_epoch = epoch
                ckpt_best_path = os.path.join("checkpoints", "best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optim.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                        "args": vars(args),
                    },
                    ckpt_best_path,
                )
                print(f"[Checkpoint] New best model (val_loss={best_val:.6f}) saved to {ckpt_best_path}")

                if writer is not None and val_metrics is not None:
                    writer.add_scalar("best/val_loss", best_val, epoch)
                    writer.add_scalar("best/f1", val_metrics["f1"], epoch)

            if epoch_pbar:
                epoch_pbar.update(1)

        # End epochs
        step_timer.print_summary("Overall Training")

        if best_metrics is not None:
            print_diagnostic_header("Best Validation Metrics")
            print_diagnostic_item("Best epoch", best_epoch, indent=1)
            print_diagnostic_item("Best val loss", f"{best_val:.6f}", indent=1)
            print_diagnostic_item("Best overall F1", f"{best_metrics['f1']*100:.2f}%", indent=1)
            print_diagnostic_item("Best overall accuracy", f"{best_metrics['accuracy']*100:.2f}%", indent=1)
            print_diagnostic_item("Best overall support", int(best_metrics["support"]), indent=1)
            print_diagnostic_item("Best overall TP", int(best_metrics["tp"]), indent=1)
            print_diagnostic_item("Best overall FP", int(best_metrics["fp"]), indent=1)
            print_diagnostic_item("Best overall FN", int(best_metrics["fn"]), indent=1)
            print_diagnostic_item("Best overall TN", int(best_metrics["tn"]), indent=1)

            per_h = best_metrics.get("per_horizon") or {}
            for h in args.horizons:
                m_h = per_h.get(h)
                if not m_h:
                    continue
                print(f"\n    Horizon +{h}:")
                print_diagnostic_item("Accuracy",  f"{m_h['accuracy']*100:.2f}%", indent=3)
                print_diagnostic_item("Precision", f"{m_h['precision']*100:.2f}%", indent=3)
                print_diagnostic_item("Recall",    f"{m_h['recall']*100:.2f}%", indent=3)
                print_diagnostic_item("F1-score",  f"{m_h['f1']*100:.2f}%", indent=3)
                print_diagnostic_item("Support (valid pixels)", int(m_h["support"]), indent=3)
                print_diagnostic_item("TP", int(m_h["tp"]), indent=3)
                print_diagnostic_item("FP", int(m_h["fp"]), indent=3)
                print_diagnostic_item("FN", int(m_h["fn"]), indent=3)
                print_diagnostic_item("TN", int(m_h["tn"]), indent=3)

        # Final test evaluation on best checkpoint (if test split exists)
        if test_loader is not None:
            print_diagnostic_header("Final Test Evaluation (best checkpoint)")
            best_ckpt_path = os.path.join("checkpoints", "best.pt")
            if os.path.exists(best_ckpt_path):
                state = torch.load(best_ckpt_path, map_location=device)
                model.load_state_dict(state["model_state"])
                print_diagnostic_item("Loaded checkpoint", best_ckpt_path, indent=1)
            else:
                print_diagnostic_item("Checkpoint", "best.pt not found; using last-epoch weights", indent=1)

            test_loss, test_metrics = evaluate(
                model,
                test_loader,
                device,
                criterion,
                args,
                use_tqdm=not args.no_tqdm,
                show_worker_stats=args.show_worker_stats,
            )
            print_diagnostic_item("Test loss", f"{test_loss:.6f}", indent=1)

            if test_metrics is not None and not math.isnan(test_metrics["f1"]):
                print_diagnostic_item("Test accuracy",  f"{test_metrics['accuracy']*100:.2f}%", indent=1)
                print_diagnostic_item("Test precision", f"{test_metrics['precision']*100:.2f}%", indent=1)
                print_diagnostic_item("Test recall",    f"{test_metrics['recall']*100:.2f}%", indent=1)
                print_diagnostic_item("Test F1",        f"{test_metrics['f1']*100:.2f}%", indent=1)
                print_diagnostic_item("Test support",   int(test_metrics["support"]), indent=1)
                print_diagnostic_item("Test TP",        int(test_metrics["tp"]), indent=1)
                print_diagnostic_item("Test FP",        int(test_metrics["fp"]), indent=1)
                print_diagnostic_item("Test FN",        int(test_metrics["fn"]), indent=1)
                print_diagnostic_item("Test TN",        int(test_metrics["tn"]), indent=1)

                per_h = test_metrics.get("per_horizon") or {}
                for h in args.horizons:
                    m_h = per_h.get(h)
                    if not m_h:
                        continue
                    print(f"\n    Horizon +{h} (test):")
                    print_diagnostic_item("Accuracy",  f"{m_h['accuracy']*100:.2f}%", indent=3)
                    print_diagnostic_item("Precision", f"{m_h['precision']*100:.2f}%", indent=3)
                    print_diagnostic_item("Recall",    f"{m_h['recall']*100:.2f}%", indent=3)
                    print_diagnostic_item("F1-score",  f"{m_h['f1']*100:.2f}%", indent=3)
                    print_diagnostic_item("Support (valid pixels)", int(m_h["support"]), indent=3)
                    print_diagnostic_item("TP", int(m_h["tp"]), indent=3)
                    print_diagnostic_item("FP", int(m_h["fp"]), indent=3)
                    print_diagnostic_item("FN", int(m_h["fn"]), indent=3)
                    print_diagnostic_item("TN", int(m_h["tn"]), indent=3)

            if writer is not None:
                writer.add_scalar("test/loss", test_loss, best_epoch or args.epochs)
                if test_metrics is not None:
                    writer.add_scalar("test/accuracy", test_metrics["accuracy"], best_epoch or args.epochs)
                    writer.add_scalar("test/precision", test_metrics["precision"], best_epoch or args.epochs)
                    writer.add_scalar("test/recall", test_metrics["recall"], best_epoch or args.epochs)
                    writer.add_scalar("test/f1", test_metrics["f1"], best_epoch or args.epochs)

    except KeyboardInterrupt:
        print("\n[TRAINING] Interrupted by user.")

    finally:
        if ram_monitor is not None:
            ram_monitor.stop()
        if epoch_pbar is not None:
            epoch_pbar.close()
        if timing_pbar is not None:
            timing_pbar.close()
        if throughput_pbar is not None:
            throughput_pbar.close()
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
