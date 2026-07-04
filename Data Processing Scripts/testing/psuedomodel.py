# peat_ignition_mamba_gnn.py
# Minimal, runnable-with-small-edits, pseudocode-level PyTorch for multi-horizon peatland fire ignition forecasting.
# Includes: TFT-lite feature gating, Mamba/SSM stack, tiny spatial coupling (graph or 3x3 conv),
# shared head with horizon embeddings, focal/CB-BCE/Brier losses, AMP+TF32, torch.compile, checkpointing, chunked BPTT,
# synthetic dataset, metrics stubs, and a tiny training loop to sanity-check throughput.

import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ========= Hardware prefs =========
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ========= Config =========
@dataclass
class Config:
    D: int = 128                  # d_model ∈ {128,192,256} for tensor cores
    depth: int = 3                # 2–4 Mamba blocks
    dropout: float = 0.1
    d_state: int = 16             # ∈ {16,32}
    expand: float = 2.0           # FFN expansion
    K_context: int = 14           # pool last K days
    horizons: Tuple[int, ...] = (1, 3, 7, 14)
    use_spatial: bool = True
    spatial_mode: str = "graph"   # "graph" | "conv"
    pos_weight_per_h: Optional[torch.Tensor] = None
    focal_gamma: float = 2.0
    brier_lambda: float = 0.05
    amp_dtype: Optional[str] = "bf16"  # "bf16" | "fp16" | None
    use_compile: bool = True
    use_checkpoint: bool = True
    chunk_len: Optional[int] = None    # e.g., 128
    pack_cells: bool = True
    # synthetic data sizes (for demo)
    C_past: int = 16
    C_known: int = 6
    C_static: int = 8
    C_doy: int = 4
    T_hist: int = 64
    B: int = 64
    H_grid: int = 16  # toy raster H (only for conv demo)
    W_grid: int = 16  # toy raster W (only for conv demo)

# ========= Utilities =========
def maybe_amp_dtype(cfg: Config):
    if cfg.amp_dtype == "bf16":
        return torch.bfloat16
    if cfg.amp_dtype == "fp16":
        return torch.float16
    return None

def gelu(x):  # explicit to keep LayerNorm in fp32 while ops can be AMP-ed
    return F.gelu(x)

# ========= Feature Projection & Gating (TFT-lite) =========
class FeatureProj(nn.Module):
    """
    Projects inputs to D and fuses via sigmoid gate.
    Inputs:
      x_past: [B, T_hist, C_past]
      x_known_future (optional aligned from t=1..T_hist+H): [B, T_hist+H, C_known] (we slice [:, :T_hist])
      x_static: [B, C_static]
      doy_enc (optional): [B, T_hist, C_doy]
    Output:
      Z0: [B, T_hist, D]
    """
    def __init__(self, C_past, C_known, C_static, C_doy, D, dropout=0.1):
        super().__init__()
        self.p_past = nn.Linear(C_past, D)
        self.p_known = nn.Linear(C_known, D) if C_known > 0 else None
        self.p_doy = nn.Linear(C_doy, D) if C_doy > 0 else None
        self.p_static = nn.Linear(C_static, D)
        self.gate = nn.Linear(4 * D, D)  # concat then gate
        self.out = nn.Linear(4 * D, D)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(D)  # keep in fp32 automatically

    def forward(self, x_past, x_known_future, x_static, doy_enc):
        # shapes
        B, T, _ = x_past.shape  # [B, T_hist, C_past]
        z_past = self.p_past(x_past)  # [B, T, D]

        if (x_known_future is not None) and (self.p_known is not None):
            z_known = self.p_known(x_known_future[:, :T, :])  # align first T days -> [B, T, D]
        else:
            z_known = torch.zeros_like(z_past)

        if (doy_enc is not None) and (self.p_doy is not None):
            z_doy = self.p_doy(doy_enc)  # [B, T, D]
        else:
            z_doy = torch.zeros_like(z_past)

        z_static = self.p_static(x_static).unsqueeze(1).expand(B, T, -1)  # [B, T, D]

        z_cat = torch.cat([z_past, z_known, z_doy, z_static], dim=-1)  # [B, T, 4D]
        g = torch.sigmoid(self.gate(z_cat))  # [B, T, D]
        z = self.out(z_cat)  # [B, T, D]
        z = g * z + (1 - g) * z_past  # gated fusion; residual bias to observed path
        z = self.dropout(self.ln(z))
        return z  # [B, T, D]

# ========= Mamba/SSM layer placeholder =========
class SomeMambaSSMLayer(nn.Module):
    """
    Placeholder for a kernelized Mamba/SSM 1D selective-scan layer.
    API stable for swapping in a C++/CUDA/Triton impl later.
    Input:  x: [B, T, D]
    Output: y: [B, T, D]
    """
    def __init__(self, d_model: int, d_state: int, expand: float):
        super().__init__()
        d_inner = int(d_model * expand)
        # Lightweight stand-in: depthwise temporal conv + pointwise proj simulating linear-time scan
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.proj_in = nn.Linear(d_model, d_inner)
        self.proj_out = nn.Linear(d_inner, d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, D = x.shape
        y = x.transpose(1, 2)  # [B, D, T]
        y = self.dw_conv(y).transpose(1, 2)  # [B, T, D]
        y = self.proj_out(self.dropout(self.act(self.proj_in(y))))
        return y  # [B, T, D]

# ========= Mamba Block =========
class Mamba1DBlock(nn.Module):
    """
    LN -> SSM -> Drop -> Resid -> FFN(GELU) -> Drop -> Resid
    Input/Output: [B, T, D]
    """
    def __init__(self, d_model, d_state, expand, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ssm = SomeMambaSSMLayer(d_model, d_state, expand)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        inner = int(d_model * expand)
        self.ff = nn.Sequential(
            nn.Linear(d_model, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        y = self.ln1(x)
        y = self.ssm(y)
        x = x + self.drop1(y)
        y = self.ln2(x)
        y = self.ff(y)
        x = x + self.drop2(y)
        return x  # [B, T, D]

# ========= Temporal Encoder =========
class TemporalMambaEncoder(nn.Module):
    """
    FeatureProj -> N * Mamba1DBlock
    Input:
      x_past [B, T, C_past], x_known_future [B, T+H, C_known]|None,
      x_static [B, C_static], doy_enc [B, T, C_doy]
    Output:
      Z_temporal [B, T, D]
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.proj = FeatureProj(cfg.C_past, cfg.C_known, cfg.C_static, cfg.C_doy, cfg.D, cfg.dropout)
        self.blocks = nn.ModuleList([Mamba1DBlock(cfg.D, cfg.d_state, cfg.expand, cfg.dropout) for _ in range(cfg.depth)])

    def forward(self, x_past, x_known_future, x_static, doy_enc, use_checkpoint: bool = False):
        z = self.proj(x_past, x_known_future, x_static, doy_enc)  # [B, T, D]
        for blk in self.blocks:
            if use_checkpoint:
                z = torch.utils.checkpoint.checkpoint(blk, z, use_reentrant=False)
            else:
                z = blk(z)
        return z  # [B, T, D]

# ========= Temporal Context Pooling =========
class ContextPool(nn.Module):
    """
    Pools last K steps by mean (simple; attention could be swapped in).
    Input: Z [B, T, D]
    Output: C [B, D]
    """
    def __init__(self, K: int):
        super().__init__()
        self.K = K

    def forward(self, Z):
        K = min(self.K, Z.size(1))
        c = Z[:, -K:, :].mean(dim=1)  # [B, D]
        return c

# ========= Tiny Spatial Coupling =========
class GraphConvStub(nn.Module):
    """
    Simple graph conv: X' = LN( X + Drop( A * X W ) )
    edge_index: [2, E], edge_weight: [E] or None
    Input/Output X: [B, N, D] (we treat batch as separate graphs with shared edges)
    """
    def __init__(self, D, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(D, D)
        self.ln = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)

    def forward(self, X, edge_index, edge_weight=None):
        # X: [B, N, D]
        B, N, D = X.shape
        src, dst = edge_index  # [E], [E]
        E = src.numel()
        h = self.lin(X)  # [B, N, D]
        # Message passing with scatter-add
        msg = torch.zeros_like(h)
        w = edge_weight if edge_weight is not None else X.new_ones(E)
        # For speed, vectorize over edges
        # Gather src features: [B, E, D]
        h_src = h[:, src, :] * w.view(1, E, 1)
        # Scatter to dst
        msg.index_add_(1, dst, h_src)  # accumulate along node dimension
        out = self.ln(X + self.drop(msg))
        return out  # [B, N, D]

class SpatialStepGraph(nn.Module):
    """1–2 GraphConvStub layers."""
    def __init__(self, D, layers=1, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([GraphConvStub(D, dropout) for _ in range(layers)])

    def forward(self, C, edge_index, edge_weight=None):
        # C: [B, N, D] -> [B, N, D]
        x = C
        for g in self.layers:
            x = g(x, edge_index, edge_weight)
        return x

class SpatialStepConv3x3(nn.Module):
    """
    Depthwise 3x3 conv for raster mode, channels_last-aware.
    Input: Cmap [B, H, W, D] channels_last
    Output: [B, H, W, D]
    """
    def __init__(self, D, dropout=0.1):
        super().__init__()
        self.dw = nn.Conv2d(D, D, kernel_size=3, padding=1, groups=D)
        self.pw = nn.Conv2d(D, D, kernel_size=1)
        self.ln = nn.LayerNorm(D)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, Cmap):
        # Expect channels_last
        x = Cmap.permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]
        y = self.pw(self.dw(x))
        y = self.drop(y)
        y = y.permute(0, 2, 3, 1).contiguous()  # [B, H, W, D]
        y = self.ln(y)
        return y

# ========= Multi-Horizon Head =========
class MultiHorizonHead(nn.Module):
    """
    Shared head with learned horizon embeddings.
    Input: C' [B, D]; Output: logits [B, |H|]
    Also supports separate heads (optional).
    """
    def __init__(self, D, horizons: Tuple[int, ...], shared=True, dropout=0.1):
        super().__init__()
        self.horizons = list(horizons)
        self.D = D
        self.shared = shared
        self.e_h = nn.Embedding(len(horizons), D)
        if shared:
            self.ln = nn.LayerNorm(D)
            self.mlp = nn.Sequential(
                nn.Linear(D, D),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(D, 1),
            )
        else:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(D),
                    nn.Linear(D, D), nn.GELU(), nn.Dropout(dropout), nn.Linear(D, 1)
                ) for _ in horizons
            ])

    def forward(self, Cprime):
        # Cprime: [B, D]
        B = Cprime.size(0)
        if self.shared:
            logits = []
            base = self.ln(Cprime)  # [B, D]
            for i in range(len(self.horizons)):
                z = base + self.e_h.weight[i]  # [D] broadcast -> [B, D]
                logit = self.mlp(z)  # [B, 1]
                logits.append(logit)
            logits = torch.cat(logits, dim=1)  # [B, |H|]
        else:
            logits = []
            for i, head in enumerate(self.heads):
                z = Cprime + self.e_h.weight[i]
                logits.append(head(z))  # [B,1]
            logits = torch.cat(logits, dim=1)
        return logits  # [B, |H|]

# ========= Full Model =========
class FireIgnitionModel(nn.Module):
    """
    encoder -> context pooling -> (optional) spatial -> head
    For graph mode, inputs are flattened cells (N = batch cells).
    For conv mode, we reshape to [H,W] toy grid purely for demo.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = TemporalMambaEncoder(cfg)
        self.pool = ContextPool(cfg.K_context)
        if cfg.use_spatial:
            if cfg.spatial_mode == "graph":
                self.spatial = SpatialStepGraph(cfg.D, layers=1, dropout=cfg.dropout)
            else:
                self.spatial = SpatialStepConv3x3(cfg.D, dropout=cfg.dropout)
        else:
            self.spatial = None
        self.head = MultiHorizonHead(cfg.D, cfg.horizons, shared=True, dropout=cfg.dropout)

    def forward(self, x_past, x_known_future, x_static, doy_enc,
                graph_info=None, grid_info=None, use_checkpoint=False):
        """
        x_past: [B, T, C_past]
        x_known_future: [B, T+Hmax, C_known] or None
        x_static: [B, C_static]
        doy_enc: [B, T, C_doy] or None
        graph_info: dict with 'edge_index' [2,E], 'edge_weight' [E] (graph mode)
        grid_info: dict with 'H','W' mapping for demo (conv mode)
        Returns: logits [B, |H|]
        """
        Z = self.encoder(x_past, x_known_future, x_static, doy_enc, use_checkpoint=use_checkpoint)  # [B,T,D]
        C = self.pool(Z)  # [B, D]
        if self.spatial is not None:
            if self.cfg.spatial_mode == "graph":
                # Treat batch as N nodes (cells); tiny step over shared edges
                # Here, we assume B == N for simplicity of the demo.
                edge_index = graph_info['edge_index'] if graph_info else torch.empty(2,0, dtype=torch.long, device=C.device)
                edge_weight = graph_info.get('edge_weight', None) if graph_info else None
                C_ = self.spatial(C.unsqueeze(0), edge_index, edge_weight)  # [1, N, D]
                Cprime = C_.squeeze(0)  # [B, D]
            else:
                # Map B cells to H×W toy grid (demo only)
                H, W = grid_info['H'], grid_info['W']
                assert H*W == C.size(0), "For demo, B must equal H*W"
                Cmap = C.view(H, W, -1).unsqueeze(0).contiguous(memory_format=torch.channels_last)  # [1,H,W,D]
                Cmap = self.spatial(Cmap)  # [1,H,W,D]
                Cprime = Cmap.view(-1, self.cfg.D)  # [B,D]
        else:
            Cprime = C  # [B, D]
        logits = self.head(Cprime)  # [B, |H|]
        return logits

# ========= Losses =========
def focal_loss(logits, targets, gamma=2.0, reduction='mean'):
    """
    logits: [B, H], targets: [B, H] in {0,1}
    """
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)
    w = (1 - pt).pow(gamma)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss = (w * bce)
    return loss.mean() if reduction == 'mean' else loss.sum()

def class_balanced_bce(logits, targets, pos_weight_per_h=None, reduction='mean'):
    """
    pos_weight_per_h: [H] tensor; if None, defaults to 1.0
    """
    if pos_weight_per_h is None:
        return F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)
    # Broadcast per-horizon weights
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss = loss * torch.where(targets > 0, pos_weight_per_h, 1.0).unsqueeze(0)
    return loss.mean() if reduction == 'mean' else loss.sum()

def brier_loss(logits, targets, reduction='mean'):
    p = torch.sigmoid(logits)
    loss = (p - targets).pow(2)
    return loss.mean() if reduction == 'mean' else loss.sum()

# ========= Calibration (temperature scaling per horizon) =========
class TemperatureScaler(nn.Module):
    """
    One scalar temperature per horizon.
    """
    def __init__(self, H):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(H))  # init T=1

    def forward(self, logits):
        T = self.log_t.exp().unsqueeze(0)  # [1,H]
        return logits / T

def temperature_scaling_per_horizon(logits, scaler: TemperatureScaler):
    return scaler(logits)

# ========= Metrics (stubs) =========
@torch.no_grad()
def pr_auc_per_horizon(logits, targets):
    # Stub: return random-ish tensor for demo; plug sklearn/torchmetrics in practice
    H = logits.size(1)
    return torch.rand(H, device=logits.device)

@torch.no_grad()
def precision_at_k(logits, targets, k=100):
    # logits/targets: [B,H]; pick top-k cells per horizon and measure precision
    B, H = logits.shape
    topk_idx = logits.topk(min(k, B), dim=0).indices  # [k,H]
    gathered = targets.gather(0, topk_idx)           # [k,H]
    prec = gathered.float().mean(dim=0)              # [H]
    return prec

@torch.no_grad()
def ece(logits, targets, n_bins=15):
    # Simple ECE stub
    p = torch.sigmoid(logits)
    H = p.size(1)
    return torch.rand(H, device=p.device)

# ========= Synthetic Dataset & Sampling =========
class SyntheticPeatlandDataset(Dataset):
    """
    Emits:
      x_past [T,C_past], x_known_future [T+Hmax,C_known] or None,
      x_static [C_static], doy_enc [T,C_doy], labels [|H|]
    """
    def __init__(self, cfg: Config, size=4096, with_known=True):
        super().__init__()
        self.cfg = cfg
        self.size = size
        self.with_known = with_known

    def __len__(self): return self.size

    def __getitem__(self, idx):
        T, Hmax = self.cfg.T_hist, max(self.cfg.horizons)
        x_past = torch.randn(T, self.cfg.C_past)
        x_known = torch.randn(T + Hmax, self.cfg.C_known) if self.with_known and self.cfg.C_known>0 else None
        x_static = torch.randn(self.cfg.C_static)
        doy = torch.randn(T, self.cfg.C_doy) if self.cfg.C_doy>0 else None
        # Labels: sparse positives
        y = (torch.rand(len(self.cfg.horizons)) < 0.05).float()
        return x_past, x_known, x_static, doy, y

def collate_batch(batch):
    # For simplicity, assume same T; stack along batch. Pad logic could be added here.
    xs, xk, xsn, doy, y = zip(*batch)
    x_past = torch.stack(xs, 0)                     # [B,T,C_past]
    x_known = None if xk[0] is None else torch.stack(xk, 0)  # [B,T+H,C_known]
    x_static = torch.stack(xsn, 0)                  # [B,C_static]
    doy_enc = None if doy[0] is None else torch.stack(doy, 0)  # [B,T,C_doy]
    y = torch.stack(y, 0)                           # [B,|H|]
    return x_past, x_known, x_static, doy_enc, y

# ========= Optimizer & Scheduler =========
def build_optimizer(model, lr=2e-4, wd=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lrs.append(base_lr * step / self.warmup_steps)
            else:
                t = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lrs.append(0.5 * base_lr * (1 + math.cos(math.pi * t)))
        return lrs

# ========= Train/Eval =========
def train_one_epoch(model, cfg: Config, loader, optimizer, scaler_fp16=None, scheduler=None,
                    use_focal=True, device="cuda"):
    model.train()
    amp_dtype = maybe_amp_dtype(cfg)
    total_loss, n = 0.0, 0
    Hmax = max(cfg.horizons)
    for batch in loader:
        x_past, x_known, x_static, doy, y = batch
        x_past = x_past.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        doy = doy.to(device, non_blocking=True) if doy is not None else None
        x_known = x_known.to(device, non_blocking=True) if x_known is not None else None

        optimizer.zero_grad(set_to_none=True)
        autocast_dtype = torch.bfloat16 if amp_dtype == torch.bfloat16 else torch.float16 if amp_dtype == torch.float16 else None
        ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype else torch.cuda.amp.autocast(enabled=False)
        with ctx:
            # Optional chunked BPTT over time
            if cfg.chunk_len is not None and cfg.T_hist > cfg.chunk_len:
                # NOTE: for true chunked BPTT, model should accept prev state; here we recompute as a placeholder.
                logits = model(x_past, x_known, x_static, doy, graph_info=None, grid_info=None, use_checkpoint=cfg.use_checkpoint)
            else:
                logits = model(x_past, x_known, x_static, doy, graph_info=None, grid_info=None, use_checkpoint=cfg.use_checkpoint)  # [B,|H|]

            if use_focal:
                loss_main = focal_loss(logits, y, gamma=cfg.focal_gamma)
            else:
                loss_main = class_balanced_bce(logits, y, pos_weight_per_h=cfg.pos_weight_per_h)

            loss = loss_main + cfg.brier_lambda * brier_loss(logits, y)

        if amp_dtype == torch.float16:
            scaler_fp16.scale(loss).backward()
            scaler_fp16.step(optimizer)
            scaler_fp16.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler: scheduler.step()
        total_loss += loss.item() * x_past.size(0)
        n += x_past.size(0)
    return total_loss / max(1, n)

@torch.no_grad()
def evaluate(model, cfg: Config, loader, device="cuda", scaler: Optional[TemperatureScaler]=None):
    model.eval()
    amp_dtype = maybe_amp_dtype(cfg)
    H = len(cfg.horizons)
    total_brier = torch.zeros(H, device=device)
    total_prauc = torch.zeros(H, device=device)
    total_prec = torch.zeros(H, device=device)
    total_ece = torch.zeros(H, device=device)
    total_n = 0

    with torch.inference_mode():
        autocast_dtype = torch.bfloat16 if amp_dtype == torch.bfloat16 else torch.float16 if amp_dtype == torch.float16 else None
        ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype else torch.cuda.amp.autocast(enabled=False)
        for batch in loader:
            x_past, x_known, x_static, doy, y = batch
            x_past = x_past.to(device, non_blocking=True)
            x_static = x_static.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            doy = doy.to(device, non_blocking=True) if doy is not None else None
            x_known = x_known.to(device, non_blocking=True) if x_known is not None else None

            with ctx:
                logits = model(x_past, x_known, x_static, doy, graph_info=None, grid_info=None, use_checkpoint=False)  # [B,H]
                if scaler is not None:
                    logits = temperature_scaling_per_horizon(logits, scaler)

            total_brier += brier_loss(logits, y, reduction='sum')
            total_prauc += pr_auc_per_horizon(logits, y)
            total_prec += precision_at_k(logits, y, k=100)
            total_ece += ece(logits, y)
            total_n += x_past.size(0)

    return {
        "brier": (total_brier / max(1, total_n)).tolist(),
        "prauc": (total_prauc / max(1, total_n)).tolist(),
        "precision@100": (total_prec / max(1, total_n)).tolist(),
        "ece": (total_ece / max(1, total_n)).tolist(),
    }

# ========= Build graph edges (8-neighbor + k downwind) STUB =========
def build_grid_edges_8nbr(H, W, k_downwind=2, wind_dir_rad=0.0, device="cuda"):
    """
    Returns edge_index [2,E] for an HxW grid with 8-neighbor + k downwind directed edges (toy).
    """
    idx = torch.arange(H * W, device=device)
    r = idx // W
    c = idx % W

    def valid(rr, cc):
        return (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)

    edges = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0: continue
            rr, cc = r + dr, c + dc
            mask = valid(rr, cc)
            src = idx[mask]
            dst = (rr * W + cc)[mask]
            edges.append(torch.stack([src, dst], 0))
    # add k directed downwind (projected by wind_dir)
    wr, wc = math.sin(wind_dir_rad), math.cos(wind_dir_rad)
    dr = int(round(wr))
    dc = int(round(wc))
    for _ in range(max(0, k_downwind)):
        rr, cc = r + dr, c + dc
        mask = valid(rr, cc)
        src = idx[mask]
        dst = (rr * W + cc)[mask]
        edges.append(torch.stack([src, dst], 0))
    edge_index = torch.cat(edges, dim=1) if edges else torch.empty(2,0, dtype=torch.long, device=device)
    return edge_index

# ========= Main =========
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    torch.manual_seed(0)

    # Dataset / Loader
    train_ds = SyntheticPeatlandDataset(cfg, size=4096, with_known=True)
    val_ds = SyntheticPeatlandDataset(cfg, size=1024, with_known=True)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.B, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.B, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True, collate_fn=collate_batch
    )

    model = FireIgnitionModel(cfg).to(device)
    # Optional channels_last for conv spatial path (kept ready)
    if cfg.spatial_mode == "conv":
        model = model.to(memory_format=torch.channels_last)

    if cfg.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")  # PyTorch 2.1+

    optimizer = build_optimizer(model, lr=2e-4, wd=1e-4)

    total_steps = len(train_loader) * 3  # small demo epochs
    warmup_steps = max(1, int(0.02 * total_steps))  # ~2%
    scheduler = CosineWithWarmup(optimizer, warmup_steps, total_steps)

    scaler_fp16 = torch.cuda.amp.GradScaler(enabled=(cfg.amp_dtype == "fp16"))
    temp_scaler = TemperatureScaler(H=len(cfg.horizons)).to(device)

    # Graph edges for demo (graph mode only): assume B==H*W or B==N nodes, reuse per batch
    graph_edges = None
    grid_info = None
    if cfg.use_spatial and cfg.spatial_mode == "graph":
        # For demo, we’ll build edges on-the-fly in forward if needed. Here, prepare a cached dict if batch size matches grid.
        graph_edges = {"edge_index": build_grid_edges_8nbr(cfg.H_grid, cfg.W_grid, device=device)}

    t0 = time.time()
    # Train loop (few steps)
    for epoch in range(3):
        loss = train_one_epoch(model, cfg, train_loader, optimizer, scaler_fp16, scheduler, use_focal=True, device=device)
        metrics = evaluate(model, cfg, val_loader, device=device, scaler=None)
        print(f"Epoch {epoch} | loss={loss:.4f} | brier={metrics['brier']} | prauc={metrics['prauc']} | p@100={metrics['precision@100']}")

    t1 = time.time()
    samples = len(train_ds) + len(val_ds)
    print(f"Throughput: {samples / (t1 - t0 + 1e-6):.1f} samples/s")

    # Inference example with torch.inference_mode + compiled model (already compiled)
    model.eval()
    with torch.inference_mode():
        batch = next(iter(val_loader))
        x_past, x_known, x_static, doy, y = batch
        x_past = x_past.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)
        doy = doy.to(device, non_blocking=True) if doy is not None else None
        x_known = x_known.to(device, non_blocking=True) if x_known is not None else None
        logits = model(x_past, x_known, x_static, doy, graph_info=graph_edges, grid_info=grid_info, use_checkpoint=False)
        probs = torch.sigmoid(logits)
        print("Inference probs (first row):", probs[0].tolist())

    # (Optional) INT8 quantization stub for head/MLPs could be added here.

if __name__ == "__main__":
    main()
