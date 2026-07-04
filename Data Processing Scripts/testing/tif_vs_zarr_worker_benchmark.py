#!/usr/bin/env python3
"""
GeoTIFF ↔ Zarr benchmark — train-like dataloading

What this models from your training loop
----------------------------------------
- Batch is a dict with {x_dyn, x_pos, y, y_mask}
- DataLoader settings (workers, prefetch, pin_memory, persistent_workers, shuffle)
- Per-worker lazy dataset open + per-worker RNG seeding
- Optional device transfer + AMP-like dtype cast timing
- Temporal stacks: read T_hist windows across daily files (or repeat from one file)

Backends
--------
- GeoTIFF via rasterio
- Zarr (chunking/compression configurable; supports consolidated metadata)

Usage examples
--------------
# Single daily tif (simulate T_hist by repeated reads from same file)
python tif_vs_zarr_trainlike_benchmark.py \
  --tif data/era5/day_2021-01-01.tif \
  --zarr-out /tmp/day_2021-01-01.zarr \
  --convert --chunk-h 512 --chunk-w 512 --compressor zstd --clevel 3 \
  --t-hist 30 --patch 256 --num-samples 1024 \
  --batch-size 24 --workers 8 --prefetch 2 --shuffle \
  --device-transfer --amp bf16

# Multiple daily tifs (one file per day)
python tif_vs_zarr_trainlike_benchmark.py \
  --tif-list paths_days.txt \
  --zarr-out /tmp/era5_days.zarr --convert \
  --t-hist 30 --patch 256 --num-samples 1024 \
  --batch-size 24 --workers 8 --prefetch 2 --shuffle \
  --device-transfer --amp bf16 --zarr-consolidated

Notes
-----
- We do NOT run your model; this isolates input-pipeline + (optional) H→D transfer.
- x_dyn shape: (T_hist, C, patch, patch) to reflect temporal stacks before your dataset reshapes.
- x_pos: lightweight synthetic feature (pos_dim) derived from window location (simulates SIREN input cost).
- y and y_mask: dummy tensors shaped like your classification head targets/masks.
"""

from __future__ import annotations
import argparse, os, time, random, math, json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext
from tqdm.auto import tqdm

# ----------------------
# Pretty-print utilities
# ----------------------

def human_bytes(n: float) -> str:
    for u in ["B","KB","MB","GB","TB","PB"]:
        if abs(n) < 1024: return f"{n:.2f}{u}"
        n /= 1024.0
    return f"{n:.2f}EB"

def print_header(title: str):
    print("\n" + "="*64)
    print(f"  {title}")
    print("="*64)

def choose_mp_context(requested: str) -> Optional[str]:
    if requested != "auto":
        return requested
    # Safer default for CUDA + workers
    if os.name == "posix":
        return "forkserver"
    return "spawn"

# -----------------
# Zarr conversion
# -----------------

def convert_tif_to_zarr(
    tif_or_list: List[str],
    zarr_out: str,
    chunk_c: int = 1,
    chunk_h: int = 512,
    chunk_w: int = 512,
    compressor: str = "zstd",
    clevel: int = 3,
    overwrite: bool = False,
    consolidate: bool = False
) -> Tuple[int, int, int, str]:
    """
    Convert one or many GeoTIFFs into a single Zarr array with shape (T, C, H, W),
    where T = number of input files. If only one tif is supplied, T=1.

    Returns (C, H, W, dtype_str). (Note: T is stored in metadata.)
    """
    import rasterio
    import zarr
    from numcodecs import Blosc

    if os.path.exists(zarr_out):
        if overwrite:
            import shutil; shutil.rmtree(zarr_out)
        else:
            print(f"[convert] Zarr exists at {zarr_out}; skipping overwrite.")
            arr = zarr.open(os.path.join(zarr_out, "data"), mode="r")
            # If stored as (T,C,H,W), expose per-slice (C,H,W)
            C, H, W = int(arr.shape[-3]), int(arr.shape[-2]), int(arr.shape[-1])
            return C, H, W, str(arr.dtype)

    print_header("GeoTIFF → Zarr Conversion")
    print(f"Inputs: {len(tif_or_list)} file(s)")
    print(f"Output: {zarr_out}")

    # Probe first file for shape/dtype
    with rasterio.open(tif_or_list[0], "r") as src0:
        C, H, W = src0.count, src0.height, src0.width
        dtype = src0.dtypes[0]
        dtype_np = np.dtype(dtype)
        print(f"Per-day raster: C={C}, H={H}, W={W}, dtype={dtype}")

    # Layout: (T, C, H, W) for temporal-first packing
    Tn = len(tif_or_list)
    store_path = os.path.join(zarr_out, "data")
    from numcodecs import Blosc
    compressor_obj = Blosc(cname=compressor, clevel=clevel, shuffle=Blosc.SHUFFLE)

    # Chunking strategy: time×spatial chunks. We chunk time by 1 unless user overrides via chunk_c used for C.
    # We'll keep (T, C, H, W) but use chunks (1, chunk_c, chunk_h, chunk_w)
    arr = zarr.open(
        store_path, mode="w",
        shape=(Tn, C, H, W),
        chunks=(1, min(chunk_c, C), min(chunk_h, H), min(chunk_w, W)),
        dtype=dtype_np, compressor=compressor_obj, zarr_format=2
    )

    # Tile size for copy
    tile_h = min(chunk_h * 4, H)
    tile_w = min(chunk_w * 4, W)
    print(f"Writing tiles of ~({tile_h}x{tile_w}) per band...")

    for t, tif_path in enumerate(tif_or_list):
        with rasterio.open(tif_path, "r") as src:
            assert src.count == C and src.height == H and src.width == W, \
                "All input tifs must share identical (C,H,W)"
            for b in range(1, C + 1):
                for r0 in range(0, H, tile_h):
                    r1 = min(H, r0 + tile_h)
                    for c0 in range(0, W, tile_w):
                        c1 = min(W, c0 + tile_w)
                        window = rasterio.windows.Window.from_slices((r0, r1), (c0, c1))
                        block = src.read(b, window=window)
                        arr[t, b - 1, r0:r1, c0:c1] = block

    meta = zarr.open_group(zarr_out, mode="a")
    meta.attrs.update({
        "inputs": tif_or_list,
        "dtype": dtype,
        "shape": [Tn, int(C), int(H), int(W)],
        "chunks": list(arr.chunks),
        "compressor": compressor,
        "clevel": clevel,
        "layout": "T,C,H,W"
    })

    if consolidate:
        from zarr.convenience import consolidate_metadata
        consolidate_metadata(zarr_out)
        print("✓ Consolidated metadata written.")
    print("✓ Conversion complete.")
    return C, H, W, dtype

# -----------------------------
# Sampler and helper utilities
# -----------------------------

@dataclass
class PatchSampler:
    H: int
    W: int
    patch: int
    seed: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def reseed(self, new_seed: int):
        self.rng = random.Random(int(new_seed) & 0x7fffffff)

    def sample_rc(self):
        if self.patch > self.H or self.patch > self.W:
            return 0, 0
        r0 = self.rng.randint(0, self.H - self.patch)
        c0 = self.rng.randint(0, self.W - self.patch)
        return int(r0), int(c0)

def make_xpos(pos_dim: int, H: int, W: int, r0: int, c0: int, patch: int, dtype=np.float32):
    # Cheap positional features derived from normalized window location & size
    y = np.array([
        r0 / max(1, H - patch),
        c0 / max(1, W - patch),
        patch / max(H, W),
        1.0
    ], dtype=dtype)
    # Expand to pos_dim with simple harmonics
    feats = [y[0], y[1], y[2], y[3]]
    k = 1
    while len(feats) < pos_dim:
        feats += [math.sin(k*math.pi*y[0]), math.cos(k*math.pi*y[0]),
                  math.sin(k*math.pi*y[1]), math.cos(k*math.pi*y[1])]
        k += 1
    return np.array(feats[:pos_dim], dtype=dtype)

# -----------------------------------------
# Train-like datasets (GeoTIFF and Zarr)
# -----------------------------------------

class TemporalWindowTif(Dataset):
    """
    Returns a dict:
      x_dyn:  (T, C, P, P) float32
      x_pos:  (1, pos_dim) float32
      y:      (Hout,) float32  (dummy)
      y_mask: (Hout,) float32  (dummy)
    """
    def __init__(self,
                 tifs: List[str],
                 patch: int,
                 t_hist: int,
                 num_samples: int,
                 pos_dim: int,
                 horizons: int,
                 seed: int = 0):
        import rasterio
        self.tifs = tifs
        self.patch = int(patch)
        self.t_hist = int(t_hist)
        self.num_samples = int(num_samples)
        self.pos_dim = int(pos_dim)
        self.horizons = int(horizons)
        self._src_first = None  # defer
        with rasterio.open(tifs[0], "r") as src:
            self.C, self.H, self.W = src.count, src.height, src.width
        self.sampler = PatchSampler(self.H, self.W, self.patch, seed)
        self._src_cache = {}  # per-worker dict: index -> opened dataset

    def _open_src(self, idx):
        import rasterio
        if idx not in self._src_cache:
            self._src_cache[idx] = rasterio.open(self.tifs[idx], "r")
        return self._src_cache[idx]

    def __len__(self): return self.num_samples

    def __getitem__(self, _):
        # Sample one window location shared across time
        r0, c0 = self.sampler.sample_rc()
        r1, c1 = r0 + self.patch, c0 + self.patch
        from rasterio.windows import Window
        window = Window.from_slices((r0, r1), (c0, c1))

        # Select T files: if only one file is provided, repeat it T times
        if len(self.tifs) >= self.t_hist:
            # take a contiguous slice if possible; otherwise sample random indices
            start = self.sampler.rng.randint(0, len(self.tifs) - self.t_hist) if len(self.tifs) > self.t_hist else 0
            t_indices = list(range(start, start + self.t_hist))
        else:
            t_indices = [0] * self.t_hist

        # Read same window across T
        frames = []
        for ti in t_indices:
            src = self._open_src(ti)
            data = src.read(window=window)  # (C, p, p)
            h, w = data.shape[1], data.shape[2]
            if h != self.patch or w != self.patch:
                tmp = np.zeros((self.C, self.patch, self.patch), dtype=data.dtype)
                tmp[:, :h, :w] = data
                data = tmp
            frames.append(np.asarray(data, dtype=np.float32))
        x_dyn = np.stack(frames, axis=0)  # (T, C, P, P)

        x_pos = make_xpos(self.pos_dim, self.H, self.W, r0, c0, self.patch)[None, :]  # (1, pos_dim)
        y      = np.zeros((self.horizons,), dtype=np.float32)
        y_mask = np.ones((self.horizons,),  dtype=np.float32)

        return {
            "x_dyn":  torch.from_numpy(x_dyn),      # float32
            "x_pos":  torch.from_numpy(x_pos),
            "y":      torch.from_numpy(y),
            "y_mask": torch.from_numpy(y_mask),
        }

class TemporalWindowZarr(Dataset):
    # Expects a Zarr with shape (T_total, C, H, W). If converted from single file, T_total=1.
    def __init__(self,
                 zarr_root: str,
                 patch: int,
                 t_hist: int,
                 num_samples: int,
                 pos_dim: int,
                 horizons: int,
                 seed: int = 0,
                 consolidated: bool = False):
        import zarr
        self.path = zarr_root
        self.patch = int(patch)
        self.t_hist = int(t_hist)
        self.num_samples = int(num_samples)
        self.pos_dim = int(pos_dim)
        self.horizons = int(horizons)
        self._arr = None
        if consolidated:
            # open_consolidated works if metadata consolidated during conversion
            try:
                from zarr.convenience import open_consolidated
                self._arr = open_consolidated(os.path.join(self.path, "data"))
            except Exception:
                self._arr = None
        if self._arr is None:
            self._arr = zarr.open(os.path.join(self.path, "data"), mode="r")
        self.Ttot, self.C, self.H, self.W = map(int, self._arr.shape)
        self.sampler = PatchSampler(self.H, self.W, self.patch, seed)

    def __len__(self): return self.num_samples

    def __getitem__(self, _):
        r0, c0 = self.sampler.sample_rc()
        r1, c1 = r0 + self.patch, c0 + self.patch

        # Choose a contiguous segment in time if possible
        if self.Ttot >= self.t_hist:
            start = self.sampler.rng.randint(0, self.Ttot - self.t_hist) if self.Ttot > self.t_hist else 0
            t_indices = slice(start, start + self.t_hist)
            data = self._arr[t_indices, :, r0:r1, c0:c1]  # (T, C, p, p)
        else:
            # Repeat the only slice
            data = np.repeat(self._arr[0:1, :, r0:r1, c0:c1], self.t_hist, axis=0)

        # Pad if needed
        T, C, h, w = data.shape
        if h != self.patch or w != self.patch:
            out = np.zeros((T, C, self.patch, self.patch), dtype=data.dtype)
            out[:, :, :h, :w] = data
            data = out
        x_dyn = np.asarray(data, dtype=np.float32)

        x_pos = make_xpos(self.pos_dim, self.H, self.W, r0, c0, self.patch)[None, :]
        y      = np.zeros((self.horizons,), dtype=np.float32)
        y_mask = np.ones((self.horizons,),  dtype=np.float32)

        return {
            "x_dyn":  torch.from_numpy(x_dyn),
            "x_pos":  torch.from_numpy(x_pos),
            "y":      torch.from_numpy(y),
            "y_mask": torch.from_numpy(y_mask),
        }

# -----------------------
# Collate & worker seeding
# -----------------------

def trainlike_collate(batch: List[Dict[str, torch.Tensor]]):
    # Mirrors your collate: pre-allocate and stack dict fields
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out

def worker_init_fn(_worker_id: int):
    info = torch.utils.data.get_worker_info()
    base_seed = torch.initial_seed()  # different per worker
    ds = info.dataset
    if hasattr(ds, "sampler") and isinstance(ds.sampler, PatchSampler):
        ds.sampler.reseed(base_seed)

# -----------------------
# Timed DataLoader wrapper
# -----------------------

class TimedIterator:
    """
    Tracks:
      - host load time per batch (DataLoader iteration)
      - optional device transfer time
    """
    def __init__(self, loader, desc, device=None, device_transfer=False, amp: Optional[str]=None):
        self.loader = loader
        self.desc = desc
        self.device = device
        self.device_transfer = device_transfer
        self.amp = amp  # None | 'bf16' | 'fp16'
        self.host_times = []
        self.h2d_times  = []

    def __iter__(self):
        it = iter(self.loader)
        try:
            total = len(self.loader)
        except TypeError:
            total = None

        pbar = tqdm(total=total, desc=self.desc, leave=True,
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                               '[{elapsed}<{remaining}] {postfix}')
        while True:
            try:
                t0 = time.time()
                batch = next(it)
                t1 = time.time()
                host = t1 - t0
                self.host_times.append(host)

                if self.device_transfer:
                    # Cast before move to simulate AMP’s smaller H2D payload
                    if self.amp in ("bf16","fp16"):
                        dtype = torch.bfloat16 if self.amp=="bf16" else torch.float16
                        def _cast(x):
                            # Cast only big tensors (x_dyn) to reduce overhead like in autocast
                            return x.to(dtype=dtype, copy=False) if x.dtype==torch.float32 else x
                    else:
                        def _cast(x): return x

                    t2 = time.time()
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            v = _cast(v)
                            batch[k] = v.to(self.device, non_blocking=True)
                    if self.device.type == "cuda":
                        torch.cuda.current_stream().synchronize()
                    t3 = time.time()
                    self.h2d_times.append(t3 - t2)

                pbar.set_postfix({"load_ms": f"{host*1000:.0f}",
                                  "h2d_ms": f"{(self.h2d_times[-1]*1000):.0f}" if self.h2d_times else "—"})
                pbar.update(1)
                yield batch
            except StopIteration:
                break
        pbar.close()

# -----------------
# Loader builder
# -----------------

def make_loader(ds: Dataset, args) -> DataLoader:
    pin = torch.cuda.is_available()
    kw: Dict[str, Any] = dict(
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=(args.workers > 0),
        collate_fn=trainlike_collate,
        worker_init_fn=worker_init_fn
    )
    if args.workers > 0:
        kw["prefetch_factor"] = args.prefetch
        ctx = choose_mp_context(args.mp_context)
        if ctx:
            kw["multiprocessing_context"] = ctx
    return DataLoader(ds, **kw)

# -----------
# Benchmark
# -----------

def run_benchmark(label: str, ds: Dataset, args) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_loader(ds, args)
    timed = TimedIterator(
        loader, desc=label,
        device=device,
        device_transfer=args.device_transfer,
        amp=args.amp
    )

    t0 = time.time()
    n_batches = 0
    n_samples = 0
    for batch in timed:
        bs = batch["x_dyn"].shape[0]
        n_batches += 1
        n_samples += bs
    t1 = time.time()

    host_avg = (sum(timed.host_times)/len(timed.host_times)) if timed.host_times else float('nan')
    h2d_avg  = (sum(timed.h2d_times) /len(timed.h2d_times))  if timed.h2d_times  else float('nan')
    return dict(
        total_time=t1-t0,
        n_batches=n_batches,
        n_samples=n_samples,
        samples_per_sec=(n_samples / (t1-t0)) if (t1>t0) else 0.0,
        host_avg_ms=host_avg*1000.0,
        h2d_avg_ms=h2d_avg*1000.0
    )

# -----------
# CLI
# -----------

def parse_args():
    p = argparse.ArgumentParser()
    # Inputs
    p.add_argument("--tif", type=str, help="Path to one input .tif file (single-day).")
    p.add_argument("--tif-list", type=str, help="Path to a text file of .tif paths (one per line or CSV).")
    p.add_argument("--zarr-out", required=True, help="Directory for Zarr store")
    p.add_argument("--convert", action="store_true", help="Build Zarr from provided tif(s) first")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--zarr-consolidated", action="store_true", help="Open Zarr with consolidated metadata if available")

    # Zarr chunking/compression
    p.add_argument("--chunk-c", type=int, default=1)
    p.add_argument("--chunk-h", type=int, default=512)
    p.add_argument("--chunk-w", type=int, default=512)
    p.add_argument("--compressor", choices=["zstd","lz4","zlib","blosclz","snappy"], default="zstd")
    p.add_argument("--clevel", type=int, default=3)

    # Train-like shapes/params
    p.add_argument("--t-hist", type=int, default=30)
    p.add_argument("--pos-dim", type=int, default=36)
    p.add_argument("--horizons", type=int, default=4)
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--num-samples", type=int, default=1024)

    # Loader knobs
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--mp-context", choices=["auto","spawn","fork","forkserver"], default="auto")
    p.add_argument("--shuffle", action="store_true")

    # Device transfer simulation
    p.add_argument("--device-transfer", action="store_true", help="Also time host→device copy like your train loop")
    p.add_argument("--amp", choices=["none","bf16","fp16"], default="none", help="Cast big tensors before H→D (approx autocast)")

    return p.parse_args()

def read_tif_list(path: str) -> List[str]:
    with open(path, "r") as f:
        raw = f.read().strip()
    if "," in raw:
        items = [s.strip() for s in raw.split(",") if s.strip()]
    else:
        items = [s.strip() for s in raw.splitlines() if s.strip()]
    return items

def main():
    args = parse_args()
    if not args.tif and not args.tif_list:
        raise SystemExit("Provide --tif or --tif-list")

    # Build input list for conversion & TIF backend
    tifs = [args.tif] if args.tif else read_tif_list(args.tif_list)

    # Optional conversion
    if args.convert:
        C, H, W, dtype = convert_tif_to_zarr(
            tifs, args.zarr_out,
            chunk_c=args.chunk_c, chunk_h=args.chunk_h, chunk_w=args.chunk_w,
            compressor=args.compressor, clevel=args.clevel,
            overwrite=args.overwrite, consolidate=args.zarr_consolidated
        )
        print(f"[convert] Zarr ready: C={C} H={H} W={W} dtype={dtype}")
    elif not os.path.exists(args.zarr_out):
        raise FileNotFoundError(f"Zarr not found at {args.zarr_out}. Use --convert to create it.")

    print_header("Benchmark Setup")
    print(f"T_hist={args.t_hist}  patch={args.patch}  pos_dim={args.pos_dim}  horizons={args.horizons}")
    print(f"samples/backend={args.num_samples}  batch={args.batch_size}")
    print(f"workers={args.workers}  prefetch={args.prefetch}  ctx={choose_mp_context(args.mp_context)}  shuffle={args.shuffle}")
    print(f"device_transfer={args.device_transfer}  amp={args.amp}")

    # Datasets
    tif_ds = TemporalWindowTif(
        tifs=tifs, patch=args.patch, t_hist=args.t_hist, num_samples=args.num_samples,
        pos_dim=args.pos_dim, horizons=args.horizons, seed=42
    )
    zarr_ds = TemporalWindowZarr(
        zarr_root=args.zarr_out, patch=args.patch, t_hist=args.t_hist, num_samples=args.num_samples,
        pos_dim=args.pos_dim, horizons=args.horizons, seed=42, consolidated=args.zarr_consolidated
    )

    print_header("Benchmark: GeoTIFF")
    res_tif = run_benchmark("GeoTIFF", tif_ds, args)

    print_header("Benchmark: Zarr")
    res_zarr = run_benchmark("Zarr", zarr_ds, args)

    # Summary
    print_header("RESULTS")
    def fmt(d):
        return (f"batches={d['n_batches']}, samples={d['n_samples']}, "
                f"host_avg={d['host_avg_ms']:.1f}ms, "
                f"h2d_avg={d['h2d_avg_ms']:.1f}ms, "
                f"total={d['total_time']:.2f}s, "
                f"samp/s={d['samples_per_sec']:.1f}")
    print(f"GeoTIFF: {fmt(res_tif)}")
    print(f"Zarr:    {fmt(res_zarr)}")
    speedup = (res_zarr['samples_per_sec']/res_tif['samples_per_sec']) if res_tif['samples_per_sec']>0 else float('nan')
    print(f"\nSpeedup (Zarr vs TIF) ≈ x{speedup:.2f}")

if __name__ == "__main__":
    main()
