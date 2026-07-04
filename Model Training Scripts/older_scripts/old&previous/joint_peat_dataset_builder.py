#!/usr/bin/env python3
"""
joint_peat_dataset.py

Joint ERA5-Land + SMAP → VIIRS multi-horizon dataset for PyTorch.

Assumes:
- You created three Zarr stores (probably via geotiff_to_zarr.py):
    era5land.zarr   (features)
    smap_wtd.zarr   (features)
    viirs.zarr      (labels; missing days → all-NaN slices if --allow-missing was used)

- Each Zarr store contains:
    "field":    (T, C, H, W)      # daily stack
    "peat_mask":(H, W) uint8      # 1=peat, 0=non-peat (optional but recommended)

This dataset:
- Builds t_hist-day history windows of ERA5+SMAP as inputs.
- Builds multi-horizon targets from VIIRS for user-specified horizons (e.g. 1,3,7,14 days ahead).
- Handles missing VIIRS days by:
    * Zarr store has NaN slices on those days;
    * Targets y use NaN→0 and a separate mask indicating where labels exist.

Output per sample:
    {
        "x":    FloatTensor, shape:
                    time_stack == "separate": (t_hist, C_total, patch, patch)
                    time_stack == "channel":  (t_hist*C_total, patch, patch)
        "y":    FloatTensor, shape (K, patch, patch)        # K=len(horizons)
        "mask": FloatTensor, shape (K, patch, patch)        # 1=label present, 0=missing
        "meta": dict, including per-sample IO timing:
                    io_time_era5, io_time_smap, io_time_viirs, io_time_total
        "coords": FloatTensor, shape (2, patch, patch) [OPTIONAL, if return_coords=True]
                  coords[0] = latitude (radians), coords[1] = longitude (radians)
    }

This file also includes a Dask cluster helper (start_dask_cluster) and CLI flags
for spinning up a LocalCluster with a dashboard, similar to geotiff_to_zarr.py.
The dataset itself does not require Dask; the cluster is just for your convenience.
"""

from __future__ import annotations
import os
import math
import json
import random
import time
from typing import Optional, Tuple, List, Literal, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import zarr

# ============================================================
# DASK PARALLELISM HELPERS (for GUI / dashboard)
# ============================================================

def start_dask_cluster(
    scheduler: str,
    workers: int,
    threads_per_worker: int,
    worker_mem: str,
    dash: bool,
):
    """
    Returns a context manager that sets up parallel execution.

    This mirrors the helper in geotiff_to_zarr.py:
    - If scheduler != 'distributed', we just set local Dask config (threads/processes).
    - If scheduler == 'distributed' and dask.distributed is available, we start
      a LocalCluster (optionally with dashboard) and return a Client context.

    Note: the joint dataset itself does not use Dask; you can use this for any
    Dask-based work you add, or just to have the dashboard around while training.
    """
    class _NoopCtx:
        def __enter__(self):
            import dask
            if scheduler == "threads":
                dask.config.set(scheduler="threads")
            elif scheduler == "processes":
                dask.config.set(scheduler="processes")
            else:
                dask.config.set(scheduler="threads")
            return None

        def __exit__(self, *exc):
            return False

    if scheduler != "distributed":
        return _NoopCtx()

    try:
        from dask.distributed import LocalCluster, Client
    except Exception:
        print("[warn] dask.distributed not available; falling back to local threads scheduler.")
        return _NoopCtx()

    class _DistCtx:
        def __enter__(self):
            processes = True
            dashboard = ":0" if dash else None
            self.cluster = LocalCluster(
                n_workers=int(workers),
                threads_per_worker=int(threads_per_worker),
                processes=processes,
                memory_limit=None if worker_mem == "auto" else worker_mem,
                dashboard_address=dashboard,
            )
            self.client = Client(self.cluster)
            try:
                link = getattr(self.client, "dashboard_link", None)
                if link:
                    print(f"[dask] Dashboard: {link}")
            except Exception:
                pass
            print(
                f"[dask] LocalCluster up: workers={workers}, threads/worker={threads_per_worker}, "
                f"processes={processes}, mem/worker={worker_mem}"
            )
            return self.client

        def __exit__(self, *exc):
            try:
                self.client.close()
            finally:
                self.cluster.close()
            print("[dask] Cluster closed.")
            return False

    return _DistCtx()


# ============================================================
# Helper: open main array from a Zarr store
# ============================================================

def _open_zarr_array(store_path: str, array_path: Optional[str] = None):
    """
    Open a Zarr array from a store and return (arr, path).

    If array_path is given:
        - first try to open that array via the root group (consolidated metadata),
        - if that fails, fall back to scanning the directory (handles arrays
          created after consolidation, or non-consolidated layouts).

    If array_path is None:
        - open as v2 group,
        - walk children to find the largest array and return it,
        - if that fails, fall back to scanning the directory.

    Compatible with folder Zarr stores written with zarr_format=2.
    """
    import zarr as _z
    import numpy as _np

    # 1) Try opening as a group (consolidated metadata, if present)
    root = None
    try:
        # Don't force zarr_format here; let zarr figure it out
        root = _z.open_group(store_path, mode="r")
    except Exception:
        root = None

    # ------------------------------------------------------------------
    # A) array_path explicitly given: try group lookup first
    # ------------------------------------------------------------------
    if root is not None and array_path is not None:
        try:
            arr = root[array_path]
            if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
                raise ValueError(
                    f"{array_path!r} is not a proper Zarr array in store {store_path!r}."
                )
            return arr, array_path
        except Exception:
            # e.g. consolidated metadata doesn't list this array, or layout is non-standard.
            # Fall back to directory-based scan below.
            root = None

    # ------------------------------------------------------------------
    # B) No explicit array_path: auto-detect largest array in the group
    # ------------------------------------------------------------------
    if root is not None and array_path is None:
        best = None
        best_path = None

        def walk(g, prefix: str = ""):
            nonlocal best, best_path

            # Arrays in this group
            names = []
            if hasattr(g, "array_keys"):
                try:
                    names = list(g.array_keys())
                except TypeError:
                    names = []
            if not names and hasattr(g, "keys"):
                for name in g.keys():
                    obj = g[name]
                    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                        names.append(name)

            for name in names:
                obj = g[name]
                if not hasattr(obj, "shape") or not hasattr(obj, "dtype"):
                    continue
                p = f"{prefix}/{name}" if prefix else name
                size = int(_np.prod(obj.shape, dtype=_np.int64))
                if best is None or size > best:
                    best = size
                    best_path = p

            # Subgroups
            grp_names = []
            if hasattr(g, "group_keys"):
                try:
                    grp_names = list(g.group_keys())
                except TypeError:
                    grp_names = []
            if not grp_names and hasattr(g, "keys"):
                for name in g.keys():
                    obj = g[name]
                    if hasattr(obj, "array_keys") or hasattr(obj, "group_keys"):
                        grp_names.append(name)

            for name in grp_names:
                sub = g[name]
                p2 = f"{prefix}/{name}" if prefix else name
                walk(sub, p2)

        walk(root)

        if best_path is None:
            # Couldn't find any arrays via group; fall back to directory scan.
            root = None
        else:
            return root[best_path], best_path

    # ------------------------------------------------------------------
    # C) Fallback: scan directory for .zarray / zarr.json and open with open_array
    # ------------------------------------------------------------------
    if not os.path.isdir(store_path):
        raise FileNotFoundError(f"Zarr directory not found: {store_path!r}")

    candidates: List[str] = []
    for dirpath, dirnames, filenames in os.walk(store_path):
        if ("zarr.json" in filenames) or (".zarray" in filenames):
            rel = os.path.relpath(dirpath, store_path)
            if rel == ".":
                rel = ""
            candidates.append(rel)

    if not candidates:
        raise ValueError(
            f"No Zarr arrays found under directory {store_path!r}. "
            f"Expected directories containing '.zarray' or 'zarr.json'."
        )

    if array_path is not None:
        # Ensure requested array exists among discovered candidates
        if array_path not in candidates:
            raise ValueError(
                f"Requested array_path {array_path!r} not found under {store_path!r}. "
                f"Discovered arrays: {candidates}"
            )
        candidates = [array_path]

    best_arr = None
    best_path = None
    best_size = -1

    for rel in candidates:
        try:
            arr = _z.open_array(store_path, path=rel, mode="r")
        except Exception:
            continue

        if not hasattr(arr, "shape"):
            continue

        size = int(_np.prod(arr.shape, dtype=_np.int64))
        if size > best_size:
            best_size = size
            best_arr = arr
            best_path = rel

    if best_arr is None:
        raise ValueError(
            f"Found candidate dirs under {store_path!r}, but none opened as Zarr array."
        )

    return best_arr, (best_path or "")


def _infer_layout(shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """
    Interpret array dims strictly as (T, C, H, W).

    This dataset assumes all main arrays are 4D:
        (T, C, H, W)

    If you accidentally point to a 3D array like (C, H, W),
    we raise an error instead of silently pretending there is a
    time dimension.
    """
    if len(shape) != 4:
        raise ValueError(f"Expected 4D array (T, C, H, W), got shape {shape}")
    T, C, H, W = shape
    return T, C, H, W



def _compute_grid_positions(H: int, W: int, patch: int, stride: Optional[int]) -> List[Tuple[int, int]]:
    if stride is None:
        stride = patch  # non-overlapping patches
    ys = list(range(0, max(H - patch + 1, 1), stride))
    xs = list(range(0, max(W - patch + 1, 1), stride))
    # Ensure coverage of far edge if patch doesn't tile evenly
    if ys and ys[-1] != H - patch:
        ys.append(max(H - patch, 0))
    if xs and xs[-1] != W - patch:
        xs.append(max(W - patch, 0))
    return [(y, x) for y in ys for x in xs]


# ============================================================
# Joint dataset
# ============================================================

class JointPeatDataset(Dataset):
    """
    Joint ERA5-Land + SMAP → VIIRS multi-horizon dataset (Zarr-backed).

    Inputs:
        - era5_zarr: path to ERA5 Zarr store
        - smap_zarr: path to SMAP Zarr store
        - viirs_zarr: path to VIIRS Zarr store (labels)
        - horizons: sequence of forecast horizons in days (e.g., [1,3,7,14])
        - t_hist: history length in days (e.g., 30)

    For each sample, returns:
        x:    (t_hist, C_total, patch, patch) or (t_hist*C_total, patch, patch)
        y:    (K, patch, patch)               where K=len(horizons)
        mask: (K, patch, patch)               1=label present, 0=missing
        meta: dict (includes IO timing fields)
    """

    def __init__(
        self,
        era5_zarr: str,
        smap_zarr: str,
        viirs_zarr: str,
        era5_array: Optional[str] = "field",
        smap_array: Optional[str] = "field",
        viirs_array: Optional[str] = "field",
        t_hist: int = 30,
        horizons: Sequence[int] = (1, 3, 7, 14),
        patch: int = 256,
        stride: Optional[int] = None,
        time_stack: Literal["separate", "channel"] = "separate",
        mode: Literal["train", "val"] = "train",
        split: float = 0.9,
        seed: int = 42,
        # normalization of *inputs* only (ERA5+SMAP)
        normalize_inputs: Optional[Literal["per_channel"]] = None,
        max_samples: Optional[int] = None,
        # use peat_mask (if present) to skip non-peat patches
        skip_nonpeat_patches: bool = True,
        peat_min_fraction: float = 0.01,   # require >=1% peat in a patch
        # optional external time_index (list[str] length T)
        time_index: Optional[List[str]] = None,
        # Whether to return lat/lon coords per patch
        return_coords: bool = False,
    ):
        super().__init__()

        self.t_hist = int(t_hist)
        self.horizons = list(int(h) for h in horizons)
        self.patch = int(patch)
        self.stride = None if stride is None else int(stride)
        self.time_stack = time_stack
        self.mode = mode
        self.split = float(split)
        self.seed = int(seed)
        self.normalize_inputs = normalize_inputs
        self.max_samples = max_samples
        self.skip_nonpeat_patches = bool(skip_nonpeat_patches)
        self.peat_min_fraction = float(peat_min_fraction)
        self.time_index = time_index
        self.return_coords = bool(return_coords)
        self.lat_grid: Optional[np.ndarray] = None
        self.lon_grid: Optional[np.ndarray] = None

        # ---------- Open Zarr arrays ----------
        self.era5_arr, self.era5_path = _open_zarr_array(era5_zarr, era5_array)
        self.smap_arr, self.smap_path = _open_zarr_array(smap_zarr, smap_array)
        self.viirs_arr, self.viirs_path = _open_zarr_array(viirs_zarr, viirs_array)

        T_e, C_e, H_e, W_e = _infer_layout(self.era5_arr.shape)
        T_s, C_s, H_s, W_s = _infer_layout(self.smap_arr.shape)
        T_v, C_v, H_v, W_v = _infer_layout(self.viirs_arr.shape)

        # Basic shape checks
        if not (T_e == T_s == T_v):
            raise ValueError(f"Time dimension mismatch: ERA5={T_e}, SMAP={T_s}, VIIRS={T_v}")
        if not (H_e == H_s == H_v and W_e == W_s == W_v):
            raise ValueError(
                f"Spatial mismatch: ERA5=({H_e},{W_e}), SMAP=({H_s},{W_s}), VIIRS=({H_v},{W_v})"
            )

        self.T = int(T_e)
        self.C_era5 = int(C_e)
        self.C_smap = int(C_s)
        self.C_total = self.C_era5 + self.C_smap
        self.H = int(H_e)
        self.W = int(W_e)
        self.C_viirs = int(C_v)

        # You stated VIIRS is 1 channel on purpose, so enforce that explicitly.
        if self.C_viirs != 1:
            raise ValueError(
                f"JointPeatDataset currently assumes a single VIIRS channel, "
                f"but viirs_arr has C={self.C_viirs}. If you add more channels "
                f"in the future, extend the label handling accordingly."
            )

        if self.t_hist < 1:
            raise ValueError("t_hist must be >= 1")
        if self.patch < 1 or self.patch > max(self.H, self.W):
            raise ValueError(f"Invalid patch={self.patch} for H={self.H}, W={self.W}")
        if self.time_index is not None and len(self.time_index) != self.T:
            raise ValueError("time_index length must match T")

        self.max_horizon = max(self.horizons) if self.horizons else 0

        # Warn if Zarr chunking is unfriendly (time/spatial chunks too small)
        self._warn_on_unfriendly_chunks(self.era5_arr, "ERA5")
        self._warn_on_unfriendly_chunks(self.smap_arr, "SMAP")
        self._warn_on_unfriendly_chunks(self.viirs_arr, "VIIRS")

        # ---------- Peat mask ----------
        self.peat_mask = None
        self._load_peat_mask(era5_zarr, smap_zarr, viirs_zarr)

        # ---------- Lat / Lon grids (optional) ----------
        if self.return_coords:
            self._load_lat_lon(era5_zarr, smap_zarr, viirs_zarr)

        # ---------- Build index of (t_end, y, x) ----------
        self._build_index()

        # ---------- Normalization stats (inputs only) ----------
        self._mean = None
        self._std = None
        if self.normalize_inputs == "per_channel":
            if len(self.index) == 0:
                raise RuntimeError("Cannot estimate normalization stats: dataset index is empty.")
            self._mean, self._std = self._estimate_input_stats(
                sample_count=min(256, len(self.index))
            )

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _warn_on_unfriendly_chunks(self, arr, name: str):
        """
        Check Zarr chunking and warn if time / spatial chunks are smaller than
        the history window or patch size. This doesn't change chunking, but
        helps you see when IO will be forced to touch many chunks per sample.
        """
        chunks = getattr(arr, "chunks", None)
        if chunks is None:
            return

        try:
            if len(chunks) == 4:
                t_chunk, c_chunk, h_chunk, w_chunk = chunks
            elif len(chunks) == 3:
                # (C, H, W) style -> treat as T=1
                c_chunk, h_chunk, w_chunk = chunks
                t_chunk = 1
            else:
                return

            msgs = []
            if t_chunk < self.t_hist:
                msgs.append(f"time chunk={t_chunk} < t_hist={self.t_hist}")
            if h_chunk < self.patch or w_chunk < self.patch:
                msgs.append(f"spatial chunk=({h_chunk},{w_chunk}) < patch={self.patch}")

            if msgs:
                print(
                    f"[warn] {name} Zarr chunks may be suboptimal for this dataset:\n"
                    f"       chunks={chunks}, t_hist={self.t_hist}, patch={self.patch}\n"
                    f"       {'; '.join(msgs)}\n"
                    f"       Consider re-chunking to (T≈{self.t_hist}, C, H≥{self.patch}, W≥{self.patch})."
                )
        except Exception:
            # Best-effort only; never crash here.
            return

    def _load_peat_mask(self, era5_zarr: str, smap_zarr: str, viirs_zarr: str):
        """
        Try to load peat_mask from one of the stores. We only keep it in memory once.
        Expected shape: (H, W) uint8.

        Uses the same robust _open_zarr_array helper so it works with your Zarr layout.
        """
        def try_open_mask(store_path: str):
            try:
                # Reuse the robust scanner, but force array_path="peat_mask"
                arr, arr_path = _open_zarr_array(store_path, "peat_mask")
                mask = np.asarray(arr, dtype=np.uint8)
                return mask
            except Exception:
                return None

        mask = try_open_mask(smap_zarr)
        if mask is not None:
            print("peat_mask loaded from SMAP store.")
        if mask is None:
            mask = try_open_mask(era5_zarr)
            print("WRONG_MASK (ERA5LAND) BEING USED")
        if mask is None:
            mask = try_open_mask(viirs_zarr)
            print("WRONG_MASK (VIIRS) BEING USED")

        if mask is None:
            msg = (
                "skip_nonpeat_patches=True but no 'peat_mask' array was found in any Zarr store.\n"
                "Available arrays should include 'peat_mask' (H,W). "
                "Either add peat_mask to one of the stores or rerun with --no-skip-nonpeat "
                "if you truly want to use all patches."
            )
            if self.skip_nonpeat_patches:
                # For your use-case, invalid non-peat data => hard error
                raise RuntimeError(msg)
            else:
                print("[warn]", msg)
                self.peat_mask = None
                return

        if mask.shape != (self.H, self.W):
            raise ValueError(
                f"peat_mask shape {mask.shape} does not match (H,W)=({self.H},{self.W})"
            )

        self.peat_mask = mask

    def _normalize_coord_array(self, arr: np.ndarray, name: str) -> np.ndarray:
        """
        Convert a lat/lon array from various possible shapes to (H, W).

        Supported shapes:
          - (H, W)
          - (H,)  for lat  -> broadcast over W
          - (W,)  for lon  -> broadcast over H
          - (1, H, W) or (H, W, 1)
          - (T, C, H, W) -> take [0,0,...] as static grid

        Raises if it cannot be converted.
        """
        arr = np.asarray(arr)

        if arr.ndim == 2:
            # (H, W)
            if arr.shape != (self.H, self.W):
                raise ValueError(
                    f"{name} array has shape {arr.shape}, expected ({self.H},{self.W})"
                )
            return arr

        if arr.ndim == 1:
            # 1D lat or lon
            if name == "lat" and arr.shape[0] == self.H:
                return np.repeat(arr[:, None], self.W, axis=1)
            if name == "lon" and arr.shape[0] == self.W:
                return np.repeat(arr[None, :], self.H, axis=0)
            raise ValueError(
                f"Cannot broadcast 1D {name} array with shape {arr.shape} "
                f"to ({self.H},{self.W})"
            )

        if arr.ndim == 3:
            # (1, H, W) or (H, W, 1)
            if arr.shape[0] == 1 and arr.shape[1:] == (self.H, self.W):
                return arr[0]
            if arr.shape[-1] == 1 and arr.shape[0:2] == (self.H, self.W):
                return arr[..., 0]

        if arr.ndim == 4 and arr.shape[2:] == (self.H, self.W):
            # (T, C, H, W) with static grid across time/channels
            return arr[0, 0]

        raise ValueError(
            f"Unsupported shape for {name} array: {arr.shape} "
            f"(cannot map to ({self.H},{self.W}))"
        )

    def _load_lat_lon(self, era5_zarr: str, smap_zarr: str, viirs_zarr: str):
        """
        Load latitude / longitude grids from one of the Zarr stores.

        We look for arrays named 'lat'/'lon' or 'latitude'/'longitude'
        in ERA5, then SMAP, then VIIRS. Shapes can be:
          - (H,W)
          - (H,) or (W,)
          - (1,H,W) or (H,W,1)
          - (T,C,H,W) with static coords (we take [0,0,...]).

        Values are assumed to be in degrees and converted later to radians
        in __getitem__.
        """
        def try_open(store_path: str, names):
            for nm in names:
                try:
                    arr, arr_path = _open_zarr_array(store_path, nm)
                    return np.asarray(arr)
                except Exception:
                    continue
            return None

        lat = None
        lon = None

        for root in (era5_zarr, smap_zarr, viirs_zarr):
            if lat is None:
                lat = try_open(root, ["lat", "latitude"])
            if lon is None:
                lon = try_open(root, ["lon", "longitude"])
            if lat is not None and lon is not None:
                break

        if lat is None or lon is None:
            raise RuntimeError(
                "return_coords=True but could not find latitude/longitude arrays "
                "named 'lat'/'lon' or 'latitude'/'longitude' in any of the Zarr "
                "stores. Please add static lat/lon arrays to one store or "
                "modify _load_lat_lon to match your layout."
            )

        lat = self._normalize_coord_array(lat, "lat")
        lon = self._normalize_coord_array(lon, "lon")

        # Store as float32 degrees; convert to radians in __getitem__
        self.lat_grid = lat.astype(np.float32)
        self.lon_grid = lon.astype(np.float32)

        print(
            f"[coords] Loaded lat/lon grids for ({self.H},{self.W}) from Zarr stores, "
            f"dtype={self.lat_grid.dtype}"
        )

    def _build_index(self):
        """
        Build the full index of (t_end, y0, x0) for this dataset.
        """
        # Time range for t_end (last index in history window)
        t_end_first = self.t_hist - 1
        t_end_last = self.T - 1 - self.max_horizon  # must fit all horizons

        if t_end_last < t_end_first:
            raise ValueError(
                f"Not enough time steps T={self.T} for t_hist={self.t_hist} "
                f"and max_horizon={self.max_horizon}"
            )

        t_ends = list(range(t_end_first, t_end_last + 1))

        # Spatial patch grid
        spatial = _compute_grid_positions(self.H, self.W, self.patch, self.stride)

        if self.skip_nonpeat_patches and self.peat_mask is not None:
            valid_spatial: List[Tuple[int, int]] = []
            area = self.patch * self.patch
            for (y0, x0) in spatial:
                patch_mask = self.peat_mask[y0:y0 + self.patch, x0:x0 + self.patch]
                frac_peat = float(patch_mask.sum()) / max(area, 1)
                if frac_peat >= self.peat_min_fraction:
                    valid_spatial.append((y0, x0))
            spatial = valid_spatial
            if not spatial:
                raise RuntimeError(
                    "After peat_mask-based filtering, no valid spatial patches remain. "
                    "Consider lowering peat_min_fraction or disabling skip_nonpeat_patches."
                )

        full_index = [(t_end, y0, x0) for t_end in t_ends for (y0, x0) in spatial]
        if not full_index:
            raise RuntimeError("No valid sampling positions derived.")

        rng = random.Random(self.seed)
        idx_shuf = full_index[:]
        rng.shuffle(idx_shuf)
        split_at = int(len(idx_shuf) * self.split)
        train_idx = idx_shuf[:split_at]
        val_idx = idx_shuf[split_at:]

        self.index: List[Tuple[int, int, int]] = train_idx if self.mode == "train" else val_idx
        if self.max_samples is not None:
            self.index = self.index[: self.max_samples]

    # --------------------------------------------------------
    # Dataset API
    # --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        t_end, y0, x0 = self.index[i]
        t0 = t_end - (self.t_hist - 1)

        # ---------- Inputs (ERA5 + SMAP) ----------
        # Measure Zarr read + decompression time for ERA5 / SMAP / VIIRS.
        io_time_era5 = 0.0
        io_time_smap = 0.0
        io_time_viirs = 0.0

        # ERA5
        t_io = time.time()
        era5_slab = self.era5_arr.get_orthogonal_selection(
            (
                slice(t0, t_end + 1),
                slice(None),
                slice(y0, y0 + self.patch),
                slice(x0, x0 + self.patch),
            )
        )  # (t_hist, C_era5, P, P)
        io_time_era5 = time.time() - t_io

        # SMAP
        t_io = time.time()
        smap_slab = self.smap_arr.get_orthogonal_selection(
            (
                slice(t0, t_end + 1),
                slice(None),
                slice(y0, y0 + self.patch),
                slice(x0, x0 + self.patch),
            )
        )  # (t_hist, C_smap, P, P)
        io_time_smap = time.time() - t_io

        era5_np = np.asarray(era5_slab, dtype=np.float32)
        smap_np = np.asarray(smap_slab, dtype=np.float32)

        era5_np = np.nan_to_num(era5_np, nan=0.0, posinf=0.0, neginf=0.0)
        smap_np = np.nan_to_num(smap_np, nan=0.0, posinf=0.0, neginf=0.0)

        x_np = np.concatenate([era5_np, smap_np], axis=1)  # (t_hist, C_total, P, P)
        x = torch.from_numpy(x_np)  # float32

        if torch.isnan(x).any():
            raise RuntimeError("NaN detected in input x after nan_to_num; check preprocessing.")

        if self.time_stack == "channel":
            # (t_hist, C_total, H, W) -> (t_hist * C_total, H, W)
            x = x.permute(1, 0, 2, 3).reshape(self.C_total * self.t_hist, self.patch, self.patch)

        # ---------- Targets (VIIRS, 1 channel) ----------
        K = len(self.horizons)
        y_np = np.zeros((K, self.patch, self.patch), dtype=np.float32)
        mask_np = np.zeros((K, self.patch, self.patch), dtype=np.float32)

        # Slice the peat_mask for this spatial patch (if available)
        peat_patch = None
        if self.peat_mask is not None:
            peat_patch = (self.peat_mask[y0:y0 + self.patch,
                                         x0:x0 + self.patch] > 0)  # bool (P, P)

        for k, h in enumerate(self.horizons):
            t_label = t_end + h
            if t_label < 0 or t_label >= self.T:
                # Outside calendar → keep y=0, mask=0
                continue

            # Measure IO/decompress for VIIRS label slice
            t_io = time.time()
            slab = self.viirs_arr.get_orthogonal_selection(
                (
                    slice(t_label, t_label + 1),
                    slice(0, 1),  # single VIIRS channel
                    slice(y0, y0 + self.patch),
                    slice(x0, x0 + self.patch),
                )
            )  # (1,1,P,P)
            io_time_viirs += time.time() - t_io

            lbl = np.asarray(slab[0, 0], dtype=np.float32)  # (P,P)

            # Valid where VIIRS has a finite value
            valid = np.isfinite(lbl)

            # If a peat mask is available, only keep peat pixels
            if peat_patch is not None:
                valid = valid & peat_patch  # both boolean (P, P)

            mask_np[k][valid] = 1.0
            safe_lbl = np.where(valid, lbl, 0.0)
            y_np[k] = safe_lbl

        y = torch.from_numpy(y_np)
        mask = torch.from_numpy(mask_np)

        # ---------- Normalize inputs (optional) ----------
        if self.normalize_inputs == "per_channel":
            if self._mean is None or self._std is None:
                raise RuntimeError(
                    "Requested per_channel normalization, but mean/std have not been set."
                )

            if self.time_stack == "separate":
                # mean/std shape: (C_total,)
                mean = self._mean.view(1, -1, 1, 1)
                std = self._std.view(1, -1, 1, 1)
            else:
                mean = self._mean.repeat(self.t_hist).view(-1, 1, 1)
                std = self._std.repeat(self.t_hist).view(-1, 1, 1)

            std = torch.where(std == 0, torch.ones_like(std), std)
            x = (x - mean) / std

        # --- OPTIONAL: debug checks for NaNs/Infs in dataset output ---
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError(
                f"Non-finite values in x from dataset: i={i}, "
                f"t0={t0}, t_end={t_end}, y0={y0}, x0={x0}"
            )
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise RuntimeError(
                f"Non-finite values in y from dataset: i={i}, "
                f"t0={t0}, t_end={t_end}, y0={y0}, x0={x0}"
            )
        if torch.isnan(mask).any() or torch.isinf(mask).any():
            raise RuntimeError(
                f"Non-finite values in mask from dataset: i={i}, "
                f"t0={t0}, t_end={t_end}, y0={y0}, x0={x0}"
            )

        # ---------- Coordinates (lat/lon in radians, optional) ----------
        coords = None
        if self.return_coords:
            if self.lat_grid is None or self.lon_grid is None:
                raise RuntimeError(
                    "return_coords=True but lat_grid / lon_grid are not loaded. "
                    "Check _load_lat_lon."
                )
            lat_patch = self.lat_grid[y0 : y0 + self.patch, x0 : x0 + self.patch]
            lon_patch = self.lon_grid[y0 : y0 + self.patch, x0 : x0 + self.patch]

            # Convert degrees -> radians
            lat_rad = np.deg2rad(lat_patch.astype(np.float32))
            lon_rad = np.deg2rad(lon_patch.astype(np.float32))

            coords_np = np.stack([lat_rad, lon_rad], axis=0)  # (2, P, P)
            coords = torch.from_numpy(coords_np)

        meta = {
            "t_end": t_end,
            "t_start": t0,
            "y0": y0,
            "x0": x0,
            "horizons": self.horizons,
            "era5_array_path": self.era5_path,
            "smap_array_path": self.smap_path,
            "viirs_array_path": self.viirs_path,
            # NEW: per-sample IO / decompression timing (seconds)
            "io_time_era5": float(io_time_era5),
            "io_time_smap": float(io_time_smap),
            "io_time_viirs": float(io_time_viirs),
            "io_time_total": float(io_time_era5 + io_time_smap + io_time_viirs),
        }
        if self.time_index is not None:
            meta["history_timestamps"] = self.time_index[t0 : t_end + 1]

        sample = {"x": x, "y": y, "mask": mask, "meta": meta}
        if self.return_coords:
            sample["coords"] = coords

        return sample
    # --------------------------------------------------------
    # Normalization helpers
    # --------------------------------------------------------

    def _estimate_input_stats(self, sample_count: int = 256):
        """
        Estimate per-channel mean/std of ERA5+SMAP inputs from a subset of samples.

        This uses float32 for the raw data and float64 for accumulation to keep
        memory use modest while retaining good numeric stability.
        """
        if len(self.index) == 0:
            raise RuntimeError("Cannot estimate input stats with an empty index.")

        rng = random.Random(self.seed + 123)
        n_samples = min(sample_count, len(self.index))
        # unique random indices
        idxs = rng.sample(range(len(self.index)), n_samples)

        sum_c = torch.zeros(self.C_total, dtype=torch.float64)
        sqsum_c = torch.zeros(self.C_total, dtype=torch.float64)
        count = 0

        for j in idxs:
            t_end, y0, x0 = self.index[j]
            t0 = t_end - (self.t_hist - 1)

            era5_slab = self.era5_arr.get_orthogonal_selection(
                (
                    slice(t0, t_end + 1),
                    slice(None),
                    slice(y0, y0 + self.patch),
                    slice(x0, x0 + self.patch),
                )
            )
            smap_slab = self.smap_arr.get_orthogonal_selection(
                (
                    slice(t0, t_end + 1),
                    slice(None),
                    slice(y0, y0 + self.patch),
                    slice(x0, x0 + self.patch),
                )
            )

            # Use float32 for the raw arrays; convert to float64 only for accumulation.
            era5_np = np.asarray(era5_slab, dtype=np.float32)
            smap_np = np.asarray(smap_slab, dtype=np.float32)

            # Clean NaNs/Infs before computing stats
            era5_np = np.nan_to_num(era5_np, nan=0.0, posinf=0.0, neginf=0.0)
            smap_np = np.nan_to_num(smap_np, nan=0.0, posinf=0.0, neginf=0.0)

            x_np = np.concatenate([era5_np, smap_np], axis=1)  # (t_hist, C_total, P, P)
            x_torch = torch.from_numpy(x_np).float()  # (t_hist, C_total, P, P)

            # Move channels to front: (C_total, t_hist, P, P)
            x_torch = x_torch.permute(1, 0, 2, 3).contiguous()

            x64 = x_torch.to(torch.float64)
            sum_c += x64.sum(dim=(1, 2, 3))
            sqsum_c += (x64 * x64).sum(dim=(1, 2, 3))
            count += x_torch.shape[1] * x_torch.shape[2] * x_torch.shape[3]

        mean = (sum_c / count).to(torch.float32)
        var = (sqsum_c / count) - (mean.double() ** 2)
        var = torch.clamp(var, min=0.0).to(torch.float32)
        std = torch.sqrt(var)
        return mean, std

    def get_normalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (mean, std) used for input normalization.

        Both tensors have shape (C_total,) and dtype float32.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("Normalization stats are not set for this dataset.")
        return self._mean.clone(), self._std.clone()

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Set per-channel input normalization stats (C_total,) and enable 'per_channel' normalization.

        This is useful when you want to:
          - compute stats on the train set, then
          - reuse them for val/test sets without re-estimating.
        """
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)

        if mean.shape != (self.C_total,) or std.shape != (self.C_total,):
            raise ValueError(
                f"mean/std must have shape ({self.C_total},), "
                f"got mean.shape={mean.shape}, std.shape={std.shape}"
            )

        self._mean = mean
        self._std = std
        self.normalize_inputs = "per_channel"


# ============================================================
# Example usage / quick bench (with optional Dask GUI)
# ============================================================

def _example():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Joint ERA5+SMAP→VIIRS Zarr dataset (with optional Dask GUI)")
    parser.add_argument("--era5-zarr", required=True)
    parser.add_argument("--smap-zarr", required=True)
    parser.add_argument("--viirs-zarr", required=True)
    parser.add_argument("--era5-array", default="field")
    parser.add_argument("--smap-array", default="field")
    parser.add_argument("--viirs-array", default="field")
    parser.add_argument("--t-hist", type=int, default=30)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 7, 14])
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stack-time", choices=["separate", "channel"], default="separate")
    parser.add_argument("--mode", choices=["train", "val"], default="train")
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--normalize-inputs", choices=[None, "per_channel"], default=None)
    parser.add_argument("--no-skip-nonpeat", action="store_true")
    parser.add_argument("--return-coords", action="store_true", help="Return lat/lon coords (radians) for SH+SIREN models")

    # --- DASK / PARALLELISM (for dashboard / extra work) ---
    parser.add_argument(
        "--scheduler",
        choices=["distributed", "processes", "threads"],
        default="distributed",
        help="Parallel execution backend (for Dask GUI / any Dask work you add)",
    )
    parser.add_argument(
        "--dask-workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) // 4),
        help="Number of Dask workers (processes for 'distributed')",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=2,
        help="Threads per Dask worker",
    )
    parser.add_argument(
        "--worker-mem",
        type=str,
        default="auto",
        help="Memory limit per Dask worker (e.g. 8GB) or 'auto'",
    )
    parser.add_argument(
        "--dask-dashboard",
        action="store_true",
        help="Enable Dask dashboard if using 'distributed'",
    )

    args = parser.parse_args()

    # Start Dask cluster context (mirrors geotiff_to_zarr behavior)
    with start_dask_cluster(
        scheduler=args.scheduler,
        workers=args.dask_workers,
        threads_per_worker=args.threads_per_worker,
        worker_mem=args.worker_mem,
        dash=args.dask_dashboard,
    ):
        ds = JointPeatDataset(
            era5_zarr=args.era5_zarr,
            smap_zarr=args.smap_zarr,
            viirs_zarr=args.viirs_zarr,
            era5_array=args.era5_array,
            smap_array=args.smap_array,
            viirs_array=args.viirs_array,
            t_hist=args.t_hist,
            horizons=args.horizons,
            patch=args.patch,
            stride=args.stride,
            time_stack=args.stack_time,
            mode=args.mode,
            split=args.split,
            seed=42,
            normalize_inputs=args.normalize_inputs,
            max_samples=args.max_samples,
            skip_nonpeat_patches=not args.no_skip_nonpeat,
            peat_min_fraction=0.01,
            time_index=None,
            return_coords=args.return_coords
        )

        print(f"T={ds.T}, C_era5={ds.C_era5}, C_smap={ds.C_smap}, C_total={ds.C_total}")
        print(f"H={ds.H}, W={ds.W}, horizons={ds.horizons}")
        print(f"Dataset length ({args.mode}): {len(ds)} samples")

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(args.mode == "train"),
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.workers > 0,
            drop_last=False,
        )

        # warmup
        it = iter(loader)
        n_warm = min(3, len(loader))
        for _ in range(n_warm):
            batch = next(it)
            _ = batch["x"], batch["y"], batch["mask"], batch["meta"]

        # small throughput test
        t0 = time.time()
        n_batches = min(20, len(loader))
        n_samples = 0
        for i, batch in enumerate(loader):
            x = batch["x"]
            y = batch["y"]
            mask = batch["mask"]
            meta = batch["meta"]
            n_samples += x.shape[0]
            if i + 1 >= n_batches:
                break
        dt = time.time() - t0
        if dt > 0:
            print(f"Throughput: {n_samples/dt:.1f} samples/sec over {n_batches} batches")


if __name__ == "__main__":
    _example()
