#!/usr/bin/env python3
"""
joint_peat_dataset_builder.py

Joint ERA5-Land + SMAP (+ optional EXTRA) → VIIRS multi-horizon dataset for PyTorch.

Key features:
- Zarr-backed, robust to slightly messy layouts via _open_zarr_array.
- Handles NaNs/Infs in inputs (nan_to_num -> 0).
- Masks out invalid VIIRS labels (NaNs/Infs/out-of-range) and peat-only label regions.
- Optional per-channel normalization (estimated from random samples).
- Optional lat/lon coordinates per patch, in radians, for SH+SIREN models.
- Optional coord_as_features: appends sin/cos(lat, lon) as 4 extra channels,
  repeated across time steps.

Coordinates loader supports x/y fallback:
    * First tries lat/lon (or latitude/longitude) arrays.
    * If not found, tries y/x axes:
        - 1D y (H,) and x (W,) → meshgrid to (H, W).
        - 2D y (H,W) and x (H,W) → use directly.
    * Assumes y = latitude (deg), x = longitude (deg).

Splits:
- If val_frac is None → 2-way split (train / val).
- If val_frac is not None → 3-way split (train / val / test):
    - train_frac = split
    - val_frac   = val_frac
    - test_frac  = 1 - split - val_frac

NEW (this version):
- Optional extra_zarr (+ extra_array) for additional input bands with same layout (T,C,H,W).
  If provided, it's concatenated onto inputs alongside ERA5+SMAP.
"""

from __future__ import annotations
import os
import random
import time
from typing import Optional, Tuple, List, Literal, Sequence

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
            try:
                import dask
                if scheduler == "threads":
                    dask.config.set(scheduler="threads")
                elif scheduler == "processes":
                    dask.config.set(scheduler="processes")
                else:
                    pass
            except ModuleNotFoundError:
                pass
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
        - if that fails, fall back to scanning the directory.

    If array_path is None:
        - open as group,
        - walk children to find the largest array and return it,
        - if that fails, fall back to scanning the directory.

    Compatible with folder Zarr stores written with zarr_format=2.
    """
    import zarr as _z
    import numpy as _np

    # 1) Try opening as a group
    root = None
    try:
        root = _z.open_group(store_path, mode="r")
    except Exception:
        root = None

    # A) array_path explicitly given
    if root is not None and array_path is not None:
        try:
            arr = root[array_path]
            if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
                raise ValueError(
                    f"{array_path!r} is not a proper Zarr array in store {store_path!r}."
                )
            return arr, array_path
        except Exception:
            root = None

    # B) Auto-detect largest array in the group
    if root is not None and array_path is None:
        best = None
        best_path = None

        def walk(g, prefix: str = ""):
            nonlocal best, best_path

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
            root = None
        else:
            return root[best_path], best_path

    # C) Fallback: scan directory for .zarray / zarr.json
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
    """Interpret array dims strictly as (T, C, H, W)."""
    if len(shape) != 4:
        raise ValueError(f"Expected 4D array (T, C, H, W), got shape {shape}")
    T, C, H, W = shape
    return T, C, H, W


def _compute_grid_positions(H: int, W: int, patch: int, stride: Optional[int]) -> List[Tuple[int, int]]:
    if stride is None:
        stride = patch
    ys = list(range(0, max(H - patch + 1, 1), stride))
    xs = list(range(0, max(W - patch + 1, 1), stride))
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
    Joint ERA5-Land + SMAP (+ optional EXTRA) → VIIRS multi-horizon dataset (Zarr-backed).

    Inputs:
        - era5_zarr: path to ERA5 Zarr store
        - smap_zarr: path to SMAP Zarr store
        - extra_zarr: optional path to EXTRA Zarr store (additional bands), same layout (T,C,H,W)
        - viirs_zarr: path to VIIRS Zarr store (labels)
        - horizons: sequence of forecast horizons in days (e.g., [1,3,7,14])
        - t_hist: history length in days (e.g., 30)

    Per sample:
        "x": (t_hist, C_total, patch, patch) or (t_hist*C_total, patch, patch)
        "y": (K, patch, patch)               (K = len(horizons))
        "mask": same shape as y, 1=label present, 0=missing
        "coords": (2, patch, patch) [lat_rad, lon_rad] if return_coords=True
        "meta": dict with IO times and indices
    """

    def __init__(
        self,
        era5_zarr: str,
        smap_zarr: str,
        viirs_zarr: str,
        era5_array: Optional[str] = "field",
        smap_array: Optional[str] = "field",
        viirs_array: Optional[str] = "field",
        # NEW:
        extra_zarr: Optional[str] = None,
        extra_array: Optional[str] = "field",
        t_hist: int = 30,
        horizons: Sequence[int] = (1, 3, 7, 14),
        patch: int = 256,
        stride: Optional[int] = None,
        time_stack: Literal["separate", "channel"] = "separate",
        mode: Literal["train", "val", "test"] = "train",
        split: float = 0.9,
        val_frac: Optional[float] = None,
        seed: int = 42,
        normalize_inputs: Optional[Literal["per_channel"]] = None,
        max_samples: Optional[int] = None,
        skip_nonpeat_patches: bool = True,
        peat_min_fraction: float = 0.01,
        time_index: Optional[List[str]] = None,
        return_coords: bool = False,
        coord_as_features: bool = False,
        # If you want to enforce "exactly 4 bands", keep this True.
        enforce_extra_c4: bool = True,
    ):
        super().__init__()

        self.t_hist = int(t_hist)
        self.horizons = list(int(h) for h in horizons)
        self.patch = int(patch)
        self.stride = None if stride is None else int(stride)
        self.time_stack = time_stack
        self.mode = mode
        self.split = float(split)
        self.val_frac = None if val_frac is None else float(val_frac)
        self.seed = int(seed)
        self.normalize_inputs = normalize_inputs
        self.max_samples = max_samples
        self.skip_nonpeat_patches = bool(skip_nonpeat_patches)
        self.peat_min_fraction = float(peat_min_fraction)
        self.time_index = time_index
        self.return_coords = bool(return_coords)
        self.coord_as_features = bool(coord_as_features)

        # NEW:
        self.extra_zarr = extra_zarr
        self.extra_array = extra_array
        self.enforce_extra_c4 = bool(enforce_extra_c4)

        self.lat_grid: Optional[np.ndarray] = None
        self.lon_grid: Optional[np.ndarray] = None

        if self.mode not in ("train", "val", "test"):
            raise ValueError(f"mode must be 'train', 'val', or 'test', got {self.mode}")

        # ---------- Open Zarr arrays ----------
        self.era5_arr, self.era5_path = _open_zarr_array(era5_zarr, era5_array)
        self.smap_arr, self.smap_path = _open_zarr_array(smap_zarr, smap_array)
        self.viirs_arr, self.viirs_path = _open_zarr_array(viirs_zarr, viirs_array)

        T_e, C_e, H_e, W_e = _infer_layout(self.era5_arr.shape)
        T_s, C_s, H_s, W_s = _infer_layout(self.smap_arr.shape)
        T_v, C_v, H_v, W_v = _infer_layout(self.viirs_arr.shape)

        if not (T_e == T_s == T_v):
            raise ValueError(f"Time dimension mismatch: ERA5={T_e}, SMAP={T_s}, VIIRS={T_v}")
        if not (H_e == H_s == H_v and W_e == W_s == W_v):
            raise ValueError(
                f"Spatial mismatch: ERA5=({H_e},{W_e}), SMAP=({H_s},{W_s}), VIIRS=({H_v},{W_v})"
            )

        self.T = int(T_e)
        self.C_era5 = int(C_e)
        self.C_smap = int(C_s)
        self.H = int(H_e)
        self.W = int(W_e)
        self.C_viirs = int(C_v)

        if self.C_viirs != 1:
            raise ValueError(
                f"JointPeatDataset currently assumes a single VIIRS channel, "
                f"but viirs_arr has C={self.C_viirs}."
            )

        # ---------- Optional EXTRA inputs ----------
        self.extra_arr = None
        self.extra_path = None
        self.C_extra = 0

        if extra_zarr is not None:
            self.extra_arr, self.extra_path = _open_zarr_array(extra_zarr, extra_array)
            T_x, C_x, H_x, W_x = _infer_layout(self.extra_arr.shape)

            if T_x != self.T:
                raise ValueError(f"Time dimension mismatch: EXTRA={T_x}, expected T={self.T}")
            if (H_x, W_x) != (self.H, self.W):
                raise ValueError(
                    f"Spatial mismatch: EXTRA=({H_x},{W_x}), expected ({self.H},{self.W})"
                )

            self.C_extra = int(C_x)
            if self.enforce_extra_c4 and self.C_extra != 4:
                raise ValueError(f"Expected EXTRA C=4 (4 bands), got C={self.C_extra}")

        if self.t_hist < 1:
            raise ValueError("t_hist must be >= 1")
        if self.patch < 1 or self.patch > self.H or self.patch > self.W:
            raise ValueError(
                f"Invalid patch={self.patch} for H={self.H}, W={self.W} "
                f"(patch must be <= both H and W)."
            )
        if self.time_index is not None and len(self.time_index) != self.T:
            raise ValueError("time_index length must match T")

        # train / val / test fractions sanity for 3-way split
        if self.val_frac is not None:
            if not (0.0 < self.split < 1.0):
                raise ValueError(f"split (train fraction) must be in (0,1), got {self.split}")
            if not (0.0 <= self.val_frac < 1.0):
                raise ValueError(f"val_frac must be in [0,1), got {self.val_frac}")
            if self.split + self.val_frac >= 1.0:
                raise ValueError(
                    f"split + val_frac must be < 1.0 (train+val); got {self.split + self.val_frac}"
                )

        # total per-time-step channels including optional coord features
        self.C_base = self.C_era5 + self.C_smap + self.C_extra
        self.C_total = self.C_base + (4 if self.coord_as_features else 0)

        self.max_horizon = max(self.horizons) if self.horizons else 0

        self._warn_on_unfriendly_chunks(self.era5_arr, "ERA5")
        self._warn_on_unfriendly_chunks(self.smap_arr, "SMAP")
        self._warn_on_unfriendly_chunks(self.viirs_arr, "VIIRS")
        if self.extra_arr is not None:
            self._warn_on_unfriendly_chunks(self.extra_arr, "EXTRA")

        # ---------- Peat mask ----------
        self.peat_mask = None
        self._load_peat_mask(era5_zarr, smap_zarr, viirs_zarr, extra_zarr=self.extra_zarr)

        # ---------- Lat / Lon grids (with lat/lon OR x/y fallback) ----------
        if self.return_coords or self.coord_as_features:
            self._load_lat_lon_with_xy_fallback(era5_zarr, smap_zarr, viirs_zarr, extra_zarr=self.extra_zarr)

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
        chunks = getattr(arr, "chunks", None)
        if chunks is None:
            return

        try:
            if len(chunks) == 4:
                t_chunk, c_chunk, h_chunk, w_chunk = chunks
            elif len(chunks) == 3:
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
            return

    def _load_peat_mask(self, era5_zarr: str, smap_zarr: str, viirs_zarr: str, extra_zarr: Optional[str] = None):
        """Try to load peat_mask from one of the stores. Expected shape: (H, W) uint8."""
        def try_open_mask(store_path: str):
            try:
                arr, _ = _open_zarr_array(store_path, "peat_mask")
                mask = np.asarray(arr, dtype=np.uint8)
                return mask
            except Exception:
                return None

        stores = [smap_zarr, era5_zarr, viirs_zarr]
        if extra_zarr is not None:
            stores = [extra_zarr] + stores

        mask = None
        for sp in stores:
            mask = try_open_mask(sp)
            if mask is not None:
                print(f"peat_mask loaded from store: {sp}")
                break

        if mask is None:
            msg = (
                "skip_nonpeat_patches=True but no 'peat_mask' array was found in any Zarr store.\n"
                "Either add peat_mask to one of the stores or rerun with --no-skip-nonpeat "
                "if you truly want to use all patches."
            )
            if self.skip_nonpeat_patches:
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
        Convert a lat/lon-like array from various possible shapes to (H, W).

        Supported shapes:
          - (H, W)
          - (H,)  for lat  -> broadcast over W
          - (W,)  for lon  -> broadcast over H
          - (1, H, W) or (H, W, 1)
          - (T, C, H, W)   -> take [0,0,...] as static grid
        """
        arr = np.asarray(arr)

        if arr.ndim == 2:
            if arr.shape != (self.H, self.W):
                raise ValueError(
                    f"{name} array has shape {arr.shape}, expected ({self.H},{self.W})"
                )
            return arr

        if arr.ndim == 1:
            if name == "lat" and arr.shape[0] == self.H:
                return np.repeat(arr[:, None], self.W, axis=1)
            if name == "lon" and arr.shape[0] == self.W:
                return np.repeat(arr[None, :], self.H, axis=0)
            raise ValueError(
                f"Cannot broadcast 1D {name} array with shape {arr.shape} "
                f"to ({self.H},{self.W})"
            )

        if arr.ndim == 3:
            if arr.shape[0] == 1 and arr.shape[1:] == (self.H, self.W):
                return arr[0]
            if arr.shape[-1] == 1 and arr.shape[0:2] == (self.H, self.W):
                return arr[..., 0]

        if arr.ndim == 4 and arr.shape[2:] == (self.H, self.W):
            return arr[0, 0]

        raise ValueError(
            f"Unsupported shape for {name} array: {arr.shape} "
            f"(cannot map to ({self.H},{self.W}))"
        )

    def _load_lat_lon_with_xy_fallback(self, era5_zarr: str, smap_zarr: str, viirs_zarr: str, extra_zarr: Optional[str] = None):
        """
        Load latitude / longitude grids with fallback:

        1. Try 'lat'/'lon' or 'latitude'/'longitude' arrays from any store.
        2. If not found, try 'y' and 'x' arrays:
             - If 1D: y(H,) + x(W,) -> meshgrid via _normalize_coord_array
             - If 2D: use directly
        3. Assume units are degrees; convert to radians in __getitem__ / coord features.
        """
        def try_open(store_path: str, names):
            for nm in names:
                try:
                    arr, _ = _open_zarr_array(store_path, nm)
                    return np.asarray(arr)
                except Exception:
                    continue
            return None

        lat = None
        lon = None
        source = None

        roots = [era5_zarr, smap_zarr, viirs_zarr]
        if extra_zarr is not None:
            roots = [extra_zarr] + roots

        # Step 1: lat/lon or latitude/longitude
        for root in roots:
            if lat is None:
                lat_candidate = try_open(root, ["lat", "latitude"])
                if lat_candidate is not None:
                    lat = self._normalize_coord_array(lat_candidate, "lat")
            if lon is None:
                lon_candidate = try_open(root, ["lon", "longitude"])
                if lon_candidate is not None:
                    lon = self._normalize_coord_array(lon_candidate, "lon")
            if lat is not None and lon is not None:
                source = "lat/lon"
                print(
                    f"[coords] Using 'lat'/'lon' from store {root} "
                    f"(shape lat={lat.shape}, lon={lon.shape})"
                )
                break

        # Step 2: fallback to y/x
        if lat is None or lon is None:
            lat = None
            lon = None
            for root in roots:
                y_arr = try_open(root, ["y"])
                x_arr = try_open(root, ["x"])
                if y_arr is not None and x_arr is not None:
                    try:
                        lat = self._normalize_coord_array(y_arr, "lat")
                        lon = self._normalize_coord_array(x_arr, "lon")
                        source = "y/x"
                        print(
                            f"[coords] Using 'y'/'x' from store {root} as lat/lon "
                            f"(shape lat={lat.shape}, lon={lon.shape})"
                        )
                        break
                    except Exception as e:
                        print(f"[coords] Failed to interpret y/x from {root} as lat/lon: {e}")
                        lat = None
                        lon = None

        if lat is None or lon is None:
            raise RuntimeError(
                "return_coords/coord_as_features=True but could not find usable coordinate arrays.\n"
                "Tried: 'lat'/'lon' or 'latitude'/'longitude' first, then 'y'/'x'. "
                "Please ensure your Zarr store has either lat/lon grids or y/x axes."
            )

        self.lat_grid = lat.astype(np.float32)
        self.lon_grid = lon.astype(np.float32)

        print(
            f"[coords] Loaded {source}-based coordinate grids for ({self.H},{self.W}), "
            f"dtype={self.lat_grid.dtype}"
        )

    def _make_coord_features_patch(self, y0: int, x0: int) -> np.ndarray:
        """Build coordinate feature patch (4, patch, patch): [sin(lat), cos(lat), sin(lon), cos(lon)] in radians."""
        if self.lat_grid is None or self.lon_grid is None:
            raise RuntimeError(
                "coord_as_features=True but lat_grid / lon_grid are not loaded. "
                "Check _load_lat_lon_with_xy_fallback."
            )

        lat_patch = self.lat_grid[y0 : y0 + self.patch, x0 : x0 + self.patch].astype(np.float32)
        lon_patch = self.lon_grid[y0 : y0 + self.patch, x0 : x0 + self.patch].astype(np.float32)

        lat_rad = np.deg2rad(lat_patch)
        lon_rad = np.deg2rad(lon_patch)

        feats = np.stack(
            [np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)],
            axis=0
        ).astype(np.float32)
        return feats

    def _build_index(self):
        """Build full index of (t_end, y0, x0), supporting 2-way or 3-way split."""
        t_end_first = self.t_hist - 1
        t_end_last = self.T - 1 - self.max_horizon

        if t_end_last < t_end_first:
            raise ValueError(
                f"Not enough time steps T={self.T} for t_hist={self.t_hist} "
                f"and max_horizon={self.max_horizon}"
            )

        t_ends = list(range(t_end_first, t_end_last + 1))
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

        n_total = len(idx_shuf)
        train_frac = self.split
        val_frac = self.val_frac

        if val_frac is None:
            split_at = int(n_total * train_frac)
            train_idx = idx_shuf[:split_at]
            val_idx = idx_shuf[split_at:]
            test_idx: List[Tuple[int, int, int]] = []
        else:
            n_train = int(n_total * train_frac)
            n_val = int(n_total * val_frac)
            n_test = max(0, n_total - n_train - n_val)

            train_idx = idx_shuf[:n_train]
            val_idx = idx_shuf[n_train:n_train + n_val]
            test_idx = idx_shuf[n_train + n_val:n_train + n_val + n_test]

        if self.mode == "train":
            chosen = train_idx
        elif self.mode == "val":
            chosen = val_idx
        elif self.mode == "test":
            chosen = test_idx
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if self.max_samples is not None:
            chosen = chosen[: self.max_samples]

        self.index: List[Tuple[int, int, int]] = chosen

    # --------------------------------------------------------
    # Dataset API
    # --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        t_end, y0, x0 = self.index[i]
        t0 = t_end - (self.t_hist - 1)

        # ---------- Inputs ----------
        io_time_era5 = 0.0
        io_time_smap = 0.0
        io_time_extra = 0.0
        io_time_viirs = 0.0

        t_io = time.time()
        era5_slab = self.era5_arr.get_orthogonal_selection(
            (
                slice(t0, t_end + 1),
                slice(None),
                slice(y0, y0 + self.patch),
                slice(x0, x0 + self.patch),
            )
        )
        io_time_era5 = time.time() - t_io

        t_io = time.time()
        smap_slab = self.smap_arr.get_orthogonal_selection(
            (
                slice(t0, t_end + 1),
                slice(None),
                slice(y0, y0 + self.patch),
                slice(x0, x0 + self.patch),
            )
        )
        io_time_smap = time.time() - t_io

        extra_np = None
        if self.extra_arr is not None:
            t_io = time.time()
            extra_slab = self.extra_arr.get_orthogonal_selection(
                (
                    slice(t0, t_end + 1),
                    slice(None),
                    slice(y0, y0 + self.patch),
                    slice(x0, x0 + self.patch),
                )
            )
            io_time_extra = time.time() - t_io
            extra_np = np.asarray(extra_slab, dtype=np.float32)
            extra_np = np.nan_to_num(extra_np, nan=0.0, posinf=0.0, neginf=0.0)

        era5_np = np.asarray(era5_slab, dtype=np.float32)
        smap_np = np.asarray(smap_slab, dtype=np.float32)

        era5_np = np.nan_to_num(era5_np, nan=0.0, posinf=0.0, neginf=0.0)
        smap_np = np.nan_to_num(smap_np, nan=0.0, posinf=0.0, neginf=0.0)

        parts = [era5_np, smap_np]
        if extra_np is not None:
            parts.append(extra_np)
        x_np = np.concatenate(parts, axis=1)

        # Optional coordinate features
        if self.coord_as_features:
            coord_feats = self._make_coord_features_patch(y0, x0)  # (4, patch, patch)
            coord_feats_t = np.repeat(coord_feats[None, ...], self.t_hist, axis=0)  # (T,4,H,W)
            x_np = np.concatenate([x_np, coord_feats_t], axis=1)

        x = torch.from_numpy(x_np)

        if torch.isnan(x).any():
            raise RuntimeError("NaN detected in input x after nan_to_num; check preprocessing.")

        if self.time_stack == "channel":
            x = x.permute(1, 0, 2, 3).reshape(self.C_total * self.t_hist, self.patch, self.patch)

        # ---------- Targets (VIIRS, 1 channel) ----------
        K = len(self.horizons)
        y_np = np.zeros((K, self.patch, self.patch), dtype=np.float32)
        mask_np = np.zeros((K, self.patch, self.patch), dtype=np.float32)

        peat_patch = None
        if self.peat_mask is not None:
            peat_patch = (self.peat_mask[y0:y0 + self.patch, x0:x0 + self.patch] > 0)

        for k, h in enumerate(self.horizons):
            t_label = t_end + h
            if t_label < 0 or t_label >= self.T:
                continue

            t_io = time.time()
            slab = self.viirs_arr.get_orthogonal_selection(
                (
                    slice(t_label, t_label + 1),
                    slice(0, 1),
                    slice(y0, y0 + self.patch),
                    slice(x0, x0 + self.patch),
                )
            )
            io_time_viirs += time.time() - t_io

            with np.errstate(over="ignore", invalid="ignore"):
                lbl = np.asarray(slab[0, 0], dtype=np.float32)

            finite = np.isfinite(lbl)
            in_range = (lbl >= 0.0) & (lbl <= 1.0)
            valid = finite & in_range

            if peat_patch is not None:
                valid = valid & peat_patch

            mask_np[k][valid] = 1.0
            y_np[k] = np.where(valid, lbl, 0.0)

        y = torch.from_numpy(y_np)
        mask = torch.from_numpy(mask_np)

        # ---------- Normalize inputs (optional) ----------
        if self.normalize_inputs == "per_channel":
            if self._mean is None or self._std is None:
                raise RuntimeError(
                    "Requested per_channel normalization, but mean/std have not been set."
                )

            if self.time_stack == "separate":
                mean = self._mean.view(1, -1, 1, 1)
                std = self._std.view(1, -1, 1, 1)
            else:
                mean = self._mean.repeat(self.t_hist).view(-1, 1, 1)
                std = self._std.repeat(self.t_hist).view(-1, 1, 1)

            std = torch.where(std == 0, torch.ones_like(std), std)
            x = (x - mean) / std

        # --- Sanity checks for non-finite outputs ---
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
                    "Check _load_lat_lon_with_xy_fallback."
                )
            lat_patch = self.lat_grid[y0 : y0 + self.patch, x0 : x0 + self.patch]
            lon_patch = self.lon_grid[y0 : y0 + self.patch, x0 : x0 + self.patch]

            lat_rad = np.deg2rad(lat_patch.astype(np.float32))
            lon_rad = np.deg2rad(lon_patch.astype(np.float32))

            coords_np = np.stack([lat_rad, lon_rad], axis=0)
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
            "io_time_era5": float(io_time_era5),
            "io_time_smap": float(io_time_smap),
            "io_time_extra": float(io_time_extra),
            "io_time_viirs": float(io_time_viirs),
            "io_time_total": float(io_time_era5 + io_time_smap + io_time_extra + io_time_viirs),
        }
        if self.extra_arr is not None:
            meta["extra_array_path"] = self.extra_path

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
        """Estimate per-channel mean/std of inputs from a subset of samples."""
        if len(self.index) == 0:
            raise RuntimeError("Cannot estimate input stats with an empty index.")

        rng = random.Random(self.seed + 123)
        n_samples = min(sample_count, len(self.index))
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

            extra_np = None
            if self.extra_arr is not None:
                extra_slab = self.extra_arr.get_orthogonal_selection(
                    (
                        slice(t0, t_end + 1),
                        slice(None),
                        slice(y0, y0 + self.patch),
                        slice(x0, x0 + self.patch),
                    )
                )
                extra_np = np.asarray(extra_slab, dtype=np.float32)
                extra_np = np.nan_to_num(extra_np, nan=0.0, posinf=0.0, neginf=0.0)

            era5_np = np.asarray(era5_slab, dtype=np.float32)
            smap_np = np.asarray(smap_slab, dtype=np.float32)

            era5_np = np.nan_to_num(era5_np, nan=0.0, posinf=0.0, neginf=0.0)
            smap_np = np.nan_to_num(smap_np, nan=0.0, posinf=0.0, neginf=0.0)

            parts = [era5_np, smap_np]
            if extra_np is not None:
                parts.append(extra_np)
            x_np = np.concatenate(parts, axis=1)

            if self.coord_as_features:
                coord_feats = self._make_coord_features_patch(y0, x0)
                coord_feats_t = np.repeat(coord_feats[None, ...], self.t_hist, axis=0)
                x_np = np.concatenate([x_np, coord_feats_t], axis=1)

            x_torch = torch.from_numpy(x_np).float()
            x_torch = x_torch.permute(1, 0, 2, 3).contiguous()  # (C_total, T_hist, H, W)

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
        if self._mean is None or self._std is None:
            raise RuntimeError("Normalization stats are not set for this dataset.")
        return self._mean.clone(), self._std.clone()

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
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

    parser = argparse.ArgumentParser(description="Joint ERA5+SMAP(+EXTRA)→VIIRS Zarr dataset")
    parser.add_argument("--era5-zarr", required=True)
    parser.add_argument("--smap-zarr", required=True)
    parser.add_argument("--viirs-zarr", required=True)
    parser.add_argument("--era5-array", default="field")
    parser.add_argument("--smap-array", default="field")
    parser.add_argument("--viirs-array", default="field")

    # NEW:
    parser.add_argument("--extra-zarr", default=None, help="Optional additional input Zarr store (T,C,H,W).")
    parser.add_argument("--extra-array", default="field", help="Array path inside --extra-zarr (default: field).")
    parser.add_argument("--no-enforce-extra-c4", action="store_true", help="Do not enforce EXTRA C==4.")

    parser.add_argument("--t-hist", type=int, default=30)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 7, 14])
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stack-time", choices=["separate", "channel"], default="separate")
    parser.add_argument("--mode", choices=["train", "val", "test"], default="train")
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--val-frac", type=float, default=None,
                        help="Optional explicit validation fraction. "
                             "If provided, test_frac = 1 - split - val_frac.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--normalize-inputs", choices=[None, "per_channel"], default=None)
    parser.add_argument("--no-skip-nonpeat", action="store_true")
    parser.add_argument("--return-coords", action="store_true", help="Return lat/lon coords (radians)")
    parser.add_argument("--coord-as-features", action="store_true",
                        help="Append sin/cos(lat,lon) as extra channels.")

    parser.add_argument(
        "--scheduler",
        choices=["distributed", "processes", "threads"],
        default="distributed",
    )
    parser.add_argument(
        "--dask-workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) // 4),
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--worker-mem",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--dask-dashboard",
        action="store_true",
    )

    args = parser.parse_args()

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
            extra_zarr=args.extra_zarr,
            extra_array=args.extra_array,
            enforce_extra_c4=not args.no_enforce_extra_c4,
            t_hist=args.t_hist,
            horizons=args.horizons,
            patch=args.patch,
            stride=args.stride,
            time_stack=args.stack_time,
            mode=args.mode,
            split=args.split,
            val_frac=args.val_frac,
            seed=42,
            normalize_inputs=args.normalize_inputs,
            max_samples=args.max_samples,
            skip_nonpeat_patches=not args.no_skip_nonpeat,
            peat_min_fraction=0.01,
            time_index=None,
            return_coords=args.return_coords,
            coord_as_features=args.coord_as_features,
        )

        print(
            f"T={ds.T}, C_era5={ds.C_era5}, C_smap={ds.C_smap}, C_extra={ds.C_extra}, C_total={ds.C_total}"
        )
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

        it = iter(loader)
        n_warm = min(3, len(loader))
        for _ in range(n_warm):
            batch = next(it)
            _ = batch["x"], batch["y"], batch["mask"], batch["meta"]

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
