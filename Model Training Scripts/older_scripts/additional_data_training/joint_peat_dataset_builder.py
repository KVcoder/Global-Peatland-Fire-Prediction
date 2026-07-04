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


from dataclasses import dataclass

@dataclass(frozen=True)
class InputSpec:
    zarr: str
    array: str = "field"

def parse_input_spec(s: str) -> InputSpec:
    # formats:
    #   "/path/store.zarr"            -> array="field"
    #   "/path/store.zarr:arrayname"  -> array="arrayname"
    s = str(s)
    if ":" in s:
        z, a = s.split(":", 1)
        z, a = z.strip(), a.strip()
        if not a:
            a = "field"
        return InputSpec(zarr=z, array=a)
    return InputSpec(zarr=s.strip(), array="field")


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
    return sorted(set((y,x) for y in ys for x in xs))



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

    def __init__(self,
    inputs: Sequence[InputSpec],
    viirs_zarr: str,
    viirs_array: Optional[str] = "field",
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
    time_index: Optional[Sequence[int]] = None,
    return_coords: bool = False,
    coord_as_features: bool = False,
    peat_mask_source: Optional[str] = None,
    coords_source: Optional[str] = None,
    t_end_index: Optional[Sequence[int]] = None,
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
        self.peat_mask_source = peat_mask_source
        self.coords_source = coords_source
        self.t_end_index = None if t_end_index is None else [int(x) for x in t_end_index]
        self.lat_grid = None
        self.lon_grid = None

        
        
        self.input_specs = list(inputs)
        if len(self.input_specs) == 0:
            raise ValueError("inputs must contain at least one InputSpec")

        self.input_arrs = []
        self.input_paths = []
        self.input_C = []

        # Open labels first (still single store)
        self.viirs_arr, self.viirs_path = _open_zarr_array(viirs_zarr, viirs_array)
        T_v, C_v, H_v, W_v = _infer_layout(self.viirs_arr.shape)
        if C_v != 1:
            raise ValueError(f"VIIRS must have C=1, got C={C_v}")

        # Open inputs
        T_ref = H_ref = W_ref = None

        for spec in self.input_specs:
            arr, path = _open_zarr_array(spec.zarr, spec.array)
            T, C, H, W = _infer_layout(arr.shape)

            if T_ref is None:
                T_ref, H_ref, W_ref = T, H, W
            else:
                if T != T_ref:
                    raise ValueError(f"Time mismatch for {spec.zarr}:{spec.array} -> T={T}, expected {T_ref}")
                if (H, W) != (H_ref, W_ref):
                    raise ValueError(f"Spatial mismatch for {spec.zarr}:{spec.array} -> {(H,W)}, expected {(H_ref,W_ref)}")
            
            self.input_arrs.append(arr)
            self.input_paths.append(path)
            self.input_C.append(int(C))

        # Validate VIIRS against inputs
        if T_v != T_ref:
            raise ValueError(f"VIIRS T={T_v} does not match inputs T={T_ref}")
        if (H_v, W_v) != (H_ref, W_ref):
            raise ValueError(f"VIIRS {(H_v,W_v)} does not match inputs {(H_ref,W_ref)}")

        self.T = int(T_ref)
        self.H = int(H_ref)
        self.W = int(W_ref)

    
        if self.patch > self.H or self.patch > self.W:
            raise ValueError(f"Patch size {self.patch} exceeds spatial dimensions {self.H}x{self.W}")
    
        self.C_inputs = sum(self.input_C)
        self.C_viirs = 1

        self.C_base = self.C_inputs
        self.C_total = self.C_base + (4 if self.coord_as_features else 0)


        self.max_horizon = max(self.horizons) if self.horizons else 0

        for idx, (arr, spec) in enumerate(zip(self.input_arrs, self.input_specs)):
            self._warn_on_unfriendly_chunks(arr, f"IN{idx}:{os.path.basename(spec.zarr)}")
        self._warn_on_unfriendly_chunks(self.viirs_arr, "VIIRS")


        # ---------- Peat mask ----------
        self.peat_mask = None
        if self.skip_nonpeat_patches:
            if not peat_mask_source:
                raise RuntimeError(
                    "skip_nonpeat_patches=True requires peat_mask_source to be set "
                    "(SMAP_WTD is authoritative in your setup)."
                )
            self._load_peat_mask_from_store(peat_mask_source)




        # ---------- Lat / Lon grids (with lat/lon OR x/y fallback) ----------
        if self.return_coords or self.coord_as_features:
            if self.coords_source is not None:
                self._load_lat_lon_with_xy_fallback_from_roots([self.coords_source])
            else:
                roots = [s.zarr for s in self.input_specs] + [viirs_zarr]
                self._load_lat_lon_with_xy_fallback_from_roots(roots)



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

    def _load_peat_mask_from_store(self, store_path: str):
        try:
            arr, _ = _open_zarr_array(store_path, "peat_mask")
            mask = np.asarray(arr, dtype=np.uint8)
        except Exception as e:
            raise RuntimeError(
                f"peat_mask_source={store_path!r} was requested, but 'peat_mask' could not be loaded: {e}"
            )

        if mask.shape != (self.H, self.W):
            raise ValueError(
                f"peat_mask from {store_path!r} has shape {mask.shape}, expected ({self.H},{self.W})"
            )

        self.peat_mask = mask
        print(f"[peat_mask] Forced peat_mask loaded ONLY from: {store_path}")

    

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

    def _load_lat_lon_with_xy_fallback_from_roots(self, roots: Sequence[str]):
        """
        Load latitude/longitude grids from exactly ONE of the given Zarr stores, with fallback:

        Priority per-store:
        1) 'lat'/'lon' or 'latitude'/'longitude' (and a few common variants)
        2) 'y'/'x' (assumed y=lat degrees, x=lon degrees)

        Rules / behavior:
        - We require BOTH lat and lon to come from the SAME store (no mixing across stores),
            to keep the grid source deterministic.
        - Supported coordinate shapes are handled by _normalize_coord_array():
            (H,W), (H,), (W,), (1,H,W), (H,W,1), (T,C,H,W)->[0,0]
        - For y/x fallback:
            * if y is (H,) and x is (W,), they broadcast to (H,W)
            * if y/x are already (H,W), they’re used directly
        - Assumes degrees; conversion to radians happens later where needed.
        """

        def try_open(store_path: str, names: Sequence[str]) -> Optional[np.ndarray]:
            for nm in names:
                try:
                    arr, _ = _open_zarr_array(store_path, nm)
                    return np.asarray(arr)
                except Exception:
                    continue
            return None

        # You can add/remove aliases here as needed
        LAT_NAMES = ["lat", "latitude", "nav_lat", "LAT", "Latitude"]
        LON_NAMES = ["lon", "longitude", "nav_lon", "LON", "Longitude"]
        Y_NAMES   = ["y", "Y"]
        X_NAMES   = ["x", "X"]

        lat = None
        lon = None
        source_store = None
        source_kind = None

        # ------------------------------------------------------------
        # 1) Try lat/lon (or latitude/longitude) in the SAME store
        # ------------------------------------------------------------
        for root in roots:
            lat_raw = try_open(root, LAT_NAMES)
            lon_raw = try_open(root, LON_NAMES)
            if lat_raw is None or lon_raw is None:
                continue

            try:
                lat = self._normalize_coord_array(lat_raw, "lat")
                lon = self._normalize_coord_array(lon_raw, "lon")
            except Exception as e:
                print(f"[coords] Found lat/lon in {root} but could not normalize: {e}")
                lat, lon = None, None
                continue

            source_store = root
            source_kind = "lat/lon"
            break

        # ------------------------------------------------------------
        # 2) Fallback: try y/x in the SAME store
        # ------------------------------------------------------------
        if lat is None or lon is None:
            lat, lon = None, None
            for root in roots:
                y_raw = try_open(root, Y_NAMES)
                x_raw = try_open(root, X_NAMES)
                if y_raw is None or x_raw is None:
                    continue

                try:
                    y_arr = np.asarray(y_raw)
                    x_arr = np.asarray(x_raw)

                    # Common case: y(H,) and x(W,) axes
                    if y_arr.ndim == 1 and x_arr.ndim == 1:
                        if y_arr.shape[0] != self.H or x_arr.shape[0] != self.W:
                            raise ValueError(
                                f"y/x are 1D but shapes are y={y_arr.shape}, x={x_arr.shape}; "
                                f"expected y=({self.H},), x=({self.W},)"
                            )
                        lat = np.repeat(y_arr[:, None], self.W, axis=1)
                        lon = np.repeat(x_arr[None, :], self.H, axis=0)

                    else:
                        # 2D grids or (1,H,W) / (H,W,1) / (T,C,H,W) etc.
                        lat = self._normalize_coord_array(y_arr, "lat")
                        lon = self._normalize_coord_array(x_arr, "lon")

                except Exception as e:
                    print(f"[coords] Found y/x in {root} but could not interpret as lat/lon: {e}")
                    lat, lon = None, None
                    continue

                source_store = root
                source_kind = "y/x"
                break

        if lat is None or lon is None:
            raise RuntimeError(
                "return_coords/coord_as_features=True but could not find usable coordinate arrays.\n"
                f"Tried per-store in roots={list(roots)}:\n"
                "  1) lat/lon (lat, latitude, nav_lat; lon, longitude, nav_lon)\n"
                "  2) y/x (assumed y=lat deg, x=lon deg)\n"
                "Ensure at least one store contains BOTH coordinate arrays in a supported shape."
            )

        # Store as float32 grids in degrees
        self.lat_grid = np.asarray(lat, dtype=np.float32)
        self.lon_grid = np.asarray(lon, dtype=np.float32)

        if self.lat_grid.shape != (self.H, self.W) or self.lon_grid.shape != (self.H, self.W):
            raise ValueError(
                f"Loaded coords have wrong shape: lat={self.lat_grid.shape}, lon={self.lon_grid.shape}, "
                f"expected ({self.H},{self.W})"
            )

        print(
            f"[coords] Loaded {source_kind}-based coordinate grids from: {source_store}\n"
            f"        lat/lon shapes: {self.lat_grid.shape} / {self.lon_grid.shape}, dtype={self.lat_grid.dtype}"
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
        
        if self.t_end_index is not None:
            allowed_set = set(self.t_end_index)
            t_ends = [t for t in t_ends if t in allowed_set]

        
        # Optional time filtering: only allow windows fully contained in allowed indices
        if self.time_index is not None:
            allowed = np.zeros(self.T, dtype=np.bool_)
            idx = np.asarray(self.time_index, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < self.T)]
            allowed[idx] = True

            filtered = []
            for t_end in t_ends:
                t0 = t_end - (self.t_hist - 1)
                t1 = t_end + self.max_horizon
                if t0 < 0 or t1 >= self.T:
                    continue
                if allowed[t0 : t1 + 1].all():
                    filtered.append(t_end)

            t_ends = filtered

        
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

        # =====================================================
        # Inputs (loop over arbitrary number of input stores)
        # =====================================================
        io_time_viirs = 0.0
        io_times_inputs: List[float] = []

        parts: List[np.ndarray] = []
        for arr in self.input_arrs:
            t_io = time.time()
            slab = arr.get_orthogonal_selection(
                (
                    slice(t0, t_end + 1),          # T
                    slice(None),                  # C
                    slice(y0, y0 + self.patch),   # H
                    slice(x0, x0 + self.patch),   # W
                )
            )
            io_times_inputs.append(time.time() - t_io)

            np_slab = np.asarray(slab, dtype=np.float32)
            np_slab = np.nan_to_num(np_slab, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(np_slab)

        # concat along channel axis: (T, sumC, patch, patch)
        x_np = np.concatenate(parts, axis=1)

        # Optional coordinate features (same behavior as before)
        if self.coord_as_features:
            coord_feats = self._make_coord_features_patch(y0, x0)              # (4, patch, patch)
            coord_feats_t = np.repeat(coord_feats[None, ...], self.t_hist, axis=0)  # (T,4,patch,patch)
            x_np = np.concatenate([x_np, coord_feats_t], axis=1)

        x = torch.from_numpy(x_np)

        #if torch.isnan(x).any():
            #raise RuntimeError("NaN detected in input x after nan_to_num; check preprocessing.")

        if self.time_stack == "channel":
            x = x.permute(1, 0, 2, 3).reshape(self.C_total * self.t_hist, self.patch, self.patch)

        # =====================================================
        # Targets (VIIRS) — unchanged from your current code
        # =====================================================
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

        # =====================================================
        # Normalize inputs (optional) — unchanged
        # =====================================================
        if self.normalize_inputs == "per_channel":
            if self._mean is None or self._std is None:
                raise RuntimeError("Requested per_channel normalization, but mean/std have not been set.")

            if self.time_stack == "separate":
                mean = self._mean.view(1, -1, 1, 1)
                std = self._std.view(1, -1, 1, 1)
            else:
                mean = self._mean.repeat(self.t_hist).view(-1, 1, 1)
                std = self._std.repeat(self.t_hist).view(-1, 1, 1)

            std = torch.where(std == 0, torch.ones_like(std), std)
            x = (x - mean) / std

        # =====================================================
        # Coordinates output (optional) — unchanged
        # =====================================================
        coords = None
        if self.return_coords:
            if self.lat_grid is None or self.lon_grid is None:
                raise RuntimeError("return_coords=True but lat_grid / lon_grid are not loaded.")
            lat_patch = self.lat_grid[y0 : y0 + self.patch, x0 : x0 + self.patch]
            lon_patch = self.lon_grid[y0 : y0 + self.patch, x0 : x0 + self.patch]
            coords_np = np.stack([np.deg2rad(lat_patch.astype(np.float32)),
                                np.deg2rad(lon_patch.astype(np.float32))], axis=0)
            coords = torch.from_numpy(coords_np)

        meta = {
            "t_end": t_end,
            "t_start": t0,
            "y0": y0,
            "x0": x0,
            "horizons": self.horizons,
            "viirs_array_path": self.viirs_path,

            # NEW: list of input array paths and IO times
            "input_array_paths": self.input_paths,
            "io_time_inputs": [float(t) for t in io_times_inputs],
            "io_time_inputs_total": float(sum(io_times_inputs)),
            "io_time_viirs": float(io_time_viirs),
            "io_time_total": float(sum(io_times_inputs) + io_time_viirs),
        }

        if self.time_index is not None:
            meta["history_time_steps"] = list(range(t0, t_end + 1))


        sample = {"x": x, "y": y, "mask": mask, "meta": meta}
        if self.return_coords:
            sample["coords"] = coords

        return sample

    # --------------------------------------------------------
    # Normalization helpers
    # --------------------------------------------------------

    def _estimate_input_stats(self, sample_count: int = 256):
        """Estimate per-channel mean/std of inputs (and coord features if enabled)."""
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

            parts = []
            for arr in self.input_arrs:
                slab = arr.get_orthogonal_selection(
                    (slice(t0, t_end + 1), slice(None), slice(y0, y0 + self.patch), slice(x0, x0 + self.patch))
                )
                np_slab = np.asarray(slab, dtype=np.float32)
                np_slab = np.nan_to_num(np_slab, nan=0.0, posinf=0.0, neginf=0.0)
                parts.append(np_slab)

            x_np = np.concatenate(parts, axis=1)  # (T, C_inputs, patch, patch)

            if self.coord_as_features:
                coord_feats = self._make_coord_features_patch(y0, x0)                    # (4, patch, patch)
                coord_feats_t = np.repeat(coord_feats[None, ...], self.t_hist, axis=0)   # (T, 4, patch, patch)
                x_np = np.concatenate([x_np, coord_feats_t], axis=1)                     # (T, C_total, patch, patch)

            x64 = torch.from_numpy(x_np).to(torch.float64)  # (T, C, H, W)
            sum_c += x64.sum(dim=(0, 2, 3))
            sqsum_c += (x64 * x64).sum(dim=(0, 2, 3))
            count += x64.shape[0] * x64.shape[2] * x64.shape[3]

        mean = (sum_c / max(count, 1)).to(torch.float32)
        var = (sqsum_c / max(count, 1)) - (mean.double() ** 2)
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

if __name__ == "__main__":
    pass
