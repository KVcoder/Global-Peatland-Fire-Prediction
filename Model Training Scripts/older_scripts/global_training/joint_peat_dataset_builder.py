#!/usr/bin/env python3
"""
joint_peat_dataset_builder.py

Joint multi-input Zarr (T,C,H,W) → VIIRS multi-horizon dataset for PyTorch.

Key features:
- Zarr-backed, robust to slightly messy layouts via _open_zarr_array.
- Handles NaNs/Infs in inputs (nan_to_num -> 0).
- Masks out invalid VIIRS labels (NaNs/Infs/out-of-range) and peat-only label regions.
- Optional per-channel normalization (estimated from random samples).
- Optional lat/lon coordinates per patch, in radians.
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

Holdout region option:
- If holdout_region_source AND holdout_t_end_index are provided:
    - TEST set is defined as: (t_end in holdout_t_end_index) AND (patch overlaps holdout region >= holdout_min_fraction)
    - TRAIN/VAL sets exclude those region patches for those holdout anchors.
    - In this mode, we do NOT create an additional random "test" split; test is the holdout.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

# ============================================================
# InputSpec helpers
# ============================================================

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
        - first try to open that array via the root group,
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

            # arrays at this level
            names: List[str] = []
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

            # child groups
            grp_names: List[str] = []
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
    for dirpath, _, filenames in os.walk(store_path):
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
    return int(T), int(C), int(H), int(W)


def _compute_grid_positions(H: int, W: int, patch: int, stride: Optional[int]) -> List[Tuple[int, int]]:
    if stride is None:
        stride = patch

    # valid top-left positions inclusive of last patch
    ys = list(range(0, max(H - patch + 1, 1), stride))
    xs = list(range(0, max(W - patch + 1, 1), stride))

    # ensure last aligns to H-patch / W-patch
    if ys and ys[-1] != max(H - patch, 0):
        ys.append(max(H - patch, 0))
    if xs and xs[-1] != max(W - patch, 0):
        xs.append(max(W - patch, 0))

    return sorted(set((y, x) for y in ys for x in xs))


# ============================================================
# Joint dataset
# ============================================================

class JointPeatDataset(Dataset):
    """
    Multi-input → VIIRS multi-horizon dataset (Zarr-backed).

    Per sample:
        "x": (t_hist, C_total, patch, patch) or (t_hist*C_total, patch, patch)
        "y": (K, patch, patch)               (K = len(horizons))
        "mask": same shape as y, 1=label present, 0=missing
        "coords": (2, patch, patch) [lat_rad, lon_rad] if return_coords=True
        "meta": dict with IO times and indices
    """

    def __init__(
        self,
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
        coords_units: Literal["auto", "degrees", "radians"] = "auto",
        # --- spatiotemporal holdout ---
        holdout_region_source: Optional[str] = None,          # Zarr with 'peat_mask' defining held-out region (valid >0)
        holdout_t_end_index: Optional[Sequence[int]] = None,  # anchors (t_end) where holdout applies (e.g. all t_end in 2023)
        holdout_min_fraction: float = 0.01,
    ):
        super().__init__()

        self.t_hist = int(t_hist)
        self.horizons = [int(h) for h in horizons]
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
        self.t_end_index = None if t_end_index is None else [int(x) for x in t_end_index]

        self.holdout_region_source = None if holdout_region_source in (None, "") else str(holdout_region_source)
        self.holdout_t_end_index = None if holdout_t_end_index is None else [int(x) for x in holdout_t_end_index]
        self.holdout_min_fraction = float(holdout_min_fraction)
        self.holdout_region_mask: Optional[np.ndarray] = None  # bool (H,W)

        self.return_coords = bool(return_coords)
        self.coord_as_features = bool(coord_as_features)
        self.peat_mask_source = peat_mask_source
        self.coords_source = coords_source
        self.coords_units = str(coords_units)
        self.coords_are_degrees: Optional[bool] = None  # resolved later if auto
        self.lat_grid: Optional[np.ndarray] = None
        self.lon_grid: Optional[np.ndarray] = None

        self.input_specs = list(inputs)
        if len(self.input_specs) == 0:
            raise ValueError("inputs must contain at least one InputSpec")

        self._viirs_zarr_path = str(viirs_zarr)
        self._viirs_array_name = viirs_array

        
        self.input_arrs = []
        self.input_paths = []
        self.input_C = []

        # Open labels first
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

        self.C_inputs = int(sum(self.input_C))
        self.C_viirs = 1

        self.C_base = self.C_inputs
        self.C_total = self.C_base + (4 if self.coord_as_features else 0)

        self.max_horizon = max(self.horizons) if self.horizons else 0

        for idx, (arr, spec) in enumerate(zip(self.input_arrs, self.input_specs)):
            self._warn_on_unfriendly_chunks(arr, f"IN{idx}:{os.path.basename(spec.zarr)}")
        self._warn_on_unfriendly_chunks(self.viirs_arr, "VIIRS")

        # ---------- Peat mask ----------
        self.peat_mask: Optional[np.ndarray] = None
        if self.skip_nonpeat_patches:
            if not peat_mask_source:
                raise RuntimeError(
                    "skip_nonpeat_patches=True requires peat_mask_source to be set "
                    "(SMAP_WTD is authoritative in your setup)."
                )
            self._load_peat_mask_from_store(peat_mask_source)


        # ---------- Holdout region mask (optional) ----------
        if self.holdout_region_source is not None:
            self._load_holdout_region_mask_from_store(self.holdout_region_source)

        # ---------- Lat / Lon grids ----------
        if self.return_coords or self.coord_as_features:
            if self.coords_source is not None:
                self._load_lat_lon_with_xy_fallback_from_roots([self.coords_source])
            else:
                roots = [s.zarr for s in self.input_specs] + [viirs_zarr]
                self._load_lat_lon_with_xy_fallback_from_roots(roots)

        # ---------- Build index ----------
        self._build_index()

        # ---------- Normalization stats ----------
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        if self.normalize_inputs == "per_channel":
            if len(self.index) == 0:
                raise RuntimeError("Cannot estimate normalization stats: dataset index is empty.")
            self._mean, self._std = self._estimate_input_stats(
                sample_count=min(256, len(self.index))
            )

        # ---------- Finally: move big grids to shared memory (DataLoader workers won't duplicate) ----------
        self._maybe_make_shared_grids()

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _get_coords_grids_for_sampling(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (lat, lon) as numpy arrays for sampling/range checks.
        Prefers numpy grids; falls back to shared tensors.
        """
        if self.lat_grid is not None and self.lon_grid is not None:
            return self.lat_grid, self.lon_grid

        has_t = (
            hasattr(self, "lat_grid_t") and self.lat_grid_t is not None and
            hasattr(self, "lon_grid_t") and self.lon_grid_t is not None
        )
        if has_t:
            # Convert *once* for unit inference / sanity sampling only.
            return self.lat_grid_t.cpu().numpy(), self.lon_grid_t.cpu().numpy()

        raise RuntimeError("Coordinate grids not available (neither numpy nor shared tensors).")

        
    def _check_coords_sanity(self, lat: np.ndarray, lon: np.ndarray):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)

        # finite fraction
        fin = np.isfinite(lat) & np.isfinite(lon)
        frac_fin = float(fin.mean())
        if frac_fin < 0.95:
            print(f"[warn][coords] Only {100*frac_fin:.2f}% of lat/lon are finite.")

        # sample stats
        lat_min, lat_max = self._sample_min_max(lat, n=50000, seed=11)
        lon_min, lon_max = self._sample_min_max(lon, n=50000, seed=12)

        # detect "projected" looking coords (meters)
        if (max(abs(lat_min), abs(lat_max)) > 1000) or (max(abs(lon_min), abs(lon_max)) > 1000):
            print(
                "[warn][coords] Values look like projected meters (very large magnitudes). "
                f"lat_min/max={lat_min:.6g}/{lat_max:.6g}, lon_min/max={lon_min:.6g}/{lon_max:.6g}"
            )

        # quick gradient dominance check (lat should vary mostly with y; lon mostly with x)
        H, W = lat.shape
        ymid = H // 2
        xmid = W // 2

        # mean abs diffs
        lat_dy = np.nanmean(np.abs(np.diff(lat[:, xmid])))
        lat_dx = np.nanmean(np.abs(np.diff(lat[ymid, :])))
        lon_dy = np.nanmean(np.abs(np.diff(lon[:, xmid])))
        lon_dx = np.nanmean(np.abs(np.diff(lon[ymid, :])))

        # warn if "swapped-ish"
        if lat_dx > 2.0 * lat_dy:
            print(f"[warn][coords] Lat varies more along X than Y (lat_dx={lat_dx:.3g}, lat_dy={lat_dy:.3g}) → possible transpose/swapped axes.")
        if lon_dy > 2.0 * lon_dx:
            print(f"[warn][coords] Lon varies more along Y than X (lon_dy={lon_dy:.3g}, lon_dx={lon_dx:.3g}) → possible transpose/swapped axes.")

        # unit consistency check after your unit resolver sets coords_are_degrees
        if self.coords_are_degrees is True:
            if max(abs(lat_min), abs(lat_max)) > 90.0 * 1.5:
                print(f"[warn][coords] Degrees mode but lat out of expected range: {lat_min:.6g}..{lat_max:.6g}")
            if max(abs(lon_min), abs(lon_max)) > 360.0 * 1.5:
                print(f"[warn][coords] Degrees mode but lon out of expected range: {lon_min:.6g}..{lon_max:.6g}")
        elif self.coords_are_degrees is False:
            if max(abs(lat_min), abs(lat_max)) > np.pi * 1.5:
                print(f"[warn][coords] Radians mode but lat out of expected range: {lat_min:.6g}..{lat_max:.6g}")
            if max(abs(lon_min), abs(lon_max)) > 2*np.pi * 1.5:
                print(f"[warn][coords] Radians mode but lon out of expected range: {lon_min:.6g}..{lon_max:.6g}")

        print(
            f"[coords] sanity: finite={100*frac_fin:.2f}% | "
            f"lat[{lat_min:.6g},{lat_max:.6g}] lon[{lon_min:.6g},{lon_max:.6g}] | "
            f"lat_dy={lat_dy:.3g},lat_dx={lat_dx:.3g},lon_dy={lon_dy:.3g},lon_dx={lon_dx:.3g}"
        )
        
    
    def _to_shared_tensor(self, a: np.ndarray, name: str) -> torch.Tensor:
        """
        Put a CPU tensor into OS shared memory so DataLoader workers don't duplicate it.
        """
        t = torch.from_numpy(np.asarray(a))  # shares with numpy view for now
        if t.device.type != "cpu":
            t = t.cpu()
        # ensure contiguous storage
        t = t.contiguous()
        # move storage into shared memory
        t.share_memory_()
        return t

    def _maybe_make_shared_grids(self):
        # Only create if present; avoid keeping both numpy+tensor copies if RAM matters
        if self.peat_mask is not None and not hasattr(self, "peat_mask_t"):
            self.peat_mask_t = self._to_shared_tensor(self.peat_mask.astype(np.uint8, copy=False), "peat_mask")
            self.peat_mask = None  # drop numpy copy to save RAM

        if self.holdout_region_mask is not None and not hasattr(self, "holdout_region_mask_t"):
            self.holdout_region_mask_t = self._to_shared_tensor(self.holdout_region_mask.astype(np.bool_, copy=False), "holdout_region")
            self.holdout_region_mask = None

        if self.lat_grid is not None and self.lon_grid is not None and not hasattr(self, "lat_grid_t"):
            self.lat_grid_t = self._to_shared_tensor(self.lat_grid.astype(np.float32, copy=False), "lat_grid")
            self.lon_grid_t = self._to_shared_tensor(self.lon_grid.astype(np.float32, copy=False), "lon_grid")
            self.lat_grid = None
            self.lon_grid = None


    def _sample_min_max(self, arr: np.ndarray, n: int = 50000, seed: int = 0) -> Tuple[float, float]:
        """Fast approximate min/max using random sampling of finite values."""
        a = np.asarray(arr).reshape(-1)
        if a.size == 0:
            return float("nan"), float("nan")

        # keep only finite values
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return float("nan"), float("nan")

        if finite.size <= n:
            s = finite
        else:
            rng = np.random.RandomState(self.seed + 999 + seed)
            idx = rng.choice(finite.size, size=n, replace=False)
            s = finite[idx]

        return float(np.min(s)), float(np.max(s))


    def _infer_coords_are_degrees(self, lat: np.ndarray, lon: np.ndarray) -> Optional[bool]:
        """
        Return True if degrees, False if radians, None if uncertain.
        Uses sampling-based range heuristics.
        """
        lat_min, lat_max = self._sample_min_max(lat, n=50000, seed=1)
        lon_min, lon_max = self._sample_min_max(lon, n=50000, seed=2)

        if not (np.isfinite(lat_min) and np.isfinite(lat_max) and np.isfinite(lon_min) and np.isfinite(lon_max)):
            return None

        lat_abs = max(abs(lat_min), abs(lat_max))
        lon_abs = max(abs(lon_min), abs(lon_max))

        # allow some slack
        rad_lat_lim = np.pi * 1.25          # ~3.93
        rad_lon_lim = (2.0 * np.pi) * 1.25  # ~7.85

        deg_lat_lim = 90.0 * 1.25           # 112.5
        deg_lon_lim = 360.0 * 1.25          # 450

        # Strong radians signature: both comfortably within rad bounds
        if lat_abs <= rad_lat_lim and lon_abs <= rad_lon_lim:
            return False

        # Strong degrees signature: lat within ~90 and lon within ~360
        if lat_abs <= deg_lat_lim and lon_abs <= deg_lon_lim:
            return True

        # ambiguous / weird coordinate system
        return None


    def _resolve_coords_units(self):
        """
        Decide self.coords_are_degrees based on coords_units and detected ranges.
        Call this after lat_grid/lon_grid are loaded.
        """
        lat, lon = self._get_coords_grids_for_sampling()


        mode = (self.coords_units or "auto").lower().strip()

        if mode == "degrees":
            self.coords_are_degrees = True
            print("[coords] Using user-specified units: degrees")
            return
        if mode == "radians":
            self.coords_are_degrees = False
            print("[coords] Using user-specified units: radians")
            return
        if mode != "auto":
            print(f"[warn] Unknown coords_units={mode!r}; falling back to auto.")
            mode = "auto"

        guess = self._infer_coords_are_degrees(lat, lon)

        # If uncertain, default to degrees (safer in most geodata), but warn loudly
        if guess is None:
            lat_min, lat_max = self._sample_min_max(lat, n=50000, seed=3)
            lon_min, lon_max = self._sample_min_max(lon, n=50000, seed=4)
            self.coords_are_degrees = True
            print(
                "[warn][coords] Could not confidently infer units; defaulting to DEGREES.\n"
                f"             sampled lat_min/max={lat_min:.6g}/{lat_max:.6g}, lon_min/max={lon_min:.6g}/{lon_max:.6g}\n"
                "             If this is wrong, pass coords_units='radians'."
            )
            return

        self.coords_are_degrees = bool(guess)
        unit_str = "degrees" if self.coords_are_degrees else "radians"
        print(f"[coords] Auto-detected coordinate units: {unit_str}")

    
    def _reopen_zarr_handles(self):
        # Reopen VIIRS
        self.viirs_arr, self.viirs_path = _open_zarr_array(self._viirs_zarr_path, self._viirs_array_name)
        T_v, C_v, H_v, W_v = _infer_layout(self.viirs_arr.shape)
        if C_v != 1:
            raise ValueError(f"VIIRS must have C=1, got C={C_v}")

        # Reopen inputs
        self.input_arrs = []
        self.input_paths = []
        self.input_C = []

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

    def __getstate__(self):
        """
        When DataLoader workers spawn, the Dataset is pickled.
        Drop non-picklable Zarr handles; reopen in __setstate__.
        """
        d = dict(self.__dict__)
        d["input_arrs"] = None
        d["viirs_arr"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

        # CRITICAL: Apply zarr thread limits BEFORE reopening handles
        # This prevents thread explosion when workers unpickle the dataset
        try:
            import zarr
            import os

            # Try to get config from environment (set by worker_init_fn or main)
            async_c = int(os.environ.get("ZARR_V3_ASYNC_CONCURRENCY", "4"))
            th_w = int(os.environ.get("ZARR_V3_THREAD_MAX_WORKERS", "4"))

            zarr.config.set({
                "async.concurrency": async_c,
                "threading.max_workers": th_w,
            })
        except Exception:
            # Fallback to safe defaults if config fails
            pass

        # Restore Zarr handles in the worker process
        self._reopen_zarr_handles()
    
    
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
        print(f"[peat_mask] Loaded from: {store_path}")

    def _load_holdout_region_mask_from_store(self, store_path: str):
        """
        Load held-out REGION mask from store_path['peat_mask'].
        Convention: region pixels are finite and >0.
        """
        try:
            arr, _ = _open_zarr_array(store_path, "peat_mask")
            raw = np.asarray(arr)
        except Exception as e:
            raise RuntimeError(
                f"holdout_region_source={store_path!r} requested, but 'peat_mask' could not be loaded: {e}"
            )

        if raw.shape != (self.H, self.W):
            raise ValueError(
                f"holdout region mask from {store_path!r} has shape {raw.shape}, expected ({self.H},{self.W})"
            )

        with np.errstate(invalid="ignore"):
            finite = np.isfinite(raw)
            gt0 = raw > 0
            region = finite & gt0

        self.holdout_region_mask = region.astype(np.bool_)
        frac = float(self.holdout_region_mask.mean())
        print(f"[holdout_region] Loaded from: {store_path} (fraction={100.0*frac:.3f}%)")

    def _warn_on_unfriendly_chunks(self, arr, name: str):
        chunks = getattr(arr, "chunks", None)
        if chunks is None:
            return
        try:
            if len(chunks) == 4:
                t_chunk, _, h_chunk, w_chunk = chunks
            elif len(chunks) == 3:
                _, h_chunk, w_chunk = chunks
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
                    f"[warn] {name} Zarr chunks may be suboptimal:\n"
                    f"       chunks={chunks}, t_hist={self.t_hist}, patch={self.patch}\n"
                    f"       {'; '.join(msgs)}\n"
                    f"       Consider rechunking to (T≈{self.t_hist}, C, H≥{self.patch}, W≥{self.patch})."
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
                raise ValueError(f"{name} array has shape {arr.shape}, expected ({self.H},{self.W})")
            return arr

        if arr.ndim == 1:
            if name == "lat" and arr.shape[0] == self.H:
                return np.repeat(arr[:, None], self.W, axis=1)
            if name == "lon" and arr.shape[0] == self.W:
                return np.repeat(arr[None, :], self.H, axis=0)
            raise ValueError(f"Cannot broadcast 1D {name} array with shape {arr.shape} to ({self.H},{self.W})")

        if arr.ndim == 3:
            if arr.shape[0] == 1 and arr.shape[1:] == (self.H, self.W):
                return arr[0]
            if arr.shape[-1] == 1 and arr.shape[0:2] == (self.H, self.W):
                return arr[..., 0]

        if arr.ndim == 4 and arr.shape[2:] == (self.H, self.W):
            return arr[0, 0]

        raise ValueError(f"Unsupported shape for {name} array: {arr.shape} (cannot map to ({self.H},{self.W}))")

    def _load_lat_lon_with_xy_fallback_from_roots(self, roots: Sequence[str]):
        """
        Load lat/lon grids from exactly ONE of the given Zarr stores.

        Priority per-store:
        1) lat/lon (or aliases)
        2) y/x (assumed y=lat deg, x=lon deg)

        We require BOTH lat and lon to come from the SAME store (no mixing).
        """
        def try_open(store_path: str, names: Sequence[str]) -> Optional[np.ndarray]:
            for nm in names:
                try:
                    arr, _ = _open_zarr_array(store_path, nm)
                    return np.asarray(arr)
                except Exception:
                    continue
            return None

        LAT_NAMES = ["lat", "latitude", "nav_lat", "LAT", "Latitude"]
        LON_NAMES = ["lon", "longitude", "nav_lon", "LON", "Longitude"]
        Y_NAMES = ["y", "Y"]
        X_NAMES = ["x", "X"]

        lat = None
        lon = None
        source_store = None
        source_kind = None

        # 1) lat/lon
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

        # 2) y/x fallback
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

                    if y_arr.ndim == 1 and x_arr.ndim == 1:
                        if y_arr.shape[0] != self.H or x_arr.shape[0] != self.W:
                            raise ValueError(
                                f"y/x are 1D but shapes are y={y_arr.shape}, x={x_arr.shape}; "
                                f"expected y=({self.H},), x=({self.W},)"
                            )
                        lat = np.repeat(y_arr[:, None], self.W, axis=1)
                        lon = np.repeat(x_arr[None, :], self.H, axis=0)
                    else:
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
                f"Tried roots={list(roots)}.\n"
                "Need BOTH lat+lon (or y+x) in one store in a supported shape."
            )

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

        self._resolve_coords_units()
        self._check_coords_sanity(self.lat_grid, self.lon_grid)

        
    def _make_coord_features_patch(self, y0: int, x0: int) -> np.ndarray:
        
        lat_src = self.lat_grid_t if hasattr(self, "lat_grid_t") and self.lat_grid_t is not None else self.lat_grid
        lon_src = self.lon_grid_t if hasattr(self, "lon_grid_t") and self.lon_grid_t is not None else self.lon_grid

        if lat_src is None or lon_src is None:
            raise RuntimeError("coords requested but coordinate grids are not loaded")

        lat_patch = lat_src[y0:y0 + self.patch, x0:x0 + self.patch]
        lon_patch = lon_src[y0:y0 + self.patch, x0:x0 + self.patch]

        # If shared tensors exist, slices may be torch.Tensor → convert to numpy
        if torch.is_tensor(lat_patch):
            lat_patch = lat_patch.cpu().numpy()
        if torch.is_tensor(lon_patch):
            lon_patch = lon_patch.cpu().numpy()

        lat_patch = np.asarray(lat_patch, dtype=np.float32)
        lon_patch = np.asarray(lon_patch, dtype=np.float32)

        if self.coords_are_degrees is None:
            self._resolve_coords_units()

        if self.coords_are_degrees:
            lat_rad = np.deg2rad(lat_patch)
            lon_rad = np.deg2rad(lon_patch)
        else:
            lat_rad = lat_patch
            lon_rad = lon_patch


        feats = np.stack(
            [np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)],
            axis=0
        ).astype(np.float32)
        return feats

    def _build_index(self):
        """Build index of (t_end, y0, x0). Supports optional holdout region logic."""
        t_end_first = self.t_hist - 1
        t_end_last = self.T - 1 - self.max_horizon
        if t_end_last < t_end_first:
            raise ValueError(
                f"Not enough time steps T={self.T} for t_hist={self.t_hist} and max_horizon={self.max_horizon}"
            )

        t_ends = list(range(t_end_first, t_end_last + 1))

        if self.t_end_index is not None:
            allowed_set = set(self.t_end_index)
            t_ends = [t for t in t_ends if t in allowed_set]

        # Optional time filtering: windows fully contained in allowed indices
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
                if allowed[t0:t1 + 1].all():
                    filtered.append(t_end)
            t_ends = filtered

        spatial = _compute_grid_positions(self.H, self.W, self.patch, self.stride)

        # Peat-only spatial filtering
        peat_src = None
        if self.skip_nonpeat_patches:
            if self.peat_mask is not None:
                peat_src = self.peat_mask
            elif hasattr(self, "peat_mask_t") and self.peat_mask_t is not None:
                peat_src = self.peat_mask_t

        if self.skip_nonpeat_patches and peat_src is not None:
            valid_spatial: List[Tuple[int, int]] = []
            area = float(self.patch * self.patch)
            for (y0, x0) in spatial:
                patch_mask = peat_src[y0:y0 + self.patch, x0:x0 + self.patch]
                if torch.is_tensor(patch_mask):
                    patch_mask = patch_mask.cpu().numpy()
                # assume peat mask is 0/1 or 0/255; treat >0 as peat
                frac_peat = float((patch_mask > 0).sum()) / max(area, 1.0)
                if frac_peat >= self.peat_min_fraction:
                    valid_spatial.append((y0, x0))
            spatial = valid_spatial
            if not spatial:
                raise RuntimeError(
                    "After peat_mask-based filtering, no valid spatial patches remain. "
                    "Lower peat_min_fraction or disable skip_nonpeat_patches."
                )

        if not t_ends:
            raise RuntimeError("No valid t_end indices after filtering.")

        # --- Holdout handling (deterministic test = holdout region) ---
        holdout_enabled = (self.holdout_region_mask is not None) and (self.holdout_t_end_index is not None)
        if holdout_enabled:
            holdout_set = set(self.holdout_t_end_index)

            # spatial positions that overlap holdout region
            region_pos = set()
            area = float(self.patch * self.patch)
            for (y0, x0) in spatial:
                holdout_src = self.holdout_region_mask
                if holdout_src is None and hasattr(self, "holdout_region_mask_t") and self.holdout_region_mask_t is not None:
                    holdout_src = self.holdout_region_mask_t

                m = holdout_src[y0:y0 + self.patch, x0:x0 + self.patch]
                if torch.is_tensor(m):
                    frac = float(m.sum().item()) / max(area, 1.0)
                else:
                    frac = float(m.sum()) / max(area, 1.0)

                if frac >= self.holdout_min_fraction:
                    region_pos.add((y0, x0))

            if not region_pos:
                raise RuntimeError(
                    "holdout_region_source provided but no spatial patches overlap the holdout region "
                    f"(holdout_min_fraction={self.holdout_min_fraction})."
                )

            test_candidates: List[Tuple[int, int, int]] = []
            trainval_candidates: List[Tuple[int, int, int]] = []

            for t_end in t_ends:
                if t_end in holdout_set:
                    # test is ONLY the region patches for those anchors
                    for (y0, x0) in region_pos:
                        test_candidates.append((t_end, y0, x0))
                    # train/val exclude region patches for those anchors
                    for (y0, x0) in spatial:
                        if (y0, x0) not in region_pos:
                            trainval_candidates.append((t_end, y0, x0))
                else:
                    # non-holdout anchors go to train/val (all spatial)
                    for (y0, x0) in spatial:
                        trainval_candidates.append((t_end, y0, x0))

            rng = random.Random(self.seed)
            rng.shuffle(trainval_candidates)
            rng.shuffle(test_candidates)

            # In holdout mode, val_frac is ambiguous (since test is already defined).
            # We use a simple train/val split: train_frac=self.split, val=rest.
            if self.val_frac is not None and abs((1.0 - self.split) - self.val_frac) > 1e-6:
                print("[warn] Holdout mode: ignoring val_frac and using val=1-split (test is holdout-defined).")

            n_total = len(trainval_candidates)
            split_at = int(n_total * self.split)
            train_idx = trainval_candidates[:split_at]
            val_idx = trainval_candidates[split_at:]

            if self.mode == "train":
                chosen = train_idx
            elif self.mode == "val":
                chosen = val_idx
            elif self.mode == "test":
                chosen = test_candidates
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

        else:
            # --- Standard random split (2-way or 3-way) ---
            full_index: List[Tuple[int, int, int]] = [(t_end, y0, x0) for t_end in t_ends for (y0, x0) in spatial]
            if not full_index:
                raise RuntimeError("No valid sampling positions derived.")

            rng = random.Random(self.seed)
            rng.shuffle(full_index)

            n_total = len(full_index)
            train_frac = self.split
            val_frac = self.val_frac

            if val_frac is None:
                split_at = int(n_total * train_frac)
                train_idx = full_index[:split_at]
                val_idx = full_index[split_at:]
                test_idx: List[Tuple[int, int, int]] = []
            else:
                n_train = int(n_total * train_frac)
                n_val = int(n_total * val_frac)
                # remainder goes to test
                train_idx = full_index[:n_train]
                val_idx = full_index[n_train:n_train + n_val]
                test_idx = full_index[n_train + n_val:]

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

        if len(self.index) == 0:
            raise RuntimeError(f"Index is empty for mode={self.mode}. Check your filters/splits/holdout settings.")

    # --------------------------------------------------------
    # Dataset API
    # --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        
        if self.input_arrs is None or self.viirs_arr is None:
            self._reopen_zarr_handles()

        
        t_end, y0, x0 = self.index[i]
        t0 = t_end - (self.t_hist - 1)

        # -------------------------
        # Inputs
        # -------------------------
        io_times_inputs: List[float] = []
        parts: List[np.ndarray] = []

        for arr in self.input_arrs:
            t_io = time.time()
            slab = arr.get_orthogonal_selection(
                (
                    slice(t0, t_end + 1),          # T
                    slice(None),                   # C
                    slice(y0, y0 + self.patch),    # H
                    slice(x0, x0 + self.patch),    # W
                )
            )
            io_times_inputs.append(time.time() - t_io)

            np_slab = np.asarray(slab, dtype=np.float32)
            np_slab = np.nan_to_num(np_slab, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(np_slab)

        x_np = np.concatenate(parts, axis=1)  # (T, sumC, patch, patch)

        if self.coord_as_features:
            coord_feats = self._make_coord_features_patch(y0, x0)                    # (4, patch, patch)
            coord_feats_t = np.repeat(coord_feats[None, ...], self.t_hist, axis=0)  # (T, 4, patch, patch)
            x_np = np.concatenate([x_np, coord_feats_t], axis=1)

        x = torch.from_numpy(x_np)

        if self.time_stack == "channel":
            x = x.permute(1, 0, 2, 3).reshape(self.C_total * self.t_hist, self.patch, self.patch)

        # -------------------------
        # Targets (VIIRS)
        # -------------------------
        K = len(self.horizons)
        y_np = np.zeros((K, self.patch, self.patch), dtype=np.float32)
        mask_np = np.zeros((K, self.patch, self.patch), dtype=np.float32)

        peat_patch = None
        if hasattr(self, "peat_mask_t") and self.peat_mask_t is not None:
            peat_patch = (self.peat_mask_t[y0:y0+self.patch, x0:x0+self.patch] > 0).numpy()
        elif self.peat_mask is not None:
            peat_patch = (self.peat_mask[y0:y0+self.patch, x0:x0+self.patch] > 0)


        io_time_viirs_total = 0.0

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
            io_time_viirs_total += (time.time() - t_io)

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

        # -------------------------
        # Normalize inputs (optional)
        # -------------------------
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

        # -------------------------
        # Coordinates output (optional)
        # -------------------------
        coords = None
        if self.return_coords:
            has_np = (self.lat_grid is not None and self.lon_grid is not None)
            has_t  = (
                hasattr(self, "lat_grid_t") and self.lat_grid_t is not None and
                hasattr(self, "lon_grid_t") and self.lon_grid_t is not None
            )
            if not (has_np or has_t):
                raise RuntimeError("return_coords=True but lat/lon grids are not loaded.")

            lat_src = self.lat_grid_t if hasattr(self, "lat_grid_t") and self.lat_grid_t is not None else self.lat_grid
            lon_src = self.lon_grid_t if hasattr(self, "lon_grid_t") and self.lon_grid_t is not None else self.lon_grid
            if lat_src is None or lon_src is None:
                raise RuntimeError("return_coords=True but lat/lon grids are not loaded.")


            lat_patch = lat_src[y0:y0 + self.patch, x0:x0 + self.patch]
            lon_patch = lon_src[y0:y0 + self.patch, x0:x0 + self.patch]

            # ensure numpy for np.deg2rad + stacking
            if torch.is_tensor(lat_patch):
                lat_patch = lat_patch.cpu().numpy()
            if torch.is_tensor(lon_patch):
                lon_patch = lon_patch.cpu().numpy()

            if self.coords_are_degrees is None:
                self._resolve_coords_units()

            latf = lat_patch.astype(np.float32)
            lonf = lon_patch.astype(np.float32)
            if self.coords_are_degrees:
                latf = np.deg2rad(latf)
                lonf = np.deg2rad(lonf)

            coords_np = np.stack([latf, lonf], axis=0)

            coords = torch.from_numpy(coords_np)

        meta = {
            "t_end": int(t_end),
            "t_start": int(t0),
            "y0": int(y0),
            "x0": int(x0),
            "horizons": list(self.horizons),
            "viirs_array_path": self.viirs_path,
            "input_array_paths": list(self.input_paths),
            "io_time_inputs": [float(t) for t in io_times_inputs],
            "io_time_inputs_total": float(sum(io_times_inputs)),
            "io_time_viirs": float(io_time_viirs_total),
            "io_time_total": float(sum(io_times_inputs) + io_time_viirs_total),
        }
        if self.time_index is not None:
            meta["history_time_steps"] = list(range(int(t0), int(t_end) + 1))

        sample = {"x": x, "y": y, "mask": mask, "meta": meta}
        if self.return_coords:
            sample["coords"] = coords

        return sample

    # --------------------------------------------------------
    # Normalization helpers
    # --------------------------------------------------------

    def _estimate_input_stats(self, sample_count: int = 256):
        
        if self.input_arrs is None or self.viirs_arr is None:
            self._reopen_zarr_handles()

        
        """Estimate per-channel mean/std over inputs (and coord features if enabled)."""
        if len(self.index) == 0:
            raise RuntimeError("Cannot estimate input stats with an empty index.")

        rng = random.Random(self.seed + 123)
        n_samples = min(sample_count, len(self.index))
        idxs = rng.sample(range(len(self.index)), n_samples)

        sum_c = torch.zeros(self.C_total, dtype=torch.float64)
        sqsum_c = torch.zeros(self.C_total, dtype=torch.float64)
        total_count = 0  # total number of pixels aggregated per channel (includes time)

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

            # accumulate
            sum_c += x64.sum(dim=(0, 2, 3))
            sqsum_c += (x64 * x64).sum(dim=(0, 2, 3))
            total_count += x64.shape[0] * x64.shape[2] * x64.shape[3]

        if total_count <= 0:
            raise RuntimeError("Normalization estimation produced zero total_count.")

        mean = (sum_c / float(total_count)).to(torch.float32)
        var = (sqsum_c / float(total_count)) - (mean.double() ** 2)
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
                f"mean/std must have shape ({self.C_total},), got mean.shape={mean.shape}, std.shape={std.shape}"
            )

        self._mean = mean
        self._std = std
        self.normalize_inputs = "per_channel"


if __name__ == "__main__":
    pass
