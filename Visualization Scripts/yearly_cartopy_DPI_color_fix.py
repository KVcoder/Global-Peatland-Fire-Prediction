#!/usr/bin/env python3
"""
deeppeat_all_in_one.py

Single, independent script that *contains* BOTH:
  (1) DeepPeat ensemble XGBoost inference (nowcast probability + VIIRS overlay + optional FWI + optional calibration)
  (2) Yearly/Daily Cartopy visualization + capture curves + confusion maps + ROC/AUC from daily GeoTIFFs

It does NOT shell out to other scripts. Everything is implemented in this file.

USAGE
-----
A) Run inference (produces pred_tXXXXX.png and optionally pred_tXXXXX.tif + fwi_tXXXXX.png):
    python deeppeat_all_in_one.py infer \
      --input era5.zarr:field --input smap.zarr:field \
      --viirs-zarr viirs.zarr:field \
      --fwi-zarr fwi.zarr:field \
      --model-dir /path/to/model_dir \
      --regime-map regime.tif --cluster-map cluster.tif \
      --output-dir outputs_2023 \
      --t-hist 3 --test-year 2023 --save-geotiff

B) Run yearly/daily viz from the GeoTIFFs created above:
    python deeppeat_all_in_one.py viz \
      --pred-dir outputs_2023 \
      --out-dir yearly_viz_2023 \
      --figures raw percentile topk \
      --top-fracs 0.01 0.05 \
      --peat-mask-zarr smap_wtd.zarr:smap_wtd --peat-nodata -9999 --peat-mask-mode valid \
      --viirs-zarr viirs.zarr:field --viirs-threshold 0.5 \
      --fwi-zarr fwi.zarr:field \
      --wrap-longitude auto \
      --plot-regions \
      --make-capture-curves \
      --make-confusion-maps \
      --make-roc --roc-include-regions

Notes
-----
- Requires: numpy, xgboost, zarr, rasterio, matplotlib, cartopy, tqdm
- Matplotlib uses Agg backend (headless-safe).
"""

from __future__ import annotations

import os
import sys
import re
import json
import glob
import csv
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import zarr
import rasterio
from rasterio.transform import from_bounds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from tqdm import tqdm

# XGBoost is only needed for "infer" mode; import here for one-file convenience
import xgboost as xgb


# =============================================================================
# Shared constants
# =============================================================================
NODATA_VALUE = -9999
VIIRS_FIRE_THRESHOLD_DEFAULT = 0.5

# Calibration numerical safety
EPS_PROB = 1e-6

# ECMWF-style discrete palettes for inference maps
ECMWF5_COLORS = ["#dce0e0", "#88c0d0", "#f4e064", "#ecb06c", "#ec7474"]

# Fire probability bins in FRACTION units (0.001 = 0.1%)
PROB_BOUNDS = [0.0, 0.001, 0.002, 0.004, 0.008, 1.0]
PROB_LABELS = ["<0.1%", "0.1–0.2%", "0.2–0.4%", "0.4–0.8%", ">0.8%"]

# FWI display scaling (inference viz only)
FWI_DISPLAY_DIVISOR = 10.0
FWI_VMIN = 0.0
FWI_VMAX = 10.0
FWI_BOUNDS = [0.0, 1.0, 2.0, 4.0, 8.0, 1e9]
FWI_LABELS = ["<1", "1–2", "2–4", "4–8", ">8"]

DEFAULT_T_HIST = 3


# =============================================================================
# Inference: Optional import from existing codebase (with fallback)
# =============================================================================
sys.path.append(os.path.join(os.path.dirname(__file__), "TRAIN_REGIMES_THE_NEW_CLUSTERS"))
try:
    from joint_peat_dataset_builder_cluster import _open_zarr_array as _open_zarr_array_external
    from joint_peat_dataset_builder_cluster import parse_input_spec as parse_input_spec_external
    from joint_peat_dataset_builder_cluster import InputSpec as InputSpec_external
    print("[Import] Successfully imported from joint_peat_dataset_builder_cluster")
except Exception:
    _open_zarr_array_external = None
    parse_input_spec_external = None
    InputSpec_external = None
    print("[Warning] Could not import from joint_peat_dataset_builder_cluster; using fallback zarr helpers.")


@dataclass(frozen=True)
class InputSpec_fallback:
    zarr: str
    array: str = "field"


def parse_input_spec_fallback(s: str) -> InputSpec_fallback:
    s = str(s)
    if ":" in s:
        z, a = s.split(":", 1)
        z, a = z.strip(), a.strip()
        if not a:
            a = "field"
        return InputSpec_fallback(zarr=z, array=a)
    return InputSpec_fallback(zarr=s.strip(), array="field")


def _open_zarr_array_fallback(zarr_path: str, array_name: str = "field"):
    if ":" in zarr_path:
        zarr_path, array_name = zarr_path.split(":", 1)

    root = zarr.open_group(zarr_path, mode="r")

    if array_name in root:
        return root[array_name], root

    for key in root.array_keys():
        print(f"[Zarr] Using array '{key}' from {zarr_path}")
        return root[key], root

    raise ValueError(f"No array '{array_name}' found in {zarr_path}")


_open_zarr_array = _open_zarr_array_external or _open_zarr_array_fallback
parse_input_spec = parse_input_spec_external or parse_input_spec_fallback
InputSpec = InputSpec_external or InputSpec_fallback


# =============================================================================
# Inference: Calibration (Platt / Isotonic) on MARGIN space
# =============================================================================
def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class PlattCalibratorNP:
    a: float
    b: float

    def apply_logits(self, margin: np.ndarray) -> np.ndarray:
        return margin * float(self.a) + float(self.b)


@dataclass(frozen=True)
class IsotonicCalibratorNP:
    x_thresholds: np.ndarray  # (M,)
    y_values: np.ndarray      # (M,)

    def apply_logits(self, margin: np.ndarray) -> np.ndarray:
        m = margin.astype(np.float64, copy=False)
        idx = np.searchsorted(self.x_thresholds, m, side="right") - 1
        idx = np.clip(idx, 0, self.y_values.size - 1)
        p = self.y_values[idx]
        p = np.clip(p, EPS_PROB, 1.0 - EPS_PROB)
        return (np.log(p) - np.log(1.0 - p)).astype(np.float32)


@dataclass(frozen=True)
class CalibrationBundle:
    method: str
    per_regime: bool
    global_by_h: Dict[int, Any]
    pair_by_rh: Dict[Tuple[int, int], Any]


def _calib_from_dict(d: dict):
    t = str(d.get("type", "")).lower()
    if t == "platt":
        return PlattCalibratorNP(a=float(d["a"]), b=float(d["b"]))
    if t == "isotonic":
        return IsotonicCalibratorNP(
            x_thresholds=np.asarray(d["x_thresholds"], dtype=np.float64),
            y_values=np.asarray(d["y_values"], dtype=np.float64),
        )
    raise ValueError(f"Unknown calibrator type in JSON: {t}")


def load_calibrators_json(path: str) -> Optional[CalibrationBundle]:
    """
    Supported structures:
      (A) {"method": "...", "per_horizon": {"0": {...}, "1": {...}}}
      (B) {"method": "...",
           "per_regime_horizon": {"3": {"0": {...}}, ...},
           "global_fallback": {"0": {...}, ...}}
    """
    if path is None or (not os.path.exists(path)):
        return None

    with open(path, "r") as f:
        payload = json.load(f)

    method = str(payload.get("method", "none")).lower()
    if method in ("none", "", "null"):
        return None

    if "per_horizon" in payload:
        per_h = {int(h): _calib_from_dict(cd) for h, cd in payload["per_horizon"].items()}
        return CalibrationBundle(method=method, per_regime=False, global_by_h=per_h, pair_by_rh={})

    if "per_regime_horizon" in payload:
        global_by_h = {int(h): _calib_from_dict(cd) for h, cd in (payload.get("global_fallback", {}) or {}).items()}
        pair: Dict[Tuple[int, int], Any] = {}
        for r_str, per_h in (payload.get("per_regime_horizon", {}) or {}).items():
            r = int(r_str)
            for h_str, cd in (per_h or {}).items():
                pair[(r, int(h_str))] = _calib_from_dict(cd)
        return CalibrationBundle(method=method, per_regime=True, global_by_h=global_by_h, pair_by_rh=pair)

    print(f"[Calib] Warning: Unrecognized calibrators.json structure at {path}")
    return None


def apply_calibration_to_logits(
    logits: np.ndarray,
    horizon: int,
    bundle: Optional[CalibrationBundle],
    regime_ids: Optional[np.ndarray] = None,
    regime_nodata_value: int = NODATA_VALUE,
) -> np.ndarray:
    if bundle is None:
        return logits

    h = int(horizon)
    calib_global = bundle.global_by_h.get(h, None)

    if not bundle.per_regime:
        return calib_global.apply_logits(logits) if calib_global is not None else logits

    if regime_ids is None:
        return calib_global.apply_logits(logits) if calib_global is not None else logits

    out = logits.copy()

    # apply global fallback first
    if calib_global is not None:
        out = calib_global.apply_logits(out)

    # override per-regime where available (using ORIGINAL logits for those pixels)
    uniq = np.unique(regime_ids)
    for r in uniq:
        r_int = int(r)
        if r_int == int(regime_nodata_value):
            continue
        calib = bundle.pair_by_rh.get((r_int, h), None)
        if calib is None:
            continue
        idx = np.nonzero(regime_ids == r)[0]
        if idx.size == 0:
            continue
        out[idx] = calib.apply_logits(logits[idx])

    return out


def _calib_debug_stats(prefix: str, margins: np.ndarray, probs: np.ndarray):
    m = margins.astype(np.float64, copy=False)
    p = probs.astype(np.float64, copy=False)

    def pct(x, q):
        return float(np.percentile(x, q))

    msg = (
        f"[CalibDebug] {prefix} | "
        f"margin(mean={float(m.mean()):.4f}, p50={pct(m,50):.4f}, p90={pct(m,90):.4f}, p99={pct(m,99):.4f}) | "
        f"prob(mean={float(p.mean()):.6f}, p50={pct(p,50):.6f}, p90={pct(p,90):.6f}, p99={pct(p,99):.6f}, max={float(p.max()):.6f}) | "
        f"frac(p>=0.5)={float((p>=0.5).mean()):.6f}, frac(p>=0.9)={float((p>=0.9).mean()):.6f}"
    )
    print(msg)


# =============================================================================
# Inference: Model loading
# =============================================================================
def load_global_model(model_dir: str, horizon: int = 0) -> xgb.Booster:
    model_path = os.path.join(model_dir, f"xgb_h{horizon}.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Global model not found: {model_path}")
    bst = xgb.Booster()
    bst.load_model(model_path)
    print(f"[Load] Global model: {model_path}")
    return bst


def load_cluster_models(model_dir: str, horizon: int = 0) -> Dict[int, xgb.Booster]:
    cluster_dir = os.path.join(model_dir, "cluster_models")
    if not os.path.exists(cluster_dir):
        print(f"[Load] No cluster models found at {cluster_dir}")
        return {}

    cluster_models: Dict[int, xgb.Booster] = {}
    for entry in os.listdir(cluster_dir):
        if not entry.startswith("cluster"):
            continue
        try:
            cluster_id = int(entry.replace("cluster", ""))
        except ValueError:
            continue

        model_path = os.path.join(cluster_dir, entry, f"xgb_h{horizon}.json")
        if not os.path.exists(model_path):
            print(f"[Load] Warning: Missing cluster {cluster_id} model: {model_path}")
            continue

        bst = xgb.Booster()
        bst.load_model(model_path)
        cluster_models[cluster_id] = bst

    print(f"[Load] Loaded {len(cluster_models)} cluster models")
    return cluster_models


def load_residual_models(model_dir: str, horizon: int = 0) -> Tuple[Dict[int, xgb.Booster], List[str]]:
    residual_dir = os.path.join(model_dir, "residual_models")
    if not os.path.exists(residual_dir):
        print(f"[Load] No residual models found at {residual_dir}")
        return {}, []

    residual_models: Dict[int, xgb.Booster] = {}
    for entry in os.listdir(residual_dir):
        if not entry.startswith("regime"):
            continue
        try:
            regime_id = int(entry.replace("regime", ""))
        except ValueError:
            continue

        model_path = os.path.join(residual_dir, entry, f"xgb_h{horizon}.json")
        if not os.path.exists(model_path):
            print(f"[Load] Warning: Missing regime {regime_id} model: {model_path}")
            continue

        bst = xgb.Booster()
        bst.load_model(model_path)
        residual_models[regime_id] = bst

    feature_names: List[str] = []
    fn_path = os.path.join(residual_dir, "feature_names.json")
    if os.path.exists(fn_path):
        with open(fn_path, "r") as f:
            feature_names = json.load(f)

    print(f"[Load] Loaded {len(residual_models)} residual models")
    if feature_names:
        print(f"[Load] Augmented features: {len(feature_names)} dims")

    return residual_models, feature_names


# =============================================================================
# Inference: Aux data loading
# =============================================================================
def init_zarr_stores(input_specs: List, viirs_spec, peat_mask_source: str, fwi_spec=None):
    print("[Init] Opening Zarr stores...")
    print(f"[Init] Number of input sources: {len(input_specs)}")

    input_arrs = []
    T_ref, H_ref, W_ref = None, None, None
    total_channels = 0

    for i, spec in enumerate(input_specs):
        print(f"  Input {i+1}: {spec.zarr}:{spec.array}")
        arr, _ = _open_zarr_array(spec.zarr, spec.array)
        T, C, H, W = arr.shape
        if T_ref is None:
            T_ref, H_ref, W_ref = T, H, W
        else:
            assert T == T_ref and H == H_ref and W == W_ref, \
                f"Shape mismatch in input {i}: ({T},{H},{W}) != ({T_ref},{H_ref},{W_ref})"
        total_channels += C
        input_arrs.append(arr)
        print(f"    Shape: ({T}, {C}, {H}, {W})")

    print(f"  VIIRS: {viirs_spec.zarr}:{viirs_spec.array}")
    viirs_arr, _ = _open_zarr_array(viirs_spec.zarr, viirs_spec.array)
    T_v, C_v, H_v, W_v = viirs_arr.shape
    assert T_v == T_ref and H_v == H_ref and W_v == W_ref, \
        f"VIIRS shape mismatch: ({T_v},{H_v},{W_v}) != ({T_ref},{H_ref},{W_ref})"
    print(f"    Shape: ({T_v}, {C_v}, {H_v}, {W_v})")

    fwi_arr = None
    if fwi_spec is not None:
        print(f"  FWI: {fwi_spec.zarr}:{fwi_spec.array}")
        fwi_arr, _ = _open_zarr_array(fwi_spec.zarr, fwi_spec.array)
        T_f, C_f, H_f, W_f = fwi_arr.shape
        assert H_f == H_ref and W_f == W_ref, \
            f"FWI spatial shape mismatch: (H,W)=({H_f},{W_f}) != ({H_ref},{W_ref})"
        assert C_f == 1, f"FWI must have 1 channel, got C={C_f}"
        print(f"    Shape: ({T_f}, {C_f}, {H_f}, {W_f})  [T may differ; will align by time]")

    stores = {
        "input_arrs": input_arrs,
        "viirs": viirs_arr,
        "fwi": fwi_arr,
        "peat_mask_source": peat_mask_source,
        "shape": (T_ref, H_ref, W_ref),
        "n_channels": total_channels
    }

    print(f"[Init] Total: T={T_ref}, H={H_ref}, W={W_ref}, Channels={total_channels}")
    if fwi_arr is not None:
        print("[Init] FWI: enabled (time-aligned)")
    return stores


def load_peat_mask(smap_zarr_path: str, H: int, W: int) -> np.ndarray:
    if smap_zarr_path is None:
        print("[Warning] peat_mask_source is None; using all pixels as peat (no masking)")
        return np.ones((H, W), dtype=np.uint8)

    try:
        root = zarr.open_group(smap_zarr_path, mode="r")
        if "peat_mask" in root:
            peat_mask = np.asarray(root["peat_mask"], dtype=np.uint8)
            assert peat_mask.shape == (H, W), f"Peat mask shape mismatch: {peat_mask.shape} != ({H},{W})"
            print(f"[Load] Peat mask: {peat_mask.sum():,} / {H*W:,} pixels ({100.0*peat_mask.sum()/(H*W):.2f}%)")
            return peat_mask
    except Exception as e:
        print(f"[Warning] Could not load peat_mask from store: {e}")

    print("[Warning] Using all pixels as peat (no masking)")
    return np.ones((H, W), dtype=np.uint8)


def load_coordinates(zarr_path: str, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    if zarr_path is None:
        raise ValueError("coords_source is None but coord features/maps require coordinates")

    root = zarr.open_group(zarr_path, mode="r")

    for lat_name in ["lat", "latitude"]:
        for lon_name in ["lon", "longitude"]:
            if lat_name in root and lon_name in root:
                lat = np.asarray(root[lat_name], dtype=np.float32)
                lon = np.asarray(root[lon_name], dtype=np.float32)

                if lat.ndim == 1 and lon.ndim == 1:
                    if len(lat) == H and len(lon) == W:
                        lon_grid, lat_grid = np.meshgrid(lon, lat)
                        print(f"[Load] Coordinates from {lat_name}/{lon_name} (1D->2D)")
                        return lat_grid, lon_grid
                elif lat.ndim == 2 and lon.ndim == 2:
                    if lat.shape == (H, W) and lon.shape == (H, W):
                        print(f"[Load] Coordinates from {lat_name}/{lon_name} (2D)")
                        return lat, lon

    if "y" in root and "x" in root:
        y = np.asarray(root["y"], dtype=np.float32)
        x = np.asarray(root["x"], dtype=np.float32)

        if y.ndim == 1 and x.ndim == 1:
            if len(y) == H and len(x) == W:
                lon_grid, lat_grid = np.meshgrid(x, y)
                print(f"[Load] Coordinates from y/x (1D->2D)")
                return lat_grid, lon_grid
        elif y.ndim == 2 and x.ndim == 2:
            if y.shape == (H, W) and x.shape == (H, W):
                print(f"[Load] Coordinates from y/x (2D)")
                return y, x

    raise ValueError("Could not find coordinate grids (lat/lon or y/x) in Zarr store")


def load_spatial_map(geotiff_path: str, expected_shape: Tuple[int, int], name: str = "map") -> np.ndarray:
    if not os.path.exists(geotiff_path):
        print(f"[Warning] {name} not found: {geotiff_path}")
        H, W = expected_shape
        return np.full((H, W), NODATA_VALUE, dtype=np.int32)

    with rasterio.open(geotiff_path) as src:
        data = src.read(1).astype(np.int32)

    assert data.shape == expected_shape, f"{name} shape mismatch: {data.shape} != {expected_shape}"

    unique_ids = np.unique(data[data != NODATA_VALUE])
    print(f"[Load] {name}: {geotiff_path} ({len(unique_ids)} unique IDs)")
    return data


# =============================================================================
# Inference: Feature engineering
# =============================================================================
def extract_features_for_time(
    zarr_stores: dict,
    t_end: int,
    t_hist: int = 30,
    peat_mask: Optional[np.ndarray] = None,
    coord_as_features: bool = True,
    lat_grid: Optional[np.ndarray] = None,
    lon_grid: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    t_start = t_end - (t_hist - 1)

    T_total = zarr_stores["shape"][0]
    if t_start < 0:
        raise ValueError(f"Insufficient history: t_end={t_end}, t_hist={t_hist}, need t_start >= 0")
    if t_end >= T_total:
        raise ValueError(f"t_end={t_end} exceeds time range [0, {T_total-1}]")

    input_slices = []
    for arr in zarr_stores["input_arrs"]:
        data = arr[t_start:t_end+1, :, :, :]
        data_np = np.nan_to_num(np.asarray(data, dtype=np.float32),
                                nan=0.0, posinf=0.0, neginf=0.0)
        input_slices.append(data_np)

    X_tcwh = np.concatenate(input_slices, axis=1)
    T, C, H, W = X_tcwh.shape

    X_fhw = X_tcwh.transpose(1, 0, 2, 3).reshape(C * T, H, W)
    X_all = X_fhw.transpose(1, 2, 0).reshape(H * W, -1)

    if peat_mask is not None:
        peat_flat = peat_mask.ravel()
        valid_mask = (peat_flat > 0)
        X = X_all[valid_mask]
        valid_pixels = np.where(valid_mask)[0]
    else:
        valid_mask = None
        X = X_all
        valid_pixels = np.arange(H * W)

    if coord_as_features and lat_grid is not None and lon_grid is not None:
        lat_rad = np.deg2rad(lat_grid).ravel()
        lon_rad = np.deg2rad(lon_grid).ravel()

        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)
        cos_lon = np.cos(lon_rad)
        sin_lon = np.sin(lon_rad)

        x = cos_lat * cos_lon
        y = cos_lat * sin_lon
        z = sin_lat

        feats = np.stack([x, y, z], axis=1).astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        if peat_mask is not None:
            feats = feats[valid_mask]

        X = np.concatenate([X, feats], axis=1)
        print("[Features] Added 3 Cartesian coord features (x,y,z)")

    print(f"[Features] t={t_end}, shape={X.shape} ({len(valid_pixels):,} peat pixels, {X.shape[1]} features)")
    return X, valid_pixels, (H, W)


def augment_features_for_residual(X_base: np.ndarray, global_probs: np.ndarray) -> np.ndarray:
    return np.column_stack([X_base, global_probs])


def align_features_to_model(X: np.ndarray, model: xgb.Booster, name: str = "model") -> np.ndarray:
    n_model = model.num_features()
    n_x = X.shape[1]
    if n_x == n_model:
        return X
    if n_x > n_model:
        print(f"[Features] {name}: X has {n_x} features but model expects {n_model}. Dropping last {n_x - n_model}.")
        return X[:, :n_model]
    print(f"[Features] {name}: X has {n_x} features but model expects {n_model}. Padding {n_model - n_x} zeros.")
    pad = np.zeros((X.shape[0], n_model - n_x), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


# =============================================================================
# Inference: Ensemble prediction (margin space + optional calibration)
# =============================================================================
def predict_ensemble(
    X: np.ndarray,
    valid_pixels: np.ndarray,
    spatial_shape: Tuple[int, int],
    regime_map: np.ndarray,
    cluster_map: np.ndarray,
    global_model: xgb.Booster,
    cluster_models: Dict[int, xgb.Booster],
    residual_models: Dict[int, xgb.Booster],
    ensemble_method: str = "weighted",
    weights: Optional[Dict[str, float]] = None,
    horizon: int = 0,
    calib_bundle: Optional[CalibrationBundle] = None,
    calib_debug: bool = False,
    calib_debug_prefix: str = "",
) -> np.ndarray:
    H, W = spatial_shape
    N_pixels = len(valid_pixels)

    if weights is None:
        weights = {"global": 0.4, "cluster": 0.3, "residual": 0.3}

    # Stage 1: Global margin
    print(f"[Ensemble] Stage 1: Global model ({N_pixels:,} pixels)")
    X_global = align_features_to_model(X, global_model, name="global")
    dmat_global = xgb.DMatrix(X_global)
    global_margin = global_model.predict(dmat_global, output_margin=True).astype(np.float32, copy=False)

    global_probs = _sigmoid_np(global_margin).astype(np.float32, copy=False)

    # Stage 2: Cluster margins
    print(f"[Ensemble] Stage 2: Cluster models ({len(cluster_models)} clusters available)")
    cluster_margin = np.full(N_pixels, np.nan, dtype=np.float32)
    if cluster_models:
        cluster_ids_flat = cluster_map.ravel()[valid_pixels]
        unique_clusters = np.unique(cluster_ids_flat)
        unique_clusters = unique_clusters[unique_clusters >= 0]
        for cluster_id in unique_clusters:
            if cluster_id not in cluster_models:
                continue
            cluster_mask = (cluster_ids_flat == cluster_id)
            n_cluster_pixels = int(cluster_mask.sum())
            if n_cluster_pixels == 0:
                continue
            X_cluster = X[cluster_mask]
            X_cluster = align_features_to_model(X_cluster, cluster_models[cluster_id], name=f"cluster{cluster_id}")
            dmat = xgb.DMatrix(X_cluster)
            cluster_margin[cluster_mask] = cluster_models[cluster_id].predict(dmat, output_margin=True).astype(np.float32, copy=False)
            print(f"  Cluster {cluster_id}: {n_cluster_pixels:,} pixels")

    # Stage 3: Residual margins
    print(f"[Ensemble] Stage 3: Residual models ({len(residual_models)} regimes available)")
    residual_margin = np.full(N_pixels, np.nan, dtype=np.float32)
    if residual_models:
        regime_ids_flat = regime_map.ravel()[valid_pixels]
        unique_regimes = np.unique(regime_ids_flat)
        unique_regimes = unique_regimes[unique_regimes >= 0]
        X_aug = augment_features_for_residual(X, global_probs)

        for regime_id in unique_regimes:
            if regime_id not in residual_models:
                continue
            regime_mask = (regime_ids_flat == regime_id)
            n_regime_pixels = int(regime_mask.sum())
            if n_regime_pixels == 0:
                continue
            X_regime = X_aug[regime_mask]
            X_regime = align_features_to_model(X_regime, residual_models[regime_id], name=f"regime{regime_id}")
            dmat = xgb.DMatrix(X_regime)
            residual_margin[regime_mask] = residual_models[regime_id].predict(dmat, output_margin=True).astype(np.float32, copy=False)
            print(f"  Regime {regime_id}: {n_regime_pixels:,} pixels")

    # Combine in margin space
    print(f"[Ensemble] Combining margins (method={ensemble_method})")
    if ensemble_method == "weighted":
        w_global = float(weights.get("global", 0.0))
        w_cluster = float(weights.get("cluster", 0.0))
        w_resid = float(weights.get("residual", 0.0))

        pred_margin = w_global * global_margin
        total_w = np.full(N_pixels, w_global, dtype=np.float32)

        cluster_valid = ~np.isnan(cluster_margin)
        pred_margin[cluster_valid] += w_cluster * cluster_margin[cluster_valid]
        total_w[cluster_valid] += w_cluster

        resid_valid = ~np.isnan(residual_margin)
        pred_margin[resid_valid] += w_resid * residual_margin[resid_valid]
        total_w[resid_valid] += w_resid

        total_w = np.where(total_w > 0, total_w, 1.0)
        pred_margin = pred_margin / total_w

    elif ensemble_method == "cascade":
        pred_margin = global_margin.copy()
        cluster_valid = ~np.isnan(cluster_margin)
        pred_margin[cluster_valid] = cluster_margin[cluster_valid]
        resid_valid = ~np.isnan(residual_margin)
        pred_margin[resid_valid] = residual_margin[resid_valid]
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")

    # Debug BEFORE calibration
    probs_pre = None
    if calib_debug:
        probs_pre = _sigmoid_np(pred_margin)
        _calib_debug_stats(f"{calib_debug_prefix} pre-calib", pred_margin, probs_pre)

    # Calibration on combined margin
    if calib_bundle is not None:
        regime_ids_for_valid = None
        if calib_bundle.per_regime and regime_map is not None:
            regime_ids_for_valid = regime_map.ravel()[valid_pixels].astype(np.int32, copy=False)

        pred_margin = apply_calibration_to_logits(
            logits=pred_margin,
            horizon=int(horizon),
            bundle=calib_bundle,
            regime_ids=regime_ids_for_valid,
            regime_nodata_value=NODATA_VALUE,
        )

    # Debug AFTER calibration
    if calib_debug:
        probs_post = _sigmoid_np(pred_margin)
        _calib_debug_stats(f"{calib_debug_prefix} post-calib", pred_margin, probs_post)
        if probs_pre is not None:
            print(
                f"[CalibDebug] {calib_debug_prefix} delta | "
                f"mean_prob: {float(probs_post.mean() - probs_pre.mean()):+.6e}, "
                f"frac(p>=0.5): {float((probs_post>=0.5).mean() - (probs_pre>=0.5).mean()):+.6e}"
            )

    pred_probs = _sigmoid_np(pred_margin).astype(np.float32, copy=False)
    pred_probs = np.clip(pred_probs, 0.0, 1.0)

    pred_map = np.zeros((H, W), dtype=np.float32)
    pred_map.ravel()[valid_pixels] = pred_probs
    return pred_map


# =============================================================================
# Inference: Visualization helpers (ECMWF-like discrete bins)
# =============================================================================
def create_scalar_map(
    data_map: np.ndarray,
    viirs_map: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    peat_mask: np.ndarray,
    output_path: str,
    title: str,
    cbar_label: Optional[str],
    vmin: float,
    vmax: float,
    cmap_name: str = "RdYlGn_r",
    dpi: int = 300,
    add_features: bool = True,
    bounds: Optional[List[float]] = None,
    bin_labels: Optional[List[str]] = None,
    extend: str = "neither",
    viirs_fire_threshold: float = VIIRS_FIRE_THRESHOLD_DEFAULT,
):
    data_masked = np.where(peat_mask > 0, data_map, np.nan)

    lat_min, lat_max = float(lat_grid.min()), float(lat_grid.max())
    lon_min, lon_max = float(lon_grid.min()), float(lon_grid.max())
    extent = [lon_min, lon_max, lat_min, lat_max]

    print(f"[Viz] Extent: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")

    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    if add_features:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray", linestyle="--", zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--", zorder=3)
        gl.top_labels = False
        gl.right_labels = False

    if bounds is not None:
        cmap = ListedColormap(ECMWF5_COLORS, name="ecmwf5")
        cmap.set_bad(color="white", alpha=0)
        norm = BoundaryNorm(bounds, cmap.N, clip=False)

        im = ax.imshow(
            data_masked,
            origin="upper",
            extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            zorder=1,
        )

        mids = [(bounds[i] + bounds[i + 1]) / 2.0 for i in range(len(bounds) - 1)]
        cbar = plt.colorbar(
            im, ax=ax, orientation="vertical", pad=0.05, fraction=0.046, shrink=0.8,
            boundaries=bounds, ticks=mids, spacing="uniform", extend=extend
        )
        if bin_labels is not None and len(bin_labels) == (len(bounds) - 1):
            cbar.ax.set_yticklabels(bin_labels)
            cbar.ax.tick_params(length=0)
    else:
        cmap = plt.get_cmap(cmap_name)
        cmap.set_bad(color="white", alpha=0)
        im = ax.imshow(
            data_masked,
            origin="upper",
            extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            zorder=1,
        )
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.05, fraction=0.046, shrink=0.8)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=12, fontweight="bold")
        cbar.ax.tick_params(labelsize=10)

    # VIIRS overlay
    viirs_fires = (viirs_map > viirs_fire_threshold) & (peat_mask > 0)
    n_fires = int(viirs_fires.sum())
    if n_fires > 0:
        print(f"[Viz] Overlaying {n_fires} VIIRS fire detections as BLACK TRIANGLES (white outline)")
        rr, cc = np.where(viirs_fires)
        fire_lats = lat_grid[rr, cc]
        fire_lons = lon_grid[rr, cc]
        ax.scatter(
            fire_lons, fire_lats,
            transform=ccrs.PlateCarree(),
            marker="^",
            s=10,
            c="black",
            edgecolors="white",
            linewidths=0.35,
            zorder=4,
            rasterized=True,
        )
    else:
        print("[Viz] No VIIRS fires detected")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {output_path}")


def create_fire_probability_map(
    pred_map: np.ndarray,
    viirs_map: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    peat_mask: np.ndarray,
    output_path: str,
    title: str = "Fire Probability Nowcast",
    dpi: int = 300,
    add_features: bool = True,
    viirs_fire_threshold: float = VIIRS_FIRE_THRESHOLD_DEFAULT,
):
    return create_scalar_map(
        pred_map, viirs_map, lat_grid, lon_grid, peat_mask,
        output_path,
        title=title,
        cbar_label=None,
        vmin=0.0,
        vmax=1.0,
        dpi=dpi,
        add_features=add_features,
        bounds=PROB_BOUNDS,
        bin_labels=PROB_LABELS,
        extend="neither",
        viirs_fire_threshold=viirs_fire_threshold,
    )


def load_viirs_for_time(viirs_arr, t_idx: int, peat_mask: np.ndarray) -> np.ndarray:
    viirs_slice = viirs_arr[t_idx, 0, :, :]
    viirs_np = np.nan_to_num(np.asarray(viirs_slice, dtype=np.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)
    viirs_binary = (viirs_np > VIIRS_FIRE_THRESHOLD_DEFAULT).astype(np.float32)
    return viirs_binary * peat_mask


def load_fwi_for_time(fwi_arr, t_idx: int, peat_mask: np.ndarray) -> np.ndarray:
    fwi_slice = fwi_arr[t_idx, 0, :, :]
    fwi_np = np.nan_to_num(np.asarray(fwi_slice, dtype=np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0)
    return fwi_np * peat_mask


def save_geotiff(pred_map: np.ndarray, output_path: str, lat_grid: np.ndarray, lon_grid: np.ndarray):
    H, W = pred_map.shape
    lat_min, lat_max = float(lat_grid.min()), float(lat_grid.max())
    lon_min, lon_max = float(lon_grid.min()), float(lon_grid.max())
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, W, H)

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=H, width=W, count=1,
        dtype=rasterio.float32,
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
        nodata=NODATA_VALUE
    ) as dst:
        dst.write(pred_map.astype(np.float32), 1)

    print(f"[Output] GeoTIFF: {output_path}")


def compute_metrics(pred_map: np.ndarray, viirs_map: np.ndarray, peat_mask: np.ndarray, threshold: float = 0.5) -> dict:
    peat_pixels = (peat_mask > 0)
    fire_pixels = (viirs_map > VIIRS_FIRE_THRESHOLD_DEFAULT) & peat_pixels
    pred_binary = (pred_map >= threshold) & peat_pixels

    TP = int(np.sum(pred_binary & fire_pixels))
    FP = int(np.sum(pred_binary & ~fire_pixels))
    FN = int(np.sum(~pred_binary & fire_pixels))
    TN = int(np.sum(~pred_binary & ~fire_pixels & peat_pixels))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_peat_pixels": int(peat_pixels.sum()),
        "n_fire_pixels": int(fire_pixels.sum()),
        "mean_prob": float(pred_map[peat_pixels].mean()) if peat_pixels.any() else 0.0,
        "mean_prob_fire": float(pred_map[fire_pixels].mean()) if fire_pixels.any() else 0.0,
        "mean_prob_nofire": float(pred_map[~fire_pixels & peat_pixels].mean())
            if (~fire_pixels & peat_pixels).any() else 0.0,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "threshold": threshold
    }


# =============================================================================
# Inference: Time index utilities (CF-ish)
# =============================================================================
def load_time_years_from_viirs(viirs_zarr_path: str) -> np.ndarray:
    root = zarr.open_group(viirs_zarr_path, mode="r")
    if "time" not in root:
        raise ValueError(f"No 'time' coordinate found in {viirs_zarr_path}")

    time_coord = np.asarray(root["time"])
    attrs = dict(getattr(root["time"], "attrs", {}))
    units = attrs.get("units", "")

    def _parse_cf_units(u: str):
        m = re.match(r"(\w+)\s+since\s+(\d{4}-\d{2}-\d{2})(?:\s+(\d{2}:\d{2}:\d{2}))?", u.strip())
        if not m:
            return None
        unit = m.group(1).lower()
        date = m.group(2)
        tstr = m.group(3) or "00:00:00"
        base = np.datetime64(f"{date}T{tstr}")
        return unit, base

    if np.issubdtype(time_coord.dtype, np.datetime64):
        return time_coord.astype("datetime64[Y]").astype(int) + 1970

    if np.issubdtype(time_coord.dtype, np.integer):
        parsed = _parse_cf_units(units) if units else None
        if parsed is not None:
            unit, base = parsed
            if unit.startswith("day"):
                dt = base + time_coord.astype("timedelta64[D]")
            elif unit.startswith("hour"):
                dt = base + time_coord.astype("timedelta64[h]")
            elif unit.startswith("min"):
                dt = base + time_coord.astype("timedelta64[m]")
            elif unit.startswith("sec"):
                dt = base + time_coord.astype("timedelta64[s]")
            else:
                raise ValueError(f"Unsupported CF time unit: {unit}")
            return dt.astype("datetime64[Y]").astype(int) + 1970

        dt = np.datetime64("1970-01-01") + time_coord.astype("timedelta64[D]")
        return dt.astype("datetime64[Y]").astype(int) + 1970

    if np.issubdtype(time_coord.dtype, np.floating):
        dt = np.datetime64("1970-01-01") + time_coord.astype("timedelta64[D]")
        return dt.astype("datetime64[Y]").astype(int) + 1970

    raise ValueError(f"Unknown time coordinate dtype: {time_coord.dtype}")


def load_time_datetimes_from_zarr(zarr_path: str) -> np.ndarray:
    root = zarr.open_group(zarr_path, mode="r")
    if "time" not in root:
        raise ValueError(f"No 'time' coordinate found in {zarr_path}")

    time_coord = np.asarray(root["time"])
    attrs = dict(getattr(root["time"], "attrs", {}))
    units = attrs.get("units", "")

    def _parse_cf_units(u: str):
        m = re.match(r"(\w+)\s+since\s+(\d{4}-\d{2}-\d{2})(?:\s+(\d{2}:\d{2}:\d{2}))?", u.strip())
        if not m:
            return None
        unit = m.group(1).lower()
        date = m.group(2)
        tstr = m.group(3) or "00:00:00"
        base = np.datetime64(f"{date}T{tstr}")
        return unit, base

    if np.issubdtype(time_coord.dtype, np.datetime64):
        return time_coord.astype("datetime64[D]")

    if np.issubdtype(time_coord.dtype, np.integer) or np.issubdtype(time_coord.dtype, np.floating):
        parsed = _parse_cf_units(units) if units else None
        if parsed is not None:
            unit, base = parsed
            if unit.startswith("day"):
                dt = base + time_coord.astype("timedelta64[D]")
            elif unit.startswith("hour"):
                dt = base + time_coord.astype("timedelta64[h]")
            elif unit.startswith("min"):
                dt = base + time_coord.astype("timedelta64[m]")
            elif unit.startswith("sec"):
                dt = base + time_coord.astype("timedelta64[s]")
            else:
                raise ValueError(f"Unsupported CF time unit: {unit}")
            return dt.astype("datetime64[D]")

        dt = np.datetime64("1970-01-01") + time_coord.astype("timedelta64[D]")
        return dt.astype("datetime64[D]")

    raise ValueError(f"Unknown time coordinate dtype: {time_coord.dtype}")


def get_test_time_indices(T_total: int, t_hist: int, test_split: float, single_date: Optional[int],
                          test_year: Optional[int], viirs_zarr_path: str) -> List[int]:
    min_t = t_hist - 1
    max_t = T_total - 1

    if single_date is not None:
        t_idx = int(single_date)
        if t_idx < min_t or t_idx > max_t:
            raise ValueError(f"Date index {t_idx} out of valid range [{min_t}, {max_t}]")
        return [t_idx]

    if test_year is not None:
        print(f"[Test] Filtering to year {test_year}...")
        years = load_time_years_from_viirs(viirs_zarr_path)
        test_indices = [t for t in range(min_t, max_t + 1) if int(years[t]) == int(test_year)]
        if not test_indices:
            raise ValueError(f"No time indices found for year {test_year}")
        print(f"[Test] Found {len(test_indices)} days in year {test_year}")
        return test_indices

    split_idx = int(T_total * float(test_split))
    return list(range(max(split_idx, min_t), max_t + 1))


# =============================================================================
# Inference: Runner
# =============================================================================
def run_infer(args: argparse.Namespace) -> None:
    input_specs = [parse_input_spec(s) for s in args.inputs]
    viirs_spec = parse_input_spec(args.viirs_zarr)
    fwi_spec = parse_input_spec(args.fwi_zarr) if args.fwi_zarr else None

    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)

    global_model = load_global_model(args.model_dir, args.horizon)
    cluster_models = load_cluster_models(args.model_dir, args.horizon)
    residual_models, _ = load_residual_models(args.model_dir, args.horizon)

    if args.calibrators_json is None:
        cand = os.path.join(args.model_dir, "calibrators.json")
        if os.path.exists(cand):
            args.calibrators_json = cand
            print(f"[CLI] Using calibrators: {args.calibrators_json}")

    calib_bundle = None
    if (not args.no_calibration) and args.calibrators_json:
        calib_bundle = load_calibrators_json(args.calibrators_json)
        if calib_bundle is not None:
            print(f"[Calib] Enabled: method={calib_bundle.method}, per_regime={calib_bundle.per_regime}")
        else:
            print("[Calib] No valid calibration loaded; running uncalibrated.")
    else:
        print("[Calib] Disabled.")

    print("\n" + "="*80)
    print("LOADING AUXILIARY DATA")
    print("="*80)

    if args.peat_mask_source is None:
        args.peat_mask_source = input_specs[0].zarr
        print(f"[CLI] Using first input for peat mask: {args.peat_mask_source}")
    if args.coords_source is None:
        args.coords_source = input_specs[0].zarr
        print(f"[CLI] Using first input for coordinates: {args.coords_source}")

    zarr_stores = init_zarr_stores(input_specs, viirs_spec, args.peat_mask_source, fwi_spec=fwi_spec)
    T, H, W = zarr_stores["shape"]

    peat_mask = load_peat_mask(args.peat_mask_source, H, W)
    lat_grid, lon_grid = load_coordinates(args.coords_source, H, W)

    if args.regime_map:
        regime_map = load_spatial_map(args.regime_map, (H, W), "Regime map")
    else:
        print("[Warning] No regime map provided; per-regime residuals and per-regime calibration will be skipped.")
        regime_map = np.full((H, W), NODATA_VALUE, dtype=np.int32)

    if args.cluster_map:
        cluster_map = load_spatial_map(args.cluster_map, (H, W), "Cluster map")
    else:
        print("[Warning] No cluster map provided; cluster models will be skipped.")
        cluster_map = np.full((H, W), NODATA_VALUE, dtype=np.int32)

    # FWI alignment by time coordinate (optional)
    fwi_index_for_viirs: Optional[np.ndarray] = None
    if zarr_stores.get("fwi") is not None and fwi_spec is not None:
        try:
            viirs_dates = load_time_datetimes_from_zarr(viirs_spec.zarr)
            fwi_dates = load_time_datetimes_from_zarr(fwi_spec.zarr)

            fwi_map = {d: i for i, d in enumerate(fwi_dates)}
            fwi_index_for_viirs = np.full(viirs_dates.shape[0], -1, dtype=np.int64)
            for i, d in enumerate(viirs_dates):
                fwi_index_for_viirs[i] = fwi_map.get(d, -1)

            n_match = int(np.sum(fwi_index_for_viirs >= 0))
            print(f"[FWI] Alignment: matched {n_match:,}/{viirs_dates.shape[0]:,} VIIRS dates to FWI dates")
            if n_match == 0:
                print("[FWI] Warning: no overlapping dates found; disabling FWI output")
                zarr_stores["fwi"] = None
                fwi_index_for_viirs = None
        except Exception as e:
            print(f"[FWI] Warning: could not align FWI by time; disabling FWI output. Error: {e}")
            zarr_stores["fwi"] = None
            fwi_index_for_viirs = None

    print("\n" + "="*80)
    print("DETERMINING TEST TIME RANGE")
    print("="*80)

    test_indices = get_test_time_indices(
        T_total=T,
        t_hist=args.t_hist,
        test_split=args.test_split,
        single_date=args.single_date,
        test_year=args.test_year,
        viirs_zarr_path=viirs_spec.zarr,
    )

    print(f"[Test] Time indices: {len(test_indices)} days")
    print(f"[Test] Range: t={test_indices[0]} to t={test_indices[-1]}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("RUNNING PREDICTIONS")
    print("="*80)

    results = []
    for t_end in tqdm(test_indices, desc="Rolling predictions"):
        try:
            X, valid_pixels, spatial_shape = extract_features_for_time(
                zarr_stores, t_end, args.t_hist, peat_mask,
                coord_as_features=args.coord_as_features,
                lat_grid=lat_grid, lon_grid=lon_grid
            )

            pred_map = predict_ensemble(
                X, valid_pixels, spatial_shape,
                regime_map, cluster_map,
                global_model, cluster_models, residual_models,
                ensemble_method=args.ensemble_method,
                weights={"global": args.global_weight, "cluster": args.cluster_weight, "residual": args.residual_weight},
                horizon=int(args.horizon),
                calib_bundle=calib_bundle,
                calib_debug=bool(args.calib_debug),
                calib_debug_prefix=f"t={t_end}",
            )

            viirs_map = load_viirs_for_time(zarr_stores["viirs"], t_end, peat_mask)

            png_path = os.path.join(args.output_dir, f"pred_t{t_end:05d}.png")
            create_fire_probability_map(
                pred_map, viirs_map, lat_grid, lon_grid, peat_mask,
                png_path,
                title=f"Fire Probability Nowcast - Day {t_end}",
                dpi=args.plot_dpi,
                add_features=args.add_map_features,
            )

            # FWI map (optional)
            if zarr_stores.get("fwi") is not None:
                fwi_idx = -1
                if fwi_index_for_viirs is not None and t_end < fwi_index_for_viirs.shape[0]:
                    fwi_idx = int(fwi_index_for_viirs[t_end])
                else:
                    if t_end < zarr_stores["fwi"].shape[0]:
                        fwi_idx = int(t_end)

                if 0 <= fwi_idx < zarr_stores["fwi"].shape[0]:
                    fwi_map = load_fwi_for_time(zarr_stores["fwi"], fwi_idx, peat_mask)
                    fwi_map = fwi_map / float(FWI_DISPLAY_DIVISOR)

                    fwi_png_path = os.path.join(args.output_dir, f"fwi_t{t_end:05d}.png")
                    create_scalar_map(
                        fwi_map, viirs_map, lat_grid, lon_grid, peat_mask,
                        fwi_png_path,
                        title=f"FWI - Day {t_end}",
                        cbar_label=None,
                        vmin=FWI_VMIN,
                        vmax=FWI_VMAX,
                        dpi=args.plot_dpi,
                        add_features=args.add_map_features,
                        bounds=FWI_BOUNDS,
                        bin_labels=FWI_LABELS,
                        extend="neither",
                    )
                else:
                    print(f"[FWI] No FWI for t={t_end} (no matching date); skipping FWI PNG")

            if args.save_geotiff:
                tif_path = png_path.replace(".png", ".tif")
                save_geotiff(pred_map, tif_path, lat_grid, lon_grid)

            metrics = compute_metrics(pred_map, viirs_map, peat_mask, args.threshold)
            results.append({"t_end": int(t_end), "metrics": metrics})

            print(f"[t={t_end}] F1={metrics['f1_score']:.4f}, Fires={metrics['n_fire_pixels']}, Mean prob (fire)={metrics['mean_prob_fire']:.4f}")

        except Exception as e:
            print(f"[ERROR] Failed at t={t_end}: {e}")
            import traceback
            traceback.print_exc()
            continue

    summary_path = os.path.join(args.output_dir, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Summary] Saved metrics to: {summary_path}")

    if results:
        avg_f1 = float(np.mean([r["metrics"]["f1_score"] for r in results]))
        avg_precision = float(np.mean([r["metrics"]["precision"] for r in results]))
        avg_recall = float(np.mean([r["metrics"]["recall"] for r in results]))

        print("\n" + "="*80)
        print("AGGREGATE METRICS")
        print("="*80)
        print(f"Average F1 Score:  {avg_f1:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall:    {avg_recall:.4f}")
        print("="*80)


# =============================================================================
# Viz/analysis section (formerly yearly_cartopy_maps_v5_all_in_one.py)
# =============================================================================
EPS = 1e-12
TIF_RE = re.compile(r".*pred_t(\d+)\.tif$")


def build_region_groups():
    """
    Regions are (lon_min, lat_min, lon_max, lat_max) in EPSG:4326 lon/lat.
    """
    BOREAL_LAT_MIN = 50.0
    BOREAL_LAT_MAX = 75.55

    EU_LON_MIN, EU_LON_MAX = -10.0, 45.0
    EU_LAT_MIN, EU_LAT_MAX = 55.0, 72.0

    WEST_SIB_LON_MIN, WEST_SIB_LON_MAX = 55.0, 95.0
    EAST_SIB_LON_MIN, EAST_SIB_LON_MAX = 95.0, 180.0

    PAT_LON_MIN, PAT_LON_MAX = -76.0, -57.0
    PAT_LAT_MIN, PAT_LAT_MAX = -56.0, -36.0

    return [
        {"name": "Tropical Americas", "boxes": [(-159.8005, -23.4394, -34.7300, 23.4394)]},
        {"name": "Tropical Africa", "boxes": [(-17.6250, -23.4394, 51.1339, 23.4394)]},
        {"name": "Tropical Asia (IDN/MYS+)", "boxes": [(95.2930, -10.6525, 156.0200, 6.9281)]},
        {"name": "Boreal North America", "boxes": [(-168.0, 50.0, -55.4070, 70.0)]},
        {"name": "Boreal Europe (Fennoscandia/NW Russia)", "boxes": [(EU_LON_MIN, EU_LAT_MIN, EU_LON_MAX, EU_LAT_MAX)]},
        {"name": "Boreal West Siberia", "boxes": [(WEST_SIB_LON_MIN, BOREAL_LAT_MIN, WEST_SIB_LON_MAX, BOREAL_LAT_MAX)]},
        {"name": "Boreal East Siberia", "boxes": [(EAST_SIB_LON_MIN, BOREAL_LAT_MIN, EAST_SIB_LON_MAX, BOREAL_LAT_MAX)]},
        {"name": "Temperate South America (Patagonia/Tierra del Fuego)", "boxes": [(PAT_LON_MIN, PAT_LAT_MIN, PAT_LON_MAX, PAT_LAT_MAX)]},
    ]


def extent_from_boxes(boxes, pad_deg: float = 0.0):
    xs, ys = [], []
    for (xmin, ymin, xmax, ymax) in boxes:
        xs += [xmin, xmax]
        ys += [ymin, ymax]

    # Dateline safety: extents that hit exactly +/-180 can trigger a Cartopy
    # seam artifact (a long quad drawn "the other way" across the map).
    # This shows up most often for Boreal East Siberia when lon_max == 180.
    # Nudge away from the seam by a tiny epsilon.
    xmin = float(min(xs) - pad_deg)
    xmax = float(max(xs) + pad_deg)
    ymin = float(min(ys) - pad_deg)
    ymax = float(max(ys) + pad_deg)

    eps = 1e-3  # 0.001 deg is far below your 0.1 deg grid, but avoids the seam.
    if xmax >= 180.0:
        xmax = 180.0 - eps
    if xmin <= -180.0:
        xmin = -180.0 + eps

    return [xmin, xmax, ymin, ymax]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_pred_tifs(pred_dir: str) -> List[Tuple[int, str]]:
    paths = sorted(glob.glob(os.path.join(pred_dir, "pred_t*.tif")))
    items: List[Tuple[int, str]] = []
    for p in paths:
        m = TIF_RE.match(p)
        if not m:
            continue
        items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return items


def open_zarr_array(spec: str) -> zarr.Array:
    zpath, field = spec.split(":", 1)
    root = zarr.open_group(zpath, mode="r")
    if field not in root:
        raise ValueError(f"Field '{field}' not found in {zpath}. Keys: {list(root.array_keys())}")
    return root[field]


def squeeze_to_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[0] in (1, 2, 3, 4) and a.shape[1] > 10 and a.shape[2] > 10:
            return np.squeeze(a[0])
        if a.shape[-1] in (1, 2, 3, 4) and a.shape[0] > 10 and a.shape[1] > 10:
            return np.squeeze(a[..., 0])
    raise ValueError(f"Expected 2D after squeeze, got shape {a.shape}.")


def read_zarr_2d(arr: zarr.Array, t_index: Optional[int] = None) -> np.ndarray:
    if arr.ndim == 2:
        return squeeze_to_2d(np.asarray(arr[:]))
    if t_index is None:
        t_index = 0
    return squeeze_to_2d(np.asarray(arr[t_index]))


@dataclass
class GeoGrid:
    x_edges: np.ndarray      # (W+1,)
    y_edges: np.ndarray      # (H+1,)
    lat_centers: np.ndarray  # (H,)
    shape: Tuple[int, int]   # (H, W)


def grid_from_tif(path: str) -> Tuple[GeoGrid, Optional[float]]:
    with rasterio.open(path) as ds:
        H, W = ds.height, ds.width
        transform = ds.transform
        nodata = ds.nodata

    x0 = transform.c
    y0 = transform.f
    dx = transform.a
    dy = transform.e  # negative for north-up

    x_edges = (x0 + np.arange(W + 1) * dx).astype(np.float64)
    y_edges = (y0 + np.arange(H + 1) * dy).astype(np.float64)
    lat_centers = ((y_edges[:-1] + y_edges[1:]) / 2.0).astype(np.float64)

    grid = GeoGrid(
        x_edges=x_edges,
        y_edges=y_edges,
        lat_centers=lat_centers,
        shape=(H, W),
    )
    return grid, nodata


def read_tif_2d(path: str, nodata: Optional[float]) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


@dataclass
class GridNormalizer:
    grid: GeoGrid
    flip_y: bool
    wrap_lon: bool
    cut_col: Optional[int]

    def apply(self, a2d: np.ndarray) -> np.ndarray:
        a = np.asarray(a2d)
        if self.flip_y:
            a = np.flipud(a)
        if self.wrap_lon and self.cut_col is not None:
            c = self.cut_col
            a = np.concatenate([a[:, c:], a[:, :c]], axis=1)
        return a


def build_normalizer(grid: GeoGrid, wrap_longitude: str) -> GridNormalizer:
    H, W = grid.shape
    x_edges = grid.x_edges.copy()
    y_edges = grid.y_edges.copy()

    flip_y = bool(y_edges[0] > y_edges[-1])
    if flip_y:
        y_edges = y_edges[::-1].copy()

    lat_centers = ((y_edges[:-1] + y_edges[1:]) / 2.0).astype(np.float64)

    wrap_lon = False
    if wrap_longitude == "always":
        wrap_lon = True
    elif wrap_longitude == "auto":
        if np.nanmin(x_edges) >= -EPS and np.nanmax(x_edges) > 180.0 + EPS:
            wrap_lon = True

    cut_col: Optional[int] = None
    if wrap_lon:
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        cut_col = int(np.searchsorted(x_centers, 180.0, side="left"))
        cut_col = cut_col if (0 < cut_col < W) else None

        if cut_col is not None:
            seg1 = x_edges[cut_col:] - 360.0
            seg2 = x_edges[1: cut_col + 1]
            x_edges = np.concatenate([seg1, seg2]).astype(np.float64)

    new_grid = GeoGrid(
        x_edges=x_edges.astype(np.float64),
        y_edges=y_edges.astype(np.float64),
        lat_centers=lat_centers,
        shape=(H, W),
    )
    return GridNormalizer(grid=new_grid, flip_y=flip_y, wrap_lon=wrap_lon, cut_col=cut_col)


def compute_daily_percentiles(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    finite = mask & np.isfinite(arr)
    n = int(finite.sum())
    if n <= 0:
        return out

    vals = arr[finite].astype(np.float64)
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)

    denom = (n - 1) if n > 1 else 1.0
    pct = 100.0 * (ranks / denom)
    out[finite] = pct.astype(np.float32)
    return out


def topk_mask_daily(score: np.ndarray, peat_mask: np.ndarray, top_frac: float, weights: Optional[np.ndarray]) -> np.ndarray:
    finite = peat_mask & np.isfinite(score)
    out = np.zeros(score.shape, dtype=bool)
    n = int(finite.sum())
    if n <= 0:
        return out

    vals = score[finite].astype(np.float64)
    idx = np.argsort(vals, kind="mergesort")[::-1]
    flat_positions = np.flatnonzero(finite)

    if weights is None:
        k = int(np.ceil(top_frac * n))
        k = max(1, min(k, n))
        chosen = idx[:k]
    else:
        w = weights[finite].astype(np.float64)
        w_sorted = w[idx]
        total = float(np.sum(w_sorted))
        target = top_frac * total
        cs = np.cumsum(w_sorted)
        k = int(np.searchsorted(cs, target, side="left")) + 1
        k = max(1, min(k, n))
        chosen = idx[:k]

    out.flat[flat_positions[chosen]] = True
    return out


def update_capture_stats(
    score: np.ndarray,
    fire: np.ndarray,
    mask: np.ndarray,
    fracs: List[float],
    weights: Optional[np.ndarray],
    numer: np.ndarray,
    denom: np.ndarray,
) -> None:
    finite = mask & np.isfinite(score)
    if finite.sum() == 0:
        return

    fire_m = fire & finite
    if weights is None:
        fire_total = float(fire_m.sum())
    else:
        fire_total = float(np.sum(weights[fire_m]))

    if fire_total <= 0:
        return

    vals = score[finite].astype(np.float64)
    idx = np.argsort(vals, kind="mergesort")[::-1]
    flat = np.flatnonzero(finite)

    if weights is None:
        n = len(idx)
        for i, frac in enumerate(fracs):
            k = max(1, min(n, int(np.ceil(frac * n))))
            chosen = flat[idx[:k]]
            alert_flat = np.zeros(score.size, dtype=bool)
            alert_flat[chosen] = True
            alert = alert_flat.reshape(score.shape)
            numer[i] += float((alert & fire_m).sum())
            denom[i] += fire_total
    else:
        w_f = weights[finite].astype(np.float64)
        w_sorted = w_f[idx]
        total_w = float(np.sum(w_sorted))
        cs = np.cumsum(w_sorted)
        for i, frac in enumerate(fracs):
            target = frac * total_w
            k = int(np.searchsorted(cs, target, side="left")) + 1
            k = max(1, min(k, len(idx)))
            chosen = flat[idx[:k]]
            alert_flat = np.zeros(score.size, dtype=bool)
            alert_flat[chosen] = True
            alert = alert_flat.reshape(score.shape)
            numer[i] += float(np.sum(weights[(alert & fire_m)]))
            denom[i] += fire_total


def daily_percentile_cmap_norm():
    # Poster-friendly percentile palette (no yellow).
    # - Darker gray for low percentiles/background so low-signal peatlands pop.
    # - Replace yellow with purple for better contrast against light land/ocean.
    colors = ["#7A7A7A", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728"]
    bounds = np.array([0, 20, 40, 60, 80, 95, 100], dtype=np.float64)
    cmap = mcolors.ListedColormap(colors, name="pct6")
    norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
    return cmap, norm, bounds


def pct_cmap_continuous():
    """Continuous percentile colormap (0–100) with a contrasting gray bottom and red maximum.

    Intended for poster readability: low percentiles show as gray (background), while high
    percentiles ramp through cool/warm colors to a strong red at 100.
    """
    stops = [
        (0.00, "#7A7A7A"),  # darker gray low end (high-contrast background)
        (0.15, "#1f77b4"),  # blue
        (0.45, "#2ca02c"),  # green
        (0.70, "#9467bd"),  # purple (replaces yellow for better contrast)
        (0.88, "#ff7f0e"),  # orange
        (1.00, "#d62728"),  # red max
    ]
    return mcolors.LinearSegmentedColormap.from_list("pct_continuous", stops, N=256)


@dataclass
class Layer:
    title: str
    data: np.ndarray
    cmap: Any = "viridis"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    cbar_label: str = ""
    extend: str = "max"
    norm: Optional[mcolors.Normalize] = None
    cbar_ticks: Optional[List[float]] = None
    cbar_ticklabels: Optional[List[str]] = None


def _auto_vmax(data: np.ndarray, fallback: float = 1.0, q: float = 99.0) -> float:
    finite = np.isfinite(data)
    if finite.sum() == 0:
        return fallback
    v = float(np.nanpercentile(data[finite], q))
    if not np.isfinite(v) or v <= 0.0:
        return fallback
    return v

# --- Fire-frequency scaling (for % of days with fire)
# Many regions have very low fire-day percentages (e.g., <1%).
# A power-law normalization (gamma<1) expands low values so you can see variation.

def make_firefreq_norm(data: 'np.ndarray', vmax: float, mode: str, gamma: float, log_vmin: float) -> 'mcolors.Normalize|None':
    """Return a Matplotlib Normalize for fire-frequency (% of days) layers.

    mode:
      - 'linear' : no norm (use vmin/vmax)
      - 'power'  : PowerNorm(gamma) with vmin=0
      - 'log'    : LogNorm with vmin=log_vmin (values <=0 should be NaN)
    """
    mode = str(mode).lower()
    if mode in ('linear', 'none', ''):
        return None

    vmax = float(vmax)
    if not (vmax > 0 and vmax == vmax):
        return None

    if mode == 'power':
        g = float(gamma)
        if not (g > 0 and g == g):
            g = 0.5
        return mcolors.PowerNorm(gamma=g, vmin=0.0, vmax=vmax)

    if mode == 'log':
        vmin = float(log_vmin)
        if not (vmin > 0 and vmin == vmin):
            vmin = 0.01
        # Safety: ensure vmin < vmax
        if vmin >= vmax:
            vmin = max(vmax * 1e-3, 1e-6)
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)

    return None



def plot_stacked_cartopy(
    layers: List[Layer],
    grid: GeoGrid,
    out_path: str,
    dpi: int,
    figwidth: float,
    row_height: float,
    coastline_lw: float,
    border_lw: float,
    gridline_alpha: float,
    extent: Optional[List[float]] = None,
    peat_mask: Optional[np.ndarray] = None,
    peat_alpha: float = 0.20,
) -> None:
    nrows = len(layers)
    figheight = max(2.0, row_height * nrows)

    fig = plt.figure(figsize=(figwidth, figheight), dpi=dpi)
    proj = ccrs.PlateCarree()

    underlay = None
    peat_cmap = None
    if peat_mask is not None:
        underlay = np.where(peat_mask, 1.0, np.nan).astype(np.float32)
        # Darker peat underlay so low-signal areas (e.g., very low percentiles) still pop on posters.
        peat_cmap = mcolors.ListedColormap(["#6E6E6E"])
        peat_cmap.set_bad(alpha=0.0)

    for i, layer in enumerate(layers, start=1):
        ax = fig.add_subplot(nrows, 1, i, projection=proj)

        if extent is None:
            ax.set_global()
        else:
            ax.set_extent(extent, crs=proj)

        ax.add_feature(cfeature.OCEAN, zorder=0)
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor="none")
        ax.add_feature(cfeature.COASTLINE, linewidth=coastline_lw, zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=border_lw, zorder=3)
        ax.gridlines(draw_labels=False, linewidth=0.3, alpha=gridline_alpha, linestyle="--")

        if underlay is not None and peat_cmap is not None:
            ax.pcolormesh(
                grid.x_edges, grid.y_edges,
                np.ma.masked_invalid(underlay),
                transform=proj, shading="auto",
                cmap=peat_cmap, vmin=0.0, vmax=1.0,
                zorder=1, alpha=peat_alpha,
            )

        data = np.array(layer.data, dtype=np.float32)
        data = np.ma.masked_invalid(data)

        if isinstance(layer.cmap, str):
            cmap_obj = plt.get_cmap(layer.cmap)
        else:
            cmap_obj = layer.cmap
        cmap = cmap_obj.copy() if hasattr(cmap_obj, "copy") else cmap_obj
        cmap.set_bad(alpha=0.0)

        filled = data.filled(np.nan)

        if layer.norm is None:
            vmin = layer.vmin
            vmax = layer.vmax

            if vmax is None:
                vmax = _auto_vmax(filled, fallback=1.0, q=99.0)
            if vmin is None:
                mn = np.nanmin(filled) if np.isfinite(filled).any() else 0.0
                vmin = 0.0 if np.isfinite(mn) and mn >= 0 else float(np.nanpercentile(filled, 1))

            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0

            im = ax.pcolormesh(
                grid.x_edges, grid.y_edges,
                data, transform=proj, shading="auto",
                cmap=cmap, vmin=vmin, vmax=vmax, zorder=2,
            )
        else:
            im = ax.pcolormesh(
                grid.x_edges, grid.y_edges,
                data, transform=proj, shading="auto",
                cmap=cmap, norm=layer.norm, zorder=2,
            )

        ax.set_title(layer.title, fontsize=10, pad=6)

        cb = plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.034, pad=0.02, extend=layer.extend)
        if layer.cbar_label:
            cb.set_label(layer.cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=8)

        if layer.cbar_ticks is not None:
            cb.set_ticks(layer.cbar_ticks)
        if layer.cbar_ticklabels is not None:
            cb.set_ticklabels(layer.cbar_ticklabels)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_capture_curve(fracs: List[float], model_rate: np.ndarray, fwi_rate: Optional[np.ndarray], title: str, out_png: str) -> None:
    x = np.array(fracs) * 100.0
    plt.figure(figsize=(5.4, 3.6), dpi=850)
    plt.plot(x, model_rate, label="Model")
    if fwi_rate is not None:
        plt.plot(x, fwi_rate, label="FWI")
    plt.xlabel("% peat area monitored (daily)")
    plt.ylabel("% VIIRS fire-days captured")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.ylim(0, 100)
    plt.xlim(0, max(x))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=850, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    fpr_model: np.ndarray,
    tpr_model: np.ndarray,
    auc_model: float,
    fpr_fwi: Optional[np.ndarray],
    tpr_fwi: Optional[np.ndarray],
    auc_fwi: Optional[float],
    title: str,
    out_png: str,
    *,
    auc_model_exact: Optional[float] = None,
    auc_fwi_exact: Optional[float] = None,
) -> None:
    """Poster-friendly ROC plot.

    - Light shading under each curve.
    - Overlap region is shaded in a neutral color (avoids brown blends).
    - Legend includes the dotted-line meaning.
    """
    # ---- poster-friendly style knobs ----
    ROC_LW = 3.2
    DIAG_LW = 2.4
    FILL_ALPHA = 0.16
    OVERLAP_ALPHA = 0.14
    LEGEND_FS = 12
    LABEL_FS = 13
    TITLE_FS = 14
    TICK_FS = 12
    SAVE_DPI = 850
    # -------------------------------------

    plt.figure(figsize=(6, 6), dpi=850)

    def _safe_auc(a: Optional[float], fallback: float) -> float:
        if a is None:
            return float(fallback)
        a = float(a)
        return a if np.isfinite(a) else float(fallback)

    def _prep_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if x.size == 0:
            return x, y
        # sort & de-dup x for interp
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]
        # clamp
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        # ensure monotone non-decreasing y (helps interpolation artifacts)
        y = np.maximum.accumulate(y)
        # unique x
        ux, ui = np.unique(x, return_index=True)
        return ux, y[ui]

    # Prefer exact AUCs if provided; display as %.
    auc_model_best = _safe_auc(auc_model_exact, auc_model)
    model_label = f"Model (AUC={auc_model_best * 100.0:.1f}% of plot filled)"

    fpr_m, tpr_m = _prep_xy(fpr_model, tpr_model)
    (model_line,) = plt.plot(fpr_m, tpr_m, label=model_label, linewidth=ROC_LW)

    # Optional FWI
    has_fwi = fpr_fwi is not None and tpr_fwi is not None
    if has_fwi:
        auc_fwi_best = _safe_auc(auc_fwi_exact, auc_fwi if auc_fwi is not None else float('nan'))
        fwi_label = f"FWI (AUC={auc_fwi_best * 100.0:.1f}% of plot filled)"
        fpr_f, tpr_f = _prep_xy(fpr_fwi, tpr_fwi)
        (fwi_line,) = plt.plot(fpr_f, tpr_f, label=fwi_label, linewidth=ROC_LW)

        # Build a shared FPR grid for clean overlap shading
        grid = np.unique(np.concatenate([
            np.linspace(0.0, 1.0, 1201),
            fpr_m,
            fpr_f,
        ]))
        tpr_m_i = np.interp(grid, fpr_m, tpr_m, left=0.0, right=1.0)
        tpr_f_i = np.interp(grid, fpr_f, tpr_f, left=0.0, right=1.0)

        overlap = np.minimum(tpr_m_i, tpr_f_i)
        m_only = np.maximum(tpr_m_i - overlap, 0.0)
        f_only = np.maximum(tpr_f_i - overlap, 0.0)

        # Shade overlap in neutral gray (avoids muddy brown blends)
        plt.fill_between(grid, 0.0, overlap, color='0.55', alpha=OVERLAP_ALPHA, linewidth=0)
        # Shade each curve's non-overlap contribution
        plt.fill_between(grid, overlap, overlap + m_only, color=model_line.get_color(), alpha=FILL_ALPHA, linewidth=0)
        plt.fill_between(grid, overlap, overlap + f_only, color=fwi_line.get_color(), alpha=FILL_ALPHA, linewidth=0)

    else:
        # Only one curve -> simple fill
        plt.fill_between(fpr_m, 0.0, tpr_m, color=model_line.get_color(), alpha=FILL_ALPHA, linewidth=0)

    # Diagonal baseline (dotted) with legend entry
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=DIAG_LW, label='Random predictions')

    plt.xlabel("False Positive Rate", fontsize=LABEL_FS)
    plt.ylabel("True Positive Rate", fontsize=LABEL_FS)
    plt.title(title, fontsize=TITLE_FS)

    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)

    plt.legend(
        loc="lower right",
        fontsize=LEGEND_FS,
        frameon=True,
        framealpha=0.92,
        borderpad=0.7,
        handlelength=3.0,
        labelspacing=0.5,
    )

    plt.grid(True, alpha=0.25, linewidth=0.9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=SAVE_DPI)
    plt.close()



@dataclass
class RocHist:
    """Histogram counts for ROC computation.

    We approximate ROC by binning scores into fixed edges, keeping separate counts
    for positives and negatives. We then sweep the threshold from high->low using
    cumulative sums.

    Fields:
      pos: counts of positive examples per bin
      neg: counts of negative examples per bin
      edges: bin edges (monotonic increasing)
    """
    pos: np.ndarray
    neg: np.ndarray
    edges: np.ndarray

def roc_hist_init(edges: np.ndarray) -> RocHist:
    nb = len(edges) - 1
    return RocHist(
        pos=np.zeros(nb, dtype=np.float64),
        neg=np.zeros(nb, dtype=np.float64),
        edges=np.asarray(edges, dtype=np.float64),
    )


def roc_hist_update(hist: RocHist, scores: np.ndarray, labels: np.ndarray) -> None:
    if scores.size == 0:
        return
    pos_scores = scores[labels]
    neg_scores = scores[~labels]
    if pos_scores.size > 0:
        h, _ = np.histogram(pos_scores, bins=hist.edges)
        hist.pos += h.astype(np.float64)
    if neg_scores.size > 0:
        h, _ = np.histogram(neg_scores, bins=hist.edges)
        hist.neg += h.astype(np.float64)


def roc_from_hist(hist: RocHist) -> Tuple[np.ndarray, np.ndarray, float]:
    P = float(hist.pos.sum())
    N = float(hist.neg.sum())
    if P <= 0 or N <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    tp = np.cumsum(hist.pos[::-1])
    fp = np.cumsum(hist.neg[::-1])

    tpr = tp / P
    fpr = fp / N

    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])

    # NumPy 2.x removed np.trapz; use trapezoid with backward-compatible fallback.
    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is None:
        trapz_fn = getattr(np, "trapz")
    auc = float(trapz_fn(tpr, fpr))
    return fpr, tpr, auc


# =============================================================================
# Yearly evaluation metrics + spider (radar) charts
# =============================================================================
SPIDER_METRIC_ORDER = ["ECE", "Brier", "LogLoss", "ROC-AUC", "Correlation"]
SPIDER_BASE_ORANGE = "#f28e2b"  # orange

# Metadata for OPTIONAL relative spider scaling (log-minmax across series).
# Keys here map the displayed label -> raw metric key in the metrics dict.
# direction: 'high' means higher-is-better; 'low' means lower-is-better.
# transform: optional pre-transform applied BEFORE scaling (e.g., map corr[-1,1]->[0,1]).
SPIDER_METRIC_INFO = {
    'ECE': {
        'key': 'ece',
        'direction': 'low',
        'log': True,
    },
    'Brier': {
        'key': 'brier',
        'direction': 'low',
        'log': True,
    },
    'LogLoss': {
        'key': 'logloss',
        'direction': 'low',
        'log': True,
    },
    'ROC-AUC': {
        'key': 'roc_auc',
        'direction': 'high',
        'log': False,
        # Map the meaningful range [0.5, 1.0] -> [0, 1] before min-max scaling.
        'transform': (lambda v: (float(v) - 0.5) / 0.5),
    },
    'Correlation': {
        'key': 'corr',
        'direction': 'high',
        'log': False,
        # Map [-1, 1] -> [0, 1] before min-max scaling.
        'transform': (lambda v: (float(v) + 1.0) / 2.0),
    },
}


@dataclass
class YearlyMetricsAccumulator:
    """
    Streaming accumulators for yearly, pixel-day metrics.

    Metrics:
      - ECE (Expected Calibration Error) over uniform probability bins in [0,1]
      - Brier score
      - Log loss
      - ROC-AUC (via histogram ROC)
      - Pearson correlation between p and y

    Notes:
      - All metrics are computed over the same valid set: mask & finite(pred) & finite(label_source)
      - ROC-AUC uses the same clipped probabilities used for logloss.
    """
    ece_edges: np.ndarray
    roc_hist: RocHist

    n: int = 0
    sum_sq: float = 0.0
    sum_logloss: float = 0.0

    sum_p: float = 0.0
    sum_y: float = 0.0
    sum_p2: float = 0.0
    sum_y2: float = 0.0
    sum_py: float = 0.0

    ece_counts: np.ndarray = None  # (B,)
    ece_sum_p: np.ndarray = None   # (B,)
    ece_sum_y: np.ndarray = None   # (B,)

    def __post_init__(self):
        nb = int(len(self.ece_edges) - 1)
        self.ece_counts = np.zeros(nb, dtype=np.float64)
        self.ece_sum_p = np.zeros(nb, dtype=np.float64)
        self.ece_sum_y = np.zeros(nb, dtype=np.float64)

    def update(self, prob2d: np.ndarray, label2d: np.ndarray, valid_mask2d: np.ndarray) -> None:
        m = valid_mask2d & np.isfinite(prob2d)
        if not np.any(m):
            return

        p = np.clip(prob2d[m].astype(np.float64), EPS_PROB, 1.0 - EPS_PROB)
        y = label2d[m].astype(np.float64)
        # Ensure labels are 0/1
        y = np.where(y > 0.0, 1.0, 0.0)

        n = int(p.size)
        self.n += n

        diff = (p - y)
        self.sum_sq += float(np.sum(diff * diff))

        self.sum_logloss += float(np.sum(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))

        self.sum_p += float(np.sum(p))
        self.sum_y += float(np.sum(y))
        self.sum_p2 += float(np.sum(p * p))
        self.sum_y2 += float(np.sum(y * y))
        self.sum_py += float(np.sum(p * y))

        # ECE bins
        bin_idx = np.digitize(p, self.ece_edges, right=False) - 1
        bin_idx = np.clip(bin_idx, 0, self.ece_counts.size - 1)

        self.ece_counts += np.bincount(bin_idx, minlength=self.ece_counts.size).astype(np.float64)
        self.ece_sum_p += np.bincount(bin_idx, weights=p, minlength=self.ece_counts.size).astype(np.float64)
        self.ece_sum_y += np.bincount(bin_idx, weights=y, minlength=self.ece_counts.size).astype(np.float64)

        # ROC-AUC hist
        roc_hist_update(self.roc_hist, p, y.astype(bool))

    def finalize(self) -> Dict[str, float]:
        if self.n <= 0:
            return {"ece": float("nan"), "brier": float("nan"), "logloss": float("nan"),
                    "roc_auc": float("nan"), "corr": float("nan"), "n": 0}

        n = float(self.n)
        brier = self.sum_sq / n
        logloss = self.sum_logloss / n

        # ECE
        counts = self.ece_counts.copy()
        good = counts > 0
        conf = np.zeros_like(counts)
        acc = np.zeros_like(counts)
        conf[good] = self.ece_sum_p[good] / counts[good]
        acc[good] = self.ece_sum_y[good] / counts[good]
        ece = float(np.sum(np.abs(acc[good] - conf[good]) * (counts[good] / n))) if np.any(good) else float("nan")

        # Correlation
        mean_p = self.sum_p / n
        mean_y = self.sum_y / n
        cov = (self.sum_py / n) - (mean_p * mean_y)
        var_p = (self.sum_p2 / n) - (mean_p * mean_p)
        var_y = (self.sum_y2 / n) - (mean_y * mean_y)
        if var_p > 0.0 and var_y > 0.0:
            corr = float(cov / np.sqrt(var_p * var_y))
            corr = float(np.clip(corr, -1.0, 1.0))
        else:
            corr = float("nan")

        # ROC-AUC
        _, _, auc = roc_from_hist(self.roc_hist)

        return {"ece": float(ece), "brier": float(brier), "logloss": float(logloss),
                "roc_auc": float(auc), "corr": float(corr), "n": int(self.n)}


def auc_exact_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute exact AUC from raw scores/labels (tie-aware, no binning).

    This uses a rank-based / Mann–Whitney formulation:
      AUC = (concordant_pairs + 0.5 * tied_pairs) / (n_pos * n_neg)

    Args:
        scores: shape (N,), larger score => more likely positive.
        labels: shape (N,), truth labels (bool or {0,1}).

    Returns:
        AUC in [0, 1], or np.nan if there are no positives or no negatives.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)

    n_pos = int(labels.sum())
    n = int(labels.size)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")  # stable for tie handling
    s = scores[order]
    y = labels[order].astype(np.int64)

    # Group equal scores.
    if s.size == 0:
        return float("nan")
    group_starts = np.r_[0, 1 + np.where(s[1:] != s[:-1])[0]]
    counts = np.diff(np.r_[group_starts, s.size])
    pos_counts = np.add.reduceat(y, group_starts)
    neg_counts = counts - pos_counts

    neg_before = np.cumsum(np.r_[0, neg_counts[:-1]])
    concordant = np.sum(pos_counts * neg_before + 0.5 * pos_counts * neg_counts)

    return float(concordant / (n_pos * n_neg))


# =============================================================================
# Reliability bias (calibration) heatmap by probability bins
# =============================================================================

RELIABILITY_BINS_PCT_DEFAULT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0
]


@dataclass
class ReliabilityBiasAccumulator:
    """Accumulates reliability-bias statistics in probability bins.

    For each bin b, we store:
      - counts[b] : number of samples
      - sum_p[b]  : sum of predicted probabilities
      - sum_y[b]  : sum of binary outcomes (0/1)

    Bias is reported as (mean_p - mean_y) in percentage points.
    """

    edges: np.ndarray  # bin edges in [0, 1], shape (B+1,)
    counts: np.ndarray = field(init=False)
    sum_p: np.ndarray = field(init=False)
    sum_y: np.ndarray = field(init=False)

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=np.float64)
        if self.edges.ndim != 1 or self.edges.size < 2:
            raise ValueError("ReliabilityBiasAccumulator: edges must be 1-D with >= 2 entries.")
        if not np.all(np.diff(self.edges) > 0):
            raise ValueError("ReliabilityBiasAccumulator: edges must be strictly increasing.")
        b = self.edges.size - 1
        self.counts = np.zeros((b,), dtype=np.int64)
        self.sum_p = np.zeros((b,), dtype=np.float64)
        self.sum_y = np.zeros((b,), dtype=np.float64)

    def update(self, prob2d: np.ndarray, fire2d: np.ndarray, valid_mask2d: np.ndarray):
        """Update with a single timestep.

        prob2d: (H, W) predicted probabilities in [0, 1]
        fire2d: (H, W) boolean or {0,1} outcomes
        valid_mask2d: (H, W) boolean mask for samples to include
        """
        if prob2d.shape != fire2d.shape or prob2d.shape != valid_mask2d.shape:
            raise ValueError("ReliabilityBiasAccumulator.update: shape mismatch.")
        m = valid_mask2d & np.isfinite(prob2d)
        if not np.any(m):
            return

        p = prob2d[m].astype(np.float64, copy=False)
        y = fire2d[m].astype(np.float64, copy=False)

        # Bin index in [0, B-1]. Anything > last edge goes into last bin.
        idx = np.digitize(p, self.edges, right=False) - 1
        b = self.edges.size - 1
        idx = np.clip(idx, 0, b - 1)

        self.counts += np.bincount(idx, minlength=b).astype(np.int64)
        self.sum_p += np.bincount(idx, weights=p, minlength=b).astype(np.float64)
        self.sum_y += np.bincount(idx, weights=y, minlength=b).astype(np.float64)

    def finalize(self):
        """Return dict with mean_p, obs, bias_pct, and counts."""
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_p = self.sum_p / self.counts
            obs = self.sum_y / self.counts
        bias = mean_p - obs
        return {
            "counts": self.counts.copy(),
            "mean_p": mean_p,
            "obs": obs,
            "bias_pct": bias * 100.0,  # percentage points
            "edges": self.edges.copy(),
        }


def _format_box(xmin, ymin, xmax, ymax) -> str:
    return f"lon[{xmin:.1f},{xmax:.1f}], lat[{ymin:.1f},{ymax:.1f}]"


def describe_region(region: Dict) -> str:
    """Human-readable region description from region dict (name + bounding boxes)."""
    name = region.get("name", "Region")
    boxes = region.get("boxes", [])
    if not boxes:
        return name
    parts = [_format_box(*b) for b in boxes]
    if len(parts) == 1:
        return f"{name}: {parts[0]}"
    joined = "; ".join(parts)
    return f"{name}: {joined}"


def write_reliability_bias_csv(out_csv: str,
                              regions_order: List[str],
                              region_desc: Dict[str, str],
                              edges_pct: List[float],
                              stats_by_region: Dict[str, Dict]) -> None:
    """Write per-region reliability bias stats to CSV."""
    import csv
    bins = len(edges_pct) - 1
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "region",
            "region_description",
            "bin_lo_pct",
            "bin_hi_pct",
            "n",
            "mean_pred_pct",
            "obs_pct",
            "bias_pct_points"
        ])
        for r in regions_order:
            st = stats_by_region[r]
            counts = st["counts"]
            mean_p = st["mean_p"] * 100.0
            obs = st["obs"] * 100.0
            bias = st["bias_pct"]
            for b in range(bins):
                w.writerow([
                    r,
                    region_desc.get(r, r),
                    float(edges_pct[b]),
                    float(edges_pct[b + 1]),
                    int(counts[b]),
                    float(mean_p[b]) if np.isfinite(mean_p[b]) else float("nan"),
                    float(obs[b]) if np.isfinite(obs[b]) else float("nan"),
                    float(bias[b]) if np.isfinite(bias[b]) else float("nan"),
                ])


def plot_reliability_bias_heatmap(out_png: str,
                                 bias_matrix_pct: np.ndarray,
                                 bin_tick_labels: List[str],
                                 region_tick_labels: List[str],
                                 region_desc_lines: List[str],
                                 vmax: float = 0.8,
                                 title: str = "Reliability bias by probability bin (Model − Observation)") -> None:
    """Save a calibration-style heatmap of reliability bias.

    bias_matrix_pct: shape (R, B) in percentage points.
    """
    import matplotlib.pyplot as plt

    bias = np.asarray(bias_matrix_pct, dtype=np.float64)
    if bias.ndim != 2:
        raise ValueError("plot_reliability_bias_heatmap: bias_matrix_pct must be 2-D.")
    r, b = bias.shape

    # Symmetric diverging scale (like the paper figure).
    vmax_use = float(vmax)
    if not np.isfinite(vmax_use) or vmax_use <= 0:
        finite = bias[np.isfinite(bias)]
        if finite.size:
            vmax_use = float(np.nanmax(np.abs(finite)))
        else:
            vmax_use = 1.0
        vmax_use = max(vmax_use, 1e-6)

    fig_h = max(2.8, 0.45 * r + 1.6)
    fig_w = 12.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=850)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4.6, 2.4], wspace=0.15)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(bias, aspect="auto", interpolation="nearest", cmap="RdBu_r",
                   vmin=-vmax_use, vmax=vmax_use)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Probability of fire bin upper edge (%)", fontsize=10)
    ax.set_ylabel("Regions", fontsize=10)

    ax.set_xticks(np.arange(b))
    ax.set_xticklabels(bin_tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(r))
    ax.set_yticklabels(region_tick_labels, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Reliability bias (Model − Observation), percentage points", fontsize=9)

    # Over/under text similar to the example figure.
    ax.text(1.02, 0.85, "Over\nprediction", transform=ax.transAxes,
            va="top", ha="left", fontsize=8)
    ax.text(1.02, 0.15, "Under\nprediction", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=8)

    # Region descriptions panel (ensures every region is explicitly described in the figure itself).
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_title("Region definitions", fontsize=10, pad=6)
    desc_text = "\n".join(region_desc_lines)
    ax2.text(0.0, 1.0, desc_text, va="top", ha="left", fontsize=7, family="monospace")

    fig.tight_layout()
    fig.savefig(out_png, dpi=850, bbox_inches="tight")
    plt.close(fig)




def plot_calibration_curve(out_png: str,
                           stats: Dict,
                           edges_pct: List[float],
                           title: str = "Calibration curve (reliability diagram)",
                           xmax_pct: Optional[float] = None,
                           show_hist: bool = True,
                           min_count: int = 1) -> None:
    """Plot a classic calibration curve (reliability diagram).

    This uses binned statistics (mean predicted probability vs observed frequency).
    Optionally overlays a histogram (fraction of samples per bin) on a secondary axis.

    Args:
        out_png: Output PNG path.
        stats: Dict from ReliabilityBiasAccumulator.finalize() (counts, mean_p, obs, edges).
        edges_pct: Bin edges in percent (for labeling).
        title: Plot title.
        xmax_pct: Optional x/y max. If None, uses max(edges_pct).
        show_hist: If True, show sample fraction bars.
        min_count: Minimum samples required to plot a bin.
    """
    import matplotlib.pyplot as plt

    # ---------------- Poster styling knobs ----------------
    CAL_LW = 3.2         # Model calibration curve thickness
    PERFECT_LW = 2.4     # Perfect-diagonal thickness
    MARKER_SIZE = 6

    TITLE_FS = 14
    LABEL_FS = 13
    TICK_FS = 12         # <-- this controls the side numbers
    LEGEND_FS = 12

    GRID_ALPHA = 0.25
    BAR_ALPHA = 0.18
    # ------------------------------------------------------

    counts = stats.get('counts')
    mean_p = stats.get('mean_p')
    obs = stats.get('obs')
    if counts is None or mean_p is None or obs is None:
        raise ValueError('plot_calibration_curve: stats must include counts, mean_p, obs.')

    counts = np.asarray(counts)
    mean_p = np.asarray(mean_p)
    obs = np.asarray(obs)

    # Convert to percent for display.
    x = mean_p * 100.0
    y = obs * 100.0

    good = (counts >= int(min_count)) & np.isfinite(x) & np.isfinite(y)

    # Bin centers for histogram bars
    e = np.asarray(edges_pct, dtype=np.float64)
    centers = 0.5 * (e[:-1] + e[1:])
    width = np.diff(e)

    if xmax_pct is None:
        xmax_pct = float(np.nanmax(e)) if e.size else 100.0
    xmax_pct = float(xmax_pct)

    fig = plt.figure(figsize=(7.2, 5.4), dpi=850)
    ax = fig.add_subplot(111)

    # Perfect calibration line
    ax.plot([0.0, xmax_pct], [0.0, xmax_pct],
            linestyle='--', linewidth=PERFECT_LW, label='Perfect calibration')

    if np.any(good):
        ax.plot(x[good], y[good],
                marker='o', markersize=MARKER_SIZE,
                linewidth=CAL_LW, label='Model')
    else:
        ax.text(0.5, 0.5, 'No data in bins', transform=ax.transAxes,
                ha='center', va='center')

    # Titles/labels (bigger for poster)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel('Mean predicted probability (%)', fontsize=LABEL_FS)
    ax.set_ylabel('Observed fire frequency (%)', fontsize=LABEL_FS)

    ax.set_xlim(0.0, xmax_pct)
    ax.set_ylim(0.0, xmax_pct)

    # Bigger tick numbers (side numbers)
    ax.tick_params(axis='both', labelsize=TICK_FS)

    ax.grid(True, alpha=GRID_ALPHA)

    if show_hist:
        ax2 = ax.twinx()
        total = float(np.sum(counts))
        frac = (counts / total) if total > 0 else np.zeros_like(counts, dtype=np.float64)

        # Only show bars where we have any samples.
        ax2.bar(centers, frac, width=width, alpha=BAR_ALPHA,
                align='center', label='Sample fraction')

        ax2.set_ylabel('Fraction of samples', fontsize=LABEL_FS)
        ax2.set_ylim(0.0, max(1e-6, float(np.nanmax(frac)) * 1.15))

        # Bigger tick numbers on the right axis too
        ax2.tick_params(axis='y', labelsize=TICK_FS)

        # Build a combined legend (ax + ax2)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(
            h1 + h2, l1 + l2,
            loc='best',
            fontsize=LEGEND_FS,
            frameon=True,
            framealpha=0.92,
            borderpad=0.7,
            handlelength=2.6,
        )
    else:
        ax.legend(
            loc='best',
            fontsize=LEGEND_FS,
            frameon=True,
            framealpha=0.92,
            borderpad=0.7,
            handlelength=2.6,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=850, bbox_inches='tight')
    plt.close(fig)



def init_yearly_metrics_accumulator(ece_bins: int = 10, roc_bins: int = 600) -> YearlyMetricsAccumulator:
    ece_edges = np.linspace(0.0, 1.0, int(ece_bins) + 1, dtype=np.float64)
    roc_edges = np.linspace(0.0, 1.0, int(roc_bins) + 1, dtype=np.float64)
    return YearlyMetricsAccumulator(ece_edges=ece_edges, roc_hist=roc_hist_init(roc_edges))


def validate_yearly_metrics(m: Dict[str, float], name: str) -> bool:
    """
    Prints a compact validity confirmation and returns True/False.
    """
    ok = True
    msgs = []

    def _finite(x): return (x is not None) and np.isfinite(x)

    # ECE, Brier, ROC-AUC are in [0,1] in principle.
    for k in ["ece", "brier", "roc_auc"]:
        v = float(m.get(k, float("nan")))
        if not _finite(v) or v < -1e-6 or v > 1.0 + 1e-6:
            ok = False
            msgs.append(f"{k}={v:.6g} (expected ~[0,1])")

    # Logloss >= 0
    v = float(m.get("logloss", float("nan")))
    if not _finite(v) or v < -1e-6:
        ok = False
        msgs.append(f"logloss={v:.6g} (expected >=0)")

    # Corr in [-1,1]
    v = float(m.get("corr", float("nan")))
    if _finite(v) and (v < -1.0 - 1e-6 or v > 1.0 + 1e-6):
        ok = False
        msgs.append(f"corr={v:.6g} (expected [-1,1])")

    # Count
    n = int(m.get("n", 0))
    if n <= 0:
        ok = False
        msgs.append("n=0 (no valid samples)")

    if ok:
        print(f"[SpiderMetrics] {name}: VALID | "
              f"ECE={m['ece']:.4f} Brier={m['brier']:.4f} LogLoss={m['logloss']:.4f} "
              f"AUC={m['roc_auc']:.4f} Corr={m['corr']:.4f} (n={m['n']:,})")
    else:
        print(f"[SpiderMetrics] {name}: INVALID | " + "; ".join(msgs))

    return ok


# Fixed absolute spider scaling (identical across all regions)
# For low-is-better metrics we use an exponential decay score:
#   score(m) = exp(-m/tau), where tau = m50 / ln(2)  => score(m50)=0.5
SPIDER_M50 = {
    "ECE": 0.02,
    "Brier": 0.05,
    "LogLoss": 0.20,
}

def _spider_decay_score(m: float, m50: float) -> float:
    if not np.isfinite(m):
        return float("nan")
    m = max(float(m), 0.0)
    m50 = float(m50)
    if (not np.isfinite(m50)) or (m50 <= 0.0):
        return float("nan")
    return float(np.exp(-m * np.log(2.0) / m50))


def spider_scores_from_metrics(m: Dict[str, float]) -> Dict[str, float]:
    """
    Convert raw metrics to [0,1] spider scores where higher=better (farther from center).

    Scaling is FIXED and IDENTICAL for all regions (no region-specific min/max):
      - ROC-AUC: linear rescale from [0.5, 1.0] -> [0, 1]
      - Correlation: linear rescale from [-1, 1] -> [0, 1]
      - ECE/Brier/LogLoss: exponential decay with per-metric m50 (score=0.5 at m=m50)
    """
    ece = float(m.get("ece", float("nan")))
    brier = float(m.get("brier", float("nan")))
    logloss = float(m.get("logloss", float("nan")))
    auc = float(m.get("roc_auc", float("nan")))
    corr = float(m.get("corr", float("nan")))

    # Higher-is-better
    score_auc = ((auc - 0.5) / 0.5) if np.isfinite(auc) else float("nan")
    score_corr = ((corr + 1.0) / 2.0) if np.isfinite(corr) else float("nan")

    # Lower-is-better (exponential decay; score(m50)=0.5)
    score_ece = _spider_decay_score(ece, SPIDER_M50["ECE"])
    score_brier = _spider_decay_score(brier, SPIDER_M50["Brier"])
    score_logloss = _spider_decay_score(logloss, SPIDER_M50["LogLoss"])

    out = {
        "ECE": float(np.clip(score_ece, 0.0, 1.0)) if np.isfinite(score_ece) else float("nan"),
        "Brier": float(np.clip(score_brier, 0.0, 1.0)) if np.isfinite(score_brier) else float("nan"),
        "LogLoss": float(np.clip(score_logloss, 0.0, 1.0)) if np.isfinite(score_logloss) else float("nan"),
        "ROC-AUC": float(np.clip(score_auc, 0.0, 1.0)) if np.isfinite(score_auc) else float("nan"),
        "Correlation": float(np.clip(score_corr, 0.0, 1.0)) if np.isfinite(score_corr) else float("nan"),
    }
    return out

def spider_scores_relative_logminmax(
    series_metrics: Dict[str, Dict[str, float]],
    *,
    eps: float = 1e-12,
) -> Dict[str, Dict[str, float]]:
    """
    Per-metric min-max normalization across the provided series.

    - For low-is-better metrics (ECE/Brier/LogLoss), we first apply log10(v + eps) to expand
      the small-value range, then invert (smaller -> closer to 1).
    - For Correlation, we map [-1, 1] to [0, 1] before normalization.
    - For ROC-AUC we keep [0, 1].

    Returns: dict[series_name] -> dict[label] -> score in [0, 1]
    """
    # Pre-compute transformed values by (metric_label, series_name)
    transformed: Dict[str, Dict[str, float]] = {name: {} for name in series_metrics.keys()}

    for label in SPIDER_METRIC_ORDER:
        info = SPIDER_METRIC_INFO.get(label)
        if info is None:
            # If a label is unknown, skip; plotting functions will handle missing keys.
            continue

        key = info["key"]
        do_log = bool(info.get("log", False))
        transform_fn = info.get("transform", None)

        # Collect transformed values for this label across all series
        vals = []
        for name, m in series_metrics.items():
            v = float(m.get(key, float("nan")))
            if transform_fn is not None and np.isfinite(v):
                v = float(transform_fn(v))

            if np.isfinite(v) and do_log:
                # Guard: ECE/Brier/LogLoss should be >= 0; still be robust.
                v = max(v, 0.0)
                v = float(np.log10(v + eps))

            transformed[name][label] = v
            if np.isfinite(v):
                vals.append(v)

        if len(vals) < 2:
            # Not enough information to scale; set a neutral value.
            for name in series_metrics.keys():
                if np.isfinite(transformed[name][label]):
                    transformed[name][label] = 0.5
            continue

        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (abs(vmax - vmin) < 1e-18):
            for name in series_metrics.keys():
                if np.isfinite(transformed[name][label]):
                    transformed[name][label] = 0.5
            continue

        denom = vmax - vmin
        direction = info.get("direction", "high")
        for name in series_metrics.keys():
            v = transformed[name][label]
            if not np.isfinite(v):
                transformed[name][label] = float("nan")
                continue

            if direction == "low":
                score = (vmax - v) / denom
            else:
                score = (v - vmin) / denom

            transformed[name][label] = float(np.clip(score, 0.0, 1.0))

    return transformed


def _radar_angles(n_axes: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_axes, endpoint=False)
    return angles


def plot_spider_chart_single(scores: Dict[str, float], title: str, out_path: str, line_color: str = SPIDER_BASE_ORANGE) -> None:
    labels = SPIDER_METRIC_ORDER
    vals = np.array([scores.get(k, np.nan) for k in labels], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise ValueError("Spider chart scores contain NaNs; cannot plot.")

    n = len(labels)
    angles = _radar_angles(n)
    angles_closed = np.concatenate([angles, [angles[0]]])
    vals_closed = np.concatenate([vals, [vals[0]]])

    fig = plt.figure(figsize=(5.2, 4.6), dpi=850)
    ax = fig.add_subplot(111, polar=True)

    # Start at top, clockwise
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)

    # Light, dotted gridlines (like the reference image)
    ax.grid(True, linestyle=":", linewidth=0.8, color="#e3e3e3")
    ax.spines["polar"].set_color("#e3e3e3")
    ax.spines["polar"].set_linewidth(0.8)

    ax.plot(angles_closed, vals_closed, linewidth=2.0, color=line_color)
    ax.fill(angles_closed, vals_closed, color=line_color, alpha=0.12)

    ax.set_title(title, fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=850, bbox_inches="tight")
    plt.close(fig)


def plot_spider_chart_multi(series_scores: Dict[str, Dict[str, float]], title: str, out_path: str) -> None:
    labels = SPIDER_METRIC_ORDER
    n = len(labels)
    angles = _radar_angles(n)
    angles_closed = np.concatenate([angles, [angles[0]]])

    series_names = list(series_scores.keys())
    n_series = len(series_names)
    if n_series <= 0:
        raise ValueError("No series provided for multi-line spider chart.")

    # Orange-themed distinct shades
    colors = plt.cm.Oranges(np.linspace(0.35, 0.95, n_series))

    fig = plt.figure(figsize=(7.6, 6.0), dpi=850)
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)

    ax.grid(True, linestyle=":", linewidth=0.8, color="#e3e3e3")
    ax.spines["polar"].set_color("#e3e3e3")
    ax.spines["polar"].set_linewidth(0.8)

    for i, name in enumerate(series_names):
        s = series_scores[name]
        vals = np.array([s.get(k, np.nan) for k in labels], dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            continue
        vals_closed = np.concatenate([vals, [vals[0]]])
        ax.plot(angles_closed, vals_closed, linewidth=2.0, color=colors[i], label=name)
        ax.fill(angles_closed, vals_closed, color=colors[i], alpha=0.06)

    ax.set_title(title, fontsize=11, pad=14)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=850, bbox_inches="tight")
    plt.close(fig)

def compute_offset_auto(t_values: List[int], zarr_len: int) -> int:
    tmin = min(t_values)
    tmax = max(t_values)
    if tmax < zarr_len:
        return 0
    span = tmax - tmin + 1
    if span <= zarr_len:
        return tmin
    raise ValueError(
        f"Cannot infer zarr index offset: tif t range [{tmin},{tmax}] span={span} but zarr_len={zarr_len}. "
        f"Pass --zarr-index-offset N explicitly."
    )


def parse_offset_arg(s: str) -> Optional[int]:
    s = str(s).strip().lower()
    if s == "auto":
        return None
    return int(s)


def daily_dirs(base: str, which: List[str], top_fracs: List[float]) -> Dict[str, str]:
    ensure_dir(base)
    out = {}
    if "raw" in which:
        out["raw"] = os.path.join(base, "raw"); ensure_dir(out["raw"])
    if "percentile" in which:
        out["percentile"] = os.path.join(base, "percentile"); ensure_dir(out["percentile"])
    if "topk" in which:
        for k in top_fracs:
            key = f"topk_{int(round(k*100)):02d}"
            out[key] = os.path.join(base, key); ensure_dir(out[key])
    if "confusion" in which:
        out["confusion"] = os.path.join(base, "confusion"); ensure_dir(out["confusion"])
    if "capture" in which:
        out["capture"] = os.path.join(base, "capture"); ensure_dir(out["capture"])
    return out


def run_viz(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)

    items = list_pred_tifs(args.pred_dir)
    if not items:
        raise SystemExit(f"No pred_t*.tif files found in {args.pred_dir}")

    if args.max_days and args.max_days > 0:
        items = items[: args.max_days]

    t_values = [t for t, _ in items]

    raw_grid, nodata = grid_from_tif(items[0][1])
    normalizer = build_normalizer(raw_grid, wrap_longitude=args.wrap_longitude)
    grid = normalizer.grid
    H, W = grid.shape

    regions = build_region_groups()
    if args.regions:
        keep = set(args.regions)
        regions = [r for r in regions if r["name"] in keep]
        if not regions:
            raise SystemExit(f"--regions specified, but none matched. Available: {[r['name'] for r in build_region_groups()]}")

    peat_mask = np.ones((H, W), dtype=bool)
    if args.peat_mask_zarr:
        peat_arr = open_zarr_array(args.peat_mask_zarr)
        peat_data = read_zarr_2d(peat_arr, t_index=0)
        if peat_data.shape != (H, W):
            raise SystemExit(f"Peat mask shape {peat_data.shape} != GeoTIFF shape {(H, W)}")

        peat_data = normalizer.apply(peat_data)
        nodp = float(args.peat_nodata)
        peat_data = np.where(np.isclose(peat_data, nodp), np.nan, peat_data)
        peat_data = np.where(np.isfinite(peat_data), peat_data, np.nan)

        if args.peat_mask_mode == "valid":
            peat_mask = np.isfinite(peat_data)
        else:
            peat_mask = np.isfinite(peat_data) & (peat_data >= float(args.peat_min_fraction))

        n_peat = int(peat_mask.sum())
        if n_peat == 0:
            raise SystemExit("Peat mask contains 0 pixels after masking. For smap_wtd use --peat-mask-mode valid --peat-nodata -9999")

    weights: Optional[np.ndarray] = None
    if args.area_weighting == "coslat":
        w_row = np.cos(np.deg2rad(grid.lat_centers)).astype(np.float64)
        w_row = np.clip(w_row, 0.0, None)
        weights = np.repeat(w_row[:, None], W, axis=1)

    viirs_arr = open_zarr_array(args.viirs_zarr) if args.viirs_zarr else None
    fwi_arr = open_zarr_array(args.fwi_zarr) if args.fwi_zarr else None

    offset_arg = parse_offset_arg(args.zarr_index_offset)
    zarr_lens = []
    if viirs_arr is not None:
        zarr_lens.append(int(viirs_arr.shape[0]))
    if fwi_arr is not None:
        zarr_lens.append(int(fwi_arr.shape[0]))
    zarr_len = min(zarr_lens) if zarr_lens else 0

    if offset_arg is None and zarr_len > 0:
        offset = compute_offset_auto(t_values, zarr_len)
    else:
        offset = offset_arg or 0

    if zarr_len > 0:
        print(f"[Indexing] Using zarr_index = tif_t - {offset} (zarr_len={zarr_len}).")
    print(f"[Grid] flip_y={normalizer.flip_y} wrap_lon={normalizer.wrap_lon} cut_col={normalizer.cut_col}")

    lon_centers = (grid.x_edges[:-1] + grid.x_edges[1:]) / 2.0
    lat_centers = (grid.y_edges[:-1] + grid.y_edges[1:]) / 2.0
    lon2 = np.repeat(lon_centers[None, :], H, axis=0)
    lat2 = np.repeat(lat_centers[:, None], W, axis=1)

    region_masks: Dict[str, np.ndarray] = {}
    for rg in regions:
        in_any = np.zeros((H, W), dtype=bool)
        for (xmin, ymin, xmax, ymax) in rg["boxes"]:
            inside = (lon2 >= xmin) & (lon2 <= xmax) & (lat2 >= ymin) & (lat2 <= ymax)
            in_any |= inside
        region_masks[rg["name"]] = peat_mask & in_any

    daily_paths: Dict[str, str] = {}
    capture_writer = None
    capture_fh = None
    if args.daily_out_dir:
        daily_paths = daily_dirs(args.daily_out_dir, args.daily_which, args.top_fracs)
        print(f"[Daily] Writing daily products to: {args.daily_out_dir}")

        if "capture" in args.daily_which and viirs_arr is not None:
            csv_path = os.path.join(daily_paths["capture"], "daily_capture_by_region.csv")
            fieldnames = [
                "tif_t", "region", "area_frac",
                "model_captured", "model_total_fire", "model_capture_rate_pct",
                "fwi_captured", "fwi_total_fire", "fwi_capture_rate_pct",
            ]
            capture_fh = open(csv_path, "w", newline="")
            capture_writer = csv.DictWriter(capture_fh, fieldnames=fieldnames)
            capture_writer.writeheader()
            print(f"[Daily] Capture CSV: {csv_path}")

    pct_cmap, pct_norm, pct_bounds = daily_percentile_cmap_norm()
    pct_cmap_yearly = pct_cmap_continuous()

    try:
        plt.register_cmap(cmap=pct_cmap)
    except Exception:
        pass
    pct_ticks = pct_bounds.tolist()
    pct_ticklabels = [str(int(b)) for b in pct_bounds]

    roc_model_global = roc_fwi_global = None
    roc_model_regions: Dict[str, RocHist] = {}
    roc_fwi_regions: Dict[str, RocHist] = {}

    # Option 1 (exact AUC): store raw scores/labels to compute exact AUC at end.
    exact_model_scores_global: Optional[List[np.ndarray]] = None
    exact_model_labels_global: Optional[List[np.ndarray]] = None
    exact_fwi_scores_global: Optional[List[np.ndarray]] = None
    exact_fwi_labels_global: Optional[List[np.ndarray]] = None
    exact_model_scores_regions: Dict[str, List[np.ndarray]] = {}
    exact_model_labels_regions: Dict[str, List[np.ndarray]] = {}
    exact_fwi_scores_regions: Dict[str, List[np.ndarray]] = {}
    exact_fwi_labels_regions: Dict[str, List[np.ndarray]] = {}

    if args.make_roc:
        pmin, pmax = args.roc_prob_minmax
        edges = np.linspace(pmin, pmax, args.roc_nbins + 1)
        roc_model_global = roc_hist_init(edges)

        f_edges = np.linspace(0.0, args.roc_fwi_max, args.roc_nbins + 1)
        roc_fwi_global = roc_hist_init(f_edges)

        if args.roc_include_regions:
            for rg in regions:
                name = rg["name"]
                roc_model_regions[name] = roc_hist_init(edges)
                roc_fwi_regions[name] = roc_hist_init(f_edges)

        if args.roc_exact:
            exact_model_scores_global = []
            exact_model_labels_global = []
            exact_fwi_scores_global = []
            exact_fwi_labels_global = []
            if args.roc_include_regions:
                for rg in regions:
                    name = rg["name"]
                    exact_model_scores_regions[name] = []
                    exact_model_labels_regions[name] = []
                    exact_fwi_scores_regions[name] = []
                    exact_fwi_labels_regions[name] = []


    # Spider chart metric accumulators (yearly only)
    spider_global_acc: Optional[YearlyMetricsAccumulator] = None
    spider_region_accs: Dict[str, YearlyMetricsAccumulator] = {}
    if args.make_spider_charts:
        if viirs_arr is None:
            raise SystemExit("--make-spider-charts requires --viirs-zarr")
        spider_global_acc = init_yearly_metrics_accumulator(
            ece_bins=args.spider_ece_bins,
            roc_bins=args.spider_roc_bins,
        )
        for rg in regions:
            name = rg["name"]
            spider_region_accs[name] = init_yearly_metrics_accumulator(
                ece_bins=args.spider_ece_bins,
                roc_bins=args.spider_roc_bins,
            )
    # Reliability bias (calibration) accumulators (yearly only)
    rel_global_acc: Optional[ReliabilityBiasAccumulator] = None
    rel_region_accs: Dict[str, ReliabilityBiasAccumulator] = {}
    rel_edges_pct: List[float] = []

    if args.make_reliability_bias or args.make_calibration_plot:
        if viirs_arr is None:
            raise SystemExit("--make-reliability-bias/--make-calibration-plot requires --viirs-zarr")

        # Default (training-matched): fixed-width bins in FRACTION units
        if not args.reliability_use_edges_pct:
            w = float(args.reliability_bin_width)
            lo = float(args.reliability_bin_min)
            hi = float(args.reliability_bin_max)
            if not (w > 0 and np.isfinite(w)):
                raise SystemExit("--reliability-bin-width must be > 0")
            if not (hi > lo and np.isfinite(lo) and np.isfinite(hi)):
                raise SystemExit("--reliability-bin-max must be > --reliability-bin-min")

            edges = np.arange(lo, hi + 0.5 * w, w, dtype=np.float64)
            # ensure exact endpoints
            if edges.size == 0 or abs(edges[0] - lo) > 1e-12:
                edges = np.r_[lo, edges]
            if abs(edges[-1] - hi) > 1e-12:
                edges = np.r_[edges, hi]
            # always start at 0 for nicer plots
            if edges[0] > 0.0:
                edges = np.r_[0.0, edges]

            rel_edges = edges
            rel_edges_pct = [float(x * 100.0) for x in rel_edges]

        # Optional: explicit edges in PERCENT units
        else:
            rel_edges_pct = [float(x) for x in args.reliability_bins_pct]
            rel_edges_pct = sorted(set(rel_edges_pct))
            if len(rel_edges_pct) < 2:
                raise SystemExit("--reliability-bins-pct must contain at least two increasing values.")
            if rel_edges_pct[0] > 0.0:
                rel_edges_pct = [0.0] + rel_edges_pct
            rel_edges = np.asarray(rel_edges_pct, dtype=np.float64) / 100.0

        rel_global_acc = ReliabilityBiasAccumulator(rel_edges)
        for rg in regions:
            name = rg["name"]
            rel_region_accs[name] = ReliabilityBiasAccumulator(rel_edges)


    used_days = 0
    sum_pred = np.zeros((H, W), dtype=np.float64)
    cnt_pred = np.zeros((H, W), dtype=np.int32)

    sum_pred_pct = np.zeros((H, W), dtype=np.float64)
    cnt_pred_pct = np.zeros((H, W), dtype=np.int32)

    sum_fwi = np.zeros((H, W), dtype=np.float64) if fwi_arr is not None else None
    cnt_fwi = np.zeros((H, W), dtype=np.int32) if fwi_arr is not None else None

    sum_fwi_pct = np.zeros((H, W), dtype=np.float64) if fwi_arr is not None else None
    cnt_fwi_pct = np.zeros((H, W), dtype=np.int32) if fwi_arr is not None else None

    topk_counts_model: Dict[float, np.ndarray] = {k: np.zeros((H, W), dtype=np.int32) for k in args.top_fracs}
    topk_counts_fwi: Dict[float, np.ndarray] = {k: np.zeros((H, W), dtype=np.int32) for k in args.top_fracs} if fwi_arr is not None else {}

    fire_days = np.zeros((H, W), dtype=np.int32) if viirs_arr is not None else None

    hit_model = miss_model = fa_model = None
    hit_fwi = miss_fwi = fa_fwi = None
    if args.make_confusion_maps and viirs_arr is not None:
        hit_model = np.zeros((H, W), dtype=np.int32)
        miss_model = np.zeros((H, W), dtype=np.int32)
        fa_model = np.zeros((H, W), dtype=np.int32)
        if fwi_arr is not None:
            hit_fwi = np.zeros((H, W), dtype=np.int32)
            miss_fwi = np.zeros((H, W), dtype=np.int32)
            fa_fwi = np.zeros((H, W), dtype=np.int32)

    capture: Dict[str, Dict[str, np.ndarray]] = {}
    if args.make_capture_curves and viirs_arr is not None:
        fracs = list(args.capture_fracs)
        for rg in regions:
            capture[rg["name"]] = {
                "fracs": np.array(fracs, dtype=np.float64),
                "model_numer": np.zeros(len(fracs), dtype=np.float64),
                "model_denom": np.zeros(len(fracs), dtype=np.float64),
                "fwi_numer": np.zeros(len(fracs), dtype=np.float64),
                "fwi_denom": np.zeros(len(fracs), dtype=np.float64),
            }

    for tif_t, tif_path in items:
        pred = read_tif_2d(tif_path, nodata)
        pred = normalizer.apply(pred)
        pred = np.where(peat_mask, pred, np.nan)

        finite_pred = np.isfinite(pred)
        sum_pred[finite_pred] += pred[finite_pred]
        cnt_pred[finite_pred] += 1

        pred_pct = None
        if "percentile" in args.figures or (args.daily_out_dir and "percentile" in args.daily_which):
            pred_pct = compute_daily_percentiles(pred, peat_mask)
            if "percentile" in args.figures:
                fp = np.isfinite(pred_pct)
                sum_pred_pct[fp] += pred_pct[fp]
                cnt_pred_pct[fp] += 1

        daily_topk_model: Dict[float, np.ndarray] = {}
        daily_topk_fwi: Dict[float, np.ndarray] = {}

        if "topk" in args.figures or (args.daily_out_dir and ("topk" in args.daily_which or "confusion" in args.daily_which)):
            for k in args.top_fracs:
                m = topk_mask_daily(pred, peat_mask, k, weights)
                daily_topk_model[k] = m
                if "topk" in args.figures:
                    topk_counts_model[k][m] += 1

        zt = tif_t - offset

        fire = None
        viirs_label_mask = None
        if viirs_arr is not None:
            if not (0 <= zt < viirs_arr.shape[0]):
                raise SystemExit(
                    f"VIIRS index out of bounds: zt={zt} (from tif_t={tif_t} - offset={offset}), viirs_len={viirs_arr.shape[0]}. "
                    f"Pass correct --zarr-index-offset."
                )
            v = read_zarr_2d(viirs_arr, t_index=zt)
            if v.shape != (H, W):
                raise SystemExit(f"VIIRS slice shape {v.shape} != GeoTIFF shape {(H, W)} for zt={zt}")
            v = normalizer.apply(v)
            v = np.where(peat_mask, v, np.nan)

            viirs_label_mask = np.isfinite(v) & peat_mask
            fire = viirs_label_mask & (v > args.viirs_threshold)

            if fire_days is not None:
                fire_days[fire] += 1

            if spider_global_acc is not None:
                y_float = fire.astype(np.float32)
                # Use the same mask used to derive the labels
                spider_global_acc.update(pred, y_float, viirs_label_mask)
                for rg in regions:
                    name = rg["name"]
                    spider_region_accs[name].update(pred, y_float, viirs_label_mask & region_masks[name])


            if rel_global_acc is not None:
                # Reliability bias uses the same label mask as yearly metrics (peat & valid label pixels).
                rel_global_acc.update(pred, fire, viirs_label_mask)
                for rg in regions:
                    name = rg["name"]
                    rel_region_accs[name].update(pred, fire, viirs_label_mask & region_masks[name])


        f = None
        f_pct = None
        if fwi_arr is not None:
            if not (0 <= zt < fwi_arr.shape[0]):
                raise SystemExit(
                    f"FWI index out of bounds: zt={zt} (from tif_t={tif_t} - offset={offset}), fwi_len={fwi_arr.shape[0]}. "
                    f"Pass correct --zarr-index-offset."
                )
            f = read_zarr_2d(fwi_arr, t_index=zt)
            if f.shape != (H, W):
                raise SystemExit(f"FWI slice shape {f.shape} != GeoTIFF shape {(H, W)} for zt={zt}")
            f = normalizer.apply(f)
            f = np.where(peat_mask, f, np.nan)

            ff = np.isfinite(f)
            sum_fwi[ff] += f[ff]
            cnt_fwi[ff] += 1

            if "percentile" in args.figures or (args.daily_out_dir and "percentile" in args.daily_which):
                f_pct = compute_daily_percentiles(f, peat_mask)
                if "percentile" in args.figures:
                    ffp = np.isfinite(f_pct)
                    sum_fwi_pct[ffp] += f_pct[ffp]
                    cnt_fwi_pct[ffp] += 1

            if "topk" in args.figures or (args.daily_out_dir and ("topk" in args.daily_which or "confusion" in args.daily_which)):
                for k in args.top_fracs:
                    mf = topk_mask_daily(f, peat_mask, k, weights)
                    daily_topk_fwi[k] = mf
                    if "topk" in args.figures:
                        topk_counts_fwi[k][mf] += 1

        if fire is not None:
            if args.make_confusion_maps and hit_model is not None:
                alert_m = topk_mask_daily(pred, peat_mask, args.confusion_top_frac, weights)
                hit_model[alert_m & fire] += 1
                miss_model[(~alert_m) & fire & peat_mask] += 1
                fa_model[alert_m & (~fire) & peat_mask] += 1

            if args.make_capture_curves:
                for rg in regions:
                    rmask = region_masks[rg["name"]]
                    st = capture[rg["name"]]
                    update_capture_stats(pred, fire, rmask, st["fracs"].tolist(), weights, st["model_numer"], st["model_denom"])

        if (fire is not None) and (f is not None):
            if args.make_confusion_maps and hit_fwi is not None:
                alert_f = topk_mask_daily(f, peat_mask, args.confusion_top_frac, weights)
                hit_fwi[alert_f & fire] += 1
                miss_fwi[(~alert_f) & fire & peat_mask] += 1
                fa_fwi[alert_f & (~fire) & peat_mask] += 1

            if args.make_capture_curves:
                for rg in regions:
                    rmask = region_masks[rg["name"]]
                    st = capture[rg["name"]]
                    update_capture_stats(f, fire, rmask, st["fracs"].tolist(), weights, st["fwi_numer"], st["fwi_denom"])

        if args.make_roc and (fire is not None) and (viirs_label_mask is not None) and (roc_model_global is not None):
            pmin, pmax = float(args.roc_prob_minmax[0]), float(args.roc_prob_minmax[1])
            mask_m = viirs_label_mask & np.isfinite(pred)
            if mask_m.any():
                scores = np.clip(pred[mask_m].astype(np.float64), pmin, pmax)
                labels = fire[mask_m].astype(bool)
                roc_hist_update(roc_model_global, scores, labels)
                if args.roc_exact:
                        assert exact_model_scores_global is not None and exact_model_labels_global is not None
                        # Store float32/uint8 to reduce memory footprint.
                        exact_model_scores_global.append(scores.astype(np.float32, copy=False))
                        exact_model_labels_global.append(labels.astype(np.uint8, copy=False))
                if args.roc_include_regions:
                    for rg in regions:
                        rm = region_masks[rg["name"]] & mask_m
                        if rm.any():
                            r_scores = np.clip(pred[rm].astype(np.float64), pmin, pmax)
                            r_labels = fire[rm].astype(bool)
                            roc_hist_update(roc_model_regions[rg["name"]], r_scores, r_labels)
                            if args.roc_exact:
                                exact_model_scores_regions[rg["name"]].append(r_scores.astype(np.float32, copy=False))
                                exact_model_labels_regions[rg["name"]].append(r_labels.astype(np.uint8, copy=False))

            if (f is not None) and (roc_fwi_global is not None):
                fmax = float(args.roc_fwi_max)
                mask_f = viirs_label_mask & np.isfinite(f)
                if mask_f.any():
                    scores_f = np.clip(f[mask_f].astype(np.float64), 0.0, fmax)
                    labels_f = fire[mask_f].astype(bool)
                    roc_hist_update(roc_fwi_global, scores_f, labels_f)
                    if args.roc_exact:
                        assert exact_fwi_scores_global is not None and exact_fwi_labels_global is not None
                        exact_fwi_scores_global.append(scores_f.astype(np.float32, copy=False))
                        exact_fwi_labels_global.append(labels_f.astype(np.uint8, copy=False))
                    if args.roc_include_regions:
                        for rg in regions:
                            rm = region_masks[rg["name"]] & mask_f
                            if rm.any():
                                r_scores_f = np.clip(f[rm].astype(np.float64), 0.0, fmax)
                                r_labels_f = fire[rm].astype(bool)
                                roc_hist_update(roc_fwi_regions[rg["name"]], r_scores_f, r_labels_f)
                                if args.roc_exact:
                                    exact_fwi_scores_regions[rg["name"]].append(r_scores_f.astype(np.float32, copy=False))
                                    exact_fwi_labels_regions[rg["name"]].append(r_labels_f.astype(np.uint8, copy=False))

        # DAILY OUTPUTS (same as your v5 script)
        if args.daily_out_dir:
            tag = f"t{tif_t:05d}"

            if "raw" in args.daily_which:
                layers_raw: List[Layer] = []
                if f is not None:
                    layers_raw.append(Layer(f"Daily FWI ({tag})", f, "plasma", 0.0, args.raw_fwi_vmax, "FWI", "max"))
                layers_raw.append(Layer(f"Daily model probability of fire ({tag})", pred, "RdYlGn_r", 0.0, args.raw_prob_vmax, "Probability", "max"))
                if fire is not None:
                    fire_layer = np.where(fire, 1.0, np.nan).astype(np.float32)
                    layers_raw.append(Layer(f"Daily VIIRS fire mask ({tag})", fire_layer, "inferno", 0.0, 1.0, "Fire (1=yes)", "neither"))

                out_png = os.path.join(daily_paths["raw"], f"daily_raw_global__{tag}.png")
                plot_stacked_cartopy(layers_raw, grid, out_png, args.daily_dpi, args.daily_figwidth, args.daily_row_height,
                                     args.coastline_lw, args.border_lw, args.gridline_alpha, extent=None, peat_mask=peat_mask, peat_alpha=0.20)

                if args.daily_include_regions:
                    for rg in regions:
                        ext = extent_from_boxes(rg["boxes"], pad_deg=args.region_pad_deg)
                        safe_name = rg["name"].replace(" ", "_").replace("/", "_")
                        out_png = os.path.join(daily_paths["raw"], f"daily_raw_region__{safe_name}__{tag}.png")
                        plot_stacked_cartopy(layers_raw, grid, out_png, args.daily_dpi, args.daily_figwidth, args.daily_row_height,
                                             args.coastline_lw, args.border_lw, args.gridline_alpha, extent=ext, peat_mask=peat_mask, peat_alpha=0.20)

            if "percentile" in args.daily_which:
                # Ensure both model and FWI percentiles are available for daily percentile products.
                if pred_pct is None:
                    pred_pct = compute_daily_percentiles(pred, peat_mask)
                if (f is not None) and (f_pct is None):
                    f_pct = compute_daily_percentiles(f, peat_mask)
                layers_pct: List[Layer] = []
                if f_pct is not None:
                    layers_pct.append(Layer(f"Daily FWI percentile within peatlands ({tag})", f_pct, pct_cmap, norm=pct_norm,
                                            vmin=0.0, vmax=100.0, cbar_label="Percentile bins", extend="neither",
                                            cbar_ticks=pct_ticks, cbar_ticklabels=pct_ticklabels))
                layers_pct.append(Layer(f"Daily model percentile within peatlands ({tag})", pred_pct, pct_cmap, norm=pct_norm,
                                        vmin=0.0, vmax=100.0, cbar_label="Percentile bins", extend="neither",
                                        cbar_ticks=pct_ticks, cbar_ticklabels=pct_ticklabels))
                if fire is not None:
                    fire_layer = np.where(fire, 1.0, np.nan).astype(np.float32)
                    layers_pct.append(Layer(f"Daily VIIRS fire mask ({tag})", fire_layer, "inferno", 0.0, 1.0, "Fire (1=yes)", "neither"))

                out_png = os.path.join(daily_paths["percentile"], f"daily_percentile_global__{tag}.png")
                plot_stacked_cartopy(layers_pct, grid, out_png, args.daily_dpi, args.daily_figwidth, args.daily_row_height,
                                     args.coastline_lw, args.border_lw, args.gridline_alpha, extent=None, peat_mask=peat_mask, peat_alpha=0.20)

                if args.daily_include_regions:
                    for rg in regions:
                        ext = extent_from_boxes(rg["boxes"], pad_deg=args.region_pad_deg)
                        safe_name = rg["name"].replace(" ", "_").replace("/", "_")
                        out_png = os.path.join(daily_paths["percentile"], f"daily_percentile_region__{safe_name}__{tag}.png")
                        plot_stacked_cartopy(layers_pct, grid, out_png, args.daily_dpi, args.daily_figwidth, args.daily_row_height,
                                             args.coastline_lw, args.border_lw, args.gridline_alpha, extent=ext, peat_mask=peat_mask, peat_alpha=0.20)

            if "topk" in args.daily_which:
                for k in args.top_fracs:
                    m_alert = daily_topk_model.get(k, topk_mask_daily(pred, peat_mask, k, weights))
                    model_alert_score = np.where(m_alert, pred, np.nan).astype(np.float32)

                    layers_topk: List[Layer] = [
                        Layer(
                            title=f"Daily MODEL top-{int(round(k*100))}% alert area (score shown) ({tag})",
                            data=model_alert_score,
                            cmap="RdYlGn_r",
                            vmin=0.0,
                            vmax=args.raw_prob_vmax,
                            cbar_label="Probability (within alert)",
                            extend="max",
                        )
                    ]
                    if f is not None and (k in daily_topk_fwi):
                        f_alert = daily_topk_fwi[k]
                        f_alert_score = np.where(f_alert, f, np.nan).astype(np.float32)
                        layers_topk.append(Layer(
                            title=f"Daily FWI top-{int(round(k*100))}% alert area (score shown) ({tag})",
                            data=f_alert_score,
                            cmap="plasma",
                            vmin=0.0,
                            vmax=args.raw_fwi_vmax,
                            cbar_label="FWI (within alert)",
                            extend="max",
                        ))
                    if fire is not None:
                        fire_layer = np.where(fire, 1.0, np.nan).astype(np.float32)
                        layers_topk.append(Layer(f"Daily VIIRS fire mask ({tag})", fire_layer, "inferno", 0.0, 1.0, "Fire (1=yes)", "neither"))

                    key = f"topk_{int(round(k*100)):02d}"
                    out_png = os.path.join(daily_paths[key], f"daily_{key}_global__{tag}.png")
                    plot_stacked_cartopy(layers_topk, grid, out_png, args.daily_dpi, args.daily_figwidth, args.daily_row_height,
                                         args.coastline_lw, args.border_lw, args.gridline_alpha, extent=None, peat_mask=peat_mask, peat_alpha=0.20)

                    if args.daily_include_regions:
                        for rg in regions:
                            ext = extent_from_boxes(rg["boxes"], pad_deg=args.region_pad_deg)
                            safe_name = rg["name"].replace(" ", "_").replace("/", "_")
                            out_png = os.path.join(daily_paths[key], f"daily_{key}_region__{safe_name}__{tag}.png")
                            plot_stacked_cartopy(layers_topk, grid, out_png, args.daily_dpi, args.daily_figwidth, args.daily_row_height,
                                                 args.coastline_lw, args.border_lw, args.gridline_alpha, extent=ext, peat_mask=peat_mask, peat_alpha=0.20)

            if "confusion" in args.daily_which and (fire is not None):
                alert_m = topk_mask_daily(pred, peat_mask, args.confusion_top_frac, weights)
                hit = np.where(alert_m & fire, 1.0, np.nan).astype(np.float32)
                fa = np.where(alert_m & (~fire) & peat_mask, 1.0, np.nan).astype(np.float32)
                miss = np.where((~alert_m) & fire & peat_mask, 1.0, np.nan).astype(np.float32)

                layers_conf = [
                    Layer(f"Daily MODEL hits (top {args.confusion_top_frac*100:.1f}% alerts) ({tag})", hit, "YlGn", 0.0, 1.0, "Hit (1=yes)", "neither"),
                    Layer(f"Daily MODEL false alarms ({tag})", fa, "inferno", 0.0, 1.0, "FA (1=yes)", "neither"),
                    Layer(f"Daily MODEL misses ({tag})", miss, "inferno", 0.0, 1.0, "Miss (1=yes)", "neither"),
                ]
                out_png = os.path.join(daily_paths["confusion"], f"daily_confusion_model_global__{tag}.png")
                plot_stacked_cartopy(layers_conf, grid, out_png, args.daily_dpi, args.daily_figwidth, args.daily_row_height,
                                     args.coastline_lw, args.border_lw, args.gridline_alpha, extent=None, peat_mask=peat_mask, peat_alpha=0.20)

            if (
                "capture" in args.daily_which
                and (fire is not None)
                and (capture_writer is not None)
            ):
                fracs = list(args.capture_fracs)
                for rg in regions:
                    rmask = region_masks[rg["name"]]
                    numer_m = np.zeros(len(fracs), dtype=np.float64)
                    denom_m = np.zeros(len(fracs), dtype=np.float64)
                    update_capture_stats(pred, fire, rmask, fracs, weights, numer_m, denom_m)

                    numer_f = denom_f = None
                    if f is not None:
                        numer_f = np.zeros(len(fracs), dtype=np.float64)
                        denom_f = np.zeros(len(fracs), dtype=np.float64)
                        update_capture_stats(f, fire, rmask, fracs, weights, numer_f, denom_f)

                    for i, frac in enumerate(fracs):
                        row = {
                            "tif_t": str(tif_t),
                            "region": rg["name"],
                            "area_frac": f"{frac:.6f}",
                            "model_captured": f"{numer_m[i]:.6f}",
                            "model_total_fire": f"{denom_m[i]:.6f}",
                            "model_capture_rate_pct": f"{(100.0 * numer_m[i] / denom_m[i]) if denom_m[i] > 0 else np.nan:.6f}",
                            "fwi_captured": "",
                            "fwi_total_fire": "",
                            "fwi_capture_rate_pct": "",
                        }
                        if numer_f is not None and denom_f is not None:
                            row.update(
                                {
                                    "fwi_captured": f"{numer_f[i]:.6f}",
                                    "fwi_total_fire": f"{denom_f[i]:.6f}",
                                    "fwi_capture_rate_pct": f"{(100.0 * numer_f[i] / denom_f[i]) if denom_f[i] > 0 else np.nan:.6f}",
                                }
                            )
                        capture_writer.writerow(row)

        used_days += 1

    if capture_fh is not None:
        capture_fh.close()

    # YEARLY finalize
    mean_pred = np.full((H, W), np.nan, dtype=np.float32)
    m = cnt_pred > 0
    mean_pred[m] = (sum_pred[m] / cnt_pred[m]).astype(np.float32)

    mean_pred_pct = None
    if "percentile" in args.figures:
        mean_pred_pct = np.full((H, W), np.nan, dtype=np.float32)
        mp = cnt_pred_pct > 0
        mean_pred_pct[mp] = (sum_pred_pct[mp] / cnt_pred_pct[mp]).astype(np.float32)

    mean_fwi = None
    mean_fwi_pct = None
    if fwi_arr is not None and sum_fwi is not None and cnt_fwi is not None:
        mean_fwi = np.full((H, W), np.nan, dtype=np.float32)
        mf = cnt_fwi > 0
        mean_fwi[mf] = (sum_fwi[mf] / cnt_fwi[mf]).astype(np.float32)

        if "percentile" in args.figures and sum_fwi_pct is not None and cnt_fwi_pct is not None:
            mean_fwi_pct = np.full((H, W), np.nan, dtype=np.float32)
            mfp = cnt_fwi_pct > 0
            mean_fwi_pct[mfp] = (sum_fwi_pct[mfp] / cnt_fwi_pct[mfp]).astype(np.float32)

    fire_count = None
    fire_freq = None
    if fire_days is not None:
        fire_count = np.where(fire_days > 0, fire_days.astype(np.float32), np.nan)
        fire_freq = np.where(
            fire_days > 0,
            (fire_days.astype(np.float32) / max(1, used_days)) * 100.0,
            np.nan,
        )

    # YEARLY global stacks
    if "raw" in args.figures:
        layers: List[Layer] = []
        if mean_fwi is not None:
            layers.append(Layer("Average Fire Weather Index (FWI)", mean_fwi, "plasma", 0.0, args.raw_fwi_vmax, "FWI", "max"))
        layers.append(Layer("Average model probability of fire (peatlands only)", mean_pred, "RdYlGn_r", 0.0, args.raw_prob_vmax, "Probability", "max"))
        if fire_count is not None:
            vmax_fire = args.fireday_vmax if args.fireday_vmax > 0 else _auto_vmax(fire_count, fallback=1.0, q=99.0)
            layers.append(Layer("Number of days with recorded fire activity (VIIRS, peatlands only)", fire_count, "inferno", 0.0, vmax_fire, "Days", "max"))

        out_path = os.path.join(args.out_dir, "yearly_raw_cartopy.png")
        plot_stacked_cartopy(
            layers, grid, out_path, args.dpi, args.figwidth, args.row_height,
            args.coastline_lw, args.border_lw, args.gridline_alpha,
            extent=None, peat_mask=peat_mask, peat_alpha=0.20,
        )
        print(f"[Saved] {out_path}")

    if "percentile" in args.figures and (mean_pred_pct is not None):
        layers = []
        if mean_fwi_pct is not None:
            layers.append(Layer("Average daily FWI percentile (within peatlands)", mean_fwi_pct, pct_cmap_yearly, 0.0, 100.0, "Percentile (0–100)", "neither"))
        layers.append(Layer("Average daily model percentile (within peatlands)", mean_pred_pct, pct_cmap_yearly, 0.0, 100.0, "Percentile (0–100)", "neither"))
        if fire_freq is not None:
            vmax_ff = args.freq_vmax if args.freq_vmax > 0 else _auto_vmax(fire_freq, fallback=5.0, q=99.0)
            ff_norm = make_firefreq_norm(fire_freq, vmax_ff, args.firefreq_scale, args.firefreq_gamma, args.firefreq_log_vmin)
            ff_data = fire_freq
            if args.firefreq_scale == "log":
                ff_data = np.where(np.isfinite(ff_data) & (ff_data > 0.0), ff_data, np.nan).astype(np.float32)
            layers.append(Layer("VIIRS fire-day frequency (percent of days, peatlands only)", ff_data, "inferno", 0.0, vmax_ff, "% of days", "max", norm=ff_norm))

        out_path = os.path.join(args.out_dir, "yearly_percentile_cartopy.png")
        plot_stacked_cartopy(
            layers, grid, out_path, args.dpi, args.figwidth, args.row_height,
            args.coastline_lw, args.border_lw, args.gridline_alpha,
            extent=None, peat_mask=peat_mask, peat_alpha=0.20,
        )
        print(f"[Saved] {out_path}")

    if "topk" in args.figures:
        for k in args.top_fracs:
            freq_model = (topk_counts_model[k].astype(np.float32) / max(1, used_days)) * 100.0
            freq_fwi = None
            if fwi_arr is not None:
                freq_fwi = (topk_counts_fwi[k].astype(np.float32) / max(1, used_days)) * 100.0

            if args.freq_vmax > 0:
                vmax_shared = args.freq_vmax
            else:
                vmax_shared = _auto_vmax(freq_model, fallback=5.0, q=99.0)
                if freq_fwi is not None:
                    vmax_shared = max(vmax_shared, _auto_vmax(freq_fwi, fallback=5.0, q=99.0))

            layers = [
                Layer(f"Model top-{int(round(k*100))}% hotspot frequency (percent of days, peatlands only)", freq_model, "RdYlGn_r", 0.0, vmax_shared, "% of days", "max")
            ]
            if freq_fwi is not None:
                layers.append(Layer(f"FWI top-{int(round(k*100))}% hotspot frequency (percent of days, peatlands only)", freq_fwi, "RdYlGn_r", 0.0, vmax_shared, "% of days", "max"))
            if fire_freq is not None:
                vmax_ff = args.freq_vmax if args.freq_vmax > 0 else _auto_vmax(fire_freq, fallback=5.0, q=99.0)
                ff_norm = make_firefreq_norm(fire_freq, vmax_ff, args.firefreq_scale, args.firefreq_gamma, args.firefreq_log_vmin)
                ff_data = fire_freq
                if args.firefreq_scale == "log":
                    ff_data = np.where(np.isfinite(ff_data) & (ff_data > 0.0), ff_data, np.nan).astype(np.float32)
                layers.append(Layer("VIIRS fire-day frequency (percent of days, peatlands only)", ff_data, "inferno", 0.0, vmax_ff, "% of days", "max", norm=ff_norm))
            layers.append(Layer("Average model probability of fire (peatlands only)", mean_pred, "RdYlGn_r", 0.0, args.raw_prob_vmax, "Probability", "max"))

            out_path = os.path.join(args.out_dir, f"yearly_topk_{int(round(k*100)):02d}_cartopy.png")
            plot_stacked_cartopy(
                layers, grid, out_path, args.dpi, args.figwidth, args.row_height,
                args.coastline_lw, args.border_lw, args.gridline_alpha,
                extent=None, peat_mask=peat_mask, peat_alpha=0.20,
            )
            print(f"[Saved] {out_path}")

    # Region zooms
    if args.plot_regions:
        for rg in regions:
            ext = extent_from_boxes(rg["boxes"], pad_deg=args.region_pad_deg)
            safe_name = rg["name"].replace(" ", "_").replace("/", "_")

            layers_raw: List[Layer] = []
            if mean_fwi is not None:
                layers_raw.append(Layer(f"{rg['name']} — Average FWI (peatlands only)", mean_fwi, "plasma", 0.0, args.raw_fwi_vmax, "FWI", "max"))
            layers_raw.append(Layer(f"{rg['name']} — Average model probability of fire (peatlands only)", mean_pred, "RdYlGn_r", 0.0, args.raw_prob_vmax, "Probability", "max"))
            if fire_count is not None and np.isfinite(fire_count).any():
                vmax_fire = args.fireday_vmax if args.fireday_vmax > 0 else _auto_vmax(fire_count, fallback=1.0, q=99.0)
                layers_raw.append(Layer(f"{rg['name']} — VIIRS fire-days (count, peatlands only)", fire_count, "inferno", 0.0, vmax_fire, "Days", "max"))

            out_png = os.path.join(args.out_dir, f"region_raw__{safe_name}.png")
            plot_stacked_cartopy(layers_raw, grid, out_png, args.dpi, args.figwidth, args.row_height,
                                 args.coastline_lw, args.border_lw, args.gridline_alpha,
                                 extent=ext, peat_mask=peat_mask, peat_alpha=0.20)
            print(f"[Saved] {out_png}")

            if ("percentile" in args.figures) and (mean_pred_pct is not None):
                layers_pct: List[Layer] = []
                if mean_fwi_pct is not None:
                    layers_pct.append(Layer(f"{rg['name']} — Avg daily FWI percentile (within peatlands)", mean_fwi_pct, pct_cmap_yearly, 0.0, 100.0, "Percentile (0–100)", "neither"))
                layers_pct.append(Layer(f"{rg['name']} — Avg daily model percentile (within peatlands)", mean_pred_pct, pct_cmap_yearly, 0.0, 100.0, "Percentile (0–100)", "neither"))
                if fire_freq is not None and np.isfinite(fire_freq).any():
                    vmax_ff = args.freq_vmax if args.freq_vmax > 0 else _auto_vmax(fire_freq, fallback=5.0, q=99.0)
                    ff_norm = make_firefreq_norm(fire_freq, vmax_ff, args.firefreq_scale, args.firefreq_gamma, args.firefreq_log_vmin)
                    ff_data = fire_freq
                    if args.firefreq_scale == "log":
                        ff_data = np.where(np.isfinite(ff_data) & (ff_data > 0.0), ff_data, np.nan).astype(np.float32)
                    layers_pct.append(Layer(f"{rg['name']} — VIIRS fire-day frequency (% of days, peatlands only)", ff_data, "inferno", 0.0, vmax_ff, "% of days", "max", norm=ff_norm))

                out_png = os.path.join(args.out_dir, f"region_percentile__{safe_name}.png")
                plot_stacked_cartopy(layers_pct, grid, out_png, args.dpi, args.figwidth, args.row_height,
                                     args.coastline_lw, args.border_lw, args.gridline_alpha,
                                     extent=ext, peat_mask=peat_mask, peat_alpha=0.20)
                print(f"[Saved] {out_png}")

    # Capture curves
    if args.make_capture_curves and viirs_arr is not None:
        for rg in regions:
            st = capture[rg["name"]]
            fracs = st["fracs"].astype(float).tolist()

            model_rate = np.where(st["model_denom"] > 0, 100.0 * st["model_numer"] / st["model_denom"], np.nan)
            fwi_rate = None
            if fwi_arr is not None:
                fwi_rate = np.where(st["fwi_denom"] > 0, 100.0 * st["fwi_numer"] / st["fwi_denom"], np.nan)

            out_png = os.path.join(args.out_dir, f"capture_curve__{rg['name'].replace(' ','_').replace('/','_')}.png")
            plot_capture_curve(fracs=fracs, model_rate=model_rate, fwi_rate=fwi_rate, title=f"Fire-capture curve — {rg['name']}", out_png=out_png)
            print(f"[Saved] {out_png}")

    # Confusion maps
    if args.make_confusion_maps and (hit_model is not None):
        hit_m = np.where(hit_model > 0, (hit_model.astype(np.float32) / max(1, used_days)) * 100.0, np.nan)
        fa_m = np.where(fa_model > 0, (fa_model.astype(np.float32) / max(1, used_days)) * 100.0, np.nan)
        miss_m = np.where(miss_model > 0, miss_model.astype(np.float32), np.nan)

        vmax_hit = args.fireday_vmax if args.fireday_vmax > 0 else _auto_vmax(hit_m, fallback=1.0, q=99.0)
        vmax_fa = args.fireday_vmax if args.fireday_vmax > 0 else _auto_vmax(fa_m, fallback=1.0, q=99.0)
        vmax_miss = args.fireday_vmax if args.fireday_vmax > 0 else _auto_vmax(miss_m, fallback=1.0, q=99.0)

        layers = [
            Layer(f"MODEL Hits (% of days) — daily top {args.confusion_top_frac*100:.1f}% peat area", hit_m, "YlGn", 0.0, vmax_hit, "% of days"),
            Layer("MODEL False alarms (% of days)", fa_m, "inferno", 0.0, vmax_fa, "% of days"),
            Layer("MODEL Missed fire-days (count)", miss_m, "inferno", 0.0, vmax_miss, "Days"),
        ]
        out_png = os.path.join(args.out_dir, f"confusion_model_top{int(round(args.confusion_top_frac*100)):02d}_global.png")
        plot_stacked_cartopy(layers, grid, out_png, args.dpi, args.figwidth, args.row_height,
                             args.coastline_lw, args.border_lw, args.gridline_alpha,
                             extent=None, peat_mask=peat_mask, peat_alpha=0.20)
        print(f"[Saved] {out_png}")

        if hit_fwi is not None:
            hit_f = np.where(hit_fwi > 0, (hit_fwi.astype(np.float32) / max(1, used_days)) * 100.0, np.nan)
            fa_f = np.where(fa_fwi > 0, (fa_fwi.astype(np.float32) / max(1, used_days)) * 100.0, np.nan)
            miss_f = np.where(miss_fwi > 0, miss_fwi.astype(np.float32), np.nan)

            vmax_hit_f = args.fireday_vmax if args.fireday_vmax > 0 else _auto_vmax(hit_f, fallback=1.0, q=99.0)
            vmax_fa_f = args.fireday_vmax if args.fireday_vmax > 0 else _auto_vmax(fa_f, fallback=1.0, q=99.0)
            vmax_miss_f = args.fireday_vmax if args.fireday_vmax > 0 else _auto_vmax(miss_f, fallback=1.0, q=99.0)

            layers = [
                Layer(f"FWI Hits (% of days) — daily top {args.confusion_top_frac*100:.1f}% peat area", hit_f, "YlGn", 0.0, vmax_hit_f, "% of days"),
                Layer("FWI False alarms (% of days)", fa_f, "inferno", 0.0, vmax_fa_f, "% of days"),
                Layer("FWI Missed fire-days (count)", miss_f, "inferno", 0.0, vmax_miss_f, "Days"),
            ]
            out_png = os.path.join(args.out_dir, f"confusion_fwi_top{int(round(args.confusion_top_frac*100)):02d}_global.png")
            plot_stacked_cartopy(layers, grid, out_png, args.dpi, args.figwidth, args.row_height,
                                 args.coastline_lw, args.border_lw, args.gridline_alpha,
                                 extent=None, peat_mask=peat_mask, peat_alpha=0.20)
            print(f"[Saved] {out_png}")

    # ROC
    if args.make_roc and (roc_model_global is not None):
        fpr_m, tpr_m, auc_m = roc_from_hist(roc_model_global)

        fpr_f: Optional[np.ndarray] = None
        tpr_f: Optional[np.ndarray] = None
        auc_f: Optional[float] = None
        if roc_fwi_global is not None:
            fpr_f, tpr_f, auc_f = roc_from_hist(roc_fwi_global)

        # Option 1: compute exact (unbinned) AUC, but keep plotting the binned ROC curves.
        auc_m_exact: Optional[float] = None
        auc_f_exact: Optional[float] = None
        if args.roc_exact:
            try:
                if exact_model_scores_global is not None and len(exact_model_scores_global) > 0:
                    auc_m_exact = auc_exact_from_scores(
                        np.concatenate(exact_model_scores_global),
                        np.concatenate(exact_model_labels_global),
                    )
                if (
                    roc_fwi_global is not None
                    and exact_fwi_scores_global is not None
                    and len(exact_fwi_scores_global) > 0
                ):
                    auc_f_exact = auc_exact_from_scores(
                        np.concatenate(exact_fwi_scores_global),
                        np.concatenate(exact_fwi_labels_global),
                    )
            except MemoryError:
                print("[ROC Exact] MemoryError: could not compute exact AUC (dataset too large).")
            except Exception as e:
                print(f"[ROC Exact] Failed to compute exact AUC: {e}")

        out_png = os.path.join(args.out_dir, "roc_global_model_vs_fwi.png")
        plot_roc_curves(
            fpr_m, tpr_m, auc_m,
            fpr_f, tpr_f, auc_f,
            "ROC — Model vs FWI (global peatlands, all days)",
            out_png,
            auc_model_exact=auc_m_exact,
            auc_fwi_exact=auc_f_exact,
        )
        print(f"[Saved] {out_png}")
        print(f"[ROC Global] Model AUC_hist={auc_m:.4f} | FWI AUC_hist={(auc_f if auc_f is not None else float('nan')):.4f}")
        if args.roc_exact:
            print(f"[ROC Global Exact] Model AUC_exact={(auc_m_exact if auc_m_exact is not None else float('nan')):.6f} | "
                  f"FWI AUC_exact={(auc_f_exact if auc_f_exact is not None else float('nan')):.6f}")


        if args.roc_include_regions:
            for rg in regions:
                rname = rg["name"]
                fpr_mr, tpr_mr, auc_mr = roc_from_hist(roc_model_regions[rname])

                fpr_fr: Optional[np.ndarray] = None
                tpr_fr: Optional[np.ndarray] = None
                auc_fr: Optional[float] = None
                if rname in roc_fwi_regions:
                    fpr_fr, tpr_fr, auc_fr = roc_from_hist(roc_fwi_regions[rname])

                # Option 1: exact AUC (unbinned), still plotting the binned ROC.
                auc_mr_exact: Optional[float] = None
                auc_fr_exact: Optional[float] = None
                if args.roc_exact:
                    try:
                        ms = exact_model_scores_regions.get(rname)
                        ml = exact_model_labels_regions.get(rname)
                        if ms is not None and len(ms) > 0:
                            auc_mr_exact = auc_exact_from_scores(np.concatenate(ms), np.concatenate(ml))

                        if rname in roc_fwi_regions:
                            fs = exact_fwi_scores_regions.get(rname)
                            fl = exact_fwi_labels_regions.get(rname)
                            if fs is not None and len(fs) > 0:
                                auc_fr_exact = auc_exact_from_scores(np.concatenate(fs), np.concatenate(fl))
                    except MemoryError:
                        print(f"[ROC Exact] MemoryError: could not compute exact AUC for region {rname}.")
                    except Exception as e:
                        print(f"[ROC Exact] Failed exact AUC for region {rname}: {e}")

                safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", rname.strip())[:60]
                out_png = os.path.join(args.out_dir, f"roc_region_{safe_name}.png")
                plot_roc_curves(
                    fpr_mr, tpr_mr, auc_mr,
                    fpr_fr, tpr_fr, auc_fr,
                    f"ROC — Model vs FWI ({rname})",
                    out_png,
                    auc_model_exact=auc_mr_exact,
                    auc_fwi_exact=auc_fr_exact,
                )
                print(f"[Saved] {out_png}")
                print(f"[ROC {rname}] Model AUC_hist={auc_mr:.4f} | FWI AUC_hist={(auc_fr if auc_fr is not None else float('nan')):.4f}")
                if args.roc_exact:
                    print(f"[ROC {rname} Exact] Model AUC_exact={(auc_mr_exact if auc_mr_exact is not None else float('nan')):.6f} | "
                          f"FWI AUC_exact={(auc_fr_exact if auc_fr_exact is not None else float('nan')):.6f}")
    


    # -------------------------------------------------------------------------
    # Reliability bias heatmap (calibration; Model - Observation)
    # -------------------------------------------------------------------------
    if args.make_reliability_bias and (rel_global_acc is not None):
        rel_out_png = os.path.join(args.out_dir, "reliability_bias_heatmap.png")
        rel_out_csv = os.path.join(args.out_dir, "reliability_bias_by_region.csv")

        edges_pct = rel_edges_pct
        # Match the example figure style: label each column by the *upper* edge of the bin.
        bin_tick_labels = [f"{edges_pct[i]:g}" for i in range(1, len(edges_pct))]

        stats_by_region: Dict[str, Dict] = {}
        region_desc: Dict[str, str] = {}
        regions_order: List[str] = []

        # Global row (all peat-valid pixels with labels)
        stats_by_region["Global"] = rel_global_acc.finalize()
        region_desc["Global"] = "Global: all peat-valid labeled pixels"
        regions_order.append("Global")

        # Region rows
        for rg in regions:
            name = rg["name"]
            stats_by_region[name] = rel_region_accs[name].finalize()
            region_desc[name] = describe_region(rg)
            regions_order.append(name)

        bias_matrix = np.vstack([stats_by_region[r]["bias_pct"] for r in regions_order])
        region_tick_labels = regions_order
        region_desc_lines = [region_desc[r] for r in regions_order]

        plot_reliability_bias_heatmap(
            out_png=rel_out_png,
            bias_matrix_pct=bias_matrix,
            bin_tick_labels=bin_tick_labels,
            region_tick_labels=region_tick_labels,
            region_desc_lines=region_desc_lines,
            vmax=args.reliability_bias_vmax,
        )
        write_reliability_bias_csv(
            out_csv=rel_out_csv,
            regions_order=regions_order,
            region_desc=region_desc,
            edges_pct=edges_pct,
            stats_by_region=stats_by_region,
        )
        print(f"[ReliabilityBias] Saved: {rel_out_png}")
        print(f"[ReliabilityBias] Saved: {rel_out_csv}")

    # -------------------------------------------------------------------------
    # Calibration curve (reliability diagram) from the same binned stats
    # -------------------------------------------------------------------------
    if args.make_calibration_plot and (rel_global_acc is not None):
        # Reuse the same bins used for reliability bias.
        edges_pct = rel_edges_pct

        # Global
        st_global = rel_global_acc.finalize()
        out_png = os.path.join(args.out_dir, 'calibration_curve_global.png')
        plot_calibration_curve(
            out_png=out_png,
            stats=st_global,
            edges_pct=edges_pct,
            title='Calibration curve — Global peatlands',
            xmax_pct=(args.calibration_xmax_pct if (args.calibration_xmax_pct is not None) else (float(args.reliability_bin_max) * 100.0 if (not args.reliability_use_edges_pct) else None)),
            show_hist=(not args.calibration_no_hist),
            min_count=args.calibration_min_count,
        )
        print(f'[CalibrationCurve] Saved: {out_png}')

        # Per-region
        if args.calibration_include_regions:
            for rg in regions:
                name = rg['name']
                st = rel_region_accs[name].finalize()
                safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())[:60]
                out_png = os.path.join(args.out_dir, f'calibration_curve_{safe}.png')
                plot_calibration_curve(
                    out_png=out_png,
                    stats=st,
                    edges_pct=edges_pct,
                    title=f'Calibration curve — {name}',
                    xmax_pct=(args.calibration_xmax_pct if (args.calibration_xmax_pct is not None) else (float(args.reliability_bin_max) * 100.0 if (not args.reliability_use_edges_pct) else None)),
                    show_hist=(not args.calibration_no_hist),
                    min_count=args.calibration_min_count,
                )
                print(f'[CalibrationCurve] Saved: {out_png}')


    # -------------------------------------------------------------------------
    # Spider/radar charts (yearly-only)
    # -------------------------------------------------------------------------
    if args.make_spider_charts and (spider_global_acc is not None):
        global_metrics = spider_global_acc.finalize()
        if not validate_yearly_metrics(global_metrics, "Global"):
            print("[SpiderCharts] Warning: Global yearly metrics out of expected range; skipping spider charts.")
        else:
            # Finalize & validate per-region metrics (skip invalid / missing)
            region_metrics: Dict[str, Dict[str, float]] = {}
            if spider_region_accs is not None:
                for rg_name, rg_acc in spider_region_accs.items():
                    m = rg_acc.finalize()
                    if validate_yearly_metrics(m, rg_name):
                        region_metrics[rg_name] = m
                    else:
                        print(f"[SpiderCharts] Skipping region '{rg_name}' (metrics invalid or missing).")

            # Absolute (fixed) score transform (higher = better)
            global_scores_abs = spider_scores_from_metrics(global_metrics)
            region_scores_abs = {name: spider_scores_from_metrics(m) for name, m in region_metrics.items()}

            # Relative normalization (log-minmax per metric across series), to make small-scale
            # metrics (ECE/Brier/LogLoss) visually comparable on radar charts.
            all_series_metrics = {"Global": global_metrics, **region_metrics}
            all_series_scores_rel = spider_scores_relative_logminmax(all_series_metrics)
            global_scores_rel = all_series_scores_rel.get("Global", {})
            region_scores_rel = {name: all_series_scores_rel.get(name, {}) for name in region_metrics.keys()}

            def _scores_ok(scores: Dict[str, float]) -> bool:
                vals = [scores.get(lbl, float("nan")) for lbl in SPIDER_METRIC_ORDER]
                return bool(np.all(np.isfinite(vals)))

            # Write a JSON summary (raw + both score spaces)
            spider_json = os.path.join(args.out_dir, "spider_metrics_yearly.json")
            with open(spider_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "global_raw": global_metrics,
                        "regions_raw": region_metrics,
                        "global_scores_absolute": global_scores_abs,
                        "regions_scores_absolute": region_scores_abs,
                        "global_scores_relative_logminmax": global_scores_rel,
                        "regions_scores_relative_logminmax": region_scores_rel,
                        "note": (
                            "relative_logminmax scores are per-metric min-max normalized across series, "
                            "with log10 scaling for ECE/Brier/LogLoss before normalization."
                        ),
                    },
                    f,
                    indent=2,
                )
            print(f"[SpiderCharts] Wrote metrics JSON: {spider_json}")

            # 1) Global charts
            # NOTE: "global_scores_abs" uses FIXED absolute scaling shared across all regions.
            if _scores_ok(global_scores_abs):
                outp = os.path.join(args.out_dir, "spider_global.png")
                plot_spider_chart_single(
                    global_scores_abs,
                    title="Yearly Metrics (Global) — Fixed absolute scaling",
                    out_path=outp,
                )
                print(f"[SpiderCharts] Wrote global (fixed): {outp}")

                # Backwards-compatible alias
                outp_alias = os.path.join(args.out_dir, "spider_global_absolute.png")
                if outp_alias != outp:
                    plot_spider_chart_single(
                        global_scores_abs,
                        title="Yearly Metrics (Global) — Fixed absolute scaling",
                        out_path=outp_alias,
                    )
                    print(f"[SpiderCharts] Wrote global (fixed alias): {outp_alias}")

            if _scores_ok(global_scores_rel):
                outp = os.path.join(args.out_dir, "spider_global_relative.png")
                plot_spider_chart_single(
                    global_scores_rel,
                    title="Yearly Metrics (Global) — Relative (log-minmax across series)",
                    out_path=outp,
                )
                print(f"[SpiderCharts] Wrote global relative: {outp}")

            def _safe_series_name(s: str) -> str:
                s = str(s)
                s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
                s = s.strip("_")
                return s if s else "region"

            # 2) Per-region charts
            for rg_name, sc_abs in region_scores_abs.items():
                if not _scores_ok(sc_abs):
                    continue
                safe = _safe_series_name(rg_name)
                outp = os.path.join(args.out_dir, f"spider_{safe}.png")
                plot_spider_chart_single(
                    sc_abs,
                    title=f"Yearly Metrics ({rg_name}) — Fixed absolute scaling",
                    out_path=outp,
                )
                print(f"[SpiderCharts] Wrote region (fixed): {outp}")

            for rg_name, sc_rel in region_scores_rel.items():
                if not _scores_ok(sc_rel):
                    continue
                safe = _safe_series_name(rg_name)
                outp = os.path.join(args.out_dir, f"spider_{safe}_relative.png")
                plot_spider_chart_single(
                    sc_rel,
                    title=f"Yearly Metrics ({rg_name}) — Relative (log-minmax across series)",
                    out_path=outp,
                )
                print(f"[SpiderCharts] Wrote region relative: {outp}")

            # 3) Multi-series overlays
            multi_fixed = {"Global": global_scores_abs, **region_scores_abs}
            multi_fixed = {k: v for k, v in multi_fixed.items() if _scores_ok(v)}
            if len(multi_fixed) >= 2:
                outp = os.path.join(args.out_dir, "spider_regions.png")
                plot_spider_chart_multi(
                    multi_fixed,
                    title="Yearly Metrics — Fixed absolute scaling",
                    out_path=outp,
                )
                print(f"[SpiderCharts] Wrote multi-series (fixed): {outp}")

                # Backwards-compatible alias
                outp_alias = os.path.join(args.out_dir, "spider_regions_absolute.png")
                if outp_alias != outp:
                    plot_spider_chart_multi(
                        multi_fixed,
                        title="Yearly Metrics — Fixed absolute scaling",
                        out_path=outp_alias,
                    )
                    print(f"[SpiderCharts] Wrote multi-series (fixed alias): {outp_alias}")

            multi_rel = {"Global": global_scores_rel, **region_scores_rel}
            multi_rel = {k: v for k, v in multi_rel.items() if _scores_ok(v)}
            if len(multi_rel) >= 2:
                outp = os.path.join(args.out_dir, "spider_regions_relative.png")
                plot_spider_chart_multi(
                    multi_rel,
                    title="Yearly Metrics — Relative (log-minmax across series)",
                    out_path=outp,
                )
                print(f"[SpiderCharts] Wrote multi-series relative: {outp}")
    # Completion message (kept inside run_viz where variables exist)
    print(f"[Done] Used {used_days} daily GeoTIFF(s) from {args.pred_dir}.")


# =============================================================================
# CLI: one script, two subcommands
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DeepPeat all-in-one: (infer) daily nowcast inference, (viz) yearly/daily cartopy analysis from GeoTIFFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # -------------------------
    # infer subcommand
    # -------------------------
    pi = sub.add_parser("infer", help="Run DeepPeat inference (XGBoost ensemble) and make per-day maps",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    pi.add_argument("--input", "--inputs", action="append", dest="inputs", required=True,
                    help="Input Zarr store(s): 'path.zarr' or 'path.zarr:arrayname'. Use multiple --input.")
    pi.add_argument("--viirs-zarr", type=str, required=True,
                    help="VIIRS Zarr store spec: 'path.zarr' or 'path.zarr:arrayname'")
    pi.add_argument("--fwi-zarr", type=str, default=None,
                    help="Optional FWI Zarr store spec (T,1,H,W). Time aligned by 'time' coordinate if present.")

    pi.add_argument("--model-dir", type=str, required=True, help="Trained model directory containing xgb_h{h}.json")
    pi.add_argument("--regime-map", type=str, default=None, help="Regime map GeoTIFF (optional)")
    pi.add_argument("--cluster-map", type=str, default=None, help="Cluster map GeoTIFF (optional)")

    # Calibration
    pi.add_argument("--calibrators-json", type=str, default=None, help="Path to calibrators.json (default: model_dir/calibrators.json if present)")
    pi.add_argument("--no-calibration", action="store_true", help="Disable calibration even if calibrators.json exists.")
    pi.add_argument("--calib-debug", action="store_true", help="Print before/after calibration stats per timestep.")

    pi.add_argument("--peat-mask-source", type=str, default=None, help="Zarr store containing peat_mask array (default: first input)")
    pi.add_argument("--coords-source", type=str, default=None, help="Zarr store containing lat/lon coordinates (default: first input)")

    pi.add_argument("--horizon", type=int, default=0, help="Prediction horizon (0=same-day nowcast)")
    pi.add_argument("--t-hist", dest="t_hist", type=int, default=DEFAULT_T_HIST, help="History window length in days")

    g = pi.add_mutually_exclusive_group()
    g.add_argument("--coord-as-features", dest="coord_as_features", action="store_true", help="Enable coordinate features (x,y,z)")
    g.add_argument("--no-coord-features", dest="coord_as_features", action="store_false", help="Disable coordinate features")
    pi.set_defaults(coord_as_features=True)

    pi.add_argument("--test-split", type=float, default=0.9, help="Train fraction (if no --test-year)")
    pi.add_argument("--test-year", type=int, default=None, help="Filter predictions to a specific year (e.g., 2023)")
    pi.add_argument("--single-date", type=int, default=None, help="Run single date prediction (time index)")

    pi.add_argument("--ensemble-method", type=str, default="weighted", choices=["weighted", "cascade"])
    pi.add_argument("--global-weight", type=float, default=0.4)
    pi.add_argument("--cluster-weight", type=float, default=0.3)
    pi.add_argument("--residual-weight", type=float, default=0.3)

    pi.add_argument("--output-dir", type=str, required=True, help="Output directory for predictions")
    pi.add_argument("--save-geotiff", action="store_true", help="Save GeoTIFF alongside PNG for each day")

    pi.add_argument("--plot-dpi", type=int, default=850, help="Output PNG DPI")
    pi.add_argument("--add-map-features", action="store_true", default=True, help="Add coastlines/borders/gridlines")
    pi.add_argument("--threshold", type=float, default=0.5, help="Binary threshold for metrics")

    # -------------------------
    # viz subcommand
    # -------------------------
    pv = sub.add_parser("viz", help="Make yearly/daily Cartopy plots + capture/confusion/ROC from pred_tXXXXX.tif outputs",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    pv.add_argument("--pred-dir", required=True, help="Directory containing pred_tXXXXX.tif files.")
    pv.add_argument("--out-dir", required=True, help="Output directory for yearly figures.")
    pv.add_argument("--figures", nargs="+", default=["raw", "percentile", "topk"], choices=["raw", "percentile", "topk"])
    pv.add_argument("--top-fracs", type=float, nargs="+", default=[0.05], help="Top-K area fractions (e.g., 0.01 0.05 0.10)")
    pv.add_argument("--area-weighting", choices=["none", "coslat"], default="coslat")

    pv.add_argument("--peat-mask-zarr", default=None, help="Optional mask Zarr spec like smap_wtd.zarr:smap_wtd")
    pv.add_argument("--peat-nodata", type=float, default=-9999.0)
    pv.add_argument("--peat-mask-mode", choices=["valid", "threshold"], default="valid")
    pv.add_argument("--peat-min-fraction", type=float, default=0.01)

    pv.add_argument("--viirs-zarr", default=None, help="Optional VIIRS Zarr spec like viirs.zarr:field")
    pv.add_argument("--viirs-threshold", type=float, default=0.5)
    pv.add_argument("--fwi-zarr", default=None, help="Optional FWI Zarr spec like fwi.zarr:field")

    pv.add_argument("--zarr-index-offset", default="auto", help="Map tif_t -> zarr_t via zarr_t = tif_t - offset. Use 'auto' or integer.")
    pv.add_argument("--dpi", type=int, default=850)
    pv.add_argument("--figwidth", type=float, default=14.0)
    pv.add_argument("--row-height", type=float, default=3.5)

    pv.add_argument("--coastline-lw", type=float, default=0.4)
    pv.add_argument("--border-lw", type=float, default=0.2)
    pv.add_argument("--gridline-alpha", type=float, default=0.15)

    pv.add_argument("--wrap-longitude", choices=["auto", "never", "always"], default="auto")

    pv.add_argument("--raw-prob-vmax", type=float, default=0.20)
    pv.add_argument("--raw-fwi-vmax", type=float, default=40.0)
    pv.add_argument("--freq-vmax", type=float, default=0.0)
    pv.add_argument("--firefreq-scale", choices=["linear", "power", "log"], default="power",
                    help="Scaling for VIIRS fire-day frequency (% of days). 'power' (gamma<1) expands low values for better contrast.")
    pv.add_argument("--firefreq-gamma", type=float, default=0.5,
                    help="Gamma for power scaling (only if --firefreq-scale=power). <1 expands low values; >1 compresses them.")
    pv.add_argument("--firefreq-log-vmin", type=float, default=0.01,
                    help="Minimum positive % shown when using log scaling (only if --firefreq-scale=log).")
    pv.add_argument("--fireday-vmax", type=float, default=0.0)

    pv.add_argument("--max-days", type=int, default=0)

    pv.add_argument("--plot-regions", action="store_true")
    pv.add_argument("--regions", nargs="*", default=[], help="Region names to plot (must match built-in names). Empty => all.")
    pv.add_argument("--region-pad-deg", type=float, default=1.0)

    pv.add_argument("--make-capture-curves", action="store_true")
    pv.add_argument("--make-confusion-maps", action="store_true")
    pv.add_argument("--confusion-top-frac", type=float, default=0.01)
    pv.add_argument("--capture-fracs", type=float, nargs="+", default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10])

    pv.add_argument("--daily-out-dir", default=None)
    pv.add_argument("--daily-which", nargs="+", default=["raw", "percentile", "topk", "confusion", "capture"],
                    choices=["raw", "percentile", "topk", "confusion", "capture"])
    pv.add_argument("--daily-dpi", type=int, default=850)
    pv.add_argument("--daily-figwidth", type=float, default=12.0)
    pv.add_argument("--daily-row-height", type=float, default=3.2)
    pv.add_argument("--daily-include-regions", action="store_true")

    pv.add_argument("--make-roc", action="store_true")
    pv.add_argument("--roc-exact", action="store_true",
                    help="Option 1: also compute exact (unbinned) AUC from raw scores/labels. "
                         "The ROC curves plotted remain binned; exact AUC is shown in the legend and printed. "
                         "WARNING: may be memory/CPU heavy because it stores and sorts all samples.")
    pv.add_argument("--roc-nbins", type=int, default=1000)
    pv.add_argument("--roc-fwi-max", type=float, default=100.0)
    pv.add_argument("--roc-prob-minmax", type=float, nargs=2, default=[0.0, 1.0])
    pv.add_argument("--roc-include-regions", action="store_true")
    # Spider (radar) charts (yearly only)
    pv.add_argument("--make-spider-charts", action="store_true",
                    help="Make yearly spider/radar charts for global + per-region metrics: ECE, Brier, LogLoss, ROC-AUC, Correlation. Requires --viirs-zarr.")
    pv.add_argument("--spider-ece-bins", type=int, default=10, help="Number of uniform probability bins for ECE.")
    pv.add_argument("--spider-roc-bins", type=int, default=600, help="Number of histogram bins for ROC-AUC estimation.")
    # Reliability bias heatmap (calibration; Model - Observation)

    pv.add_argument("--make-calibration-plot", action="store_true",
                    help="Make calibration curve (reliability diagram). By default uses the SAME fixed-width binning as training (see --reliability-bin-width/min/max). Use --reliability-use-edges-pct to instead use --reliability-bins-pct. Requires --viirs-zarr.")
    pv.add_argument("--calibration-xmax-pct", type=float, default=None,
                    help="Optional x/y max (in percent) for calibration curve axes (e.g., 6). Default: last bin edge.")
    pv.add_argument("--calibration-no-hist", action="store_true",
                    help="Disable the sample-fraction histogram overlay on the calibration curve.")
    pv.add_argument("--calibration-min-count", type=int, default=50,
                    help="Minimum samples required to plot a bin point on the calibration curve.")
    pv.add_argument("--calibration-include-regions", action="store_true",
                    help="Also write per-region calibration curves (one PNG per region).")

    pv.add_argument("--make-reliability-bias", action="store_true",
                    help="Make reliability bias (Model-Observation) heatmap by probability bins per region. Requires --viirs-zarr.")

    # Training-matched fixed-width binning in FRACTION units
    pv.add_argument("--reliability-bin-width", type=float, default=0.005,
                    help="Training-matched calibration bin width in FRACTION units (default 0.005 = 0.5%).")
    pv.add_argument("--reliability-bin-min", type=float, default=0.0,
                    help="Minimum probability for reliability/calibration binning (FRACTION; default 0.0).")
    pv.add_argument("--reliability-bin-max", type=float, default=0.06,
                    help="Maximum probability for reliability/calibration binning (FRACTION; default 0.06 = 6%). Values above are clipped into the last bin.")

    # Optional: custom edges in PERCENT units (overrides fixed-width bins)
    pv.add_argument("--reliability-use-edges-pct", action="store_true",
                    help="Use --reliability-bins-pct (percent edges) instead of fixed-width bins.")
    pv.add_argument("--reliability-bins-pct", type=float, nargs="+", default=RELIABILITY_BINS_PCT_DEFAULT,
                    help="Bin edges in percent for reliability/calb curves (e.g., 0 0.5 1 ... 6). Values above the last edge are clipped into the last bin.")
    pv.add_argument("--reliability-bias-vmax", type=float, default=0.8,
                    help="Colorbar limit (±vmax) in percentage points for reliability bias heatmap.")

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.cmd == "infer":
        run_infer(args)
    elif args.cmd == "viz":
        run_viz(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
