# Random Forest Command Examples

This file contains example commands for running the RF backend with train_XGB_global_new.py.

## Basic Random Forest Training (CPU)

```bash
python train_XGB_global_new.py \
  --backend random-forest \
  --train-mode tabularize \
  --input /path/to/zarr/file1.zarr \
  --input /path/to/zarr/file2.zarr \
  --horizons 1 3 7 14 \
  --rf-n-estimators 400 \
  --rf-max-depth 20 \
  --rf-max-features sqrt \
  --rf-max-samples 0.8 \
  --rf-min-samples-leaf 1 \
  --rf-n-jobs -1 \
  --rf-use-class-weight \
  --calibration-method isotonic \
  --logdir runs/peat_rf
```

## Random Forest with GPU (cuML)

```bash
python train_XGB_global_new.py \
  --backend random-forest \
  --use-gpu \
  --train-mode tabularize \
  --input /path/to/zarr/file1.zarr \
  --input /path/to/zarr/file2.zarr \
  --horizons 1 3 7 14 \
  --rf-n-estimators 400 \
  --rf-max-depth 20 \
  --rf-max-features auto \
  --rf-min-samples-leaf 1 \
  --calibration-method isotonic \
  --logdir runs/peat_rf_gpu
```

## Converting from XGBoost Command

If you have an existing XGBoost command, here's how to convert it:

### Original XGBoost Command:
```bash
python train_XGB_global_new.py \
  --backend xgboost \
  --train-mode tabularize \
  --input /data/peat_fires_indonesia.zarr \
  --input /data/peat_fires_malaysia.zarr \
  --horizons 1 3 7 14 30 \
  --xgb-n-estimators 400 \
  --xgb-learning-rate 0.05 \
  --xgb-max-depth 5 \
  --xgb-subsample 0.8 \
  --xgb-tree-method hist \
  --xgb-device cuda \
  --calibration-method isotonic \
  --logdir runs/peat_xgb
```

### Adapted Random Forest Command:
```bash
python train_XGB_global_new.py \
  --backend random-forest \
  --train-mode tabularize \
  --input /data/peat_fires_indonesia.zarr \
  --input /data/peat_fires_malaysia.zarr \
  --horizons 1 3 7 14 30 \
  --rf-n-estimators 400 \
  --rf-max-depth 20 \
  --rf-max-features sqrt \
  --rf-max-samples 0.8 \
  --rf-min-samples-leaf 1 \
  --rf-n-jobs -1 \
  --rf-use-class-weight \
  --calibration-method isotonic \
  --logdir runs/peat_rf
```

### Key Changes:
1. `--backend xgboost` → `--backend random-forest`
2. Remove `--xgb-learning-rate` (no boosting in RF)
3. `--xgb-max-depth 5` → `--rf-max-depth 20` (RF needs deeper trees)
4. `--xgb-subsample 0.8` → `--rf-max-samples 0.8`
5. Remove `--xgb-tree-method` and `--xgb-device` (use `--rf-n-jobs -1` or `--use-gpu` instead)
6. Add `--rf-use-class-weight` for class imbalance handling

## Important Notes

### Training Mode Restrictions
- **Tabularize mode**: ✅ Supported - Use `--train-mode tabularize`
- **Stream mode**: ❌ NOT supported - Random Forest requires all data in memory

If you try to use `--train-mode stream` with `--backend random-forest`, you'll get a helpful error message telling you to use `--train-mode tabularize` instead.

### GPU Support (cuML)
To use GPU acceleration with Random Forest:
1. Install cuML: `pip install cuml-cu11` (CUDA 11) or `cuml-cu12` (CUDA 12)
2. Add `--use-gpu` flag to your command
3. Note: cuML has different parameter support than sklearn - some parameters like `class_weight` are not available

### Hyperparameter Mapping

| XGBoost Param | Default | RF Param | Default | Notes |
|---------------|---------|----------|---------|-------|
| `--xgb-n-estimators` | 400 | `--rf-n-estimators` | 400 | Number of trees |
| `--xgb-max-depth` | 5 | `--rf-max-depth` | 20 | RF typically needs deeper trees |
| `--xgb-subsample` | 0.8 | `--rf-max-samples` | 0.8 | Row sampling |
| `--xgb-colsample-bynode` | 0.5 | `--rf-max-features` | "sqrt" | Feature sampling |
| `--xgb-min-child-weight` | 1.0 | `--rf-min-samples-leaf` | 1 | Leaf size control |
| `--xgb-device` | cuda | `--use-gpu` or `--rf-n-jobs` | -1 | GPU vs CPU parallelism |
| `--xgb-learning-rate` | 0.05 | ❌ Remove | - | No boosting in RF |
| `--xgb-tree-method` | hist | ❌ Remove | - | Auto-optimized in RF |

### Class Imbalance Handling

Random Forest offers two options:
1. **Auto-balancing** (recommended): `--rf-use-class-weight`
2. **Manual weighting**: `--rf-use-sample-weight`

Note: cuML doesn't support `class_weight`, so when using `--use-gpu`, the code automatically falls back to manual sample weighting.

### Calibration

Calibration works exactly the same for both backends:
- `--calibration-method none` (default, no calibration)
- `--calibration-method platt` (Platt scaling)
- `--calibration-method isotonic` (Isotonic regression)

Use `--calib-frac 0.1` to hold out 10% of training data for calibration fitting.

### Performance Expectations

- **Training speed**: RF is typically 2-5x slower than XGBoost GPU, similar to XGBoost CPU
- **Memory usage**: RF uses 1.5-3x more RAM than XGBoost
- **Prediction quality**: ROC-AUC should be within ±0.02 of XGBoost
- **Calibration quality**: Similar to XGBoost after isotonic/platt calibration
