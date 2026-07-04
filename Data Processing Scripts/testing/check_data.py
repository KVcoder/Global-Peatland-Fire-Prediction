
#!/usr/bin/env python3
"""
infer_era_tif_temporal_structure.py

Infer temporal structure (variables per timestep, number of timesteps, temporal frequency)
directly from a stacked ERA5-style GeoTIFF, using only the file contents.

What it tries:
1) Parse per-band descriptions/tags to extract timestamps and variable names (CF-like short names).
2) If timestamps aren't per-band, parse file-level tags or filename for date hints.
3) If variable names aren't in metadata, infer the number of variables per timestep by
   detecting a repeating pattern across bands using sampling statistics and similarity.

Outputs:
- Basic raster info
- Inferred variables per timestep
- Inferred variable name order (guessed)
- Inferred time steps (and any remainder bands)
- Temporal frequency (DAILY / MONTHLY / UNKNOWN) and reasoning
- Sample of discovered timestamps (if any)

Usage:
  python infer_era_tif_temporal_structure.py /path/to/stack.tif
"""

import argparse
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio

# --- Regex helpers for timestamps ---
ISO_DATE_RE = re.compile(
    r'(?P<y>\d{4})[-_/]?(?P<m>\d{2})(?:[-_/]?(?P<d>\d{2}))?(?:[ T](?P<h>\d{2}):(?P<min>\d{2})(?::(?P<s>\d{2}))?)?'
)
YEAR_MONTH_ONLY_RE = re.compile(r'(?P<y>\d{4})[-_/]?(?P<m>\d{2})(?![-_/]?\d{2})')

# Known ERA5 short names / hints for variable extraction
KNOWN_VAR_HINTS = [
    "t2m", "d2m", "u10", "v10", "sp", "tp", "ssr", "msl", "tcc", "t2m_max", "t2m_min",
    "precip", "precipitation", "total_precipitation", "skin_temperature", "sst",
    "wind_u", "wind_v", "u_component", "v_component", "dewpoint", "surface_pressure",
    "total_cloud_cover", "evaporation", "runoff"
]

def parse_date_strings(text: str) -> List[datetime]:
    dates = []
    for m in ISO_DATE_RE.finditer(text):
        y = int(m.group('y'))
        mo = int(m.group('m'))
        d = int(m.group('d') or 1)
        h = int(m.group('h') or 0)
        mi = int(m.group('min') or 0)
        s = int(m.group('s') or 0)
        try:
            dates.append(datetime(y, mo, d, h, mi, s))
        except ValueError:
            continue
    if not dates:
        for m in YEAR_MONTH_ONLY_RE.finditer(text):
            y = int(m.group('y')); mo = int(m.group('m'))
            try:
                dates.append(datetime(y, mo, 1))
            except ValueError:
                continue
    return dates

def extract_candidate_datetimes_and_vars(ds: rasterio.io.DatasetReader, tif_path: Path) -> Tuple[List[datetime], List[Optional[str]]]:
    dates: List[datetime] = []
    var_labels: List[Optional[str]] = [None] * ds.count

    # File-level tags for global dates
    try:
        file_tags = ds.tags()
        for k, v in file_tags.items():
            dates.extend(parse_date_strings(f"{k}={v}"))
    except Exception:
        pass

    # Per-band descriptions may include var and/or date
    try:
        descs = ds.descriptions or [None] * ds.count
    except Exception:
        descs = [None] * ds.count

    # Per-band tags too
    for b in range(1, ds.count + 1):
        band_texts = []
        d = None
        try:
            d = descs[b - 1]
        except Exception:
            d = None
        if d:
            band_texts.append(d)
        try:
            bt = ds.tags(b)
            if bt:
                band_texts.extend([f"{k}={v}" for k, v in bt.items()])
        except Exception:
            pass

        merged = " | ".join([t for t in band_texts if t])
        if merged:
            # dates
            bd = parse_date_strings(merged)
            dates.extend(bd)
            # variables
            var_guess = None
            low = merged.lower()
            for hint in KNOWN_VAR_HINTS:
                if hint in low:
                    var_guess = hint
                    break
            # Look for 'variable=XXX' or 'var=XXX'
            if var_guess is None:
                m = re.search(r'(?:variable|var|name|short_name)\s*[:=]\s*([A-Za-z0-9_]+)', merged, re.IGNORECASE)
                if m:
                    var_guess = m.group(1)
            var_labels[b - 1] = var_guess

    # If we still have no dates, try filename
    if not dates:
        dates = parse_date_strings(tif_path.name)

    # Deduplicate/sort dates
    dates = sorted(set(dates))
    return dates, var_labels

def sample_band_stats(ds, band_index: int, n_samples: int = 2000, rng_seed: int = 42) -> Tuple[float, float]:
    """
    Return (mean, std) from a sparse uniform sample of pixels in a band.
    """
    rng = np.random.default_rng(rng_seed + band_index)
    h, w = ds.height, ds.width
    # choose random windows to limit IO: sample N points
    rows = rng.integers(0, h, size=n_samples)
    cols = rng.integers(0, w, size=n_samples)
    # rasterio window read of scattered points: read a small window then index
    # For simplicity and speed, read a grid window if possible
    # We'll read 8 small tiles and aggregate until reaching ~n_samples
    tile_means = []
    tile_stds = []
    remaining = n_samples
    step = max(1, min(h, w) // 64)
    for r0 in range(0, h, max(1, h // 8)):
        for c0 in range(0, w, max(1, w // 8)):
            if remaining <= 0:
                break
            r1 = min(h, r0 + step)
            c1 = min(w, c0 + step)
            window = rasterio.windows.Window.from_slices((r0, r1), (c0, c1))
            arr = ds.read(band_index, window=window, masked=True)
            if arr.size == 0:
                continue
            tile_means.append(float(np.nanmean(arr)))
            tile_stds.append(float(np.nanstd(arr)))
            remaining -= arr.size
        if remaining <= 0:
            break
    if tile_means:
        return float(np.nanmean(tile_means)), float(np.nanmean(tile_stds))
    # Fallback: read full band (may be slow)
    arr = ds.read(band_index, masked=True)
    return float(np.nanmean(arr)), float(np.nanstd(arr))

def infer_period_via_similarity(ds, max_period: int = 24) -> Tuple[Optional[int], Dict[int, float]]:
    """
    Try to infer repeating period p such that bands separated by p have more similar stats
    than neighbors at other offsets. Returns best p and score per p.
    Score is average negative L1 distance of (mean,std) pairs for same-mod classes.
    Larger score => better.
    """
    count = ds.count
    max_p = min(max_period, max(1, count // 2))
    # precompute quick stats per band
    stats = [sample_band_stats(ds, i+1) for i in range(count)]
    means = np.array([m for m, s in stats], dtype=float)
    stds = np.array([s for m, s in stats], dtype=float)

    scores: Dict[int, float] = {}
    for p in range(1, max_p + 1):
        # group by i % p
        dists = []
        for r in range(p):
            idx = np.arange(r, count, p)
            if len(idx) < 2:
                continue
            # pairwise distance between consecutive items in this residue class
            for i in range(len(idx) - 1):
                a = idx[i]; b = idx[i+1]
                d = abs(means[a] - means[b]) + abs(stds[a] - stds[b])
                dists.append(d)
        if dists:
            # smaller distance means more repeatability; we invert
            score = -float(np.mean(dists))
            scores[p] = score

    if not scores:
        return None, {}
    # pick best p with most negative mean distance (i.e., max score)
    best_p = max(scores, key=lambda k: scores[k])
    # sanity check: require improvement over p=1
    base = scores.get(1, None)
    if base is not None and scores[best_p] <= base * 0.95:
        # Not much improvement; uncertain
        return None, scores
    return best_p, scores

def classify_frequency(dates: List[datetime], time_steps: Optional[int]) -> Tuple[str, str]:
    if dates and len(dates) >= 3:
        deltas = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        deltas = [d for d in deltas if d > 0]
        if deltas:
            common = Counter(deltas).most_common(1)[0][0]
            if common in (1, 2):
                return "DAILY", f"Inferred from timestamp gaps; most common gap = {common} day(s)."
            if 27 <= common <= 31:
                return "MONTHLY", f"Inferred from timestamp gaps; most common gap ≈ {common} days."
            avg = sum(deltas) / len(deltas)
            if 0.75 <= avg <= 1.75:
                return "DAILY", f"Inferred from timestamps; average gap ≈ {avg:.2f} days."
            if 25 <= avg <= 35:
                return "MONTHLY", f"Inferred from timestamps; average gap ≈ {avg:.2f} days."
    if time_steps:
        if time_steps % 12 == 0:
            return "MONTHLY", f"Heuristic: {time_steps} time steps is a multiple of 12."
        if 360 <= time_steps <= 370 or 720 <= time_steps <= 740:
            return "DAILY", f"Heuristic: {time_steps} time steps looks like ~1–2 years of daily data."
        if time_steps >= 1000 and time_steps % 365 in (0, 1, 2):
            return "DAILY", f"Heuristic: {time_steps} time steps ≈ multiple years of daily data."
    return "UNKNOWN", "Could not confidently classify from metadata or heuristics."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif_path", type=Path, help="Path to the GeoTIFF stack")
    ap.add_argument("--max-period", type=int, default=24, help="Maximum repeating period to consider when inferring variables per timestep (default: 24)")
    args = ap.parse_args()

    tif_path: Path = args.tif_path
    if not tif_path.exists():
        print(f"ERROR: File not found: {tif_path}")
        return

    with rasterio.open(tif_path) as ds:
        print("=== Basic Info ===")
        print(f"Path: {tif_path}")
        print(f"Driver: {ds.driver}")
        print(f"Size: {ds.width} x {ds.height}")
        print(f"Bands: {ds.count}")
        print(f"CRS: {ds.crs}")
        print(f"Dtype: {ds.dtypes[0] if ds.count > 0 else 'n/a'}")

        # Extract timestamps and any variable hints
        dates, var_labels = extract_candidate_datetimes_and_vars(ds, tif_path)
        if dates:
            print("\n=== Temporal Metadata (from tags/descriptions/filename) ===")
            print(f"Found {len(dates)} candidate timestamp(s).")
            sample = dates[:5]
            print("First timestamps:", ", ".join(dt.isoformat() for dt in sample), ("..." if len(dates) > 5 else ""))
        else:
            print("\nNo explicit timestamps discovered in metadata or filename.")

        # Try to infer variables per timestep from labels first
        label_period = None
        if any(v is not None for v in var_labels):
            # Reduce to sequence of labels; compress consecutive duplicates
            seq = [v if v is not None else f"band{idx+1}" for idx, v in enumerate(var_labels)]
            # Try to find smallest repeating period using KMP-like approach on labels
            # We consider the period that exactly tiles the sequence if possible; otherwise fall back to similarity
            def smallest_period(arr: Sequence[str]) -> Optional[int]:
                n = len(arr)
                for p in range(1, min(args.max_period, n) + 1):
                    if n % p != 0:
                        continue
                    ok = True
                    for i in range(n):
                        if arr[i] != arr[i % p]:
                            ok = False; break
                    if ok:
                        return p
                return None
            label_period = smallest_period(seq)

        if label_period is not None:
            vars_per_ts = label_period
            inference_source = "labels"
        else:
            # Infer via similarity of sampled stats
            inferred_p, scores = infer_period_via_similarity(ds, max_period=args.max_period)
            vars_per_ts = inferred_p if inferred_p and inferred_p > 0 else None
            inference_source = "similarity" if vars_per_ts else "none"

        # Compute time steps from inferred period
        if vars_per_ts:
            time_steps = ds.count // vars_per_ts
            leftover = ds.count % vars_per_ts
        else:
            time_steps = None
            leftover = None

        print("\n=== Inferred Stack Structure ===")
        print(f"Variables per timestep: {vars_per_ts if vars_per_ts else 'UNKNOWN'} (source: {inference_source})")
        print(f"Estimated time steps: {time_steps if time_steps is not None else 'UNKNOWN'}", end="")
        if leftover is not None and leftover != 0:
            print(f"  [WARNING: remainder bands = {leftover}; stack may be irregular]")
        else:
            print("")

        # Guess variable name order for one timestep
        if vars_per_ts and any(var_labels):
            # Take first block of length p and report labels (fallback to band indices)
            block = []
            for i in range(vars_per_ts):
                label = var_labels[i] if var_labels[i] else f"band{i+1}"
                block.append(label)
            print(f"Guessed variable order for each timestep: {block}")
        elif vars_per_ts:
            print("Guessed variable order for each timestep: [band1 .. band{p}]".replace("{p}", str(vars_per_ts)))
        else:
            print("Could not guess variable order due to unknown period.")

        # Classify frequency
        temporal_res, why = classify_frequency(dates, time_steps)
        print("\n=== Temporal Resolution ===")
        print(f"Classification: {temporal_res}")
        print(f"Reason: {why}")

        # Hints
        if temporal_res == "UNKNOWN":
            if time_steps:
                print("\nHint: If this is an ERA5 MONTHLY stack, time_steps is typically a multiple of 12.")
                print("      For DAILY stacks, ~365 per year (leap years 366).")
            else:
                print("\nHint: Add per-band descriptions (e.g., 't2m 2021-01', 'sp 2021-01') to improve inference.")

if __name__ == "__main__":
    main()
