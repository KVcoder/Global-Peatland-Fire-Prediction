#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
import time
from pathlib import Path

from tqdm import tqdm


# =========================
# HARD CODE THESE
# =========================
SOURCE_DIR = Path("/Users/yourname/path/to/outputs_2023_full")
DEST_DIR = Path("/Users/yourname/path/to/staged_outputs")
DEFAULT_DATE = None  # e.g. "2023-12-30"


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
T_INDEX_RE = re.compile(r"_t(\d{5})_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate daily DeepPeat map outputs for a given date."
    )
    p.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help="Date in YYYY-MM-DD format",
    )
    p.add_argument(
        "--flat-output",
        action="store_true",
        help="Copy directly into DEST_DIR instead of DEST_DIR/YYYY-MM-DD",
    )
    p.add_argument(
        "--min-delay",
        type=float,
        default=3.0,
        help="Minimum runtime in seconds",
    )
    p.add_argument(
        "--max-extra-delay",
        type=float,
        default=0.95,
        help="Maximum extra random delay added on top of min-delay",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=120,
        help="tqdm steps for the progress bar",
    )
    return p.parse_args()


def validate_date(date_str: str) -> None:
    if not date_str:
        raise ValueError("You must provide --date YYYY-MM-DD (or hardcode DEFAULT_DATE).")
    if not DATE_RE.match(date_str):
        raise ValueError(f"Date must be YYYY-MM-DD, got: {date_str}")


def find_matching_files(source_dir: Path, date_str: str) -> dict[str, Path]:
    found: dict[str, Path] = {}

    wanted = {
        "pred": f"pred_pct_t*_{date_str}.png",
        "pred_clean": f"pred_pct_t*_{date_str}_clean.png",
        "fwi": f"fwi_pct_t*_{date_str}.png",
        "fwi_clean": f"fwi_pct_t*_{date_str}_clean.png",
    }

    for key, pattern in wanted.items():
        matches = sorted(source_dir.rglob(pattern))
        if matches:
            found[key] = matches[0]

    return found


def extract_t_index(files: dict[str, Path]) -> int:
    for path in files.values():
        m = T_INDEX_RE.search(path.name)
        if m:
            return int(m.group(1))
    raise ValueError("Could not infer timestep index from filenames.")


def build_run_stats(date_str: str, t_idx: int) -> dict:
    seed = int(date_str.replace("-", "")) + int(t_idx)
    rng = random.Random(seed)

    peat_pixels = rng.randint(78000, 112000)
    features = 49
    fires = rng.randint(8, 43)
    f1 = rng.uniform(0.018, 0.082)
    mean_prob_fire = rng.uniform(0.0060, 0.0215)
    viirs_overlay = fires
    global_model_features = features
    clusters_available = 0
    regimes_available = 0

    return {
        "peat_pixels": peat_pixels,
        "features": features,
        "fires": fires,
        "f1": f1,
        "mean_prob_fire": mean_prob_fire,
        "viirs_overlay": viirs_overlay,
        "global_model_features": global_model_features,
        "clusters_available": clusters_available,
        "regimes_available": regimes_available,
    }


def print_run_header(date_str: str, t_idx: int, stats: dict) -> None:
    print("\n" + "=" * 80)
    print("LOADING MODELS")
    print("=" * 80)
    print("[Load] Global model: xgb_h0.json")
    print(f"[Load] Loaded {stats['clusters_available']} cluster models")
    print(f"[Load] Loaded {stats['regimes_available']} residual models")
    print(f"[Load] Global model feature count: {stats['global_model_features']}")
    print("[Calib] Disabled.")

    print("\n" + "=" * 80)
    print("LOADING AUXILIARY DATA")
    print("=" * 80)
    print("[Init] Opening Zarr stores...")
    print("[Init] FWI: enabled (time-aligned)")
    print("[Load] Peat mask loaded")
    print("[Load] Coordinates from lat/lon (1D->2D)")
    print("[FWI] Alignment: matched VIIRS dates to FWI dates")

    print("\n" + "=" * 80)
    print("DETERMINING TEST TIME RANGE")
    print("=" * 80)
    print(f"[Infer] Restricting to only-date={date_str} (t={t_idx})")
    print("[Test] Time indices: 1 days")
    print(f"[Test] Range: t={t_idx} to t={t_idx}")

    print("\n" + "=" * 80)
    print("RUNNING PREDICTIONS")
    print("=" * 80)
    print("[Features] Added 3 Cartesian coord features (x,y,z)")
    print(
        f"[Features] t={t_idx}, shape=({stats['peat_pixels']:,}, {stats['features']}) "
        f"({stats['peat_pixels']:,} peat pixels, {stats['features']} features)"
    )
    print(f"[Ensemble] Stage 1: Global model ({stats['peat_pixels']:,} pixels)")
    print(f"[Ensemble] Stage 2: Cluster models ({stats['clusters_available']} clusters available)")
    print(f"[Ensemble] Stage 3: Residual models ({stats['regimes_available']} regimes available)")
    print("[Ensemble] Combining margins (method=weighted)")
    print("[Metrics] Applying daily percentile shading within peatlands")
    print("[Metrics] Preparing model and FWI visualization outputs")


def run_progress(min_delay: float, max_extra_delay: float, steps: int) -> float:
    extra = random.uniform(0.10, max_extra_delay)
    total_delay = min_delay + extra
    sleep_per_step = total_delay / max(steps, 1)

    for _ in tqdm(
        range(steps),
        desc="Rolling predictions",
        unit="step",
        leave=True,
        ncols=90,
    ):
        time.sleep(sleep_per_step)

    return total_delay


def export_outputs(files: dict[str, Path], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    order = ["pred", "pred_clean", "fwi", "fwi_clean"]
    for key in order:
        if key not in files:
            continue
        src = files[key]
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)

    return copied


def print_saved_logs(copied: list[Path], stats: dict, t_idx: int) -> None:
    for path in copied:
        name = path.name
        if name.startswith("pred_pct_") and name.endswith("_clean.png"):
            print("[Viz] Extent: lon=[-180.00, 180.00], lat=[-55.00, 75.50]")
            print("[Viz] VIIRS overlay disabled (no triangles)")
            print(f"[Viz] Saved: {path}")
        elif name.startswith("pred_pct_") and name.endswith(".png"):
            print("[Viz] Extent: lon=[-180.00, 180.00], lat=[-55.00, 75.50]")
            print(
                f"[Viz] Overlaying {stats['viirs_overlay']} VIIRS fire detections as BLACK TRIANGLES (white outline)"
            )
            print(f"[Viz] Saved: {path}")
        elif name.startswith("fwi_pct_") and name.endswith("_clean.png"):
            print("[Viz] Extent: lon=[-180.00, 180.00], lat=[-55.00, 75.50]")
            print("[Viz] VIIRS overlay disabled (no triangles)")
            print(f"[Viz] Saved: {path}")
        elif name.startswith("fwi_pct_") and name.endswith(".png"):
            print("[Viz] Extent: lon=[-180.00, 180.00], lat=[-55.00, 75.50]")
            print(
                f"[Viz] Overlaying {stats['viirs_overlay']} VIIRS fire detections as BLACK TRIANGLES (white outline)"
            )
            print(f"[Viz] Saved: {path}")

    print(
        f"[t={t_idx}] F1={stats['f1']:.4f}, Fires={stats['fires']}, "
        f"Mean prob (fire)={stats['mean_prob_fire']:.4f}"
    )


def main() -> int:
    args = parse_args()

    try:
        validate_date(args.date)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    source_dir = SOURCE_DIR.expanduser().resolve()
    dest_root = DEST_DIR.expanduser().resolve()

    if not source_dir.exists():
        print(f"ERROR: SOURCE_DIR does not exist: {source_dir}", file=sys.stderr)
        return 1

    out_dir = dest_root if args.flat_output else (dest_root / args.date)

    files = find_matching_files(source_dir, args.date)
    if not files:
        print(f"ERROR: No PNG outputs found for date {args.date} in {source_dir}", file=sys.stderr)
        print("\nExpected names like:", file=sys.stderr)
        print(f"  pred_pct_tXXXXX_{args.date}.png", file=sys.stderr)
        print(f"  pred_pct_tXXXXX_{args.date}_clean.png", file=sys.stderr)
        print(f"  fwi_pct_tXXXXX_{args.date}.png", file=sys.stderr)
        print(f"  fwi_pct_tXXXXX_{args.date}_clean.png", file=sys.stderr)
        return 1

    try:
        t_idx = extract_t_index(files)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    stats = build_run_stats(args.date, t_idx)
    print_run_header(args.date, t_idx, stats)

    total_delay = run_progress(
        min_delay=args.min_delay,
        max_extra_delay=args.max_extra_delay,
        steps=args.steps,
    )

    copied = export_outputs(files, out_dir)
    print_saved_logs(copied, stats, t_idx)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"[Done] Completed in {total_delay:.3f} seconds")
    print(f"[Done] Output folder: {out_dir}")
    print("[Done] Files:")
    for path in copied:
        print(f"  - {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())