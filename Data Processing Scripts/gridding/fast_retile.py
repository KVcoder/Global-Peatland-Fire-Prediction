#!/usr/bin/env python3
import argparse, subprocess, sys, os, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def check_ok(proc, cmd_str):
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd_str}\n{proc.stderr}")

def gdalinfo_json(path: Path) -> dict | None:
    p = run(["gdalinfo", "-json", str(path)])
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout)
    except Exception:
        return None

def already_matches(dst: Path, block: int, compress: str, predictor: int, zstd_level: int, cog: bool) -> bool:
    """Return True if an existing dst file already matches our desired layout options."""
    if not dst.exists():
        return False
    info = gdalinfo_json(dst)
    if not info:
        return False
    md = info.get("metadata", {})
    dmd = md.get("IMAGE_STRUCTURE", {})
    comp = dmd.get("COMPRESSION")
    tiled = dmd.get("TILED") == "YES"
    bs = dmd.get("BLOCK_SIZE")
    # BLOCK_SIZE like "256x256"
    if bs and "x" in bs:
        try:
            bx, by = map(int, bs.split("x"))
        except Exception:
            bx = by = -1
    else:
        bx = by = -1

    # For GTiff/COG creation options, predictor and zstd level are in metadata only sometimes.
    # We just match on compression and block size, which are the heavy hitters.
    # If you need strict matching, remove the comments below and be stricter.
    # Strict predictor/level check (best-effort):
    pred_ok = True
    zstd_ok = True
    for k, v in md.get("", {}).items():
        if k.upper().endswith("PREDICTOR"):
            pred_ok = (str(v) == str(predictor))
        if k.upper().endswith("ZSTD_LEVEL"):
            zstd_ok = (str(v) == str(zstd_level))

    return (tiled and comp == compress.upper() and bx == block and by == block and pred_ok and zstd_ok)

def build_cmd(src: Path, tmp: Path, block: int, compress: str, predictor: int,
              threads: int, cog: bool, zstd_level: int, bigtiff: str) -> list[str]:
    compress = compress.upper()
    co = [
        "-co", "TILED=YES",
        "-co", f"BLOCKXSIZE={block}",
        "-co", f"BLOCKYSIZE={block}",
        "-co", f"COMPRESS={compress}",
        "-co", f"PREDICTOR={predictor}",
        "-co", f"BIGTIFF={bigtiff}",
        "-co", f"NUM_THREADS={threads}",
    ]
    if compress == "ZSTD":
        co += ["-co", f"ZSTD_LEVEL={zstd_level}"]
    if cog:
        # Use COG driver (may be slightly slower but great for browsing).
        # You can add overview options as needed.
        return ["gdal_translate", "-of", "COG", str(src), str(tmp), *co, "-q"]
    else:
        return ["gdal_translate", str(src), str(tmp), *co, "-q"]

def process_one(p: Path, src_root: Path, dst_root: Path, block: int, compress: str, predictor: int,
                threads: int, cog: bool, zstd_level: int, bigtiff: str, smart_skip: bool, skip_existing: bool) -> tuple[Path, str]:
    rel = p.relative_to(src_root)
    out_path = (dst_root / rel)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and out_path.exists():
        return (out_path, "skipped_exists")

    if smart_skip and already_matches(out_path, block, compress, predictor, zstd_level, cog):
        return (out_path, "skipped_smart")

    tmp = out_path.with_suffix(out_path.suffix + ".tmp.tif")
    cmd = build_cmd(p, tmp, block, compress, predictor, threads, cog, zstd_level, bigtiff)
    proc = run(cmd)
    check_ok(proc, " ".join(cmd))
    tmp.replace(out_path)
    return (out_path, "done")

def main():
    ap = argparse.ArgumentParser(description="Fast, parallel retiling/recompression of GeoTIFF folders using GDAL.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input folders (e.g., ERA5LAND_..., RESAMPLED_SMAP_..., RESAMPLED_NEW_VIIRS_...)")
    ap.add_argument("--out", required=True, help="Output root folder. Input folder names will be mirrored inside.")
    ap.add_argument("--block", type=int, default=512, help="Tile size (both X and Y). 512 is a good modern default.")
    ap.add_argument("--compress", default="ZSTD", choices=["ZSTD", "LZW", "DEFLATE", "JPEG", "NONE"])
    ap.add_argument("--predictor", type=int, default=3, help="Use 3 for floating point, 2 for integers.")
    ap.add_argument("--zstd-level", type=int, default=9, help="ZSTD level (1=fast, 9=default, 19=very slow).")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4)//4),
                    help="Number of parallel files to process.")
    ap.add_argument("--threads-per-job", type=int, default=max(1, (os.cpu_count() or 4)//(max(1, (os.cpu_count() or 4)//4))),
                    help="Threads per GDAL encode (NUM_THREADS). Tune with --jobs.")
    ap.add_argument("--gdal-cachemax", type=int, default=1024, help="GDAL cache in MB (per process).")
    ap.add_argument("--cog", action="store_true", help="Write Cloud Optimized GeoTIFFs (-of COG).")
    ap.add_argument("--smart-skip", action="store_true", help="Skip outputs that already match tiling/compression.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if output file already exists.")
    ap.add_argument("--bigtiff", default="IF_SAFER", choices=["YES", "NO", "IF_NEEDED", "IF_SAFER"], help="BIGTIFF policy.")
    args = ap.parse_args()

    # Check GDAL tools
    p = run(["gdal_translate", "--version"])
    if p.returncode != 0:
        print("ERROR: gdal_translate not found. Please install GDAL.", file=sys.stderr)
        sys.exit(1)
    p = run(["gdalinfo", "--version"])
    if p.returncode != 0 and args.smart_skip:
        print("WARNING: gdalinfo not found. --smart-skip will be ineffective.", file=sys.stderr)

    # Set GDAL cache (per worker)
    os.environ["GDAL_CACHEMAX"] = str(args.gdal_cachemax)

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    # Collect work
    tasks = []
    mirror_pairs = []
    for in_dir in args.inputs:
        src = Path(in_dir).resolve()
        if not src.is_dir():
            print(f"WARNING: {src} is not a directory; skipping", file=sys.stderr)
            continue
        dst = (out_root / src.name).resolve()
        dst.mkdir(parents=True, exist_ok=True)
        mirror_pairs.append((src, dst))
        for pth in src.rglob("*"):
            if pth.is_file() and pth.suffix.lower() in (".tif", ".tiff"):
                tasks.append((pth, src, dst))

    if not tasks:
        print("No GeoTIFF files found. Nothing to do.")
        return

    worker = partial(
        process_one,
        block=args.block,
        compress=args.compress,
        predictor=args.predictor,
        threads=args.threads_per_job,
        cog=args.cog,
        zstd_level=args.zstd_level,
        bigtiff=args.bigtiff,
        smart_skip=args.smart_skip,
        skip_existing=args.skip_existing,
    )

    done = skipped_e = skipped_s = failed = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(worker, p, s, d): (p, s, d) for (p, s, d) in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                out_path, status = fut.result()
                if status == "done":
                    done += 1
                elif status == "skipped_exists":
                    skipped_e += 1
                elif status == "skipped_smart":
                    skipped_s += 1
                if i % 25 == 0:
                    print(f"  Processed {i}/{len(futs)} (done={done}, skip_exist={skipped_e}, skip_smart={skipped_s})")
            except Exception as e:
                failed += 1
                print(f"ERROR: {e}", file=sys.stderr)

    total = len(tasks)
    print(f"\nFinished. Total files: {total} | done={done} | skip_exist={skipped_e} | skip_smart={skipped_s} | failed={failed}")

if __name__ == "__main__":
    main()
