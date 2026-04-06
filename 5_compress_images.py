"""
Script 5: GPU-Accelerated Image Compression

Compresses images to AVIF (recommended) or high-quality JPEG while
preserving EXIF/IPTC/XMP metadata. Uses Pillow with SIMD optimizations
and pillow-avif-plugin for AV1 encoding.

Supports:
  - AVIF: 50-70% smaller than JPEG, modern format, wide compatibility
  - JPEG: Universal compatibility fallback at quality 85
  - PNG:  Lossless compression

Usage:
    python 5_compress_images.py
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path

from PIL import Image

# Try to import AVIF support
AVIF_AVAILABLE = False
try:
    import pillow_avif
    AVIF_AVAILABLE = True
except ImportError:
    pass

# ============ CONFIGURATION ============
INPUT_DIR       = r"D:\Photos"             # Source: Archive or raw photos
OUTPUT_DIR      = r"D:\Photos\Compressed"   # Output directory
FORMAT          = "AVIF"                     # AVIF, JPEG, or PNG
QUALITY         = 80                         # 1-100 (higher = better quality, larger file)
MAX_DIMENSION   = 0                          # Max width/height in px (0 = no resize)
SKIP_EXISTING   = True                       # Skip already-compressed files
BATCH_SIZE      = 25                         # Progress report interval

# ---- Load overrides from pipeline_config.json ----
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path, 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    INPUT_DIR     = _cfg.get("compress_input",   INPUT_DIR)
    OUTPUT_DIR    = _cfg.get("compress_output",  OUTPUT_DIR)
    FORMAT        = _cfg.get("compress_format",  FORMAT).upper()
    QUALITY       = int(_cfg.get("compress_quality", QUALITY))
    MAX_DIMENSION = int(_cfg.get("compress_max_dim", MAX_DIMENSION))
# =======================================

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif'}

FORMAT_MAP = {
    "AVIF": {"ext": ".avif", "pil_format": "AVIF",  "params": lambda q: {"quality": q, "speed": 6}},
    "JPEG": {"ext": ".jpg",  "pil_format": "JPEG",  "params": lambda q: {"quality": q, "optimize": True, "subsampling": "4:2:0"}},
    "PNG":  {"ext": ".png",  "pil_format": "PNG",   "params": lambda q: {"optimize": True}},
}


def get_exif_data(img):
    """Extract EXIF data from image for preservation."""
    exif_data = None
    try:
        exif_data = img.info.get('exif', None)
    except Exception:
        pass
    return exif_data


def compress_image(src_path, dst_path, fmt_config, quality, max_dim):
    """
    Compress a single image.
    Returns (success, original_size, compressed_size) tuple.
    """
    original_size = os.path.getsize(src_path)

    try:
        img = Image.open(src_path)

        # Preserve EXIF
        exif_data = get_exif_data(img)

        # Convert to RGB if needed (AVIF/JPEG don't support RGBA well)
        if img.mode in ('RGBA', 'P', 'LA'):
            if fmt_config["pil_format"] in ("JPEG", "AVIF"):
                # Create white background for transparency
                bg = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                bg.paste(img, mask=img.split()[-1] if 'A' in img.mode else None)
                img = bg
            else:
                img = img.convert('RGBA')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Optional resize
        if max_dim > 0:
            w, h = img.size
            if max(w, h) > max_dim:
                ratio = max_dim / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.LANCZOS)

        # Prepare save parameters
        save_params = fmt_config["params"](quality)
        if exif_data and fmt_config["pil_format"] in ("JPEG", "AVIF"):
            save_params["exif"] = exif_data

        # Save compressed
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img.save(dst_path, format=fmt_config["pil_format"], **save_params)

        compressed_size = os.path.getsize(dst_path)

        # Safety: if compressed is larger, keep original
        if compressed_size >= original_size and fmt_config["pil_format"] != "PNG":
            os.remove(dst_path)
            shutil.copy2(src_path, dst_path)
            compressed_size = original_size

        return True, original_size, compressed_size

    except Exception as e:
        print(f"  [ERROR] {src_path}: {e}")
        return False, original_size, 0


def find_images(input_dir):
    """Recursively find all image files."""
    images = []
    for dirpath, _, filenames in os.walk(input_dir):
        for f in filenames:
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(dirpath, f))
    return sorted(images)


def main():
    print(f"\n{'=' * 60}")
    print(f"  Script 5: Image Compression")
    print(f"{'=' * 60}")

    # Validate format
    fmt_key = FORMAT.upper()
    if fmt_key not in FORMAT_MAP:
        print(f"  [ERROR] Unknown format: {FORMAT}. Use AVIF, JPEG, or PNG.")
        return

    if fmt_key == "AVIF" and not AVIF_AVAILABLE:
        print(f"  [WARN] AVIF not available (pip install pillow-avif-plugin)")
        print(f"  Falling back to JPEG at quality {QUALITY}")
        fmt_key = "JPEG"

    fmt_config = FORMAT_MAP[fmt_key]

    print(f"\n  Input:       {INPUT_DIR}")
    print(f"  Output:      {OUTPUT_DIR}")
    print(f"  Format:      {fmt_key} ({fmt_config['ext']})")
    print(f"  Quality:     {QUALITY}")
    if MAX_DIMENSION > 0:
        print(f"  Max size:    {MAX_DIMENSION}px")
    print(f"  Skip exist:  {SKIP_EXISTING}")

    # Find images
    images = find_images(INPUT_DIR)
    print(f"\n  Total images found: {len(images)}")

    if not images:
        print("  Nothing to compress.")
        return

    # Filter already compressed
    pending = []
    for src in images:
        # Compute exact relative path to strictly maintain subfolders
        try:
            rel_path = os.path.relpath(src, INPUT_DIR)
        except ValueError:
            rel_path = os.path.basename(src) # edge case fallback

        orig_path = Path(rel_path)
        
        # 100% keep original name logic
        if fmt_key == "JPEG" and orig_path.suffix.lower() in ['.jpg', '.jpeg', '.jpe']:
            # Literally keep exact same name string (e.g., IMG_123.JPG stays IMG_123.JPG)
            dst_name = orig_path
        elif fmt_key == "PNG" and orig_path.suffix.lower() == '.png':
            dst_name = orig_path
        else:
            # Change extension safely for AVIF
            dst_name = orig_path.with_suffix(fmt_config["ext"])

        dst_path = os.path.join(OUTPUT_DIR, str(dst_name))

        if SKIP_EXISTING and os.path.exists(dst_path):
            continue
        pending.append((src, dst_path))

    print(f"  Already compressed: {len(images) - len(pending)}")
    print(f"  Pending: {len(pending)}")

    if not pending:
        print("  Nothing to do. All images already compressed.")
        return

    start_time = time.time()
    compressed = 0
    errors = 0
    total_original = 0
    total_compressed = 0

    for idx, (src_path, dst_path) in enumerate(pending, 1):
        success, orig_sz, comp_sz = compress_image(src_path, dst_path, fmt_config, QUALITY, MAX_DIMENSION)

        if success:
            compressed += 1
            total_original += orig_sz
            total_compressed += comp_sz
        else:
            errors += 1

        # Progress report
        if idx % BATCH_SIZE == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            eta = (len(pending) - idx) / rate if rate > 0 else 0

            if total_original > 0:
                ratio = (1 - total_compressed / total_original) * 100
            else:
                ratio = 0

            print(f"  [{idx:>6}/{len(pending)}] "
                  f"compressed: {compressed} | "
                  f"{rate:.1f} img/s | "
                  f"saved {ratio:.0f}% | "
                  f"ETA: {eta/60:.0f} min")

    elapsed = time.time() - start_time

    # ---- Print summary ----
    if total_original > 0:
        savings_pct = (1 - total_compressed / total_original) * 100
        savings_mb = (total_original - total_compressed) / (1024 * 1024)
    else:
        savings_pct = 0
        savings_mb = 0

    print(f"\n{'=' * 60}")
    print(f"  Compression complete!")
    print(f"  Images compressed : {compressed}")
    print(f"  Errors            : {errors}")
    print(f"  Time              : {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"  Speed             : {compressed/elapsed:.1f} images/sec")
    print(f"\n  Original size     : {total_original / (1024*1024):.1f} MB")
    print(f"  Compressed size   : {total_compressed / (1024*1024):.1f} MB")
    print(f"  Space saved       : {savings_mb:.1f} MB ({savings_pct:.1f}%)")
    print(f"\n  Output location   : {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
