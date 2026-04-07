"""
Script 3: Semantic Image Classification using CLIP / SigLIP 2 + DirectML

Classifies images that have NO detected faces into semantic categories
using OpenAI CLIP (ViT-B/32) via ONNX Runtime with DirectML acceleration.

CLIP performs "zero-shot" classification — categories are defined as plain
text strings. You can add/remove/modify categories without retraining.

Prerequisites:
    1. Run scripts/setup_clip_model.py once to export the ONNX model
    2. pip install onnxruntime-directml transformers Pillow numpy

Usage:
    python scripts/3_classify_images.py
"""

import os
import sys
import json
import sqlite3
import numpy as np
import time
from PIL import Image

# ============ CONFIGURATION ============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = r"D:\PhotoAI\photo_catalog.db"
CLIP_MODEL_DIR_FAST = os.path.join(PROJECT_ROOT,
                                   "models", "clip-vit-base-patch32-onnx")
SIGLIP_MODEL_DIR = os.path.join(PROJECT_ROOT,
                                "models", "siglip2-so400m-patch14-384-onnx")
CLIP_MODEL_DIR = CLIP_MODEL_DIR_FAST
CONFIDENCE_THRESHOLD = 0.25
BATCH_SIZE = 50
HYBRID_MODE = True
HIGH_CONF_THRESHOLD = 0.45
CLIP_BATCH_SIZE = 32     # Images per GPU call for CLIP ViT-B/32
SIGLIP_BATCH_SIZE = 2   # Images per GPU call for SigLIP2 SO400M
CLASSIFY_FACE_IMAGES = False
ENSEMBLE_MODE = False

# ---- Load overrides from pipeline_config.json (written by GUI) ----
_cfg_path = os.path.join(PROJECT_ROOT, "pipeline_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path, 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    DB_PATH              = _cfg.get("db_path", DB_PATH)
    CONFIDENCE_THRESHOLD = float(_cfg.get("confidence_threshold", CONFIDENCE_THRESHOLD))
    _hybrid_mode_raw     = _cfg.get("hybrid_mode", HYBRID_MODE)
    if isinstance(_hybrid_mode_raw, str):
        HYBRID_MODE = _hybrid_mode_raw.strip().lower() in ("1", "true", "yes", "on")
    else:
        HYBRID_MODE = bool(_hybrid_mode_raw)
    HIGH_CONF_THRESHOLD  = float(_cfg.get("high_conf_threshold", HIGH_CONF_THRESHOLD))
    CLIP_BATCH_SIZE      = int(_cfg.get("clip_batch_size", CLIP_BATCH_SIZE))
    SIGLIP_BATCH_SIZE    = int(_cfg.get("siglip_batch_size", SIGLIP_BATCH_SIZE))
    _ensemble_mode_raw   = _cfg.get("ensemble_mode", ENSEMBLE_MODE)
    if isinstance(_ensemble_mode_raw, str):
        ENSEMBLE_MODE = _ensemble_mode_raw.strip().lower() in ("1", "true", "yes", "on")
    else:
        ENSEMBLE_MODE = bool(_ensemble_mode_raw)
    clip_model_size      = str(_cfg.get("clip_model_size", "clip")).lower()
    if clip_model_size in ("clip", "large"):
        CLIP_MODEL_DIR = os.path.join(PROJECT_ROOT,
                                      "models", "clip-vit-large-patch14-onnx")
    elif clip_model_size in ("siglip2", "ultra"):
        CLIP_MODEL_DIR = os.path.join(PROJECT_ROOT,
                                      "models", "siglip2-so400m-patch14-384-onnx")
    elif clip_model_size == "base":
        # Backward compatibility for older configs.
        CLIP_MODEL_DIR = os.path.join(PROJECT_ROOT,
                                      "models", "clip-vit-base-patch32-onnx")
    # Direct model path override — takes priority over clip_model_size radio logic
    if "clip_model_dir" in _cfg:
        CLIP_MODEL_DIR = _cfg["clip_model_dir"]

# ---- CATEGORY DEFINITIONS ----
# CLIP uses natural language descriptions for zero-shot classification.
# Each category maps to a list of text prompts that describe it.
# Multiple prompts per category improve accuracy through ensembling.
CATEGORIES = {
    "Documents": [
        "a scanned document or paper",
        "a screenshot of a text document",
        "a receipt or invoice",
        "printed text on paper",
    ],
    "Sensitive_Documents": [
        "a photo of an ID card or passport",
        "a photo of a bank statement or financial document",
        "a legal document or contract",
        "a photo of a credit card or personal identification",
    ],
    "Landscapes": [
        "a landscape photograph of nature",
        "a scenic outdoor view with mountains or fields",
        "a photo of the sky, sunset, or sunrise",
        "a beach or ocean view",
    ],
    "Architecture": [
        "a photo of a building or architecture",
        "an interior of a room or house",
        "a cityscape or urban skyline",
        "a photo of a mosque, church, or monument",
    ],
    "Vehicles": [
        "a photo of a car, truck, or motorcycle",
        "a vehicle on a road",
        "an airplane or aircraft",
        "a photo of a boat or ship",
    ],
    "Animals": [
        "a photo of a pet cat or dog",
        "a wild animal in nature",
        "a bird or fish",
        "a close-up of an animal",
    ],
    "Food": [
        "a photo of food or a meal",
        "a dish or plate of food at a restaurant",
        "cooking or food preparation",
        "a photo of fruits, vegetables, or groceries",
    ],
    "Events": [
        "a party or celebration",
        "a wedding ceremony",
        "a gathering of people at an event",
        "a concert or performance",
    ],
    "Electronics_Gadgets": [
        "a photo of a smartphone or tablet",
        "a computer setup or monitor",
        "electronic gadgets or hardware",
        "a photo of a gaming console or tech device",
    ],
    "Office_Workspace": [
        "a photo of a desk or office workspace",
        "an organized workstation with a computer",
        "office supplies and equipment",
        "a home office or study room",
    ],
    "Coding_Screens": [
        "a screenshot of code or a code editor",
        "a terminal or command line interface",
        "a screenshot of a development environment or IDE",
        "study materials or educational content on a screen",
    ],
    "Wallpapers": [
        "a desktop wallpaper or abstract background",
        "a colorful abstract digital art wallpaper",
        "a minimalist wallpaper for a phone or computer",
        "a high-resolution artistic background image",
    ],
    "Objects": [
        "a close-up photo of an object or product",
        "a photo of household items",
        "a still life or product photography",
        "a tool or utility item",
    ],
    "Art": [
        "a drawing or painting",
        "digital art or illustration",
        "a sketch or handwritten artwork",
        "graphic design or artistic creation",
    ],
    "Memes_Screenshots": [
        "a social media screenshot",
        "a meme or funny internet image",
        "a screenshot from a messaging app",
        "a screenshot of a tweet or social media post",
    ],
    "Books": [
        "a photograph of a book cover",
        "a stack of books on a shelf or table",
        "an open book showing printed pages",
        "a bookshelf filled with books in a library",
    ],
    "Icons_Logos": [
        "an app icon or application logo on a screen",
        "a brand logo or corporate emblem",
        "a badge, sticker, or graphic symbol design",
        "a simple flat icon or pictogram",
    ],
    "Historical_Art": [
        "a classical oil painting from the Renaissance era",
        "a historical sculpture or ancient artifact",
        "a museum exhibit of a famous painting",
        "a medieval or baroque art piece",
    ],
    "NSFW": [
        "an explicit adult photograph with nudity",
        "a suggestive or provocative photo of a person",
        "an adult or mature content image",
        "an intimate or sexually explicit photograph",
    ],
    "Social_Media_Posts": [
        "a screenshot of an Instagram post or story",
        "a Facebook post or status update screenshot",
        "a TikTok video still or thumbnail",
        "a social media feed showing posts and comments",
    ],
    "Holy_Places": [
        "a photograph of a mosque with minarets and domes",
        "a church, cathedral, or chapel interior or exterior",
        "a Hindu temple, Buddhist shrine, or sacred monument",
        "a holy site or religious pilgrimage destination",
    ],
}
# =======================================


def setup_classifications_table(conn):
    """Create the classifications table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER UNIQUE,
            category TEXT,
            confidence REAL,
            top3_categories TEXT,
            has_faces INTEGER DEFAULT 0,
            FOREIGN KEY(image_id) REFERENCES images(id)
        )
    ''')
    conn.commit()


def load_clip_model(model_dir=None):
    """Load the CLIP or SigLIP ONNX model with DirectML acceleration."""
    model_dir = model_dir or CLIP_MODEL_DIR

    try:
        from optimum.onnxruntime import ORTModelForZeroShotImageClassification
        from transformers import AutoProcessor
    except ImportError:
        print("[ERROR] Required packages not found.")
        print("Install: pip install optimum[onnxruntime] transformers")
        sys.exit(1)

    if not os.path.exists(model_dir):
        print(f"[ERROR] Classification model not found at: {model_dir}")
        print("Run scripts/setup_clip_model.py first to export the model.")
        sys.exit(1)

    # Detect if this is a SigLIP model
    is_siglip = "siglip" in model_dir.lower()
    model_type = "SigLIP 2" if is_siglip else "CLIP"
    print(f"  Loading {model_type} model from ONNX...")

    # Determine the provider
    import onnxruntime as ort
    available_providers = ort.get_available_providers()

    provider = "CPUExecutionProvider"
    if 'DmlExecutionProvider' in available_providers:
        provider = "DmlExecutionProvider"

    print(f"  Execution provider: {provider}")

    model = ORTModelForZeroShotImageClassification.from_pretrained(
        model_dir,
        provider=provider,
    )
    processor = AutoProcessor.from_pretrained(model_dir)

    return model, processor, is_siglip


def build_candidate_labels(categories_dict, ensemble_mode=False):
    """
    Build the list of candidate text labels for CLIP/SigLIP.
    """
    labels = []
    label_to_category = {}
    for cat_name, prompts in categories_dict.items():
        selected_prompts = prompts if ensemble_mode else prompts[:1]
        for prompt in selected_prompts:
            labels.append(prompt)
            label_to_category[prompt] = cat_name

    return labels, label_to_category


def classify_with_ensemble(model, processor, images, categories_dict, is_siglip=False, ensemble_mode=False):
    """
    Classify one image or a batch using ensemble of prompts per category.
    Returns one result list for single input, or list of result lists for batch input.
    """
    single_input = not isinstance(images, list)
    if single_input:
        images = [images]

    all_labels, label_to_category = build_candidate_labels(
        categories_dict,
        ensemble_mode=ensemble_mode,
    )

    # Run inference
    inputs = processor(
        text=all_labels,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    outputs = model(**inputs)

    all_results = []
    for i in range(len(images)):
        logits = outputs.logits_per_image[i]
        if is_siglip:
            probs = logits.sigmoid().detach().numpy()
        else:
            probs = logits.softmax(dim=0).detach().numpy()

        # Aggregate probabilities by category (average across prompts)
        cat_scores = {}
        for label, prob in zip(all_labels, probs):
            cat_name = label_to_category[label]
            if cat_name not in cat_scores:
                cat_scores[cat_name] = []
            cat_scores[cat_name].append(float(prob))

        # Average probabilities per category
        results = []
        for cat_name, scores in cat_scores.items():
            avg_score = sum(scores) / len(scores)
            results.append((cat_name, avg_score))

        # Normalize if it's CLIP (SigLIP scores don't necessarily sum to 1)
        if not is_siglip:
            total = sum(s for _, s in results)
            if total > 0:
                results = [(name, score / total) for name, score in results]

        results.sort(key=lambda x: x[1], reverse=True)
        all_results.append(results)

    return all_results[0] if single_input else all_results


def unload_model(model):
    """Best-effort model cleanup between hybrid passes to free VRAM."""
    try:
        del model
        import gc
        gc.collect()
    except Exception:
        pass


def run_single_model_pipeline(conn, cursor, rows):
    """Run the existing single-model classification pipeline with batched inference."""
    model, processor, is_siglip = load_clip_model()

    start_time = time.time()
    classified = 0
    errors = 0
    model_batch_size = SIGLIP_BATCH_SIZE if is_siglip else CLIP_BATCH_SIZE
    batch_items = []  # (row_idx, image_id, file_path, face_count, pil_image)

    for idx, (image_id, file_path, face_count) in enumerate(rows, 1):
        try:
            if not os.path.exists(file_path):
                print(f"  [SKIP] File not found: {file_path}")
            else:
                # Load image (per-image guard so one bad file doesn't kill the batch)
                with Image.open(file_path) as img:
                    image = img.convert("RGB")
                batch_items.append((idx, image_id, file_path, face_count, image))

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  [ERROR] {file_path}: {e}")
            elif errors == 11:
                print(f"  ... suppressing further error messages")

        if batch_items and (len(batch_items) == model_batch_size or idx == len(rows)):
            try:
                # Classify a full/partial batch in a single forward pass
                pil_images = [item[4] for item in batch_items]
                batch_results = classify_with_ensemble(
                    model,
                    processor,
                    pil_images,
                    CATEGORIES,
                    is_siglip=is_siglip,
                    ensemble_mode=ENSEMBLE_MODE,
                )

                for (row_idx, b_image_id, _b_path, b_face_count, _), results in zip(batch_items, batch_results):
                    # Determine category
                    top_cat, top_score = results[0]
                    final_category = top_cat

                    # Build top-3 JSON
                    top3 = json.dumps(results[:3])

                    # Store in database
                    cursor.execute('''
                        INSERT OR REPLACE INTO classifications
                        (image_id, category, confidence, top3_categories, has_faces)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (b_image_id, final_category, top_score, top3, 1 if b_face_count > 0 else 0))

                    classified += 1

                    # Batch commit + progress
                    if row_idx % BATCH_SIZE == 0:
                        conn.commit()
                        elapsed = time.time() - start_time
                        rate = row_idx / elapsed
                        eta = (len(rows) - row_idx) / rate
                        print(f"  [{row_idx:>6}/{len(rows)}] "
                              f"classified: {classified} | "
                              f"{rate:.1f} img/s | "
                              f"ETA: {eta/60:.0f} min | "
                                f"last: {final_category} ({top_score:.4f})")

            except Exception as e:
                for _, _, b_path, _, _ in batch_items:
                    errors += 1
                    if errors <= 10:
                        print(f"  [ERROR] {b_path}: {e}")
                    elif errors == 11:
                        print(f"  ... suppressing further error messages")
            finally:
                for _, _, _, _, pil in batch_items:
                    try:
                        pil.close()
                    except Exception:
                        pass
                batch_items = []

    conn.commit()
    unload_model(model)
    return classified, errors


def run_hybrid_pipeline(conn, cursor, rows):
    """Run CLIP-first high-confidence pass, then refine ambiguous images with SigLIP2."""
    total = len(rows)
    errors = 0

    def flush_batch(batch, model, processor, is_siglip, error_label, on_result, on_batch_error=None):
        nonlocal errors

        if not batch:
            return

        pil_images = [item[3] for item in batch]
        try:
            batch_results = classify_with_ensemble(
                model,
                processor,
                pil_images,
                CATEGORIES,
                is_siglip=is_siglip,
                ensemble_mode=ENSEMBLE_MODE,
            )
        except Exception as e:
            print(f"  [ERROR] {error_label}: {e}")
            errors += len(batch)
            if on_batch_error is not None:
                on_batch_error(batch)
            for _, _, _, pil in batch:
                try:
                    pil.close()
                except Exception:
                    pass
            batch.clear()
            return

        for item, results in zip(batch, batch_results):
            on_result(item, results)

        for _, _, _, pil in batch:
            try:
                pil.close()
            except Exception:
                pass
        batch.clear()

    # ---- PASS 1: CLIP ViT-B/32 ----
    print(f"\n  [Hybrid Mode] Pass 1: CLIP ViT-B/32 -- batch_size={CLIP_BATCH_SIZE}")
    clip_model, clip_processor, _ = load_clip_model(CLIP_MODEL_DIR_FAST)

    ambiguous_queue = []   # (image_id, file_path, face_count)
    classified_p1 = 0
    batch = []             # (image_id, file_path, face_count, pil_image)
    processed = 0
    start = time.time()

    def on_clip_result(item, results):
        nonlocal classified_p1
        image_id, file_path, face_count, _ = item
        top_cat, top_score = results[0]
        top3 = json.dumps(results[:3])

        if top_score >= HIGH_CONF_THRESHOLD:
            cursor.execute('''
                INSERT OR REPLACE INTO classifications
                (image_id, category, confidence, top3_categories, has_faces)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_id, top_cat, top_score, top3, 1 if face_count > 0 else 0))
            classified_p1 += 1
        else:
            ambiguous_queue.append((image_id, file_path, face_count))

    def on_clip_batch_error(items):
        for image_id, file_path, face_count, _ in items:
            ambiguous_queue.append((image_id, file_path, face_count))

    for image_id, file_path, face_count in rows:
        processed += 1

        try:
            if not os.path.exists(file_path):
                print(f"  [SKIP] File not found: {file_path}")
            else:
                with Image.open(file_path) as img:
                    pil = img.convert("RGB")
                batch.append((image_id, file_path, face_count, pil))
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  [ERROR] {file_path}: {e}")
            elif errors == 11:
                print("  ... suppressing further error messages")

        if len(batch) >= CLIP_BATCH_SIZE:
            flush_batch(
                batch,
                clip_model,
                clip_processor,
                is_siglip=False,
                error_label="CLIP batch inference",
                on_result=on_clip_result,
                on_batch_error=on_clip_batch_error,
            )

        if processed % BATCH_SIZE == 0:
            conn.commit()
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta = (total - processed) / rate if rate > 0 else 0.0
            print(f"  [P1 {processed:>6}/{total}] "
                  f"confirmed: {classified_p1} | "
                  f"ambiguous: {len(ambiguous_queue)} | "
                  f"{rate:.1f} img/s | ETA: {eta/60:.0f} min")

    flush_batch(
        batch,
        clip_model,
        clip_processor,
        is_siglip=False,
        error_label="CLIP batch inference",
        on_result=on_clip_result,
        on_batch_error=on_clip_batch_error,
    )
    conn.commit()
    unload_model(clip_model)

    pct = (len(ambiguous_queue) / total * 100.0) if total else 0.0
    print("\n  Pass 1 complete.")
    print(f"  -> Confirmed by CLIP : {classified_p1} ({100 - pct:.1f}%)")
    print(f"  -> Sent to SigLIP2   : {len(ambiguous_queue)} ({pct:.1f}%)")

    # ---- PASS 2: SigLIP2 SO400M ----
    if not ambiguous_queue:
        print("\n  No ambiguous images. Skipping Pass 2.")
        return classified_p1, 0, errors

    print(f"\n  [Hybrid Mode] Pass 2: SigLIP2 SO400M -- batch_size={SIGLIP_BATCH_SIZE}")
    siglip_model, siglip_processor, _ = load_clip_model(SIGLIP_MODEL_DIR)

    classified_p2 = 0
    batch2 = []  # (image_id, file_path, face_count, pil_image)
    amb_total = len(ambiguous_queue)
    processed2 = 0
    start2 = time.time()

    def on_siglip_result(item, results):
        nonlocal classified_p2
        image_id, _file_path, face_count, _ = item
        top_cat, top_score = results[0]
        top3 = json.dumps(results[:3])

        cursor.execute('''
            INSERT OR REPLACE INTO classifications
            (image_id, category, confidence, top3_categories, has_faces)
            VALUES (?, ?, ?, ?, ?)
        ''', (image_id, top_cat, top_score, top3, 1 if face_count > 0 else 0))
        classified_p2 += 1

    for image_id, file_path, face_count in ambiguous_queue:
        processed2 += 1

        try:
            if not os.path.exists(file_path):
                print(f"  [SKIP] File not found: {file_path}")
            else:
                with Image.open(file_path) as img:
                    pil = img.convert("RGB")
                batch2.append((image_id, file_path, face_count, pil))
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  [ERROR] {file_path}: {e}")
            elif errors == 11:
                print("  ... suppressing further error messages")

        if len(batch2) >= SIGLIP_BATCH_SIZE:
            flush_batch(
                batch2,
                siglip_model,
                siglip_processor,
                is_siglip=True,
                error_label="SigLIP2 batch inference",
                on_result=on_siglip_result,
                on_batch_error=None,
            )

        if processed2 % BATCH_SIZE == 0:
            conn.commit()
            elapsed2 = time.time() - start2
            rate2 = processed2 / elapsed2 if elapsed2 > 0 else 0.0
            eta2 = (amb_total - processed2) / rate2 if rate2 > 0 else 0.0
            print(f"  [P2 {processed2:>6}/{amb_total}] "
                  f"refined: {classified_p2} | "
                  f"{rate2:.1f} img/s | ETA: {eta2/60:.0f} min")

    flush_batch(
        batch2,
        siglip_model,
        siglip_processor,
        is_siglip=True,
        error_label="SigLIP2 batch inference",
        on_result=on_siglip_result,
        on_batch_error=None,
    )
    conn.commit()
    unload_model(siglip_model)

    print(f"\n  Pass 2 complete. Refined: {classified_p2} images.")
    return classified_p1, classified_p2, errors


def main():
    print("=" * 60)
    print("  Script 3: Semantic Image Classification (CLIP/SigLIP + DirectML)")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    setup_classifications_table(conn)
    cursor = conn.cursor()

    # Get images to classify
    if CLASSIFY_FACE_IMAGES:
        # Classify all images not yet classified
        cursor.execute('''
            SELECT images.id, images.file_path, images.face_count
            FROM images
            LEFT JOIN classifications ON images.id = classifications.image_id
            WHERE classifications.id IS NULL
        ''')
    else:
        # Only classify images with NO faces
        cursor.execute('''
            SELECT images.id, images.file_path, images.face_count
            FROM images
            LEFT JOIN classifications ON images.id = classifications.image_id
            WHERE classifications.id IS NULL
            AND images.face_count = 0
        ''')

    rows = cursor.fetchall()
    print(f"\n  Images to classify: {len(rows)}")
    print(f"  Categories: {len(CATEGORIES)}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Hybrid mode: {HYBRID_MODE}")
    print(f"  High-conf threshold: {HIGH_CONF_THRESHOLD}")
    print(f"  CLIP batch size: {CLIP_BATCH_SIZE}")
    print(f"  SigLIP batch size: {SIGLIP_BATCH_SIZE}")
    print(f"  Classify face images: {CLASSIFY_FACE_IMAGES}\n")

    if not rows:
        print("  Nothing to classify. All eligible images already processed.")
        conn.close()
        return

    start_time = time.time()
    if HYBRID_MODE:
        p1, p2, errors = run_hybrid_pipeline(conn, cursor, rows)
        classified = p1 + p2
    else:
        classified, errors = run_single_model_pipeline(conn, cursor, rows)

    conn.commit()

    # ---- Print summary ----
    cursor.execute('''
        SELECT category, COUNT(*) as cnt
        FROM classifications
        GROUP BY category
        ORDER BY cnt DESC
    ''')
    category_counts = cursor.fetchall()

    conn.close()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Classification complete!")
    print(f"  Classified : {classified}")
    print(f"  Errors     : {errors}")
    print(f"  Time       : {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"  Speed      : {classified/elapsed:.1f} images/sec")

    print(f"\n  Category distribution:")
    for cat, cnt in category_counts:
        bar = "█" * min(cnt // 5, 40)
        print(f"    {cat:<25} {cnt:>5}  {bar}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
