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
CLIP_MODEL_DIR = os.path.join(PROJECT_ROOT,
                              "models", "clip-vit-base-patch32-onnx")
CONFIDENCE_THRESHOLD = 0.18
BATCH_SIZE = 50
CLASSIFY_FACE_IMAGES = False

# ---- Load overrides from pipeline_config.json (written by GUI) ----
_cfg_path = os.path.join(PROJECT_ROOT, "pipeline_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path, 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    DB_PATH              = _cfg.get("db_path", DB_PATH)
    CONFIDENCE_THRESHOLD = float(_cfg.get("confidence_threshold", CONFIDENCE_THRESHOLD))
    clip_model_size      = _cfg.get("clip_model_size", "base")
    if clip_model_size == "large":
        CLIP_MODEL_DIR = os.path.join(PROJECT_ROOT,
                                      "models", "clip-vit-large-patch14-onnx")

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


def load_clip_model():
    """Load the CLIP or SigLIP ONNX model with DirectML acceleration."""
    try:
        from optimum.onnxruntime import ORTModelForZeroShotImageClassification
        from transformers import AutoProcessor
    except ImportError:
        print("[ERROR] Required packages not found.")
        print("Install: pip install optimum[onnxruntime] transformers")
        sys.exit(1)

    if not os.path.exists(CLIP_MODEL_DIR):
        print(f"[ERROR] Classification model not found at: {CLIP_MODEL_DIR}")
        print("Run scripts/setup_clip_model.py first to export the model.")
        sys.exit(1)

    # Detect if this is a SigLIP model
    is_siglip = "siglip" in CLIP_MODEL_DIR.lower()
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
        CLIP_MODEL_DIR,
        provider=provider,
    )
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_DIR)

    return model, processor, is_siglip


def build_candidate_labels():
    """
    Build the list of candidate text labels for CLIP/SigLIP.
    """
    labels = []
    category_names = []
    for cat_name, prompts in CATEGORIES.items():
        labels.append(prompts[0])  # Primary prompt
        category_names.append(cat_name)

    return labels, category_names


def classify_with_ensemble(model, processor, image, categories_dict, is_siglip=False):
    """
    Classify an image using ensemble of multiple prompts per category.
    Returns list of (category_name, score) sorted by score descending.
    """
    all_labels = []
    label_to_category = {}
    for cat_name, prompts in categories_dict.items():
        for prompt in prompts:
            all_labels.append(prompt)
            label_to_category[prompt] = cat_name

    # Run inference
    inputs = processor(
        text=all_labels,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    outputs = model(**inputs)

    # SigLIP and CLIP handle logits differently
    # SigLIP 2 uses sigmoid-based probs usually, but ORTModelForZeroShot
    # might wrap it. Let's handle both.
    if is_siglip:
        # SigLIP probabilities are independent (multilabel style)
        # but for classification we look at relative strength
        logits = outputs.logits_per_image[0]
        probs = logits.sigmoid().detach().numpy()
    else:
        # CLIP uses softmax across candidates
        logits = outputs.logits_per_image[0]
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
    return results


def main():
    print("=" * 60)
    print("  Script 3: Semantic Image Classification (CLIP/SigLIP + DirectML)")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    setup_classifications_table(conn)
    cursor = conn.cursor()

    # Load model
    model, processor, is_siglip = load_clip_model()

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
    print(f"  Classify face images: {CLASSIFY_FACE_IMAGES}\n")

    if not rows:
        print("  Nothing to classify. All eligible images already processed.")
        conn.close()
        return

    start_time = time.time()
    classified = 0
    errors = 0

    for idx, (image_id, file_path, face_count) in enumerate(rows, 1):
        try:
            if not os.path.exists(file_path):
                print(f"  [SKIP] File not found: {file_path}")
                continue

            # Load image
            image = Image.open(file_path).convert("RGB")

            # Classify using ensemble of prompts
            results = classify_with_ensemble(model, processor, image, CATEGORIES, is_siglip=is_siglip)

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
            ''', (image_id, final_category, top_score, top3, 1 if face_count > 0 else 0))

            classified += 1

            # Batch commit + progress
            if idx % BATCH_SIZE == 0:
                conn.commit()
                elapsed = time.time() - start_time
                rate = idx / elapsed
                eta = (len(rows) - idx) / rate
                print(f"  [{idx:>6}/{len(rows)}] "
                      f"classified: {classified} | "
                      f"{rate:.1f} img/s | "
                      f"ETA: {eta/60:.0f} min | "
                      f"last: {final_category} ({top_score:.2f})")

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  [ERROR] {file_path}: {e}")
            elif errors == 11:
                print(f"  ... suppressing further error messages")

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
