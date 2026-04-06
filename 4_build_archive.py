"""
Script 4: Archive Builder — Final Directory Organization

Reads the complete database (face clusters + semantic classifications)
and builds the final organized archive directory structure.

Uses hard links (os.link) for same-drive organization (zero disk cost,
100% metadata preservation). Falls back to shutil.copy2 for cross-drive.

Prerequisites:
    - Script 1 has been run (face extraction)
    - Script 2 has been run (face clustering — populates cluster labels)
    - Script 3 has been run (semantic classification)

Usage:
    python 4_build_archive.py
"""

import os
import sys
import shutil
import sqlite3
import pickle
import json
import numpy as np
import time

# ============ CONFIGURATION ============
DB_PATH         = r"D:\PhotoAI\photo_catalog.db"
BASE_PHOTOS_DIR = r"D:\Photos"
MASTER_DIR      = r"D:\POCOP - Copy\+"
FILE_MODE       = "auto"

MIN_CLUSTER_SIZE = 4
MERGE_THRESHOLD  = 0.40

# ---- Load overrides from pipeline_config.json (written by GUI) ----
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path, 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    DB_PATH          = _cfg.get("db_path", DB_PATH)
    BASE_PHOTOS_DIR  = _cfg.get("output_dir", BASE_PHOTOS_DIR)
    MASTER_DIR       = _cfg.get("master_dir", MASTER_DIR)
    MIN_CLUSTER_SIZE = int(_cfg.get("min_cluster_size", MIN_CLUSTER_SIZE))
    MERGE_THRESHOLD  = float(_cfg.get("merge_threshold", MERGE_THRESHOLD))
# =======================================

placed_source_paths = set()

def get_next_archive_dir(merge_thresh, conf_thresh):
    """Create the next sequential archive directory."""
    os.makedirs(BASE_PHOTOS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(BASE_PHOTOS_DIR)
                if os.path.isdir(os.path.join(BASE_PHOTOS_DIR, f))
                and f.startswith("Archive_")]

    max_num = 0
    for folder in existing:
        parts = folder.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            max_num = max(max_num, int(parts[1]))

    next_num = max_num + 1
    return os.path.join(BASE_PHOTOS_DIR, f"Archive_{next_num}_merge{merge_thresh}_conf{conf_thresh}")


def place_file(src, dst, mode="auto"):
    """
    Attempts to place a file at dst from src using the specified mode.
    Returns True if successful.
    """
    success = False
    if os.path.exists(dst):
        success = True  # Already exists
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if mode == "hardlink":
            try:
                os.link(src, dst)
                success = True
            except Exception as e:
                print(f"  [LINK FAIL] {e}")
                
        elif mode == "copy":
            try:
                shutil.copy2(src, dst)
                success = True
            except Exception as e:
                print(f"  [COPY FAIL] {e}")

        elif mode == "auto":
            try:
                os.link(src, dst)
                success = True
            except OSError:
                try:
                    shutil.copy2(src, dst)
                    success = True
                except Exception as e:
                    print(f"  [PLACE FAIL] {src}: {e}")

    if success:
        placed_source_paths.add(os.path.normcase(os.path.abspath(src)))
    return success


def load_cluster_labels(db_path):
    """
    Load face cluster assignments directly from the persistent database.
    Returns:
      image_persons: dict of image_id -> set of resolved string names (e.g., 'Mom', 'Person_001_5faces', 'Unknown_Faces')
      image_paths: dict of image_id -> original file_path
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Pre-compute face counts per person to append to the untagged folder names
        cursor.execute("SELECT person_id, COUNT(*) FROM faces WHERE person_id > 0 GROUP BY person_id")
        face_counts = {r[0]: r[1] for r in cursor.fetchall()}

        cursor.execute('''
            SELECT faces.image_id, images.file_path, faces.person_id, people.custom_name
            FROM faces
            JOIN images ON faces.image_id = images.id
            LEFT JOIN people ON faces.person_id = people.id
            WHERE faces.person_id IS NOT NULL AND faces.person_id != ""
        ''')
        rows = cursor.fetchall()
        conn.close()

    except sqlite3.OperationalError:
        print("  Database missing mapping columns? Run Script 1 to rebuild DB.")
        return {}, {}

    image_persons = {}
    image_paths = {}

    for img_id, file_path, person_id, custom_name in rows:
        if person_id == -1:
            name = "Unknown_Faces"
        else:
            if custom_name and custom_name.strip():
                name = custom_name.strip()
            else:
                count = face_counts.get(person_id, 0)
                name = f"Person_{person_id:03d}_{count}faces"

        if img_id not in image_persons:
            image_persons[img_id] = set()
        image_persons[img_id].add(name)
        image_paths[img_id] = file_path

    return image_persons, image_paths


def main():
    print("=" * 60)
    print("  Script 4: Archive Builder")
    print("=" * 60)

    # Get thresholds from config for the directory name
    merge_t = _cfg.get("merge_threshold", 0.40) if '_cfg' in globals() else 0.40
    conf_t  = _cfg.get("confidence_threshold", 0.15) if '_cfg' in globals() else 0.15
    archive_dir = get_next_archive_dir(merge_t, conf_t)
    print(f"\n  Building archive: {archive_dir}")
    print(f"  File mode: {FILE_MODE}\n")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ---- Load all images with reliable face count from faces table ----
    # We use COUNT(faces.id) rather than images.face_count because the
    # face_count column may be stale (e.g. from a pre-ArcFace database).
    cursor.execute("""
        SELECT images.id, images.file_path, COUNT(faces.id) as real_face_count
        FROM images
        LEFT JOIN faces ON faces.image_id = images.id
        GROUP BY images.id
    """)
    all_images = cursor.fetchall()  # (id, file_path, real_face_count)
    print(f"  Total images in database: {len(all_images)}")

    # ---- Load classifications ----
    cursor.execute("SELECT image_id, category, confidence FROM classifications")
    classification_map = {}
    for img_id, category, confidence in cursor.fetchall():
        classification_map[img_id] = (category, confidence)
    print(f"  Images with classifications: {len(classification_map)}")

    conn.close()

    # ---- Split by actual face presence (faces table is ground truth) ----
    face_images    = [img for img in all_images if img[2] > 0]
    no_face_images = [img for img in all_images if img[2] == 0]
    print(f"  Images WITH faces   : {len(face_images)}")
    print(f"  Images without faces: {len(no_face_images)}")

    image_persons = {}
    if face_images:
        image_persons, _ = load_cluster_labels(DB_PATH)

    # ---- Create directory structure ----
    people_dir = os.path.join(archive_dir, "People")
    small_group_dir = os.path.join(people_dir, "SmallGroups_2-3_faces")
    large_group_dir = os.path.join(people_dir, "LargeGroups_4+_faces")
    unknown_dir = os.path.join(people_dir, "Unknown_Faces")
    os.makedirs(people_dir, exist_ok=True)
    os.makedirs(small_group_dir, exist_ok=True)
    os.makedirs(large_group_dir, exist_ok=True)
    os.makedirs(unknown_dir, exist_ok=True)

    start_time = time.time()
    placed = 0
    skipped = 0
    errors = 0

    # ---- Non-personal categories: images with faces in these categories
    #      go to their semantic folder instead of People/ ----
    NON_PERSONAL_CATEGORIES = {
        "Memes_Screenshots", "Documents", "Sensitive_Documents",
        "Coding_Screens", "Wallpapers", "Art", "Historical_Art",
        "Icons_Logos", "Social_Media_Posts", "NSFW", "Books",
    }

    # ---- Phase 1: Place face images into People folders ----
    print(f"\n  Phase 1: Organizing face images...")

    # Compute face counts per image for group photo detection
    face_count_map = {img[0]: img[2] for img in all_images}

    # Get unique person labels and create directories
    all_person_labels = set()
    for persons in image_persons.values():
        all_person_labels.update(persons)

    person_dirs = {}
    for label in sorted(all_person_labels):
        if label == "Unknown_Faces":
            person_dirs[label] = unknown_dir
        else:
            pdir = os.path.join(people_dir, str(label))
            os.makedirs(pdir, exist_ok=True)
            person_dirs[label] = pdir

    rerouted_to_semantic = 0

    # Place each face image into its person folder(s)
    for img_id, file_path, face_count in face_images:
        if not os.path.exists(file_path):
            skipped += 1
            continue

        fname = os.path.basename(file_path)
        unique_fname = f"{img_id}_{fname}"

        # --- Smart routing: check if CLIP says this is a non-personal image ---
        if img_id in classification_map:
            clip_cat, clip_conf = classification_map[img_id]
            if clip_cat in NON_PERSONAL_CATEGORIES and clip_conf >= conf_t:
                # Route to semantic category instead of People/
                cat_dir = os.path.join(archive_dir, clip_cat)
                os.makedirs(cat_dir, exist_ok=True)
                dst = os.path.join(cat_dir, unique_fname)
                if place_file(file_path, dst, FILE_MODE):
                    placed += 1
                    rerouted_to_semantic += 1
                continue  # Skip People/ placement entirely

        if img_id in image_persons:
            labels = image_persons[img_id]
            for label in labels:
                pdir = person_dirs.get(label, unknown_dir)
                dst = os.path.join(pdir, unique_fname)
                if place_file(file_path, dst, FILE_MODE):
                    placed += 1
                else:
                    errors += 1

            # Small groups (2-3 faces)
            if 2 <= face_count <= 3:
                dst = os.path.join(small_group_dir, unique_fname)
                place_file(file_path, dst, FILE_MODE)
            # Large groups (4+ faces)
            elif face_count >= 4:
                dst = os.path.join(large_group_dir, unique_fname)
                place_file(file_path, dst, FILE_MODE)
        else:
            # Face detected but no cluster (shouldn't happen, fallback)
            dst = os.path.join(unknown_dir, unique_fname)
            if place_file(file_path, dst, FILE_MODE):
                placed += 1

    print(f"    Placed {placed - rerouted_to_semantic} files into People folders")
    if rerouted_to_semantic > 0:
        print(f"    Rerouted {rerouted_to_semantic} non-personal images (memes/docs/screenshots) to semantic categories")

    # ---- Phase 2: Place non-face images into semantic categories ----
    print(f"\n  Phase 2: Organizing non-face images by category...")

    category_placed = 0
    uncategorized = 0

    for img_id, file_path, face_count in no_face_images:
        if not os.path.exists(file_path):
            skipped += 1
            continue

        fname = os.path.basename(file_path)
        unique_fname = f"{img_id}_{fname}"

        if img_id in classification_map:
            category, confidence = classification_map[img_id]
            # Enforce dynamic classification confidence threshold
            if confidence < conf_t:
                category = "Other"
        else:
            category = "Uncategorized"

        cat_dir = os.path.join(archive_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        dst = os.path.join(cat_dir, unique_fname)
        if place_file(file_path, dst, FILE_MODE):
            placed += 1
            category_placed += 1
    print(f"    Placed {category_placed} files into category folders")

    # ---- Phase 3: Handle unprocessed images (no-face, no-classification) ----
    # Images in DB but not classified (e.g., Script 3 wasn't run yet).
    # Use faces table join (not face_count column) as source of truth.
    cursor_check = sqlite3.connect(DB_PATH).cursor()
    cursor_check.execute('''
        SELECT images.id, images.file_path
        FROM images
        LEFT JOIN faces ON faces.image_id = images.id
        LEFT JOIN classifications ON classifications.image_id = images.id
        WHERE faces.id IS NULL
          AND classifications.image_id IS NULL
        GROUP BY images.id
    ''')
    unprocessed = cursor_check.fetchall()

    if unprocessed:
        print(f"\n  Phase 3: {len(unprocessed)} unclassified images -> Uncategorized/")
        uncat_dir = os.path.join(archive_dir, "Uncategorized")
        os.makedirs(uncat_dir, exist_ok=True)

        for img_id, file_path in unprocessed:
            if not os.path.exists(file_path):
                continue
            fname = os.path.basename(file_path)
            dst = os.path.join(uncat_dir, f"{img_id}_{fname}")
            if place_file(file_path, dst, FILE_MODE):
                placed += 1
                uncategorized += 1

        print(f"    Placed {uncategorized} files into Uncategorized/")

    cursor_check.connection.close()

    # ---- Phase 4 removed (User requested to skip unsupported formats entirely) ----

    # ---- Print final summary ----
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"  Archive build complete!")
    print(f"  Total files placed : {placed}")
    print(f"  Skipped (missing)  : {skipped}")
    print(f"  Errors             : {errors}")
    print(f"  Time               : {elapsed:.1f}s")

    # Show directory tree summary
    print(f"\n  Archive structure:")
    if os.path.exists(archive_dir):
        for entry in sorted(os.listdir(archive_dir)):
            entry_path = os.path.join(archive_dir, entry)
            if os.path.isdir(entry_path):
                count = sum(1 for f in os.listdir(entry_path)
                            if os.path.isfile(os.path.join(entry_path, f)))
                subdirs = sum(1 for f in os.listdir(entry_path)
                              if os.path.isdir(os.path.join(entry_path, f)))
                if subdirs > 0:
                    print(f"    📁 {entry}/  ({subdirs} subfolders, {count} files)")
                else:
                    print(f"    📁 {entry}/  ({count} files)")

    print(f"\n  Archive location: {archive_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
