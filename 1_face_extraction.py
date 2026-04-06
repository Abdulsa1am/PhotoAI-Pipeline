"""
Script 1: Face Extraction using InsightFace + ONNX Runtime DirectML

Scans a directory of images, detects faces using RetinaFace, extracts
512D ArcFace embeddings, and stores everything in an SQLite database.

Runs on AMD RX 5700 XT via DirectML for GPU acceleration.

Requirements:
    pip install insightface onnxruntime-directml opencv-python numpy Pillow
"""

import os
import sqlite3
import pickle
import json
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import Base as ExifBase
from insightface.app import FaceAnalysis
import time

# ============ CONFIGURATION ============
MASTER_DIR = r"D:\POCOP - Copy\+"
DB_PATH    = r"D:\PhotoAI\photo_catalog.db"

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
DET_SIZE = (640, 640)
DET_THRESH = 0.5
MAX_DIM = 1920
BATCH_SIZE = 50

# ---- Load overrides from pipeline_config.json (written by GUI) ----
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path, 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    MASTER_DIR = _cfg.get("master_dir", MASTER_DIR)
    DB_PATH    = _cfg.get("db_path", DB_PATH)
    _ds        = _cfg.get("det_size", DET_SIZE[0] if isinstance(DET_SIZE, tuple) else DET_SIZE)
    DET_SIZE   = (int(_ds), int(_ds))
    DET_THRESH = float(_cfg.get("det_thresh", DET_THRESH))
    MAX_DIM    = int(_cfg.get("max_dim", MAX_DIM))
# =======================================


def setup_database():
    """Create/connect to the SQLite database with the required schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            custom_name TEXT,
            centroid BLOB
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            face_count INTEGER DEFAULT 0
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            encoding BLOB,
            bbox TEXT,
            det_score REAL,
            FOREIGN KEY(image_id) REFERENCES images(id)
        )
    ''')

    # Ensure face_count column exists (for databases from older schema)
    try:
        cursor.execute("SELECT face_count FROM images LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE images ADD COLUMN face_count INTEGER DEFAULT 0")

    # Ensure bbox/det_score columns exist
    try:
        cursor.execute("SELECT bbox, det_score, person_id FROM faces LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cursor.execute("ALTER TABLE faces ADD COLUMN bbox TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE faces ADD COLUMN det_score REAL")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE faces ADD COLUMN person_id INTEGER")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    return conn


def init_face_app():
    """Initialize InsightFace with DirectML (AMD GPU) acceleration."""
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']

    # 'buffalo_l' includes:
    #   - RetinaFace-10GF detector
    #   - ArcFace-R100 recognition (512D embeddings)
    app = FaceAnalysis(
        name='buffalo_l',
        providers=providers
    )
    app.prepare(ctx_id=0, det_size=DET_SIZE, det_thresh=DET_THRESH)

    print(f"  InsightFace initialized")
    print(f"  Providers: {providers}")
    print(f"  Det size:  {DET_SIZE}")
    return app


def load_with_exif_rotation(file_path):
    """Load image and apply EXIF orientation correction."""
    try:
        pil_img = Image.open(file_path)

        # Apply EXIF rotation if present
        try:
            exif = pil_img.getexif()
            orientation = exif.get(ExifBase.Orientation, 1)
            rotation_map = {3: 180, 6: 270, 8: 90}
            if orientation in rotation_map:
                pil_img = pil_img.rotate(rotation_map[orientation], expand=True)
        except Exception:
            pass

        # Convert to RGB numpy array then to BGR for OpenCV
        pil_img = pil_img.convert('RGB')
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    except Exception:
        # Fallback to OpenCV if PIL fails
        return cv2.imread(file_path)


def collect_image_paths(master_dir):
    """Pre-scan all image paths."""
    paths = []
    for root, _, files in os.walk(master_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return paths


def process_images():
    conn = setup_database()
    cursor = conn.cursor()

    print("=" * 60)
    print("  Script 1: Face Extraction (InsightFace + DirectML)")
    print("=" * 60)

    app = init_face_app()

    # Collect all image paths
    all_paths = collect_image_paths(MASTER_DIR)
    print(f"\n  Source directory: {MASTER_DIR}")
    print(f"  Total images found: {len(all_paths)}")

    # Filter out already-processed
    cursor.execute("SELECT file_path FROM images")
    processed = set(row[0] for row in cursor.fetchall())
    pending = [p for p in all_paths if p not in processed]
    print(f"  Already processed: {len(processed)}")
    print(f"  Pending: {len(pending)}\n")

    if not pending:
        print("Nothing to process. All images already in database.")
        conn.close()
        return

    start_time = time.time()
    total_faces = 0
    errors = 0

    for idx, file_path in enumerate(pending, 1):
        try:
            # Load image with EXIF rotation handling
            img = load_with_exif_rotation(file_path)
            if img is None:
                print(f"  [SKIP] Cannot read: {file_path}")
                continue

            # Resize very large images for speed
            h, w = img.shape[:2]
            if max(h, w) > MAX_DIM:
                scale = MAX_DIM / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)

            # Detect faces + extract embeddings (GPU accelerated)
            faces = app.get(img)
            face_count = len(faces)

            # Insert image record with face count
            cursor.execute(
                "INSERT INTO images (file_path, face_count) VALUES (?, ?)",
                (file_path, face_count)
            )
            image_id = cursor.lastrowid

            # Insert each face encoding
            for face in faces:
                encoding_bytes = pickle.dumps(face.embedding)
                bbox_str = str(face.bbox.tolist())
                det_score = float(face.det_score)
                cursor.execute(
                    "INSERT INTO faces (image_id, encoding, bbox, det_score) VALUES (?, ?, ?, ?)",
                    (image_id, encoding_bytes, bbox_str, det_score)
                )
                total_faces += 1

            # Periodic batch commit + progress report
            if idx % BATCH_SIZE == 0:
                conn.commit()
                elapsed = time.time() - start_time
                rate = idx / elapsed
                eta = (len(pending) - idx) / rate
                print(f"  [{idx:>6}/{len(pending)}] "
                      f"faces: {total_faces} | "
                      f"{rate:.1f} img/s | "
                      f"ETA: {eta/60:.0f} min")

        except Exception as e:
            errors += 1
            print(f"  [ERROR] {file_path}: {e}")

    # Final commit
    conn.commit()
    conn.close()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Extraction complete!")
    print(f"  Images processed : {len(pending)}")
    print(f"  Faces extracted  : {total_faces}")
    print(f"  Errors           : {errors}")
    print(f"  Total time       : {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"  Average speed    : {len(pending)/elapsed:.1f} images/sec")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    process_images()