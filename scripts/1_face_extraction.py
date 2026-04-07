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
import traceback
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import Base as ExifBase
from insightface.app import FaceAnalysis
import time

try:
    import onnxruntime as ort
except Exception:
    ort = None

# ============ CONFIGURATION ============
MASTER_DIR = r"D:\POCOP - Copy\+"
DB_PATH    = r"D:\PhotoAI\photo_catalog.db"

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
DET_SIZE = (640, 640)
DET_THRESH = 0.65
MIN_FACE_AREA_PX = 1600
MAX_DIM = 1920
BATCH_SIZE = 50

# ---- Load overrides from pipeline_config.json (written by GUI) ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cfg_path = os.path.join(PROJECT_ROOT, "pipeline_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path, 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    MASTER_DIR = _cfg.get("master_dir", MASTER_DIR)
    DB_PATH    = _cfg.get("db_path", DB_PATH)
    _ds        = _cfg.get("det_size", DET_SIZE[0] if isinstance(DET_SIZE, tuple) else DET_SIZE)
    DET_SIZE   = (int(_ds), int(_ds))
    DET_THRESH = float(_cfg.get("det_thresh", DET_THRESH))
    MIN_FACE_AREA_PX = int(_cfg.get("min_face_area_px", MIN_FACE_AREA_PX))
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
            centroid BLOB,
            n_confirmed INTEGER DEFAULT 0,
            baseline_cohesion REAL DEFAULT 1.0,
            current_cohesion REAL DEFAULT 1.0,
            created_at TEXT,
            last_updated TEXT
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

    # Ensure assignment/review metadata columns exist
    for alter_sql in [
        "ALTER TABLE faces ADD COLUMN assignment_status TEXT",
        "ALTER TABLE faces ADD COLUMN review_candidate_ids TEXT",
        "ALTER TABLE faces ADD COLUMN review_candidate_scores TEXT",
        "ALTER TABLE faces ADD COLUMN review_margin REAL",
        "ALTER TABLE faces ADD COLUMN review_best_similarity REAL",
    ]:
        try:
            cursor.execute(alter_sql)
        except sqlite3.OperationalError:
            pass

    # Ensure identity stability tracking columns exist on older DBs
    for alter_sql in [
        "ALTER TABLE people ADD COLUMN n_confirmed INTEGER DEFAULT 0",
        "ALTER TABLE people ADD COLUMN baseline_cohesion REAL DEFAULT 1.0",
        "ALTER TABLE people ADD COLUMN current_cohesion REAL DEFAULT 1.0",
        "ALTER TABLE people ADD COLUMN created_at TEXT",
        "ALTER TABLE people ADD COLUMN last_updated TEXT",
    ]:
        try:
            cursor.execute(alter_sql)
        except sqlite3.OperationalError:
            pass

    conn.commit()
    return conn


def _runtime_providers():
    if ort is None:
        return []
    try:
        return ort.get_available_providers()
    except Exception:
        return []


def _is_dml_reshape_runtime_error(exc):
    msg = f"{type(exc).__name__}: {exc}"
    return (
        "Reshape_223" in msg
        and "ONNXRuntimeError" in msg
        and ("DmlExecutionProvider" in msg or "80070057" in msg)
    )


def init_face_app(providers=None, det_size=None):
    """Initialize InsightFace with requested providers."""
    if providers is None:
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    if det_size is None:
        det_size = DET_SIZE

    # 'buffalo_l' includes:
    #   - RetinaFace-10GF detector
    #   - ArcFace-R100 recognition (512D embeddings)
    app = FaceAnalysis(
        name='buffalo_l',
        providers=providers
    )
    app.prepare(ctx_id=0, det_size=det_size, det_thresh=DET_THRESH)

    print(f"  InsightFace initialized")
    print(f"  Providers: {providers}")
    print(f"  Det size:  {det_size}")
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

    print(f"  Effective config:")
    print(f"    Source dir : {MASTER_DIR}")
    print(f"    DB path    : {DB_PATH}")
    print(f"    Det size   : {DET_SIZE}")
    print(f"    Det thresh : {DET_THRESH}")
    print(f"    Min face area  : {MIN_FACE_AREA_PX}px²")
    print(f"    Max dim    : {MAX_DIM}")

    if not os.path.isdir(MASTER_DIR):
        print(f"\n[FATAL] Source directory does not exist: {MASTER_DIR}")
        conn.close()
        return

    available_providers = _runtime_providers()
    if ort is not None:
        print(f"  ONNX Runtime version: {ort.__version__}")
    print(f"  Available providers : {available_providers if available_providers else 'unknown'}")

    requested_det_side = DET_SIZE[0] if isinstance(DET_SIZE, tuple) else int(DET_SIZE)
    runtime_det_size = DET_SIZE
    gpu_safe_upscale_factor = 1.0

    # Known DirectML bug in some ORT versions causes RetinaFace reshape failures for
    # det_size values other than 640. Keep GPU active by running 640 internally and
    # upscaling input image to preserve small-face sensitivity.
    if 'DmlExecutionProvider' in available_providers and requested_det_side != 640:
        runtime_det_size = (640, 640)
        gpu_safe_upscale_factor = max(1.0, requested_det_side / 640.0)
        print(f"  [INFO] GPU-safe high-accuracy mode enabled")
        print(f"  [INFO] Requested det_size={requested_det_side}, runtime det_size=(640, 640)")
        print(f"  [INFO] Input upscaling factor: x{gpu_safe_upscale_factor:.2f}")

    app_providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    dml_fallback_applied = False
    app = init_face_app(providers=app_providers, det_size=runtime_det_size)

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
    processing_errors = 0
    unreadable_skips = 0
    traceback_samples = 0
    smoke_test_done = False

    for idx, file_path in enumerate(pending, 1):
        attempts = 0
        while attempts < 2:
            attempts += 1
            try:
            # Load image with EXIF rotation handling
                img = load_with_exif_rotation(file_path)
                if img is None:
                    unreadable_skips += 1
                    print(f"  [SKIP] Cannot read: {file_path}")
                    break

            # Resize very large images for speed
                h, w = img.shape[:2]
                if max(h, w) > MAX_DIM:
                    scale = MAX_DIM / max(h, w)
                    img = cv2.resize(img, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)

            # Fail early on systemic runtime/provider failures.
                detect_img = img
                if (
                    "DmlExecutionProvider" in app_providers
                    and gpu_safe_upscale_factor > 1.0
                ):
                    # Keep memory bounded while improving small-face detectability.
                    max_side = max(img.shape[:2])
                    max_safe_side = max(MAX_DIM * 2, 2560)
                    safe_factor = min(gpu_safe_upscale_factor, max_safe_side / max_side)
                    if safe_factor > 1.01:
                        detect_img = cv2.resize(
                            img,
                            None,
                            fx=safe_factor,
                            fy=safe_factor,
                            interpolation=cv2.INTER_CUBIC
                        )

                if not smoke_test_done:
                    faces = app.get(detect_img)
                    smoke_test_done = True
                    print(f"  [SMOKE] First image inference OK ({len(faces)} faces)")
                else:
                    # Detect faces + extract embeddings (GPU accelerated)
                    faces = app.get(detect_img)
                face_count = len(faces)

                cursor.execute("SAVEPOINT image_tx")

            # Insert image record with face count
                cursor.execute(
                    "INSERT INTO images (file_path, face_count) VALUES (?, ?)",
                    (file_path, face_count)
                )
                image_id = cursor.lastrowid

            # Insert each face encoding
                for face in faces:
                    bbox = face.bbox  # [x1, y1, x2, y2]
                    face_w = float(bbox[2]) - float(bbox[0])
                    face_h = float(bbox[3]) - float(bbox[1])
                    face_area = face_w * face_h

                    if face_area < MIN_FACE_AREA_PX:
                        continue

                    encoding_bytes = pickle.dumps(face.embedding)
                    bbox_str = str(face.bbox.tolist())
                    det_score = float(face.det_score)
                    cursor.execute(
                        "INSERT INTO faces (image_id, encoding, bbox, det_score) VALUES (?, ?, ?, ?)",
                        (image_id, encoding_bytes, bbox_str, det_score)
                    )
                    total_faces += 1

                cursor.execute("RELEASE SAVEPOINT image_tx")

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
                break

            except Exception as e:
                try:
                    cursor.execute("ROLLBACK TO SAVEPOINT image_tx")
                    cursor.execute("RELEASE SAVEPOINT image_tx")
                except sqlite3.Error:
                    pass

                # Known DirectML issue in some ORT builds for det-size != 640.
                if (
                    not dml_fallback_applied
                    and "DmlExecutionProvider" in app_providers
                    and _is_dml_reshape_runtime_error(e)
                ):
                    print("  [WARN] DirectML RetinaFace runtime failure detected (Reshape_223 / 80070057).")
                    print("  [WARN] Retrying with CPUExecutionProvider so det_size can remain unchanged.")
                    app_providers = ['CPUExecutionProvider']
                    # CPU path can use the requested det_size directly.
                    app = init_face_app(providers=app_providers, det_size=DET_SIZE)
                    dml_fallback_applied = True
                    gpu_safe_upscale_factor = 1.0
                    continue

                processing_errors += 1
                print(f"  [ERROR] {file_path}: {type(e).__name__}: {e}")
                if traceback_samples < 3:
                    tb = traceback.format_exc(limit=6).strip()
                    print(f"    Traceback sample #{traceback_samples + 1}: {tb}")
                    traceback_samples += 1
                break

    # Final commit
    conn.commit()
    conn.close()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Extraction complete!")
    print(f"  Images processed : {len(pending)}")
    print(f"  Faces extracted  : {total_faces}")
    print(f"  Errors           : {processing_errors}")
    print(f"  Skipped unreadable: {unreadable_skips}")
    print(f"  Total time       : {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"  Average speed    : {len(pending)/elapsed:.1f} images/sec")
    if len(pending) > 0 and processing_errors == len(pending):
        print("  [HINT] Every image failed during processing. Check dependency/provider logs above.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    process_images()