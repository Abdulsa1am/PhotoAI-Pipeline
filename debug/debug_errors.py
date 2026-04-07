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
import traceback

# ============ CONFIGURATION ============
MASTER_DIR = r"D:\POCOP - Copy"
DB_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photo_catalog_test.db")
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
DET_SIZE = (640, 640)
DET_THRESH = 0.5
MAX_DIM = 1920

def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY AUTOINCREMENT, file_path TEXT UNIQUE, face_count INTEGER DEFAULT 0)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, image_id INTEGER, encoding BLOB, bbox TEXT, det_score REAL)''')
    conn.commit()
    return conn

def init_face_app():
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=DET_SIZE, det_thresh=DET_THRESH)
    return app

def load_with_exif_rotation(file_path):
    try:
        pil_img = Image.open(file_path)
        pil_img = pil_img.convert('RGB')
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        # Fallback to OpenCV
        return cv2.imread(file_path)

def collect_image_paths(master_dir):
    paths = []
    for root, _, files in os.walk(master_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return paths

def debug_process():
    conn = setup_database()
    cursor = conn.cursor()
    app = init_face_app()
    all_paths = collect_image_paths(MASTER_DIR)
    
    print(f"Total paths: {len(all_paths)}")
    
    for idx, file_path in enumerate(all_paths):
        try:
            img = load_with_exif_rotation(file_path)
            if img is None:
                # If we are expecting this to be an error in the summary, 
                # but our code says SKIP, then maybe the user's code has something 
                # that throws here.
                # Let's check if cv2.imread(file_path) can throw.
                continue
            
            # This is where 30,152 errors might be happening
            faces = app.get(img)
            # ...
        except Exception as e:
            print(f"FATAL ERROR on file: {file_path}")
            print(f"Error: {e}")
            traceback.print_exc()
            return # Exit after first error

if __name__ == "__main__":
    debug_process()
