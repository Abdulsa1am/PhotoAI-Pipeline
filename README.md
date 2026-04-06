# 📸 PhotoAI — Comprehensive Image Archiving Pipeline

A fully offline, automated photo organization pipeline for Windows that sorts massive photo libraries by **face identity** and **semantic content**.

Built for AMD hardware — leverages the **RX 5700 XT GPU** via DirectML and the **Ryzen 7 7800X3D** multi-core CPU.

---

## 🖥️ Running the GUI

The recommended way to run PhotoAI is through the **graphical interface** (`app.py`). It gives you real-time output, a live system monitor (CPU/GPU/img/s), and one-click pipeline control.

### Option A — With activated virtual environment (recommended)

```powershell
# Step 1: Activate the venv (one-time per terminal session)
cd d:\PhotoAI
.\env\Scripts\activate

# Step 2: Launch the GUI
python app.py
```

### Option B — Direct path to venv Python (no activation needed)

```powershell
d:\PhotoAI\env\Scripts\python.exe d:\PhotoAI\app.py
```

### Option C — Create a launch shortcut

Right-click the Desktop → New → Shortcut, then enter:

```
d:\PhotoAI\env\Scripts\pythonw.exe d:\PhotoAI\app.py
```

> `pythonw.exe` runs without a console window — the GUI is all you see.

### Option D — Double-click launcher script

Create `run.bat` in `d:\PhotoAI\`:

```bat
@echo off
cd /d d:\PhotoAI
.\env\Scripts\pythonw.exe app.py
```

Double-click `run.bat` to launch.

---

## ✨ Features

- **Graphical UI** — Dark-themed dashboard with real-time log, CPU/GPU meters, and progress bars
- **Face Detection & Recognition** — Detects faces using RetinaFace and extracts 512D ArcFace identity embeddings (GPU-accelerated)
- **Face Clustering** — Groups faces by identity using HDBSCAN with age-progression tolerance and automatic centroid merging
- **Semantic Classification** — Classifies non-face images into 17 categories using CLIP zero-shot classification (GPU-accelerated)
- **Archive Builder** — Assembles the final organized directory with hard links (zero disk cost, 100% metadata preservation)
- **EXIF/IPTC/XMP Integrity** — All original metadata is preserved through hard links or `shutil.copy2` fallback
- **Resumable** — All scripts skip already-processed images on re-run

---

## 🧠 AI Model Architecture

PhotoAI uses a multi-stage "Ensemble" of state-of-the-art computer vision models, all running locally on your hardware.

### 1. Face Analysis (InsightFace / ArcFace)
Used in **Step 1** to identify individuals across your entire library.
- **Detector (RetinaFace)**: A high-precision localized detector that finds faces even in low light or at difficult angles.
- **Recognition (ArcFace-R100)**: The "gold standard" for open-source face recognition. It transforms a face into a **92-Dimensional mathematical vector** (embedding). If two faces have similar vectors, they are the same person.
- **Persistence**: These vectors are saved to SQLite, meaning the AI only has to "look" at each photo once.

### 2. Semantic Classification (OpenAI CLIP)
Used in **Step 3** to understand the *meaning* of photos (Memes, Documents, Landscapes, etc.).
- **Architecture**: Vision Transformer (ViT). It "aligns" images with natural language descriptions.
- **Base (ViT-B/32)**:
  - **Size**: ~340 MB
  - **Performance**: ~30 images/sec (RX 5700 XT)
  - **Best for**: Fast sorting of general categories (Animals, Landscapes, Vehicles).
- **Large (ViT-L/14)**:
  - **Size**: ~1.7 GB
  - **Performance**: ~10 images/sec (RX 5700 XT)
  - **Best for**: High-precision detection of nuanced categories like *NSFW*, *Holy Places*, *Sensitive Documents*, and *Art Styles*.

### 3. Identity Discovery (PCA + HDBSCAN)
Used in **Step 2** to automatically "invent" person folders without being told who is who.
- **Dimensionality Reduction (PCA)**: Compresses the 512D face vectors down to 96D. This preserves 95% of facial variance while speeding up the math by **over 20x** for large libraries (30k+ faces).
- **Clustering (HDBSCAN)**: A density-based algorithm that handles "Noise" (people you don't know) elegantly. It groups faces into clusters based on their mathematical proximity in 96D space.
- **Centroid Matching**: Once a person is "named", the AI calculates their mathematical "Average Face" (Centroid). New photos are matched against these centroids instantly using **Cosine Similarity**, skipping the heavy clustering math entirely.

---

## 🗂️ Final Archive Structure

```
D:\Photos\Archive_1\
├── People\
│   ├── Person_000\                  ← Face cluster 0
│   ├── Person_001\                  ← Face cluster 1
│   ├── ...
│   ├── SmallGroups_2-3_faces\       ← Images with 2-3 detected faces
│   ├── LargeGroups_4+_faces\        ← Images with 4+ detected faces
│   └── Unknown_Faces\               ← Unclusterable / noise faces
├── Documents\
├── Sensitive_Documents\
├── Landscapes\
├── Architecture\
├── Vehicles\
├── Animals\
├── Food\
├── Events\
├── Electronics_Gadgets\
├── Office_Workspace\
├── Coding_Screens\
├── Wallpapers\
├── Objects\
├── Art\
├── Memes_Screenshots\
├── Other\                           ← Low confidence on all categories
├── Uncategorized\                   ← Valid images not yet classified
└── Skipped_Unprocessed\             ← Phase 4 Sweep: Videos, RAW files, or corrupted unreadable photos
```

---

## 🔧 Requirements

### Hardware
| Component | Specification |
|---|---|
| CPU | AMD Ryzen 7 7800X3D (8C/16T) |
| RAM | 32 GB |
| GPU | AMD RX 5700 XT (8 GB VRAM) |
| OS | Windows 10/11 |

### Software Dependencies

All packages must be installed **inside the venv** (`d:\PhotoAI\env`):

```powershell
# Activate venv first
.\env\Scripts\activate

# Core — Face extraction (GPU accelerated)
pip install insightface onnxruntime-directml opencv-python numpy Pillow

# Clustering
pip install hdbscan scikit-learn scipy

# Semantic classification + GUI monitoring
pip install "optimum[onnxruntime]" transformers psutil

# Image Compression Support (AVIF)
pip install pillow-avif-plugin

# ONNX format library (required by optimum exporter)
pip install onnx onnxruntime

# One-time model export only
pip install torch
```

> ⚠️ **Important:** `onnxruntime` (CPU) and `onnxruntime-directml` must **both** be installed — `onnxruntime` is needed by the CLIP exporter, `onnxruntime-directml` is needed for GPU inference.

---

## 🚀 First-Time Setup

```powershell
cd d:\PhotoAI
.\env\Scripts\activate

# 1. Export CLIP model to ONNX format (~5 min, downloads ~600MB once)
python setup_clip_model.py

# 2. Launch the GUI and run the full pipeline from there
python app.py
```

In the GUI:
1. Set your **Source Directory**, **Database Path**, and **Output Directory** in the config panel
2. Click **▶▶ Run Full Pipeline** to execute all 5 steps sequentially
3. Watch the live log and system monitor (CPU/GPU/img/s)

---

## 📁 Pipeline Files

| File | Purpose | Accelerator |
|---|---|---|
| `app.py` | **GUI — start here** | — |
| `setup_clip_model.py` | One-time CLIP → ONNX model export | — |
| `1_face_extraction.py` | Face detection + embedding extraction | RX 5700 XT (DirectML) |
| `2_face_clustering.py` | HDBSCAN clustering + centroid merge | Ryzen 7 (all cores) |
| `3_classify_images.py` | CLIP zero-shot semantic classification | RX 5700 XT (DirectML) |
| `4_build_archive.py` | Final archive assembly | SSD I/O |
| `pipeline_config.json` | Config saved by GUI (auto-created) | — |
| `photo_catalog.db` | SQLite database (auto-created) | — |

---

## 🖥️ GUI Reference

| Element | Description |
|---|---|
| **Configuration panel** | Set source dir, DB path, output dir, detection size, cluster parameters |
| **Script cards** (0–4) | Run any individual step; shows ● Ready / ● Running / ✓ Done / ✗ Error |
| **Run Full Pipeline** | Executes all 5 steps sequentially; aborts on first error |
| **CPU bar** (blue) | Live CPU utilization + session average |
| **GPU bar** (green) | Live GPU utilization + session average (via Windows PDH counters) |
| **Progress bar** (yellow) | Images processed / total, parsed from script output |
| **Throughput** | img/sec + ETA, parsed from script output |
| **Output Log** | Real-time colored output from scripts |
| **💾 Save Config** | Writes GUI parameters to disk |
| **Reset Database** | Deletes `photo_catalog.db` to force full re-extraction (Required if you change `Det Size` or `Det Thresh`) |
| **■ Stop** | Terminates the running script immediately |

---

## ⚙️ Configuration

All settings can be changed in the GUI. They are saved to `pipeline_config.json` and applied automatically. 

> 💡 **Dynamic vs Static Settings:**
> - **Dynamic settings:** `Min Cluster`, `Merge Thresh`, and `Confidence` are evaluated dynamically during **Step 4 (Build Archive)**. If you change these, you DO NOT need to reset the database. Just click "Run Full Pipeline" again and the AI will reorganize your archive differently in under 2 seconds.
> - **Static settings:** `Det Size` and `Det Thresh` control the raw data extracted from pixels into the database. If you change these, you **MUST** click "Reset Database" for them to take effect, otherwise the AI will skip your photos thinking they are already processed.

### Script 1 — Face Extraction (Static)
| Parameter | Default | Description |
|---|---|---|
| Source Directory | `D:\POCOP - Copy\+` | Source photo directory to scan |
| Database Path | `D:\PhotoAI\photo_catalog.db` | SQLite database path |
| Det Size | `640` | Detection resolution px. Use `320` for speed, `1024` for accuracy |
| Det Thresh | `0.6` | Strictness of the face detector. Higher = fewer false positives |

### Script 2 & 4 — Face Clustering (Dynamic)
| Parameter | Default | Description |
|---|---|---|
| Min Cluster | `4` | Minimum faces required to form a person folder |
| Merge Thresh | `0.40` | Centroid merge cosine distance to group identical faces (0 = disabled) |

### Script 3 & 4 — Image Classification (Dynamic)
| Parameter | Default | Description |
|---|---|---|
| Confidence | `0.15` | Minimum confidence to assign a category (below → "Other") |
| CLIP Model | `Base` | Toggle between `Base` (Fast/340MB) and `Large` (Accurate/1.7GB) models |

---

## 🧠 Semantic Categories

Categories are defined as plain-text CLIP prompts in `3_classify_images.py`. Each uses 4 ensemble prompts for accuracy.

| # | Category | Examples |
|---|---|---|
| 1 | Documents | Scanned papers, receipts, text-heavy images |
| 2 | Sensitive_Documents | IDs, passports, bank statements, legal papers |
| 3 | Landscapes | Nature, scenery, sunsets, beaches |
| 4 | Architecture | Buildings, interiors, cityscapes |
| 5 | Vehicles | Cars, motorcycles, airplanes, boats |
| 6 | Animals | Pets, wildlife, birds, fish |
| 7 | Food | Meals, cooking, groceries, restaurants |
| 8 | Events | Parties, weddings, concerts, ceremonies |
| 9 | Electronics_Gadgets | Phones, computers, gaming hardware |
| 10 | Office_Workspace | Desks, workstations, offices |
| 11 | Coding_Screens | Code editors, terminals, study materials |
| 12 | Wallpapers | Desktop/phone backgrounds, abstract art |
| 13 | Objects | Products, household items, tools |
| 14 | Art | Drawings, paintings, digital art |
| 15 | Historical_Art | Ancient art, museum pieces, sculptures |
| 16 | Icons_Logos | Brand logos, application icons, UI symbols |
| 17 | Social_Media_Posts | Instagram, Twitter, Snapchat UI, chat bubbles |
| 18 | NSFW | Explicit content (requires Large model for accuracy) |
| 19 | Books | Book covers, pages, library shelves |
| 20 | Holy_Places | Mosques, churches, temples, religious monuments |
| 21 | Memes_Screenshots | Internet jokes, mobile screenshots |
| 22 | Other | Low-confidence catch-all |
| 23 | Uncategorized | Not yet processed |

**Adding custom categories:** Edit the `CATEGORIES` dict in `3_classify_images.py`:
```python
CATEGORIES["Wedding"] = [
    "a wedding ceremony",
    "a bride and groom",
    "a wedding reception",
    "a marriage celebration",
]
```

---

## 💾 Database Schema

```sql
-- All images in the library
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE,
    face_count INTEGER DEFAULT 0
);

-- Detected faces with 512D ArcFace embeddings
CREATE TABLE faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER,
    encoding BLOB,
    bbox TEXT,
    det_score REAL,
    FOREIGN KEY(image_id) REFERENCES images(id)
);

-- Semantic classifications (non-face images)
CREATE TABLE classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER UNIQUE,
    category TEXT,
    confidence REAL,
    top3_categories TEXT,   -- JSON array of top 3 predictions
    has_faces INTEGER DEFAULT 0,
    FOREIGN KEY(image_id) REFERENCES images(id)
);
```

---

## 🔒 Metadata Preservation

| Method | EXIF | IPTC | XMP | Timestamps | Disk Cost |
|---|---|---|---|---|---|
| `os.link()` (hard link) | ✅ 100% | ✅ 100% | ✅ 100% | ✅ Shared | **Zero** |
| `shutil.copy2()` (fallback) | ✅ 100% | ✅ 100% | ✅ 100% | ✅ Preserved | Full copy |

Hard links are the same file on disk. Cross-drive operations fall back to `shutil.copy2()` automatically.

---

## 📝 Notes

- **Resumable:** All scripts skip previously processed images. Safe to interrupt and re-run.
- **Reset DB:** Use the **Reset Database** button in the GUI (bottom-right) if you need to force full re-extraction (e.g. switching from old dlib encodings to ArcFace).
- **Same-drive:** Hard links only work within the same NTFS partition. The `"auto"` file mode handles this transparently.
- **GPU fallback:** If DirectML is unavailable, all processing continues on CPU — just slower.
