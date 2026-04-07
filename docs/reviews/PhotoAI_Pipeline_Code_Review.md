# PhotoAI Pipeline — Complete Post-Session Code Review

**Repository:** [Abdulsa1am/PhotoAI-Pipeline](https://github.com/Abdulsa1am/PhotoAI-Pipeline)
**Reviewed files:** `app.py`, `1_face_extraction.py`, `2_face_clustering.py`, `3_classify_images.py`, `4_build_archive.py`, `5_compress_images.py`
**Language:** Python 3 / Tkinter GUI
**Review Date:** April 7, 2026

---

## Executive Summary

The PhotoAI Pipeline is a well-structured, feature-rich desktop application for face extraction, clustering, classification, and image archiving. The overall architecture is sound — each script has a clean single responsibility, configuration flows correctly from GUI → `pipeline_config.json` → each script, and the tracing/logging system is genuinely impressive for a personal project. However, there are several bugs and risks that need addressing before this is production-grade, ranging from a **critical database race condition** and **hardcoded credential paths committed in source**, to important UX gaps and several anti-patterns that will cause maintenance pain.

---

## 1. Bug Detection

### 1.1 🔴 Database Connection Leaked on Early Return (Script 2)

**File:** `2_face_clustering.py`, lines ~60–72

When `unassigned_rows` is empty, the function prints a message and calls `conn.close()` and returns — correct. But when `leftover_idx` has fewer items than `MIN_CLUSTER_SIZE`, the function marks the faces as Unknown and calls `conn.close()` inside the nested `if leftover_idx:` block. If HDBSCAN raises an exception after the commit but before `conn.close()` at the bottom, the outer connection is silently leaked.

```python
# PROBLEMATIC — two separate conn.close() paths
if len(leftover_idx) < MIN_CLUSTER_SIZE:
    ...
    conn.commit()
    conn.close()   # ← early return, outer close never reached
    return

# ... HDBSCAN code ...
conn.close()  # ← outer close can be skipped
```

**Fix:** Use a `try/finally` block:
```python
conn = sqlite3.connect(DB_PATH)
try:
    ...
    main logic
finally:
    conn.close()
```

---

### 1.2 🔴 `_on_script_done` Checks `self.is_pipeline_run` After Setting It to False

**File:** `app.py`, `_on_script_done` method (~line 620)

When a pipeline step **fails**, the code does:
```python
self.is_pipeline_run = False
# ... then later:
if not self.is_pipeline_run:
    self._close_run_trace(returncode=returncode)
```

The `_close_run_trace(returncode)` call triggers `_archive_last_run_trace()`, which archives the trace correctly. But in the **success** path:
```python
if self.is_pipeline_run:
    self.root.after(500, self._run_next_pipeline_step)
    return   # ← early return means the trace is NOT closed here
```

The trace is only closed in `_run_next_pipeline_step` when `pipeline_queue` is empty. If the process is killed between steps, `_close_run_trace` is never called with a `returncode`, so the trace file won't be archived. The `on_close` handler calls `_close_run_trace()` without a returncode — this means `should_archive` will be `False` and history will be silently lost.

**Fix:**
```python
def on_close(self):
    ...
    self._close_run_trace(returncode=-1)  # force archive even on forced close
```

---

### 1.3 🔴 `NameFacesUI` Binds `<MouseWheel>` Globally and Never Unbinds

**File:** `app.py`, `NameFacesUI._build_ui` (~line 820)

```python
canvas.bind_all("<MouseWheel>", _on_mousewheel)
```

`bind_all` registers the binding on the **root window** — it captures mouse wheel events application-wide. When the `NameFacesUI` window is closed, this binding remains active on the root. If the user opens and closes the Name Faces UI multiple times, each instance stacks a new global binding. This causes multiple simultaneous scroll callbacks and can scroll invisible canvases from a destroyed widget.

**Fix:**
```python
# In NameFacesUI.__init__:
self.protocol("WM_DELETE_WINDOW", self._on_close)

def _on_close(self):
    self.unbind_all("<MouseWheel>")
    self.destroy()
```

---

### 1.4 🟡 Silent Config Load Failure Hides Real Errors

**File:** `app.py`, `_load_config` method (~line 490)

```python
except Exception:
    pass  # ← silently swallows JSON decode errors, permission errors, etc.
```

If `pipeline_config.json` becomes malformed (e.g., truncated mid-write during a crash), the GUI silently loads all defaults and the user has no idea their saved config was lost.

**Fix:**
```python
except json.JSONDecodeError as e:
    messagebox.showwarning(
        "Config Load Warning",
        f"Config file is corrupted and has been reset to defaults.\n\nDetails: {e}"
    )
except Exception as e:
    messagebox.showwarning("Config Load Warning", f"Could not load config: {e}")
```

---

### 1.5 🟡 `_parse_log_line` Regex Requires Leading Space — Misses Some Patterns

**File:** `app.py`, `_parse_log_line` (~line 510)

```python
m = re.search(r'\[\s+(\d+)\s*/\s*(\d+)\s*\]', line)
```

The comment says "Require at least one space before the number to avoid matching setup steps like [1/2]". But `\s+` requires **one or more** spaces inside the brackets. The actual script output format is:
```
  [   500/20000] faces: ...
```
This works for Step 1, but Step 5 (`compress_images.py`) prints:
```
  [    25/10000] compressed: ...
```
...which also has leading spaces. The problem is when scripts print something like `[100/500]` with no padding spaces — this is silently ignored, leaving the progress bar at 0%. Worse, the `compress` run key is set but `self.status_labels["compress"]` doesn't exist (compress is not in `PIPELINE_ORDER` and has no card), so `_set_status` for "compress" silently does nothing.

**Fix:** Make the regex accept optional whitespace: `r'\[\s*(\d+)\s*/\s*(\d+)\s*\]'` and add size guard (e.g., require total > 10) to avoid false positives.

---

### 1.6 🟡 `place_file` Returns `True` for Existing Files Without Validation

**File:** `4_build_archive.py`, `place_file` function (~line 60)

```python
if os.path.exists(dst):
    success = True  # Already exists
```

If a previous run placed a **different file** at `dst` (e.g., two images with the same filename in different source subdirectories, both mapped to the same person folder), the second file silently "succeeds" without overwriting. The `placed_source_paths` set still gets the new `src` added, but the **file at `dst` is the wrong image**. With the `unique_fname = f"{img_id}_{fname}"` prefix this is very rare, but possible if `img_id` somehow collides (if the DB was reset and IDs restarted while the archive directory was not deleted).

**Fix:** Add a warning log when skipping due to existing file:
```python
if os.path.exists(dst):
    # Verify it's the same source file before silently treating as success
    if os.path.getsize(dst) != os.path.getsize(src):
        print(f"  [WARN] Destination exists but sizes differ: {dst}")
    return True
```

---

### 1.7 🟡 CLIP Confidence Threshold Default Mismatch

**File:** `3_classify_images.py` line 15 vs `app.py` config default

- Script 3 default: `CONFIDENCE_THRESHOLD = 0.18`
- GUI default shown to user: `"0.15"` (in `_build_config`)
- `pipeline_config.json` sample value: needs checking

The user sees 0.15 in the GUI, saves config, and the script runs at 0.15 — but without the GUI, running the script standalone uses 0.18. This is a minor discrepancy but a source of reproducibility bugs in headless use.

**Fix:** Align both defaults to the same value (0.15 is a better UX-visible default) and document it.

---

### 1.8 🟢 `face_count` Column Is Stale — Script 4 Correctly Works Around It But Script 3 Doesn't

**File:** `3_classify_images.py` (~line 195)

```sql
WHERE classifications.id IS NULL AND images.face_count = 0
```

Script 4 explicitly notes that `face_count` column can be stale and uses a `COUNT(faces.id)` join instead. Script 3 still trusts the `face_count` column. If a database was migrated from an older schema where `face_count` wasn't updated correctly, Script 3 may classify images that actually have faces (because `face_count` shows 0 when it shouldn't).

**Fix:** Mirror Script 4's approach:
```sql
WHERE classifications.id IS NULL
AND (SELECT COUNT(*) FROM faces WHERE faces.image_id = images.id) = 0
```

---

## 2. Logic Analysis

### 2.1 Pipeline Abort on Error Doesn't Reset ALL Cards

**File:** `app.py`, `_on_script_done` (~line 618)

When a step fails mid-pipeline:
```python
if self.pipeline_queue:
    self.pipeline_queue.clear()
    self._set_status("all", "error")
```

The cards for **future steps** (those still in `pipeline_queue`) remain in `◌ Pending` state — they never get reset to `● Ready`. The user sees "◌ Pending" on steps 3, 4 even after the error, which implies those steps ran but haven't finished, rather than showing they were skipped.

**Fix:**
```python
if self.pipeline_queue:
    for remaining_key in self.pipeline_queue:
        self._set_status(remaining_key, "ready")  # reset pending cards
    self.pipeline_queue.clear()
    self._set_status("all", "error")
```

---

### 2.2 `run_all` Skips Validation for Steps Other Than `extract`

**File:** `app.py`, `run_all` method

`run_all` only calls `self._validate_extract_config()` — but steps 3 and 4 also require `db_path` to exist and the `output_dir` parent to be valid. If the user hasn't run Step 1, the DB won't exist, but `run_all` will happily launch Step 1 which will try to create the DB in the configured path. If the configured DB directory doesn't exist, Step 1 will fail at runtime rather than being caught cleanly in the GUI.

The `_validate_extract_config` already checks that `db_path`'s parent directory exists — this is fine. But `output_dir` is never validated before Step 4 runs, meaning Step 4 can silently fail or create unexpected directories.

---

### 2.3 `_run_next_pipeline_step` Starts Timer But Doesn't Stop Previous One

**File:** `app.py`, `_run_next_pipeline_step`

Each step calls `self._update_timer()`, which polls every 1 second. When a step finishes, `_update_timer` checks `if not self.running: return`. However, between step N finishing and step N+1 starting (the 500ms `after` delay), `self.running` is set to `True` in `_run_next_pipeline_step` — so the old timer keeps ticking AND a new timer starts. The timer display updates twice per second mid-pipeline.

**Fix:** Cancel the timer before starting a new one:
```python
if hasattr(self, '_timer_after_id') and self._timer_after_id:
    self.root.after_cancel(self._timer_after_id)
```

---

### 2.4 `classify_with_ensemble` Normalizes CLIP Scores Twice

**File:** `3_classify_images.py`, `classify_with_ensemble` function (~line 185)

```python
probs = logits.softmax(dim=0).detach().numpy()
# ... aggregate by category ...
if not is_siglip:
    total = sum(s for _, s in results)
    if total > 0:
        results = [(name, score / total) for name, score in results]
```

`softmax` already produces values that sum to 1.0. After averaging them per category, the aggregated values sum to slightly less than 1.0 (due to multi-prompt averaging). The second normalization step is therefore redundant but harmless. However, it creates a false impression that the scores are "re-calibrated" when they are not — they're just re-scaled within the same distribution. This isn't a bug but is misleading code.

---

### 2.5 Script 4's `_cfg` Globals Check Is Fragile

**File:** `4_build_archive.py`, `main()` (~line 95)

```python
merge_t = _cfg.get("merge_threshold", 0.40) if '_cfg' in globals() else 0.40
conf_t  = _cfg.get("confidence_threshold", 0.15) if '_cfg' in globals() else 0.15
```

Since `_cfg` is defined at module-level in the config-loading block at the top, it will always be in `globals()` if `pipeline_config.json` exists. But if the file doesn't exist, `_cfg` is never defined, so `'_cfg' in globals()` is `False` and the hardcoded defaults are used — this is actually correct behavior. However, the pattern is fragile: if the config file exists but has a read error, `_cfg` might be partially set. The cleaner solution is to define `_cfg = {}` as the default before the `if os.path.exists` block (as done in Scripts 2 and 3).

---

## 3. UX & General Experience

### 3.1 🔴 No "Stop" Button Reset After Pipeline Abort

When a pipeline run is aborted (either via stop button or script error), the "■ Stop" button on the **active step card** is converted back to "▶ Run". But the "▶▶ Start" button on the "Run Full Pipeline" card is reset via `_set_buttons_enabled(True)` → `_set_status("all", ...)`. If an error occurs, `_set_status("all", "error")` is called which sets the "all" button back to "▶▶ Start" — correct. However, if the user **manually stops** the script, `stop_script()` calls `self.pipeline_queue.clear()` but does NOT call `_set_status("all", "ready")`. The "all" card stays in "● Running" state with "■ Stop" visible even after the process was terminated.

**Fix:** Add `self._set_status("all", "ready")` inside `stop_script()`.

---

### 3.2 🟡 Compress Tab Has No Input Folder Validation

Before running compression, there's no check that `compress_input` is a valid directory. If the user types a nonexistent path and clicks Start, the script runs, prints `Total images found: 0`, and exits with code 0. The GUI shows "✓ Compression finished!" — misleading the user into thinking everything worked.

**Fix:** Add validation similar to `_validate_extract_config`:
```python
def _validate_compress_config(self):
    input_dir = self.config_vars.get("compress_input", tk.StringVar()).get().strip()
    if not input_dir or not os.path.isdir(input_dir):
        messagebox.showerror("Invalid Input Folder", 
            "Input folder does not exist. Please select a valid directory.")
        return False
    return True
```

---

### 3.3 🟡 Auto-Open Name Faces After Archive Runs Silently If DB Is Missing

**File:** `app.py`, `_on_script_done`

```python
if key == "archive" and not self.is_pipeline_run:
    self.root.after(1000, self._open_name_faces_ui)
```

`_open_name_faces_ui` shows an error dialog if the DB doesn't exist, but this dialog appears 1 second after the archive finishes with no warning that it's coming. Users who run Step 4 standalone for the first time (before having a valid DB) will be confused by an error popup appearing unprompted after the success message.

**Fix:** Add a condition to check DB existence before scheduling the auto-open, or show a gentler info dialog.

---

### 3.4 🟡 Log Panel Has No Maximum Size / Memory Bound

**File:** `app.py`, log `tk.Text` widget

The `log` widget accumulates all text indefinitely. For very large photo libraries (50,000+ images), a full pipeline run will print ~10,000+ progress lines, causing significant memory usage and potential UI slowdown as the text widget renders tens of thousands of lines.

**Fix:** Add a line-trimming mechanism in `_append_log`:
```python
MAX_LOG_LINES = 5000
line_count = int(self.log.index("end-1c").split(".")[0])
if line_count > MAX_LOG_LINES:
    self.log.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
```

---

### 3.5 🟢 No Keyboard Shortcut for Common Actions

The interface has no keyboard shortcuts for high-frequency actions. Power users running the pipeline repeatedly have to click through the UI every time.

**Suggestions:**
- `Ctrl+R` → Run Full Pipeline
- `Ctrl+S` → Save Config
- `Escape` → Stop running script
- `Ctrl+L` → Clear log

---

### 3.6 🟢 NameFacesUI Has No "Skip" vs "Clear Name" Distinction

In `_save_names`, empty entries are simply skipped:
```python
if not new_name:
    continue
```

This means if a user clears a name they previously assigned, that change is silently discarded — the old name stays in the DB. There's no way to "un-name" a person via the UI without going directly to the database.

**Fix:** Treat an explicitly cleared name differently:
```python
if new_name == "":
    cursor.execute("UPDATE people SET custom_name = NULL WHERE id = ?", (p_id,))
```

---

## 4. Code Quality

### 4.1 🟡 Hardcoded Personal Paths Committed to Public Repository

**Files:** All 5 pipeline scripts

```python
MASTER_DIR = r"D:\POCOP - Copy\+"
DB_PATH    = r"D:\PhotoAI\photo_catalog.db"
BASE_PHOTOS_DIR = r"D:\Photos"
```

These paths are hardcoded as defaults across all scripts. Since the repository is **public**, these reveal the user's personal drive layout. While the GUI overrides them via `pipeline_config.json`, any contributor or person running the scripts standalone without a config file will hit these paths. The `pipeline_config.json` should be the single source of truth and these inline defaults should be generic placeholders.

**Fix:**
```python
MASTER_DIR = r"C:\Photos\Source"
DB_PATH    = os.path.join(os.path.dirname(__file__), "photo_catalog.db")
BASE_PHOTOS_DIR = r"C:\Photos\Archive"
```

---

### 4.2 🟡 `photo_catalog.db` Is Committed to the Repository

**File:** `photo_catalog.db` (471 KB in the repository root)

A live SQLite database is committed directly to the repository. This database likely contains extracted face embeddings, file paths from the user's personal machine, and potentially face clustering data. Face embeddings are biometric data — committing them to a public repo is a **privacy and GDPR concern**.

**Fix:**
1. Add `*.db` to `.gitignore` immediately
2. Remove the file from Git history using `git filter-repo` or BFG Repo Cleaner

---

### 4.3 🟡 `SystemMonitor._loop` Unbounded History Lists

**File:** `app.py`, `SystemMonitor` class

```python
self._cpu_hist.append(c)
self.cpu_avg = sum(self._cpu_hist) / len(self._cpu_hist)
```

The CPU and GPU history lists grow without bound. For a long-running pipeline (e.g., 8 hours, 1.5s polling interval = ~19,200 samples), these lists will consume unnecessary memory, and the `sum() / len()` average recalculation becomes linearly slower over time.

**Fix:** Use a fixed-size deque for a rolling average:
```python
from collections import deque
self._cpu_hist = deque(maxlen=300)  # Last 7.5 minutes
```

---

### 4.4 🟡 `_set_buttons_enabled` Has Asymmetric Enable Logic

**File:** `app.py`, `_set_buttons_enabled` method

```python
def _set_buttons_enabled(self, enabled):
    state = "normal" if enabled else "disabled"
    for key, btn in self.run_buttons.items():
        if enabled or key == self.current_key:
            btn.configure(state="normal")
        else:
            btn.configure(state="disabled")
```

When `enabled=False`, the button for `self.current_key` is left enabled (so the Stop button works). But when `enabled=True` (re-enabling after run), the `compress_btn` (which is NOT in `self.run_buttons`) is never re-enabled here — it has its own reset logic in `_on_script_done`. This split responsibility is confusing and a future maintainer could easily break one path while fixing the other.

---

### 4.5 🟢 Scripts 1–5 Repeat the Same Config Loading Pattern

Every script contains this identical boilerplate:
```python
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path, 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    DB_PATH = _cfg.get("db_path", DB_PATH)
    ...
```

This is a DRY violation. A shared `config_loader.py` utility would eliminate 30+ lines of duplication and mean that changes to the config loading logic (e.g., adding error handling) only need to happen once.

**Fix:** Create `config_loader.py`:
```python
def load_pipeline_config():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[WARN] pipeline_config.json is corrupted, using script defaults.")
    return {}
```

---

### 4.6 🟢 `NameFacesUI` Loads All Person Images into Memory at Once

**File:** `app.py`, `_load_folders`

All face thumbnails for all persons are loaded and kept in `self.images` simultaneously. For a database with 200 recognized people × 3 samples × 150×150px = 90,000 thumbnail images, this is roughly 40–80 MB of Python objects in RAM — and none are released until the window is closed.

**Fix:** Use lazy loading or a virtual scroll approach — only load thumbnails for rows visible in the canvas viewport.

---

## 5. Performance

### 5.1 🔴 `_poll_output` Drains Queue as Fast as Possible But Calls `after(50ms)`

**File:** `app.py`, `_poll_output`

The output queue is drained in a tight `while True` loop with `get_nowait()`. During active script execution (especially Step 1 with thousands of images), the script output pipe can generate hundreds of lines per second. All of these are processed in a single Tkinter event loop tick, freezing the UI for potentially hundreds of milliseconds.

**Fix:** Drain a maximum number of lines per tick:
```python
MAX_LINES_PER_TICK = 50
try:
    for _ in range(MAX_LINES_PER_TICK):
        msg_type, data = self.output_queue.get_nowait()
        ...
except queue.Empty:
    pass
```

---

### 5.2 🟡 Script 2 Fetches ALL Unassigned Encodings Into RAM

**File:** `2_face_clustering.py`, ~line 55

```python
cursor.execute("SELECT id, encoding FROM faces WHERE person_id IS NULL OR person_id = -1")
unassigned_rows = cursor.fetchall()
unassigned_encodings = np.array([pickle.loads(r[1]) for r in unassigned_rows])
```

For a library of 100,000 photos with average 2 faces each = 200,000 face embeddings × 512 floats × 4 bytes = **~390 MB** loaded into RAM at once. The `fetchall()` additionally holds all the raw pickle BLOBs in memory simultaneously before converting them to numpy. This could be particularly problematic on memory-constrained systems.

**Fix:** Use chunked loading or process in batches:
```python
cursor.execute("SELECT id, encoding FROM faces WHERE ... LIMIT ? OFFSET ?", (chunk_size, offset))
```

---

### 5.3 🟡 `cdist` Computes Full Pairwise Distance Matrix

**File:** `2_face_clustering.py`, Phase 1 KNN

```python
distances = cdist(unassigned_norm, people_centroids_norm, metric='cosine')
```

With 200,000 unassigned faces × 500 known people = 100 million distance computations. For very large libraries, this produces a 200,000 × 500 float64 matrix = **~800 MB**. The `np.argmin` on this matrix is then O(N×M).

**Fix:** For large datasets, use `sklearn.neighbors.NearestNeighbors` with `algorithm='ball_tree'` or `'brute'` (metric='cosine') which avoids materializing the full matrix.

---

### 5.4 🟡 Script 3 Opens Every Image Fresh for Each Classification (No Caching)

**File:** `3_classify_images.py`, main loop

Each iteration calls `Image.open(file_path).convert("RGB")` and then passes the full resolution image to CLIP. Pillow does decode-time downsampling when the processor resizes for CLIP (224×224), but the full file is still read from disk. For JPEG images, using draft mode would be faster:
```python
img = Image.open(file_path)
img.draft('RGB', (224, 224))  # tells JPEG decoder to decode at reduced resolution
img = img.convert('RGB')
```

---

### 5.5 🟢 GPU Monitor Spawns a `powershell` Process Every 1.5 Seconds

**File:** `app.py`, `SystemMonitor._loop`

```python
result = subprocess.run(["powershell", "-NoProfile", "-Command", _GPU_PS], ...)
```

Each poll spawns a new PowerShell process. PowerShell startup overhead is ~200–400ms on a cold process, meaning this loop can take 400ms+ per iteration even with `timeout=3`. On a low-powered system, this single background thread is consuming measurable CPU for the entire pipeline duration.

**Fix:** Cache the PDH counter in a persistent PowerShell session, or use the `wmi` Python library for direct counter access:
```python
import wmi
c = wmi.WMI(namespace="root\\cimv2")
```

---

## 6. Security

### 6.1 🔴 `photo_catalog.db` Committed to Public Repo Contains Biometric Data

As noted in 4.2, the committed SQLite database contains pickled ArcFace embeddings, which are biometric identifiers derived from real face images. Committing these to a public repository:
- May violate GDPR/privacy laws depending on jurisdiction
- Exposes the embedding space of real people's faces to public extraction
- The file paths within the DB also reveal the user's personal directory structure

**Immediate action required:** Remove the file from the repository and its Git history.

---

### 6.2 🔴 `pickle.loads()` on Database-Stored Blobs Is an Arbitrary Code Execution Risk

**Files:** `1_face_extraction.py`, `2_face_clustering.py`, `4_build_archive.py`

```python
people_centroids.append(pickle.loads(row[1]))
unassigned_encodings = np.array([pickle.loads(r[1]) for r in unassigned_rows])
```

`pickle.loads()` on untrusted data can execute arbitrary Python code. While the DB is locally created and owned by the user, if the `photo_catalog.db` file is replaced by a malicious actor (e.g., via a shared drive), it could execute arbitrary code on the machine. Given the DB path is now public knowledge from the committed database, this is a realistic social engineering vector.

**Fix:** Replace `pickle` for numpy array serialization with `numpy`'s own binary format:
```python
# Store
buf = io.BytesIO()
np.save(buf, embedding)
encoding_bytes = buf.getvalue()

# Load
buf = io.BytesIO(encoding_bytes)
embedding = np.load(buf, allow_pickle=False)
```

---

### 6.3 🟡 `safe_name` Sanitization Is Overly Permissive

**File:** `app.py`, `NameFacesUI._save_names`

```python
safe_name = "".join(c for c in new_name if c.isalnum() or c in " _-")
```

This sanitization is used for `custom_name` stored in the DB and later used as **directory names** in Script 4. The character set allows spaces, which are legal in directory names but problematic in shell scripts, paths with `os.path.join`, and log output. More importantly, there's no length limit — a 10,000-character name would create a path that exceeds Windows MAX_PATH (260 chars).

**Fix:**
```python
safe_name = "".join(c for c in new_name if c.isalnum() or c in "_-")[:64]
```

---

### 6.4 🟡 SQL Queries in Script 4 Use `WHERE person_id != ""`  (String Comparison on Integer Column)

**File:** `4_build_archive.py`, `load_cluster_labels`

```sql
WHERE faces.person_id IS NOT NULL AND faces.person_id != ""
```

`person_id` is an `INTEGER` column, but the comparison `!= ""` is a string comparison. SQLite will silently coerce this (comparing integer to empty string always evaluates to True for any non-null integer), so it works accidentally — but it's semantically wrong and confusing. The intent is clearly `!= 0` or `> 0` (since -1 = Unknown).

**Fix:**
```sql
WHERE faces.person_id IS NOT NULL AND faces.person_id > 0
```

---

### 6.5 🟢 No Input Validation on `det_size` Could Cause Resource Exhaustion

**File:** `app.py`, `_validate_extract_config`

`det_size` is validated to be between 160 and 2048. At `det_size=2048` with the GPU-safe upscaling factor, the script could resize images up to `2048/640 * MAX_DIM = ~6,144px` before passing to the detector. For a library of 10MP+ RAW-converted images, this could exhaust VRAM/RAM. The check is good but the upper bound of 2048 might need a warning at values above 1280.

---

## 7. Complete Improvement Plan

### 🔴 Critical — Fix Immediately

| # | Issue | File | Root Cause |
|---|-------|------|------------|
| C1 | Remove `photo_catalog.db` from repo; add `*.db` to `.gitignore` | repo root | Biometric data in public repo |
| C2 | Replace `pickle.loads()` with `numpy` binary format | Scripts 1, 2, 4 | Arbitrary code execution risk |
| C3 | Wrap all `sqlite3.connect()` in `try/finally conn.close()` | Script 2 | Connection leak on error |
| C4 | Fix `on_close` to call `_close_run_trace(returncode=-1)` | `app.py` | Trace archive silently lost on force-quit |
| C5 | Unbind `<MouseWheel>` when `NameFacesUI` is destroyed | `app.py` | Stacking global event bindings, potential crash |

### 🟡 Important — Fix Before Release / Sharing

| # | Issue | File |
|---|-------|------|
| I1 | Replace hardcoded personal paths with generic defaults | All scripts |
| I2 | Add `_validate_compress_config()` before running Script 5 | `app.py` |
| I3 | Reset pending cards to "Ready" on pipeline abort | `app.py` |
| I4 | Fix `stop_script()` to reset the "all" card status | `app.py` |
| I5 | Fix SQL `person_id != ""` to `person_id > 0` | Script 4 |
| I6 | Align confidence threshold defaults (0.15 everywhere) | Script 3, app.py |
| I7 | Replace `sum/len` average in `SystemMonitor` with `deque(maxlen=300)` | `app.py` |
| I8 | Cap drain per `_poll_output` tick at 50 lines to prevent UI freezes | `app.py` |
| I9 | Replace `cdist` full matrix with `NearestNeighbors` for large datasets | Script 2 |
| I10 | Add error-reporting to `_load_config` (don't silently swallow exceptions) | `app.py` |
| I11 | Limit `safe_name` length to 64 chars; remove spaces from allowed chars | `app.py` |
| I12 | Add `config_loader.py` to eliminate DRY violation across all 5 scripts | All scripts |

### 🟢 Nice to Have — Future Improvements

| # | Issue | File |
|---|-------|------|
| N1 | Add keyboard shortcuts (`Ctrl+R`, `Ctrl+S`, `Esc`, `Ctrl+L`) | `app.py` |
| N2 | Implement lazy loading for `NameFacesUI` thumbnails | `app.py` |
| N3 | Replace PowerShell GPU polling with `wmi` library for efficiency | `app.py` |
| N4 | Add duplicate timer fix in `_run_next_pipeline_step` | `app.py` |
| N5 | Allow "un-naming" a person in `NameFacesUI` (clear → `NULL` in DB) | `app.py` |
| N6 | Add `JPEG draft mode` in Script 3 for faster image loading | Script 3 |
| N7 | Use subquery for `face_count` in Script 3 (mirror Script 4's approach) | Script 3 |
| N8 | Add max log line cap to prevent memory growth on large runs | `app.py` |
| N9 | Chunked DB embedding loading in Script 2 for 100k+ face libraries | Script 2 |
| N10 | Remove double CLIP softmax normalization comment/clean-up | Script 3 |

---

*Reviewed by: AI Senior Code Review | PhotoAI-Pipeline @ commit `512160a`*
