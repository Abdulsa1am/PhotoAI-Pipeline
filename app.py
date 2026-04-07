"""
PhotoAI Pipeline — GUI Application
Dark-themed Tkinter interface for controlling the archiving pipeline.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import queue
import json
import os
import sys
import time
import re
try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

import glob
import random
import sqlite3
from collections import deque
try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

# DPI awareness on Windows
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "pipeline_config.json")
TRACE_DIR = os.path.join(SCRIPT_DIR, "logs")
TRACE_HISTORY_DIR = os.path.join(TRACE_DIR, "history")
LAST_RUN_TRACE_PATH = os.path.join(TRACE_DIR, "last_run.log")
TRACE_HISTORY_MAX_RUNS = 10

# ---- Theme Colors (GitHub Dark) ----
BG          = "#0d1117"
CARD_BG     = "#161b22"
BORDER      = "#30363d"
ACCENT      = "#58a6ff"
TEXT        = "#c9d1d9"
TEXT_DIM    = "#8b949e"
GREEN       = "#238636"
GREEN_LIT   = "#3fb950"
RED         = "#da3633"
RED_LIT     = "#f85149"
YELLOW      = "#d29922"
INPUT_BG    = "#0d1117"
LOG_BG      = "#010409"

FONT        = ("Segoe UI", 10)
FONT_SM     = ("Segoe UI", 9)
FONT_BOLD   = ("Segoe UI", 10, "bold")
FONT_HEAD   = ("Segoe UI", 16, "bold")
FONT_NUM    = ("Segoe UI", 20, "bold")
FONT_LOG    = ("Cascadia Code", 9)

SCRIPTS = {
    "setup":    ("0", "Setup CLIP",       "Export CLIP model to ONNX (one-time)",     os.path.join("scripts", "setup_clip_model.py")),
    "extract":  ("1", "Face Extraction",  "Detect faces & extract ArcFace embeddings", os.path.join("scripts", "1_face_extraction.py")),
    "cluster":  ("2", "Face Clustering",  "HDBSCAN clustering + centroid merge",      os.path.join("scripts", "2_face_clustering.py")),
    "classify": ("3", "Classify Images",  "CLIP zero-shot semantic classification",   os.path.join("scripts", "3_classify_images.py")),
    "archive":  ("4", "Build Archive",    "Assemble final organized directory",       os.path.join("scripts", "4_build_archive.py")),
    "compress": ("5", "Compress Images",  "GPU-accelerated image compression",        os.path.join("scripts", "5_compress_images.py")),
}
PIPELINE_ORDER = ["setup", "extract", "cluster", "classify", "archive"]


# ---- GPU query via PowerShell PDH counters (Robust against AMD name variations) ----
_GPU_PS = (
    r"try { "
    r"$v=0; (Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction Stop).CounterSamples | "
    r"Where-Object { $_.Path -match 'engtype_3D|engtype_Graphics' } | "
    r"ForEach-Object { $v += $_.CookedValue }; "
    r"[math]::Min([math]::Round($v),100) "
    r"} catch { 'N/A' }"
)


class ToolTip:
    """Creates a tooltip that displays text when hovering over a widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.id = None
        self.widget.bind("<Enter>", self.schedule_show)
        self.widget.bind("<Leave>", self.schedule_hide)
        self.widget.bind("<ButtonPress>", self.hide_tooltip)

    def schedule_show(self, event=None):
        self.unschedule()
        # Wait 250ms before showing to prevent flashing when casually moving mouse
        self.id = self.widget.after(250, self.show_tooltip)

    def schedule_hide(self, event=None):
        self.unschedule()
        # 100ms tolerance before hiding allows hopping between label/field without flickering
        self.id = self.widget.after(100, self.hide_tooltip)

    def unschedule(self):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        x_offset, y_offset = 20, 25
        x = self.widget.winfo_rootx() + x_offset
        y = self.widget.winfo_rooty() + y_offset

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        self.tooltip_window.attributes('-topmost', True)

        label = tk.Label(self.tooltip_window, text=self.text, justify='left',
                         background=INPUT_BG, foreground=TEXT, relief='solid', borderwidth=1,
                         font=("Segoe UI", 9), padx=10, pady=8, wraplength=450)
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class SystemMonitor:
    """Background thread that samples CPU% and GPU% every 1.5 seconds."""

    def __init__(self, interval=1.5):
        self.interval   = interval
        self.cpu_cur    = 0.0
        self.cpu_avg    = 0.0
        self.gpu_cur    = -1.0
        self.gpu_avg    = -1.0
        self._cpu_hist  = deque(maxlen=300)
        self._gpu_hist  = deque(maxlen=300)
        self._running   = False
        self._thread    = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def reset_averages(self):
        self._cpu_hist.clear()
        self._gpu_hist.clear()
        self.cpu_avg = 0.0
        self.gpu_avg = -1.0

    def _loop(self):
        if PSUTIL_OK:
            psutil.cpu_percent(interval=None)   # prime the pump
        while self._running:
            # ---- CPU ----
            if PSUTIL_OK:
                c = psutil.cpu_percent(interval=None)
            else:
                c = 0.0
            self.cpu_cur = c
            self._cpu_hist.append(c)
            self.cpu_avg = sum(self._cpu_hist) / len(self._cpu_hist)

            # ---- GPU via PowerShell PDH ----
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", _GPU_PS],
                    capture_output=True, text=True, timeout=3
                )
                raw = result.stdout.strip()
                g = float(raw) if raw not in ('', 'N/A') else -1.0
            except Exception:
                g = -1.0
            if g >= 0:
                self.gpu_cur = g
                self._gpu_hist.append(g)
                self.gpu_avg = sum(self._gpu_hist) / len(self._gpu_hist)

            time.sleep(self.interval)


class PhotoAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PhotoAI Pipeline")
        self.root.geometry("960x780")
        self.root.configure(bg=BG)
        self.root.minsize(800, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.output_queue   = queue.Queue()
        self.current_process = None
        self.running         = False
        self.is_pipeline_run = False
        self.current_key     = None
        self.pipeline_queue  = []
        self.start_time      = 0

        # stats tracking
        self._img_done   = 0
        self._img_total  = 0
        self._img_sec    = 0.0
        self._eta_str    = ""

        self.status_labels = {}
        self.run_buttons   = {}
        self.card_frames   = {}
        self.config_vars   = {}
        self._trace_fp     = None
        self._current_trace_path = None

        self.monitor = SystemMonitor()
        self.monitor.start()

        self._build_ui()
        self._load_config()
        self._poll_output()
        self._poll_stats()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG, pady=10)
        hdr.pack(fill="x", padx=20)
        tk.Label(hdr, text="📸  PhotoAI Pipeline", font=FONT_HEAD,
                 fg=TEXT, bg=BG).pack(side="left")
        tk.Label(hdr, text="Comprehensive Image Archiving System",
                 font=FONT_SM, fg=TEXT_DIM, bg=BG).pack(side="left", padx=(12, 0), pady=(6, 0))

        sep = tk.Frame(self.root, bg=BORDER, height=1)
        sep.pack(fill="x", padx=20)

        # ---- Tabbed Notebook ----
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook", background=BG, borderwidth=0)
        style.configure("Dark.TNotebook.Tab", background=CARD_BG, foreground=TEXT_DIM,
                        padding=[14, 6], font=FONT_BOLD)
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", CARD_BG), ("!selected", BG)],
                  foreground=[("selected", ACCENT), ("!selected", TEXT_DIM)])

        self.notebook = ttk.Notebook(self.root, style="Dark.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=18, pady=(8, 0))

        # ---- Tab 1: Pipeline ----
        pipeline_tab = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(pipeline_tab, text="  🔧 Pipeline  ")
        self._build_pipeline_tab(pipeline_tab)

        # ---- Tab 2: Compress ----
        compress_tab = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(compress_tab, text="  🗜️ Compress  ")
        self._build_compress_tab(compress_tab)

        # ---- Shared Log Panel (below tabs) ----
        log_header = tk.Frame(self.root, bg=BG)
        log_header.pack(fill="x", padx=20, pady=(8, 0))
        tk.Label(log_header, text="Output Log", font=FONT_BOLD,
                 fg=TEXT, bg=BG).pack(side="left")
        self.timer_label = tk.Label(log_header, text="", font=FONT_SM,
                                    fg=TEXT_DIM, bg=BG)
        self.timer_label.pack(side="right")
        tk.Button(log_header, text="Clear", font=FONT_SM, fg=TEXT_DIM, bg=CARD_BG,
                  relief="flat", padx=8, command=self._clear_log).pack(side="right", padx=(0, 8))

        log_frame = tk.Frame(self.root, bg=BORDER, padx=1, pady=1)
        log_frame.pack(fill="both", expand=True, padx=20, pady=(4, 12))

        self.log = tk.Text(log_frame, bg=LOG_BG, fg=TEXT_DIM, font=FONT_LOG,
                           relief="flat", wrap="word", insertbackground=TEXT,
                           selectbackground=ACCENT, padx=10, pady=8, state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

        # Log tags for colored output
        self.log.tag_configure("error",   foreground=RED_LIT)
        self.log.tag_configure("success", foreground=GREEN_LIT)
        self.log.tag_configure("warning", foreground=YELLOW)
        self.log.tag_configure("accent",  foreground=ACCENT)
        self.log.tag_configure("bold",    foreground=TEXT, font=(FONT_LOG[0], FONT_LOG[1], "bold"))

        self._append_log("Ready. Select a script or click 'Run Full Pipeline' to start.\n", "accent")

        # Status bar
        status_bar = tk.Frame(self.root, bg=CARD_BG, height=32)
        status_bar.pack(fill="x", side="bottom")
        self.status_text = tk.Label(status_bar, text="● Idle", font=FONT_SM,
                                    fg=TEXT_DIM, bg=CARD_BG, padx=12)
        self.status_text.pack(side="left")

        tk.Button(status_bar, text="Reset Database", font=FONT_SM, fg=YELLOW,
                  bg=CARD_BG, relief="flat", padx=10, cursor="hand2",
                  command=self._reset_db).pack(side="right", padx=8)

        tk.Button(status_bar, text="🏷️ Name Faces", font=FONT_SM, fg=ACCENT,
                  bg=CARD_BG, relief="flat", padx=10, cursor="hand2",
                  command=self._open_name_faces_ui).pack(side="right", padx=8)

    def _build_pipeline_tab(self, parent):
        """Build the main pipeline tab content."""
        # Config panel
        self._build_config(parent)

        # Cards
        cards_frame = tk.Frame(parent, bg=BG)
        cards_frame.pack(fill="x", padx=2, pady=(10, 0))
        for col in range(3):
            cards_frame.columnconfigure(col, weight=1)

        for idx, key in enumerate(PIPELINE_ORDER):
            num, title, desc, _ = SCRIPTS[key]
            r, c = divmod(idx, 3)
            self._build_card(cards_frame, key, num, title, desc, r, c)

        # "Run All" card
        self._build_card(cards_frame, "all", "▶▶", "Run Full Pipeline",
                         "Execute all 5 steps sequentially", 1, 2, accent=True)

        # Stats panel
        self._build_stats(parent)

    def _build_compress_tab(self, parent):
        """Build the Compress tab with input/output, format, quality controls."""
        # ---- Header ----
        hdr = tk.Frame(parent, bg=BG, pady=8)
        hdr.pack(fill="x", padx=10)
        tk.Label(hdr, text="🗜️  Image Compression", font=("Segoe UI", 14, "bold"),
                 fg=TEXT, bg=BG).pack(side="left")
        tk.Label(hdr, text="Compress images while preserving quality & metadata",
                 font=FONT_SM, fg=TEXT_DIM, bg=BG).pack(side="left", padx=(12, 0), pady=(4, 0))

        # ---- Settings Card ----
        cfg = tk.LabelFrame(parent, text="  Compression Settings  ", font=FONT_BOLD,
                            fg=TEXT_DIM, bg=CARD_BG, padx=14, pady=10,
                            highlightbackground=BORDER, highlightthickness=1)
        cfg.pack(fill="x", padx=10, pady=(4, 6))
        cfg.columnconfigure(1, weight=1)

        # Input folder
        tk.Label(cfg, text="Input Folder:", font=FONT_SM, fg=TEXT_DIM, bg=CARD_BG
                 ).grid(row=0, column=0, sticky="w", pady=3)
        self.config_vars["compress_input"] = tk.StringVar(value=r"D:\Photos")
        ent_in = tk.Entry(cfg, textvariable=self.config_vars["compress_input"], font=FONT_SM,
                          bg=INPUT_BG, fg=TEXT, insertbackground=TEXT, relief="flat",
                          highlightbackground=BORDER, highlightthickness=1)
        ent_in.grid(row=0, column=1, sticky="ew", padx=(8, 4), pady=3)
        tk.Button(cfg, text="📁", font=FONT_SM, bg=CARD_BG, fg=TEXT_DIM, relief="flat", padx=4,
                  command=lambda: self._browse(self.config_vars["compress_input"], "compress_input_dir")
                  ).grid(row=0, column=2, padx=(0, 2), pady=3)
        ToolTip(ent_in, "The folder containing images to compress. Archives or any photo folder.")

        # Output folder
        tk.Label(cfg, text="Output Folder:", font=FONT_SM, fg=TEXT_DIM, bg=CARD_BG
                 ).grid(row=1, column=0, sticky="w", pady=3)
        self.config_vars["compress_output"] = tk.StringVar(value=r"D:\Photos\Compressed")
        ent_out = tk.Entry(cfg, textvariable=self.config_vars["compress_output"], font=FONT_SM,
                           bg=INPUT_BG, fg=TEXT, insertbackground=TEXT, relief="flat",
                           highlightbackground=BORDER, highlightthickness=1)
        ent_out.grid(row=1, column=1, sticky="ew", padx=(8, 4), pady=3)
        tk.Button(cfg, text="📁", font=FONT_SM, bg=CARD_BG, fg=TEXT_DIM, relief="flat", padx=4,
                  command=lambda: self._browse(self.config_vars["compress_output"], "compress_output_dir")
                  ).grid(row=1, column=2, padx=(0, 2), pady=3)
        ToolTip(ent_out, "Where compressed images will be saved. Maintains original folder structure.")

        # ---- Row: Format + Quality + Max Dim ----
        opts_row = tk.Frame(cfg, bg=CARD_BG)
        opts_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 4))

        # Format selector
        fmt_frame = tk.Frame(opts_row, bg=CARD_BG)
        fmt_frame.pack(side="left", padx=(0, 20))
        tk.Label(fmt_frame, text="Format:", font=FONT_SM, fg=ACCENT, bg=CARD_BG).pack(side="left", padx=(0, 6))
        self.config_vars["compress_format"] = tk.StringVar(value="AVIF")
        for fmt in ["AVIF", "JPEG", "PNG"]:
            rb = tk.Radiobutton(fmt_frame, text=fmt, variable=self.config_vars["compress_format"],
                                value=fmt, font=FONT_SM, fg=TEXT, bg=CARD_BG,
                                selectcolor=INPUT_BG, activebackground=CARD_BG,
                                activeforeground=TEXT, indicatoron=True, cursor="hand2")
            rb.pack(side="left", padx=(0, 8))
        ToolTip(fmt_frame, "AVIF: Best compression (50-70% smaller than JPEG), modern & widely supported.\n"
                           "JPEG: Universal compatibility at quality 85.\n"
                           "PNG: Lossless, no quality loss but larger files.")

        # Quality slider
        q_frame = tk.Frame(opts_row, bg=CARD_BG)
        q_frame.pack(side="left", padx=(0, 20))
        tk.Label(q_frame, text="Quality:", font=FONT_SM, fg=ACCENT, bg=CARD_BG).pack(side="left", padx=(0, 6))
        self.config_vars["compress_quality"] = tk.StringVar(value="80")
        q_scale = tk.Scale(q_frame, from_=10, to=100, orient="horizontal", length=120,
                           bg=CARD_BG, fg=TEXT, troughcolor=INPUT_BG, highlightthickness=0,
                           activebackground=ACCENT, font=FONT_SM,
                           command=lambda v: self.config_vars["compress_quality"].set(str(int(float(v)))))
        q_scale.set(80)
        q_scale.pack(side="left")
        ToolTip(q_frame, "Higher = better quality, larger file.\n"
                         "80 is the sweet spot for AVIF (near-lossless visual quality).\n"
                         "For JPEG, 85 is recommended.")

        # Max dimension
        d_frame = tk.Frame(opts_row, bg=CARD_BG)
        d_frame.pack(side="left", padx=(0, 20))
        tk.Label(d_frame, text="Max Dim:", font=FONT_SM, fg=ACCENT, bg=CARD_BG).pack(side="left", padx=(0, 6))
        self.config_vars["compress_max_dim"] = tk.StringVar(value="0")
        dim_ent = tk.Entry(d_frame, textvariable=self.config_vars["compress_max_dim"], font=FONT_SM,
                           bg=INPUT_BG, fg=TEXT, insertbackground=TEXT, relief="flat", width=6,
                           highlightbackground=BORDER, highlightthickness=1)
        dim_ent.pack(side="left")
        tk.Label(d_frame, text="px", font=FONT_SM, fg=TEXT_DIM, bg=CARD_BG).pack(side="left", padx=(4, 0))
        ToolTip(d_frame, "Maximum width or height in pixels. 0 = keep original size.\n"
                         "E.g., 4096 will resize 6000x4000 → 4096x2731.")

        # ---- Compress Button ----
        btn_row = tk.Frame(parent, bg=BG)
        btn_row.pack(fill="x", padx=10, pady=(6, 4))

        self.compress_btn = tk.Button(
            btn_row, text="🗜️  Start Compression", font=("Segoe UI", 11, "bold"),
            fg="white", bg=ACCENT, activebackground=ACCENT, activeforeground="white",
            relief="flat", padx=24, pady=8, cursor="hand2",
            command=self.run_compress
        )
        self.compress_btn.pack(side="left")

        # Info label
        self._compress_info = tk.Label(btn_row, text="Select input folder and format, then click Start.",
                                       font=FONT_SM, fg=TEXT_DIM, bg=BG)
        self._compress_info.pack(side="left", padx=(16, 0))



    def _build_config(self, parent=None):
        if parent is None:
            parent = self.root
        wrapper = tk.Frame(parent, bg=BG, padx=2, pady=8)
        wrapper.pack(fill="x")

        cfg = tk.LabelFrame(wrapper, text="  Configuration  ", font=FONT_BOLD,
                            fg=TEXT_DIM, bg=CARD_BG, padx=12, pady=8,
                            highlightbackground=BORDER, highlightthickness=1)
        cfg.pack(fill="x")
        cfg.columnconfigure(1, weight=1)

        paths = [
            ("Source Directory:", "master_dir",  r"D:\POCOP - Copy\+"),
            ("Database Path:",   "db_path",     r"D:\PhotoAI\photo_catalog.db"),
            ("Output Directory:","output_dir",  r"D:\Photos"),
        ]
        for i, (label, key, default) in enumerate(paths):
            tk.Label(cfg, text=label, font=FONT_SM, fg=TEXT_DIM, bg=CARD_BG
                     ).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=default)
            self.config_vars[key] = var
            ent = tk.Entry(cfg, textvariable=var, font=FONT_SM, bg=INPUT_BG,
                           fg=TEXT, insertbackground=TEXT, relief="flat",
                           highlightbackground=BORDER, highlightthickness=1)
            ent.grid(row=i, column=1, sticky="ew", padx=(8, 4), pady=2)
            tk.Button(cfg, text="📁", font=FONT_SM, bg=CARD_BG, fg=TEXT_DIM,
                      relief="flat", padx=4,
                      command=lambda v=var, k=key: self._browse(v, k)
                      ).grid(row=i, column=2, padx=(0, 2), pady=2)

            tip_text = ""
            if key == "master_dir":
                tip_text = "The folder containing all your raw photos. Subdirectories are searched automatically."
            elif key == "db_path":
                tip_text = "Location of the SQLite file to save metadata, facial embeddings, and classifications."
            elif key == "output_dir":
                tip_text = "Where the final Archive folders will be created."

            # Bind tooltips for folder paths too!
            lbl_widget = cfg.grid_slaves(row=i, column=0)[0]
            ToolTip(lbl_widget, tip_text)
            ToolTip(ent, tip_text)

        # Numeric params row
        params_row = tk.Frame(cfg, bg=CARD_BG)
        params_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 2))

        nums = [
            ("Det Size ⓘ:", "det_size", "640",
             "Detection Resolution (Det Size):\n"
             "The internal resolution the AI scales your images to before looking for faces.\n\n"
             "• Lower (e.g. 320): Process images lightning-fast. Good if faces are large and clear.\n"
             "• Default (640): Balanced speed and accuracy.\n"
             "• Raise (e.g. 1024 or 1280): Find extremely small faces far in the background of large group shots (significantly slower processing)."),

            ("Det Thresh ⓘ:", "det_thresh", "0.5",
             "Detection Threshold (Det Thresh):\n"
             "The confidence cutoff (from 0.0 to 1.0) for considering a shape to be a human face.\n\n"
             "• Default: 0.5\n"
             "• Higher (e.g. 0.6 or 0.7): Makes the detector stricter. Use this if you are getting 'false positives' (e.g., dogs, statues, clouds being tagged as faces)."),

            ("Min Cluster ⓘ:", "min_cluster_size", "4",
             "Minimum Cluster Size (Min Cluster):\n"
             "The absolute minimum number of photos a person must appear in to be granted their own 'Person_XYZ' folder.\n\n"
             "• Lower: Catches rare guests who only appear in 2-3 photos.\n"
             "• Higher: Strictly only captures core family members. People appearing less than this number will be tossed into 'Unknown_Faces'."),

            ("Merge Thresh ⓘ:", "merge_threshold", "0.40",
             "Merge Threshold (Merge Thresh):\n"
             "The post-clustering 'forgiveness' threshold. Used to merge clusters that the AI incorrectly separated (e.g., photos from 2010 vs 2020).\n\n"
             "• Default (0.40): Safe sweet spot for ArcFace.\n"
             "• Raise (e.g. 0.50 or 0.60): Use if ONE person is being split into multiple folders.\n"
             "• Lower (e.g. 0.35 or 0.30): Use if TWO different people (like siblings) are grouped together."),

            ("Confidence ⓘ:", "confidence_threshold", "0.15",
             "Semantic Confidence (Confidence):\n"
             "The minimum confidence for the CLIP semantic model (Step 3).\n\n"
             "• Default (0.15): Good balance for 17 classes.\n"
             "• If the AI cannot be at least this confident that the image belongs to a category, it throws it into the 'Other' folder.\n"
             "• Raise this if you are getting miscategorized images."),
        ]

        for j, (label, key, default, tip_text) in enumerate(nums):
            # Wrap each Param in its own Frame so the tooltips don't drop when picking up gaps between label and entry
            param_frame = tk.Frame(params_row, bg=CARD_BG, padx=4, pady=2)
            param_frame.pack(side="left", padx=(0 if j == 0 else 6, 2))

            lbl = tk.Label(param_frame, text=label, font=FONT_SM, fg=ACCENT, bg=CARD_BG, cursor="hand2")
            lbl.pack(side="left", padx=(0, 4))

            var = tk.StringVar(value=default)
            self.config_vars[key] = var
            ent = tk.Entry(param_frame, textvariable=var, font=FONT_SM, bg=INPUT_BG,
                           fg=TEXT, insertbackground=TEXT, relief="flat", width=5,
                           highlightbackground=BORDER, highlightthickness=1)
            ent.pack(side="left")
            
            # Bind tooltip to the frame and all inner widgets to create a solid, large hover area
            ToolTip(param_frame, tip_text)
            ToolTip(lbl, tip_text)
            ToolTip(ent, tip_text)

        # Model Size selector
        model_frame = tk.Frame(params_row, bg=CARD_BG, padx=4, pady=2)
        model_frame.pack(side="left", padx=(10, 10))
        tk.Label(model_frame, text="Vision Model:", font=FONT_SM, fg=ACCENT, bg=CARD_BG).pack(side="left", padx=(0, 4))
        
        self.config_vars["clip_model_size"] = tk.StringVar(value="clip")
        # clip = previous Ultra CLIP variant, siglip2 = new Ultra option
        for m_size in [("clip", "Base"), ("siglip2", "Ultra")]:
            rb = tk.Radiobutton(model_frame, text=m_size[1], variable=self.config_vars["clip_model_size"],
                                value=m_size[0], font=FONT_SM, fg=TEXT, bg=CARD_BG,
                                selectcolor=INPUT_BG, activebackground=CARD_BG,
                                activeforeground=TEXT, indicatoron=False, cursor="hand2")
            rb.pack(side="left", padx=(0, 2))
        ToolTip(model_frame, "Base: CLIP (previous Ultra model)\nUltra: SigLIP 2 SO400M")

        save_btn = tk.Button(params_row, text="💾 Save Config", font=FONT_SM, fg=TEXT, bg=CARD_BG, cursor="hand2", relief="flat", command=self._manual_save_config)
        save_btn.pack(side="right", padx=(12, 2))

    def _build_card(self, parent, key, num, title, desc, row, col, accent=False):
        card = tk.Frame(parent, bg=CARD_BG, padx=14, pady=10,
                        highlightbackground=ACCENT if accent else BORDER,
                        highlightthickness=1)
        card.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        self.card_frames[key] = card

        top = tk.Frame(card, bg=CARD_BG)
        top.pack(fill="x")
        tk.Label(top, text=num, font=FONT_NUM, fg=ACCENT if not accent else GREEN_LIT,
                 bg=CARD_BG).pack(side="left")
        tk.Label(top, text=title, font=FONT_BOLD, fg=TEXT,
                 bg=CARD_BG).pack(side="left", padx=(8, 0))

        tk.Label(card, text=desc, font=FONT_SM, fg=TEXT_DIM, bg=CARD_BG,
                 anchor="w").pack(fill="x", pady=(2, 8))

        bot = tk.Frame(card, bg=CARD_BG)
        bot.pack(fill="x")

        status = tk.Label(bot, text="● Ready", font=FONT_SM, fg=TEXT_DIM, bg=CARD_BG)
        status.pack(side="left")
        self.status_labels[key] = status

        btn_bg = GREEN if not accent else ACCENT
        btn = tk.Button(bot, text="▶ Run" if not accent else "▶▶ Start",
                        font=("Segoe UI", 9, "bold"), fg="white", bg=btn_bg,
                        activebackground=btn_bg, activeforeground="white",
                        relief="flat", padx=14, pady=2, cursor="hand2",
                        command=lambda: self.run_all() if key == "all" else self.run_script(key))
        btn.pack(side="right")
        self.run_buttons[key] = btn

    def _browse(self, var, key):
        if "dir" in key:
            path = filedialog.askdirectory(initialdir=var.get())
        else:
            path = filedialog.askopenfilename(initialdir=os.path.dirname(var.get()),
                                              filetypes=[("SQLite DB", "*.db"), ("All", "*.*")])
        if path:
            path = os.path.normpath(path)
            var.set(path)
            
            # Auto-update output to "same folder name + compressed"
            if key == "compress_input_dir":
                self.config_vars["compress_output"].set(path + "_compressed")

    # ---------------------------------------------------------- Stats Panel
    def _build_stats(self, parent=None):
        if parent is None:
            parent = self.root
        wrapper = tk.Frame(parent, bg=BG)
        wrapper.pack(fill="x", padx=2, pady=(6, 0))

        panel = tk.Frame(wrapper, bg=CARD_BG,
                         highlightbackground=BORDER, highlightthickness=1)
        panel.pack(fill="x")

        # ---- Left: CPU ----
        cpu_col = tk.Frame(panel, bg=CARD_BG, padx=14, pady=8)
        cpu_col.pack(side="left", fill="x", expand=True)

        tk.Label(cpu_col, text="CPU", font=FONT_BOLD, fg=TEXT_DIM,
                 bg=CARD_BG).pack(anchor="w")
        self._cpu_bar = ttk.Progressbar(cpu_col, length=180, maximum=100,
                                        mode="determinate", style="CPU.Horizontal.TProgressbar")
        self._cpu_bar.pack(anchor="w", pady=(3, 0))
        self._cpu_lbl = tk.Label(cpu_col, text="Current: --  Avg: --",
                                 font=FONT_SM, fg=TEXT_DIM, bg=CARD_BG)
        self._cpu_lbl.pack(anchor="w")

        tk.Frame(panel, bg=BORDER, width=1).pack(side="left", fill="y", pady=6)

        # ---- Middle: GPU ----
        gpu_col = tk.Frame(panel, bg=CARD_BG, padx=14, pady=8)
        gpu_col.pack(side="left", fill="x", expand=True)

        tk.Label(gpu_col, text="GPU  (RX 5700 XT)", font=FONT_BOLD, fg=TEXT_DIM,
                 bg=CARD_BG).pack(anchor="w")
        self._gpu_bar = ttk.Progressbar(gpu_col, length=180, maximum=100,
                                        mode="determinate", style="GPU.Horizontal.TProgressbar")
        self._gpu_bar.pack(anchor="w", pady=(3, 0))
        self._gpu_lbl = tk.Label(gpu_col, text="Current: --  Avg: --",
                                 font=FONT_SM, fg=TEXT_DIM, bg=CARD_BG)
        self._gpu_lbl.pack(anchor="w")

        tk.Frame(panel, bg=BORDER, width=1).pack(side="left", fill="y", pady=6)

        # ---- Right: Progress / Speed ----
        prog_col = tk.Frame(panel, bg=CARD_BG, padx=14, pady=8)
        prog_col.pack(side="left", fill="x", expand=True)

        tk.Label(prog_col, text="Progress", font=FONT_BOLD, fg=TEXT_DIM,
                 bg=CARD_BG).pack(anchor="w")
        self._prog_bar = ttk.Progressbar(prog_col, length=200, maximum=100,
                                         mode="determinate", style="Prog.Horizontal.TProgressbar")
        self._prog_bar.pack(anchor="w", pady=(3, 0))
        self._prog_lbl = tk.Label(prog_col, text="0 / 0 images",
                                  font=FONT_SM, fg=TEXT_DIM, bg=CARD_BG)
        self._prog_lbl.pack(anchor="w")

        tk.Frame(panel, bg=BORDER, width=1).pack(side="left", fill="y", pady=6)

        # ---- Far right: Speed / ETA ----
        speed_col = tk.Frame(panel, bg=CARD_BG, padx=14, pady=8)
        speed_col.pack(side="left", fill="x", expand=True)

        tk.Label(speed_col, text="Throughput", font=FONT_BOLD, fg=TEXT_DIM,
                 bg=CARD_BG).pack(anchor="w")
        self._speed_lbl = tk.Label(speed_col, text="-- img/s",
                                   font=("Segoe UI", 16, "bold"), fg=ACCENT, bg=CARD_BG)
        self._speed_lbl.pack(anchor="w")
        self._eta_lbl = tk.Label(speed_col, text="ETA: --",
                                 font=FONT_SM, fg=TEXT_DIM, bg=CARD_BG)
        self._eta_lbl.pack(anchor="w")

        # Style the progress bars
        style = ttk.Style()
        style.theme_use("default")
        for name, color in [("CPU", "#58a6ff"), ("GPU", "#3fb950"), ("Prog", "#d29922")]:
            style.configure(f"{name}.Horizontal.TProgressbar",
                            troughcolor=INPUT_BG, background=color,
                            bordercolor=BORDER, lightcolor=color, darkcolor=color,
                            thickness=10)

    def _poll_stats(self):
        """Refresh the stats panel every second."""
        # CPU
        cpu_c = self.monitor.cpu_cur
        cpu_a = self.monitor.cpu_avg
        self._cpu_bar["value"] = cpu_c
        self._cpu_lbl.configure(
            text=f"Current: {cpu_c:.0f}%   Avg: {cpu_a:.0f}%",
            fg=RED_LIT if cpu_c > 85 else (YELLOW if cpu_c > 60 else TEXT_DIM)
        )

        # GPU
        gpu_c = self.monitor.gpu_cur
        gpu_a = self.monitor.gpu_avg
        if gpu_a >= 0:
            self._gpu_bar["value"] = gpu_c
            self._gpu_lbl.configure(
                text=f"Current: {gpu_c:.0f}%   Avg: {gpu_a:.0f}%",
                fg=RED_LIT if gpu_c > 85 else (YELLOW if gpu_c > 60 else TEXT_DIM)
            )
        else:
            self._gpu_lbl.configure(text="N/A — GPU undetected or CPU fallback active", fg=TEXT_DIM)

        # Progress
        if self._img_total > 0:
            pct = min(100, self._img_done / self._img_total * 100)
            self._prog_bar["value"] = pct
            self._prog_lbl.configure(
                text=f"{self._img_done:,} / {self._img_total:,} images  ({pct:.0f}%)"
            )
        elif not self.running:
            self._prog_bar["value"] = 0
            self._prog_lbl.configure(text="0 / 0 images")

        # Speed / ETA
        if self._img_sec > 0:
            self._speed_lbl.configure(text=f"{self._img_sec:.1f} img/s", fg=ACCENT)
            self._eta_lbl.configure(text=f"ETA: {self._eta_str}" if self._eta_str else "")
        elif not self.running:
            self._speed_lbl.configure(text="-- img/s", fg=TEXT_DIM)
            self._eta_lbl.configure(text="ETA: --")

        self.root.after(1000, self._poll_stats)

    def _parse_log_line(self, line):
        """Extract progress/speed info from a script output line."""
        # Pattern: [   500/20000] ... 15.3 img/s ... ETA: 5 min
        # Require at least one space before the number to avoid matching setup steps like [1/2]
        m = re.search(r'\[\s+(\d+)\s*/\s*(\d+)\s*\]', line)
        if m:
            self._img_done  = int(m.group(1))
            self._img_total = int(m.group(2))

        m2 = re.search(r'([\d.]+)\s*img/s', line)
        if m2:
            self._img_sec = float(m2.group(1))

        m3 = re.search(r'ETA:\s*([\d.]+\s*\w+)', line)
        if m3:
            self._eta_str = m3.group(1).strip()



    # ------------------------------------------------------------ Config I/O
    def _save_config(self):
        cfg = {}
        for key, var in self.config_vars.items():
            val = var.get()
            # Try numeric conversion
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
            cfg[key] = val
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)

    def _manual_save_config(self):
        self._save_config()
        self.status_text.configure(text="● Config Saved", fg=GREEN_LIT)
        self.root.after(3000, lambda: self.status_text.configure(text="● Idle", fg=TEXT_DIM) if not self.running else None)

    def _load_config(self):
        if not os.path.exists(CONFIG_PATH):
            return
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            for key, val in cfg.items():
                if key in self.config_vars:
                    if key == "clip_model_size":
                        val_str = str(val).lower()
                        if val_str == "base":
                            val = "clip"
                        elif val_str == "large":
                            val = "siglip2"
                    self.config_vars[key].set(str(val))
        except Exception:
            pass

    def _strip_ansi(self, text):
        """Remove ANSI escape/control characters for clean trace files."""
        cleaned = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)
        return cleaned.replace('\x1b', '')

    def _make_trace_history_path(self):
        """Build a unique timestamped trace file path under logs/history/."""
        os.makedirs(TRACE_HISTORY_DIR, exist_ok=True)
        base = time.strftime("run_%Y%m%d_%H%M%S")
        candidate = os.path.join(TRACE_HISTORY_DIR, f"{base}.log")
        if not os.path.exists(candidate):
            return candidate
        for idx in range(1, 1000):
            candidate = os.path.join(TRACE_HISTORY_DIR, f"{base}_{idx:02d}.log")
            if not os.path.exists(candidate):
                return candidate
        # Extremely unlikely fallback: append process id.
        return os.path.join(TRACE_HISTORY_DIR, f"{base}_{os.getpid()}.log")

    def _prune_trace_history(self):
        """Keep only the latest TRACE_HISTORY_MAX_RUNS trace files."""
        os.makedirs(TRACE_HISTORY_DIR, exist_ok=True)
        history_files = []
        for name in os.listdir(TRACE_HISTORY_DIR):
            if not name.lower().endswith('.log'):
                continue
            full_path = os.path.join(TRACE_HISTORY_DIR, name)
            if os.path.isfile(full_path):
                history_files.append(full_path)

        if len(history_files) <= TRACE_HISTORY_MAX_RUNS:
            return

        history_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        stale_files = history_files[TRACE_HISTORY_MAX_RUNS:]
        for stale_path in stale_files:
            removed = False
            for _ in range(3):
                try:
                    os.remove(stale_path)
                    removed = True
                    break
                except PermissionError:
                    time.sleep(0.1)
                except OSError:
                    break
            if not removed:
                self._append_log(f"[WARN] Could not prune old trace file: {stale_path}\n", "warn")

    def _archive_last_run_trace(self):
        """Copy latest run trace into logs/history and enforce retention."""
        if not self._current_trace_path or not os.path.exists(self._current_trace_path):
            return
        history_path = self._make_trace_history_path()
        try:
            with open(self._current_trace_path, 'r', encoding='utf-8', errors='replace') as src_fp:
                content = src_fp.read()
            with open(history_path, 'w', encoding='utf-8', newline='') as dst_fp:
                dst_fp.write(content)
            self._prune_trace_history()
        except Exception as exc:
            self._append_log(f"[WARN] Failed to archive run trace: {type(exc).__name__}: {exc}\n", "warn")

    def _start_run_trace(self, run_name, script_path=None):
        """Start (overwrite) the rolling last-run trace file."""
        os.makedirs(TRACE_DIR, exist_ok=True)
        os.makedirs(TRACE_HISTORY_DIR, exist_ok=True)
        self._close_run_trace()
        self._prune_trace_history()

        self._trace_fp = open(LAST_RUN_TRACE_PATH, 'w', encoding='utf-8', newline='')
        self._current_trace_path = LAST_RUN_TRACE_PATH
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        mode = "full-pipeline" if self.is_pipeline_run else "single-step"

        self._trace_fp.write("=" * 70 + "\n")
        self._trace_fp.write("PhotoAI Last Run Trace\n")
        self._trace_fp.write(f"Started   : {started}\n")
        self._trace_fp.write(f"Run       : {run_name}\n")
        self._trace_fp.write(f"Mode      : {mode}\n")
        if script_path:
            self._trace_fp.write(f"Script    : {script_path}\n")
        self._trace_fp.write(f"Python    : {sys.executable}\n")
        self._trace_fp.write("-" * 70 + "\n")
        self._trace_fp.write("Config snapshot:\n")

        cfg = {k: v.get() for k, v in self.config_vars.items()}
        self._trace_fp.write(json.dumps(cfg, indent=2, ensure_ascii=False))
        self._trace_fp.write("\n" + "=" * 70 + "\n")
        self._trace_fp.flush()

    def _append_run_trace(self, text):
        if not self._trace_fp:
            return
        cleaned = self._strip_ansi(text)
        if not cleaned:
            return
        self._trace_fp.write(cleaned)
        if not cleaned.endswith("\n"):
            self._trace_fp.write("\n")
        self._trace_fp.flush()

    def _close_run_trace(self, returncode=None):
        if not self._trace_fp:
            return
        should_archive = returncode is not None
        if returncode is not None:
            finished = time.strftime("%Y-%m-%d %H:%M:%S")
            self._trace_fp.write("\n" + "=" * 70 + "\n")
            self._trace_fp.write(f"Finished  : {finished}\n")
            self._trace_fp.write(f"Exit code : {returncode}\n")
            self._trace_fp.write("=" * 70 + "\n")
        self._trace_fp.flush()
        self._trace_fp.close()
        self._trace_fp = None
        if should_archive:
            self._archive_last_run_trace()

    def _validate_extract_config(self):
        """Validate config required for face extraction before launch."""
        master_dir = self.config_vars.get("master_dir", tk.StringVar()).get().strip()
        db_path = self.config_vars.get("db_path", tk.StringVar()).get().strip()
        det_size_raw = self.config_vars.get("det_size", tk.StringVar(value="640")).get().strip()
        det_thresh_raw = self.config_vars.get("det_thresh", tk.StringVar(value="0.5")).get().strip()

        if not master_dir or not os.path.isdir(master_dir):
            messagebox.showerror("Invalid Source Directory", "Source Directory does not exist. Please choose a valid folder before running extraction.")
            return False

        if not db_path:
            messagebox.showerror("Invalid Database Path", "Database Path is empty. Please set a valid .db path.")
            return False

        db_parent = os.path.dirname(db_path) or "."
        if not os.path.isdir(db_parent):
            messagebox.showerror("Invalid Database Path", "Database folder does not exist. Please choose an existing folder for the database.")
            return False

        try:
            det_size = int(float(det_size_raw))
        except ValueError:
            messagebox.showerror("Invalid Det Size", "Det Size must be a whole number (e.g. 640).")
            return False

        if det_size < 160 or det_size > 2048:
            messagebox.showerror("Invalid Det Size", "Det Size must be between 160 and 2048.")
            return False

        try:
            det_thresh = float(det_thresh_raw)
        except ValueError:
            messagebox.showerror("Invalid Det Thresh", "Det Thresh must be a number between 0.0 and 1.0.")
            return False

        if det_thresh < 0.0 or det_thresh > 1.0:
            messagebox.showerror("Invalid Det Thresh", "Det Thresh must be between 0.0 and 1.0.")
            return False

        if det_size != 640:
            proceed = messagebox.askyesno(
                "Detection Size Warning",
                "On some DirectML setups, Step 1 may fail for Det Size values other than 640. "
                "If that happens, PhotoAI will fall back to CPU for extraction (slower but stable). Continue?"
            )
            if not proceed:
                return False

        return True

    # ---------------------------------------------------------- Script Exec
    def run_script(self, key):
        if self.running:
            messagebox.showwarning("Busy", "A script is already running.")
            return
        if key == "extract" and not self._validate_extract_config():
            return
        self._save_config()

        _, title, _, script_file = SCRIPTS[key]
        script_path = os.path.join(SCRIPT_DIR, script_file)

        self.current_key = key
        self.running = True
        self.is_pipeline_run = False
        self.pipeline_queue = []
        self._set_buttons_enabled(False)
        self._set_status(key, "running")
        self.status_text.configure(text="● Running...", fg=YELLOW)
        self.start_time = time.time()
        self._update_timer()
        self._clear_log()
        # Reset progress stats for new run
        self._img_done  = 0
        self._img_total = 0
        self._img_sec   = 0.0
        self._eta_str   = ""
        self.monitor.reset_averages()

        self._start_run_trace(f"Step {SCRIPTS[key][0]} - {title}", script_path=script_path)
        self._append_log(f"Trace file: {LAST_RUN_TRACE_PATH}\n", "accent")
        self._append_log(f"━━━ Starting: {title} ━━━\n", "accent")

        thread = threading.Thread(target=self._exec_subprocess,
                                  args=(script_path,), daemon=True)
        thread.start()

    def run_compress(self):
        """Run the compression script from the Compress tab."""
        if self.running:
            messagebox.showwarning("Busy", "A script is already running.")
            return
        self._save_config()

        script_path = os.path.join(SCRIPT_DIR, "scripts", "5_compress_images.py")

        self.current_key = "compress"
        self.running = True
        self.is_pipeline_run = False
        self.pipeline_queue = []
        self._set_buttons_enabled(False)
        self.compress_btn.configure(state="disabled", text="⏳ Compressing...", bg=YELLOW)
        self._compress_info.configure(text="Compression running... check Output Log below.", fg=YELLOW)
        self.status_text.configure(text="● Compressing...", fg=YELLOW)
        self.start_time = time.time()
        self._update_timer()
        self._clear_log()
        self._img_done  = 0
        self._img_total = 0
        self._img_sec   = 0.0
        self._eta_str   = ""
        self.monitor.reset_averages()

        self._start_run_trace("Step 5 - Compress Images", script_path=script_path)
        self._append_log(f"Trace file: {LAST_RUN_TRACE_PATH}\n", "accent")
        self._append_log("━━━ Starting: Image Compression ━━━\n", "accent")

        thread = threading.Thread(target=self._exec_subprocess,
                                  args=(script_path,), daemon=True)
        thread.start()

    def run_all(self):
        if self.running:
            messagebox.showwarning("Busy", "A script is already running.")
            return
        if not self._validate_extract_config():
            return
        self._save_config()

        self.current_key = "all"
        self.pipeline_queue = list(PIPELINE_ORDER)
        self.is_pipeline_run = True
        self._start_run_trace("Full Pipeline")

        self._clear_log()
        self._append_log(f"Trace file: {LAST_RUN_TRACE_PATH}\n", "accent")
        self._append_log("━━━ Running Full Pipeline ━━━\n\n", "accent")

        # Reset all statuses
        for key in PIPELINE_ORDER:
            self._set_status(key, "pending")
        self._set_status("all", "running")

        self._run_next_pipeline_step()

    def _run_next_pipeline_step(self):
        if not self.pipeline_queue:
            self._set_status("all", "complete")
            self.status_text.configure(text="● Pipeline complete!", fg=GREEN_LIT)
            self.running = False
            self.is_pipeline_run = False
            self._set_buttons_enabled(True)
            
            # Print stats to the log / CLI
            cpu_a = self.monitor.cpu_avg
            gpu_a = self.monitor.gpu_avg
            self._append_log(
                f"\n━━━ Full Pipeline Complete! ━━━\n"
                f"  Total Session Stats:\n"
                f"  CPU Avg: {cpu_a:.0f}%\n"
                f"  GPU Avg: " + (f"{gpu_a:.0f}%\n" if gpu_a >= 0 else "(N/A)\n"),
                "success"
            )

            self._close_run_trace(returncode=0)

            # Auto-open the naming UI after a successful full run
            self.root.after(1000, self._open_name_faces_ui)
            return

        key = self.pipeline_queue.pop(0)
        self.current_key = key
        self.running = True
        self._set_buttons_enabled(False)
        self._set_status(key, "running")
        self.status_text.configure(text=f"● Running step {SCRIPTS[key][0]}...", fg=YELLOW)
        self.start_time = time.time()
        self._update_timer()
        # Reset progress stats per step
        self._img_done  = 0
        self._img_total = 0
        self._img_sec   = 0.0
        self._eta_str   = ""
        self.monitor.reset_averages()

        _, title, _, script_file = SCRIPTS[key]
        self._append_log(f"\n━━━ Step {SCRIPTS[key][0]}: {title} ━━━\n", "accent")

        script_path = os.path.join(SCRIPT_DIR, script_file)
        self._append_run_trace(f"Script: {script_path}\n")
        thread = threading.Thread(target=self._exec_subprocess,
                                  args=(script_path,), daemon=True)
        thread.start()

    def stop_script(self):
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            self.pipeline_queue.clear()
            self._append_log("\n[STOPPED] Script terminated by user.\n", "warning")

    def _exec_subprocess(self, script_path):
        try:
            # Force UTF-8 I/O so Unicode chars (arrows, ticks, etc.) don't
            # crash on Windows cp1252 consoles
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            if getattr(self, "is_pipeline_run", False):
                env["PHOTOAI_FULL_PIPELINE"] = "1"

            proc = subprocess.Popen(
                [sys.executable, "-u", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=SCRIPT_DIR,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            self.current_process = proc
            for line in iter(proc.stdout.readline, ''):
                self.output_queue.put(("line", line))
            proc.wait()
            self.output_queue.put(("done", proc.returncode))
        except Exception as e:
            self.output_queue.put(("error", str(e)))

    # ----------------------------------------------------------- Output Poll
    def _poll_output(self):
        try:
            while True:
                msg_type, data = self.output_queue.get_nowait()
                if msg_type == "line":
                    tag = None
                    if "[ERROR]" in data or "Error" in data:
                        tag = "error"
                    elif "[OK]" in data or "complete" in data.lower():
                        tag = "success"
                    elif "[SKIP]" in data or "[WARN]" in data:
                        tag = "warning"
                    elif data.strip().startswith("="):
                        tag = "accent"
                    self._parse_log_line(data)
                    self._append_log(data, tag)

                elif msg_type == "done":
                    self._on_script_done(data)

                elif msg_type == "error":
                    self._append_log(f"\n[ERROR] {data}\n", "error")
                    self._on_script_done(1)

        except queue.Empty:
            pass
        self.root.after(50, self._poll_output)

    def _on_script_done(self, returncode):
        key = self.current_key
        elapsed = time.time() - self.start_time

        if returncode == 0:
            self._set_status(key, "complete")
            self._append_log(f"  ✓ Finished in {elapsed:.1f}s\n", "success")

            if key == "archive" and not self.is_pipeline_run:
                self.root.after(1000, self._open_name_faces_ui)

            # Continue pipeline if it's a pipeline run
            if self.is_pipeline_run:
                self.root.after(500, self._run_next_pipeline_step)
                return
        else:
            self._set_status(key, "error")
            self._append_log(f"  ✗ Failed (exit code {returncode})\n", "error")
            # Abort pipeline on error
            if self.pipeline_queue:
                self.pipeline_queue.clear()
                self._set_status("all", "error")
                self._append_log("\n[PIPELINE ABORTED] Fix the error and retry.\n", "error")
            self.is_pipeline_run = False

        self.running = False
        self.current_process = None
        self._set_buttons_enabled(True)

        # Reset compress button if it was a compression run
        if key == "compress" and hasattr(self, 'compress_btn'):
            self.compress_btn.configure(state="normal", text="🗜️  Start Compression", bg=ACCENT)
            self._compress_info.configure(
                text="✓ Compression finished!" if returncode == 0 else "✗ Compression failed.",
                fg=GREEN_LIT if returncode == 0 else RED_LIT
            )

        self.status_text.configure(
            text="● Complete" if returncode == 0 else "● Error",
            fg=GREEN_LIT if returncode == 0 else RED_LIT
        )
        self.timer_label.configure(text=f"⏱ {elapsed:.1f}s")

        if not self.is_pipeline_run:
            self._close_run_trace(returncode=returncode)

    # --------------------------------------------------------------- Helpers
    def _set_status(self, key, state):
        colors = {
            "ready":   ("● Ready",   TEXT_DIM),
            "pending": ("◌ Pending", TEXT_DIM),
            "running": ("● Running", YELLOW),
            "complete":("✓ Done",    GREEN_LIT),
            "error":   ("✗ Error",   RED_LIT),
        }
        text, color = colors.get(state, ("● Ready", TEXT_DIM))
        if key in self.status_labels:
            self.status_labels[key].configure(text=text, fg=color)

        # Toggle Run/Stop button for active script
        if key in self.run_buttons:
            btn = self.run_buttons[key]
            if state == "running":
                btn.configure(text="■ Stop", bg=RED, command=self.stop_script)
            else:
                is_all = key == "all"
                btn.configure(
                    text="▶▶ Start" if is_all else "▶ Run",
                    bg=ACCENT if is_all else GREEN,
                    command=self.run_all if is_all else lambda k=key: self.run_script(k)
                )

    def _set_buttons_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for key, btn in self.run_buttons.items():
            if enabled or key == self.current_key:
                btn.configure(state="normal")
            else:
                btn.configure(state="disabled")

    def _append_log(self, text, tag=None):
        self.log.configure(state="normal")
        if tag:
            self.log.insert("end", text, tag)
        else:
            self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")
        self._append_run_trace(text)

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _update_timer(self):
        if not self.running:
            return
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        self.timer_label.configure(text=f"⏱ {mins:02d}:{secs:02d}")
        self.root.after(1000, self._update_timer)

    def _reset_db(self):
        """Delete the database so faces can be re-extracted with new ArcFace embeddings."""
        db_path = self.config_vars.get("db_path", tk.StringVar()).get()
        if not db_path:
            db_path = os.path.join(SCRIPT_DIR, "photo_catalog.db")
        if not os.path.exists(db_path):
            messagebox.showinfo("Reset DB", f"Database not found:\n{db_path}")
            return
        if self.running:
            messagebox.showwarning("Busy", "Stop the running script before resetting the database.")
            return
        if messagebox.askyesno(
            "Reset Database",
            f"This will delete ALL extracted faces and classifications:\n\n{db_path}\n\n"
            "No pipeline steps will run automatically.\n\n"
            "Continue?",
            icon="warning"
        ):
            try:
                os.remove(db_path)
                self._append_log(f"[OK] Database deleted: {db_path}\n", "warning")
                # Reset all card statuses
                for key in PIPELINE_ORDER:
                    self._set_status(key, "ready")
                self.status_text.configure(text="● Database reset", fg=YELLOW)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete database:\n{e}")

    def _open_name_faces_ui(self):
        if not PIL_OK:
            messagebox.showerror("Error", "Pillow library is required for the UI. (pip install Pillow)")
            return

        db_path = self.config_vars.get("db_path", tk.StringVar()).get()
        if not db_path or not os.path.exists(db_path):
            messagebox.showerror("Error", "Database not found. Have you run the pipeline yet?")
            return

        NameFacesUI(self.root, db_path)

    def on_close(self):
        if self.current_process and self.current_process.poll() is None:
            if messagebox.askyesno("Confirm", "A script is running. Stop and exit?"):
                self.current_process.terminate()
            else:
                return
        self._save_config()
        self._close_run_trace()
        self.monitor.stop()
        self.root.destroy()


class NameFacesUI(tk.Toplevel):
    def __init__(self, parent, db_path):
        super().__init__(parent)
        self.title("Name Faces")
        self.geometry("900x700")
        self.configure(bg=BG)
        self.db_path = db_path
        self.entries = {}
        self.images = [] # Prevent garbage collection

        self._build_ui()

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG, pady=10)
        hdr.pack(fill="x", padx=20)
        tk.Label(hdr, text="🏷️ Name the Faces", font=FONT_HEAD, fg=TEXT, bg=BG).pack(side="left")
        
        save_btn = tk.Button(hdr, text="💾 Save Names", font=("Segoe UI", 10, "bold"),
                             fg="white", bg=GREEN, activebackground=GREEN_LIT,
                             relief="flat", padx=16, pady=4, cursor="hand2",
                             command=self._save_names)
        save_btn.pack(side="right")

        # Scrollable Canvas
        container = tk.Frame(self, bg=BORDER, padx=1, pady=1)
        container.pack(fill="both", expand=True, padx=20, pady=(10, 20))

        canvas = tk.Canvas(container, bg=LOG_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        
        self.scroll_frame = tk.Frame(canvas, bg=LOG_BG)
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw", width=830)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mousewheel scroll binding
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._load_folders()

    def _load_folders(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, custom_name FROM people")
            people = cursor.fetchall()
            
            if not people:
                tk.Label(self.scroll_frame, text="No established people found in database.\nRun the pipeline first to cluster faces.",
                         font=FONT_BOLD, fg=TEXT_DIM, bg=LOG_BG).pack(pady=40)
                conn.close()
                return

            # Randomize the display order but keep the IDs attached
            for p_id, current_name in people:
                # Find 3 random face images assigned to this person
                cursor.execute('''
                    SELECT images.file_path 
                    FROM faces 
                    JOIN images ON faces.image_id = images.id 
                    WHERE faces.person_id = ? 
                    ORDER BY RANDOM() LIMIT 3
                ''', (p_id,))
                
                samples = [r[0] for r in cursor.fetchall() if os.path.exists(r[0])]
                if not samples:
                    continue

                display_name = current_name if current_name else f"Person_{p_id:03d}"

                row = tk.Frame(self.scroll_frame, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
                row.pack(fill="x", padx=10, pady=10)

                # Left side: Images
                img_container = tk.Frame(row, bg=CARD_BG)
                img_container.pack(side="left", padx=10, pady=10)

                for img_path in samples:
                    try:
                        img = Image.open(img_path)
                        img.thumbnail((150, 150), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        self.images.append(photo) # Keep ref
                        
                        lbl = tk.Label(img_container, image=photo, bg=LOG_BG)
                        lbl.pack(side="left", padx=5)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")

                # Right side: Input
                input_container = tk.Frame(row, bg=CARD_BG)
                input_container.pack(side="left", fill="x", expand=True, padx=20)
                
                tk.Label(input_container, text=f"Identify {display_name}:",
                         font=FONT_SM, fg=TEXT_DIM, bg=CARD_BG).pack(anchor="w")
                
                var = tk.StringVar(value=current_name if current_name else "")
                self.entries[p_id] = var
                ent = tk.Entry(input_container, textvariable=var, font=FONT_BOLD,
                               bg=INPUT_BG, fg=TEXT, insertbackground=TEXT, relief="flat",
                               highlightbackground=BORDER, highlightthickness=1)
                ent.pack(fill="x", pady=5)

            conn.close()
        except Exception as e:
            tk.Label(self.scroll_frame, text=f"Error accessing Database: {e}", fg=RED_LIT, bg=LOG_BG).pack()

    def _save_names(self):
        renamed = 0
        errors = 0
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for p_id, var in self.entries.items():
                new_name = var.get().strip()
                if not new_name:
                    continue
                    
                safe_name = "".join(c for c in new_name if c.isalnum() or c in " _-")
                if not safe_name:
                    continue
                    
                cursor.execute("UPDATE people SET custom_name = ? WHERE id = ?", (safe_name, p_id))
                renamed += 1
                
            conn.commit()
            conn.close()
            
            messagebox.showinfo("Success", f"Successfully saved {renamed} names to database!\nRe-run Step 4 to rebuild physical folders instantly.")
        except Exception as e:
            messagebox.showerror("Error", f"Database error: {e}")
            errors += 1
            
        self.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoAIApp(root)
    root.mainloop()
