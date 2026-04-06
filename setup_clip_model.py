"""
setup_clip_model.py — One-time CLIP model export to ONNX format.

Run this ONCE before using 3_classify_images.py.
It downloads the CLIP ViT-B/32 model from Hugging Face and exports it
to ONNX format for use with DirectML (AMD GPU acceleration).

Requirements (one-time):
    pip install optimum[onnxruntime] transformers torch Pillow

After export, you can optionally uninstall torch to reclaim ~2.5GB:
    pip uninstall torch
"""

import os
import sys
import subprocess
import json

# ---- Load overrides from pipeline_config.json ----
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_config.json")
CLIP_MODEL_SIZE = "base"
if os.path.exists(_cfg_path):
    with open(_cfg_path, 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    CLIP_MODEL_SIZE = _cfg.get("clip_model_size", "base")

if CLIP_MODEL_SIZE == "large":
    # Google SigLIP 2 SO400M - State-of-the-Art Vision-Language model
    MODEL_ID = "google/siglip2-so400m-patch14-384"
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "siglip2-so400m-patch14-384-onnx")
else:
    MODEL_ID = "openai/clip-vit-base-patch32"
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "clip-vit-base-patch32-onnx")

def export_model():
    """Export CLIP to ONNX using optimum-cli."""
    print("=" * 60)
    print("  CLIP Model Export to ONNX")
    print("=" * 60)
    print(f"\n  Model:  {MODEL_ID}")
    print(f"  Output: {OUTPUT_DIR}\n")

    if os.path.exists(os.path.join(OUTPUT_DIR, "model.onnx")):
        print("[OK] ONNX model already exists. Skipping export.")
        print(f"     Location: {OUTPUT_DIR}")
        verify_model()
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/2] Exporting CLIP to ONNX format...")
    print("      This downloads ~600MB and may take a few minutes.\n")

    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", MODEL_ID,
        "--task", "zero-shot-image-classification",
        OUTPUT_DIR
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n[OK] Export completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Export failed with code {e.returncode}")
        print("Make sure you have installed: pip install optimum[onnxruntime] transformers torch")
        sys.exit(1)

    verify_model()


def verify_model():
    """Verify the exported model loads with DirectML."""
    print("\n[2/2] Verifying ONNX model with DirectML...")

    try:
        import onnxruntime as ort

        # Check available providers
        available = ort.get_available_providers()
        print(f"      Available providers: {available}")

        if 'DmlExecutionProvider' in available:
            print("      [OK] DirectML (AMD GPU) is available!")
        else:
            print("      [WARN] DirectML not found. Will use CPU.")
            print("      Install: pip install onnxruntime-directml")

        # Try loading the model
        model_path = os.path.join(OUTPUT_DIR, "model.onnx")
        if os.path.exists(model_path):
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            providers = [p for p in providers if p in available]
            session = ort.InferenceSession(model_path, providers=providers)
            active = session.get_providers()
            print(f"      Active providers: {active}")
            print("      [OK] Model loaded successfully!")
        else:
            # optimum may export as different filenames
            onnx_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.onnx')]
            print(f"      ONNX files found: {onnx_files}")

    except ImportError:
        print("      [WARN] onnxruntime not installed. Skipping verification.")
        print("      Install: pip install onnxruntime-directml")

    print("\n" + "=" * 60)
    print("  Setup complete! You can now run 3_classify_images.py")
    print("=" * 60)


if __name__ == "__main__":
    export_model()
