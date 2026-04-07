import json
import os


def load_pipeline_config(base_dir):
    """Load pipeline_config.json from base_dir with safe fallback."""
    cfg_path = os.path.join(base_dir, "pipeline_config.json")
    if not os.path.exists(cfg_path):
        return {}

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        print("[WARN] pipeline_config.json is corrupted; script defaults will be used.")
        return {}
    except Exception as exc:
        print(f"[WARN] Could not load pipeline_config.json: {exc}")
        return {}
