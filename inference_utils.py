import json
import os

import config


DEFAULT_INFERENCE_SETTINGS = {
    "top_k_groups": 5,
    "top_k_classes": 5,
    "temperature": 1.0,
    "group_score_power": getattr(config, "GROUP_SCORE_POWER", 1.0),
}

INFERENCE_SETTINGS_PATH = getattr(
    config,
    "INFERENCE_SETTINGS_PATH",
    os.path.join(os.path.dirname(__file__), "inference_settings.json"),
)


def load_inference_settings(path=INFERENCE_SETTINGS_PATH):
    settings = dict(DEFAULT_INFERENCE_SETTINGS)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        settings.update(saved)
    return settings


def save_inference_settings(settings, path=INFERENCE_SETTINGS_PATH):
    merged = dict(DEFAULT_INFERENCE_SETTINGS)
    merged.update(settings)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return merged
