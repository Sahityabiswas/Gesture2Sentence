# config.py - Tuned for ~2000 classes, Windows-compatible

import os


ARTIFACT_DIR = os.getenv("SLD_ARTIFACT_DIR", ".")


def _artifact_path(*parts):
    return os.path.join(ARTIFACT_DIR, *parts)


# Data paths
CLASS_MAP_PATH = "class_map_FDMSE-ISL.csv"
DATA_MAP_PATH = "data_map_FDMSE-ISL_keypoints.pkl"
VID_CLASS_PATH = "vid_class_FDMSE-ISL.pkl"
VID_SPLITS_PATH = "vid_splits_FDMSE-ISL.pkl"

# Output artifacts
STATS_PATH = _artifact_path("global_stats.pkl")
LABEL_MAP_PATH = _artifact_path("label_map.pkl")
GROUP_MAP_PATH = _artifact_path("group_map.pkl")
GROUP_MODEL_PATH = _artifact_path("group_model.pt")
SUBMODEL_DIR = _artifact_path("submodels")
INFERENCE_SETTINGS_PATH = _artifact_path("inference_settings.json")

# Data settings
MAX_CLASSES = None
VAL_SPLIT = 0.15
SEED = int(os.getenv("SLD_SEED", "42"))

# Grouping
NUM_GROUPS = 72

# Model architecture
USE_TRANSFORMER = False
HIDDEN_SIZE = 384
NUM_LAYERS = 4
DROPOUT = 0.2

# Training
BATCH_SIZE = 64
EPOCHS = 120
PATIENCE = 20
LR = 7e-4
WEIGHT_DECAY = 5e-5
LABEL_SMOOTH = 0.05
GRAD_CLIP = 1.0

# Speed settings
NUM_WORKERS = 0
PIN_MEMORY = True

# Augmentation settings (applied to training sequences only)
AUGMENT_TRAIN = True
AUG_NOISE_STD = 0.01
AUG_FRAME_DROP_PROB = 0.05
AUG_TIME_MASK_PROB = 0.20
AUG_TIME_MASK_RATIO = 0.08
AUG_SCALE_JITTER = 0.08
AUG_SHIFT_STD = 0.015

# Inference re-ranking settings
GROUP_SCORE_POWER = 1.0
