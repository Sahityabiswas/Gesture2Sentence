import os


WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKSPACE_DIR)
ARTIFACT_DIR = os.path.join(WORKSPACE_DIR, "artifacts")


DATA_MAP_PATH = os.path.join(PROJECT_ROOT, "data_map_FDMSE-ISL_keypoints.pkl")
VID_CLASS_PATH = os.path.join(PROJECT_ROOT, "vid_class_FDMSE-ISL.pkl")
VID_SPLITS_PATH = os.path.join(PROJECT_ROOT, "vid_splits_FDMSE-ISL.pkl")
CLASS_MAP_PATH = os.path.join(PROJECT_ROOT, "class_map_FDMSE-ISL.csv")


LABEL_MAP_PATH = os.path.join(ARTIFACT_DIR, "label_map.pkl")
STATS_PATH = os.path.join(ARTIFACT_DIR, "global_stats.pkl")
BEST_MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.pt")
LAST_MODEL_PATH = os.path.join(ARTIFACT_DIR, "last_model.pt")
HISTORY_PATH = os.path.join(ARTIFACT_DIR, "history.json")


SEED = 42
VAL_SPLIT = 0.15
MAX_CLASSES = None


NUM_KEYPOINTS = 29
INPUT_CHANNELS = 2
BASE_CHANNELS = 64
NUM_STAGES = 4
DROPOUT = 0.2


BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


NUM_WORKERS = 0
PIN_MEMORY = True


FRAME_DROP_PROB = 0.05
COORD_NOISE_STD = 0.01
