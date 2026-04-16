import os
from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results directory
RESULTS_DIR = ROOT_DIR / "results"

# Image folders
REAL_IMAGES_DIR = RAW_DATA_DIR / "real"
FAKE_IMAGES_DIR = RAW_DATA_DIR / "fake"

# Common file names
DATASET_INDEX_CSV = RAW_DATA_DIR / "dataset_index.csv"
TRAIN_CSV = PROCESSED_DATA_DIR / "train.csv"
TEST_CSV = PROCESSED_DATA_DIR / "test.csv"

# Global function to ensure directories (already handled by script but good for robustness)
def ensure_dirs():
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
