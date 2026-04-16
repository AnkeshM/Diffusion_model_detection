import os
import csv
import numpy as np
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from numpy.fft import fft2, fftshift

# Add root to sys.path to allow config/utils import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TEST_CSV, PROCESSED_DATA_DIR, RAW_DATA_DIR
from utils.jpeg_utils import jpeg_compress

# ===============================
# CONFIG
# ===============================
CSV_FILE = TEST_CSV
OUTPUT_X = PROCESSED_DATA_DIR / "X_test.npy"
OUTPUT_Y = PROCESSED_DATA_DIR / "y_test.npy"
OUTPUT_SRC = PROCESSED_DATA_DIR / "sources_test.npy"

PERIODS = [0, 2, 4, 8]

def load_image(path, jpeg_quality=None):    # or 95, 90, 80, 70
    img = Image.open(path).convert("RGB")
    if jpeg_quality is not None:
        img = jpeg_compress(img, jpeg_quality)
    return np.asarray(img, dtype=np.float32)

def cross_difference(channel):
    return (
        channel[:-1, :-1]
        + channel[1:, 1:]
        - channel[:-1, 1:]
        - channel[1:, :-1]
    )

def fft_magnitude(cd):
    H, W = cd.shape
    F = fftshift(np.abs(fft2(cd)))
    return F / (H * W)

def extract_peaks(F):
    H, W = F.shape
    cx, cy = H // 2, W // 2

    features = []

    # DC
    features.append(F[cx, cy])

    periods = [2, 4, 8]

    for px in periods:
        dx = H // px

        # horizontal axis
        features.extend([F[cx + dx, cy], F[cx - dx, cy]])

    for py in periods:
        dy = W // py

        # vertical axis
        features.extend([F[cx, cy + dy], F[cx, cy - dy]])

    for px in periods:
        dx = H // px
        for py in periods:
            dy = W // py

            # diagonals
            features.extend([
                F[cx + dx, cy + dy],
                F[cx + dx, cy - dy],
                F[cx - dx, cy + dy],
                F[cx - dx, cy - dy],
            ])

    # 1 + 6 + 6 + 32 = 45
    return features

rows = []
with open(CSV_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["label"] = int(row["label"])
        rows.append(row)

print(f"[INFO] Loaded {len(rows)} test samples")

# ===============================
# FEATURE EXTRACTION
# ===============================
X, y, sources = [], [], []

for row in tqdm(rows, desc="Extracting test features"):
    # Prefix with RAW_DATA_DIR
    img_path = RAW_DATA_DIR / row["path"]
    img = load_image(img_path)
    feats = []

    for c in range(3):
        cd = cross_difference(img[:, :, c])
        F = fft_magnitude(cd)
        feats.extend(extract_peaks(F))

    X.append(feats)
    y.append(row["label"])
    sources.append(row["source"])

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int8)
sources = np.asarray(sources)

np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)
np.save(OUTPUT_SRC, sources)

print("\n[SUCCESS]")
print(f"X_test shape: {X.shape} (should be [N, 135])")
print(f"Saved: {OUTPUT_X}, {OUTPUT_Y}, {OUTPUT_SRC}")
