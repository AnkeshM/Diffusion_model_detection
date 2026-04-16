import csv
import json
import random
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from numpy.fft import fft2, fftshift
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TRAIN_CSV, TEST_CSV, RESULTS_DIR, RAW_DATA_DIR

# ===============================
# CONFIG
# ===============================
OUTPUT_JSON = RESULTS_DIR / "crop_results.json"

random.seed(42)
np.random.seed(42)

def load_image(path):
    return Image.open(path).convert("RGB")

def center_crop(img, ratio):
    w, h = img.size
    cw, ch = int(w * ratio), int(h * ratio)
    left = (w - cw) // 2
    top = (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))

# ---- feature extraction (same as before) ----
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
    features = [F[cx, cy]]
    periods = [2, 4, 8]

    for px in periods:
        dx = H // px
        features.extend([F[cx + dx, cy], F[cx - dx, cy]])

    for py in periods:
        dy = W // py
        features.extend([F[cx, cy + dy], F[cx, cy - dy]])

    for px in periods:
        dx = H // px
        for py in periods:
            dy = W // py
            features.extend([
                F[cx + dx, cy + dy],
                F[cx + dx, cy - dy],
                F[cx - dx, cy + dy],
                F[cx - dx, cy - dy],
            ])
    return features

def extract_features(img):
    arr = np.asarray(img, dtype=np.float32)
    feats = []
    for c in range(3):
        cd = cross_difference(arr[:, :, c])
        F = fft_magnitude(cd)
        feats.extend(extract_peaks(F))
    return feats

def load_rows(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            rows.append(row)
    return rows

train_rows = load_rows(TRAIN_CSV)
test_rows = load_rows(TEST_CSV)

print("[INFO] Training model...")
X_train, y_train = [], []
for row in tqdm(train_rows):
    img_path = RAW_DATA_DIR / row["path"]
    img = load_image(img_path)
    feats = extract_features(img)
    X_train.append(feats)
    y_train.append(row["label"])

model = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=300,
    random_state=42
)
model.fit(np.array(X_train), np.array(y_train))

def evaluate(ratio):
    X, y = [], []
    for row in tqdm(test_rows, desc=f"crop_{ratio}"):
        img_path = RAW_DATA_DIR / row["path"]
        img = load_image(img_path)
        img = center_crop(img, ratio)
        feats = extract_features(img)
        X.append(feats)
        y.append(row["label"])

    X = np.asarray(X)
    y = np.asarray(y)

    scores = model.decision_function(X)
    y_pred = (scores > 0).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    return {
        "MCC": matthews_corrcoef(y, y_pred),
        "F1": f1_score(y, y_pred),
        "FPR": fp / (fp + tn),
        "FNR": fn / (fn + tp)
    }

ratios = [0.5]

if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON) as f:
        results = json.load(f)
else:
    results = {}

for r in ratios:
    key = f"crop_center_{int(r*100)}"
    if key in results:
        print(f"[SKIP] {key} already done.")
        continue

    results[key] = evaluate(r)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

print("[DONE] Crop experiments complete.")
