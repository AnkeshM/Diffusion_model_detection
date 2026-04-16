import os
import csv
import json
import random
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from tqdm import tqdm
from numpy.fft import fft2, fftshift
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TRAIN_CSV, TEST_CSV, RESULTS_DIR, RAW_DATA_DIR

TRAIN_CSV = TRAIN_CSV
TEST_CSV = TEST_CSV
OUTPUT_JSON = RESULTS_DIR / "robust_detector_results.json"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_image(path):
    return Image.open(path).convert("RGB")

def random_resize(img):
    scale = random.choice([0.5, 0.75, 1.0, 1.5])
    w, h = img.size
    return img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

def random_crop(img):
    if random.random() < 0.5:
        w, h = img.size
        cw, ch = int(w * 0.5), int(h * 0.5)
        left = random.randint(0, w - cw)
        top = random.randint(0, h - ch)
        img = img.crop((left, top, left + cw, top + ch))
    return img

def random_blur(img):
    if random.random() < 0.5:
        radius = random.choice([1, 2, 3])
        img = img.filter(ImageFilter.GaussianBlur(radius))
    return img

def random_noise(img):
    if random.random() < 0.5:
        sigma = random.choice([5, 10, 20])
        arr = np.asarray(img, dtype=np.float32)
        noise = np.random.normal(0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
    return img

def augment_image(img):
    img = random_resize(img)
    img = random_crop(img)
    img = random_blur(img)
    img = random_noise(img)
    return img

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
    features.append(F[cx, cy])

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

def load_dataset(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            rows.append(row)
    return rows

train_rows = load_dataset(TRAIN_CSV)
test_rows = load_dataset(TEST_CSV)

print("\n[INFO] Extracting augmented training features...")
X_train = []
y_train = []

for row in tqdm(train_rows):
    img_path = RAW_DATA_DIR / row["path"]
    img = load_image(img_path)
    img = augment_image(img)
    feats = extract_features(img)

    X_train.append(feats)
    y_train.append(row["label"])

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train)

model = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=300,
    random_state=42
)
model.fit(X_train, y_train)

print("\n[INFO] Evaluating robust detector...")
X_test = []
y_test = []

for row in tqdm(test_rows):
    img_path = RAW_DATA_DIR / row["path"]
    img = load_image(img_path)
    img = augment_image(img)  # same distortions
    feats = extract_features(img)

    X_test.append(feats)
    y_test.append(row["label"])

X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test)

scores = model.decision_function(X_test)
y_pred = (scores > 0).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

results = {
    "MCC": matthews_corrcoef(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "confusion_matrix": {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    },
    "rates": {
        "FPR": fp / (fp + tn),
        "FNR": fn / (fn + tp)
    }
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=4)

print("\n[SUCCESS] Robust detector results saved to", OUTPUT_JSON)
