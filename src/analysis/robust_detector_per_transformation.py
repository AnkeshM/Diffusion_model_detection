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

# ===============================
# CONFIG
# ===============================
OUTPUT_JSON = RESULTS_DIR / "robust_per_transformation_results.json"

random.seed(42)
np.random.seed(42)

def load_image(path):
    return Image.open(path).convert("RGB")

def resize(img, scale):
    w, h = img.size
    return img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

def center_crop(img, ratio=0.5):
    w, h = img.size
    cw, ch = int(w * ratio), int(h * ratio)
    left = (w - cw) // 2
    top = (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))

def blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius))

def noise(img, sigma):
    arr = np.asarray(img, dtype=np.float32)
    n = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + n, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def augment(img):
    # mixed distortions for robust training
    if random.random() < 0.5:
        img = resize(img, random.choice([0.5, 0.75, 1.5]))
    if random.random() < 0.5:
        img = center_crop(img, 0.5)
    if random.random() < 0.5:
        img = blur(img, random.choice([1, 2, 3]))
    if random.random() < 0.5:
        img = noise(img, random.choice([5, 10, 20]))
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
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            rows.append(row)
    return rows

train_rows = load_rows(TRAIN_CSV)
test_rows = load_rows(TEST_CSV)

print("\n[INFO] Extracting augmented training features...")
X_train, y_train = [], []

for row in tqdm(train_rows):
    img_path = RAW_DATA_DIR / row["path"]
    img = load_image(img_path)
    img = augment(img)
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

def evaluate(name, transform_fn):
    X, y = [], []

    for row in tqdm(test_rows, desc=name):
        img_path = RAW_DATA_DIR / row["path"]
        img = load_image(img_path)
        img = transform_fn(img)
        feats = extract_features(img)
        X.append(feats)
        y.append(row["label"])

    X = np.asarray(X, dtype=np.float32)
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

results = {}

# Clean
results["clean"] = evaluate("clean", lambda x: x)

# Resizing
for s in [0.5, 0.75, 1.5, 2.0]:
    results[f"resize_{s}"] = evaluate(
        f"resize_{s}",
        lambda img, s=s: resize(img, s)
    )

# Cropping
results["crop_center_50"] = evaluate(
    "crop_center_50",
    lambda img: center_crop(img, 0.5)
)

# Blur
for b in [1, 2, 3]:
    results[f"blur_{b}"] = evaluate(
        f"blur_{b}",
        lambda img, b=b: blur(img, b)
    )

# Noise
for n in [5, 10, 20]:
    results[f"noise_{n}"] = evaluate(
        f"noise_{n}",
        lambda img, n=n: noise(img, n)
    )

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=4)

print("\n[SUCCESS] Results saved to", OUTPUT_JSON)
