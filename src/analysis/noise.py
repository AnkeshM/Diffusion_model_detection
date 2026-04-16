import csv
import json
import random
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import TEST_CSV, RESULTS_DIR, RAW_DATA_DIR
from utils.feature_extraction import extract_features, load_image
from models.network import DetectorNet

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = RESULTS_DIR / "robust_model.pth"
OUTPUT_JSON = RESULTS_DIR / "noise_results.json"

random.seed(42)
np.random.seed(42)

def add_noise(img_pil, sigma):
    arr = np.asarray(img_pil, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

# ===============================
# DATA LOADING
# ===============================
def load_rows(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            rows.append(row)
    return rows

test_rows = load_rows(TEST_CSV)

# ===============================
# EVALUATION
# ===============================
def evaluate(model, sigma):
    all_preds = []
    all_labels = []

    for row in tqdm(test_rows, desc=f"noise_{sigma}"):
        img_path = RAW_DATA_DIR / row["path"]
        img = load_image(img_path)
        img_pil = Image.fromarray(img.astype(np.uint8))
        
        # Add Noise
        img_noisy = add_noise(img_pil, sigma)
        img_np = np.asarray(img_noisy, dtype=np.float32)
        
        # Extract features
        feats = extract_features(img_np)
        feats_tensor = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = torch.sigmoid(model(feats_tensor))
            pred = (output > 0.5).int().cpu().item()
            
        all_preds.append(pred)
        all_labels.append(row["label"])

    y = np.array(all_labels)
    y_pred = np.array(all_preds)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    return {
        "MCC": matthews_corrcoef(y, y_pred),
        "F1": f1_score(y, y_pred),
        "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "FNR": fn / (fn + tp) if (fn + tp) > 0 else 0
    }

# ===============================
# MAIN
# ===============================
def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found at {MODEL_PATH}. Please train first.")
        return

    print(f"[INFO] Loading robust model from {MODEL_PATH}...")
    model = DetectorNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    sigmas = [5, 10, 20]

    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            results = json.load(f)
    else:
        results = {}

    for s in sigmas:
        key = f"noise_{s}"
        if key in results:
            print(f"[SKIP] {key} already done.")
            continue

        results[key] = evaluate(model, s)

        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=4)

    print(f"[DONE] Noise experiments complete. Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
