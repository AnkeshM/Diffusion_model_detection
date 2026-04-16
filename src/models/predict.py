import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

# Add root to sys.path to allow config/utils import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import RESULTS_DIR
from utils.feature_extraction import extract_features
from network import DetectorNet

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = RESULTS_DIR / "robust_model.pth"

def predict(image_path):
    """
    Run inference on a single image.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model weights not found at {MODEL_PATH}")
        print("Please run training first: python src/models/train_torch.py")
        return

    print(f"[INFO] Loading model from {MODEL_PATH}...")
    model = DetectorNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Process Image
    print(f"[INFO] Processing image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img_np = np.asarray(img, dtype=np.float32)
    
    # Extract features (135 dimensions)
    features = extract_features(img_np)
    features_tensor = torch.from_numpy(features).unsqueeze(0).to(DEVICE)

    # 3. Predict
    with torch.no_grad():
        logits = model(features_tensor)
        prob = torch.sigmoid(logits).item()
        prediction = "FAKE" if prob > 0.5 else "REAL"

    # 4. Output
    print("\n" + "="*30)
    print(f"IMAGE      : {os.path.basename(image_path)}")
    print(f"PREDICTION : {prediction}")
    print(f"CONFIDENCE : {prob if prediction == 'FAKE' else (1-prob):.4%}")
    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Image Detector Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to test")
    args = parser.parse_args()

    predict(args.image)
