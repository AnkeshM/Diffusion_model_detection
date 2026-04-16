import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix

# Import robust pipeline modules
from preprocessing import pipeline_preprocess_train, load_image
from feature_extraction import extract_features_from_multiscale
from model import train_model, save_model_package

def get_image_files(directory):
    path = Path(directory)
    # We fetch typical formats plus .TIF/.tif
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    files = []
    for f in path.rglob("*"):
        if f.suffix.lower() in extensions and f.is_file():
            files.append(f)
    return files

def extract_dataset_features(real_dir, fake_dir):
    """
    Iterates over real and fake directories, extracting robust features.
    applies on-the-fly random real-world distortions.
    """
    print(f"Scanning for REAL images in {real_dir} ...")
    real_files = get_image_files(real_dir)
    print(f"Found {len(real_files)} REAL images.")
    
    print(f"Scanning for FAKE images in {fake_dir} ...")
    fake_files = get_image_files(fake_dir)
    print(f"Found {len(fake_files)} FAKE images.")
    
    all_files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
    
    X, y = [], []
    
    print("\nExtracting features and applying stochastic augmentations...")
    for file_path, label in tqdm(all_files, desc="Processing Images"):
        try:
            # 1. Load image natively
            img = load_image(str(file_path))
            
            # 2. Preprocess (applies random augmentations implicitly, multiscales generation)
            multiscales = pipeline_preprocess_train(img, apply_distortions=True)
            
            # 3. Extract robust multi-scale features
            features = extract_features_from_multiscale(multiscales)
            
            X.append(features)
            y.append(label)
        except Exception as e:
            # Skip corrupted/unreadable files without crashing
            print(f"Error processing {file_path}: {e}")
            
    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser(description="Train Robust Synthbuster Model.")
    parser.add_argument("--real_dir", type=str, default=r"d:\IP_Proj\Images", help="Path to real images directory")
    parser.add_argument("--fake_dir", type=str, default=r"d:\IP_Proj\synthbuster", help="Path to fake images directory")
    parser.add_argument("--output", type=str, default=r"d:\IP_Proj\src\robust_pipeline\robust_synthbuster.pkl", help="Output model path")
    args = parser.parse_args()
    
    print("=== Robust Synthbuster Pipeline Training ===")
    
    # Extract features
    X, y = extract_dataset_features(args.real_dir, args.fake_dir)
    
    if len(X) == 0:
        print("No valid images found or processed. Aborting training.")
        sys.exit(1)
        
    print(f"\nFeature matrix built: {X.shape[0]} samples, {X.shape[1]} features.")
    
    # 70/30 Train/Validation Split standardizing class distributions
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # 6. Feature Scaling & Filtering
    print("\nApplying VarianceThreshold and StandardScaler...")
    # FFT magnitude features are typically extremely small floats. A threshold of 0.01 was destroying the dataset.
    selector = VarianceThreshold(threshold=1e-6)
    X_train_sel = selector.fit_transform(X_train)
    X_val_sel = selector.transform(X_val)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)
    
    # 7. Model Training
    # class_weight='balanced' natively handles the 1:9 imbalance during LightGBM init
    print("\nStarting LightGBM Model Training (GPU execution configured)...")
    model = train_model(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val)
    
    # Validation Evaluation
    print("\n=== Evaluating on Validation Set ===")
    y_pred_probs = model.predict_proba(X_val_scaled)[:, 1]
    
    # Optimization Loop: Find Optimal Threshold maximizing Macro F1
    best_thresh = 0.5
    best_macro_f1 = 0.0
    
    for thresh in np.arange(0.05, 0.95, 0.05):
        preds = (y_pred_probs >= thresh).astype(int)
        f1 = f1_score(y_val, preds, average="macro")
        if f1 > best_macro_f1:
            best_macro_f1 = f1
            best_thresh = thresh
            
    print(f"\n[OPTIMIZATION] Found Optimal Classification Threshold: {best_thresh:.3f} (Macro F1: {best_macro_f1:.4f})")
    
    # Evaluate firmly on Optimal Threshold
    y_pred_opt = (y_pred_probs >= best_thresh).astype(int)
    
    cm = confusion_matrix(y_val, y_pred_opt)
    # cm format: [ [TN, FP], [FN, TP] ] -> TN=REAL correctly predicted, FN=FAKE predicted as REAL (wait, FAKE=1 so FN is FAKE predicted as REAL. TP is FAKE predicted correctly)
    real_recall = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    fake_recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    
    print("\n[CONFUSION MATRIX]")
    print(f"Predicted REAL | Predicted FAKE")
    print(f"Actual REAL: {cm[0,0]:<14} {cm[0,1]}")
    print(f"Actual FAKE: {cm[1,0]:<14} {cm[1,1]}")
    
    print(f"\n[CRITICAL METRICS] -> Real Class Recall:   {real_recall:.4f}")
    print(f"[CRITICAL METRICS] -> Fake Class Recall:   {fake_recall:.4f}")
    
    auc_score = roc_auc_score(y_val, y_pred_probs)
    print(f"[CRITICAL METRICS] -> ROC-AUC Score:       {auc_score:.4f}")
    
    # Package into Dictionary
    package = {
        "model": model,
        "scaler": scaler,
        "selector": selector,
        "optimal_threshold": best_thresh
    }
    
    # Save Model
    save_model_package(package, args.output)

if __name__ == "__main__":
    main()
