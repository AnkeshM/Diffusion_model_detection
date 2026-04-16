import os
import csv
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, f1_score

# Add root to sys.path to allow config/utils import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TRAIN_CSV, TEST_CSV, RAW_DATA_DIR, RESULTS_DIR
from utils.feature_extraction import extract_features, load_image
from utils.augmentation import augment_image
from network import DetectorNet

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.pth"
MODEL_PATH = RESULTS_DIR / "robust_model.pth"

# (Moved print to main)

# ===============================
# DATASET
# ===============================
class ImageFeatureDataset(Dataset):
    """
    On-the-fly feature extraction dataset.
    Augments images and extracts FFT features for robust training.
    """
    def __init__(self, csv_path, root_dir, augment=False):
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["label"] = int(row["label"])
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = self.root_dir / row["path"]
        
        # Load and potentially augment
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        
        if self.augment:
            img = augment_image(img)
            
        # Extract features (135 dimensions)
        img_np = np.asarray(img, dtype=np.float32)
        features = extract_features(img_np)
        
        return torch.from_numpy(features), torch.tensor([row["label"]], dtype=torch.float32)

# ===============================
# TRAINING UTILS
# ===============================
def save_checkpoint(model, optimizer, epoch, val_mcc, is_best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_mcc': val_mcc
    }
    torch.save(state, CHECKPOINT_PATH)
    if is_best:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  --> Saved best model to {MODEL_PATH}")

def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[INFO] Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'] + 1, checkpoint['val_mcc']
    return 0, -1.0

# ===============================
# MAIN TRAINING LOOP
# ===============================
def train():
    print(f"[INFO] Using device: {DEVICE}")
    # 1. Setup Dataloaders
    train_dataset = ImageFeatureDataset(TRAIN_CSV, RAW_DATA_DIR, augment=True)
    test_dataset = ImageFeatureDataset(TEST_CSV, RAW_DATA_DIR, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 2. Setup Model, Loss, Optimizer
    model = DetectorNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Resume if possible
    start_epoch, best_mcc = load_checkpoint(model, optimizer)

    print(f"\n[START] Training for {EPOCHS} epochs...")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for features, labels in pbar:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(DEVICE)
                outputs = torch.sigmoid(model(features))
                preds = (outputs > 0.5).int().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_mcc = matthews_corrcoef(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        
        print(f"  [RESULT] Val MCC: {val_mcc:.4f} | Val F1: {val_f1:.4f}")

        # Checkpointing
        is_best = val_mcc > best_mcc
        if is_best:
            best_mcc = val_mcc
        
        save_checkpoint(model, optimizer, epoch, val_mcc, is_best=is_best)

    print("\n[SUCCESS] Training complete.")

if __name__ == "__main__":
    train()
