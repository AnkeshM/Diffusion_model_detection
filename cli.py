import os
import csv
import sys
import numpy as np
import random
from pathlib import Path
from PIL import Image
from collections import defaultdict
import click

# Add root to sys.path to allow config/utils import
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

try:
    from src.config import DATASET_INDEX_CSV, TRAIN_CSV, TEST_CSV, RAW_DATA_DIR, RESULTS_DIR
    from src.utils.feature_extraction import extract_features
    from src.utils.augmentation import augment_image
except ImportError as e:
    print(f"Import Error: {e}")
    from config import DATASET_INDEX_CSV, TRAIN_CSV, TEST_CSV, RAW_DATA_DIR, RESULTS_DIR
    from utils.feature_extraction import extract_features
    from utils.augmentation import augment_image

# ===============================
# CONFIG & HELPERS
# ===============================
def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = RESULTS_DIR / "robust_model.pth"
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.pth"

# ===============================
# DATASET CLASS
# ===============================
class ImageFeatureDataset:
    def __init__(self, csv_path, root_dir, augment=False):
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.rows = []
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
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
        
        try:
            img = Image.open(img_path).convert("RGB")
            
            # Resize BEFORE augmentation for performance (1024x1024)
            img = img.resize((1024, 1024), Image.BICUBIC)
            
            if self.augment:
                img = augment_image(img)
                # Resize BACK to 1024x1024 after augmentation so FFT bins are consistent
                if img.size != (1024, 1024):
                    img = img.resize((1024, 1024), Image.BICUBIC)
            
            import torch
            img_np = np.asarray(img, dtype=np.float32)
            features = extract_features(img_np)
            return torch.from_numpy(features), torch.tensor([row["label"]], dtype=torch.float32)

        except Exception as e:
            import torch
            # Return a dummy if image fails to load
            print(f"\n[WARNING] Error loading image {img_path}: {e}")
            return torch.zeros(147), torch.tensor([row["label"]], dtype=torch.float32)


@click.group()
def cli():
    """Unified AI Image Detection CLI"""
    pass

@cli.command()
@click.option('--ratio', default=0.8, help="Split ratio for training (default: 0.8)")
@click.option('--seed', default=42, help="Random seed for reproducibility")
def split(ratio, seed):
    """Split the dataset index into training and testing sets."""
    random.seed(seed)
    
    if not os.path.exists(DATASET_INDEX_CSV):
        click.echo(f"[ERROR] {DATASET_INDEX_CSV} not found.")
        return

    rows = []
    with open(DATASET_INDEX_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    click.echo(f"[INFO] Loaded {len(rows)} samples.")

    # Group by index to avoid leakage
    def get_index(path):
        return os.path.basename(path).split(".")[0]

    groups = defaultdict(list)
    for row in rows:
        groups[get_index(row["path"])].append(row)

    indices = list(groups.keys())
    random.shuffle(indices)
    
    split_idx = int(len(indices) * ratio)
    train_indices = set(indices[:split_idx])
    
    train_set, test_set = [], []
    for idx, samples in groups.items():
        if idx in train_indices:
            train_set.extend(samples)
        else:
            test_set.extend(samples)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    with open(TRAIN_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source"])
        writer.writeheader()
        writer.writerows(train_set)

    with open(TEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source"])
        writer.writeheader()
        writer.writerows(test_set)

    click.echo(f"[SUCCESS] Split into Train: {len(train_set)}, Test: {len(test_set)}")

@cli.command()
@click.option('--epochs', default=50, help="Number of training epochs")
@click.option('--batch-size', default=64, help="Batch size")
@click.option('--lr', default=0.0005, help="Learning rate")
@click.option('--resume', is_flag=True, help="Resume from last checkpoint")
@click.option('--workers', default=4, help="Number of data loading workers")
def train(epochs, batch_size, lr, resume, workers):
    """Train the DetectorNet model with robust augmentations."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from tqdm import tqdm
    from sklearn.metrics import matthews_corrcoef
    try:
        from src.models.network import DetectorNet
    except ImportError:
        from models.network import DetectorNet

    device = get_device()
    click.echo(f"[INFO] Using device: {device}")
    click.echo(f"[INFO] Data loading workers: {workers}")
    
    # Check if files exist
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        click.echo("[ERROR] Train/Test CSVs missing. Please run 'split' first.")
        return

    train_dataset = ImageFeatureDataset(TRAIN_CSV, RAW_DATA_DIR, augment=True)
    test_dataset = ImageFeatureDataset(TEST_CSV, RAW_DATA_DIR, augment=False)
    
    # Compute class distribution
    num_pos = sum(1 for r in train_dataset.rows if r["label"] == 1)
    num_neg = sum(1 for r in train_dataset.rows if r["label"] == 0)
    total = num_pos + num_neg
    click.echo(f"[INFO] Train labels: {num_neg} real (0), {num_pos} fake (1)")
    click.echo(f"[INFO] Test samples: {len(test_dataset)}")

    # WeightedRandomSampler for balanced batches (more stable than pos_weight)
    class_weights = {0: total / (2.0 * num_neg), 1: total / (2.0 * num_pos)}
    sample_weights = [class_weights[r["label"]] for r in train_dataset.rows]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    click.echo(f"[INFO] Balanced sampling: w0={class_weights[0]:.3f}, w1={class_weights[1]:.3f}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers)

    model = DetectorNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    start_epoch = 0
    best_mcc = -1.0
    patience = 10
    epochs_without_improvement = 0

    if resume and os.path.exists(CHECKPOINT_PATH):
        click.echo(f"[INFO] Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_mcc = checkpoint.get('best_mcc', checkpoint['val_mcc'])
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

    click.echo(f"\n[START] Training for {epochs} epochs (LR={lr})...")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = train_loss / max(batch_count, 1)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Evaluation after each epoch
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                outputs = torch.sigmoid(model(features))
                preds = (outputs > 0.5).int().cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy().flatten())

        # Diagnostics: prediction distribution
        pred_0 = sum(1 for p in all_preds if p == 0)
        pred_1 = sum(1 for p in all_preds if p == 1)
        label_0 = sum(1 for l in all_labels if l == 0)
        label_1 = sum(1 for l in all_labels if l == 1)

        epoch_mcc = matthews_corrcoef(all_labels, all_preds)
        click.echo(f"  [RESULT] Loss: {avg_loss:.4f} | MCC: {epoch_mcc:.4f} | LR: {current_lr:.6f} | Preds(0/1): {pred_0}/{pred_1} | Labels(0/1): {label_0}/{label_1}")

        # Checkpoint
        is_best = epoch_mcc > best_mcc
        if is_best:
            best_mcc = epoch_mcc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            click.echo(f"  --> Best model saved to {MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            click.echo(f"  --> No improvement for {epochs_without_improvement} epoch(s)")

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_mcc': epoch_mcc,
            'best_mcc': best_mcc,
            'epochs_without_improvement': epochs_without_improvement
        }, CHECKPOINT_PATH)

        if epochs_without_improvement >= patience:
            click.echo(f"\n[INFO] Early stopping triggered after {patience} epochs without improvement.")
            break

    click.echo("\n[SUCCESS] Training complete.")

@cli.command()
@click.option('--workers', default=4, help="Number of data loading workers")
@click.option('--batch-size', default=64, help="Batch size for evaluation")
def evaluate(workers, batch_size):
    """Evaluate the trained model on the test set with per-augmentation breakdown."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix, accuracy_score
    from PIL import ImageFilter
    try:
        from src.models.network import DetectorNet
    except ImportError:
        from models.network import DetectorNet

    device = get_device()
    click.echo(f"[INFO] Evaluating on device: {device}")
    click.echo(f"[INFO] Data loading workers: {workers}")
    
    if not os.path.exists(MODEL_PATH):
        click.echo(f"[ERROR] Model not found at {MODEL_PATH}")
        return

    model = DetectorNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load test CSV rows
    test_rows = []
    with open(TEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            test_rows.append(row)

    click.echo(f"[INFO] Test samples: {len(test_rows)}")

    # Define augmentation variants for evaluation
    STANDARD_SIZE = (1024, 1024)
    augmentation_variants = {
        "Normal": lambda img: img,
        "Blurred": lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)),
        "Noised": lambda img: Image.fromarray(
            np.clip(
                np.asarray(img, dtype=np.float32) + np.random.normal(0, 15, np.asarray(img).shape),
                0, 255
            ).astype(np.uint8)
        ),
        "Resized": lambda img: img.resize(
            (512, 512), Image.BICUBIC
        ).resize(STANDARD_SIZE, Image.BICUBIC),
    }

    def compute_metrics(labels, preds):
        """Compute metrics from flat lists."""
        mcc = matthews_corrcoef(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        return {
            "acc": acc, "mcc": mcc, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "fpr": fpr, "fnr": fnr
        }

    def evaluate_variant(variant_name, transform_fn):
        """Evaluate model on test set with a specific augmentation applied."""
        all_preds, all_labels = [], []

        pbar = tqdm(test_rows, desc=f"  {variant_name}")
        for row in pbar:
            img_path = Path(RAW_DATA_DIR) / row["path"]
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(STANDARD_SIZE, Image.BICUBIC)
                img = transform_fn(img)
                # Ensure consistent size after transform
                if img.size != STANDARD_SIZE:
                    img = img.resize(STANDARD_SIZE, Image.BICUBIC)

                img_np = np.asarray(img, dtype=np.float32)
                features = extract_features(img_np)
                features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)

                with torch.no_grad():
                    logit = model(features_tensor)
                    prob = torch.sigmoid(logit).item()
                    pred = 1 if prob > 0.5 else 0

                all_preds.append(pred)
                all_labels.append(row["label"])
            except Exception as e:
                click.echo(f"\n    [WARNING] Error on {img_path}: {e}")
                continue

        return compute_metrics(all_labels, all_preds)

    # ---- Run per-augmentation evaluation ----
    results = {}
    for variant_name, transform_fn in augmentation_variants.items():
        click.echo(f"\n[EVAL] Running evaluation: {variant_name}")
        results[variant_name] = evaluate_variant(variant_name, transform_fn)

    # ---- Print results ----
    click.echo("\n" + "=" * 70)
    click.echo("                    EVALUATION RESULTS")
    click.echo("=" * 70)

    # Header
    click.echo(f"{'Variant':<12} {'Acc':>8} {'MCC':>8} {'F1':>8} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6} {'FPR':>8} {'FNR':>8}")
    click.echo("-" * 70)

    for name, m in results.items():
        click.echo(
            f"{name:<12} {m['acc']:>8.4f} {m['mcc']:>8.4f} {m['f1']:>8.4f} "
            f"{m['tp']:>6} {m['tn']:>6} {m['fp']:>6} {m['fn']:>6} "
            f"{m['fpr']:>8.4f} {m['fnr']:>8.4f}"
        )

    click.echo("=" * 70)

@cli.command()
@click.option('--image', required=True, type=click.Path(exists=True), help="Path to image for prediction")
def predict(image):
    """Predict label for a single image."""
    import torch
    try:
        from src.models.network import DetectorNet
    except ImportError:
        from models.network import DetectorNet

    device = get_device()
    if not os.path.exists(MODEL_PATH):
        click.echo("[ERROR] Model weights not found. Please train first.")
        return

    model = DetectorNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    try:
        img = Image.open(image).convert("RGB")
        img_np = np.asarray(img, dtype=np.float32)
        features = extract_features(img_np)
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(features_tensor)
            prob = torch.sigmoid(logit).item()
            prediction = "AI-GENERATED (FAKE)" if prob > 0.5 else "REAL"

        click.echo("\n" + "="*40)
        click.echo(f"Image File : {os.path.basename(image)}")
        click.echo(f"Result     : {prediction}")
        click.echo(f"Confidence : {prob if prob > 0.5 else (1-prob):.4%}")
        click.echo("="*40 + "\n")

    except Exception as e:
        click.echo(f"[ERROR] Failed to process image: {e}")

if __name__ == '__main__':
    cli()
