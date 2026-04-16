import numpy as np
import sys
import os
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR

X_train = np.load(PROCESSED_DATA_DIR / "X_train.npy")
y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")

X_test = np.load(PROCESSED_DATA_DIR / "X_test.npy")
y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")
sources_test = np.load(PROCESSED_DATA_DIR / "sources_test.npy")

print(f"[INFO] Train shape: {X_train.shape}")
print(f"[INFO] Test  shape: {X_test.shape}")

model = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=300,
    random_state=42
)

model.fit(X_train, y_train)

scores = model.decision_function(X_test)
y_pred = (scores > 0).astype(int)

mcc = matthews_corrcoef(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, scores)

print("\n=== GLOBAL RESULTS ===")
print(f"MCC : {mcc:.4f}")
print(f"F1  : {f1:.4f}")
print(f"AUC : {auc:.4f}")

print("\n=== PER-GENERATOR MCC ===")

by_source = defaultdict(list)

for i, src in enumerate(sources_test):
    by_source[src].append(i)

for src, idxs in sorted(by_source.items()):
    y_true = y_test[idxs]
    y_hat = y_pred[idxs]

    # skip real-only bucket
    if len(set(y_true)) < 2:
        continue

    src_mcc = matthews_corrcoef(y_true, y_hat)
    print(f"{src:25s} : MCC = {src_mcc:.4f}")
