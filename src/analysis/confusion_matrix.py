import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR

# ===============================
# LOAD DATA
# ===============================
X_test = np.load(PROCESSED_DATA_DIR / "X_test.npy")
y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")
sources_test = np.load(PROCESSED_DATA_DIR / "sources_test.npy")

# Load trained model outputs
# Recompute predictions the same way as train_and_evaluate.py
X_train = np.load(PROCESSED_DATA_DIR / "X_train.npy")
y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")

model = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=300,
    random_state=42
)

model.fit(X_train, y_train)

scores = model.decision_function(X_test)
y_pred = (scores > 0).astype(int)

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n=== CONFUSION MATRIX ===")
print("          Predicted")
print("          Real  Fake")
print(f"Actual Real  {tn:5d} {fp:5d}")
print(f"Actual Fake  {fn:5d} {tp:5d}")

# ===============================
# DERIVED METRICS
# ===============================
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / Sensitivity
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

print("\n=== DERIVED RATES ===")
print(f"TPR (Recall, Sensitivity) : {tpr:.4f}")
print(f"FPR (False Positive Rate) : {fpr:.4f}")
print(f"TNR (Specificity)         : {tnr:.4f}")
print(f"FNR (False Negative Rate) : {fnr:.4f}")

# ===============================
# CLASSIFICATION REPORT
# ===============================
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Real", "Fake"],
    digits=4
))
