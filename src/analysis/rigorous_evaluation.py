import json
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, RESULTS_DIR

OUTPUT_JSON = RESULTS_DIR / "evaluation_results.json"
THRESHOLDS = np.linspace(-2.0, 2.0, 81)  # rigorous sweep

X_train = np.load(PROCESSED_DATA_DIR / "X_train.npy")
y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")

X_test = np.load(PROCESSED_DATA_DIR / "X_test.npy")
y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")
sources_test = np.load(PROCESSED_DATA_DIR / "sources_test.npy")

model = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=300,
    random_state=42
)
model.fit(X_train, y_train)

scores = model.decision_function(X_test)

threshold = 0.0
y_pred = (scores > threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

global_results = {
    "threshold": threshold,
    "MCC": matthews_corrcoef(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, scores),
    "accuracy": float((tp + tn) / len(y_test)),
    "confusion_matrix": {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    },
    "rates": {
        "TPR": tp / (tp + fn),
        "FPR": fp / (fp + tn),
        "TNR": tn / (tn + fp),
        "FNR": fn / (fn + tp)
    }
}

per_generator = {}

for src in np.unique(sources_test):
    idx = np.where(sources_test == src)[0]
    y_true = y_test[idx]
    y_hat = y_pred[idx]

    # skip real-only bucket
    if len(np.unique(y_true)) < 2:
        continue

    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()

    per_generator[src] = {
        "samples": int(len(idx)),
        "MCC": matthews_corrcoef(y_true, y_hat),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "rates": {
            "TPR": tp / (tp + fn),
            "FPR": fp / (fp + tn),
            "TNR": tn / (tn + fp),
            "FNR": fn / (fn + tp)
        }
    }

threshold_sweep = []

for t in THRESHOLDS:
    y_hat = (scores > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()

    threshold_sweep.append({
        "threshold": float(t),
        "MCC": matthews_corrcoef(y_test, y_hat),
        "F1": f1_score(y_test, y_hat),
        "FPR": fp / (fp + tn),
        "FNR": fn / (fn + tp)
    })

results = {
    "global_evaluation": global_results,
    "per_generator_evaluation": per_generator,
    "threshold_sweep": threshold_sweep
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"\n[SUCCESS] Rigorous evaluation saved to {OUTPUT_JSON}")
