import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, RESULTS_DIR

# ===============================
# LOAD DATA
# ===============================
X_train = np.load(PROCESSED_DATA_DIR / "X_train.npy")
y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")
src_train = np.load(PROCESSED_DATA_DIR / "sources_train.npy")

X_test = np.load(PROCESSED_DATA_DIR / "X_test.npy")
y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")
src_test = np.load(PROCESSED_DATA_DIR / "sources_test.npy")

generators = sorted(set(src_train) - {"real"})

results = {}

for gen in generators:
    print(f"[SPECIFIC] {gen}")

    train_idx = (src_train == "real") | (src_train == gen)
    test_idx  = (src_test  == "real") | (src_test  == gen)

    Xtr, ytr = X_train[train_idx], y_train[train_idx]
    Xte, yte = X_test[test_idx], y_test[test_idx]

    model = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=300, random_state=42
    )
    model.fit(Xtr, ytr)

    scores = model.decision_function(Xte)
    ypred = (scores > 0).astype(int)

    mcc = matthews_corrcoef(yte, ypred)
    results[gen] = mcc

    print(f"  MCC = {mcc:.4f}")

OUTPUT_FILE = RESULTS_DIR / "table1_specific.npy"
np.save(OUTPUT_FILE, results)
print(f"\nSaved → {OUTPUT_FILE}")
