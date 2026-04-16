import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, RESULTS_DIR

JPEG_QUALITIES = ["raw", 95, 90, 80, 70]

results = {}

for q in JPEG_QUALITIES:
    print(f"[JPEG Q={q}]")

    train_file = PROCESSED_DATA_DIR / f"X_train_q{q}.npy"
    test_file = PROCESSED_DATA_DIR / f"X_test_q{q}.npy"
    y_train_file = PROCESSED_DATA_DIR / "y_train.npy"
    y_test_file = PROCESSED_DATA_DIR / "y_test.npy"

    if not train_file.exists() or not test_file.exists():
        print(f"  [SKIP] Files not found for Q={q}")
        continue

    X_train = np.load(train_file)
    y_train = np.load(y_train_file)

    X_test = np.load(test_file)
    y_test = np.load(y_test_file)

    model = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=300, random_state=42
    )
    model.fit(X_train, y_train)

    scores = model.decision_function(X_test)
    ypred = (scores > 0).astype(int)

    mcc = matthews_corrcoef(y_test, ypred)
    results[q] = mcc

    print(f"  MCC = {mcc:.4f}")

OUTPUT_FILE = RESULTS_DIR / "table2_jpeg.npy"
np.save(OUTPUT_FILE, results)
print(f"\nSaved → {OUTPUT_FILE}")
