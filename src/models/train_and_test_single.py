import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, RESULTS_DIR

X_train = np.load(PROCESSED_DATA_DIR / "X_train.npy")
y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")

print(f"[INFO] Train shape: {X_train.shape}")

# single_image_features.npy is a result of extract_single_feature.py, so it goes in results or processed.
# Given it's a test feature, results/ seems appropriate for one-offs.
X_single = np.load(RESULTS_DIR / "single_image_features.npy")

print(f"[INFO] Single test shape: {X_single.shape}")

model = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.05,
    max_iter=300,
    random_state=42
)

model.fit(X_train, y_train)

score = model.decision_function(X_single)[0]
prediction = int(score > 0)

print("\n=== SINGLE IMAGE RESULT ===")
# print(f"Score      : {score:.4f}")

if prediction == 1:
    print("Prediction : FAKE")
else:
    print("Prediction : REAL")
