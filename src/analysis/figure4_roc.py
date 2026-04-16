import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from sklearn.metrics import roc_curve
from sklearn.ensemble import HistGradientBoostingClassifier

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, RESULTS_DIR

GENERATOR = "stable-diffusion-xl"

# ===============================
# LOAD DATA
# ===============================
X_train = np.load(PROCESSED_DATA_DIR / "X_train.npy")
y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")
src_train = np.load(PROCESSED_DATA_DIR / "sources_train.npy")

X_test = np.load(PROCESSED_DATA_DIR / "X_test.npy")
y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")
src_test = np.load(PROCESSED_DATA_DIR / "sources_test.npy")

# Helper
def train_and_score(train_idx, test_idx):
    model = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=300, random_state=42
    )
    model.fit(X_train[train_idx], y_train[train_idx])
    return model.decision_function(X_test[test_idx]), y_test[test_idx]

# ===============================
# GENERIC
# ===============================
generic_train = src_train != ""  # all
generic_test  = (src_test == "real") | (src_test == GENERATOR)

scores_g, y_g = train_and_score(generic_train, generic_test)

# ===============================
# SPECIFIC
# ===============================
specific_train = (src_train == "real") | (src_train == GENERATOR)
specific_test  = (src_test == "real") | (src_test == GENERATOR)

scores_s, y_s = train_and_score(specific_train, specific_test)

# ===============================
# GENERALIZATION
# ===============================
general_train = (src_train == "real") | ((src_train != GENERATOR) & (src_train != "real"))
general_test  = (src_test == "real") | (src_test == GENERATOR)

scores_u, y_u = train_and_score(general_train, general_test)

# ===============================
# ROC CURVES
# ===============================
fpr_g, tpr_g, _ = roc_curve(y_g, scores_g)
fpr_s, tpr_s, _ = roc_curve(y_s, scores_s)
fpr_u, tpr_u, _ = roc_curve(y_u, scores_u)

plt.figure()
plt.plot(fpr_g, tpr_g, label="Generic training")
plt.plot(fpr_s, tpr_s, label="Specific training")
plt.plot(fpr_u, tpr_u, label="Generalization training")
plt.plot([0,1], [0,1], "--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC curves for {GENERATOR}")
plt.legend()
plt.grid(True)

OUTPUT_IMG = RESULTS_DIR / f"figure4_{GENERATOR}.png"
plt.savefig(OUTPUT_IMG)
print(f"\n[SUCCESS] ROC curve saved to {OUTPUT_IMG}")
# plt.show() # Disabled for headless or script execution
