import csv
import random
import re
import sys
from pathlib import Path
from collections import defaultdict

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATASET_INDEX_CSV, TRAIN_CSV, TEST_CSV

# ===============================
# CONFIG
# ===============================
INPUT_CSV = DATASET_INDEX_CSV
TRAIN_RATIO = 0.7
RANDOM_SEED = 42

# ===============================
# HELPER: extract index from filename
# ===============================
def extract_index(path):
    """
    Extract the filename without extension as the index.
    Example:
        real/r000da54ft.TIF -> r000da54ft
        fake/dalle2/r000da54ft.png -> r000da54ft
    """
    # Use forward slash for consistency as per build_dataset_index.py
    filename = path.split("/")[-1]
    index = filename.split(".")[0]
    return index


random.seed(RANDOM_SEED)

rows = []
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["label"] = int(row["label"])
        rows.append(row)

print(f"[INFO] Loaded {len(rows)} total samples")

index_groups = defaultdict(list)

for row in rows:
    idx = extract_index(row["path"])
    index_groups[idx].append(row)

indices = sorted(index_groups.keys())
print(f"[INFO] Found {len(indices)} unique indices")

random.shuffle(indices)
split_idx = int(len(indices) * TRAIN_RATIO)

train_indices = set(indices[:split_idx])
test_indices  = set(indices[split_idx:])

print(f"[SPLIT] Indices → Train: {len(train_indices)}, Test: {len(test_indices)}")

train_set = []
test_set = []

for idx, samples in index_groups.items():
    if idx in train_indices:
        train_set.extend(samples)
    else:
        test_set.extend(samples)

random.shuffle(train_set)
random.shuffle(test_set)

print(f"\n[FINAL]")
print(f"  Train samples: {len(train_set)}")
print(f"  Test samples : {len(test_set)}")

fieldnames = ["path", "label", "source"]

with open(TRAIN_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(train_set)

with open(TEST_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(test_set)

print("\n[SUCCESS]")
print(f"  -> {TRAIN_CSV}")
print(f"  -> {TEST_CSV}")
