import os
import csv
import sys
from pathlib import Path

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, DATASET_INDEX_CSV

# ===============================
# CONFIG
# ===============================
ROOT_DIR = RAW_DATA_DIR
REAL_DIR = "real"
FAKE_DIR = "fake"
OUTPUT_CSV = DATASET_INDEX_CSV

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# ===============================
# HELPERS
# ===============================
def is_image(filename):
    return filename.lower().endswith(VALID_EXTENSIONS)

# ===============================
# MAIN
# ===============================
rows = []

# ---- REAL IMAGES ----
real_path = RAW_DATA_DIR / REAL_DIR

if not real_path.is_dir():
    raise RuntimeError(f"Missing folder: {real_path}")

for fname in sorted(os.listdir(real_path)):
    if is_image(fname):
        rel_path = os.path.join(REAL_DIR, fname)
        rows.append({
            "path": rel_path.replace("\\", "/"),
            "label": 0,
            "source": "real"
        })

print(f"[INFO] Found {len(rows)} real images")

# ---- FAKE IMAGES ----
fake_root = RAW_DATA_DIR / FAKE_DIR

if not fake_root.is_dir():
    raise RuntimeError(f"Missing folder: {fake_root}")

fake_count = 0

for generator in sorted(os.listdir(fake_root)):
    gen_path = os.path.join(fake_root, generator)

    if not os.path.isdir(gen_path):
        continue

    gen_images = 0
    for fname in sorted(os.listdir(gen_path)):
        if is_image(fname):
            rel_path = os.path.join(FAKE_DIR, generator, fname)
            rows.append({
                "path": rel_path.replace("\\", "/"),
                "label": 1,
                "source": generator
            })
            gen_images += 1
            fake_count += 1

    print(f"[INFO] Found {gen_images} images for generator: {generator}")

print(f"[INFO] Total fake images: {fake_count}")

# ===============================
# WRITE CSV
# ===============================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["path", "label", "source"]
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print("\n[SUCCESS] Dataset index created:")
print(f"  -> {OUTPUT_CSV}")
print(f"  -> Total samples: {len(rows)}")

# ===============================
# BASIC SANITY CHECK
# ===============================
num_real = sum(1 for r in rows if r["label"] == 0)
num_fake = sum(1 for r in rows if r["label"] == 1)

print("\n[SANITY CHECK]")
print(f"  Real images : {num_real}")
print(f"  Fake images : {num_fake}")

if num_real == 0 or num_fake == 0:
    print("[WARNING] One of the classes is empty!")
else:
    print("[OK] Dataset looks valid")
