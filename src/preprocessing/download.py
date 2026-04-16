import csv
import os
import requests
import sys
from pathlib import Path
from urllib.parse import urlparse
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import tempfile

# Add root to sys.path to allow config import (though not strictly needed here yet)
sys.path.append(str(Path(__file__).resolve().parent.parent))

# If there was a config for download paths, I'd use it here.
# For now, keeping the original relative paths but making them more robust.

CSV_FILE = "dataset.csv"
OUT_DIR = "paralleldataset/images/tiff"

TIMEOUT = 30
RETRIES = 3
MAX_WORKERS = 12
SLEEP_BETWEEN = 0.05

os.makedirs(OUT_DIR, exist_ok=True)

lock = threading.Lock()

stats = {
    "downloaded": 0,
    "failed": 0,
    "skipped": 0,
    "processed": 0,
    "total": 0
}

def download_tiff(task):
    idx, url = task

    if not url or url.strip() == "":
        with lock:
            stats["skipped"] += 1
            stats["processed"] += 1
        return

    filename = os.path.basename(urlparse(url).path)
    out_path = os.path.join(OUT_DIR, filename)

    # 🔒 Thread-safe skip check
    with lock:
        if os.path.exists(out_path):
            stats["skipped"] += 1
            stats["processed"] += 1
            print(f"[{idx}] ⏭️  SKIP | {filename}")
            return

    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT, stream=True)
            r.raise_for_status()

            # Write to temp file first (atomic write)
            with tempfile.NamedTemporaryFile(delete=False, dir=OUT_DIR) as tmp:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                temp_path = tmp.name

            # Atomic rename
            os.replace(temp_path, out_path)

            with lock:
                stats["downloaded"] += 1
                stats["processed"] += 1
                print(f"[{idx}] ✅ DOWNLOADED ({stats['downloaded']}) | {filename}")
            return

        except Exception as e:
            sleep(0.5)

    with lock:
        stats["failed"] += 1
        stats["processed"] += 1
        print(f"[{idx}] ❌ FAILED | {filename}")

def main():
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] CSV file not found: {CSV_FILE}")
        return

    tasks = []

    with open(CSV_FILE, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "TIFF" not in reader.fieldnames:
            raise Exception("CSV must contain a column named 'TIFF'")

        rows = list(reader)
        stats["total"] = len(rows)

        for i, row in enumerate(rows, start=1):
            tasks.append((i, row["TIFF"]))

    print(f"\n📦 Total images: {stats['total']}")
    print(f"⚡ Parallel workers: {MAX_WORKERS}")
    print("🔁 Existing files will be skipped safely\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_tiff, task) for task in tasks]

        for _ in as_completed(futures):
            pass

    print("\n========== SUMMARY ==========")
    print(f"Total entries : {stats['total']}")
    print(f"Downloaded    : {stats['downloaded']}")
    print(f"Skipped       : {stats['skipped']}")
    print(f"Failed        : {stats['failed']}")
    print("=============================")
    print("TIFF dataset ready.\n")

if __name__ == "__main__":
    main()
