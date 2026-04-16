import numpy as np
import sys
from pathlib import Path

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR

specific_file = RESULTS_DIR / "table1_specific.npy"
general_file = RESULTS_DIR / "table1_generalization.npy"

specific = np.load(specific_file, allow_pickle=True).item()
general  = np.load(general_file, allow_pickle=True).item()

print("\n=== TABLE 1 (MCC) ===")
print(f"{'Model':25s}  Specific   Generalization")

for gen in specific:
    print(f"{gen:25s}  {specific[gen]:8.4f}   {general[gen]:8.4f}")
