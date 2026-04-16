import numpy as np
from PIL import Image
from numpy.fft import fft2, fftshift
import sys
from pathlib import Path

# Add root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)

def cross_difference(channel):
    return (
        channel[:-1, :-1]
        + channel[1:, 1:]
        - channel[:-1, 1:]
        - channel[1:, :-1]
    )

def fft_magnitude(cd):
    H, W = cd.shape
    F = fftshift(np.abs(fft2(cd)))
    return F / (H * W)

def extract_peaks(F):
    H, W = F.shape
    cx, cy = H // 2, W // 2

    features = []

    # DC
    features.append(F[cx, cy])

    periods = [2, 4, 8]

    for px in periods:
        dx = H // px
        features.extend([F[cx + dx, cy], F[cx - dx, cy]])

    for py in periods:
        dy = W // py
        features.extend([F[cx, cy + dy], F[cx, cy - dy]])

    for px in periods:
        dx = H // px
        for py in periods:
            dy = W // py
            features.extend([
                F[cx + dx, cy + dy],
                F[cx + dx, cy - dy],
                F[cx - dx, cy + dy],
                F[cx - dx, cy - dy],
            ])

    return features  # total = 49

if len(sys.argv) != 2:
    print("Usage: python extract_single_feature.py image.png")
    exit()

image_path = sys.argv[1]

img = load_image(image_path)
features = []

for c in range(3):
    cd = cross_difference(img[:, :, c])
    F = fft_magnitude(cd)
    features.extend(extract_peaks(F))

X_single = np.asarray(features, dtype=np.float32).reshape(1, -1)

OUTPUT_FILE = RESULTS_DIR / "single_image_features.npy"
np.save(OUTPUT_FILE, X_single)

print("\n[SUCCESS]")
print(f"Feature file: {OUTPUT_FILE}")
print("Shape:", X_single.shape)
