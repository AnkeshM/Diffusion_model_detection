import numpy as np
from PIL import Image
from numpy.fft import fft2, fftshift
from .jpeg_utils import jpeg_compress

def load_image(path, jpeg_quality=None):
    """
    Load an image and optionally apply JPEG compression.
    Returns: np.ndarray (H, W, 3)
    """
    img = Image.open(path).convert("RGB")
    if jpeg_quality is not None:
        img = jpeg_compress(img, jpeg_quality)
    return np.asarray(img, dtype=np.float32)

def cross_difference(channel):
    """
    Apply cross-difference operator to a single channel.
    """
    return (
        channel[:-1, :-1]
        + channel[1:, 1:]
        - channel[:-1, 1:]
        - channel[1:, :-1]
    )

def fft_magnitude(cd):
    """
    Compute the FFT magnitude spectrum of a cross-differenced channel.
    """
    H, W = cd.shape
    F = fftshift(np.abs(fft2(cd)))
    return F / (H * W)

def extract_peaks(F):
    """
    Extract peak values from the FFT magnitude spectrum at specific frequencies.
    Returns: list of 45 features per channel.
    """
    H, W = F.shape
    cx, cy = H // 2, W // 2

    features = []

    # DC component
    features.append(F[cx, cy])

    periods = [2, 4, 8]

    # Horizontal axis peaks
    for px in periods:
        dx = H // px
        features.extend([F[cx + dx, cy], F[cx - dx, cy]])

    # Vertical axis peaks
    for py in periods:
        dy = W // py
        features.extend([F[cx, cy + dy], F[cx, cy - dy]])

    # Diagonal peaks
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

    return features

def extract_features(img_np):
    """
    Extract the full 135-dimensional feature vector (45 per RGB channel).
    """
    all_features = []
    for c in range(3):  # R, G, B
        cd = cross_difference(img_np[:, :, c])
        F = fft_magnitude(cd)
        all_features.extend(extract_peaks(F))
    return np.asarray(all_features, dtype=np.float32)
