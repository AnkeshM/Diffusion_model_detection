import numpy as np
from numpy.fft import fft2, fftshift

def cross_difference(channel):
    """
    Computes the cross-difference filter residual for a single channel.
    C(x,y) = I(x,y) + I(x+1,y+1) - I(x,y+1) - I(x+1,y)
    """
    return (
        channel[:-1, :-1]
        + channel[1:, 1:]
        - channel[:-1, 1:]
        - channel[1:, :-1]
    )

def fft_magnitude(cd_channel):
    """
    Computes the normalized 2D FFT magnitude spectrum of the residual.
    Requires input array to be floating point.
    """
    H, W = cd_channel.shape
    # Compute 2D FFT and shift zero-frequency component to the center of the spectrum
    F = fftshift(np.abs(fft2(cd_channel)))
    return F / (H * W)

def extract_neighborhood_features(F, r, c, window=1):
    """
    Extracts robust features from a (2*window+1)x(2*window+1) neighborhood.
    Extracts mean, variance, and max magnitude to be resilient against peak shifting.
    """
    H, W = F.shape
    
    # Clip window to spectrum bounds
    r_start = max(0, r - window)
    r_end = min(H, r + window + 1)
    c_start = max(0, c - window)
    c_end = min(W, c + window + 1)
    
    patch = F[r_start:r_end, c_start:c_end]
    
    if patch.size == 0:
        return [0.0, 0.0, 0.0]
        
    return [
        float(np.mean(patch)),
        float(np.var(patch)),
        float(np.max(patch))
    ]

def extract_robust_peaks(F):
    """
    Extracts features for specific expected peak periods taking the neighborhood into account.
    Returns a unified flat list of characteristics.
    """
    H, W = F.shape
    cx, cy = H // 2, W // 2

    features = []

    # DC peak (0 frequency)
    features.extend(extract_neighborhood_features(F, cx, cy))

    periods = [2, 4, 8]

    # Horizontal and vertical peaks
    for px in periods:
        dx = H // px
        features.extend(extract_neighborhood_features(F, cx + dx, cy))
        features.extend(extract_neighborhood_features(F, cx - dx, cy))

    for py in periods:
        dy = W // py
        features.extend(extract_neighborhood_features(F, cx, cy + dy))
        features.extend(extract_neighborhood_features(F, cx, cy - dy))

    # Diagonal peaks
    for px in periods:
        dx = H // px
        for py in periods:
            dy = W // py
            features.extend(extract_neighborhood_features(F, cx + dx, cy + dy))
            features.extend(extract_neighborhood_features(F, cx + dx, cy - dy))
            features.extend(extract_neighborhood_features(F, cx - dx, cy + dy))
            features.extend(extract_neighborhood_features(F, cx - dx, cy - dy))

    return features

def extract_features_from_multiscale(multiscales):
    """
    Process dictionary mapped multi-scaled arrays: {scale: img_array}
    Expected scale ordering to enforce consistency: 256, 128, 64.
    Channels are assumed to be RGB (shape HxWx3).
    """
    all_features = []
    
    # Enforce order
    scales = sorted(list(multiscales.keys()), reverse=True)
    
    for scale in scales:
        img = multiscales[scale]
        # Iterate over channels (R, G, B)
        for c in range(3):
            # Compute CD for this channel
            cd = cross_difference(img[:, :, c])
            # FFT and feature extraction
            F_mag = fft_magnitude(cd)
            peaks = extract_robust_peaks(F_mag)
            all_features.extend(peaks)

    # Return as unified single vector suitable for LightGBM
    return np.array(all_features, dtype=np.float32)
