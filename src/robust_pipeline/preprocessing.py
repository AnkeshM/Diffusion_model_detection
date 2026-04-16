import cv2
import numpy as np
import random
from PIL import Image
import io

def load_image(path):
    """
    Load an image from disk, convert to RGB, and return as a numpy array.
    Used for clean loading.
    """
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32)

def normalize_image(img):
    """ Normalize image to [0, 1] for processing. """
    return img / 255.0

def apply_mild_enhancements(img):
    """
    Optionally apply light denoising and mild edge enhancement.
    We apply this to help recover high-frequency signals.
    Expects image in standard [0, 255] uint8 for cv2 operations.
    """
    # Convert float32 back to uint8 safely
    if img.dtype != np.uint8:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img

    # Light denoising (GaussianBlur instead of robust Non-Local Means to prevent destroying artifact structure)
    denoised = cv2.GaussianBlur(img_uint8, (3, 3), 0.5)

    # Mild edge enhancement using a weak Laplacian formulation (Unsharp Masking)
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    enhanced = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    
    return enhanced.astype(np.float32)

def generate_multiscale(img, base_res=256, scales=[256, 128, 64]):
    """
    Resize image to a base resolution, then generate downscaled versions.
    Returns a dictionary mapping resolution -> image array.
    """
    resolutions = {}
    
    # First convert to base resolution
    base_img = cv2.resize(img, (base_res, base_res), interpolation=cv2.INTER_AREA)
    
    # Generate all requested scales
    for scale in scales:
        if scale == base_res:
            resolutions[scale] = base_img
        else:
            resolutions[scale] = cv2.resize(base_img, (scale, scale), interpolation=cv2.INTER_AREA)
            
    return resolutions

def add_real_world_distortions(img):
    """
    Applies random real-world distortions for training robustness:
    - Gaussian noise
    - Gaussian blur
    - Random Rescaling
    - JPEG compression
    Expects input image array.
    """
    img_distorted = img.copy()

    # Random Blur
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img_distorted = cv2.GaussianBlur(img_distorted, (ksize, ksize), 0)

    # Random Rescale (downscale then upscale back to cause interpolation artifacts)
    if random.random() < 0.3:
        h, w = img_distorted.shape[:2]
        factor = random.uniform(0.5, 0.9)
        small_h, small_w = int(h * factor), int(w * factor)
        small = cv2.resize(img_distorted, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        img_distorted = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    # Ensure valid range for uint8 before doing JPEG compression
    img_distorted = np.clip(img_distorted, 0, 255).astype(np.uint8)

    # Random JPEG Compression
    if random.random() < 0.4:
        quality = random.randint(30, 90)
        # Using PIL for pure JPEG compression matching standard user flow
        pil_img = Image.fromarray(img_distorted)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        pil_img = Image.open(buffer)
        img_distorted = np.array(pil_img)

    # Random Noise (Add after JPEG so it doesn't get completely flattened)
    if random.random() < 0.3:
        # Avoid overflowing uint8
        img_distorted = img_distorted.astype(np.float32)
        noise = np.random.normal(0, random.uniform(2, 15), img_distorted.shape)
        img_distorted = img_distorted + noise
        img_distorted = np.clip(img_distorted, 0, 255)

    return img_distorted.astype(np.float32)

def pipeline_preprocess_inference(path_or_array):
    """ Standard preprocessing entrypoint for inference. """
    if isinstance(path_or_array, str):
        img = load_image(path_or_array)
    else:
        img = path_or_array
        
    img = apply_mild_enhancements(img)
    multiscales = generate_multiscale(img, base_res=256, scales=[256, 128, 64])
    return multiscales

def pipeline_preprocess_train(path_or_array, apply_distortions=True):
    """ Standard preprocessing entrypoint for training (includes augmentations). """
    if isinstance(path_or_array, str):
        img = load_image(path_or_array)
    else:
        img = path_or_array
        
    if apply_distortions:
        img = add_real_world_distortions(img)
        
    img = apply_mild_enhancements(img)
    multiscales = generate_multiscale(img, base_res=256, scales=[256, 128, 64])
    return multiscales
