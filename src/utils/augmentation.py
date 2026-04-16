import random
import numpy as np
from PIL import Image, ImageFilter
from .jpeg_utils import jpeg_compress

def random_resize(img):
    scale = random.choice([0.5, 0.75, 1.0, 1.5])
    w, h = img.size
    return img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

def random_crop(img):
    if random.random() < 0.5:
        w, h = img.size
        cw, ch = int(w * 0.5), int(h * 0.5)
        left = random.randint(0, w - cw)
        top = random.randint(0, h - ch)
        img = img.crop((left, top, left + cw, top + ch))
    return img

def random_blur(img):
    if random.random() < 0.5:
        radius = random.choice([1, 2, 3])
        img = img.filter(ImageFilter.GaussianBlur(radius))
    return img

def random_denoise(img):
    if random.random() < 0.5:
        # Median filter is a common choice for denoising
        size = random.choice([3, 5])
        img = img.filter(ImageFilter.MedianFilter(size=size))
    return img

def random_jpeg(img):
    if random.random() < 0.5:
        quality = random.randint(50, 95)
        img = jpeg_compress(img, quality)
    return img

def random_noise(img):
    if random.random() < 0.5:
        sigma = random.choice([5, 10, 20])
        arr = np.asarray(img, dtype=np.float32)
        noise = np.random.normal(0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
    return img

def augment_image(img):
    """
    Apply a sequence of random augmentations to an image.
    Used for building robust models.
    """
    img = random_resize(img)
    img = random_crop(img)
    img = random_blur(img)
    img = random_noise(img)
    img = random_denoise(img)
    img = random_jpeg(img)
    return img
