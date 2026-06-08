# src/preprocessing.py
import cv2
import numpy as np
from pathlib import Path

# Basic image utilities (resize, blur)
class BasicPreprocessor:
    def __init__(self, img_size=640):
        self.img_size = img_size

    def resize_image(self, image):
        """Resize image to target size."""
        return cv2.resize(image, (self.img_size, self.img_size))

    def apply_gaussian_blur(self, image, kernel_size=5):
        """Apply Gaussian blur for noise reduction."""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Advanced preprocessing techniques (median filter, CLAHE)
class AdvancedPreprocessor:
    def _init_(self, img_size=640):
        self.img_size = img_size

    def apply_median_filter(self, image, kernel_size=5):
        """Apply median filter for noise reduction."""
        return cv2.medianBlur(image, kernel_size)

    def normalize_brightness(self, image):
        """Normalize image brightness using CLAHE."""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            return cv2.merge([l, a, b])
        return image

    def preprocess_frame(self, frame):
        """Complete preprocessing pipeline (resize → blur → CLAHE)."""
        basic = BasicPreprocessor(self.img_size)
        frame = basic.resize_image(frame)
        frame = basic.apply_gaussian_blur(frame)
        frame = self.normalize_brightness(frame)
        return frame


