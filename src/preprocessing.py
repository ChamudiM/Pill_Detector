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

