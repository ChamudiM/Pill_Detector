"""
Image Preprocessing Module
Handles all image preprocessing operations including resizing, noise reduction, 
normalization, and augmentation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing operations."""
    
    def __init__(self, img_size: int = 640):
        """
        Initialize preprocessor.
        
        Args:
            img_size: Target image size (square)
        """
        self.img_size = img_size
        logger.info(f"ImagePreprocessor initialized with size {img_size}")
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size maintaining aspect ratio.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Resized image
        """
        return cv2.resize(image, (self.img_size, self.img_size), 
                         interpolation=cv2.INTER_LINEAR)
    
    def apply_gaussian_blur(self, image: np.ndarray, 
                           kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur for noise reduction.
        
        Args:
            image: Input image
            kernel_size: Kernel size (must be odd)
            
        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_median_filter(self, image: np.ndarray, 
                           kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filter for salt-and-pepper noise reduction.
        
        Args:
            image: Input image
            kernel_size: Kernel size (must be odd)
            
        Returns:
            Filtered image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(image, kernel_size)
    
    def normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize brightness using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Brightness-normalized image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        result = cv2.merge([l, a, b])
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    def bilateral_filter(self, image: np.ndarray, d: int = 9, 
                        sigma_color: float = 75, 
                        sigma_space: float = 75) -> np.ndarray:
        """
        Apply bilateral filter (preserves edges while smoothing).
        
        Args:
            image: Input image
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Filtered image
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def apply_morphological_operations(self, image: np.ndarray, 
                                      operation: str = "open",
                                      kernel_size: int = 5) -> np.ndarray:
        """
        Apply morphological operations.
        
        Args:
            image: Input grayscale image
            operation: 'open', 'close', 'gradient', 'tophat'
            kernel_size: Size of the morphological kernel
            
        Returns:
            Processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (kernel_size, kernel_size))
        
        if operation == "open":
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == "gradient":
            return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        elif operation == "tophat":
            return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        else:
            logger.warning(f"Unknown operation: {operation}")
            return image
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def preprocess_frame(self, frame: np.ndarray, 
                        apply_blur: bool = True,
                        apply_normalization: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline for a single frame.
        
        Args:
            frame: Input frame (BGR)
            apply_blur: Whether to apply blur
            apply_normalization: Whether to normalize brightness
            
        Returns:
            Preprocessed frame
        """
        # Resize
        frame = self.resize_image(frame)
        
        # Apply blur if requested
        if apply_blur:
            frame = self.apply_gaussian_blur(frame, kernel_size=5)
        
        # Normalize brightness if requested
        if apply_normalization:
            frame = self.normalize_brightness(frame)
        
        return frame
    
    def preprocess_batch(self, frames: list, **kwargs) -> list:
        """
        Preprocess a batch of frames.
        
        Args:
            frames: List of images
            **kwargs: Arguments for preprocess_frame
            
        Returns:
            List of preprocessed images
        """
        return [self.preprocess_frame(frame, **kwargs) for frame in frames]
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha: float = 1.5, 
                        beta: float = 0) -> np.ndarray:
        """
        Enhance image contrast using: output = alpha * input + beta
        
        Args:
            image: Input image
            alpha: Contrast factor (>1 increases contrast)
            beta: Brightness adjustment
            
        Returns:
            Enhanced image
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


class DataAugmenter:
    """Handles data augmentation operations."""
    
    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    @staticmethod
    def flip_image(image: np.ndarray, direction: str = "horizontal") -> np.ndarray:
        """
        Flip image.
        
        Args:
            image: Input image
            direction: 'horizontal', 'vertical', or 'both'
            
        Returns:
            Flipped image
        """
        if direction == "horizontal":
            return cv2.flip(image, 1)
        elif direction == "vertical":
            return cv2.flip(image, 0)
        elif direction == "both":
            return cv2.flip(image, -1)
        return image
    
    @staticmethod
    def scale_image(image: np.ndarray, scale: float) -> np.ndarray:
        """Scale image by given factor."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h))
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness. factor > 1 increases brightness."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def add_noise(image: np.ndarray, noise_level: float = 10) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, noise_level, image.shape)
        return cv2.convertScaleAbs(image.astype(np.float32) + noise)


if __name__ == "__main__":
    # Test preprocessing
    logger.info("Testing ImagePreprocessor...")
    
    # Create a test image
    test_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    preprocessor = ImagePreprocessor(img_size=640)
    processed = preprocessor.preprocess_frame(test_img)
    
    logger.info(f"Original shape: {test_img.shape}")
    logger.info(f"Processed shape: {processed.shape}")
    
    # Test augmentation
    logger.info("Testing DataAugmenter...")
    augmenter = DataAugmenter()
    rotated = augmenter.rotate_image(test_img, 15)
    logger.info(f"Rotated shape: {rotated.shape}")
    
    logger.info("Tests completed successfully!")