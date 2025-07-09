"""Image preprocessing for improved OCR accuracy."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from src.core.config import PreprocessingConfig


class ImagePreprocessor:
    """Handles image preprocessing for OCR."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply multiple preprocessing strategies and return all variants.
        
        Args:
            image: Input image (color or grayscale)
            
        Returns:
            List of preprocessed images
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(
            gray, 
            None, 
            self.config.denoise_strength, 
            7, 
            21
        )
        
        preprocessed_images = []
        
        # Apply different thresholding methods
        for method in self.config.threshold_methods:
            processed = self._apply_threshold(denoised, method)
            if processed is not None:
                preprocessed_images.append(processed)
        
        # Create a combined image using multiple methods
        combined = self._create_combined_image(preprocessed_images)
        if combined is not None:
            preprocessed_images.append(combined)
        
        # Add a variant specifically optimized for thin letters
        thin_optimized = self._optimize_for_thin_letters(denoised)
        if thin_optimized is not None:
            preprocessed_images.append(thin_optimized)
        
        self.logger.info(f"Created {len(preprocessed_images)} preprocessed variants")
        
        return preprocessed_images
    
    def _apply_threshold(self, image: np.ndarray, method: str) -> Optional[np.ndarray]:
        """Apply a specific thresholding method."""
        try:
            if method == 'adaptive_gaussian':
                blurred = cv2.GaussianBlur(image, self.config.blur_kernel_size, 0)
                return cv2.adaptiveThreshold(
                    blurred, 
                    255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 
                    21,  # Block size
                    10   # C constant
                )
            
            elif method == 'otsu':
                blurred = cv2.GaussianBlur(image, self.config.blur_kernel_size, 0)
                _, result = cv2.threshold(
                    blurred, 
                    0, 
                    255, 
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                return result
            
            elif method.startswith('manual_'):
                threshold_value = int(method.split('_')[1])
                _, result = cv2.threshold(
                    image, 
                    threshold_value, 
                    255, 
                    cv2.THRESH_BINARY_INV
                )
                return result
            
            else:
                self.logger.warning(f"Unknown threshold method: {method}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to apply {method} threshold: {e}")
            return None
    
    def _create_combined_image(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """Combine multiple binary images using OR operation."""
        if not images:
            return None
        
        try:
            # Start with the first image
            combined = images[0].copy()
            
            # OR with each subsequent image
            for img in images[1:]:
                combined = cv2.bitwise_or(combined, img)
            
            # Clean up with morphological operations
            kernel = np.ones(self.config.morph_kernel_size, np.uint8)
            
            # Close small gaps
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            
            # Remove small noise
            kernel_small = np.ones((2, 2), np.uint8)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small)
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Failed to create combined image: {e}")
            return None
    
    def _optimize_for_thin_letters(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Create a variant specifically optimized for thin letters like 'I'."""
        try:
            # Use a lower threshold to capture thin strokes
            _, binary = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)
            
            # Enhance vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            vertical_enhanced = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, vertical_kernel)
            
            # Slightly dilate to make letters more visible
            dilate_kernel = np.ones((2, 1), np.uint8)
            dilated = cv2.dilate(vertical_enhanced, dilate_kernel, iterations=1)
            
            return dilated
            
        except Exception as e:
            self.logger.error(f"Failed to optimize for thin letters: {e}")
            return None
    
    def crop_game_area(self, image: np.ndarray, window_rect: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop the game area from a screenshot.
        
        Args:
            image: Full screenshot
            window_rect: (x, y, width, height) of the game window
            
        Returns:
            Cropped image and crop coordinates (x, y)
        """
        x, y, width, height = window_rect
        
        # Calculate crop area based on percentages from config
        # Note: These values should come from ScreenConfig, not PreprocessingConfig
        # Using default values here since PreprocessingConfig doesn't have screen settings
        crop_x_start = 0.2
        crop_y_start = 0.625
        crop_width_pct = 0.6
        crop_height_pct = 0.275
        
        crop_x = int(width * crop_x_start)
        crop_y = int(height * crop_y_start)
        crop_width = int(width * crop_width_pct)
        crop_height = int(height * crop_height_pct)
        
        # Ensure we don't go out of bounds
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)
        crop_width = min(crop_width, width - crop_x)
        crop_height = min(crop_height, height - crop_y)
        
        # Crop the image
        cropped = image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
        
        return cropped, (crop_x, crop_y)