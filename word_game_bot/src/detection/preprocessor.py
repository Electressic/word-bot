"""Updated ImagePreprocessor to use only Otsu method"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
import logging

from src.core.config import PreprocessingConfig


class ImagePreprocessor:
    """Handles image preprocessing for OCR."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.debug_dir = getattr(config, 'debug_dir', 'debug')
        self._setup_debug_dir()
    
    def _setup_debug_dir(self):
        """Create debug directory if it doesn't exist."""
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
            self.logger.info(f"Created debug directory: {self.debug_dir}")
    
    def save_debug_image(self, name: str, image: np.ndarray):
        """Save debug image to the debug directory."""
        filename = os.path.join(self.debug_dir, f"{name}.png")
        cv2.imwrite(filename, image)
        self.logger.debug(f"Saved debug image: {filename}")
    
    def preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply Otsu preprocessing and return only that variant.
        Focus on creating clean black-on-white images for OCR.
        
        Args:
            image: Input image (color or grayscale)
            
        Returns:
            List containing only the Otsu-processed black-on-white image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Save original for debugging
        self.save_debug_image("1_original_cropped", gray)
        self.logger.info(f"Original image stats: mean={np.mean(gray):.1f}, min={np.min(gray)}, max={np.max(gray)}")
        
        # Apply light denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        self.save_debug_image("2_denoised", denoised)
        
        preprocessed_images = []
        
        # Use ONLY Otsu's method
        otsu_bw = self._create_otsu_black_on_white(denoised)
        if otsu_bw is not None:
            preprocessed_images.append(otsu_bw)
            self.save_debug_image("3_otsu_black_white", otsu_bw)
            self.logger.info("Created Otsu black-on-white (primary method)")
        else:
            # Fallback: simple threshold if Otsu fails
            _, fallback = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
            preprocessed_images.append(fallback)
            self.save_debug_image("3_fallback_black_white", fallback)
            self.logger.warning("Otsu failed, using fallback threshold method")
        
        self.logger.info(f"Created {len(preprocessed_images)} black-on-white variants")
        
        return preprocessed_images
    
    def _create_otsu_black_on_white(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Create black-on-white using Otsu's method."""
        try:
            # Apply blur
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            
            # Otsu's threshold
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Ensure correct polarity
            binary = self._ensure_black_letters_white_background(binary)
            
            # Clean up
            binary = self._clean_binary_image(binary)
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Otsu threshold failed: {e}")
            return None
    
    def _ensure_black_letters_white_background(self, binary: np.ndarray) -> np.ndarray:
        """Ensure binary image has black letters on white background."""
        # Count black vs white pixels in center region (where letters likely are)
        h, w = binary.shape
        center_region = binary[h//4:3*h//4, w//4:3*w//4]
        
        black_pixels = np.sum(center_region == 0)
        white_pixels = np.sum(center_region == 255)
        
        # If more black than white in center, likely white letters on black background
        if black_pixels > white_pixels * 1.5:
            self.logger.debug("Inverting image - detected white letters on black background")
            return cv2.bitwise_not(binary)
        
        return binary
    
    def _clean_binary_image(self, binary: np.ndarray) -> np.ndarray:
        """Clean up binary image noise while preserving letter structure."""
        # Small morphological opening to remove isolated noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Small closing to connect broken parts
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
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
        crop_x_start = getattr(self.config, 'crop_x_start', 0.2)
        crop_y_start = getattr(self.config, 'crop_y_start', 0.625)
        crop_width_pct = getattr(self.config, 'crop_width_pct', 0.6)
        crop_height_pct = getattr(self.config, 'crop_height_pct', 0.275)
        
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