"""Image preprocessing for improved OCR accuracy."""

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
        Apply multiple preprocessing strategies and return all variants.
        Focus on creating clean black-on-white images for OCR.
        
        Args:
            image: Input image (color or grayscale)
            
        Returns:
            List of preprocessed images (all black letters on white background)
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
        
        # Method 1: Adaptive Threshold (primary method)
        adaptive_bw = self._create_adaptive_black_on_white(denoised)
        if adaptive_bw is not None:
            preprocessed_images.append(adaptive_bw)
            self.save_debug_image("3_adaptive_black_white", adaptive_bw)
        
        # Method 2: Otsu's method
        otsu_bw = self._create_otsu_black_on_white(denoised)
        if otsu_bw is not None:
            preprocessed_images.append(otsu_bw)
            self.save_debug_image("4_otsu_black_white", otsu_bw)
        
        # Method 3: Manual threshold (for high contrast images)
        manual_bw = self._create_manual_black_on_white(denoised)
        if manual_bw is not None:
            preprocessed_images.append(manual_bw)
            self.save_debug_image("5_manual_black_white", manual_bw)
        
        # Method 4: Enhanced for thin letters
        thin_bw = self._create_thin_letter_black_on_white(denoised)
        if thin_bw is not None:
            preprocessed_images.append(thin_bw)
            self.save_debug_image("6_thin_letter_black_white", thin_bw)
        
        # Create a combined image if we have multiple variants
        if len(preprocessed_images) > 1:
            combined = self._create_combined_black_on_white(preprocessed_images)
            if combined is not None:
                preprocessed_images.append(combined)
                self.save_debug_image("7_combined_black_white", combined)
        
        self.logger.info(f"Created {len(preprocessed_images)} black-on-white variants")
        
        # Ensure we have at least one image
        if not preprocessed_images:
            # Fallback: simple threshold
            _, fallback = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
            preprocessed_images.append(fallback)
            self.save_debug_image("8_fallback_black_white", fallback)
            self.logger.warning("Using fallback threshold method")
        
        return preprocessed_images
    
    def _create_adaptive_black_on_white(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Create black-on-white using adaptive threshold."""
        try:
            # Apply gentle blur
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            
            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21,  # Block size
                10   # C constant
            )
            
            # Check if we need to invert (letters should be black)
            binary = self._ensure_black_letters_white_background(binary)
            
            # Clean up noise
            binary = self._clean_binary_image(binary)
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Adaptive threshold failed: {e}")
            return None
    
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
    
    def _create_manual_black_on_white(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Create black-on-white using manual threshold."""
        try:
            # Determine threshold based on image characteristics
            mean_val = np.mean(image)
            if mean_val > 200:
                threshold = 150  # High contrast image
            elif mean_val > 100:
                threshold = int(mean_val * 0.8)  # Medium contrast
            else:
                threshold = 80   # Low contrast
            
            _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            
            # Ensure correct polarity
            binary = self._ensure_black_letters_white_background(binary)
            
            # Clean up
            binary = self._clean_binary_image(binary)
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Manual threshold failed: {e}")
            return None
    
    def _create_thin_letter_black_on_white(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Create black-on-white optimized for thin letters like 'I'."""
        try:
            # Lower threshold to capture thin strokes
            _, binary = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
            
            # Ensure correct polarity
            binary = self._ensure_black_letters_white_background(binary)
            
            # Enhance thin vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, vertical_kernel)
            
            # Clean up while preserving thin letters
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Thin letter optimization failed: {e}")
            return None
    
    def _create_combined_black_on_white(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """Combine multiple black-on-white images."""
        try:
            if not images:
                return None
            
            # Start with first image
            combined = images[0].copy()
            
            # OR operation to combine all letter detections
            for img in images[1:]:
                # Invert both images, OR them, then invert back
                # This ensures we capture letters from all methods
                inv_combined = cv2.bitwise_not(combined)
                inv_img = cv2.bitwise_not(img)
                inv_result = cv2.bitwise_or(inv_combined, inv_img)
                combined = cv2.bitwise_not(inv_result)
            
            # Final cleanup
            combined = self._clean_binary_image(combined)
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Combined image creation failed: {e}")
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
        # Remove small noise with opening
        kernel_open = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        
        # Fill small gaps in letters with closing
        kernel_close = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        
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
    
    def get_base_image_for_visualization(self) -> Optional[np.ndarray]:
        """Get the best black-on-white image for visualization overlay."""
        # Return the path to the best processed image for visualization
        best_image_path = os.path.join(self.debug_dir, "3_adaptive_black_white.png")
        if os.path.exists(best_image_path):
            return cv2.imread(best_image_path, cv2.IMREAD_GRAYSCALE)
        return None