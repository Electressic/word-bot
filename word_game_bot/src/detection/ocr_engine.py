"""Enhanced OCR engine with multiple detection strategies."""

import easyocr
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass

from src.core.config import OCRConfig


@dataclass
class OCRResult:
    """Represents an OCR detection result."""
    text: str
    bbox: List[List[float]]  # Changed to float to handle any numeric type
    confidence: float
    center: Optional[Tuple[int, int]] = None
    
    def __post_init__(self):
        """Calculate center if not provided."""
        if not self.center and self.bbox:
            xs = [p[0] for p in self.bbox]
            ys = [p[1] for p in self.bbox]
            self.center = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))


class OCREngine:
    """Enhanced OCR engine with multiple detection strategies."""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.reader = easyocr.Reader(config.languages, gpu=config.gpu)
        self.logger = logging.getLogger(__name__)
        
        # Build allowlist from uppercase letters and mapped characters
        self.allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + ''.join(config.char_mappings.keys())
    
    def detect_text(self, image: np.ndarray) -> List[OCRResult]:
        """
        Detect text using multiple strategies and combine results.
        
        Args:
            image: Preprocessed image (grayscale or binary)
            
        Returns:
            List of OCR results
        """
        all_results = []
        
        # Strategy 1: Multiple parameter sets on original image
        for i, params in enumerate(self.config.parameter_sets):
            self.logger.info(f"Trying OCR parameter set {i+1}")
            results = self._run_ocr(image, params)
            all_results.extend(results)
        
        # Strategy 2: Enhanced image for thin letters (especially 'I')
        enhanced_image = self._enhance_thin_letters(image)
        if enhanced_image is not None:
            self.logger.info("Trying OCR on enhanced image for thin letters")
            # Use the most aggressive parameters for thin letters
            results = self._run_ocr(enhanced_image, self.config.parameter_sets[0])
            all_results.extend(results)
        
        # Strategy 3: Morphologically processed image
        morphed_image = self._apply_morphology(image)
        if morphed_image is not None:
            self.logger.info("Trying OCR on morphologically processed image")
            results = self._run_ocr(morphed_image, self.config.parameter_sets[1])
            all_results.extend(results)
        
        # Deduplicate and filter results
        unique_results = self._deduplicate_results(all_results)
        
        return unique_results
    
    def _run_ocr(self, image: np.ndarray, params: Dict) -> List[OCRResult]:
        """Run OCR with specific parameters."""
        try:
            raw_results = self.reader.readtext(
                image,
                allowlist=self.allowlist,
                paragraph=False,
                **params
            )
            
            results = []
            for bbox, text, confidence in raw_results:
                if len(bbox) >= 4:
                    result = OCRResult(
                        text=text,
                        bbox=bbox,
                        confidence=float(confidence),
                        center=None  # Will be calculated in __post_init__
                    )
                    results.append(result)
            
            self.logger.info(f"OCR run returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.warning(f"OCR run failed: {e}")
            return []
    
    def _enhance_thin_letters(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Enhance thin letters like 'I' by applying specific filters."""
        try:
            # Convert to binary if not already
            if len(np.unique(image)) > 2:
                _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            else:
                binary = image.copy()
            
            # Apply vertical line detection to enhance 'I'
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            # Dilate slightly to make lines thicker
            dilate_kernel = np.ones((2, 2), np.uint8)
            enhanced = cv2.dilate(vertical_lines, dilate_kernel, iterations=1)
            
            # Combine with original
            result = cv2.bitwise_or(binary, enhanced)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance thin letters: {e}")
            return None
    
    def _apply_morphology(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Apply morphological operations to improve letter connectivity."""
        try:
            # Convert to binary if needed
            if len(np.unique(image)) > 2:
                _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            else:
                binary = image.copy()
            
            # Close small gaps
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (2, 2)  # Use fixed kernel size
            )
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Remove small noise
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            
            return opened
            
        except Exception as e:
            self.logger.warning(f"Failed to apply morphology: {e}")
            return None
    
    def _deduplicate_results(self, results: List[OCRResult], 
                           radius: int = 20) -> List[OCRResult]:
        """Remove duplicate detections, keeping highest confidence."""
        if not results:
            return []
        
        # Sort by confidence (highest first)
        sorted_results = sorted(results, key=lambda x: x.confidence, reverse=True)
        
        unique = []
        for result in sorted_results:
            # Check if too close to any already kept result
            too_close = False
            for kept in unique:
                dist = np.hypot(
                    result.center[0] - kept.center[0],
                    result.center[1] - kept.center[1]
                )
                if dist < radius:
                    too_close = True
                    break
            
            if not too_close:
                unique.append(result)
        
        return unique
    
    def map_character(self, char: str) -> Optional[str]:
        """Map ambiguous characters to letters."""
        if not char:
            return None
        
        # Check mapping
        if char in self.config.char_mappings:
            return self.config.char_mappings[char]
        
        # Return uppercase if already a letter
        if char.isalpha():
            return char.upper()
        
        return None
    
    def get_confidence_threshold(self, letter: str) -> float:
        """Get the confidence threshold for a specific letter."""
        return self.config.letter_thresholds.get(
            letter, 
            self.config.letter_thresholds.get('DEFAULT', 0.4)
        )