"""Improved letter detection with better handling of problematic letters."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import math

from src.core.config import BotConfig
from src.detection.ocr_engine import OCREngine, OCRResult
from src.detection.preprocessor import ImagePreprocessor


class LetterDetector:
    """Detects individual letters from game screenshots."""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.ocr_engine = OCREngine(config.ocr)
        self.preprocessor = ImagePreprocessor(config.preprocessing)
        self.logger = logging.getLogger(__name__)
    
    def detect_letters(self, image: np.ndarray) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Detect letters and their positions from an image.
        """
        # Preprocess the image (now returns black-on-white images)
        processed_images = self.preprocessor.preprocess(image)
        
        all_detections = []
        
        # Run OCR on each preprocessed variant
        for i, processed in enumerate(processed_images):
            self.logger.info(f"Running OCR on black-on-white variant {i+1}")
            ocr_results = self.ocr_engine.detect_text(processed)
            letter_results = self._process_ocr_results(ocr_results)
            all_detections.extend(letter_results)
        
        # Special handling for 'I' detection if needed
        if not any(letter == 'I' for letter, _ in all_detections):
            self.logger.warning("No 'I' detected, trying specialized detection")
            i_detections = self._detect_letter_i_specialized(processed_images[0] if processed_images else image)
            all_detections.extend(i_detections)
        
        # Deduplicate all detections
        unique_detections = self._deduplicate_detections(all_detections)
        
        # Log final results
        detected_letters = [letter for letter, _ in unique_detections]
        self.logger.info(f"Final detected letters: {detected_letters}")
        
        return unique_detections
    
    def _process_ocr_results(self, ocr_results: List[OCRResult]) -> List[Tuple[str, Tuple[int, int]]]:
        """Convert OCR results to letter detections."""
        detections = []
        
        for result in ocr_results:
            text = result.text.strip()
            
            if len(text) == 1:
                # Single character detection
                letter = self.ocr_engine.map_character(text)
                if letter:
                    threshold = self.ocr_engine.get_confidence_threshold(letter)
                    if result.confidence >= threshold:
                        detections.append((letter, result.center))
                        self.logger.debug(f"Detected '{letter}' at {result.center} with confidence {result.confidence:.3f}")
            
            elif len(text) > 1:
                # Multi-character detection - split it
                self.logger.info(f"Splitting multi-character detection: '{text}'")
                split_detections = self._split_multi_letter(result)
                detections.extend(split_detections)
        
        return detections
    
    def _split_multi_letter(self, result: OCRResult) -> List[Tuple[str, Tuple[int, int]]]:
        """Split a multi-letter detection into individual letters."""
        text = result.text
        
        # Map each character
        letters = []
        for char in text:
            letter = self.ocr_engine.map_character(char)
            if letter:
                letters.append(letter)
        
        if not letters:
            return []
        
        # Calculate individual letter positions
        bbox = result.bbox
        if len(bbox) >= 4:
            # Get bounding box dimensions
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            width = max_x - min_x
            height = max_y - min_y
            
            detections = []
            num_letters = len(letters)
            
            for i, letter in enumerate(letters):
                # Estimate position for this letter
                letter_x = min_x + (i + 0.5) * (width / num_letters)
                letter_y = min_y + height / 2
                
                # Check confidence threshold
                threshold = self.ocr_engine.get_confidence_threshold(letter)
                if result.confidence >= threshold:
                    detections.append((letter, (int(letter_x), int(letter_y))))
                    self.logger.debug(f"Split letter '{letter}' at ({int(letter_x)}, {int(letter_y)})")
            
            return detections
        
        return []
    
    def _detect_letter_i_specialized(self, image: np.ndarray) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Specialized detection for the letter 'I' using line detection.
        This is a fallback when normal OCR fails to detect 'I'.
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Threshold
            _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
            
            # Detect vertical lines using Hough transform
            lines = cv2.HoughLinesP(
                binary,
                rho=1,
                theta=np.pi/180,
                threshold=30,
                minLineLength=20,
                maxLineGap=5
            )
            
            if lines is None:
                return []
            
            detections = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line is mostly vertical
                angle = abs(math.atan2(y2 - y1, x2 - x1))
                if angle > math.pi * 0.4:  # Within 45 degrees of vertical
                    # Check aspect ratio
                    length = math.hypot(x2 - x1, y2 - y1)
                    width = abs(x2 - x1)
                    
                    if length > 20 and width < length * 0.3:  # Thin and tall
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        detections.append(('I', (center_x, center_y)))
                        self.logger.info(f"Detected 'I' using line detection at ({center_x}, {center_y})")
            
            return detections
            
        except Exception as e:
            self.logger.warning(f"Specialized 'I' detection failed: {e}")
            return []
    
    def _deduplicate_detections(self, detections: List[Tuple[str, Tuple[int, int]]]) -> List[Tuple[str, Tuple[int, int]]]:
        """Remove duplicate detections within a radius."""
        if not detections:
            return []
        
        # Group by position with confidence scores
        detection_data = []
        for letter, pos in detections:
            # Assign confidence based on letter (some letters are more reliable)
            confidence = 1.0 if letter in ['A', 'E', 'T'] else 0.5
            detection_data.append((letter, pos, confidence))
        
        # Sort by confidence
        detection_data.sort(key=lambda x: x[2], reverse=True)
        
        unique = []
        radius = self.config.layout.deduplication_radius
        
        for letter, pos, conf in detection_data:
            # Special handling for 'I' - always include if no other I nearby
            if letter == 'I':
                has_nearby_i = any(
                    kept_letter == 'I' and 
                    math.hypot(pos[0] - kept_pos[0], pos[1] - kept_pos[1]) < radius
                    for kept_letter, kept_pos in unique
                )
                if not has_nearby_i:
                    unique.append((letter, pos))
                    continue
            
            # Check if too close to existing detection
            too_close = False
            for kept_letter, kept_pos in unique:
                dist = math.hypot(pos[0] - kept_pos[0], pos[1] - kept_pos[1])
                if dist < radius:
                    too_close = True
                    break
            
            if not too_close:
                unique.append((letter, pos))
        
        return unique