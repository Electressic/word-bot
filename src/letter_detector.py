import easyocr
import cv2
import numpy as np
from typing import List, Tuple, Dict
import os
import logging

class LetterDetector:
    """
    Detects letters and their positions from an image.
    """

    def __init__(self, lang: str = 'en', min_confidence: float = 0.8):
        """
        Initializes the LetterDetector.

        Args:
            lang: The language to use for OCR.
            min_confidence: The minimum confidence score for a detected letter.
        """
        self.reader = easyocr.Reader([lang], gpu=True)
        self.min_confidence = min_confidence
        
        # Mapping of visually similar characters to their intended letters
        self.char_mapping = {
            '1': 'I',  # Digit one -> I
            '|': 'I',  # Vertical bar -> I
            'l': 'I',  # Lower-case L -> I
            'i': 'I',  # Lower-case i -> I (in case)
            '0': 'O',  # Digit zero -> O
            'o': 'O',  # Lower-case o -> O
        }
        
        # Lower confidence thresholds for problematic letters
        self.letter_confidence_thresholds = {
            'I': 0.1,  # Very low threshold for 'I'
            'N': 0.3,  # Even lower threshold for N
            'O': 0.3,  # Even lower threshold for O
            'C': 0.4,  # O and C can be confused
            'D': 0.4,  # O and D can be confused
            'Q': 0.4,  # O and Q can be confused
            'G': 0.4,  # Can be confused with other letters
            'P': 0.4,  # Can be confused with other letters
            'R': 0.4,  # Can be confused with other letters
            # Require higher certainty for the more reliably detected letters
            'A': 0.8,
            'E': 0.8,
            'T': 0.8,
            'J': 0.8,
            'K': 0.8,
            'C': 0.8,
        }

    def _map_char(self, ch: str) -> str:
        """Map ambiguous OCR characters to their corresponding uppercase letter.

        Args:
            ch: Single character detected by OCR.

        Returns:
            Mapped uppercase letter if mapping exists or if the character is an
            alphabetic letter. Returns an empty string if the character cannot
            be mapped to a valid letter.
        """
        if not ch:
            return ''
        if ch in self.char_mapping:
            return self.char_mapping[ch]
        if ch.isalpha():
            return ch.upper()
        return ''

    def _split_multi_letter_detection(self, bbox, text: str, confidence: float) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Split a multi-letter detection into individual letters with estimated positions.
        
        Args:
            bbox: The bounding box of the multi-letter detection
            text: The detected text (e.g., "ON", "AB")
            confidence: The confidence of the detection
            
        Returns:
            List of individual letter detections with estimated positions
        """
        # Normalize each character using mapping and filter out invalid ones
        normalized_chars = [self._map_char(c) for c in text]
        clean_text = ''.join(c for c in normalized_chars if c)
        
        # Discard if no real letters were extracted
        if len(clean_text) <= 1:
            return []
        # Do NOT discard based solely on overall confidence – individual
        # letters will be filtered by their own thresholds below.  This keeps
        # low-confidence groups like "iRE" so we can still rescue an isolated
        # 'I'.
        
        # Calculate the bounding box dimensions
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        total_width = bottom_right[0] - top_left[0]
        total_height = bottom_right[1] - top_left[1]
        
        individual_letters = []
        num_letters = len(clean_text)
        
        # Estimate individual letter positions by dividing the total width
        for i, letter in enumerate(clean_text):
            # Calculate the center position for this letter
            # Assume letters are evenly distributed across the width
            letter_width = total_width / num_letters
            letter_center_x = top_left[0] + (i + 0.5) * letter_width
            letter_center_y = top_left[1] + total_height / 2
            
            # Use the same confidence as the original detection
            # but apply letter-specific thresholds
            required_confidence = self.letter_confidence_thresholds.get(letter, self.min_confidence)
            
            if confidence >= required_confidence:
                individual_letters.append((letter, (int(letter_center_x), int(letter_center_y))))
                logging.info(f"SPLIT LETTER: '{letter}' at ({int(letter_center_x)}, {int(letter_center_y)}) from '{text}' with confidence {confidence:.3f}")
        
        return individual_letters

    def detect_letters(self, image: np.ndarray) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Detects letters and their positions from an image.

        Args:
            image: The pre-processed image.

        Returns:
            A list of tuples, where each tuple contains the detected letter
            and its center coordinates (x, y).
        """
        # a list of tuples, each containing (letter, (x,y), confidence)
        all_letters: List[Tuple[str, Tuple[int, int], float]] = []
        
        # Create a dilated version of the image to make letters thicker
        kernel = np.ones((2,2), np.uint8)
        dilated_image = cv2.dilate(image, kernel, iterations=1)
        
        # Save dilated image for debugging if needed
        # cv2.imwrite("debug_screenshots/dilated.png", dilated_image)
        
        # The first parameter set will run on the dilated image
        parameter_sets = [
            # Dilated image with permissive settings
            {'image': dilated_image, 'text_threshold': 0.4, 'low_text': 0.2, 'link_threshold': 0.2, 'width_ths': 0.5, 'height_ths': 0.5},
            
            # Default-ish CRAFT parameters on original image
            {'image': image, 'text_threshold': 0.7, 'low_text': 0.4, 'link_threshold': 0.4, 'width_ths': 0.7, 'height_ths': 0.7},
            
            # More permissive on original image
            {'image': image, 'text_threshold': 0.4, 'low_text': 0.2, 'link_threshold': 0.4, 'width_ths': 0.7, 'height_ths': 0.7},

            # Try magnification on original image
            {'image': image, 'text_threshold': 0.4, 'low_text': 0.2, 'link_threshold': 0.4, 'width_ths': 0.7, 'height_ths': 0.7, 'mag_ratio': 1.5},
        ]
        
        for i, params in enumerate(parameter_sets):
            logging.info(f"Trying parameter set {i+1}: {params}")
            
            # Extract the image to be used for this run
            current_image = params.pop('image')
            
            try:
                # Build allowlist dynamically so that ambiguous characters we
                # later map (e.g. 'i', 'l', '|', '0', 'o') are not discarded
                # outright by EasyOCR.
                allow_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + ''.join(self.char_mapping.keys())
                results = self.reader.readtext(
                    current_image,
                    allowlist=allow_chars,
                    paragraph=False,
                    **params
                )
                
                logging.info(f"Parameter set {i+1} returned {len(results)} results")
                
                # Log ALL detections for debugging, regardless of confidence
                for j, (bbox, text, prob) in enumerate(results):
                    confidence = float(prob) if isinstance(prob, (int, float, str)) else 0.0
                    
                    # Calculate position for logging
                    if len(bbox) >= 4:
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        center_x = int(sum(xs) / len(xs))
                        center_y = int(sum(ys) / len(ys))
                        
                        logging.info(f"  Result {j+1}: Text='{text}', Confidence={confidence:.3f}, Position=({center_x}, {center_y})")
                        
                        # Handle single letters
                        if len(text) == 1:
                            # Map potentially ambiguous character to proper letter
                            mapped = self._map_char(text)
                            if not mapped:
                                continue  # Skip unmappable character
                            logging.info(f"    -> SINGLE LETTER: '{text}' at ({center_x}, {center_y}) with confidence {confidence:.3f}")
                            
                            letter_upper = mapped
                            required_confidence = self.letter_confidence_thresholds.get(
                                letter_upper, self.min_confidence
                            )
                            
                            if confidence >= required_confidence:
                                # Check if this letter is already detected nearby
                                duplicate = False
                                for existing_letter, (existing_x, existing_y), _ in all_letters:
                                    if (abs(center_x - existing_x) < 30 and 
                                        abs(center_y - existing_y) < 30 and 
                                        existing_letter == letter_upper):
                                        duplicate = True
                                        break
                                
                                if not duplicate:
                                    all_letters.append((letter_upper, (center_x, center_y), confidence))
                                    logging.info(f"ACCEPTED (Set {i+1}): '{letter_upper}' with confidence {confidence:.3f} (threshold: {required_confidence:.3f})")
                        
                        # Handle multi-letter detections (like "ON", "AB", etc.)
                        elif len(text) > 1:
                            logging.info(f"    -> MULTI-LETTER: '{text}' - attempting to split")
                            
                            # Split the multi-letter detection into individual letters
                            individual_letters = self._split_multi_letter_detection(bbox, text, confidence)
                            
                            for letter, position in individual_letters:
                                # Check for duplicates
                                duplicate = False
                                for existing_letter, (existing_x, existing_y), _ in all_letters:
                                    if (abs(position[0] - existing_x) < 30 and 
                                        abs(position[1] - existing_y) < 30 and 
                                        existing_letter == letter):
                                        duplicate = True
                                        break
                                
                                if not duplicate:
                                    all_letters.append((letter, position, confidence))
                        
                        # Also log any text that might be N or O related
                        if any(char in text.upper() for char in ['N', 'O', 'C', 'D', 'Q', 'G', 'P', 'R']):
                            logging.info(f"    -> CONTAINS TARGET LETTERS: '{text}' might be relevant")
                            
            except Exception as e:
                logging.warning(f"Parameter set {i+1} failed: {e}")
                continue
        
        # If we found very few letters, try one more time with even more aggressive settings
        if len(all_letters) < 4:
            logging.info("Very few letters detected, trying ultra-permissive settings...")
            try:
                # Build allowlist dynamically so that ambiguous characters we
                # later map (e.g. 'i', 'l', '|', '0', 'o') are not discarded
                # outright by EasyOCR.
                allow_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + ''.join(self.char_mapping.keys())
                results = self.reader.readtext(
                    image,
                    allowlist=allow_chars,
                    paragraph=False,
                    width_ths=0.05,
                    height_ths=0.05,
                    text_threshold=0.1,
                    low_text=0.1,
                    link_threshold=0.1,
                    mag_ratio=2.0
                )
                
                logging.info(f"Ultra-permissive mode returned {len(results)} results")
                
                for j, (bbox, text, prob) in enumerate(results):
                    confidence = float(prob) if isinstance(prob, (int, float, str)) else 0.0
                    
                    if len(bbox) >= 4:
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        center_x = int(sum(xs) / len(xs))
                        center_y = int(sum(ys) / len(ys))
                        
                        logging.info(f"  Ultra Result {j+1}: Text='{text}', Confidence={confidence:.3f}, Position=({center_x}, {center_y})")
                    
                    # Handle both single and multi-letter detections in ultra mode (mapping aware)
                    if len(text) >= 1:
                        if len(text) == 1:
                            mapped = self._map_char(text)
                            if not mapped:
                                continue
                            logging.info(f"DEBUG (Ultra): Detected '{text}' (mapped to '{mapped}') with confidence {confidence:.3f}")

                            # Use even lower thresholds for ultra mode
                            ultra_threshold = self.letter_confidence_thresholds.get(mapped, 0.1)
                            if confidence >= ultra_threshold:
                                # Duplicate check identical to earlier
                                duplicate = False
                                for existing_letter, (existing_x, existing_y), _ in all_letters:
                                    if (abs(center_x - existing_x) < 30 and 
                                        abs(center_y - existing_y) < 30 and 
                                        existing_letter == mapped):
                                        duplicate = True
                                        break
                                if not duplicate:
                                    all_letters.append((mapped, (center_x, center_y), confidence))
                                    logging.info(f"ACCEPTED (Ultra): '{mapped}' with confidence {confidence:.3f}")
                        else:
                            # Handle multi-letter in ultra mode too
                            individual_letters = self._split_multi_letter_detection(bbox, text, confidence)
                            for letter, position in individual_letters:
                                duplicate = False
                                for existing_letter, (existing_x, existing_y), _ in all_letters:
                                    if (abs(position[0] - existing_x) < 30 and 
                                        abs(position[1] - existing_y) < 30 and 
                                        existing_letter == letter):
                                        duplicate = True
                                        break
                                if not duplicate:
                                    all_letters.append((letter, position, confidence))
                                    logging.info(f"ACCEPTED (Ultra Split): '{letter}' at {position}")
                                
            except Exception as e:
                logging.warning(f"Ultra-permissive attempt failed: {e}")
        
        # Also try without any allowlist to see what EasyOCR detects freely
        logging.info("Testing without allowlist to see all possible detections...")
        try:
            # Build allowlist dynamically so that ambiguous characters we
            # later map (e.g. 'i', 'l', '|', '0', 'o') are not discarded
            # outright by EasyOCR.
            allow_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + ''.join(self.char_mapping.keys())
            results = self.reader.readtext(
                image,
                allowlist=allow_chars,
                paragraph=False,
                width_ths=0.3,
                height_ths=0.3,
                text_threshold=0.2,
                low_text=0.2,
                link_threshold=0.2
            )
            
            logging.info(f"No-allowlist mode returned {len(results)} results")
            
            for j, (bbox, text, prob) in enumerate(results):
                confidence = float(prob) if isinstance(prob, (int, float, str)) else 0.0
                
                if len(bbox) >= 4:
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    center_x = int(sum(xs) / len(xs))
                    center_y = int(sum(ys) / len(ys))
                    
                    logging.info(f"  No-allowlist Result {j+1}: Text='{text}', Confidence={confidence:.3f}, Position=({center_x}, {center_y})")
                    
                    # Process detections from no-allowlist run using same mapping logic
                    if len(text) >= 1:
                        if len(text) == 1:
                            mapped = self._map_char(text)
                            if not mapped:
                                continue
                            req_conf = self.letter_confidence_thresholds.get(mapped, self.min_confidence)
                            if confidence >= req_conf:
                                duplicate = False
                                for existing_letter, (existing_x, existing_y), _ in all_letters:
                                    if (abs(center_x - existing_x) < 30 and 
                                        abs(center_y - existing_y) < 30 and 
                                        existing_letter == mapped):
                                        duplicate = True
                                        break
                                if not duplicate:
                                    all_letters.append((mapped, (center_x, center_y), confidence))
                                    logging.info(f"ACCEPTED (No-Allowlist): '{mapped}' with confidence {confidence:.3f}")
                        else:
                            letters = self._split_multi_letter_detection(bbox, text, confidence)
                            for letter, position in letters:
                                duplicate = False
                                for existing_letter, (existing_x, existing_y), _ in all_letters:
                                    if (abs(position[0] - existing_x) < 30 and 
                                        abs(position[1] - existing_y) < 30 and 
                                        existing_letter == letter):
                                        duplicate = True
                                        break
                                if not duplicate:
                                    all_letters.append((letter, position, confidence))
                                    logging.info(f"ACCEPTED (No-Allowlist Split): '{letter}' at {position}")

        except Exception as e:
            logging.warning(f"No-allowlist attempt failed: {e}")

        unique_letters = set()
        for letter, coords, conf in all_letters:
            unique_letters.add((letter, coords, conf))

        # Final de-duplication pass
        final_detections = []
        # Sort by x, then y to process close detections together
        unique_letters = sorted(list(unique_letters), key=lambda item: (item[1][0], item[1][1]))
        
        i = 0
        while i < len(unique_letters):
            # Find a cluster of detections that are close to each other
            cluster = [unique_letters[i]]
            j = i + 1
            while j < len(unique_letters):
                dist = self._distance(unique_letters[i][1], unique_letters[j][1])
                if dist < 30: # 30px proximity threshold
                    cluster.append(unique_letters[j])
                    j += 1
                else:
                    break
            
            # From that cluster, choose the one with the highest confidence
            best_in_cluster = max(cluster, key=lambda item: item[2]) # item[2] is confidence
            final_detections.append((best_in_cluster[0], best_in_cluster[1]))
            
            # Move index past the processed cluster
            i = j
            
        logging.info("Deduplicated letters: %s", ", ".join([d[0] for d in final_detections]))
        return final_detections

    def _get_params(self, image: np.ndarray) -> List[Dict]:
        return [
            # Dilated image with permissive settings
            {'image': cv2.dilate(image, np.ones((2,2), np.uint8), iterations=1), 'text_threshold': 0.4, 'low_text': 0.2, 'link_threshold': 0.2, 'width_ths': 0.5, 'height_ths': 0.5},
            
            # Default-ish CRAFT parameters on original image
            {'image': image, 'text_threshold': 0.7, 'low_text': 0.4, 'link_threshold': 0.4, 'width_ths': 0.7, 'height_ths': 0.7},
            
            # More permissive on original image
            {'image': image, 'text_threshold': 0.4, 'low_text': 0.2, 'link_threshold': 0.4, 'width_ths': 0.7, 'height_ths': 0.7},

            # Try magnification on original image
            {'image': image, 'text_threshold': 0.4, 'low_text': 0.2, 'link_threshold': 0.4, 'width_ths': 0.7, 'height_ths': 0.7, 'mag_ratio': 1.5},
        ]

    # ------------------------------------------------------------------
    # Duplicate filtering
    # ------------------------------------------------------------------
    def _filter_duplicates(self,
                           letters: List[Tuple[str, Tuple[int, int], float]],
                           radius: int) -> List[Tuple[str, Tuple[int, int], float]]:
        """Collapse multiple detections that fall within a radius.

        By default we treat *any* detections closer than the radius as the
        same letter regardless of their label – this catches situations where
        two different letters are predicted inside one glyph (e.g. both 'T'
        and 'I' inside the central letter).
        """
        to_remove = set()
        for i in range(len(letters)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(letters)):
                dist = self._distance(letters[i][1], letters[j][1])
                if dist < radius:
                    # Keep the one with higher confidence
                    if letters[i][2] > letters[j][2]:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
        
        filtered = []
        for i, letter_data in enumerate(letters):
            if i not in to_remove:
                filtered.append(letter_data)
        return filtered

    def _distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _split_and_filter(self, text: str, bbox: np.ndarray, conf: float) -> List[Tuple[str, Tuple[int, int], float]]:
        # ...
        # (Make sure this function returns the confidence score in each tuple)
        letters: List[Tuple[str, Tuple[int, int], float]] = []
        if len(text) == 1:
            return [(text, (center_x, center_y), conf)]
        # ...
        for i, char in enumerate(text):
            split_x = int(bbox[0][0] + (i + 0.5) * letter_width)
            letters.append((char, (split_x, center_y), conf))
        return letters
