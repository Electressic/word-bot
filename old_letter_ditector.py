import easyocr
import cv2
import numpy as np
from typing import List, Tuple

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

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the image to improve letter detection.

        Args:
            image: The input image.

        Returns:
            The preprocessed image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply a binary threshold to make the letters stand out
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return thresh

    def detect_letters(self, image_path: str) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Detects letters and their positions from an image.

        Args:
            image_path: The path to the image file.

        Returns:
            A list of tuples, where each tuple contains the detected letter
            and its center coordinates (x, y).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        processed_image = self._preprocess_image(image)
        results = self.reader.readtext(
            processed_image,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        )

        letters = []
        for (bbox, text, prob) in results:
            if prob >= self.min_confidence and len(text) == 1 and text.isalpha():
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                center_x = (top_left[0] + bottom_right[0]) // 2
                center_y = (top_left[1] + bottom_right[1]) // 2
                letters.append((text.upper(), (center_x, center_y)))

        return letters
