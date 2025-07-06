import time
import os
from typing import List, Tuple
import pyautogui
import mss
import pygetwindow as gw

import cv2
import numpy as np
import logging

class PCAutomator:
    """
    Automates actions on a PC for the word game.
    """

    def __init__(self, window_title: str, debug: bool = False, debug_dir: str = "debug_screenshots"):
        """
        Initializes the PCAutomator.

        Args:
            window_title: The title of the game window.
            debug: Whether to run in debug mode.
            debug_dir: The directory to save debug files to.
        """
        self.window_title = window_title
        self.debug = debug
        self.debug_dir = debug_dir
        self.window = self._get_window()

    def _get_window(self):
        """
        Gets the game window using its title.
        """
        windows = gw.getWindowsWithTitle(self.window_title)
        if not windows:
            raise RuntimeError(f"Window with title '{self.window_title}' not found.")
        return windows[0]

    def get_processed_screenshot(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Takes a screenshot, crops it, and preprocesses it.

        Returns:
            A tuple containing the processed image and the crop coordinates (x, y).
        """
        with mss.mss() as sct:
            monitor = {
                "top": self.window.top,
                "left": self.window.left,
                "width": self.window.width,
                "height": self.window.height,
            }
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            
            # Crop and preprocess the image
            height, width, _ = img.shape
            # The letter wheel sometimes sits slightly higher than the
            # previous crop assumed.  Start 5 % higher and capture 5 % more
            # height so both the top and bottom rows remain visible.
            crop_x, crop_y = int(width * 0.2), int(height * 0.625)
            crop_width, crop_height = int(width * 0.6), int(height * 0.275)
            cropped_image = img[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
            
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            
            # Improved preprocessing for better N and O detection
            # Apply light denoising first
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Try multiple preprocessing approaches with gentler settings
            # Approach 1: Adaptive threshold with larger block size (better for circular letters)
            blurred1 = cv2.GaussianBlur(denoised, (3, 3), 0)
            processed_image1 = cv2.adaptiveThreshold(
                blurred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 21, 10  # Larger block size, more conservative
            )
            
            # Approach 2: OTSU threshold (good for varying lighting)
            blurred2 = cv2.GaussianBlur(denoised, (3, 3), 0)
            _, processed_image2 = cv2.threshold(
                blurred2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            
            # Approach 3: Gentler manual threshold
            _, processed_image3 = cv2.threshold(denoised, 140, 255, cv2.THRESH_BINARY_INV)
            
            # Approach 4: Conservative threshold for light text
            _, processed_image4 = cv2.threshold(denoised, 160, 255, cv2.THRESH_BINARY_INV)
            
            # Combine approaches more carefully - start with most conservative
            processed_image = cv2.bitwise_or(processed_image2, processed_image3)
            processed_image = cv2.bitwise_or(processed_image, processed_image4)
            
            # Add some of the adaptive threshold results but more selectively
            # Use opening to clean up noise first
            kernel_clean = np.ones((2,2), np.uint8)
            processed_image1_cleaned = cv2.morphologyEx(processed_image1, cv2.MORPH_OPEN, kernel_clean)
            
            # Only add adaptive threshold pixels that are part of larger structures
            kernel_dilate = np.ones((3,3), np.uint8)
            processed_image1_dilated = cv2.dilate(processed_image1_cleaned, kernel_dilate, iterations=1)
            processed_image1_filtered = cv2.bitwise_and(processed_image1_cleaned, processed_image1_dilated)
            
            processed_image = cv2.bitwise_or(processed_image, processed_image1_filtered)
            
            # Very light morphological closing to connect small gaps without destroying letter shapes
            kernel_close = np.ones((2,2), np.uint8)
            final_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel_close)
            
            # Light opening to remove very small noise
            kernel_open = np.ones((1,1), np.uint8) 
            final_image = cv2.morphologyEx(final_image, cv2.MORPH_OPEN, kernel_open)

            if self.debug:
                if not os.path.exists(self.debug_dir):
                    os.makedirs(self.debug_dir)
                cv2.imwrite(os.path.join(self.debug_dir, "1_cropped_screenshot.png"), cropped_image)
                cv2.imwrite(os.path.join(self.debug_dir, "2_processed_screenshot.png"), final_image)

            return final_image, (crop_x, crop_y)

    def swipe_word(self, coordinates: List[Tuple[int, int]], word: str):
        """
        Swipes a word by moving the mouse over the letters.

        Args:
            coordinates: A list of (x, y) coordinates for the word's letters.
            word: The word being swiped (used for debug screenshot naming).
        """
        if len(coordinates) < 2:
            return

        # Move to the starting position
        start_x, start_y = coordinates[0]
        pyautogui.moveTo(self.window.left + start_x, self.window.top + start_y)
        pyautogui.mouseDown()

        # Swipe through the letters
        for x, y in coordinates[1:]:
            pyautogui.moveTo(self.window.left + x, self.window.top + y, duration=0.1)
        
        pyautogui.mouseUp()

        if self.debug:
            time.sleep(0.2)
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir)
            debug_screenshot_path = os.path.join(self.debug_dir, f"{word}_swiped.png")
            with mss.mss() as sct:
                monitor = {
                    "top": self.window.top,
                    "left": self.window.left,
                    "width": self.window.width,
                    "height": self.window.height,
                }
                sct_img = sct.grab(monitor)
                img_array = np.array(sct_img)
                cv2.imwrite(debug_screenshot_path, img_array)

    # ------------------------------------------------------------------
    # Popup handling helpers
    # ------------------------------------------------------------------

    def _click_absolute(self, abs_x: int, abs_y: int):
        """Move the mouse to an absolute screen coordinate and click."""
        pyautogui.moveTo(abs_x, abs_y)
        pyautogui.click()

    def try_close_popup(self) -> bool:
        """Attempt to close common in-game pop-ups.

        This method is intentionally heuristic – it simply clicks a few
        well-known close / continue areas:

        1. The small ⓧ icon that most overlays display in the top-right
           (~95 % width, ~10 % height inside the game window).
        2. The big green "NEXT LEVEL" button that appears after finishing
           a round (~50 % width, ~85 % height).

        Returns True if at least one click was performed.
        """
        if self.window is None:
            # Safety – should not happen because we set it in __init__
            return False

        performed_click = False

        # Coordinates relative to window
        w, h = self.window.width, self.window.height

        candidates = [
            (int(w * 0.95), int(h * 0.08)),   # typical small X icon
            (int(w * 0.5),  int(h * 0.88)),   # NEXT LEVEL / CONTINUE button
        ]

        for rel_x, rel_y in candidates:
            abs_x = self.window.left + rel_x
            abs_y = self.window.top  + rel_y
            logging.info(f"Attempting to close popup by clicking at ({abs_x}, {abs_y})")
            self._click_absolute(abs_x, abs_y)
            performed_click = True
            # Give UI a brief moment to react
            time.sleep(0.3)

        return performed_click
