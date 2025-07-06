import time
import os
from typing import List, Tuple
import pyautogui
import mss
import pygetwindow as gw

import cv2
import numpy as np

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
            crop_x, crop_y = int(width * 0.2), int(height * 0.63)
            crop_width, crop_height = int(width * 0.6), int(height * 0.25)
            cropped_image = img[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
            
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            processed_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
            
            kernel = np.ones((2,2), np.uint8)
            dilated_image = cv2.dilate(processed_image, kernel, iterations = 1)

            if self.debug:
                if not os.path.exists(self.debug_dir):
                    os.makedirs(self.debug_dir)
                cv2.imwrite(os.path.join(self.debug_dir, "1_cropped_screenshot.png"), cropped_image)
                cv2.imwrite(os.path.join(self.debug_dir, "2_processed_screenshot.png"), dilated_image)

            return dilated_image, (crop_x, crop_y)

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
                mss.tools.to_png(sct_img.rgb, sct_img.size, output=debug_screenshot_path)
