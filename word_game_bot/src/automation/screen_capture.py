"""Screen capture functionality."""

import mss
import numpy as np
from typing import Tuple
import logging


class ScreenCapture:
    """Handles screenshot capture."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def capture_window(self, window_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Capture screenshot of a window.
        
        Args:
            window_rect: (x, y, width, height) of the window
            
        Returns:
            Screenshot as numpy array
        """
        x, y, width, height = window_rect
        
        with mss.mss() as sct:
            monitor = {
                "top": y,
                "left": x,
                "width": width,
                "height": height
            }
            sct_img = sct.grab(monitor)
            
            # Convert to numpy array
            img = np.array(sct_img)
            
            # Convert BGRA to BGR
            if img.shape[2] == 4:
                img = img[:, :, :3]
            
            return img
