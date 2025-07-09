"""Mouse control and automation."""

import time
from typing import List, Tuple
import pyautogui
import logging


class MouseController:
    """Handles mouse movements and clicks."""
    
    def __init__(self, window_rect: Tuple[int, int, int, int]):
        self.window_x, self.window_y, self.window_width, self.window_height = window_rect
        self.logger = logging.getLogger(__name__)
        
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01
    
    def swipe_word(self, coordinates: List[Tuple[int, int]], 
                   duration_per_letter: float = 0.1) -> bool:
        """
        Swipe through a word by dragging the mouse.
        
        Args:
            coordinates: List of (x, y) coordinates for each letter
            duration_per_letter: Time to spend moving between letters
            
        Returns:
            True if swipe completed successfully
        """
        if len(coordinates) < 2:
            self.logger.warning("Need at least 2 coordinates to swipe")
            return False
        
        try:
            # Convert relative to absolute coordinates
            abs_coords = [
                (self.window_x + x, self.window_y + y) 
                for x, y in coordinates
            ]
            
            # Move to start position
            start_x, start_y = abs_coords[0]
            pyautogui.moveTo(start_x, start_y, duration=0.2)
            
            # Press mouse down
            pyautogui.mouseDown()
            
            # Drag through each letter
            for i, (x, y) in enumerate(abs_coords[1:], 1):
                pyautogui.moveTo(x, y, duration=duration_per_letter)
                
                # Small pause at each letter for better recognition
                if i < len(abs_coords) - 1:
                    time.sleep(0.05)
            
            # Release mouse
            pyautogui.mouseUp()
            
            # Small pause after swipe
            time.sleep(0.1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Swipe failed: {e}")
            pyautogui.mouseUp()  # Ensure mouse is released
            return False
    
    def click_position(self, x: int, y: int):
        """Click at a specific position relative to the window."""
        abs_x = self.window_x + x
        abs_y = self.window_y + y
        
        try:
            pyautogui.click(abs_x, abs_y)
            self.logger.debug(f"Clicked at ({abs_x}, {abs_y})")
        except Exception as e:
            self.logger.error(f"Click failed: {e}")
