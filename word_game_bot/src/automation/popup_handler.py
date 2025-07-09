"""Popup detection and handling."""

import time
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import ScreenConfig
    from src.automation.mouse_control import MouseController


class PopupHandler:
    """Handles game popups and overlays."""
    
    def __init__(self, config: 'ScreenConfig', mouse: 'MouseController'):
        self.config = config
        self.mouse = mouse
        self.logger = logging.getLogger(__name__)
    
    def try_close_popups(self) -> bool:
        """
        Attempt to close common popups.
        
        Returns:
            True if any popup close was attempted
        """
        clicked = False
        
        for rel_x, rel_y in self.config.popup_close_positions:
            # Convert percentage to pixel coordinates
            x = int(self.mouse.window_width * rel_x)
            y = int(self.mouse.window_height * rel_y)
            
            self.logger.info(f"Attempting to close popup at ({x}, {y})")
            self.mouse.click_position(x, y)
            clicked = True
            
            # Wait for UI to respond
            time.sleep(0.3)
        
        return clicked
    
    def check_for_popup(self, num_detections: int) -> bool:
        """
        Check if a popup might be blocking the game.
        
        Args:
            num_detections: Number of letters detected
            
        Returns:
            True if popup is likely present
        """
        # If very few letters are detected, a popup might be blocking
        return num_detections < 3