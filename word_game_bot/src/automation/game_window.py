"""Game window management."""

import time
from typing import Tuple
import pygetwindow as gw
import logging


class GameWindow:
    """Manages the game window."""
    
    def __init__(self, window_title: str):
        self.window_title = window_title
        self.logger = logging.getLogger(__name__)
        self._window = None
        self._find_window()
    
    def _find_window(self):
        """Find the game window by title."""
        windows = gw.getWindowsWithTitle(self.window_title)
        if not windows:
            raise RuntimeError(f"Window with title '{self.window_title}' not found.")
        self._window = windows[0]
        self.logger.info(f"Found window: {self.window_title}")
    
    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """Get window rectangle (x, y, width, height)."""
        if self._window is None:
            raise RuntimeError("Window not found or not initialized")
        return (self._window.left, self._window.top, 
                self._window.width, self._window.height)
    
    def focus(self):
        """Bring window to foreground."""
        if self._window is None:
            raise RuntimeError("Window not found or not initialized")
        try:
            self._window.activate()
            time.sleep(0.1)
        except Exception as e:
            self.logger.warning(f"Could not focus window: {e}")
