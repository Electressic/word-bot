"""Main entry point for the Word Game Bot."""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from typing import List, Tuple, Set

import keyboard
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.config import BotConfig
from src.detection.letter_detector import LetterDetector
from src.detection.preprocessor import ImagePreprocessor
from src.solving.word_solver import WordSolver
from src.solving.layout_matcher import LayoutMatcher
from src.automation.screen_capture import ScreenCapture
from src.automation.mouse_control import MouseController
from src.automation.popup_handler import PopupHandler
from src.automation.game_window import GameWindow
from src.utils.logger import setup_logging
from src.utils.debug import DebugVisualizer


class WordGameBot:
    """Main bot controller."""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.stop_requested = False
        self.pause_requested = False
        
        # Initialize components
        self._init_components()
        self._setup_hotkeys()
    
    def _init_components(self):
        """Initialize all bot components."""
        try:
            # Window and automation
            self.window = GameWindow(self.config.screen.window_title)
            self.screen_capture = ScreenCapture()
            self.mouse = MouseController(self.window.rect)
            self.popup_handler = PopupHandler(self.config.screen, self.mouse)
            
            # Detection and solving
            self.preprocessor = ImagePreprocessor(self.config.preprocessing)
            self.letter_detector = LetterDetector(self.config)
            self.layout_matcher = LayoutMatcher(self.config.layout)
            self.word_solver = WordSolver(self.config.words_file)
            
            # Debug
            if self.config.debug:
                self.debug_viz = DebugVisualizer(self.config.debug_dir)
            else:
                self.debug_viz = None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_hotkeys(self):
        """Setup keyboard hotkeys."""
        try:
            # Stop hotkey
            keyboard.add_hotkey(
                self.config.stop_hotkey, 
                self._request_stop,
                suppress=True
            )
            self.logger.info(f"Press {self.config.stop_hotkey} to stop the bot")
            
            # Pause hotkey
            pause_key = "ctrl+shift+f7"
            keyboard.add_hotkey(
                pause_key,
                self._toggle_pause,
                suppress=True
            )
            self.logger.info(f"Press {pause_key} to pause/resume the bot")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup hotkeys: {e}")
    
    def _request_stop(self):
        """Handle stop request."""
        self.stop_requested = True
        self.logger.info("Stop requested - will halt immediately")
    
    def _toggle_pause(self):
        """Toggle pause state."""
        self.pause_requested = not self.pause_requested
        state = "paused" if self.pause_requested else "resumed"
        self.logger.info(f"Bot {state}")
    
    def _check_interrupts(self) -> bool:
        """Check for stop/pause interrupts. Returns True if should stop."""
        # Check for stop
        if self.stop_requested:
            return True
            
        # Handle pause
        while self.pause_requested and not self.stop_requested:
            time.sleep(0.1)
            
        return self.stop_requested
    
    def run_once(self) -> bool:
        """
        Run one iteration of the bot.
        
        Returns:
            True if successful, False if should retry
        """
        try:
            # Step 1: Capture screenshot
            self.logger.info("Capturing screenshot...")
            screenshot = self.screen_capture.capture_window(self.window.rect)
            
            if self._check_interrupts():
                return True
            
            # Step 2: Crop to game area
            cropped, (crop_x, crop_y) = self.preprocessor.crop_game_area(
                screenshot, 
                self.window.rect
            )
            
            if self.debug_viz:
                self.debug_viz.save_screenshot("1_raw_screenshot", screenshot)
                self.debug_viz.save_screenshot("2_cropped_area", cropped)
            
            # Step 3: Detect letters
            self.logger.info("Detecting letters...")
            detections = self.letter_detector.detect_letters(cropped)
            
            # Adjust coordinates to screen space
            adjusted_detections = [
                (letter, (x + crop_x, y + crop_y))
                for letter, (x, y) in detections
            ]
            
            # Check for popup
            if self.popup_handler.check_for_popup(len(detections)):
                self.logger.warning("Possible popup detected, attempting to close...")
                self.popup_handler.try_close_popups()
                return False  # Retry
            
            if self._check_interrupts():
                return True
            
            # Step 4: Match to layout
            self.logger.info("Matching to wheel layout...")
            
            # Try automatic layout detection first
            best_matches = None
            best_count = 0
            
            for wheel_size in self.layout_matcher.layouts.keys():
                matches = self.layout_matcher.match(adjusted_detections, wheel_size)
                if len(matches) > best_count:
                    best_count = len(matches)
                    best_matches = matches
            
            if best_matches is None or best_count < 3:
                self.logger.warning("Poor layout matching, using raw detections")
                matched_detections = adjusted_detections
            else:
                matched_detections = best_matches
                self.logger.info(f"Matched {len(matched_detections)} letters")
            
            # Debug visualization
            if self.debug_viz:
                debug_info = self.layout_matcher.get_debug_info()
                self.debug_viz.visualize_detections(
                    cropped,
                    [(letter, (x - crop_x, y - crop_y)) for letter, (x, y) in matched_detections],
                    debug_info
                )
            
            if self._check_interrupts():
                return True
            
            # Step 5: Find words
            letters = [letter for letter, _ in matched_detections]
            self.logger.info(f"Letters found: {' '.join(letters)}")
            
            words = self.word_solver.find_words(letters)
            self.logger.info(f"Found {len(words)} valid words")
            
            if not words:
                self.logger.warning("No words found!")
                return True
            
            # Step 6: Swipe words (convert set to list)
            self._swipe_words(list(words), matched_detections)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Bot error: {e}", exc_info=True)
            return True
    
    def _swipe_words(self, words: List[str], 
                     letter_positions: List[Tuple[str, Tuple[int, int]]]):
        """Swipe all found words."""
        # Build position lookup
        letter_pos_map = defaultdict(list)
        for letter, pos in letter_positions:
            letter_pos_map[letter].append(pos)
        
        # Sort words by length (longer first)
        sorted_words = sorted(words, key=len, reverse=True)
        
        # Limit number of words to swipe to avoid repetition
        max_words = min(len(sorted_words), 20)
        
        for i, word in enumerate(sorted_words[:max_words]):
            if self._check_interrupts():
                break
            
            # Build coordinate sequence
            coords = []
            letter_usage = defaultdict(int)
            valid = True
            
            for letter in word:
                available_positions = letter_pos_map.get(letter, [])
                usage_index = letter_usage[letter]
                
                if usage_index < len(available_positions):
                    coords.append(available_positions[usage_index])
                    letter_usage[letter] += 1
                else:
                    valid = False
                    break
            
            if valid and len(coords) == len(word):
                self.logger.info(f"Swiping word: {word} ({i+1}/{max_words})")
                success = self.mouse.swipe_word(coords, self.config.swipe_duration)
                
                if self.debug_viz and success:
                    time.sleep(0.2)
                    post_swipe = self.screen_capture.capture_window(self.window.rect)
                    self.debug_viz.save_screenshot(f"swipe_{word}", post_swipe)
                
                # Check for interrupts between words
                if self._check_interrupts():
                    break
                
                # Small delay between words
                time.sleep(0.2)
    
    def run(self):
        """Main bot loop."""
        self.logger.info("Starting Word Game Bot...")
        
        try:
            # Focus game window
            self.window.focus()
            
            while not self.stop_requested:
                success = self.run_once()
                
                if self.stop_requested:
                    break
                
                if not success:
                    # Retry after short delay
                    time.sleep(1)
                else:
                    # Wait before next round
                    self.logger.info("Waiting for next round...")
                    
                    # Check for interrupts during wait
                    for _ in range(30):  # 3 seconds in 0.1s chunks
                        if self._check_interrupts():
                            break
                        time.sleep(0.1)
                    
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        keyboard.unhook_all()
        self.logger.info("Bot stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Word Game Bot")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--window-title",
        type=str,
        help="Override window title"
    )
    parser.add_argument(
        "--words-file",
        type=str,
        help="Override words file path"
    )
    parser.add_argument(
        "--wheel-size",
        type=int,
        help="Force specific wheel size"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = BotConfig.from_file(args.config)
    
    # Apply command line overrides
    if args.debug:
        config.debug = True
    if args.window_title:
        config.screen.window_title = args.window_title
    if args.words_file:
        config.words_file = args.words_file
    
    # Create and run bot
    bot = WordGameBot(config)
    
    # Handle forced wheel size
    if args.wheel_size:
        # Create a wrapper function that forces the wheel size
        original_match = bot.layout_matcher.match
        bot.layout_matcher.match = lambda detections, wheel_size=None: original_match(
            detections, wheel_size=args.wheel_size
        )
    
    bot.run()


if __name__ == "__main__":
    main()