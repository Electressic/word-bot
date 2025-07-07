import argparse
import os
import logging
import shutil
import keyboard
import cv2
from src.letter_detector import LetterDetector
from src.word_solver import WordSolver
from src.pc_automator import PCAutomator
from src.layout_matcher import LayoutMatcher
from collections import defaultdict

# Global flag to stop the bot
stop_bot = False

def setup_logging():
    """
    Configures logging to print to the console and write to a file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("console.log"),
            logging.StreamHandler()
        ]
    )

def stop_bot_handler():
    """
    Handler for the stop bot hotkey.
    """
    global stop_bot
    stop_bot = True
    logging.info("STOP SIGNAL RECEIVED - Bot will stop after current operation!")

def setup_hotkeys():
    """
    Sets up global hotkeys for controlling the bot.
    """
    try:
        # Register Ctrl+Shift+F8 to stop the bot
        keyboard.add_hotkey('ctrl+shift+f8', stop_bot_handler)
        logging.info("Hotkey registered: Ctrl+Shift+F8 to stop the bot")
    except Exception as e:
        logging.warning(f"Failed to register hotkeys: {e}")

def main():
    """
    Main function to run the word solver bot.
    """
    global stop_bot
    setup_logging()
    setup_hotkeys()
    
    parser = argparse.ArgumentParser(description="A bot to solve word games on Android.")
    parser.add_argument(
        "--words",
        type=str,
        default="words.txt",
        help="Path to the words file.",
    )
    parser.add_argument(
        "--window-title",
        type=str,
        default="SM-A346B",
        help="The title of the game window.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to capture screenshots of swipes."
    )
    parser.add_argument(
        "--wheel-size",
        type=int,
        default=None,
        help="Force a specific wheel size (number of letters) for layout matching."
    )
    args = parser.parse_args()

    debug_dir = "debug_screenshots"
    if args.debug:
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)
        logging.info(f"Debug mode enabled. Screenshots will be saved in '{debug_dir}'")
    else:
        debug_dir = None

    # Initialize components
    letter_detector = LetterDetector()
    word_solver = WordSolver(args.words)
    pc_automator = PCAutomator(
        args.window_title,
        debug=args.debug,
        debug_dir=debug_dir
    )
    layout_matcher = LayoutMatcher()

    # Run the bot
    try:
        if stop_bot:
            logging.info("Bot stopped before starting due to stop signal")
            return
            
        logging.info("Getting processed screenshot...")
        processed_image, (crop_x, crop_y) = pc_automator.get_processed_screenshot()

        if stop_bot:
            logging.info("Bot stopped after screenshot")
            return

        logging.info("Detecting letters...")
        letters_with_coords = letter_detector.detect_letters(processed_image)

        # If not enough letters are visible there is a good chance an overlay
        # or level-complete screen is hiding the game board. In that case try
        # to close the popup and stop this run – the user may launch us again
        # or we can later loop.
        if len(letters_with_coords) < 3:
            logging.warning("Very few letters detected – likely a pop-up/overlay. Attempting to close it.")
            pc_automator.try_close_popup()
            return

        # Adjust coordinates to be relative to the original screen for layout matching
        adjusted_letters_with_coords = [
            (letter, (x + crop_x, y + crop_y))
            for letter, (x, y) in letters_with_coords
        ]

        # ------------------------------------------------------------------
        # Layout selection: allow user override via --wheel-size. If provided,
        # use that template directly; otherwise fall back to automatic
        # selection logic.
        # ------------------------------------------------------------------
        if args.wheel_size is not None:
            forced_size = args.wheel_size
            if str(forced_size) not in layout_matcher.layouts:
                logging.error(f"Wheel size {forced_size} not supported. Available: {', '.join(layout_matcher.layouts.keys())}")
                return
            matched_letters_with_coords = layout_matcher.match(
                adjusted_letters_with_coords,
                wheel_size=forced_size
            )
            logging.info(f"Forced {forced_size}-letter wheel template with {len(matched_letters_with_coords)} matched detections.")
        else:
            # ------------------------------------------------------------------
            # Automatic template selection (as implemented earlier)
            # ------------------------------------------------------------------
            best_layout = None
            best_match_count = -1
            min_size_diff = float('inf')

            raw_detection_count = len(adjusted_letters_with_coords)

            # Try all available layouts
            for size_str in layout_matcher.layouts.keys():
                size = int(size_str)
                matches = layout_matcher.match(
                    adjusted_letters_with_coords,
                    wheel_size=size
                )
                num_matches = len(matches)

                # The current layout is better if:
                # 1. It matches more letters than the best one so far.
                # 2. It matches the same number of letters, but its wheel size is
                #    closer to the number of raw detections found by OCR.
                size_diff = abs(size - raw_detection_count)
                if num_matches > best_match_count or \
                   (num_matches == best_match_count and size_diff < min_size_diff):
                    best_match_count = num_matches
                    min_size_diff = size_diff
                    best_layout = {
                        "size": size,
                        "matches": matches
                    }

            # Re-run match on the chosen size so that LayoutMatcher stores the
            # slot coordinates of the winning layout for later debugging output.
            if best_layout and best_match_count > 0:
                best_size = best_layout["size"]
                # This re-run is for the side-effect of populating `layout_matcher.slots_abs`
                matched_letters_with_coords = layout_matcher.match(
                    adjusted_letters_with_coords,
                    wheel_size=best_size
                )
                logging.info(f"Selected {best_size}-letter wheel template with {len(matched_letters_with_coords)} matched detections.")
            else:
                # Fallback – no template produced any matches within tolerance.
                matched_letters_with_coords = adjusted_letters_with_coords
                logging.warning("No template matched – using raw detections.")

        # ------------------------------------------------------------------
        # Debug: visualise detected letters on the processed image
        # ------------------------------------------------------------------
        if args.debug and debug_dir:
            annotated = cv2.cvtColor(processed_image.copy(), cv2.COLOR_GRAY2BGR)
            # Draw template slots for debugging (blue)
            # The slots are in absolute screen coordinates, so we need to adjust them
            # for the cropped debug image.
            for (x_s, y_s) in layout_matcher.get_last_slots():
                cv2.circle(annotated, (int(x_s - crop_x), int(y_s - crop_y)), 50, (255, 0, 0), 1)

            # Draw matched letters, which are also in absolute screen coordinates.
            for letter, (x, y) in matched_letters_with_coords:
                # Adjust coordinates for the cropped debug image.
                x_adj, y_adj = x - crop_x, y - crop_y
                # Outer circle shows duplicate-suppression radius (debug aid)
                cv2.circle(annotated, (x_adj, y_adj), 50, (0, 255, 255), 1)  # yellow-ish
                # Inner circle marks the detection centre
                cv2.circle(annotated, (x_adj, y_adj), 12, (0, 255, 0), 2)
                cv2.putText(annotated, letter, (x_adj - 10, y_adj - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(debug_dir, "3_detected_letters.png"), annotated)

        if stop_bot:
            logging.info("Bot stopped after letter detection")
            return
        
        detected_letters = [letter for letter, coords in matched_letters_with_coords]
        logging.info(f"Detected letters: {', '.join(detected_letters)}")

        if stop_bot:
            logging.info("Bot stopped after letter detection")
            return

        logging.info("Finding words...")
        words = word_solver.find_words(detected_letters)
        logging.info(f"Found {len(words)} words: {', '.join(words)}")

        if stop_bot:
            logging.info("Bot stopped after word finding")
            return

        logging.info("Swiping words...")
        # ------------------------------------------------------------------
        # Build a mapping from each letter to *all* of its positions so we can
        # handle words that contain the same letter multiple times (e.g.
        # "ACCORD" with two Cs). A simple dict would overwrite duplicates;
        # using lists preserves every occurrence.
        # ------------------------------------------------------------------
        letter_coords_map = defaultdict(list)  # letter -> list of (x, y)
        for letter, coords in matched_letters_with_coords:
            letter_coords_map[letter].append(coords)

        for word in sorted(list(words)):
            if stop_bot:
                logging.info(f"Bot stopped before swiping word: {word}")
                break

            # Build coordinate list while accounting for duplicate letters
            temp_usage = defaultdict(int)  # how many times we've used each letter in this word
            coordinates = []
            valid = True
            for char in word:
                occurrences = letter_coords_map.get(char, [])
                usage_idx = temp_usage[char]
                if usage_idx < len(occurrences):
                    coordinates.append(occurrences[usage_idx])
                    temp_usage[char] += 1
                else:
                    # Not enough occurrences of this letter detected – skip the word
                    valid = False
                    break

            if valid and len(coordinates) == len(word):
                logging.info(f"Swiping: {word}")
                pc_automator.swipe_word(coordinates, word)

                if stop_bot:
                    logging.info(f"Bot stopped after swiping word: {word}")
                    break

        if not stop_bot:
            logging.info("Done!")
        else:
            logging.info("Bot stopped by user!")

    except (FileNotFoundError, RuntimeError) as e:
        logging.error(f"Error: {e}")
    except KeyboardInterrupt:
        logging.info("Bot stopped by Ctrl+C")
    finally:
        # Clean up keyboard listeners
        keyboard.unhook_all()


if __name__ == "__main__":
    main()
