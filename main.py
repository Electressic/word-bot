import argparse
import os
import logging
import shutil
import keyboard
from src.letter_detector import LetterDetector
from src.word_solver import WordSolver
from src.pc_automator import PCAutomator

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
        
        if stop_bot:
            logging.info("Bot stopped after letter detection")
            return
        
        # Adjust coordinates to be relative to the original screen
        adjusted_letters_with_coords = [
            (letter, (x + crop_x, y + crop_y))
            for letter, (x, y) in letters_with_coords
        ]

        detected_letters = [letter for letter, coords in adjusted_letters_with_coords]
        logging.info(f"Detected letters: {', '.join(detected_letters)}")

        if stop_bot:
            logging.info("Bot stopped after coordinate adjustment")
            return

        logging.info("Finding words...")
        words = word_solver.find_words(detected_letters)
        logging.info(f"Found {len(words)} words: {', '.join(words)}")

        if stop_bot:
            logging.info("Bot stopped after word finding")
            return

        logging.info("Swiping words...")
        letter_coords = {letter: coords for letter, coords in adjusted_letters_with_coords}
        for word in sorted(list(words)):
            if stop_bot:
                logging.info(f"Bot stopped before swiping word: {word}")
                break
                
            coordinates = [letter_coords[char] for char in word if char in letter_coords]
            if len(coordinates) == len(word):
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
