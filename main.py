import argparse
import os
import logging
import shutil
from src.letter_detector import LetterDetector
from src.word_solver import WordSolver
from src.android_automator import AndroidAutomator

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

def main():
    """
    Main function to run the word solver bot.
    """
    setup_logging()
    parser = argparse.ArgumentParser(description="A bot to solve word games on Android.")
    parser.add_argument(
        "--screenshot",
        type=str,
        default="screenshot.png",
        help="Path to save the screenshot.",
    )
    parser.add_argument(
        "--words",
        type=str,
        default="words.txt",
        help="Path to the words file.",
    )
    parser.add_argument(
        "--adb-path",
        type=str,
        default="adb",
        help="Path to the ADB executable.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to capture screenshots of swipes."
    )
    args = parser.parse_args()

    if args.debug:
        debug_dir = "debug_screenshots"
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)
        logging.info(f"Debug mode enabled. Screenshots will be saved in '{debug_dir}'")

    # Initialize components
    letter_detector = LetterDetector()
    word_solver = WordSolver(args.words)
    android_automator = AndroidAutomator(args.adb_path, debug=args.debug)

    # Run the bot
    try:
        logging.info("Taking screenshot...")
        android_automator.take_screenshot(args.screenshot)

        logging.info("Detecting letters...")
        letters_with_coords = letter_detector.detect_letters(args.screenshot)
        detected_letters = [letter for letter, coords in letters_with_coords]
        logging.info(f"Detected letters: {', '.join(detected_letters)}")

        logging.info("Finding words...")
        words = word_solver.find_words(detected_letters)
        logging.info(f"Found {len(words)} words: {', '.join(words)}")

        logging.info("Swiping words...")
        letter_coords = {letter: coords for letter, coords in letters_with_coords}
        for word in sorted(list(words)):
            coordinates = [letter_coords[char] for char in word if char in letter_coords]
            if len(coordinates) == len(word):
                logging.info(f"Swiping: {word}")
                android_automator.swipe_word(coordinates, word)

        logging.info("Done!")

    except (FileNotFoundError, RuntimeError) as e:
        logging.error(f"Error: {e}")
    finally:
        # Clean up the screenshot file
        if os.path.exists(args.screenshot):
            os.remove(args.screenshot)

if __name__ == "__main__":
    main()
