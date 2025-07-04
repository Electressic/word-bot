import collections
import time
import logging
import os
import cv2
import numpy as np
from ppadb.client import Client as AdbClient

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("word_bot.log"),
                        logging.StreamHandler()
                    ])

def find_words(letters):
    """Finds all possible words from the given letters."""
    with open("words.txt") as f:
        words = [line.strip().lower() for line in f]
    letter_counts = collections.Counter(letters.lower())
    valid_words = [word for word in words if all(collections.Counter(word)[char] <= letter_counts[char] for char in collections.Counter(word))]
    return valid_words

def filter_words_by_length(words, lengths):
    """Filters words by specific lengths."""
    filtered = collections.defaultdict(list)
    for word in words:
        if len(word) in lengths:
            filtered[len(word)].append(word)
    return filtered

def get_letter_coordinates_from_templates(device):
    """Takes a screenshot and finds letter coordinates using template matching."""
    logging.info("Taking screenshot for template matching...")
    screenshot = device.screencap()
    with open("screen.png", "wb") as f:
        f.write(screenshot)

    logging.info("Loading screenshot and templates...")
    img_rgb = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        logging.error(f"'templates' directory not found. Please create it and add letter images.")
        return None

    templates = {f.split('.')[0]: cv2.imread(os.path.join(templates_dir, f), 0) for f in os.listdir(templates_dir) if f.endswith('.png')}
    if not templates:
        logging.error(f"No .png templates found in the '{templates_dir}' directory.")
        return None

    logging.info(f"Found {len(templates)} templates: {list(templates.keys())}")
    coordinates = {}
    # Set a threshold for matching accuracy
    threshold = 0.8

    for letter, template in templates.items():
        if template is None:
            logging.warning(f"Could not load template for letter '{letter}'. Skipping.")
            continue
        
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        # Find the best match location for the letter
        if loc[0].size > 0:
            # Convert numpy types to standard python types for cleaner logs and commands
            y_coord, x_coord = loc[0][0], loc[1][0]
            point = (int(x_coord + w // 2), int(y_coord + h // 2))
            coordinates[letter] = point
            logging.info(f"Found letter '{letter}' at {point}")
        else:
            logging.warning(f"Letter '{letter}' not found on screen.")

    return coordinates

def swipe_word(device, word, coordinates):
    """
    Simulates a swipe gesture on the device for a given word.
    This chains multiple 'input swipe' commands for a continuous gesture.
    """
    if len(word) < 2:
        logging.warning(f"Word '{word}' is too short to swipe.")
        return

    word_coords = []
    for letter in word:
        if letter not in coordinates:
            logging.warning(f"Cannot swipe word '{word}'. Missing coordinate for letter '{letter}'.")
            return
        word_coords.append(coordinates[letter])

    # Build a single shell command to chain swipe events for a continuous gesture
    # The duration of each swipe segment is set to 50ms
    swipe_duration = 50
    command = ""
    for i in range(len(word_coords) - 1):
        start_x, start_y = word_coords[i]
        end_x, end_y = word_coords[i+1]
        command += f"input swipe {start_x} {start_y} {end_x} {end_y} {swipe_duration}"
        # Add a separator for the next command in the chain
        if i < len(word_coords) - 2:
            command += " && "
    
    logging.info(f"Executing swipe for '{word}': {command}")
    device.shell(command)

if __name__ == "__main__":
    client = AdbClient(host="127.0.0.1", port=5037)
    try:
        device = client.devices()[0]
        logging.info(f"Connected to device: {device.serial}")
    except IndexError:
        logging.error("No device found. Please ensure USB debugging is enabled.")
        exit()

    # The script now detects available letters from the templates folder
    coordinates = get_letter_coordinates_from_templates(device)

    if not coordinates:
        logging.error("Could not determine any letter coordinates. Exiting.")
        exit()

    available_letters = "".join(coordinates.keys())
    logging.info(f"Letters detected on screen: {available_letters}")

    word_lengths = [3, 4, 5, 6, 7, 8]  # Desired word lengths
    all_words = find_words(available_letters)
    filtered_words = filter_words_by_length(all_words, word_lengths)

    for length in sorted(filtered_words.keys()):
        logging.info(f"--- Processing {length}-letter words ---")
        for word in filtered_words[length]:
            swipe_word(device, word, coordinates)
            time.sleep(1.0) # Increased pause between words

    logging.info("All words sent. Bot finished.")