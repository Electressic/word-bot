import requests
from bs4 import BeautifulSoup
import re
import time
import sys
import os

def load_existing_words(filename="word_collect_answers.txt"):
    """Load existing words from the file to avoid duplicates."""
    existing_words = set()
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                existing_words = {line.strip().upper() for line in f if line.strip()}
            print(f"Loaded {len(existing_words)} existing words from {filename}")
        except Exception as e:
            print(f"Error reading existing file: {e}")
    else:
        print(f"No existing file found. Starting fresh.")
    return existing_words

def scrape_word_collect(start_chapter=1, end_chapter=3):
    """
    Scrapes words from wordcollectanswers.com for specified chapter range.
    
    Args:
        start_chapter (int): Starting chapter number (1-based)
        end_chapter (int): Ending chapter number (1-based, inclusive)
    """
    base_url = "https://wordcollectanswers.com/"
    filename = "word_collect_answers.txt"
    
    # Load existing words to avoid duplicates
    existing_words = load_existing_words(filename)
    new_words = []

    try:
        # 1. Fetch the main page
        print("Fetching the main page...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        main_page_response = requests.get(base_url, headers=headers)
        main_page_response.raise_for_status()

        # 2. Find all the chapter links
        print("Finding all the chapter links...")
        soup = BeautifulSoup(main_page_response.content, 'html.parser')
        
        # Look for chapter links using the pattern from the screenshots
        chapter_links = []
        
        # Find all links that contain "chapter" and "answers"
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'chapter' in href and 'answers' in href:
                if href.startswith('/'):
                    chapter_links.append(base_url.rstrip('/') + href)
                elif href.startswith('http'):
                    chapter_links.append(href)
                else:
                    chapter_links.append(base_url + href)
        
        # Remove duplicates while preserving order
        chapter_links = list(dict.fromkeys(chapter_links))

        if not chapter_links:
            print("Could not find any chapter links.")
            return

        print(f"Found {len(chapter_links)} total chapter links.")
        
        # Filter chapters based on the specified range
        chapters_to_scrape = chapter_links[start_chapter-1:end_chapter]
        print(f"Scraping chapters {start_chapter} to {end_chapter} ({len(chapters_to_scrape)} chapters)")

        # 3. Visit each chapter page to get level links
        for i, chapter_url in enumerate(chapters_to_scrape):
            actual_chapter_num = start_chapter + i
            try:
                print(f"Scraping chapter {actual_chapter_num}: {chapter_url}...")
                chapter_response = requests.get(chapter_url, headers=headers)
                chapter_response.raise_for_status()
                chapter_soup = BeautifulSoup(chapter_response.content, 'html.parser')

                # Find all level links in this chapter (like "/en/level-2.html")
                level_links = []
                for link in chapter_soup.find_all('a', href=True):
                    href = link['href']
                    if '/level-' in href and href.endswith('.html'):
                        if href.startswith('/'):
                            level_links.append(base_url.rstrip('/') + href)
                        elif href.startswith('http'):
                            level_links.append(href)
                        else:
                            level_links.append(base_url + href)

                # Remove duplicates
                level_links = list(dict.fromkeys(level_links))
                print(f"Found {len(level_links)} level links in chapter {actual_chapter_num}.")

                # 4. Visit each level page to extract words
                for j, level_url in enumerate(level_links):
                    try:
                        print(f"  Scraping level {j+1}/{len(level_links)}: {level_url}...")
                        level_response = requests.get(level_url, headers=headers)
                        level_response.raise_for_status()
                        level_soup = BeautifulSoup(level_response.content, 'html.parser')

                        words = []
                        
                        # Extract words from ALL versions (tabs)
                        # Look for all div elements with class="words" (each version has its own)
                        words_divs = level_soup.find_all('div', class_='words')
                        
                        for version_idx, words_div in enumerate(words_divs):
                            version_words = []
                            # Find all divs inside this words div
                            word_divs = words_div.find_all('div')
                            for word_div in word_divs:
                                # Extract all span elements with class="let"
                                letter_spans = word_div.find_all('span', class_='let')
                                if letter_spans:
                                    # Combine the letters to form a word
                                    word = ''.join(span.get_text(strip=True) for span in letter_spans)
                                    if word and word.isalpha():
                                        version_words.append(word.upper())
                            
                            if version_words:
                                print(f"    Version {version_idx + 1}: {', '.join(version_words)}")
                                words.extend(version_words)

                        # Extract additional words from "Here are more words, useful to this puzzle level" section
                        more_words_h3 = level_soup.find('h3', string=re.compile(r'Here are more words.*useful.*puzzle.*level', re.IGNORECASE))
                        if more_words_h3:
                            # Find the next p tag after this h3
                            next_p = more_words_h3.find_next_sibling('p')
                            if next_p:
                                additional_text = next_p.get_text(strip=True)
                                # Split by common separators and clean up
                                additional_words = re.split(r'[,\s]+', additional_text)
                                additional_words = [word.strip().upper() for word in additional_words if word.strip() and word.strip().isalpha() and len(word.strip()) > 1]
                                if additional_words:
                                    print(f"    Additional words: {', '.join(additional_words)}")
                                    words.extend(additional_words)

                        if words:
                            # Filter out words that already exist
                            level_new_words = []
                            level_duplicate_count = 0
                            
                            for word in words:
                                if word not in existing_words:
                                    level_new_words.append(word)
                                    existing_words.add(word)  # Add to set to avoid duplicates within this session
                                else:
                                    level_duplicate_count += 1
                            
                            if level_new_words:
                                new_words.extend(level_new_words)
                                print(f"    Added {len(level_new_words)} new words (skipped {level_duplicate_count} duplicates)")
                            else:
                                print(f"    No new words (all {len(words)} were duplicates)")
                        else:
                            print(f"    No words found in {level_url}")
                        
                        # Be respectful to the server
                        time.sleep(0.5)

                    except requests.exceptions.RequestException as e:
                        print(f"    Could not scrape {level_url}: {e}")
                
                # Delay between chapters
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"Could not scrape {chapter_url}: {e}")

        # 5. Save new words to the file (append mode)
        if new_words:
            # Remove duplicates from new words (shouldn't happen but just in case)
            unique_new_words = list(dict.fromkeys(new_words))
            
            with open(filename, "a", encoding="utf-8") as f:
                for word in unique_new_words:
                    f.write(word + "\n")
            
            print(f"\nSuccessfully added {len(unique_new_words)} new unique words to {filename}")
            print(f"Total words in file: {len(existing_words)}")
        else:
            print(f"\nNo new words found. All words were already in {filename}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) == 1:
        # No arguments provided, use default range
        start_chapter, end_chapter = 1, 3
        print("No chapter range specified. Using default: chapters 1-3")
        print("Usage: python scraper.py <start_chapter> <end_chapter>")
        print("Example: python scraper.py 1 5")
    elif len(sys.argv) == 3:
        try:
            start_chapter = int(sys.argv[1])
            end_chapter = int(sys.argv[2])
            
            if start_chapter < 1 or end_chapter < 1:
                print("Error: Chapter numbers must be positive integers")
                return
            
            if start_chapter > end_chapter:
                print("Error: Start chapter must be less than or equal to end chapter")
                return
                
        except ValueError:
            print("Error: Please provide valid integer chapter numbers")
            print("Usage: python scraper.py <start_chapter> <end_chapter>")
            return
    else:
        print("Error: Invalid number of arguments")
        print("Usage: python scraper.py <start_chapter> <end_chapter>")
        print("Example: python scraper.py 1 5")
        return
    
    print(f"Starting scraper for chapters {start_chapter} to {end_chapter}...")
    scrape_word_collect(start_chapter, end_chapter)

if __name__ == '__main__':
    main()