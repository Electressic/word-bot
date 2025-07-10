import requests
from bs4 import BeautifulSoup
import re
import time

def scrape_word_collect():
    """
    Scrapes all the words from wordcollectanswers.com and saves them to a file.
    """
    base_url = "https://wordcollectanswers.com/"
    all_words = []

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

        print(f"Found {len(chapter_links)} chapter links.")

        # 3. Visit each chapter page to get level links
        for i, chapter_url in enumerate(chapter_links[:3]):  # Limit to first 3 chapters for testing
            try:
                print(f"Scraping chapter {i+1}/{len(chapter_links[:3])}: {chapter_url}...")
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
                print(f"Found {len(level_links)} level links in this chapter.")

                # 4. Visit each level page to extract words
                for j, level_url in enumerate(level_links):
                    try:
                        print(f"  Scraping level {j+1}/{len(level_links)}: {level_url}...")
                        level_response = requests.get(level_url, headers=headers)
                        level_response.raise_for_status()
                        level_soup = BeautifulSoup(level_response.content, 'html.parser')

                        # Save first level page for debugging
                        if i == 0 and j == 0:
                            with open("debug_level_page.html", "w", encoding="utf-8") as f:
                                f.write(level_soup.prettify())
                            print("  Saved first level page HTML to debug_level_page.html")

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
                                        version_words.append(word)
                            
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
                                additional_words = [word.strip() for word in additional_words if word.strip() and word.strip().isalpha() and len(word.strip()) > 1]
                                if additional_words:
                                    print(f"    Additional words: {', '.join(additional_words)}")
                                    words.extend(additional_words)

                        if words:
                            # Remove duplicates from this level
                            unique_level_words = list(dict.fromkeys(words))
                            print(f"    Total found {len(unique_level_words)} unique words for this level")
                            all_words.extend(unique_level_words)
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

        # 5. Save the words to a file
        if all_words:
            # Remove duplicates and empty strings
            unique_words = list(dict.fromkeys([word for word in all_words if word and word.strip()]))
            print(f"Successfully scraped {len(unique_words)} unique words.")
            with open("word_collect_answers.txt", "w", encoding="utf-8") as f:
                for word in unique_words:
                    f.write(word + "\n")
            print("All words have been saved to word_collect_answers.txt")
        else:
            print("No words were scraped. Check the debug HTML files to understand the page structure.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    scrape_word_collect()