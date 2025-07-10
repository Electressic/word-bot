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

        # 2. Find all the chapter links with better debugging
        print("Finding all the chapter links...")
        soup = BeautifulSoup(main_page_response.content, 'html.parser')
        
        # Debug: Save the HTML to see what we're working with
        with open("debug_main_page.html", "w", encoding="utf-8") as f:
            f.write(soup.prettify())
        print("Saved main page HTML to debug_main_page.html for inspection")
        
        # Try multiple approaches to find chapter links
        chapter_links = []
        
        # Approach 1: Look for entry-content div
        content_div = soup.find('div', class_='entry-content')
        if content_div:
            for link in content_div.find_all('a', href=re.compile(r'chapter')):
                chapter_links.append(link['href'])
        
        # Approach 2: Look for any links containing 'chapter'
        if not chapter_links:
            for link in soup.find_all('a', href=re.compile(r'chapter')):
                chapter_links.append(link['href'])
        
        # Approach 3: Look for links in article or main content areas
        if not chapter_links:
            content_areas = soup.find_all(['article', 'main', 'div'], class_=re.compile(r'content|post|entry'))
            for area in content_areas:
                for link in area.find_all('a', href=re.compile(r'chapter')):
                    chapter_links.append(link['href'])
        
        # Remove duplicates while preserving order
        chapter_links = list(dict.fromkeys(chapter_links))

        if not chapter_links:
            print("Could not find any chapter links. Checking for any game-related links...")
            # Look for any links that might be game levels
            all_links = soup.find_all('a', href=True)
            for link in all_links:
                href = link['href']
                if any(keyword in href.lower() for keyword in ['level', 'answer', 'solution', 'pack']):
                    chapter_links.append(href)
            
            if not chapter_links:
                print("No relevant links found. Please check debug_main_page.html to see the page structure.")
                return

        print(f"Found {len(chapter_links)} chapter links. Now scraping each chapter...")

        # 3. Visit each chapter page and extract words
        for i, link in enumerate(chapter_links[:5]):  # Limit to first 5 for testing
            # Handle relative URLs
            if link.startswith('/'):
                chapter_url = base_url.rstrip('/') + link
            elif link.startswith('http'):
                chapter_url = link
            else:
                chapter_url = base_url + link
                
            try:
                print(f"Scraping {i+1}/{len(chapter_links[:5])}: {chapter_url}...")
                chapter_response = requests.get(chapter_url, headers=headers)
                chapter_response.raise_for_status()
                chapter_soup = BeautifulSoup(chapter_response.content, 'html.parser')

                # Save first chapter page for debugging
                if i == 0:
                    with open("debug_chapter_page.html", "w", encoding="utf-8") as f:
                        f.write(chapter_soup.prettify())
                    print("Saved first chapter page HTML to debug_chapter_page.html")

                # Try multiple approaches to extract words
                words = []
                
                # Approach 1: Look for word-answers div
                answer_div = chapter_soup.find('div', class_='word-answers')
                if answer_div:
                    words = [li.get_text(strip=True) for li in answer_div.find_all('li')]
                
                # Approach 2: Look for any div containing answers
                if not words:
                    answer_divs = chapter_soup.find_all('div', class_=re.compile(r'answer|solution|word'))
                    for div in answer_divs:
                        word_elements = div.find_all(['li', 'span', 'p', 'div'])
                        for elem in word_elements:
                            text = elem.get_text(strip=True)
                            if text and len(text) > 1 and text.isalpha():
                                words.append(text)
                
                # Approach 3: Look in common content areas
                if not words:
                    content_areas = chapter_soup.find_all(['article', 'main', 'div'], class_=re.compile(r'content|entry|post'))
                    for area in content_areas:
                        # Look for lists of words
                        lists = area.find_all(['ul', 'ol'])
                        for ul in lists:
                            for li in ul.find_all('li'):
                                text = li.get_text(strip=True)
                                if text and len(text) > 1 and text.isalpha():
                                    words.append(text)

                if words:
                    print(f"Found {len(words)} words from {chapter_url}")
                    all_words.extend(words)
                else:
                    print(f"No words found in {chapter_url}")
                
                # Be respectful to the server
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"Could not scrape {chapter_url}: {e}")

        # 4. Save the words to a file
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