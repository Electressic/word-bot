"""Word solver with efficient word finding algorithm."""

import os
from typing import List, Set, Dict, Tuple, Optional
from itertools import permutations
import logging


class TrieNode:
    """Node in a Trie data structure for efficient word lookup."""
    
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word: Optional[str] = None


class WordSolver:
    """Efficiently finds valid words from given letters."""
    
    def __init__(self, words_file: str):
        self.logger = logging.getLogger(__name__)
        self.min_word_length = 3
        
        # Load words into Trie for efficient lookup
        self.trie = TrieNode()
        self.all_words = set()
        self._load_words(words_file)
    
    def _load_words(self, words_file: str):
        """Load words from file into Trie structure."""
        if not os.path.exists(words_file):
            raise FileNotFoundError(f"Words file not found: {words_file}")
        
        count = 0
        with open(words_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().upper()
                if len(word) >= self.min_word_length:
                    self._add_to_trie(word)
                    self.all_words.add(word)
                    count += 1
        
        self.logger.info(f"Loaded {count} words into solver")
    
    def _add_to_trie(self, word: str):
        """Add a word to the Trie."""
        node = self.trie
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.word = word
    
    def find_words(self, letters: List[str]) -> Set[str]:
        """
        Find all valid words that can be formed from the given letters.
        
        Args:
            letters: List of available letters
            
        Returns:
            Set of valid words
        """
        if len(letters) < self.min_word_length:
            return set()
        
        # Count available letters
        letter_counts = {}
        for letter in letters:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
        
        found_words = set()
        
        # Method 1: Use Trie for efficient search (for smaller letter sets)
        if len(letters) <= 7:
            self._find_words_trie(self.trie, letter_counts, "", found_words)
        
        # Method 2: Check permutations (for larger letter sets or if Trie is incomplete)
        if len(letters) <= 8:
            for length in range(self.min_word_length, min(len(letters) + 1, 8)):
                for perm in permutations(letters, length):
                    word = ''.join(perm)
                    if word in self.all_words:
                        found_words.add(word)
        
        # Filter out subwords if we have longer words with same letters
        filtered_words = self._filter_subwords(found_words)
        
        self.logger.info(f"Found {len(filtered_words)} words from {len(letters)} letters")
        return filtered_words
    
    def _find_words_trie(self, node: TrieNode, available: Dict[str, int], 
                        current: str, found: Set[str]):
        """Recursively find words using Trie."""
        # Check if current path forms a word
        if node.is_word and len(current) >= self.min_word_length and node.word:
            found.add(node.word)
        
        # Try each available letter
        for letter, count in available.items():
            if count > 0 and letter in node.children:
                # Use this letter
                available[letter] -= 1
                self._find_words_trie(
                    node.children[letter],
                    available,
                    current + letter,
                    found
                )
                # Backtrack
                available[letter] += 1
    
    def _filter_subwords(self, words: Set[str]) -> Set[str]:
        """
        Filter out shorter words that are subsets of longer words.
        Keep all words of the same maximum length.
        """
        if not words:
            return words
        
        # Group by sorted letters
        letter_groups = {}
        for word in words:
            key = ''.join(sorted(word))
            if key not in letter_groups:
                letter_groups[key] = []
            letter_groups[key].append(word)
        
        # Keep only longest words in each group
        filtered = set()
        for group in letter_groups.values():
            max_len = max(len(w) for w in group)
            filtered.update(w for w in group if len(w) == max_len)
        
        return filtered
    
    def is_valid_word(self, word: str) -> bool:
        """Check if a word is valid."""
        return word.upper() in self.all_words
    
    def get_word_score(self, word: str) -> int:
        """
        Calculate a score for a word (for prioritizing swipes).
        Longer words get higher scores.
        """
        base_score = len(word)
        
        # Bonus for less common letters
        letter_scores = {
            'Q': 10, 'Z': 10, 'X': 8, 'J': 8,
            'K': 5, 'V': 4, 'W': 4, 'Y': 4,
            'B': 3, 'C': 3, 'F': 3, 'H': 3, 'M': 3, 'P': 3,
            'D': 2, 'G': 2, 'L': 2, 'N': 2, 'R': 2, 'S': 2, 'T': 2,
            'A': 1, 'E': 1, 'I': 1, 'O': 1, 'U': 1
        }
        
        bonus = sum(letter_scores.get(c, 1) for c in word.upper())
        
        return base_score * 10 + bonus
    
    def sort_words_by_priority(self, words: Set[str]) -> List[str]:
        """Sort words by priority for swiping."""
        # Sort by score (descending) and then alphabetically
        return sorted(words, key=lambda w: (-self.get_word_score(w), w))