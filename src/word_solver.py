from itertools import permutations
from typing import List, Set

class WordSolver:
    """
    Finds all valid words from a given set of letters.
    """

    def __init__(self, words_file: str):
        """
        Initializes the WordSolver.

        Args:
            words_file: The path to the file containing the list of valid words.
        """
        with open(words_file, 'r') as f:
            self.valid_words = set(word.strip().upper() for word in f)

    def find_words(self, letters: List[str]) -> Set[str]:
        """
        Finds all valid words that can be formed from the given letters.

        Args:
            letters: A list of letters.

        Returns:
            A set of valid words.
        """
        found_words = set()
        for i in range(3, len(letters) + 1):
            for p in permutations(letters, i):
                word = "".join(p)
                if word in self.valid_words:
                    found_words.add(word)
        return found_words
