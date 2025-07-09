"""Game state tracking (for future enhancements)."""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from datetime import datetime


@dataclass
class GameState:
    """Tracks the current state of the game."""
    
    # Current level information
    level: Optional[int] = None
    
    # Words found in current level
    words_found: Set[str] = field(default_factory=set)
    
    # Letters available in current wheel
    current_letters: List[str] = field(default_factory=list)
    
    # Statistics
    total_words_found: int = 0
    session_start_time: datetime = field(default_factory=datetime.now)
    
    # Error tracking
    consecutive_errors: int = 0
    last_error: Optional[str] = None
    
    def reset_level(self):
        """Reset state for a new level."""
        self.words_found.clear()
        self.current_letters.clear()
        self.consecutive_errors = 0
    
    def add_word(self, word: str):
        """Record a found word."""
        self.words_found.add(word)
        self.total_words_found += 1
    
    def set_letters(self, letters: List[str]):
        """Update the current letters."""
        self.current_letters = letters.copy()
    
    def record_error(self, error_msg: str):
        """Record an error occurrence."""
        self.consecutive_errors += 1
        self.last_error = error_msg
    
    def clear_errors(self):
        """Clear error tracking after successful operation."""
        self.consecutive_errors = 0
        self.last_error = None