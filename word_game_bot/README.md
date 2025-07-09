# Word Game Bot

An automated bot for solving word games using computer vision and OCR.

## Project Structure

```
word_game_bot/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   ├── game_state.py      # Game state tracking
│   │   └── exceptions.py      # Custom exceptions
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── ocr_engine.py      # OCR wrapper with multiple strategies
│   │   ├── letter_detector.py # Letter detection logic
│   │   └── preprocessor.py    # Image preprocessing
│   ├── solving/
│   │   ├── __init__.py
│   │   ├── word_solver.py     # Word finding logic
│   │   └── layout_matcher.py  # Layout matching
│   ├── automation/
│   │   ├── __init__.py
│   │   ├── screen_capture.py  # Screenshot handling
│   │   ├── mouse_control.py   # Mouse automation
│   │   └── popup_handler.py   # Popup detection/closing
│   └── utils/
│       ├── __init__.py
│       ├── logger.py          # Logging setup
│       └── debug.py           # Debug utilities
├── data/
│   ├── layouts.json
│   └── words.txt
├── tests/
│   └── ...
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the bot:
```bash
python main.py
```

## Features

- Computer vision-based letter detection
- OCR with multiple strategies
- Automatic word solving
- Mouse automation for word swiping
- Popup handling
- Debug utilities

## Development

This project is structured as a modular word game bot with separate components for:
- Detection (OCR and image processing)
- Solving (word finding algorithms)
- Automation (mouse control and screen capture)
- Core utilities (configuration, state management)
