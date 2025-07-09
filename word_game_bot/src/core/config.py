"""Configuration management for the Word Game Bot."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
import os


@dataclass
class OCRConfig:
    """OCR-specific configuration."""
    min_confidence: float = 0.3
    languages: List[str] = field(default_factory=lambda: ['en'])
    gpu: bool = True
    
    # Character mappings for ambiguous detections
    char_mappings: Dict[str, str] = field(default_factory=lambda: {
        '1': 'I', '|': 'I', 'l': 'I', 'i': 'I',
        '0': 'O', 'o': 'O',
    })
    
    # Letter-specific confidence thresholds
    letter_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'I': 0.05,  # Very low for problematic I
        'N': 0.2,
        'O': 0.2,
        'C': 0.3,
        'D': 0.3,
        'Q': 0.3,
        'G': 0.3,
        'P': 0.3,
        'R': 0.3,
        'T': 0.5,
        'A': 0.6,
        'E': 0.6,
        'DEFAULT': 0.4
    })
    
    # OCR parameter sets for multiple attempts
    parameter_sets: List[Dict] = field(default_factory=lambda: [
        # Set 1: Aggressive for thin letters like 'I'
        {
            'text_threshold': 0.1,
            'low_text': 0.05,
            'link_threshold': 0.1,
            'width_ths': 0.3,
            'height_ths': 0.3,
            'mag_ratio': 2.0
        },
        # Set 2: Balanced
        {
            'text_threshold': 0.4,
            'low_text': 0.2,
            'link_threshold': 0.3,
            'width_ths': 0.5,
            'height_ths': 0.5,
        },
        # Set 3: Conservative
        {
            'text_threshold': 0.7,
            'low_text': 0.4,
            'link_threshold': 0.4,
            'width_ths': 0.7,
            'height_ths': 0.7,
        }
    ])


@dataclass
class PreprocessingConfig:
    """Image preprocessing configuration."""
    # Denoising
    denoise_strength: int = 10
    
    # Thresholding options
    threshold_methods: List[str] = field(default_factory=lambda: [
        'adaptive_gaussian',
        'otsu',
        'manual_140',
        'manual_160'
    ])
    
    # Morphological operations
    morph_kernel_size: Tuple[int, int] = (2, 2)
    dilate_iterations: int = 1
    
    # Blur settings
    blur_kernel_size: Tuple[int, int] = (3, 3)
    
    # Black on white conversion
    enable_black_on_white: bool = True


@dataclass
class ScreenConfig:
    """Screen capture and window configuration."""
    window_title: str = "SM-A346B"
    
    # Crop percentages for the letter wheel area
    crop_x_start: float = 0.2
    crop_y_start: float = 0.625
    crop_width: float = 0.6
    crop_height: float = 0.275
    
    # Popup close button locations (as percentages)
    popup_close_positions: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.95, 0.08),  # X button
        (0.5, 0.88),   # Next level button
    ])


@dataclass
class LayoutConfig:
    """Layout matching configuration."""
    tolerance_px: int = 60
    deduplication_radius: int = 30
    
    # Path to layouts file
    layouts_file: str = "data/layouts.json"


@dataclass
class BotConfig:
    """Main bot configuration."""
    # Sub-configurations
    ocr: OCRConfig = field(default_factory=OCRConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    
    # General settings
    debug: bool = False
    debug_dir: str = "debug_screenshots"
    words_file: str = "data/words.txt"
    
    # Hotkeys
    stop_hotkey: str = "ctrl+shift+f8"
    
    # Timing
    swipe_duration: float = 0.1
    popup_wait_time: float = 0.3
    
    @classmethod
    def from_file(cls, config_file: str) -> 'BotConfig':
        """Load configuration from a JSON file."""
        if not os.path.exists(config_file):
            return cls()
        
        with open(config_file, 'r') as f:
            data = json.load(f)
        
        # Recursively create dataclass instances
        ocr_data = data.get('ocr', {})
        preprocessing_data = data.get('preprocessing', {})
        screen_data = data.get('screen', {})
        layout_data = data.get('layout', {})
        
        return cls(
            ocr=OCRConfig(**ocr_data) if ocr_data else OCRConfig(),
            preprocessing=PreprocessingConfig(**preprocessing_data) if preprocessing_data else PreprocessingConfig(),
            screen=ScreenConfig(**screen_data) if screen_data else ScreenConfig(),
            layout=LayoutConfig(**layout_data) if layout_data else LayoutConfig(),
            debug=data.get('debug', False),
            debug_dir=data.get('debug_dir', 'debug_screenshots'),
            words_file=data.get('words_file', 'data/words.txt'),
            stop_hotkey=data.get('stop_hotkey', 'ctrl+shift+f8'),
            swipe_duration=data.get('swipe_duration', 0.1),
            popup_wait_time=data.get('popup_wait_time', 0.3)
        )
    
    def to_file(self, config_file: str):
        """Save configuration to a JSON file."""
        import dataclasses
        
        def dataclass_to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {
                    field.name: dataclass_to_dict(getattr(obj, field.name))
                    for field in dataclasses.fields(obj)
                }
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        data = dataclass_to_dict(self)
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)