import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
import logging


class DebugVisualizer:
    """Handles debug visualization and screenshot saving."""
    
    def __init__(self, debug_dir: str):
        self.debug_dir = debug_dir
        self.logger = logging.getLogger(__name__)
        self._setup_debug_dir()
    
    def _setup_debug_dir(self):
        """Create or clean debug directory."""
        if os.path.exists(self.debug_dir):
            # Clean existing files
            for file in os.listdir(self.debug_dir):
                if file.endswith(('.png', '.jpg')):
                    os.remove(os.path.join(self.debug_dir, file))
        else:
            os.makedirs(self.debug_dir)
        
        self.logger.info(f"Debug directory ready: {self.debug_dir}")
    
    def save_screenshot(self, name: str, image: np.ndarray):
        """Save a screenshot with the given name."""
        filename = os.path.join(self.debug_dir, f"{name}.png")
        cv2.imwrite(filename, image)
        self.logger.debug(f"Saved debug image: {filename}")
    
    def visualize_detections(self, 
                           image: np.ndarray,
                           detections: List[Tuple[str, Tuple[int, int]]],
                           layout_info: Optional[Dict] = None):
        """Create visualization of letter detections."""
        # Convert to color if grayscale
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Draw layout info if available
        if layout_info:
            # Draw wheel circle
            if layout_info.get('wheel_center') and layout_info.get('wheel_radius'):
                center = layout_info['wheel_center']
                radius = layout_info['wheel_radius']
                cv2.circle(vis_image, 
                          (int(center[0]), int(center[1])), 
                          int(radius), 
                          (255, 255, 0), 2)  # Yellow circle
            
            # Draw slot positions
            if layout_info.get('slot_positions'):
                for slot_x, slot_y in layout_info['slot_positions']:
                    cv2.circle(vis_image, (slot_x, slot_y), 30, (255, 0, 0), 1)  # Blue slots
        
        # Draw detections
        for letter, (x, y) in detections:
            # Detection circle
            cv2.circle(vis_image, (x, y), 15, (0, 255, 0), 2)  # Green circle
            
            # Letter text
            cv2.putText(vis_image, letter, 
                       (x - 10, y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 0, 255), 2)  # Red text
            
            # Position coordinates
            cv2.putText(vis_image, f"({x},{y})",
                       (x - 30, y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (255, 255, 255), 1)  # White text
        
        # Add detection count
        cv2.putText(vis_image, f"Detections: {len(detections)}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 0), 2)
        
        self.save_screenshot("3_detected_letters", vis_image)
    
    def create_preprocessing_comparison(self, images: List[Tuple[str, np.ndarray]]):
        """Create side-by-side comparison of preprocessing methods."""
        if not images:
            return
        
        # Calculate grid dimensions
        num_images = len(images)
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols
        
        # Get dimensions from first image
        h, w = images[0][1].shape[:2]
        
        # Create combined image
        combined = np.zeros((h * rows, w * cols), dtype=np.uint8)
        
        for i, (name, img) in enumerate(images):
            row = i // cols
            col = i % cols
            
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Place image in grid
            y1, y2 = row * h, (row + 1) * h
            x1, x2 = col * w, (col + 1) * w
            combined[y1:y2, x1:x2] = img
            
            # Add label
            cv2.putText(combined, name,
                       (x1 + 10, y1 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), thickness=1)
        
        self.save_screenshot("preprocessing_comparison", combined)