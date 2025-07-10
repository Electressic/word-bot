"""Debug utilities for the word game bot."""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import math


class DebugVisualizer:
    """Enhanced debug visualization with positioning analysis."""
    
    def __init__(self, debug_dir: str = "debug_screenshots"):
        self.debug_dir = debug_dir
        self.logger = logging.getLogger(__name__)
        self._ensure_debug_dir()
    
    def _ensure_debug_dir(self):
        """Ensure debug directory exists."""
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
            self.logger.info(f"Debug directory ready: {self.debug_dir}")
    
    def save_detections_overlay(self, image: np.ndarray, detections: List[Tuple[str, Tuple[int, int]]], 
                               filename: str = "final_detections_overlay.png"):
        """
        Save image with detection overlays and positioning analysis.
        
        Args:
            image: Base image
            detections: List of (letter, (x, y)) tuples
            filename: Output filename
        """
        # Create a copy for drawing
        overlay = image.copy()
        
        # Convert to BGR for color drawing
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        elif len(overlay.shape) == 3 and overlay.shape[2] == 1:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        
        # Colors for visualization
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 255, 128), # Light green
            (255, 128, 128), # Light blue
        ]
        
        # Analyze clustering
        cluster_analysis = self._analyze_clustering(detections)
        
        # Draw detections
        for i, (letter, (x, y)) in enumerate(detections):
            color = colors[i % len(colors)]
            
            # Draw circle at detection point
            cv2.circle(overlay, (x, y), 20, color, 2)
            
            # Draw letter label with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(letter, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(overlay, 
                         (x - text_width//2 - 5, y - text_height - 10),
                         (x + text_width//2 + 5, y + baseline - 5),
                         (255, 255, 255), -1)
            
            # Draw letter
            cv2.putText(overlay, letter, (x - text_width//2, y - 5), 
                       font, font_scale, (0, 0, 0), thickness)
            
            # Draw index number
            cv2.putText(overlay, str(i), (x + 25, y + 5), 
                       font, 0.5, color, 1)
        
        # Draw clustering lines for letters that are too close
        for (letter1, pos1), (letter2, pos2), distance in cluster_analysis['close_pairs']:
            if distance < 50:  # Draw line for very close letters
                cv2.line(overlay, pos1, pos2, (0, 0, 255), 2)  # Red line
                # Add distance label
                mid_x = (pos1[0] + pos2[0]) // 2
                mid_y = (pos1[1] + pos2[1]) // 2
                cv2.putText(overlay, f"{distance:.0f}px", (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Add statistics text
        stats_text = [
            f"Detections: {len(detections)}",
            f"Close pairs (<50px): {len(cluster_analysis['close_pairs'])}",
            f"Min distance: {cluster_analysis['min_distance']:.1f}px",
            f"Avg distance: {cluster_analysis['avg_distance']:.1f}px"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(overlay, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Save the overlay
        output_path = os.path.join(self.debug_dir, filename)
        cv2.imwrite(output_path, overlay)
        self.logger.info(f"Saved detection overlay: {output_path}")
        
        return cluster_analysis
    
    def _analyze_clustering(self, detections: List[Tuple[str, Tuple[int, int]]]) -> Dict:
        """Analyze letter clustering and positioning issues."""
        if len(detections) < 2:
            return {
                'close_pairs': [],
                'min_distance': float('inf'),
                'avg_distance': 0,
                'clustering_score': 0
            }
        
        close_pairs = []
        distances = []
        
        for i, (letter1, pos1) in enumerate(detections):
            for j, (letter2, pos2) in enumerate(detections[i+1:], i+1):
                distance = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
                distances.append(distance)
                
                if distance < 60:  # Consider pairs within 60px as potentially problematic
                    close_pairs.append(((letter1, pos1), (letter2, pos2), distance))
        
        return {
            'close_pairs': close_pairs,
            'min_distance': min(distances) if distances else float('inf'),
            'avg_distance': sum(distances) / len(distances) if distances else 0,
            'clustering_score': len(close_pairs) / len(detections) if detections else 0
        }
    
    def save_step_image(self, image: np.ndarray, step_name: str, step_number: int = None):
        """Save an intermediate processing step image."""
        if step_number is not None:
            filename = f"{step_number}_{step_name}.png"
        else:
            filename = f"{step_name}.png"
        
        output_path = os.path.join(self.debug_dir, filename)
        cv2.imwrite(output_path, image)
        self.logger.debug(f"Saved debug image: {output_path}")
    
    def save_screenshot(self, name: str, image: np.ndarray):
        """Save a screenshot with the given name (backward compatibility)."""
        filename = os.path.join(self.debug_dir, f"{name}.png")
        cv2.imwrite(filename, image)
        self.logger.debug(f"Saved debug image: {filename}")
    
    def visualize_detections(self, 
                       image: np.ndarray,
                       detections: List[Tuple[str, Tuple[int, int]]],
                       layout_info: Optional[Dict] = None,
                       base_bw_image: Optional[np.ndarray] = None,
                       wheel_info: Optional[Dict] = None):
        """Visualize detections and save the final overlay."""
        # Use black-and-white base if provided (Otsu result)
        if base_bw_image is not None:
            # Convert B&W to color for overlay
            vis_image = cv2.cvtColor(base_bw_image, cv2.COLOR_GRAY2BGR)
        else:
            # Fallback to original method
            if len(image.shape) == 2:
                vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                vis_image = image.copy()
        
        # Draw detected wheel circle if available
        if wheel_info:
            center = wheel_info.get('center')
            radius = wheel_info.get('radius')
            confidence = wheel_info.get('confidence', 0)
            
            if center and radius:
                # Draw wheel circle in cyan
                cv2.circle(vis_image, center, radius, (255, 255, 0), 2)  # Cyan circle
                
                # Draw center point
                cv2.circle(vis_image, center, 5, (255, 255, 0), -1)  # Filled cyan center
                
                # Draw inner boundary (min distance for letters)
                inner_radius = int(radius * 0.3)
                cv2.circle(vis_image, center, inner_radius, (128, 128, 0), 1)  # Darker cyan
                
                # Draw outer boundary (max distance for letters)
                outer_radius = int(radius * 1.2)
                cv2.circle(vis_image, center, outer_radius, (128, 128, 0), 1)  # Darker cyan
                
                # Add wheel info text
                wheel_text = f"Wheel: ({center[0]},{center[1]}) r={radius} conf={confidence:.2f}"
                cv2.putText(vis_image, wheel_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(vis_image, wheel_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw layout info if available
        if layout_info:
            # Draw wheel circle in yellow
            if layout_info.get('wheel_center') and layout_info.get('wheel_radius'):
                center = layout_info['wheel_center']
                radius = layout_info['wheel_radius']
                cv2.circle(vis_image, 
                          (int(center[0]), int(center[1])), 
                          int(radius), 
                          (0, 255, 255), 2)  # Yellow circle
            
            # Draw slot positions in blue
            if layout_info.get('slot_positions'):
                for slot_x, slot_y in layout_info['slot_positions']:
                    cv2.circle(vis_image, (int(slot_x), int(slot_y)), 8, (255, 0, 0), 2)  # Blue circles
        
        # Colors for different letters
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 255, 128), # Light green
            (255, 128, 128), # Light blue
        ]
        
        # Draw detections with better visualization
        for i, (letter, (x, y)) in enumerate(detections):
            color = colors[i % len(colors)]
            
            # Detection circle in bright color
            cv2.circle(vis_image, (x, y), 20, color, 3)
            
            # Letter label with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(letter, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(vis_image, 
                         (x - text_width//2 - 5, y - text_height - 10),
                         (x + text_width//2 + 5, y + baseline - 5),
                         (255, 255, 255), -1)
            
            # Draw letter in black
            cv2.putText(vis_image, letter, (x - text_width//2, y - 5), 
                       font, font_scale, (0, 0, 0), thickness)
            
            # Add position coordinates as small text
            coord_text = f"({x},{y})"
            cv2.putText(vis_image, coord_text, (x - 30, y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add detection count and info
        info_text = f"Detections: {len(detections)}"
        cv2.putText(vis_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Save the final overlay
        output_path = os.path.join(self.debug_dir, "9_final_detections_overlay.png")
        cv2.imwrite(output_path, vis_image)
        self.logger.info(f"Saved final detection overlay: {output_path}")
        
        # Also call the enhanced overlay method for detailed analysis
        self.save_detections_overlay(vis_image, detections, "9_enhanced_analysis.png")
    
    def log_positioning_stats(self, detections: List[Tuple[str, Tuple[int, int]]]):
        """Log detailed positioning statistics."""
        if len(detections) < 2:
            return
        
        self.logger.info("=== Letter Positioning Analysis ===")
        
        # Calculate all pairwise distances
        for i, (letter1, pos1) in enumerate(detections):
            for j, (letter2, pos2) in enumerate(detections[i+1:], i+1):
                distance = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
                if distance < 50:
                    self.logger.warning(f"  '{letter1}' and '{letter2}' are very close: {distance:.1f}px")
                else:
                    self.logger.debug(f"  '{letter1}' to '{letter2}': {distance:.1f}px")
        
        # Find the most problematic pairs
        close_pairs = []
        for i, (letter1, pos1) in enumerate(detections):
            for j, (letter2, pos2) in enumerate(detections[i+1:], i+1):
                distance = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
                close_pairs.append((distance, letter1, letter2))
        
        close_pairs.sort()
        if close_pairs:
            self.logger.info(f"Closest pair: '{close_pairs[0][1]}' - '{close_pairs[0][2]}' ({close_pairs[0][0]:.1f}px)")
            
        self.logger.info("=== End Positioning Analysis ===")


# For backward compatibility
def save_debug_image(image: np.ndarray, filename: str, debug_dir: str = "debug_screenshots"):
    """Save debug image (legacy function)."""
    visualizer = DebugVisualizer(debug_dir)
    visualizer.save_step_image(image, filename.replace('.png', ''))


def ensure_debug_dir(debug_dir: str = "debug_screenshots"):
    """Ensure debug directory exists (legacy function)."""
    visualizer = DebugVisualizer(debug_dir)
    return debug_dir