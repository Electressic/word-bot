"""Improved layout matching with better assignment algorithm."""

import json
import os
import math
from typing import List, Tuple, Dict, Optional
import numpy as np
import logging

from src.core.config import LayoutConfig


class LayoutMatcher:
    """Maps OCR detections to fixed wheel positions."""
    
    def __init__(self, config: LayoutConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.layouts = self._load_layouts()
        self.last_matched_slots = []
        self.last_wheel_center = None
        self.last_wheel_radius = None
    
    def _load_layouts(self) -> Dict[int, List[Tuple[float, float]]]:
        """Load layout templates from file."""
        if not os.path.exists(self.config.layouts_file):
            self.logger.warning(f"Layouts file not found: {self.config.layouts_file}")
            return self._generate_default_layouts()
        
        try:
            with open(self.config.layouts_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Handle different possible JSON structures
            layouts = {}
            
            # If the JSON has a nested structure with layout type
            if isinstance(raw_data, dict) and 'circular_layout' in raw_data:
                # Extract the circular layout data
                circular_data = raw_data.get('circular_layout', {})
                for size_str, positions in circular_data.items():
                    try:
                        size = int(size_str)
                        layouts[size] = [tuple(pos) for pos in positions]
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Skipping invalid layout entry {size_str}: {e}")
            
            # If the JSON is directly the layout data
            elif isinstance(raw_data, dict):
                for size_str, positions in raw_data.items():
                    try:
                        size = int(size_str)
                        layouts[size] = [tuple(pos) for pos in positions]
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Skipping invalid layout entry {size_str}: {e}")
            
            if not layouts:
                self.logger.warning("No valid layouts found in file, using defaults")
                return self._generate_default_layouts()
            
            self.logger.info(f"Loaded layouts for wheel sizes: {list(layouts.keys())}")
            return layouts
            
        except Exception as e:
            self.logger.error(f"Error loading layouts: {e}")
            return self._generate_default_layouts()
    
    def _generate_default_layouts(self) -> Dict[int, List[Tuple[float, float]]]:
        """Generate default circular layouts for common wheel sizes."""
        layouts = {}
        
        # Common wheel sizes
        for size in [5, 6, 7, 8]:
            positions = []
            for i in range(size):
                angle = 2 * math.pi * i / size - math.pi / 2  # Start from top
                x = math.cos(angle)
                y = math.sin(angle)
                positions.append((x, y))
            layouts[size] = positions
        
        self.logger.info(f"Generated default layouts for sizes: {list(layouts.keys())}")
        return layouts
    
    def match(self, detections: List[Tuple[str, Tuple[int, int]]], 
              wheel_size: Optional[int] = None) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Match detections to template slots.
        
        Args:
            detections: List of (letter, (x, y)) tuples
            wheel_size: Force a specific wheel size, or None for auto-detection
            
        Returns:
            List of matched (letter, (x, y)) tuples
        """
        if not detections:
            return []
        
        # Auto-detect wheel size if not specified
        if wheel_size is None:
            wheel_size = self._estimate_wheel_size(len(detections))
        
        # Special handling for forced wheel sizes with many detections
        if wheel_size and len(detections) > wheel_size * 2:
            self.logger.warning(f"Many detections ({len(detections)}) for wheel size {wheel_size}. "
                               "Trying progressive tolerance matching.")
            
            # Try increasing tolerances until we get a good match
            for tolerance_multiplier in [1.0, 1.5, 2.0, 3.0]:
                temp_tolerance = int(self.config.tolerance_px * tolerance_multiplier)
                matches = self._match_with_tolerance(detections, wheel_size, temp_tolerance)
                
                if len(matches) >= wheel_size:
                    self.logger.info(f"Found good match with {tolerance_multiplier}x tolerance "
                                   f"({temp_tolerance}px): {len(matches)} letters")
                    return matches[:wheel_size]  # Take first N matches
        
        # If we don't have a layout for this size, generate one
        if wheel_size not in self.layouts:
            self.logger.warning(f"No layout for wheel size {wheel_size}, generating one")
            self._add_generated_layout(wheel_size)
        
        # Estimate wheel geometry
        center, radius = self._estimate_wheel_geometry(detections)
        if radius == 0:
            self.logger.warning("Could not estimate wheel geometry")
            return detections
        
        self.last_wheel_center = center
        self.last_wheel_radius = radius
        
        # Get absolute slot positions
        template = self.layouts[wheel_size]
        slots = self._calculate_slot_positions(template, center, radius)
        self.last_matched_slots = slots
        
        # Find optimal assignment
        matches = self._optimal_assignment(detections, slots)
        
        self.logger.info(f"Matched {len(matches)}/{len(detections)} detections to {wheel_size}-letter wheel")
        
        return matches
    
    def _add_generated_layout(self, size: int):
        """Generate and add a layout for a specific size."""
        positions = []
        for i in range(size):
            angle = 2 * math.pi * i / size - math.pi / 2  # Start from top
            x = math.cos(angle)
            y = math.sin(angle)
            positions.append((x, y))
        self.layouts[size] = positions
    
    def _estimate_wheel_size(self, num_detections: int) -> int:
        """Estimate wheel size based on number of detections."""
        # Find the closest available wheel size
        available_sizes = list(self.layouts.keys())
        if not available_sizes:
            return num_detections
        
        # Find closest size
        closest_size = min(available_sizes, key=lambda x: abs(x - num_detections))
        
        # If the difference is too large, prefer the actual detection count
        if abs(closest_size - num_detections) > 2:
            self.logger.warning(f"No good wheel size match for {num_detections} detections")
            return num_detections
        
        return closest_size
    
    def _estimate_wheel_geometry(self, detections: List[Tuple[str, Tuple[int, int]]]) -> Tuple[Tuple[float, float], float]:
        """
        Estimate the center and radius of the wheel from detections.
        
        Returns:
            (center_x, center_y), radius
        """
        points = np.array([pos for _, pos in detections], dtype=np.float32)
        
        # Method 1: Simple centroid and average distance
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        
        # Method 2: Use RANSAC-like approach to filter outliers
        median_dist = np.median(distances)
        inlier_mask = np.abs(distances - median_dist) < median_dist * 0.3
        
        if np.sum(inlier_mask) >= 3:
            # Recalculate with inliers only
            inlier_points = points[inlier_mask]
            center = np.mean(inlier_points, axis=0)
            distances = np.linalg.norm(inlier_points - center, axis=1)
            radius = np.mean(distances)
        else:
            # Fall back to median
            radius = median_dist
        
        return tuple(center), float(radius)
    
    def _calculate_slot_positions(self, template: List[Tuple[float, float]], 
                                 center: Tuple[float, float], 
                                 radius: float) -> List[Tuple[int, int]]:
        """Calculate absolute positions for template slots."""
        cx, cy = center
        
        slots = []
        for dx, dy in template:
            x = cx + dx * radius
            y = cy + dy * radius
            slots.append((int(x), int(y)))
        
        return slots
    
    def _optimal_assignment(self, detections: List[Tuple[str, Tuple[int, int]]], 
                           slots: List[Tuple[int, int]]) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Find optimal assignment using Hungarian algorithm approach.
        """
        if not detections or not slots:
            return []
        
        # Build cost matrix (distances)
        num_detections = len(detections)
        num_slots = len(slots)
        
        # Create distance matrix
        distances = np.full((num_detections, num_slots), np.inf)
        
        # Log all detections and their distances to slots
        self.logger.info(f"Analyzing {num_detections} detections for {num_slots} slots:")
        
        for i, (letter, det_pos) in enumerate(detections):
            min_distance = float('inf')
            closest_slot = -1
            
            for j, slot_pos in enumerate(slots):
                dist = math.hypot(det_pos[0] - slot_pos[0], 
                                det_pos[1] - slot_pos[1])
                if dist < min_distance:
                    min_distance = dist
                    closest_slot = j
                    
                if dist <= self.config.tolerance_px:
                    distances[i, j] = dist
            
            # Log each detection with its closest slot distance
            status = "ACCEPTED" if min_distance <= self.config.tolerance_px else "REJECTED"
            self.logger.info(f"  Detection {i}: '{letter}' at {det_pos} -> closest slot {closest_slot} "
                            f"(dist: {min_distance:.1f}px) [{status}]")

        # Simple greedy assignment (can be replaced with scipy.optimize.linear_sum_assignment)
        matches = []
        used_detections = set()
        used_slots = set()
        
        # Create list of all valid assignments sorted by distance
        assignments = []
        for i in range(num_detections):
            for j in range(num_slots):
                if distances[i, j] < np.inf:
                    assignments.append((distances[i, j], i, j))
        
        # Sort by distance
        assignments.sort(key=lambda x: x[0])
        
        # Assign greedily
        for dist, det_idx, slot_idx in assignments:
            if det_idx not in used_detections and slot_idx not in used_slots:
                letter, det_pos = detections[det_idx]
                matches.append((letter, det_pos))
                used_detections.add(det_idx)
                used_slots.add(slot_idx)
                
                self.logger.info(f"MATCHED '{letter}' to slot {slot_idx} (distance: {dist:.1f})")

        # Log which detections were not matched
        for i, (letter, pos) in enumerate(detections):
            if i not in used_detections:
                self.logger.warning(f"UNMATCHED detection: '{letter}' at {pos}")
        
        # Sort matches by slot order to maintain consistency
        if matches:
            # Create mapping from detection to slot
            match_to_slot = {}
            for dist, det_idx, slot_idx in assignments:
                if det_idx in used_detections:
                    letter, pos = detections[det_idx]
                    for match in matches:
                        if match[0] == letter and match[1] == pos:
                            match_to_slot[match] = slot_idx
                            break
            
            # Sort by slot index
            matches.sort(key=lambda m: match_to_slot.get(m, float('inf')))
        
        return matches
    
    def _match_with_tolerance(self, detections: List[Tuple[str, Tuple[int, int]]], 
                              wheel_size: int, tolerance_px: int) -> List[Tuple[str, Tuple[int, int]]]:
        """Helper method to match with specific tolerance."""
        original_tolerance = self.config.tolerance_px
        self.config.tolerance_px = tolerance_px
        
        try:
            # Auto-detect wheel size if not specified
            if wheel_size is None:
                wheel_size = self._estimate_wheel_size(len(detections))
            
            # If we don't have a layout for this size, generate one
            if wheel_size not in self.layouts:
                self.logger.warning(f"No layout for wheel size {wheel_size}, generating one")
                self._add_generated_layout(wheel_size)
            
            # Estimate wheel geometry
            center, radius = self._estimate_wheel_geometry(detections)
            if radius == 0:
                return detections
            
            self.last_wheel_center = center
            self.last_wheel_radius = radius
            
            # Get absolute slot positions
            template = self.layouts[wheel_size]
            slots = self._calculate_slot_positions(template, center, radius)
            self.last_matched_slots = slots
            
            # Find optimal assignment
            matches = self._optimal_assignment(detections, slots)
            
            self.logger.info(f"Matched {len(matches)}/{len(detections)} detections to {wheel_size}-letter wheel "
                            f"with {tolerance_px}px tolerance")
            
            return matches
        finally:
            self.config.tolerance_px = original_tolerance
    
    def get_debug_info(self) -> Dict:
        """Get debug information about the last match."""
        return {
            'wheel_center': self.last_wheel_center,
            'wheel_radius': self.last_wheel_radius,
            'slot_positions': self.last_matched_slots
        }