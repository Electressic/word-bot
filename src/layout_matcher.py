import json
import os
import math
from typing import List, Tuple

class LayoutMatcher:
    """Maps raw OCR detections onto fixed wheel slots to remove duplicates / noise."""

    def __init__(self, layout_file: str = None):
        if layout_file is None:
            layout_file = os.path.join(os.path.dirname(__file__), "layouts.json")
        with open(layout_file, "r", encoding="utf-8") as f:
            self.layouts = json.load(f)

    def match(self,
              detections: List[Tuple[str, Tuple[int, int]]],
              wheel_size: int,
              tolerance_px: int = 60) -> List[Tuple[str, Tuple[int, int]]]:
        """Return one letter per template slot if a detection falls within tolerance.

        Args:
            detections: list of (letter, (x, y)) tuples in absolute image coords.
            wheel_size: number of letters in the wheel (e.g. 6).
            tolerance_px: max pixel distance between a template slot and a detection.
        """
        if str(wheel_size) not in self.layouts:
            # No template – fall back to raw detections
            return detections

        if not detections:
            return []

        # Estimate centre and radius of the current wheel from detections
        xs = [x for _, (x, y) in detections]
        ys = [y for _, (x, y) in detections]
        centre_x = sum(xs) / len(xs)
        centre_y = sum(ys) / len(ys)
        # mean distance to centre = radius estimate
        radius = sum(math.hypot(x - centre_x, y - centre_y) for x, y in zip(xs, ys)) / len(xs)
        if radius == 0:
            return detections

        template = self.layouts[str(wheel_size)]
        self.slots_abs = [
            (centre_x + dx * radius, centre_y + dy * radius)
            for dx, dy in template
        ]

        slots_abs = self.slots_abs
        chosen: List[Tuple[str, Tuple[int, int]]] = []
        remaining = detections.copy()
        for slot_x, slot_y in slots_abs:
            best_idx = None
            best_dist = tolerance_px + 1
            for idx, (letter, (x, y)) in enumerate(remaining):
                d = math.hypot(x - slot_x, y - slot_y)
                if d < best_dist and d <= tolerance_px:
                    best_dist = d
                    best_idx = idx
            if best_idx is not None:
                letter, (x, y) = remaining.pop(best_idx)
                chosen.append((letter, (int(x), int(y))))
        return chosen

    # ------------------------------------------------------------------
    def get_last_slots(self) -> List[Tuple[int, int]]:
        """Return absolute coordinates of the most recently computed slots."""
        return getattr(self, "slots_abs", []) 