import subprocess
import time
import os
import re
import numpy as np
from typing import List, Tuple, Optional

class AndroidAutomator:
    """
    Automates actions on an Android device using ADB.
    """

    def __init__(self, adb_path: str = 'adb', debug: bool = False):
        """
        Initializes the AndroidAutomator.

        Args:
            adb_path: The path to the ADB executable.
            debug: Whether to run in debug mode.
        """
        self.adb_path = adb_path
        self.debug = debug
        self.touch_device = None
        self._find_touch_device()

    def _run_command(self, command: List[str], capture_output=True, text=True) -> subprocess.CompletedProcess:
        """
        Runs an ADB command.

        Args:
            command: The command to run.
        """
        try:
            return subprocess.run(
                [self.adb_path] + command,
                check=True,
                capture_output=capture_output,
                text=text
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"ADB not found at '{self.adb_path}'. "
                f"Please ensure ADB is installed and in your system's PATH, "
                f"or specify the path to the ADB executable."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ADB command failed: {e.stderr}")

    def _find_touch_device(self):
        """
        Finds the input device path for the touchscreen.
        """
        if self.touch_device:
            return
        
        result = self._run_command(["shell", "getevent", "-lp"])
        devices = result.stdout.split('/dev/input/')
        for device in devices:
            if 'ABS_MT_POSITION_X' in device and 'ABS_MT_POSITION_Y' in device:
                match = re.search(r'event\d+', device)
                if match:
                    self.touch_device = f"/dev/input/{match.group(0)}"
                    print(f"Found touchscreen device: {self.touch_device}")
                    return
        raise RuntimeError("Could not find a touchscreen device. Please ensure your device is connected and ADB is authorized.")


    def take_screenshot(self, output_path: str):
        """
        Takes a screenshot of the device.

        Args:
            output_path: The path to save the screenshot to.
        """
        self._run_command(["shell", "screencap", "-p", "/sdcard/screen.png"])
        self._run_command(["pull", "/sdcard/screen.png", output_path])
        self._run_command(["shell", "rm", "/sdcard/screen.png"])

    def swipe_word(self, coordinates: List[Tuple[int, int]], word: str, steps: int = 20):
        """
        Swipes a word on the screen using low-level sendevent commands
        to simulate a continuous touch gesture.

        Args:
            coordinates: A list of (x, y) coordinates to swipe through.
            word: The word being swiped (for debug naming).
            steps: The number of interpolated steps between each letter.
        """
        if not self.touch_device:
            raise RuntimeError("Touchscreen device not found.")
        if len(coordinates) < 2:
            return

        # Start building the command sequence
        commands = []
        
        # 1. Finger Down
        start_x, start_y = coordinates[0]
        commands.extend([
            f"sendevent {self.touch_device} 1 330 1",  # BTN_TOUCH DOWN
            f"sendevent {self.touch_device} 3 53 {start_x}",  # ABS_MT_POSITION_X
            f"sendevent {self.touch_device} 3 54 {start_y}",  # ABS_MT_POSITION_Y
            f"sendevent {self.touch_device} 0 0 0",  # SYN_REPORT
            "sleep 0.05"
        ])

        # 2. Finger Move (interpolate between points)
        for i in range(len(coordinates) - 1):
            p1 = np.array(coordinates[i])
            p2 = np.array(coordinates[i+1])
            for j in range(1, steps + 1):
                alpha = j / steps
                point = p1 * (1 - alpha) + p2 * alpha
                ix, iy = int(point[0]), int(point[1])
                commands.extend([
                    f"sendevent {self.touch_device} 3 53 {ix}",
                    f"sendevent {self.touch_device} 3 54 {iy}",
                    f"sendevent {self.touch_device} 0 0 0",
                    "sleep 0.005" # Small delay for smoother movement
                ])

        # 3. Finger Up
        commands.extend([
            "sleep 0.05",
            f"sendevent {self.touch_device} 1 330 0",  # BTN_TOUCH UP
            f"sendevent {self.touch_device} 0 0 0"
        ])

        # Execute the entire swipe as a single command
        full_command = " && ".join(commands)
        self._run_command(["shell", full_command])

        # Take a screenshot after the swipe in debug mode
        if self.debug:
            time.sleep(0.5) # Wait for the swipe to register visually
            debug_screenshot_path = os.path.join("debug_screenshots", f"{word}_swiped.png")
            self.take_screenshot(debug_screenshot_path)
