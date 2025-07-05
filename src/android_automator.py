import subprocess
import time
from typing import List, Tuple

class AndroidAutomator:
    """
    Automates actions on an Android device using ADB.
    """

    def __init__(self, adb_path: str = 'adb'):
        """
        Initializes the AndroidAutomator.

        Args:
            adb_path: The path to the ADB executable.
        """
        self.adb_path = adb_path

    def _run_command(self, command: List[str]):
        """
        Runs an ADB command.

        Args:
            command: The command to run.
        """
        try:
            subprocess.run([self.adb_path] + command, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(
                f"ADB not found at '{self.adb_path}'. "
                f"Please ensure ADB is installed and in your system's PATH, "
                f"or specify the path to the ADB executable."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ADB command failed: {e.stderr}")

    def take_screenshot(self, output_path: str):
        """
        Takes a screenshot of the device.

        Args:
            output_path: The path to save the screenshot to.
        """
        self._run_command(["shell", "screencap", "-p", "/sdcard/screen.png"])
        self._run_command(["pull", "/sdcard/screen.png", output_path])
        self._run_command(["shell", "rm", "/sdcard/screen.png"])

    def swipe_word(self, coordinates: List[Tuple[int, int]], duration_per_segment_ms: int = 50):
        """
        Swipes a word on the screen by chaining multiple swipe commands.

        Args:
            coordinates: A list of (x, y) coordinates to swipe through.
            duration_per_segment_ms: The duration for each segment of the swipe.
        """
        if len(coordinates) < 2:
            return

        commands = []
        for i in range(len(coordinates) - 1):
            start_x, start_y = coordinates[i]
            end_x, end_y = coordinates[i+1]
            commands.append(
                f"input swipe {start_x} {start_y} {end_x} {end_y} {duration_per_segment_ms}"
            )
        
        full_command = "; ".join(commands)
        self._run_command(["shell", full_command])
