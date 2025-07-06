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

    def swipe_word(self, coordinates: List[Tuple[int, int]], duration_ms: int = 300):
        """
        Swipes a word on the screen.

        Args:
            coordinates: A list of (x, y) coordinates to swipe through.
            duration_ms: The duration of the swipe in milliseconds.
        """
        if not coordinates:
            return

        start_x, start_y = coordinates[0]
        command = ["shell", "input", "swipe", str(start_x), str(start_y), str(start_x), str(start_y), str(duration_ms)]

        for i in range(1, len(coordinates)):
            end_x, end_y = coordinates[i]
            command[4] = str(end_x)
            command[5] = str(end_y)
            self._run_command(command)
            start_x, start_y = end_x, end_y
            time.sleep(duration_ms / 1000)
