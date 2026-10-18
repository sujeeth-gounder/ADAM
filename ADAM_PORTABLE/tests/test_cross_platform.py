"""Unit and Integration Tests for Cross-Platform Compatibility Layer."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.platform_utils import (
    get_os_type,
    is_windows,
    is_linux,
    is_macos,
    get_user_data_dir,
    find_executable_cross_platform,
    find_brave_browser,
    find_blender_binary,
    open_url_in_browser,
)


class TestCrossPlatformUtils(unittest.TestCase):
    def test_os_detection(self):
        os_name = get_os_type()
        self.assertIn(os_name, ["linux", "windows", "darwin"])
        if sys.platform.startswith("linux"):
            self.assertTrue(is_linux())
            self.assertFalse(is_windows())
        elif sys.platform == "win32":
            self.assertTrue(is_windows())
            self.assertFalse(is_linux())

    def test_user_data_dir_dynamic(self):
        data_dir = get_user_data_dir("ADAM")
        self.assertTrue(isinstance(data_dir, Path))
        # Must not contain hardcoded /home/sujeethkani if running under another user
        self.assertTrue(str(Path.home()) in str(data_dir))

    def test_find_brave_browser(self):
        brave = find_brave_browser()
        # On this Linux system, Brave is installed; verify if found it returns a valid string
        if brave:
            self.assertTrue(isinstance(brave, str))
            self.assertTrue("brave" in brave.lower())

    def test_find_blender_binary(self):
        blender = find_blender_binary()
        if blender:
            self.assertTrue(isinstance(blender, str))
            self.assertTrue("blender" in blender.lower())

    def test_executable_cross_platform_windows_simulation(self):
        with patch("sys.platform", "win32"):
            with patch("os.environ", {"ProgramFiles": "/simulated/ProgramFiles", "LOCALAPPDATA": "/simulated/AppData"}):
                # When files do not exist, returns None gracefully
                res = find_executable_cross_platform("non_existent_tool_12345")
                self.assertIsNone(res)

    def test_open_url_in_browser_safe_format(self):
        # Test malformed / non-http URLs are rejected
        ok, msg = open_url_in_browser("javascript:alert(1)")
        self.assertFalse(ok)
        ok2, msg2 = open_url_in_browser("file:///etc/passwd")
        self.assertFalse(ok2)


if __name__ == "__main__":
    unittest.main()
