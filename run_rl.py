#!/usr/bin/env python3
"""
Launch script for RL Pathfinding Visualizer with proper environment setup.
This script sets all necessary environment variables before importing Qt/PySide6.
"""

import os
import sys

# Set all Qt environment variables BEFORE any Qt imports
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
os.environ['QT_SCALE_FACTOR'] = '1'
os.environ['QT_SCREEN_SCALE_FACTORS'] = '1'
os.environ['QT_DEVICE_PIXEL_RATIO'] = '1'
os.environ['QT_LOGGING_RULES'] = '*=false;qt.qpa.backingstore=false;qt.qpa.drawing=false'

# Now import and run the main application
from rl.__main__ import main

if __name__ == "__main__":
    sys.exit(main())