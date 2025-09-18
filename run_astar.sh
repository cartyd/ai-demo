#!/bin/bash

# A* Pathfinding Visualizer Runner
# Usage: ./run_astar.sh

echo "🎯 Starting A* Pathfinding Visualizer..."

# Navigate to project directory
cd "$(dirname "$0")"

# Set Qt environment variables to prevent DPI scaling issues on macOS
export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_SCALE_FACTOR=1
export QT_SCREEN_SCALE_FACTORS=1
export QT_DEVICE_PIXEL_RATIO=1
export QT_LOGGING_RULES='*=false;qt.qpa.backingstore=false;qt.qpa.drawing=false'

# Activate virtual environment and run the application
source astar/.venv/bin/activate && PYTHONPATH=. python -m astar

echo "👋 A* Pathfinding Visualizer closed."
