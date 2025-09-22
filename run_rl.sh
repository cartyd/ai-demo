#!/bin/bash

# RL Pathfinding Visualizer Launch Script

echo "🧠 Starting RL Pathfinding Visualizer..."

# Check if virtual environment exists, if not create it
if [ ! -d "rl/.venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv rl/.venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source rl/.venv/bin/activate

# Install/update dependencies
echo "📋 Installing dependencies..."
pip install -q -r rl/requirements.txt

# Launch the application
echo "🚀 Launching RL Pathfinding Visualizer..."
python run_rl.py

echo "👋 RL Pathfinding Visualizer closed."