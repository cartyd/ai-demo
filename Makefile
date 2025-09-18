# A* Pathfinding Visualizer Makefile

.PHONY: run clean test help

# Default target
run:
	@echo "🎯 Launching A* Pathfinding Visualizer with DPI fixes..."
	@./run_astar.sh

# Alternative run target using Python launcher
run-py:
	@echo "🎯 Launching A* Pathfinding Visualizer (Python launcher)..."
	@python run_astar.py

# Run original version (may have DPI issues)
run-original:
	@echo "⚠️  Launching original version (may have screen blinking)..."
	@source astar/.venv/bin/activate && python -m astar

# Clean Python cache files
clean:
	@echo "🧹 Cleaning up Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Run tests
test:
	@echo "🧪 Running tests..."
	@source astar/.venv/bin/activate && python -m pytest astar/tests/ -v

# Show help
help:
	@echo "A* Pathfinding Visualizer - Available Commands:"
	@echo ""
	@echo "  make run        - Launch the visualizer (recommended, with DPI fixes)"
	@echo "  make run-py     - Launch using Python launcher"
	@echo "  make run-original - Launch original version (may have issues)"
	@echo "  make test       - Run the test suite"
	@echo "  make clean      - Clean Python cache files"
	@echo "  make help       - Show this help message"