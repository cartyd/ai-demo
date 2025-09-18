# A* Pathfinding Visualizer - Mouse & Cheese

A production-quality, interactive A* pathfinding visualizer that demonstrates the algorithm step-by-step on a grid. The theme features **a mouse (start) finding cheese (target)** with smooth animations and educational controls.

![A* Visualizer Screenshot](docs/screenshot.png)

## Features

### 🎯 Core Visualization
- **Interactive Grid**: Adjustable size (10×60 each dimension)
- **Real-time Algorithm Visualization**: Watch A* explore the grid step-by-step
- **Cost Display**: Shows G-cost, H-cost, and F-cost for each node
- **Visual States**: Distinct colors for open set, closed set, current node, and optimal path
- **SVG Assets**: Vector-based mouse and cheese with smooth rotations

### 🎮 Interactive Controls
- **Edit Modes**: Add/remove walls, set start/target positions
- **Algorithm Control**: Step, Run, Pause, Reset functionality
- **Speed Control**: Adjustable animation speed (50-1000ms)
- **Random Generation**: Create random solvable maps with density control
- **Multiple Heuristics**: Manhattan, Euclidean, Diagonal (Chebyshev), Octile
- **Movement Options**: 4-directional or 8-directional with corner-cutting rules

### 🧠 Algorithm Features
- **Pure A* Implementation**: Framework-agnostic core with proper tie-breaking
- **Binary Heap Priority Queue**: Efficient O(log n) operations
- **Multiple Heuristics**: Choose the best heuristic for your movement rules
- **Weighted Tiles**: Support for terrain with different movement costs
- **Path Reconstruction**: Animate the optimal path with mouse rotation
- **Deterministic Results**: Reproducible pathfinding with seeded random generation

### 🛠 Technical Excellence
- **Clean Architecture**: MVC separation with domain/UI/app layers
- **Type Safety**: Full type hints with mypy checking
- **Comprehensive Tests**: Table-driven unit tests with edge cases
- **Modern Python**: Uses dataclasses, type hints, and modern Python features
- **Qt6 UI**: Professional desktop interface with PySide6

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Quick Start
```bash
# Clone or download the project
cd astar

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m astar
```

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
black --check .
mypy .

# Setup pre-commit hooks
pre-commit install
```

## Usage

### Basic Operations

1. **Grid Setup**
   - Adjust grid size with width/height controls
   - Generate random maps with wall density slider
   - Place start (mouse) and target (cheese) by clicking

2. **Algorithm Execution**
   - **Step**: Execute one algorithm step
   - **Run**: Continuous execution with speed control
   - **Pause**: Pause during execution
   - **Reset**: Clear algorithm state and start over

3. **Visualization Options**
   - Toggle G/H/F cost display
   - Change heuristic function
   - Switch between 4-dir and 8-dir movement
   - Enable/disable corner cutting for diagonal moves

### Algorithm Configuration

#### Heuristics
- **Manhattan**: Best for 4-directional movement
- **Euclidean**: General distance, may be too optimistic
- **Diagonal**: Good for 8-dir when diagonal cost = orthogonal cost  
- **Octile**: Most accurate for 8-dir with √2 diagonal cost (recommended)

#### Movement Rules
- **4-directional**: Only up/down/left/right movement
- **8-directional**: Includes diagonal movement
- **Corner Cutting**: Allow/disallow diagonal moves when corners are blocked

### Keyboard Shortcuts
- **Space**: Step algorithm
- **Enter**: Run/Pause algorithm
- **R**: Reset algorithm
- **Ctrl+N**: New grid
- **Ctrl+R**: Generate random grid

## A* Algorithm Primer

A* (A-star) is a graph traversal and pathfinding algorithm that finds the least-cost path between nodes. It uses a heuristic function to guide its search toward the goal.

### Key Concepts

**F-cost = G-cost + H-cost**
- **G-cost**: Actual cost from start to current node
- **H-cost**: Heuristic estimate from current node to goal
- **F-cost**: Total estimated cost of path through current node

### Algorithm Steps
1. Start with the initial node in the open set
2. Repeat until goal found or open set empty:
   - Select node with lowest F-cost from open set
   - Move it to closed set
   - For each neighbor:
     - Skip if in closed set or blocked
     - If not in open set, add it
     - If already in open set with higher G-cost, update it
3. Reconstruct path from goal to start using parent pointers

### Heuristic Properties
- **Admissible**: Never overestimates the true cost
- **Consistent**: h(n) ≤ h(n') + cost(n, n') for any edge n→n'
- **Optimal**: A* finds optimal path if heuristic is admissible

## Architecture

```
astar/
├── domain/          # Pure algorithm logic
│   ├── types.py     # Data structures and type definitions
│   ├── astar.py     # Core A* implementation
│   ├── heuristics.py # Heuristic functions
│   ├── neighbors.py  # Movement rules and neighbor generation
│   ├── priority_queue.py # Binary heap with tie-breaking
│   └── path.py      # Path reconstruction utilities
├── ui/              # Qt user interface
│   ├── main_window.py # Main application window
│   ├── grid_view.py   # Grid visualization
│   ├── tiles.py       # Individual tile rendering
│   ├── svg_mouse.py   # Rotatable mouse sprite
│   └── svg_cheese.py  # Cheese target sprite
├── app/             # Application controllers
│   ├── controller.py  # Main controller (UI ↔ Domain)
│   └── fsm.py        # Finite state machine
├── utils/           # Utilities
│   ├── grid_factory.py # Grid creation and randomization
│   └── rng.py         # Seeded random number generation
└── tests/           # Test suite
    ├── test_astar.py
    ├── test_heuristics.py
    └── test_neighbors.py
```

## Configuration Examples

### Performance Testing
```python
# Generate large grid for performance testing
grid, start, target = create_preset_grid(60, 60, "dense", seed=42)
config = AlgoConfig(movement="8-dir", heuristic="octile")
result = find_path(start, target, grid, config)
```

### Custom Heuristic
```python
def custom_heuristic(start: Coord, target: Coord) -> float:
    # Your custom heuristic implementation
    dx, dy = abs(start[0] - target[0]), abs(start[1] - target[1])
    return max(dx, dy) + 0.5 * min(dx, dy)

# Register and use custom heuristic
HEURISTICS["custom"] = custom_heuristic
```

## Testing

The project includes comprehensive table-driven tests covering:

- **Algorithm Correctness**: Valid/invalid inputs, edge cases
- **Path Optimality**: Verification of shortest paths
- **Heuristic Admissibility**: All heuristics find valid paths
- **Movement Rules**: 4-dir vs 8-dir, corner cutting
- **Reproducibility**: Deterministic results with seeds
- **Performance**: Large grids, dense obstacles

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=astar --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip performance tests
pytest tests/test_astar.py -v  # Verbose output
```

## Performance

- **Grid Sizes**: Smooth performance up to 60×60 grids
- **Algorithm Speed**: Configurable 50-1000ms per step
- **Memory Usage**: Efficient binary heap and node storage
- **Rendering**: Hardware-accelerated Qt graphics with SVG assets

Tested on modern laptops with no frame drops at default speeds.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes with tests: `pytest tests/`
4. Run quality checks: `ruff check . && black . && mypy .`
5. Commit changes: `git commit -am "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Submit pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Attribution

Mouse & Cheese A* Visualizer inspired by classic pathfinding demos and educational visualizations.

## Troubleshooting

### Common Issues

**"Module 'PySide6' not found"**
```bash
pip install PySide6>=6.5.0
```

**"SVG assets not loading"**
- Ensure `assets/mouse.svg` and `assets/cheese.svg` exist
- Fallback renderings will be used if SVG files are missing

**"Tests failing on import"**
```bash
# Run from project root directory
cd astar
python -m pytest
```

**"Algorithm not finding path"**
- Check that start and target are not on walls
- Ensure there's actually a valid path (try empty grid first)
- Verify wall placement isn't blocking all routes

### Performance Tips

- Use smaller grids (25×25) for smooth stepping
- Reduce animation speed for better visualization
- Use "octile" heuristic for best 8-directional performance
- Generate solvable grids to avoid "no path" scenarios