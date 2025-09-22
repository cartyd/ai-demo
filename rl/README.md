# RL Pathfinding Visualizer

A reinforcement learning approach to pathfinding that learns optimal paths through Q-Learning, providing visual comparison to traditional algorithms like A*.

## Overview

This project implements a Q-Learning agent that learns to navigate grid environments. Unlike traditional pathfinding algorithms that have perfect knowledge, the RL agent learns through trial and error, discovering optimal paths through exploration and exploitation.

## Features

- **Q-Learning Algorithm**: Tabular Q-Learning implementation for pathfinding
- **Interactive Training**: Watch the agent learn in real-time
- **Visual Q-Values**: See Q-values displayed on grid tiles
- **Multiple Grid Types**: Empty grids, mazes, and random obstacles
- **Training Statistics**: Track episodes, success rate, and convergence
- **Policy Testing**: Test the learned policy step-by-step
- **Grid Editing**: Create custom environments

## Key Components

### Domain Layer
- `types.py`: Core types and data structures for RL
- `qlearning.py`: Q-Learning algorithm and environment implementation

### Application Layer  
- `controller.py`: Main controller connecting UI and RL logic
- `fsm.py`: State machine for training/testing phases

### UI Layer
- `main_window.py`: Main application window
- `grid_view.py`: Grid visualization with RL-specific features
- `tiles.py`: Individual grid tiles with Q-value display

### Utilities
- `grid_factory.py`: Grid generation and manipulation
- `rng.py`: Random number generation for reproducibility

## Usage

### Running the Application

```bash
# Using the shell script (recommended)
./run_rl.sh

# Or directly with Python
python run_rl.py
```

### Training Process

1. **Create/Generate Grid**: Start with empty grid or generate maze
2. **Place Start/Target**: Set mouse (start) and cheese (target) positions  
3. **Configure Parameters**: Adjust learning rate, epsilon, rewards
4. **Train Agent**: Click "Train" to start Q-Learning episodes
5. **Monitor Progress**: Watch success rate and epsilon decay
6. **Test Policy**: Click "Test" to see learned behavior

### Visualization Features

- **Q-Value Display**: Shows learned values for each action
- **Training Progress**: Real-time episode statistics
- **Color Coding**: 
  - Green: Positive Q-values (good actions)
  - Red: Negative Q-values (bad actions)
  - Yellow: Current agent position
  - Orange: Learned optimal path

## Configuration

### Q-Learning Parameters

- **Learning Rate (α)**: How much to update Q-values (default: 0.1)
- **Discount Factor (γ)**: Importance of future rewards (default: 0.9)
- **Epsilon (ε)**: Exploration rate (default: 0.1)
- **Epsilon Decay**: Rate of exploration reduction (default: 0.995)

### Reward Structure

- **Goal Reward**: +100 for reaching target
- **Step Penalty**: -1 for each move (encourages efficiency)
- **Wall Penalty**: -10 for hitting walls/boundaries

## Comparison with A*

| Aspect | A* | Q-Learning |
|--------|----|-----------| 
| Knowledge | Perfect | Learns through experience |
| Optimality | Guaranteed optimal | Converges to optimal |
| Speed | Fast | Slow initial training |
| Adaptability | None | Can adapt to changes |
| Memory | No learning | Stores Q-values |

## Dependencies

- PySide6 >= 6.5.0 (GUI framework)
- NumPy >= 1.24.0 (Numerical computations)
- Python 3.8+ (Required for type hints)

## Architecture

The project follows a clean architecture pattern:

```
rl/
├── domain/          # Core RL algorithms and types
├── app/            # Application logic and state management  
├── ui/             # User interface components
├── utils/          # Utility functions and helpers
└── __main__.py     # Application entry point
```

## Learning Process

1. **Exploration**: Agent tries random actions (high epsilon)
2. **Learning**: Q-values updated based on rewards received
3. **Exploitation**: Agent uses learned Q-values (low epsilon)
4. **Convergence**: Policy stabilizes to optimal behavior

## Future Enhancements

- Deep Q-Networks (DQN) for larger state spaces
- Policy gradient methods (REINFORCE, Actor-Critic)
- Multi-agent reinforcement learning
- Dynamic environments with moving obstacles
- 3D grid environments