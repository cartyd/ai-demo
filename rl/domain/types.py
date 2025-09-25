"""Core type definitions for the RL pathfinding algorithm."""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict
import numpy as np

# Coordinate type for grid positions
Coord = Tuple[int, int]

# Node states for visualization - extended for RL
NodeState = Literal[
    "empty", "wall", "start", "target",
    "visited", "current", "path", "optimal_path", "q_high", "q_medium", "q_low",
    "training_current", "training_visited", "training_considering"
]

# Actions the RL agent can take
Action = Literal["up", "down", "left", "right"]
ActionInt = Literal[0, 1, 2, 3]  # Numerical representation

# Training phases
TrainingPhase = Literal["training", "testing", "converged"]

# Training modes
TrainingMode = Literal["background", "visual"]


@dataclass
class QValues:
    """Stores Q-values for all actions at a state."""
    up: float = 0.0
    down: float = 0.0
    left: float = 0.0
    right: float = 0.0
    
    def as_array(self) -> np.ndarray:
        """Return Q-values as numpy array."""
        return np.array([self.up, self.down, self.left, self.right])
    
    def from_array(self, values: np.ndarray) -> None:
        """Set Q-values from numpy array."""
        self.up = float(values[0])
        self.down = float(values[1])
        self.left = float(values[2])
        self.right = float(values[3])
    
    def max_value(self) -> float:
        """Get the maximum Q-value."""
        return max(self.up, self.down, self.left, self.right)
    
    def best_action(self) -> ActionInt:
        """Get the action with highest Q-value."""
        values = self.as_array()
        return int(np.argmax(values))


@dataclass
class GridNode:
    """Represents a single node in the pathfinding grid."""
    id: str
    coord: Coord
    q_values: QValues
    walkable: bool = True
    weight: float = 1.0
    state: NodeState = "empty"
    visit_count: int = 0
    last_reward: float = 0.0

    def reset_rl_data(self):
        """Reset RL-related data."""
        self.q_values = QValues()
        self.visit_count = 0
        self.last_reward = 0.0

    def is_passable(self) -> bool:
        """Check if this node can be traversed."""
        return self.walkable and self.state not in ["wall"]


@dataclass
class Grid:
    """Represents the entire pathfinding grid."""
    width: int
    height: int
    nodes: Dict[str, GridNode]

    def get_node(self, coord: Coord) -> Optional[GridNode]:
        """Get node at coordinate, returns None if out of bounds."""
        node_id = f"{coord[0]},{coord[1]}"
        return self.nodes.get(node_id)

    def set_node_state(self, coord: Coord, state: NodeState):
        """Set the state of a node at given coordinate."""
        node = self.get_node(coord)
        if node:
            node.state = state

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if coordinate is within grid bounds."""
        x, y = coord
        return 0 <= x < self.width and 0 <= y < self.height

    def reset_visualization_states(self):
        """Reset all nodes to remove visualization states."""
        for node in self.nodes.values():
            if node.state in ["visited", "current", "path", "optimal_path", "q_high", "q_medium", "q_low",
                             "training_current", "training_visited", "training_considering"]:
                node.state = "empty"

    def reset_q_values(self):
        """Reset all Q-values in the grid."""
        for node in self.nodes.values():
            node.reset_rl_data()


@dataclass
class RLConfig:
    """Configuration for the RL algorithm."""
    learning_rate: float = 0.15  # Faster learning for complex mazes
    discount_factor: float = 0.98  # Very high for 50-step paths
    epsilon: float = 0.4  # Even higher initial exploration
    epsilon_decay: float = 0.995  # Balanced faster decay for better convergence
    epsilon_min: float = 0.03  # Lower minimum exploration
    max_episodes: int = 8000  # More episodes for very complex mazes
    max_steps_per_episode: int = 500  # Much higher limit for deep exploration
    reward_goal: float = 100.0
    reward_wall: float = -10.0
    reward_step: float = -0.5  # Reduced step penalty for long paths
    step_mode: bool = False
    show_q_values: bool = True
    # Testing configuration
    instant_testing: bool = True  # Show complete path immediately vs step-by-step
    # Visual training configuration
    training_mode: TrainingMode = "background"
    visual_step_delay: int = 10   # milliseconds between steps in visual mode (ultra-fast!)
    visual_episode_delay: int = 50  # milliseconds between episodes in visual mode (ultra-fast!)
    
    # Smart reward system - optimized for complex mazes
    use_smart_rewards: bool = True
    reward_progress: float = 5.0  # Higher reward for getting closer to goal
    reward_exploration: float = 1.0  # Strong bonus for visiting new states
    reward_dead_end: float = -15.0  # Less harsh penalty to allow exploration
    reward_revisit: float = -2.0  # Lighter penalty for revisiting (needed in complex mazes)
    reward_stuck: float = -10.0  # Lighter penalty for being stuck
    reward_backward: float = -1.0  # Very light penalty for moving away
    
    # Distance-based guidance
    use_distance_guidance: bool = True
    distance_reward_scale: float = 1.0
    
    # Dead-end detection
    detect_dead_ends: bool = True
    stuck_threshold: int = 5  # Steps without progress to consider "stuck"
    
    # Early stopping configuration
    enable_early_stopping: bool = True
    early_stop_patience: int = 100  # More patience for complex mazes
    min_improvement_threshold: float = 0.02  # Smaller threshold (2%) for gradual improvement


@dataclass
class Episode:
    """Represents a single training episode."""
    number: int
    steps: int
    total_reward: float
    reached_goal: bool
    epsilon_used: float
    elapsed_time: float = 0.0  # Time taken for this episode in seconds


@dataclass
class TrainingResult:
    """Result of RL training."""
    episodes: list[Episode]
    total_episodes: int
    successful_episodes: int
    average_reward: float
    final_epsilon: float
    converged: bool
    early_stopped: bool = False
    stopping_reason: str = ""  # Reason for early stopping
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_episodes / self.total_episodes if self.total_episodes > 0 else 0.0
    
    @property
    def best_episode_time(self) -> Optional[float]:
        """Get the best (shortest) episode time for successful episodes."""
        successful_times = [ep.elapsed_time for ep in self.episodes if ep.reached_goal and ep.elapsed_time > 0]
        return min(successful_times) if successful_times else None
    
    @property
    def worst_episode_time(self) -> Optional[float]:
        """Get the worst (longest) episode time for successful episodes."""
        successful_times = [ep.elapsed_time for ep in self.episodes if ep.reached_goal and ep.elapsed_time > 0]
        return max(successful_times) if successful_times else None
    
    @property
    def average_episode_time(self) -> Optional[float]:
        """Get the average episode time for successful episodes."""
        successful_times = [ep.elapsed_time for ep in self.episodes if ep.reached_goal and ep.elapsed_time > 0]
        return sum(successful_times) / len(successful_times) if successful_times else None


@dataclass
class PathfindingResult:
    """Result of RL pathfinding operation."""
    path: Optional[list[Coord]] = None
    path_length: int = 0
    total_reward: float = 0.0
    steps_taken: int = 0
    found: bool = False
    training_episodes: int = 0
    
    @property
    def success(self) -> bool:
        """Whether pathfinding was successful."""
        return self.found and self.path is not None and len(self.path) > 0


# Action mappings
ACTION_TO_INT: Dict[Action, ActionInt] = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3
}

INT_TO_ACTION: Dict[ActionInt, Action] = {
    0: "up",
    1: "down", 
    2: "left",
    3: "right"
}

ACTION_DELTAS: Dict[ActionInt, Coord] = {
    0: (0, -1),  # up
    1: (0, 1),   # down
    2: (-1, 0),  # left
    3: (1, 0)    # right
}