"""Main application controller connecting UI and RL domain logic."""

from typing import Optional, List
from PySide6.QtCore import QObject, QTimer, Signal, QThread
from PySide6.QtWidgets import QMessageBox

from ..domain.types import Grid, Coord, RLConfig, PathfindingResult, Episode, TrainingResult
from ..domain.qlearning import QLearningAgent, QLearningEnvironment
from ..utils.grid_factory import (
    create_empty_grid, generate_solvable_grid, generate_maze_grid,
    place_start_and_target
)
from ..utils.rng import set_global_seed
from .fsm import RLStateMachine, RLState


class TrainingWorker(QObject):
    """Worker thread for RL training to avoid blocking UI."""
    
    episode_completed = Signal(object)  # Episode
    training_finished = Signal(object)  # TrainingResult
    error_occurred = Signal(str)
    
    def __init__(self, agent: QLearningAgent, grid: Grid, start: Coord, target: Coord, episodes: int):
        super().__init__()
        self.agent = agent
        self.grid = grid
        self.start = start
        self.target = target
        self.episodes = episodes
        self.should_stop = False
    
    def stop(self):
        """Stop training."""
        self.should_stop = True
    
    def run(self):
        """Run training."""
        try:
            env = QLearningEnvironment(self.grid, self.start, self.target, self.agent.config)
            
            # Reset grid Q-values
            self.grid.reset_q_values()
            
            self.agent.training_phase = "training"
            episodes_list = []
            successful_episodes = 0
            
            for episode_num in range(self.episodes):
                if self.should_stop:
                    break
                    
                episode = self.agent.train_episode(env)
                episodes_list.append(episode)
                
                if episode.reached_goal:
                    successful_episodes += 1
                
                # Emit episode completed signal
                self.episode_completed.emit(episode)
            
            # Calculate final statistics
            total_reward = sum(ep.total_reward for ep in episodes_list)
            average_reward = total_reward / len(episodes_list) if episodes_list else 0.0
            
            # Check convergence
            converged = False
            if len(episodes_list) >= 100:
                recent_success = sum(1 for ep in episodes_list[-100:] if ep.reached_goal)
                converged = recent_success >= 90
            
            if converged:
                self.agent.training_phase = "converged"
            
            result = TrainingResult(
                episodes=episodes_list,
                total_episodes=len(episodes_list),
                successful_episodes=successful_episodes,
                average_reward=average_reward,
                final_epsilon=self.agent.epsilon,
                converged=converged
            )
            
            self.training_finished.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class RLController(QObject):
    """
    Controller that manages RL algorithm execution and connects UI to domain logic.
    
    Signals:
        state_changed: Emitted when RL state changes
        episode_completed: Emitted when training episode is completed
        training_completed: Emitted when training finishes
        testing_completed: Emitted when testing finishes
        grid_updated: Emitted when grid needs to be redrawn
        error_occurred: Emitted when an error occurs
    """
    
    # Qt Signals
    state_changed = Signal(object)  # RLState
    episode_completed = Signal(object)  # Episode
    training_completed = Signal(object)  # TrainingResult
    testing_completed = Signal(object)  # PathfindingResult
    grid_updated = Signal()
    error_occurred = Signal(str)  # Error message
    
    def __init__(self):
        super().__init__()
        
        # Core components
        self._agent = QLearningAgent(RLConfig())
        self._state_machine = RLStateMachine()
        self._grid: Optional[Grid] = None
        self._start_coord: Optional[Coord] = None
        self._target_coord: Optional[Coord] = None
        self._config = RLConfig()
        
        # Training worker thread
        self._training_thread: Optional[QThread] = None
        self._training_worker: Optional[TrainingWorker] = None
        
        # Timer for testing mode
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_timer_tick)
        self._timer_interval = 500  # milliseconds
        
        # Testing state
        self._current_test_path: Optional[List[Coord]] = None
        self._current_test_step = 0
        
        # Setup state machine callbacks
        self._setup_state_callbacks()
        
        # Initialize with a pre-generated maze for better demonstration
        self.generate_maze_grid(15, 15, seed=42)  # Fixed seed for consistent initial experience
    
    def _setup_state_callbacks(self):
        """Setup callbacks for state machine transitions."""
        self._state_machine.on_state_enter(RLState.TRAINING, self._on_training_entered)
        self._state_machine.on_state_enter(RLState.PAUSED, self._on_paused_entered)
        self._state_machine.on_state_enter(RLState.IDLE, self._on_idle_entered)
        self._state_machine.on_state_enter(RLState.TESTING, self._on_testing_entered)
        self._state_machine.on_state_enter(RLState.CONVERGED, self._on_converged_entered)
        self._state_machine.on_state_enter(RLState.ERROR, self._on_error_entered)
    
    # Properties
    
    @property
    def grid(self) -> Optional[Grid]:
        """Get the current grid."""
        return self._grid
    
    @property
    def start_coord(self) -> Optional[Coord]:
        """Get the start coordinate."""
        return self._start_coord
    
    @property
    def target_coord(self) -> Optional[Coord]:
        """Get the target coordinate."""
        return self._target_coord
    
    @property
    def config(self) -> RLConfig:
        """Get the RL configuration."""
        return self._config
    
    @property
    def current_state(self) -> RLState:
        """Get the current RL state."""
        return self._state_machine.current_state
    
    @property
    def agent(self) -> QLearningAgent:
        """Get the RL agent."""
        return self._agent
    
    # Grid Management
    
    def create_new_grid(self, width: int, height: int) -> bool:
        """Create a new empty grid."""
        try:
            self._grid = create_empty_grid(width, height)
            self._start_coord = None
            self._target_coord = None
            self._agent.reset()
            self._state_machine.reset_to_idle()
            self.grid_updated.emit()
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to create grid: {str(e)}")
            return False
    
    def generate_maze_grid(self, width: int, height: int, seed: Optional[int] = None) -> bool:
        """Generate a maze grid."""
        try:
            set_global_seed(seed)
            self._grid, self._start_coord, self._target_coord = generate_maze_grid(
                width, height, seed=seed
            )
            self._agent.reset()
            self._state_machine.reset_to_idle()
            self.grid_updated.emit()
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to generate maze: {str(e)}")
            return False
    
    def generate_solvable_grid(self, width: int, height: int, density: float = 0.3, 
                              seed: Optional[int] = None) -> bool:
        """Generate a solvable grid with walls."""
        try:
            set_global_seed(seed)
            self._grid, self._start_coord, self._target_coord = generate_solvable_grid(
                width, height, density, seed=seed
            )
            self._agent.reset()
            self._state_machine.reset_to_idle()
            self.grid_updated.emit()
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to generate grid: {str(e)}")
            return False
    
    def set_node_state(self, coord: Coord, state: str) -> bool:
        """Set the state of a node at the given coordinate."""
        if not self._grid or not self._grid.is_valid_coord(coord):
            return False
        
        if self._state_machine.is_active():
            return False  # Don't allow modifications while running
        
        try:
            if state == "start":
                # Clear old start
                if self._start_coord:
                    old_node = self._grid.get_node(self._start_coord)
                    if old_node:
                        old_node.state = "empty"
                self._start_coord = coord
            elif state == "target":
                # Clear old target
                if self._target_coord:
                    old_node = self._grid.get_node(self._target_coord)
                    if old_node:
                        old_node.state = "empty"
                self._target_coord = coord
            
            self._grid.set_node_state(coord, state)
            
            # Update node walkability
            node = self._grid.get_node(coord)
            if node:
                node.walkable = state != "wall"
            
            self.grid_updated.emit()
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to set node state: {str(e)}")
            return False
    
    def place_random_start_target(self, seed: Optional[int] = None) -> bool:
        """Place start and target at random valid positions."""
        if not self._grid:
            return False
        
        try:
            set_global_seed(seed)
            self._start_coord, self._target_coord = place_start_and_target(self._grid)
            self.grid_updated.emit()
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to place start/target: {str(e)}")
            return False
    
    # RL Algorithm Control
    
    def can_start_training(self) -> bool:
        """Check if training can be started."""
        return (self._state_machine.can_start() and 
                self._grid is not None and 
                self._start_coord is not None and 
                self._target_coord is not None)
    
    def start_training(self, episodes: int = None) -> bool:
        """Start RL training."""
        if not self.can_start_training():
            return False
        
        episodes = episodes or self._config.max_episodes
        
        try:
            # Create worker thread for training
            self._training_thread = QThread()
            self._training_worker = TrainingWorker(
                self._agent, self._grid, self._start_coord, self._target_coord, episodes
            )
            self._training_worker.moveToThread(self._training_thread)
            
            # Connect signals
            self._training_worker.episode_completed.connect(self._on_episode_completed)
            self._training_worker.training_finished.connect(self._on_training_finished)
            self._training_worker.error_occurred.connect(self._on_training_error)
            self._training_thread.started.connect(self._training_worker.run)
            self._training_thread.finished.connect(self._cleanup_training_thread)
            
            # Start training
            self._state_machine.start_training()
            self._training_thread.start()
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to start training: {str(e)}")
            return False
    
    def pause_training(self) -> bool:
        """Pause training."""
        if self._training_worker:
            self._training_worker.stop()
        return self._state_machine.pause()
    
    def resume_training(self) -> bool:
        """Resume training."""
        # For simplicity, just restart training from current state
        return self._state_machine.resume()
    
    def start_testing(self) -> bool:
        """Start testing the learned policy."""
        if not self.can_start_training():
            return False
        
        try:
            # Find path using learned policy
            result = self._agent.find_path(self._grid, self._start_coord, self._target_coord)
            
            if result.success:
                self._current_test_path = result.path
                self._current_test_step = 0
                self._state_machine.start_testing()
                
                # Start timer for step-by-step visualization
                if not self._config.step_mode:
                    self._timer.start(self._timer_interval)
                
                return True
            else:
                self.error_occurred.emit("Agent could not find path. Training may be insufficient.")
                return False
                
        except Exception as e:
            self.error_occurred.emit(f"Failed to start testing: {str(e)}")
            return False
    
    def step_testing(self) -> bool:
        """Execute one step of testing."""
        if not self._state_machine.is_testing() or not self._current_test_path:
            return False
        
        try:
            if self._current_test_step < len(self._current_test_path):
                # Update visualization
                coord = self._current_test_path[self._current_test_step]
                if coord != self._start_coord and coord != self._target_coord:
                    self._grid.set_node_state(coord, "current")
                
                self._current_test_step += 1
                self.grid_updated.emit()
                
                # Check if testing is complete
                if self._current_test_step >= len(self._current_test_path):
                    self._complete_testing()
                
                return True
            else:
                self._complete_testing()
                return False
                
        except Exception as e:
            self.error_occurred.emit(f"Testing error: {str(e)}")
            return False
    
    def _complete_testing(self):
        """Complete testing and show final path."""
        if self._current_test_path:
            # Mark entire path
            for coord in self._current_test_path:
                if coord != self._start_coord and coord != self._target_coord:
                    self._grid.set_node_state(coord, "path")
            
            result = PathfindingResult(
                path=self._current_test_path,
                path_length=len(self._current_test_path),
                found=True,
                training_episodes=self._agent.episodes_completed
            )
            
            self.testing_completed.emit(result)
        
        self._timer.stop()
        self._state_machine.reset_to_idle()
        self.grid_updated.emit()
    
    def reset_algorithm(self) -> bool:
        """Reset the RL algorithm."""
        # Stop any running operations
        if self._training_worker:
            self._training_worker.stop()
        self._timer.stop()
        
        # Reset agent and grid
        self._agent.reset()
        if self._grid:
            self._grid.reset_visualization_states()
            self._grid.reset_q_values()
        
        self._current_test_path = None
        self._current_test_step = 0
        
        self.grid_updated.emit()
        return self._state_machine.reset_to_idle()
    
    # Configuration
    
    def update_config(self, **kwargs):
        """Update RL configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        # Update agent config
        self._agent.config = self._config
    
    # State Machine Callbacks
    
    def _on_training_entered(self, context):
        """Called when entering TRAINING state."""
        self.state_changed.emit(RLState.TRAINING)
    
    def _on_paused_entered(self, context):
        """Called when entering PAUSED state."""
        self._timer.stop()
        self.state_changed.emit(RLState.PAUSED)
    
    def _on_idle_entered(self, context):
        """Called when entering IDLE state."""
        self._timer.stop()
        self.state_changed.emit(RLState.IDLE)
    
    def _on_testing_entered(self, context):
        """Called when entering TESTING state."""
        self.state_changed.emit(RLState.TESTING)
    
    def _on_converged_entered(self, context):
        """Called when entering CONVERGED state."""
        self.state_changed.emit(RLState.CONVERGED)
    
    def _on_error_entered(self, context):
        """Called when entering ERROR state."""
        self._timer.stop()
        self.state_changed.emit(RLState.ERROR)
    
    def _on_timer_tick(self):
        """Called on each timer tick during testing."""
        if self._state_machine.is_testing():
            self.step_testing()
    
    # Training callbacks
    
    def _on_episode_completed(self, episode: Episode):
        """Called when a training episode is completed."""
        self.episode_completed.emit(episode)
        self.grid_updated.emit()  # Update Q-value visualization
    
    def _on_training_finished(self, result: TrainingResult):
        """Called when training is finished."""
        if result.converged:
            self._state_machine.converge()
        else:
            self._state_machine.reset_to_idle()
        
        self.training_completed.emit(result)
        self.grid_updated.emit()
    
    def _on_training_error(self, error_message: str):
        """Called when training error occurs."""
        self._state_machine.fail_error()
        self.error_occurred.emit(f"Training error: {error_message}")
    
    def _cleanup_training_thread(self):
        """Clean up training thread."""
        if self._training_thread:
            self._training_thread.deleteLater()
            self._training_thread = None
        if self._training_worker:
            self._training_worker.deleteLater()
            self._training_worker = None
    
    # Utility methods
    
    def get_statistics(self) -> dict:
        """Get current RL statistics."""
        return {
            "episodes_completed": self._agent.episodes_completed,
            "current_epsilon": self._agent.epsilon,
            "training_phase": self._agent.training_phase,
            "current_state": self._state_machine.current_state.value,
            "state_description": self._state_machine.get_state_description(),
        }