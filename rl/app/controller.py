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
    
    episode_completed = Signal(object)  # Episode (throttled)
    progress_updated = Signal(int, int)  # current_episode, total_episodes
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
        self.is_paused = False
        self.episode_update_interval = max(1, episodes // 100)  # Update UI ~100 times max
        
        # Early stopping tracking
        self.successful_times = []  # Track times of successful episodes
        self.no_improvement_count = 0
        self.best_time = float('inf')
    
    def stop(self):
        """Stop training."""
        self.should_stop = True
    
    def pause(self):
        """Pause training."""
        self.is_paused = True
    
    def resume(self):
        """Resume training."""
        self.is_paused = False
    
    def _should_stop_early(self, episode: Episode) -> tuple[bool, str]:
        """Check if training should stop early due to convergence in performance."""
        if not self.agent.config.enable_early_stopping:
            return False, ""
        
        # Only consider successful episodes
        if not episode.reached_goal or episode.elapsed_time <= 0:
            return False, ""
        
        current_time = episode.elapsed_time
        self.successful_times.append(current_time)
        
        # First successful episode sets the baseline
        if len(self.successful_times) == 1:
            self.best_time = current_time
            return False, ""
        
        # Check if current time is an improvement
        improvement_threshold = self.agent.config.min_improvement_threshold
        if current_time < self.best_time * (1 - improvement_threshold):
            # Significant improvement found
            self.best_time = current_time
            self.no_improvement_count = 0
        else:
            # No significant improvement
            self.no_improvement_count += 1
        
        # Check if we've reached the patience limit
        if self.no_improvement_count >= self.agent.config.early_stop_patience:
            reason = f"No time improvement for {self.no_improvement_count} consecutive successful episodes (best: {self.best_time:.3f}s)"
            return True, reason
        
        return False, ""
    def run(self):
        """Run training with frequent stop checks."""
        try:
            env = QLearningEnvironment(self.grid, self.start, self.target, self.agent.config)
            
            # Reset grid Q-values
            self.grid.reset_q_values()
            
            self.agent.training_phase = "training"
            episodes_list = []
            successful_episodes = 0
            early_stopped = False
            stopping_reason = ""
            
            for episode_num in range(self.episodes):
                # Check for stop signal at the beginning of each episode
                if self.should_stop:
                    stopping_reason = "Training stopped by user"
                    break
                
                # Handle pause with frequent stop checks
                while self.is_paused and not self.should_stop:
                    import time
                    time.sleep(0.01)  # Very short sleep for rapid response
                
                if self.should_stop:
                    stopping_reason = "Training stopped by user"
                    break
                
                # Train episode with stop check capability
                try:
                    episode = self.agent.train_episode(env)
                    
                    # Check for stop again after episode completion
                    if self.should_stop:
                        stopping_reason = "Training stopped by user"
                        break
                    
                    episodes_list.append(episode)
                    
                    if episode.reached_goal:
                        successful_episodes += 1
                        
                        # Check for early stopping after successful episode
                        should_stop, reason = self._should_stop_early(episode)
                        if should_stop:
                            early_stopped = True
                            stopping_reason = reason
                            self.agent.training_phase = "converged"
                            break
                    
                    # Emit progress signal for every episode (lightweight)
                    self.progress_updated.emit(episode_num + 1, self.episodes)
                    
                    # Only emit detailed episode data occasionally to avoid overwhelming UI
                    if (episode_num + 1) % self.episode_update_interval == 0 or episode_num == self.episodes - 1:
                        self.episode_completed.emit(episode)
                        
                except Exception as episode_error:
                    # If training episode fails, check if we should stop
                    if self.should_stop:
                        stopping_reason = "Training stopped by user"
                        break
                    # Otherwise re-raise the error
                    raise episode_error
            
            # Final stop check before emitting results
            if self.should_stop and not stopping_reason:
                stopping_reason = "Training stopped by user"
            
            # Calculate final statistics
            total_reward = sum(ep.total_reward for ep in episodes_list)
            average_reward = total_reward / len(episodes_list) if episodes_list else 0.0
            
            # Check convergence (only if not already early stopped)
            converged = early_stopped
            if not early_stopped and len(episodes_list) >= 100:
                recent_success = sum(1 for ep in episodes_list[-100:] if ep.reached_goal)
                converged = recent_success >= 90
                if converged:
                    self.agent.training_phase = "converged"
                    stopping_reason = f"Converged with {recent_success}% success rate over last 100 episodes"
            
            result = TrainingResult(
                episodes=episodes_list,
                total_episodes=len(episodes_list),
                successful_episodes=successful_episodes,
                average_reward=average_reward,
                final_epsilon=self.agent.epsilon,
                converged=converged,
                early_stopped=early_stopped,
                stopping_reason=stopping_reason
            )
            
            # Only emit result if we haven't been asked to stop
            if not self.should_stop:
                self.training_finished.emit(result)
            
        except Exception as e:
            # Check if we're stopping before emitting error
            if not self.should_stop:
                self.error_occurred.emit(str(e))
        
        # Always print completion message for debugging
        print(f"TrainingWorker.run() completed (stopped: {self.should_stop})")


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
    training_progress = Signal(int, int)  # current_episode, total_episodes
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
        self._timer_interval = 100  # milliseconds - faster testing visualization
        
        # Testing state
        self._current_test_path: Optional[List[Coord]] = None
        self._current_test_step = 0
        self._testing_phase = "moving"  # "considering" or "moving"
        
        # Visual training state
        self._visual_episodes_remaining = 0
        self._visual_episode_active = False
        self._visual_successful_times = []  # Track times for visual training early stopping
        self._visual_no_improvement_count = 0
        self._visual_best_time = float('inf')
        self._original_episode_target = 0  # Track original training target for mode transitions
        
        # Setup state machine callbacks
        self._setup_state_callbacks()
        
        # Initialize with a pre-generated maze for better demonstration
        self.generate_maze_grid(15, 15, seed=42)  # Fixed seed for consistent initial experience
    
    def _should_stop_visual_training_early(self, episode: Episode) -> tuple[bool, str]:
        """Check if visual training should stop early due to time convergence."""
        if not self._config.enable_early_stopping:
            return False, ""
        
        # Only consider successful episodes
        if not episode.reached_goal or episode.elapsed_time <= 0:
            return False, ""
        
        current_time = episode.elapsed_time
        self._visual_successful_times.append(current_time)
        
        # First successful episode sets the baseline
        if len(self._visual_successful_times) == 1:
            self._visual_best_time = current_time
            return False, ""
        
        # Check if current time is an improvement
        improvement_threshold = self._config.min_improvement_threshold
        if current_time < self._visual_best_time * (1 - improvement_threshold):
            # Significant improvement found
            self._visual_best_time = current_time
            self._visual_no_improvement_count = 0
        else:
            # No significant improvement
            self._visual_no_improvement_count += 1
        
        # Check if we've reached the patience limit
        if self._visual_no_improvement_count >= self._config.early_stop_patience:
            reason = f"No time improvement for {self._visual_no_improvement_count} consecutive successful episodes (best: {self._visual_best_time:.3f}s)"
            return True, reason
        
        return False, ""
    
    def _validate_training_data(self) -> tuple[bool, str]:
        """Validate if agent has sufficient training data for reliable testing."""
        episodes_completed = self._agent.episodes_completed
        training_history = self._agent.training_history
        
        # Define minimum requirements
        min_episodes = 50  # Minimum total episodes
        min_successful = 10  # Minimum successful episodes
        min_success_rate = 0.2  # Minimum 20% success rate
        
        # Check if any training has been done
        if episodes_completed == 0:
            return False, "No training completed yet. Click 'Train' to start learning before testing."
        
        # Check minimum episodes
        if episodes_completed < min_episodes:
            return False, f"Insufficient training data. Completed {episodes_completed} episodes, but need at least {min_episodes}. Run more training episodes."
        
        # Check successful episodes if we have training history
        if training_history:
            successful_episodes = sum(1 for ep in training_history if ep.reached_goal)
            success_rate = successful_episodes / len(training_history)
            
            if successful_episodes < min_successful:
                return False, f"Agent needs more successful training. Only {successful_episodes} successful episodes out of {len(training_history)}. Need at least {min_successful} successful episodes."
            
            if success_rate < min_success_rate:
                return False, f"Agent success rate too low ({success_rate:.1%}). Need at least {min_success_rate:.1%} success rate. Try training longer or adjusting maze difficulty."
            
            # Check recent performance (last 20 episodes)
            if len(training_history) >= 20:
                recent_episodes = training_history[-20:]
                recent_successes = sum(1 for ep in recent_episodes if ep.reached_goal)
                recent_success_rate = recent_successes / len(recent_episodes)
                
                if recent_success_rate < 0.1:  # Less than 10% success in recent episodes
                    return False, f"Agent performance has declined. Recent success rate: {recent_success_rate:.1%}. Consider more training or resetting the algorithm."
        
        return True, ""  # Validation passed
    
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
        """Start RL training (background or visual based on config)."""
        if not self.can_start_training():
            return False
        
        episodes = episodes or self._config.max_episodes
        
        if self._config.training_mode == "visual":
            return self._start_visual_training(episodes)
        else:
            return self._start_background_training(episodes)
    
    def _start_background_training(self, episodes: int) -> bool:
        """Start background training in worker thread."""
        try:
            # Store original target for potential mode switching
            self._original_episode_target = episodes
            
            # Clean up any existing thread first
            self._cleanup_training_thread()
            
            # Create worker thread for training
            self._training_thread = QThread()
            # Set the thread to auto-delete when finished to prevent destructor issues
            self._training_thread.setObjectName("RL-TrainingThread")
            
            self._training_worker = TrainingWorker(
                self._agent, self._grid, self._start_coord, self._target_coord, episodes
            )
            self._training_worker.moveToThread(self._training_thread)
            
            # Connect signals
            self._training_worker.episode_completed.connect(self._on_episode_completed)
            self._training_worker.progress_updated.connect(self._on_progress_updated)
            self._training_worker.training_finished.connect(self._on_training_finished)
            self._training_worker.training_finished.connect(self._cleanup_training_thread)
            self._training_worker.error_occurred.connect(self._on_training_error)
            self._training_thread.started.connect(self._training_worker.run)
            # Also connect to thread finished signal as backup cleanup
            self._training_thread.finished.connect(self._cleanup_training_thread)
            
            # Start training
            self._state_machine.start_training()
            self._training_thread.start()
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to start background training: {str(e)}")
            return False
    
    def _start_visual_training(self, episodes: int) -> bool:
        """Start visual training with step-by-step visualization."""
        try:
            self._visual_episodes_remaining = episodes
            self._visual_episode_active = False
            self._original_episode_target = episodes  # Store original target
            
            # Reset early stopping state
            self._visual_successful_times = []
            self._visual_no_improvement_count = 0
            self._visual_best_time = float('inf')
            
            # Reset grid states
            if self._grid:
                self._grid.reset_q_values()
                self._grid.reset_visualization_states()
            
            # Start first episode
            self._state_machine.start_training()
            self._start_next_visual_episode()
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to start visual training: {str(e)}")
            return False
    
    def _start_next_visual_episode(self):
        """Start the next visual training episode."""
        if self._visual_episodes_remaining <= 0:
            self._complete_visual_training()
            return
        
        # Check if we should stop early due to achieving validation criteria
        episodes = self._agent.training_history
        if episodes:
            episodes_completed = len(episodes)
            successful_episodes = sum(1 for ep in episodes if ep.reached_goal)
            success_rate = successful_episodes / episodes_completed if episodes_completed > 0 else 0.0
            
            # Stop early if we meet validation criteria and have high success rate
            if (episodes_completed >= 50 and successful_episodes >= 10 and success_rate >= 0.95):
                self._complete_visual_training(
                    early_stopped=True, 
                    stopping_reason=f"Training criteria exceeded: {success_rate:.1%} success rate with {successful_episodes} successful episodes out of {episodes_completed} total"
                )
                return
        
        # Start new episode
        if self._agent.start_visual_episode(self._grid, self._start_coord, self._target_coord):
            self._visual_episode_active = True
            self._visual_episodes_remaining -= 1
            
            # Start timer for step progression
            self._timer.setInterval(self._config.visual_step_delay)
            self._timer.start()
            
            self.grid_updated.emit()
        else:
            self.error_occurred.emit("Failed to start visual episode")
    
    def _complete_visual_training(self, early_stopped: bool = False, stopping_reason: str = ""):
        """Complete visual training and emit results."""
        # Calculate training results
        episodes = self._agent.training_history
        successful_episodes = sum(1 for ep in episodes if ep.reached_goal)
        total_reward = sum(ep.total_reward for ep in episodes)
        average_reward = total_reward / len(episodes) if episodes else 0.0
        
        # Check convergence (only if not already early stopped)
        converged = early_stopped
        if not early_stopped:
            # Check basic validation criteria first (matches _validate_training_data requirements)
            episodes_completed = len(episodes)
            success_rate = successful_episodes / episodes_completed if episodes_completed > 0 else 0.0
            
            if (episodes_completed >= 50 and successful_episodes >= 10 and success_rate >= 0.2):
                # Check for high-performance convergence
                if len(episodes) >= 100:
                    recent_success = sum(1 for ep in episodes[-100:] if ep.reached_goal)
                    recent_success_rate = recent_success / 100
                    if recent_success_rate >= 0.9:
                        converged = True
                        stopping_reason = f"Converged with {recent_success_rate:.1%} success rate over last 100 episodes"
                # Also check for very high success rate with fewer episodes
                elif episodes_completed >= 50 and success_rate >= 0.95:
                    converged = True
                    stopping_reason = f"High performance achieved: {success_rate:.1%} success rate over {episodes_completed} episodes"
        
        if converged:
            self._agent.training_phase = "converged"
            self._state_machine.converge()
        else:
            self._state_machine.reset_to_idle()
        
        result = TrainingResult(
            episodes=episodes,
            total_episodes=len(episodes),
            successful_episodes=successful_episodes,
            average_reward=average_reward,
            final_epsilon=self._agent.epsilon,
            converged=converged,
            early_stopped=early_stopped,
            stopping_reason=stopping_reason
        )
        
        self._timer.stop()
        self._visual_episode_active = False
        self.training_completed.emit(result)
        self.grid_updated.emit()
    
    def pause_training(self) -> bool:
        """Pause training."""
        if self._config.training_mode == "background" and self._training_worker:
            self._training_worker.pause()
        elif self._config.training_mode == "visual":
            self._timer.stop()
        return self._state_machine.pause()
    
    def resume_training(self) -> bool:
        """Resume training."""
        if self._config.training_mode == "background" and self._training_worker:
            self._training_worker.resume()
        elif self._config.training_mode == "visual" and self._visual_episode_active:
            self._timer.start(self._config.visual_step_delay)
        return self._state_machine.resume()
    
    def stop_training(self) -> bool:
        """Stop training completely."""
        if self._training_worker:
            self._training_worker.stop()
            # Wait for the thread to finish gracefully
            if self._training_thread and self._training_thread.isRunning():
                if not self._training_thread.wait(2000):  # Wait up to 2 seconds
                    self._training_thread.terminate()  # Force termination if needed
                    self._training_thread.wait(500)  # Give it time to terminate
        if self._config.training_mode == "visual":
            self._timer.stop()
            self._visual_episode_active = False
        return self._state_machine.reset_to_idle()
    
    def start_testing(self) -> bool:
        """Start testing the learned policy."""
        if not self.can_start_training():
            return False
        
        # Check if agent has sufficient training data
        validation_result = self._validate_training_data()
        if not validation_result[0]:
            self.error_occurred.emit(validation_result[1])
            return False
        
        try:
            # Clear any previous visualization states
            if self._grid:
                self._grid.reset_visualization_states()
            
            # Reset testing state
            self._current_test_path = None
            self._current_test_step = 0
            self._testing_phase = "moving"
            
            # Find path using learned policy
            result = self._agent.find_path(self._grid, self._start_coord, self._target_coord)
            
            if result.success:
                self._current_test_path = result.path
                self._current_test_step = 0
                
                # Transition to testing state
                if not self._state_machine.start_testing():
                    self.error_occurred.emit("Could not transition to testing state")
                    return False
                
                # Option 1: Instant testing - show complete path immediately
                # Option 2: Step-by-step visualization for demonstration
                
                if self._config.instant_testing:
                    # Show complete path immediately
                    self._complete_testing()
                else:
                    # Start timer for step-by-step visualization
                    if not self._config.step_mode:
                        self._timer.start(self._timer_interval)
                
                # Emit initial grid update to show clean state
                self.grid_updated.emit()
                
                return True
            else:
                self.error_occurred.emit(f"Agent could not find path. Training may be insufficient. Path length: {result.path_length}, Episodes: {result.training_episodes}")
                return False
                
        except Exception as e:
            self.error_occurred.emit(f"Failed to start testing: {str(e)}")
            return False
    
    def step_testing(self) -> bool:
        """Execute one step of testing with decision visualization."""
        if not self._state_machine.is_testing():
            return False
            
        if not self._current_test_path:
            self.error_occurred.emit("No test path available")
            return False
        
        try:
            if self._current_test_step < len(self._current_test_path):
                coord = self._current_test_path[self._current_test_step]
                
                if self._testing_phase == "moving":
                    # First show decision consideration
                    if self._current_test_step < len(self._current_test_path) - 1:
                        self._show_decision_process(coord)
                        self._testing_phase = "considering"
                    else:
                        # Final step, just move
                        self._move_to_position(coord)
                        self._current_test_step += 1
                        
                        if self._current_test_step >= len(self._current_test_path):
                            self._complete_testing()
                    
                elif self._testing_phase == "considering":
                    # Now actually move to the position
                    self._clear_decision_highlights()
                    self._move_to_position(coord)
                    self._current_test_step += 1
                    self._testing_phase = "moving"
                    
                    if self._current_test_step >= len(self._current_test_path):
                        self._complete_testing()
                        return True  # Return early since testing is complete
                
                # Always emit grid update to ensure UI refreshes
                self.grid_updated.emit()
                return True
            else:
                self._complete_testing()
                return False
                
        except Exception as e:
            self.error_occurred.emit(f"Testing error: {str(e)}")
            return False
    
    def _show_decision_process(self, current_coord: Coord):
        """Show the decision process by highlighting possible moves."""
        from ..domain.types import ACTION_DELTAS
        
        # Get valid actions from current position
        valid_actions = []
        for action in range(4):
            delta = ACTION_DELTAS[action]
            next_pos = (current_coord[0] + delta[0], current_coord[1] + delta[1])
            
            if (self._grid.is_valid_coord(next_pos) and 
                self._grid.get_node(next_pos) and 
                self._grid.get_node(next_pos).is_passable()):
                valid_actions.append((action, next_pos))
        
        # Highlight possible moves and show Q-values
        for action, next_pos in valid_actions:
            if next_pos != self._start_coord and next_pos != self._target_coord:
                node = self._grid.get_node(next_pos)
                if node and node.state not in ["current", "path", "optimal_path"]:
                    self._grid.set_node_state(next_pos, "training_considering")
    
    def _clear_decision_highlights(self):
        """Clear decision process highlights."""
        for node in self._grid.nodes.values():
            if node.state == "training_considering":
                node.state = "empty"
    
    def _move_to_position(self, coord: Coord):
        """Move to a position and update visualization."""
        if coord != self._start_coord and coord != self._target_coord:
            self._grid.set_node_state(coord, "current")
    
    def _complete_testing(self):
        """Complete testing and show final path."""
        # Clear any decision highlights
        self._clear_decision_highlights()
        
        if self._current_test_path:
            # Check if validation criteria are met to determine path highlighting
            validation_result = self._validate_training_data()
            path_state = "optimal_path" if validation_result[0] else "path"
            
            # Mark entire path with appropriate state
            for coord in self._current_test_path:
                if coord != self._start_coord and coord != self._target_coord:
                    self._grid.set_node_state(coord, path_state)
            
            result = PathfindingResult(
                path=self._current_test_path,
                path_length=len(self._current_test_path),
                found=True,
                training_episodes=self._agent.episodes_completed
            )
            
            self.testing_completed.emit(result)
        
        self._timer.stop()
        self._testing_phase = "moving"
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
        
        # Reset testing state
        self._current_test_path = None
        self._current_test_step = 0
        self._testing_phase = "moving"
        
        # Reset visual training state
        self._visual_episodes_remaining = 0
        self._visual_episode_active = False
        self._visual_successful_times = []
        self._visual_no_improvement_count = 0
        self._visual_best_time = float('inf')
        self._original_episode_target = 0
        
        self.grid_updated.emit()
        return self._state_machine.reset_to_idle()
    
    def step_visual_training(self) -> bool:
        """Manually execute one step of visual training (for step mode)."""
        if (not self._state_machine.is_training() or 
            self._config.training_mode != "visual" or 
            not self._visual_episode_active):
            return False
        
        self._step_visual_training()
        return True
    
    # Configuration
    
    def update_config(self, **kwargs):
        """Update RL configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        # Update agent config
        self._agent.config = self._config
    
    def handle_training_mode_transition(self, new_mode: str):
        """Handle seamless transition between training modes during active training."""
        if not self._state_machine.is_training():
            return
            
        if new_mode == "background" and self._config.training_mode == "background":
            # Switching from visual to background
            # Stop visual training timer and convert to background training
            self._timer.stop()
            self._visual_episode_active = False
            
            # Calculate remaining episodes
            remaining_episodes = max(0, self._visual_episodes_remaining)
            if remaining_episodes > 0:
                # Start background training for remaining episodes
                self._start_background_training(remaining_episodes)
                
        elif new_mode == "visual" and self._config.training_mode == "visual":
            # Switching from background to visual
            # Stop background training and convert to visual training
            if self._training_worker:
                self._training_worker.stop()
                
            # Clean up background training
            self._cleanup_training_thread()
            
            # Calculate remaining episodes based on original plan
            current_episodes = self._agent.episodes_completed
            remaining_episodes = max(0, self._original_episode_target - current_episodes)
            
            if remaining_episodes > 0:
                # Start visual training for remaining episodes
                self._visual_episodes_remaining = remaining_episodes
                self._visual_episode_active = False
                self._start_next_visual_episode()
    
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
    
    def _step_visual_training(self):
        """Execute one step of visual training."""
        if not self._visual_episode_active:
            return
        
        try:
            episode_done, episode = self._agent.step_visual_training()
            
            if episode_done:
                # Episode finished
                if episode:
                    self.episode_completed.emit(episode)
                    
                    # Check for early stopping after successful episode
                    if episode.reached_goal:
                        should_stop, reason = self._should_stop_visual_training_early(episode)
                        if should_stop:
                            # Stop visual training early
                            self._complete_visual_training(early_stopped=True, stopping_reason=reason)
                            return
                
                # Delay before starting next episode
                self._visual_episode_active = False
                self._timer.setInterval(self._config.visual_episode_delay)
                self._timer.singleShot(self._config.visual_episode_delay, self._start_next_visual_episode)
            
            self.grid_updated.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"Visual training error: {str(e)}")
    
    def _on_timer_tick(self):
        """Called on each timer tick during testing or visual training."""
        try:
            if self._state_machine.is_testing():
                if not self.step_testing():
                    # If step_testing returns False, stop the timer
                    self._timer.stop()
            elif self._state_machine.is_training():
                # Only step visual training if we're in visual mode and have an active episode
                if self._config.training_mode == "visual" and self._visual_episode_active:
                    self._step_visual_training()
                elif self._config.training_mode == "background":
                    # If we switched to background mode but timer is still running, stop it
                    self._timer.stop()
        except Exception as e:
            self.error_occurred.emit(f"Timer tick error: {str(e)}")
            self._timer.stop()
    
    # Training callbacks
    
    def _on_episode_completed(self, episode: Episode):
        """Called when a training episode is completed."""
        self.episode_completed.emit(episode)
        self.grid_updated.emit()  # Update Q-value visualization
    
    def _on_progress_updated(self, current_episode: int, total_episodes: int):
        """Called when training progress is updated."""
        self.training_progress.emit(current_episode, total_episodes)
    
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
    
    def _cleanup_training_thread(self, shutdown_mode=False):
        """Clean up training thread with proper synchronization."""
        try:
            # Stop and disconnect worker first
            if self._training_worker:
                try:
                    self._training_worker.stop()
                    # Disconnect all signals to prevent issues during cleanup
                    self._training_worker.blockSignals(True)
                except RuntimeError:
                    pass  # Worker already deleted or disconnected
                # Don't delete worker yet - let thread cleanup handle it
            
            # Clean up thread with proper synchronization
            if self._training_thread:
                try:
                    if self._training_thread.isRunning():
                        # First try to stop gracefully
                        self._training_thread.quit()
                        
                        # Wait longer for graceful termination
                        wait_time = 50 if shutdown_mode else 1000  # 50ms vs 1s
                        if not self._training_thread.wait(wait_time):
                            # If graceful shutdown fails, force termination
                            print("Warning: Force terminating training thread")
                            self._training_thread.terminate()
                            # Give more time for forced termination
                            final_wait = 100 if shutdown_mode else 500
                            if not self._training_thread.wait(final_wait):
                                print("Error: Training thread failed to terminate")
                    
                    # Only delete thread after it's completely stopped
                    if not self._training_thread.isRunning():
                        # Disconnect all signals first to prevent callbacks during deletion
                        try:
                            self._training_thread.blockSignals(True)
                        except RuntimeError:
                            pass
                        
                        # Now safe to delete the thread object
                        self._training_thread.deleteLater()
                        
                        # Process events to ensure deletion happens
                        try:
                            from PySide6.QtWidgets import QApplication
                            app = QApplication.instance()
                            if app:
                                # Process events multiple times to ensure proper cleanup
                                for _ in range(3):
                                    app.processEvents()
                        except Exception:
                            pass
                    
                except RuntimeError:
                    pass  # Thread already deleted
                
                # Clear reference regardless of success
                self._training_thread = None
            
            # Now safe to delete worker
            if self._training_worker:
                try:
                    self._training_worker.deleteLater()
                except RuntimeError:
                    pass
                self._training_worker = None
                
        except Exception as e:
            # Log cleanup errors but ensure references are cleared
            print(f"Thread cleanup error: {e}")
            self._training_thread = None
            self._training_worker = None
    
    def cleanup(self):
        """Clean up all resources before application shutdown."""
        try:
            # Stop timer first (catch Qt internal object deletion)
            try:
                if hasattr(self, '_timer') and self._timer is not None:
                    self._timer.stop()
                    self._timer.deleteLater()
            except RuntimeError:
                pass  # Qt object already deleted
            
            # Force stop any running operations immediately
            try:
                if self._training_worker:
                    self._training_worker.stop()
            except Exception:
                pass
            
            # Reset visual training state to prevent timer issues
            self._visual_episode_active = False
            self._visual_episodes_remaining = 0
            
            # Clean up training thread and worker with shutdown mode
            self._cleanup_training_thread(shutdown_mode=True)
            
            # Final safety check: ensure thread is completely stopped before returning
            max_attempts = 10
            attempt = 0
            while (hasattr(self, '_training_thread') and self._training_thread and 
                   self._training_thread.isRunning() and attempt < max_attempts):
                try:
                    print(f"Waiting for thread termination (attempt {attempt + 1})")
                    self._training_thread.terminate()
                    if self._training_thread.wait(10):  # 10ms wait
                        break
                    attempt += 1
                except Exception:
                    break
            
            # Force clear references regardless
            self._training_thread = None
            self._training_worker = None
            
            # Disconnect all signals to prevent issues during shutdown
            try:
                self.blockSignals(True)
            except RuntimeError:
                pass  # Qt object already deleted
            
            # Process events to ensure cleanup operations complete
            try:
                from PySide6.QtWidgets import QApplication
                app = QApplication.instance()
                if app:
                    # Multiple event processing rounds for thorough cleanup
                    for _ in range(5):
                        app.processEvents()
            except Exception:
                pass
                
        except Exception as e:
            # Log cleanup warnings during shutdown but ensure references are cleared
            print(f"Cleanup warning: {e}")
            self._training_thread = None
            self._training_worker = None
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.cleanup()
    
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