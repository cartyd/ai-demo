"""Main application controller connecting UI and domain logic."""

from typing import Optional, Callable
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QMessageBox

from ..domain.types import Grid, Coord, AlgoConfig, PathfindingResult
from ..domain.astar import AStarAlgorithm
from ..utils.grid_factory import (
    create_empty_grid, generate_solvable_grid, generate_maze_grid, 
    generate_multipath_maze, generate_branching_maze,
    place_start_and_target
)
from ..utils.rng import set_global_seed
from .fsm import AlgoStateMachine, AlgoState


class AStarController(QObject):
    """
    Controller that manages the A* algorithm execution and connects UI to domain logic.
    
    Signals:
        state_changed: Emitted when algorithm state changes
        step_completed: Emitted when one algorithm step is completed  
        algorithm_completed: Emitted when algorithm finishes (success or failure)
        grid_updated: Emitted when grid needs to be redrawn
        error_occurred: Emitted when an error occurs
    """
    
    # Qt Signals
    state_changed = Signal(object)  # AlgoState
    step_completed = Signal(object)  # Optional[PathfindingResult]
    algorithm_completed = Signal(object)  # PathfindingResult
    grid_updated = Signal()
    error_occurred = Signal(str)  # Error message
    
    def __init__(self):
        super().__init__()
        
        # Core components
        self._algorithm = AStarAlgorithm()
        self._state_machine = AlgoStateMachine()
        self._grid: Optional[Grid] = None
        self._start_coord: Optional[Coord] = None
        self._target_coord: Optional[Coord] = None
        self._config = AlgoConfig()
        
        # Timer for run mode
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_timer_tick)
        self._timer_interval = 300  # milliseconds
        
        # Setup state machine callbacks
        self._setup_state_callbacks()
        
        # Initialize with default grid
        self.create_new_grid(25, 25)
    
    def _setup_state_callbacks(self):
        """Setup callbacks for state machine transitions."""
        self._state_machine.on_state_enter(AlgoState.RUNNING, self._on_running_entered)
        self._state_machine.on_state_enter(AlgoState.PAUSED, self._on_paused_entered)
        self._state_machine.on_state_enter(AlgoState.IDLE, self._on_idle_entered)
        self._state_machine.on_state_enter(AlgoState.COMPLETE, self._on_complete_entered)
        self._state_machine.on_state_enter(AlgoState.NO_PATH, self._on_no_path_entered)
        self._state_machine.on_state_enter(AlgoState.ERROR, self._on_error_entered)
        
        self._state_machine.on_transition(AlgoState.IDLE, AlgoState.RUNNING, self._on_start_transition)
    
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
    def config(self) -> AlgoConfig:
        """Get the algorithm configuration."""
        return self._config
    
    @property
    def current_state(self) -> AlgoState:
        """Get the current algorithm state."""
        return self._state_machine.current_state
    
    @property
    def speed(self) -> int:
        """Get the current speed (timer interval in ms)."""
        return self._timer_interval
    
    @speed.setter
    def speed(self, interval_ms: int):
        """Set the speed (timer interval in ms)."""
        self._timer_interval = max(50, min(1000, interval_ms))
        if self._timer.isActive():
            self._timer.setInterval(self._timer_interval)
    
    # Grid Management
    
    def create_new_grid(self, width: int, height: int) -> bool:
        """Create a new empty grid."""
        try:
            self._grid = create_empty_grid(width, height)
            self._start_coord = None
            self._target_coord = None
            self._algorithm.reset()
            self._state_machine.reset_to_idle()
            self.grid_updated.emit()
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to create grid: {str(e)}")
            return False
    
    
    def generate_maze_grid(self, width: int, height: int, seed: Optional[int] = None) -> bool:
        """Generate a proper maze using recursive backtracking."""
        try:
            set_global_seed(seed)
            self._grid, self._start_coord, self._target_coord = generate_maze_grid(
                width, height, seed=seed
            )
            self._algorithm.reset()
            self._state_machine.reset_to_idle()
            self.grid_updated.emit()
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to generate maze: {str(e)}")
            return False
    
    
    def generate_corridor_maze(self, width: int, height: int, seed: Optional[int] = None) -> bool:
        """Generate a corridor-style maze with wider passages."""
        try:
            set_global_seed(seed)
            # Use the contiguous walls generator with high density for corridor effect
            self._grid, self._start_coord, self._target_coord = generate_solvable_grid(
                width, height, 0.6, seed=seed  # 60% walls for corridor effect
            )
            self._algorithm.reset()
            self._state_machine.reset_to_idle()
            self.grid_updated.emit()
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to generate corridor maze: {str(e)}")
            return False
    
    def generate_multipath_maze(self, width: int, height: int, seed: Optional[int] = None) -> bool:
        """Generate a maze with multiple paths to the goal."""
        try:
            set_global_seed(seed)
            self._grid, self._start_coord, self._target_coord = generate_multipath_maze(
                width, height, seed=seed
            )
            self._algorithm.reset()
            self._state_machine.reset_to_idle()
            self.grid_updated.emit()
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to generate multi-path maze: {str(e)}")
            return False
    
    def generate_branching_maze(self, width: int, height: int, seed: Optional[int] = None) -> bool:
        """Generate a branching maze with many decision points."""
        try:
            set_global_seed(seed)
            self._grid, self._start_coord, self._target_coord = generate_branching_maze(
                width, height, seed=seed
            )
            self._algorithm.reset()
            self._state_machine.reset_to_idle()
            self.grid_updated.emit()
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to generate branching maze: {str(e)}")
            return False
    
    def set_node_state(self, coord: Coord, state: str) -> bool:
        """Set the state of a node at the given coordinate."""
        if not self._grid or not self._grid.is_valid_coord(coord):
            return False
        
        if self._state_machine.is_running():
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
    
    # Algorithm Control
    
    def can_start(self) -> bool:
        """Check if algorithm can be started."""
        return (self._state_machine.can_start() and 
                self._grid is not None and 
                self._start_coord is not None and 
                self._target_coord is not None)
    
    def start_algorithm(self) -> bool:
        """Start the A* algorithm."""
        if not self.can_start():
            return False
        
        try:
            self._algorithm.initialize(self._start_coord, self._target_coord, self._grid, self._config)
            return self._state_machine.start()
        except Exception as e:
            self.error_occurred.emit(f"Failed to start algorithm: {str(e)}")
            return False
    
    def step_algorithm(self) -> bool:
        """Execute one step of the algorithm."""
        if not self._state_machine.is_running() and not self._state_machine.is_paused():
            # Allow stepping from idle if we can start
            if not self.start_algorithm():
                return False
        
        try:
            result = self._algorithm.step(self._grid)
            self.step_completed.emit(result)
            
            if result is not None:
                # Algorithm completed
                if result.success:
                    self._state_machine.complete({"result": result})
                else:
                    self._state_machine.fail_no_path({"result": result})
            
            self.grid_updated.emit()
            return True
        except Exception as e:
            self._state_machine.fail_error({"error": str(e)})
            self.error_occurred.emit(f"Algorithm error: {str(e)}")
            return False
    
    def pause_algorithm(self) -> bool:
        """Pause the algorithm."""
        return self._state_machine.pause()
    
    def resume_algorithm(self) -> bool:
        """Resume the algorithm."""
        return self._state_machine.resume()
    
    def reset_algorithm(self) -> bool:
        """Reset the algorithm to idle state."""
        self._algorithm.reset()
        if self._grid:
            self._grid.reset_pathfinding_states()
        self.grid_updated.emit()
        return self._state_machine.reset_to_idle()
    
    # Configuration
    
    def update_config(self, **kwargs):
        """Update algorithm configuration."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
    
    # State Machine Callbacks
    
    def _on_running_entered(self, context):
        """Called when entering RUNNING state."""
        if not self._config.step_mode:
            self._timer.start(self._timer_interval)
        self.state_changed.emit(AlgoState.RUNNING)
    
    def _on_paused_entered(self, context):
        """Called when entering PAUSED state."""
        self._timer.stop()
        self.state_changed.emit(AlgoState.PAUSED)
    
    def _on_idle_entered(self, context):
        """Called when entering IDLE state."""
        self._timer.stop()
        self.state_changed.emit(AlgoState.IDLE)
    
    def _on_complete_entered(self, context):
        """Called when entering COMPLETE state."""
        self._timer.stop()
        result = context.get("result") if context else None
        if result:
            # Display the optimal path on the grid
            self._display_optimal_path(result)
            self.algorithm_completed.emit(result)
        self.state_changed.emit(AlgoState.COMPLETE)
    
    def _on_no_path_entered(self, context):
        """Called when entering NO_PATH state."""
        self._timer.stop()
        result = context.get("result") if context else None
        if result:
            self.algorithm_completed.emit(result)
        self.state_changed.emit(AlgoState.NO_PATH)
    
    def _on_error_entered(self, context):
        """Called when entering ERROR state."""
        self._timer.stop()
        self.state_changed.emit(AlgoState.ERROR)
    
    def _on_start_transition(self, from_state, to_state, context):
        """Called when transitioning from IDLE to RUNNING."""
        pass  # Additional logic if needed
    
    def _on_timer_tick(self):
        """Called on each timer tick during run mode."""
        if self._state_machine.is_running():
            self.step_algorithm()
    
    # Utility methods
    
    def get_statistics(self) -> dict:
        """Get current algorithm statistics."""
        return {
            "nodes_explored": self._algorithm.nodes_explored,
            "open_set_size": self._algorithm.open_set.size(),
            "closed_set_size": len(self._algorithm.closed_set),
            "current_state": self._state_machine.current_state.value,
            "state_description": self._state_machine.get_state_description(),
        }
    
    def get_current_node(self) -> Optional[Coord]:
        """Get the currently processing node coordinate."""
        return self._algorithm.get_current_node_coord()
    
    def _display_optimal_path(self, result):
        """Display the optimal path on the grid."""
        if not result.success or not result.path or not self._grid:
            return
        
        # Mark all path nodes (except start and target) as path nodes
        for coord in result.path:
            if coord != self._start_coord and coord != self._target_coord:
                node = self._grid.get_node(coord)
                if node:
                    node.state = "path"
        
        # Emit grid update to refresh the visualization
        self.grid_updated.emit()
