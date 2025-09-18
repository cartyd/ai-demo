"""Finite State Machine for A* algorithm execution phases."""

from enum import Enum
from typing import Optional, Set, Callable
from dataclasses import dataclass


class AlgoState(Enum):
    """States for the A* algorithm execution."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETE = "complete"
    NO_PATH = "no_path"
    ERROR = "error"


@dataclass
class StateTransition:
    """Represents a state transition with optional condition."""
    from_state: AlgoState
    to_state: AlgoState
    condition: Optional[Callable[[], bool]] = None
    
    def can_transition(self) -> bool:
        """Check if transition is allowed."""
        return self.condition is None or self.condition()


class AlgoStateMachine:
    """
    Finite State Machine for managing A* algorithm execution states.
    
    State Transitions:
    IDLE -> RUNNING (when start/run is pressed)
    RUNNING -> PAUSED (when pause is pressed)
    RUNNING -> COMPLETE (when path is found)
    RUNNING -> NO_PATH (when no path exists)
    RUNNING -> ERROR (when error occurs)
    PAUSED -> RUNNING (when resume is pressed)
    PAUSED -> IDLE (when reset is pressed)
    COMPLETE -> IDLE (when reset is pressed)
    NO_PATH -> IDLE (when reset is pressed)
    ERROR -> IDLE (when reset is pressed)
    """
    
    def __init__(self):
        self._current_state = AlgoState.IDLE
        self._state_callbacks = {}
        self._transition_callbacks = {}
        self._valid_transitions = self._build_transition_map()
    
    def _build_transition_map(self) -> dict[AlgoState, Set[AlgoState]]:
        """Build the valid state transition map."""
        return {
            AlgoState.IDLE: {AlgoState.RUNNING},
            AlgoState.RUNNING: {AlgoState.PAUSED, AlgoState.COMPLETE, AlgoState.NO_PATH, AlgoState.ERROR},
            AlgoState.PAUSED: {AlgoState.RUNNING, AlgoState.IDLE},
            AlgoState.COMPLETE: {AlgoState.IDLE},
            AlgoState.NO_PATH: {AlgoState.IDLE},
            AlgoState.ERROR: {AlgoState.IDLE},
        }
    
    @property
    def current_state(self) -> AlgoState:
        """Get the current state."""
        return self._current_state
    
    def can_transition_to(self, target_state: AlgoState) -> bool:
        """Check if transition to target state is valid."""
        return target_state in self._valid_transitions.get(self._current_state, set())
    
    def transition_to(self, target_state: AlgoState, context: dict = None) -> bool:
        """
        Attempt to transition to the target state.
        
        Args:
            target_state: The state to transition to
            context: Optional context data for the transition
            
        Returns:
            True if transition was successful, False otherwise
        """
        if not self.can_transition_to(target_state):
            return False
        
        old_state = self._current_state
        self._current_state = target_state
        
        # Call transition callbacks
        transition_key = (old_state, target_state)
        if transition_key in self._transition_callbacks:
            self._transition_callbacks[transition_key](old_state, target_state, context)
        
        # Call state entry callbacks
        if target_state in self._state_callbacks:
            self._state_callbacks[target_state](context)
        
        return True
    
    def on_state_enter(self, state: AlgoState, callback: Callable[[Optional[dict]], None]):
        """Register a callback for when entering a specific state."""
        self._state_callbacks[state] = callback
    
    def on_transition(self, from_state: AlgoState, to_state: AlgoState, 
                     callback: Callable[[AlgoState, AlgoState, Optional[dict]], None]):
        """Register a callback for a specific state transition."""
        self._transition_callbacks[(from_state, to_state)] = callback
    
    def reset(self):
        """Reset the state machine to IDLE."""
        self._current_state = AlgoState.IDLE
    
    # Convenience methods for common operations
    
    def can_start(self) -> bool:
        """Check if algorithm can be started."""
        return self.can_transition_to(AlgoState.RUNNING)
    
    def can_pause(self) -> bool:
        """Check if algorithm can be paused."""
        return self._current_state == AlgoState.RUNNING and self.can_transition_to(AlgoState.PAUSED)
    
    def can_resume(self) -> bool:
        """Check if algorithm can be resumed."""
        return self._current_state == AlgoState.PAUSED and self.can_transition_to(AlgoState.RUNNING)
    
    def can_reset(self) -> bool:
        """Check if algorithm can be reset."""
        return self.can_transition_to(AlgoState.IDLE)
    
    def is_running(self) -> bool:
        """Check if algorithm is currently running."""
        return self._current_state == AlgoState.RUNNING
    
    def is_paused(self) -> bool:
        """Check if algorithm is paused."""
        return self._current_state == AlgoState.PAUSED
    
    def is_complete(self) -> bool:
        """Check if algorithm has completed successfully."""
        return self._current_state == AlgoState.COMPLETE
    
    def is_failed(self) -> bool:
        """Check if algorithm failed (no path or error)."""
        return self._current_state in [AlgoState.NO_PATH, AlgoState.ERROR]
    
    def is_idle(self) -> bool:
        """Check if algorithm is idle."""
        return self._current_state == AlgoState.IDLE
    
    def is_finished(self) -> bool:
        """Check if algorithm has finished (complete, no path, or error)."""
        return self._current_state in [AlgoState.COMPLETE, AlgoState.NO_PATH, AlgoState.ERROR]
    
    def start(self, context: dict = None) -> bool:
        """Start the algorithm."""
        return self.transition_to(AlgoState.RUNNING, context)
    
    def pause(self, context: dict = None) -> bool:
        """Pause the algorithm."""
        return self.transition_to(AlgoState.PAUSED, context)
    
    def resume(self, context: dict = None) -> bool:
        """Resume the algorithm."""
        return self.transition_to(AlgoState.RUNNING, context)
    
    def complete(self, context: dict = None) -> bool:
        """Mark the algorithm as complete."""
        return self.transition_to(AlgoState.COMPLETE, context)
    
    def fail_no_path(self, context: dict = None) -> bool:
        """Mark the algorithm as failed (no path)."""
        return self.transition_to(AlgoState.NO_PATH, context)
    
    def fail_error(self, context: dict = None) -> bool:
        """Mark the algorithm as failed (error)."""
        return self.transition_to(AlgoState.ERROR, context)
    
    def reset_to_idle(self, context: dict = None) -> bool:
        """Reset to idle state."""
        return self.transition_to(AlgoState.IDLE, context)
    
    def get_state_description(self) -> str:
        """Get a human-readable description of the current state."""
        descriptions = {
            AlgoState.IDLE: "Ready to start",
            AlgoState.RUNNING: "Algorithm running",
            AlgoState.PAUSED: "Algorithm paused",
            AlgoState.COMPLETE: "Path found",
            AlgoState.NO_PATH: "No path exists",
            AlgoState.ERROR: "Error occurred",
        }
        return descriptions.get(self._current_state, "Unknown state")