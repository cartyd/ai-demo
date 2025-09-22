"""Finite State Machine for RL algorithm execution states."""

from enum import Enum, auto
from typing import Dict, Callable, Optional, Any


class RLState(Enum):
    """States for RL algorithm execution."""
    IDLE = auto()
    TRAINING = auto()
    PAUSED = auto()
    TESTING = auto()
    CONVERGED = auto()
    ERROR = auto()


class RLStateMachine:
    """State machine for managing RL algorithm execution."""
    
    def __init__(self):
        self.current_state = RLState.IDLE
        self._enter_callbacks: Dict[RLState, Callable[[Optional[Dict]], None]] = {}
        self._exit_callbacks: Dict[RLState, Callable[[Optional[Dict]], None]] = {}
        self._transition_callbacks: Dict[tuple, Callable[[RLState, RLState, Optional[Dict]], None]] = {}
        
        # Define valid state transitions
        self._valid_transitions = {
            RLState.IDLE: {RLState.TRAINING, RLState.TESTING},
            RLState.TRAINING: {RLState.PAUSED, RLState.CONVERGED, RLState.ERROR, RLState.IDLE},
            RLState.PAUSED: {RLState.TRAINING, RLState.IDLE},
            RLState.TESTING: {RLState.IDLE, RLState.ERROR},
            RLState.CONVERGED: {RLState.TESTING, RLState.IDLE, RLState.TRAINING},
            RLState.ERROR: {RLState.IDLE},
        }
    
    def on_state_enter(self, state: RLState, callback: Callable[[Optional[Dict]], None]):
        """Register callback for state entry."""
        self._enter_callbacks[state] = callback
    
    def on_state_exit(self, state: RLState, callback: Callable[[Optional[Dict]], None]):
        """Register callback for state exit."""
        self._exit_callbacks[state] = callback
    
    def on_transition(self, from_state: RLState, to_state: RLState, 
                     callback: Callable[[RLState, RLState, Optional[Dict]], None]):
        """Register callback for state transition."""
        self._transition_callbacks[(from_state, to_state)] = callback
    
    def can_transition(self, to_state: RLState) -> bool:
        """Check if transition to target state is valid."""
        return to_state in self._valid_transitions.get(self.current_state, set())
    
    def transition(self, to_state: RLState, context: Optional[Dict] = None) -> bool:
        """Attempt to transition to target state."""
        if not self.can_transition(to_state):
            return False
        
        from_state = self.current_state
        
        # Call exit callback for current state
        if from_state in self._exit_callbacks:
            self._exit_callbacks[from_state](context)
        
        # Call transition callback
        transition_key = (from_state, to_state)
        if transition_key in self._transition_callbacks:
            self._transition_callbacks[transition_key](from_state, to_state, context)
        
        # Change state
        self.current_state = to_state
        
        # Call enter callback for new state
        if to_state in self._enter_callbacks:
            self._enter_callbacks[to_state](context)
        
        return True
    
    # Convenience methods for common transitions
    
    def start_training(self, context: Optional[Dict] = None) -> bool:
        """Start training mode."""
        return self.transition(RLState.TRAINING, context)
    
    def pause(self, context: Optional[Dict] = None) -> bool:
        """Pause current operation."""
        return self.transition(RLState.PAUSED, context)
    
    def resume(self, context: Optional[Dict] = None) -> bool:
        """Resume training from paused state."""
        if self.current_state == RLState.PAUSED:
            return self.transition(RLState.TRAINING, context)
        return False
    
    def start_testing(self, context: Optional[Dict] = None) -> bool:
        """Start testing mode."""
        return self.transition(RLState.TESTING, context)
    
    def converge(self, context: Optional[Dict] = None) -> bool:
        """Mark as converged."""
        return self.transition(RLState.CONVERGED, context)
    
    def reset_to_idle(self, context: Optional[Dict] = None) -> bool:
        """Reset to idle state."""
        return self.transition(RLState.IDLE, context)
    
    def fail_error(self, context: Optional[Dict] = None) -> bool:
        """Transition to error state."""
        return self.transition(RLState.ERROR, context)
    
    # State checking methods
    
    def is_idle(self) -> bool:
        """Check if in idle state."""
        return self.current_state == RLState.IDLE
    
    def is_training(self) -> bool:
        """Check if in training state."""
        return self.current_state == RLState.TRAINING
    
    def is_paused(self) -> bool:
        """Check if in paused state."""
        return self.current_state == RLState.PAUSED
    
    def is_testing(self) -> bool:
        """Check if in testing state."""
        return self.current_state == RLState.TESTING
    
    def is_converged(self) -> bool:
        """Check if in converged state."""
        return self.current_state == RLState.CONVERGED
    
    def is_error(self) -> bool:
        """Check if in error state."""
        return self.current_state == RLState.ERROR
    
    def is_active(self) -> bool:
        """Check if algorithm is actively running."""
        return self.current_state in {RLState.TRAINING, RLState.TESTING}
    
    def can_start(self) -> bool:
        """Check if can start training or testing."""
        return self.current_state in {RLState.IDLE, RLState.CONVERGED}
    
    def get_state_description(self) -> str:
        """Get human-readable state description."""
        descriptions = {
            RLState.IDLE: "Ready - Click Train to start learning",
            RLState.TRAINING: "Training agent with Q-Learning",
            RLState.PAUSED: "Training paused",
            RLState.TESTING: "Testing learned policy",
            RLState.CONVERGED: "Training converged - agent learned optimal policy",
            RLState.ERROR: "Error occurred during execution",
        }
        return descriptions.get(self.current_state, "Unknown state")