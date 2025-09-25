"""Q-Learning algorithm implementation for pathfinding."""

import numpy as np
import time
from typing import Optional, List, Callable, Dict
from .types import (
    Coord, Grid, RLConfig, ActionInt, QValues, Episode, TrainingResult, 
    PathfindingResult, ACTION_DELTAS, TrainingPhase, TrainingMode
)


class QLearningEnvironment:
    """Environment for Q-Learning pathfinding with smart reward system."""
    
    def __init__(self, grid: Grid, start: Coord, target: Coord, config: RLConfig):
        self.grid = grid
        self.start = start
        self.target = target
        self.config = config
        self.current_pos = start
        self.steps_taken = 0
        self.total_reward = 0.0
        
        # Smart reward tracking
        self.visited_states: set[Coord] = set()
        self.position_history: list[Coord] = []
        self.last_distance_to_goal = self._manhattan_distance(start, target)
        self.steps_without_progress = 0
        
    def reset(self) -> Coord:
        """Reset environment to initial state."""
        self.current_pos = self.start
        self.steps_taken = 0
        self.total_reward = 0.0
        
        # Reset smart reward tracking
        self.visited_states = {self.start}
        self.position_history = [self.start]
        self.last_distance_to_goal = self._manhattan_distance(self.start, self.target)
        self.steps_without_progress = 0
        
        return self.current_pos
    
    def step(self, action: ActionInt) -> tuple[Coord, float, bool]:
        """
        Execute action and return (next_state, reward, done).
        
        Args:
            action: Action to take (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            Tuple of (next_position, reward, episode_done)
        """
        self.steps_taken += 1
        old_pos = self.current_pos
        
        # Calculate next position
        delta = ACTION_DELTAS[action]
        next_pos = (self.current_pos[0] + delta[0], self.current_pos[1] + delta[1])
        
        # Determine base reward and new position
        if not self.grid.is_valid_coord(next_pos):
            # Hit boundary - stay in place, negative reward
            base_reward = self.config.reward_wall
            new_pos = self.current_pos  # Stay in place
            done = False
        else:
            next_node = self.grid.get_node(next_pos)
            if not next_node or not next_node.is_passable():
                # Hit wall - stay in place, negative reward
                base_reward = self.config.reward_wall
                new_pos = self.current_pos  # Stay in place
                done = False
            else:
                # Valid move
                self.current_pos = next_pos
                new_pos = next_pos
                
                # Check if reached goal
                if next_pos == self.target:
                    base_reward = self.config.reward_goal
                    done = True
                else:
                    base_reward = self.config.reward_step
                    done = False
        
        # Check if max steps reached
        if self.steps_taken >= self.config.max_steps_per_episode:
            done = True
        
        # Calculate smart reward
        reward = self._calculate_smart_reward(old_pos, new_pos, base_reward, done)
        
        self.total_reward += reward
        return self.current_pos, reward, done
    
    def get_valid_actions(self, pos: Coord) -> List[ActionInt]:
        """Get list of valid actions from current position."""
        valid_actions = []
        for action in range(4):
            delta = ACTION_DELTAS[action]
            next_pos = (pos[0] + delta[0], pos[1] + delta[1])
            
            if (self.grid.is_valid_coord(next_pos) and 
                self.grid.get_node(next_pos) and 
                self.grid.get_node(next_pos).is_passable()):
                valid_actions.append(action)
        
        return valid_actions
    
    def _manhattan_distance(self, pos1: Coord, pos2: Coord) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _is_dead_end(self, pos: Coord) -> bool:
        """Check if a position is a dead end (only one exit)."""
        if not self.config.detect_dead_ends:
            return False
            
        valid_actions = self.get_valid_actions(pos)
        return len(valid_actions) <= 1  # Dead end or isolated
    
    def _calculate_smart_reward(self, old_pos: Coord, new_pos: Coord, 
                               base_reward: float, done: bool) -> float:
        """Calculate smart reward with multiple components."""
        if not self.config.use_smart_rewards:
            return base_reward
        
        total_reward = base_reward
        
        # 1. Distance-based progress reward
        if self.config.use_distance_guidance and new_pos != old_pos:
            old_dist = self._manhattan_distance(old_pos, self.target)
            new_dist = self._manhattan_distance(new_pos, self.target)
            
            if new_dist < old_dist:
                # Got closer to goal
                progress_reward = self.config.reward_progress
                total_reward += progress_reward
                self.steps_without_progress = 0
            elif new_dist > old_dist:
                # Got further from goal - use stronger backward penalty
                total_reward += self.config.reward_backward
                self.steps_without_progress += 1
            else:
                # Same distance
                self.steps_without_progress += 1
            
            self.last_distance_to_goal = new_dist
        
        # 2. Exploration bonus (for visiting new states)
        if new_pos not in self.visited_states:
            total_reward += self.config.reward_exploration
            self.visited_states.add(new_pos)
        else:
            # Penalty for revisiting states
            total_reward += self.config.reward_revisit
        
        # 3. Dead-end penalty
        if self._is_dead_end(new_pos) and new_pos != self.target:
            total_reward += self.config.reward_dead_end
        
        # 4. Stuck penalty (no progress for too long)
        if self.steps_without_progress >= self.config.stuck_threshold:
            total_reward += self.config.reward_stuck
            # Reset counter to avoid accumulating too much penalty
            self.steps_without_progress = 0
        
        # 5. Update position history (keep last 10 positions for analysis)
        self.position_history.append(new_pos)
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        
        # 6. Enhanced loop detection and immediate backtracking penalty
        if len(self.position_history) >= 2:
            # Immediate backtracking penalty (going back to previous position)
            if self.position_history[-1] == self.position_history[-2]:
                total_reward -= 8.0  # Strong penalty for immediate backtrack
            
        if len(self.position_history) >= 4:
            # 2-step oscillation (A->B->A->B)
            if (self.position_history[-1] == self.position_history[-3] and
                self.position_history[-2] == self.position_history[-4]):
                total_reward -= 10.0  # Strong oscillation penalty
                
        if len(self.position_history) >= 6:
            # 3-step loops (A->B->C->A->B->C)
            if (self.position_history[-1] == self.position_history[-4] and
                self.position_history[-2] == self.position_history[-5] and
                self.position_history[-3] == self.position_history[-6]):
                total_reward -= 12.0  # Very strong loop penalty
                
        # 7. Recent position revisit penalty (visited within last 3 steps)
        if len(self.position_history) >= 4:
            recent_positions = self.position_history[-4:-1]  # Last 3 positions (excluding current)
            if new_pos in recent_positions:
                steps_ago = len(self.position_history) - recent_positions.index(new_pos)
                penalty = max(-6.0, -2.0 * steps_ago)  # Stronger penalty for more recent revisits
                total_reward += penalty
        
        return total_reward


class QLearningAgent:
    """Q-Learning agent for pathfinding."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.epsilon = config.epsilon
        self.training_phase: TrainingPhase = "training"
        self.episodes_completed = 0
        self.training_history: List[Episode] = []
        
        # Visual training state
        self._visual_env: Optional[QLearningEnvironment] = None
        self._visual_state: Optional[Coord] = None
        self._visual_episode_reward = 0.0
        self._visual_episode_steps = 0
        self._visual_epsilon_used = 0.0
        
    def reset(self):
        """Reset the agent."""
        self.epsilon = self.config.epsilon
        self.training_phase = "training"
        self.episodes_completed = 0
        self.training_history.clear()
    
    def get_q_value(self, grid: Grid, state: Coord, action: ActionInt) -> float:
        """Get Q-value for state-action pair."""
        node = grid.get_node(state)
        if not node:
            return 0.0
        return node.q_values.as_array()[action]
    
    def set_q_value(self, grid: Grid, state: Coord, action: ActionInt, value: float):
        """Set Q-value for state-action pair."""
        node = grid.get_node(state)
        if node:
            q_values = node.q_values.as_array()
            q_values[action] = value
            node.q_values.from_array(q_values)
    
    def get_best_action(self, grid: Grid, state: Coord, valid_actions: List[ActionInt]) -> ActionInt:
        """Get best action for state based on Q-values."""
        if not valid_actions:
            return 0  # Default action if no valid actions
        
        node = grid.get_node(state)
        if not node:
            return valid_actions[0]
        
        # Get Q-values for valid actions only
        q_values = node.q_values.as_array()
        valid_q_values = [q_values[action] for action in valid_actions]
        best_idx = np.argmax(valid_q_values)
        return valid_actions[best_idx]
    
    def select_action(self, grid: Grid, state: Coord, valid_actions: List[ActionInt]) -> ActionInt:
        """Select action using smart epsilon-greedy policy with heuristics."""
        if not valid_actions:
            return 0
        
        if self.training_phase == "training" and np.random.random() < self.epsilon:
            # Smart exploration: bias toward goal direction and avoid dead ends
            return self._smart_explore(grid, state, valid_actions)
        else:
            # Smart exploitation: combine Q-values with heuristics
            return self._smart_exploit(grid, state, valid_actions)
    
    def _smart_explore(self, grid: Grid, state: Coord, valid_actions: List[ActionInt]) -> ActionInt:
        """Smart exploration that avoids obvious bad moves and recent positions."""
        if not self.config.use_smart_rewards:
            return np.random.choice(valid_actions)
        
        # Get environment to access position history
        env = getattr(self, '_current_env', None)
        
        # Filter out actions that lead to dead ends or recent positions
        good_actions = []
        recent_positions = []
        if env and hasattr(env, 'position_history') and len(env.position_history) >= 3:
            recent_positions = env.position_history[-3:]  # Last 3 positions
        
        for action in valid_actions:
            delta = ACTION_DELTAS[action]
            next_pos = (state[0] + delta[0], state[1] + delta[1])
            
            # Skip if this would revisit a recent position
            if next_pos in recent_positions:
                continue
            
            # Avoid dead ends during exploration
            env_temp = QLearningEnvironment(grid, state, (0, 0), self.config)
            if not env_temp._is_dead_end(next_pos):
                good_actions.append(action)
        
        # If all actions lead to dead ends, just pick randomly
        if not good_actions:
            good_actions = valid_actions
        
        # Bias exploration toward goal direction
        if len(good_actions) > 1 and hasattr(self, '_target_coord'):
            target = getattr(self, '_target_coord', None)
            if target:
                # Calculate which actions move toward goal
                toward_goal = []
                for action in good_actions:
                    delta = ACTION_DELTAS[action]
                    next_pos = (state[0] + delta[0], state[1] + delta[1])
                    
                    current_dist = abs(state[0] - target[0]) + abs(state[1] - target[1])
                    next_dist = abs(next_pos[0] - target[0]) + abs(next_pos[1] - target[1])
                    
                    if next_dist < current_dist:
                        toward_goal.append(action)
                
                # 85% chance to pick goal-directed action during exploration
                if toward_goal and np.random.random() < 0.85:
                    return np.random.choice(toward_goal)
        
        return np.random.choice(good_actions)
    
    def _smart_exploit(self, grid: Grid, state: Coord, valid_actions: List[ActionInt]) -> ActionInt:
        """Smart exploitation that combines Q-values with heuristics."""
        if not valid_actions:
            return 0
        
        node = grid.get_node(state)
        if not node:
            return valid_actions[0]
        
        # Get Q-values for valid actions
        q_values = node.q_values.as_array()
        
        # Calculate combined scores (Q-value + heuristics)
        action_scores = []
        for action in valid_actions:
            q_score = q_values[action]
            
            # Add distance-based heuristic if enabled
            heuristic_score = 0.0
            if (self.config.use_distance_guidance and 
                hasattr(self, '_target_coord') and 
                getattr(self, '_target_coord', None)):
                
                target = getattr(self, '_target_coord')
                delta = ACTION_DELTAS[action]
                next_pos = (state[0] + delta[0], state[1] + delta[1])
                
                current_dist = abs(state[0] - target[0]) + abs(state[1] - target[1])
                next_dist = abs(next_pos[0] - target[0]) + abs(next_pos[1] - target[1])
                
                if next_dist < current_dist:
                    heuristic_score = 1.0  # Bonus for moving toward goal
                elif next_dist > current_dist:
                    heuristic_score = -0.5  # Small penalty for moving away
            
            # Avoid dead ends during exploitation
            dead_end_penalty = 0.0
            if self.config.detect_dead_ends:
                delta = ACTION_DELTAS[action]
                next_pos = (state[0] + delta[0], state[1] + delta[1])
                env_temp = QLearningEnvironment(grid, state, (0, 0), self.config)
                if env_temp._is_dead_end(next_pos):
                    dead_end_penalty = -5.0
            
            combined_score = q_score + heuristic_score + dead_end_penalty
            action_scores.append((action, combined_score))
        
        # Select action with highest combined score
        best_action = max(action_scores, key=lambda x: x[1])[0]
        return best_action
    
    def update_q_value(self, grid: Grid, state: Coord, action: ActionInt, 
                      reward: float, next_state: Coord, done: bool):
        """Update Q-value using Q-learning update rule."""
        current_q = self.get_q_value(grid, state, action)
        
        if done:
            # Terminal state
            next_q_max = 0.0
        else:
            # Get maximum Q-value for next state
            next_node = grid.get_node(next_state)
            if next_node:
                next_q_max = next_node.q_values.max_value()
            else:
                next_q_max = 0.0
        
        # Q-learning update
        target = reward + self.config.discount_factor * next_q_max
        new_q = current_q + self.config.learning_rate * (target - current_q)
        
        self.set_q_value(grid, state, action, new_q)
    
    def decay_epsilon(self):
        """Decay epsilon for less exploration over time."""
        self.epsilon = max(self.config.epsilon_min, 
                          self.epsilon * self.config.epsilon_decay)
    
    def get_agent_state(self) -> Dict:
        """Get current agent state for checkpointing."""
        # Convert training history to serializable format
        training_history_dict = []
        for ep in self.training_history:
            training_history_dict.append({
                "number": ep.number,
                "steps": ep.steps,
                "total_reward": ep.total_reward,
                "reached_goal": ep.reached_goal,
                "epsilon_used": ep.epsilon_used,
                "elapsed_time": getattr(ep, 'elapsed_time', 0.0)
            })
        
        return {
            "epsilon": self.epsilon,
            "training_phase": self.training_phase,
            "episodes_completed": self.episodes_completed,
            "training_history": training_history_dict,
            "config": {
                "learning_rate": self.config.learning_rate,
                "discount_factor": self.config.discount_factor,
                "epsilon": self.config.epsilon,
                "epsilon_decay": self.config.epsilon_decay,
                "epsilon_min": self.config.epsilon_min,
                "max_episodes": self.config.max_episodes,
                "max_steps_per_episode": self.config.max_steps_per_episode,
                "reward_goal": self.config.reward_goal,
                "reward_wall": self.config.reward_wall,
                "reward_step": self.config.reward_step
            }
        }
    
    def load_agent_state(self, state: Dict):
        """Load agent state from checkpoint."""
        from .types import Episode
        
        self.epsilon = state.get("epsilon", self.config.epsilon)
        self.training_phase = state.get("training_phase", "training")
        self.episodes_completed = state.get("episodes_completed", 0)
        
        # Reconstruct Episode objects from dictionaries
        training_history_data = state.get("training_history", [])
        self.training_history = []
        
        for ep_data in training_history_data:
            if isinstance(ep_data, dict):
                episode = Episode(
                    number=ep_data.get("number", 0),
                    steps=ep_data.get("steps", 0),
                    total_reward=ep_data.get("total_reward", 0.0),
                    reached_goal=ep_data.get("reached_goal", False),
                    epsilon_used=ep_data.get("epsilon_used", 0.0),
                    elapsed_time=ep_data.get("elapsed_time", 0.0)
                )
                self.training_history.append(episode)
            else:
                # Already an Episode object
                self.training_history.append(ep_data)
    
    def train_background_with_checkpoints(self, grid: Grid, start: Coord, target: Coord, 
                                        episodes: Optional[int] = None,
                                        checkpoint_interval: int = 100,
                                        checkpoint_callback: Optional[Callable[[int, Dict], None]] = None) -> TrainingResult:
        """Train in background mode with automatic checkpoint saving."""
        max_episodes = episodes or self.config.max_episodes
        env = QLearningEnvironment(grid, start, target, self.config)
        
        # Store target for smart exploration
        self._target_coord = target
        self._current_env = env
        
        # Reset if starting fresh
        if self.episodes_completed == 0:
            grid.reset_q_values()
        
        self.training_phase = "training"
        episodes_list = list(self.training_history)  # Start with existing history
        successful_episodes = sum(1 for ep in episodes_list if ep.reached_goal)
        
        # Early stopping tracking
        best_success_rate = 0.0
        episodes_without_improvement = 0
        
        print(f"Starting background training from episode {self.episodes_completed}...")
        
        for episode_num in range(self.episodes_completed, max_episodes):
            episode_start_time = time.time()
            episode = self.train_episode(env)
            episode.elapsed_time = time.time() - episode_start_time
            
            episodes_list.append(episode)
            
            if episode.reached_goal:
                successful_episodes += 1
            
            # Update training history
            self.training_history = episodes_list
            self.episodes_completed = episode_num + 1
            
            # Check for checkpointing
            if checkpoint_callback and (episode_num + 1) % checkpoint_interval == 0:
                agent_state = self.get_agent_state()
                checkpoint_callback(episode_num + 1, agent_state)
            
            # Print progress occasionally
            if (episode_num + 1) % 50 == 0:
                recent_episodes = episodes_list[-50:]
                recent_success = sum(1 for ep in recent_episodes if ep.reached_goal)
                recent_success_rate = recent_success / len(recent_episodes)
                print(f"Episode {episode_num + 1}: Success rate: {recent_success_rate:.1%}, Epsilon: {self.epsilon:.3f}")
            
            # Early stopping check
            if self.config.enable_early_stopping and len(episodes_list) >= 100:
                recent_100_episodes = episodes_list[-100:]
                recent_success = sum(1 for ep in recent_100_episodes if ep.reached_goal)
                current_success_rate = recent_success / 100
                
                if current_success_rate > best_success_rate + self.config.min_improvement_threshold:
                    best_success_rate = current_success_rate
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += 1
                
                if episodes_without_improvement >= self.config.early_stop_patience:
                    print(f"Early stopping at episode {episode_num + 1}. No improvement for {episodes_without_improvement} episodes.")
                    break
        
        # Calculate final statistics
        total_reward = sum(ep.total_reward for ep in episodes_list)
        average_reward = total_reward / len(episodes_list) if episodes_list else 0.0
        
        # Check convergence
        converged = False
        if len(episodes_list) >= 100:
            recent_success = sum(1 for ep in episodes_list[-100:] if ep.reached_goal)
            converged = recent_success >= 90
        
        if converged:
            self.training_phase = "converged"
            print(f"Training converged! Success rate: {recent_success}%")
        
        # Final checkpoint
        if checkpoint_callback:
            agent_state = self.get_agent_state()
            checkpoint_callback(self.episodes_completed, agent_state)
        
        result = TrainingResult(
            episodes=episodes_list,
            total_episodes=len(episodes_list),
            successful_episodes=successful_episodes,
            average_reward=average_reward,
            final_epsilon=self.epsilon,
            converged=converged,
            early_stopped=episodes_without_improvement >= self.config.early_stop_patience if self.config.enable_early_stopping else False,
            stopping_reason=f"Early stopped after {episodes_without_improvement} episodes without improvement" if episodes_without_improvement >= self.config.early_stop_patience else "Completed normally"
        )
        
        return result
    
    def train_episode(self, env: QLearningEnvironment) -> Episode:
        """Train for one episode with timing."""
        episode_start_time = time.time()
        
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        epsilon_used = self.epsilon
        
        # Set target and environment for smart heuristics
        self._target_coord = env.target
        self._current_env = env
        
        while True:
            # Select action
            valid_actions = env.get_valid_actions(state)
            action = self.select_action(env.grid, state, valid_actions)
            
            # Take action
            next_state, reward, done = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Update Q-value
            self.update_q_value(env.grid, state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            if done:
                break
        
        # Calculate episode elapsed time
        episode_elapsed_time = time.time() - episode_start_time
        
        # Create episode record
        episode = Episode(
            number=self.episodes_completed,
            steps=episode_steps,
            total_reward=episode_reward,
            reached_goal=(state == env.target),
            epsilon_used=epsilon_used,
            elapsed_time=episode_elapsed_time
        )
        
        self.training_history.append(episode)
        self.episodes_completed += 1
        
        # Decay epsilon
        self.decay_epsilon()
        
        return episode
    
    def train(self, grid: Grid, start: Coord, target: Coord, 
              episodes: Optional[int] = None) -> TrainingResult:
        """Train the agent for specified number of episodes."""
        max_episodes = episodes or self.config.max_episodes
        env = QLearningEnvironment(grid, start, target, self.config)
        
        # Reset grid Q-values
        grid.reset_q_values()
        
        self.training_phase = "training"
        episodes_list = []
        successful_episodes = 0
        
        for episode_num in range(max_episodes):
            episode = self.train_episode(env)
            episodes_list.append(episode)
            
            if episode.reached_goal:
                successful_episodes += 1
        
        # Calculate final statistics
        total_reward = sum(ep.total_reward for ep in episodes_list)
        average_reward = total_reward / len(episodes_list) if episodes_list else 0.0
        
        # Check convergence (simplified: if last 100 episodes have >90% success rate)
        converged = False
        if len(episodes_list) >= 100:
            recent_success = sum(1 for ep in episodes_list[-100:] if ep.reached_goal)
            converged = recent_success >= 90
        
        if converged:
            self.training_phase = "converged"
        
        return TrainingResult(
            episodes=episodes_list,
            total_episodes=len(episodes_list),
            successful_episodes=successful_episodes,
            average_reward=average_reward,
            final_epsilon=self.epsilon,
            converged=converged
        )
    
    def find_path(self, grid: Grid, start: Coord, target: Coord) -> PathfindingResult:
        """Find path using learned Q-values."""
        self.training_phase = "testing"
        env = QLearningEnvironment(grid, start, target, self.config)
        
        # Set target for smart heuristics
        self._target_coord = target
        
        path = [start]
        state = env.reset()
        total_reward = 0.0
        
        # Follow policy to find path
        max_steps = self.config.max_steps_per_episode
        for step in range(max_steps):
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break
                
            # Always choose best action (no exploration)
            action = self.get_best_action(grid, state, valid_actions)
            
            # Take action
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            path.append(next_state)
            state = next_state
            
            if done and state == target:
                # Successfully reached target
                return PathfindingResult(
                    path=path,
                    path_length=len(path),
                    total_reward=total_reward,
                    steps_taken=step + 1,
                    found=True,
                    training_episodes=self.episodes_completed
                )
            elif done:
                # Episode ended without reaching target
                break
        
        # Did not reach target
        return PathfindingResult(
            path=path,
            path_length=len(path),
            total_reward=total_reward,
            steps_taken=len(path) - 1,
            found=False,
            training_episodes=self.episodes_completed
        )
    
    # Visual Training Methods
    
    def start_visual_episode(self, grid: Grid, start: Coord, target: Coord) -> bool:
        """Start a new visual training episode."""
        try:
            self._visual_env = QLearningEnvironment(grid, start, target, self.config)
            self._visual_state = self._visual_env.reset()
            self._visual_episode_reward = 0.0
            self._visual_episode_steps = 0
            self._visual_epsilon_used = self.epsilon
            self._visual_episode_start_time = time.time()  # Track episode start time
            
            # Set target and environment for smart heuristics
            self._target_coord = target
            self._current_env = self._visual_env
            
            # Clear any previous visual states
            grid.reset_visualization_states()
            
            # Mark current position
            if self._visual_state != start and self._visual_state != target:
                grid.set_node_state(self._visual_state, "training_current")
            
            return True
        except Exception:
            return False
    
    def step_visual_training(self) -> tuple[bool, Optional[Episode]]:
        """
        Execute one step of visual training.
        Returns (episode_done, episode_result)
        """
        if not self._visual_env or self._visual_state is None:
            return True, None
        
        # Get valid actions and show considering states
        valid_actions = self._visual_env.get_valid_actions(self._visual_state)
        if not valid_actions:
            return True, None
        
        # Briefly highlight possible moves (for visualization)
        self._highlight_possible_moves(valid_actions)
        
        # Select action
        action = self.select_action(self._visual_env.grid, self._visual_state, valid_actions)
        
        # Take action
        next_state, reward, done = self._visual_env.step(action)
        self._visual_episode_reward += reward
        self._visual_episode_steps += 1
        
        # Update Q-value
        self.update_q_value(self._visual_env.grid, self._visual_state, action, reward, next_state, done)
        
        # Update visual states
        self._update_visual_states(next_state, done)
        
        # Update state
        self._visual_state = next_state
        
        # Check if episode is done
        if done:
            episode = self._finish_visual_episode()
            return True, episode
        
        return False, None
    
    def _highlight_possible_moves(self, valid_actions: List[ActionInt]):
        """Highlight possible next moves for visualization."""
        if not self._visual_env or self._visual_state is None:
            return
        
        for action in valid_actions:
            delta = ACTION_DELTAS[action]
            next_pos = (self._visual_state[0] + delta[0], self._visual_state[1] + delta[1])
            
            if (self._visual_env.grid.is_valid_coord(next_pos) and 
                next_pos != self._visual_env.start and 
                next_pos != self._visual_env.target):
                node = self._visual_env.grid.get_node(next_pos)
                if node and node.is_passable() and node.state not in ["training_current", "training_visited"]:
                    self._visual_env.grid.set_node_state(next_pos, "training_considering")
    
    def _update_visual_states(self, next_state: Coord, done: bool):
        """Update visual states after a step."""
        if not self._visual_env:
            return
        
        # Clear considering states
        for node in self._visual_env.grid.nodes.values():
            if node.state == "training_considering":
                node.state = "empty"
        
        # Mark previous position as visited (unless it's start/target)
        if (self._visual_state != self._visual_env.start and 
            self._visual_state != self._visual_env.target):
            self._visual_env.grid.set_node_state(self._visual_state, "training_visited")
        
        # Mark new position as current (unless it's start/target or episode is done)
        if not done and next_state != self._visual_env.start and next_state != self._visual_env.target:
            self._visual_env.grid.set_node_state(next_state, "training_current")
    
    def _finish_visual_episode(self) -> Episode:
        """Finish the current visual episode and return episode data."""
        # Calculate episode elapsed time
        episode_elapsed_time = time.time() - self._visual_episode_start_time if hasattr(self, '_visual_episode_start_time') else 0.0
        
        # Create episode record
        episode = Episode(
            number=self.episodes_completed,
            steps=self._visual_episode_steps,
            total_reward=self._visual_episode_reward,
            reached_goal=(self._visual_state == self._visual_env.target if self._visual_env else False),
            epsilon_used=self._visual_epsilon_used,
            elapsed_time=episode_elapsed_time
        )
        
        # Add to history
        self.training_history.append(episode)
        self.episodes_completed += 1
        
        # Decay epsilon
        self.decay_epsilon()
        
        # Clear visual states (except start/target)
        if self._visual_env:
            for node in self._visual_env.grid.nodes.values():
                if node.state in ["training_current", "training_visited", "training_considering"]:
                    node.state = "empty"
        
        # Reset visual training state
        self._visual_env = None
        self._visual_state = None
        self._visual_episode_reward = 0.0
        self._visual_episode_steps = 0
        
        return episode
    
    def is_visual_training_active(self) -> bool:
        """Check if visual training is currently active."""
        return self._visual_env is not None and self._visual_state is not None
