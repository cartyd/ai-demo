"""Q-Learning algorithm implementation for pathfinding."""

import numpy as np
from typing import Optional, List
from .types import (
    Coord, Grid, RLConfig, ActionInt, QValues, Episode, TrainingResult, 
    PathfindingResult, ACTION_DELTAS, TrainingPhase
)


class QLearningEnvironment:
    """Environment for Q-Learning pathfinding."""
    
    def __init__(self, grid: Grid, start: Coord, target: Coord, config: RLConfig):
        self.grid = grid
        self.start = start
        self.target = target
        self.config = config
        self.current_pos = start
        self.steps_taken = 0
        self.total_reward = 0.0
        
    def reset(self) -> Coord:
        """Reset environment to initial state."""
        self.current_pos = self.start
        self.steps_taken = 0
        self.total_reward = 0.0
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
        
        # Calculate next position
        delta = ACTION_DELTAS[action]
        next_pos = (self.current_pos[0] + delta[0], self.current_pos[1] + delta[1])
        
        # Check if next position is valid
        if not self.grid.is_valid_coord(next_pos):
            # Hit boundary - stay in place, negative reward
            reward = self.config.reward_wall
            done = False
        else:
            next_node = self.grid.get_node(next_pos)
            if not next_node or not next_node.is_passable():
                # Hit wall - stay in place, negative reward
                reward = self.config.reward_wall
                done = False
            else:
                # Valid move
                self.current_pos = next_pos
                
                # Check if reached goal
                if next_pos == self.target:
                    reward = self.config.reward_goal
                    done = True
                else:
                    reward = self.config.reward_step
                    done = False
        
        # Check if max steps reached
        if self.steps_taken >= self.config.max_steps_per_episode:
            done = True
            
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


class QLearningAgent:
    """Q-Learning agent for pathfinding."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.epsilon = config.epsilon
        self.training_phase: TrainingPhase = "training"
        self.episodes_completed = 0
        self.training_history: List[Episode] = []
        
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
        """Select action using epsilon-greedy policy."""
        if not valid_actions:
            return 0
        
        if self.training_phase == "training" and np.random.random() < self.epsilon:
            # Explore: choose random valid action
            return np.random.choice(valid_actions)
        else:
            # Exploit: choose best action
            return self.get_best_action(grid, state, valid_actions)
    
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
    
    def train_episode(self, env: QLearningEnvironment) -> Episode:
        """Train for one episode."""
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        epsilon_used = self.epsilon
        
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
        
        # Create episode record
        episode = Episode(
            number=self.episodes_completed,
            steps=episode_steps,
            total_reward=episode_reward,
            reached_goal=(state == env.target),
            epsilon_used=epsilon_used
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