"""Training checkpoint management for RL pathfinding."""

import json
import os
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from pathlib import Path

from ..domain.types import (
    TrainingCheckpoint, CheckpointMetadata, Grid, Coord, Episode, 
    RLConfig, QValues
)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.maze_serialization import MazeData, save_maze, load_maze, extract_maze_from_grid


class CheckpointManager:
    """Manages training checkpoints with maze correlation."""
    
    def __init__(self, checkpoints_dir: str = "training_checkpoints"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Ensure saved_mazes directory exists
        self.mazes_dir = Path("saved_mazes")
        self.mazes_dir.mkdir(exist_ok=True)
    
    def generate_maze_hash(self, grid: Grid, start: Coord, target: Coord) -> str:
        """Generate a unique hash for maze structure."""
        # Create a string representation of the maze structure
        maze_data = {
            "width": grid.width,
            "height": grid.height,
            "start": start,
            "target": target,
            "walls": []
        }
        
        # Add wall positions
        for coord_str, node in grid.nodes.items():
            if node.state == "wall":
                maze_data["walls"].append(node.coord)
        
        # Sort walls for consistent hashing
        maze_data["walls"].sort()
        
        # Generate hash
        maze_str = json.dumps(maze_data, sort_keys=True)
        return hashlib.md5(maze_str.encode()).hexdigest()
    
    def ensure_maze_saved(self, grid: Grid, start: Coord, target: Coord, 
                         maze_name: Optional[str] = None) -> str:
        """Ensure the maze is saved and return the file path."""
        maze_hash = self.generate_maze_hash(grid, start, target)
        
        # Check if maze already exists by hash
        existing_maze = self.find_maze_by_hash(maze_hash)
        if existing_maze:
            return existing_maze
        
        # Generate maze name if not provided
        if not maze_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            maze_name = f"Training_Maze_{grid.width}x{grid.height}_{timestamp}"
        
        # Create maze data and save
        maze_data = extract_maze_from_grid(grid, start, target)
        maze_data.name = maze_name
        
        from utils.maze_serialization import generate_maze_filename
        maze_file_path = generate_maze_filename(maze_data)
        
        if save_maze(maze_data, maze_file_path):
            return maze_file_path
        else:
            raise Exception("Failed to save maze")
    
    def find_maze_by_hash(self, maze_hash: str) -> Optional[str]:
        """Find existing maze file by hash."""
        for maze_file in self.mazes_dir.glob("*.json"):
            try:
                with open(maze_file, 'r') as f:
                    maze_data = json.load(f)
                    if maze_data.get("maze_hash") == maze_hash:
                        return str(maze_file)
            except (json.JSONDecodeError, KeyError):
                continue
        return None
    
    def create_checkpoint(self, checkpoint_id: str, maze_name: str, grid: Grid, 
                         start: Coord, target: Coord, agent_state: Dict,
                         config: RLConfig, episode_number: int, 
                         total_episodes: int) -> str:
        """Create a training checkpoint."""
        # Ensure maze is saved
        maze_file_path = self.ensure_maze_saved(grid, start, target, maze_name)
        maze_hash = self.generate_maze_hash(grid, start, target)
        
        # Serialize Q-table
        q_table = {}
        for coord_str, node in grid.nodes.items():
            if any(abs(q_val) > 0.01 for q_val in node.q_values.as_array()):
                q_table[coord_str] = {
                    "up": node.q_values.up,
                    "down": node.q_values.down,
                    "left": node.q_values.left,
                    "right": node.q_values.right
                }
        
        # Calculate statistics
        training_history = agent_state.get("training_history", [])
        # Convert Episode objects to dicts if needed
        if training_history and hasattr(training_history[0], 'reached_goal'):
            training_history = [
                {
                    "number": ep.number,
                    "steps": ep.steps,
                    "total_reward": ep.total_reward,
                    "reached_goal": ep.reached_goal,
                    "epsilon_used": ep.epsilon_used,
                    "elapsed_time": getattr(ep, 'elapsed_time', 0.0)
                } for ep in training_history
            ]
        successful_episodes = sum(1 for ep in training_history if ep.get('reached_goal', False))
        success_rate = successful_episodes / len(training_history) if training_history else 0.0
        
        # Create checkpoint
        checkpoint = TrainingCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            maze_name=maze_name,
            maze_file_path=maze_file_path,
            maze_hash=maze_hash,
            episode_number=episode_number,
            total_episodes=total_episodes,
            episodes_completed=len(training_history),
            success_rate=success_rate,
            epsilon=agent_state.get("epsilon", config.epsilon),
            agent_state=agent_state,
            q_table=q_table,
            training_history=training_history,
            config=config,
            grid_info={
                "width": grid.width,
                "height": grid.height,
                "start": start,
                "target": target
            }
        )
        
        # Save checkpoint
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
        with open(checkpoint_file, 'w') as f:
            # Convert checkpoint to dict for JSON serialization
            checkpoint_dict = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "timestamp": checkpoint.timestamp,
                "maze_name": checkpoint.maze_name,
                "maze_file_path": checkpoint.maze_file_path,
                "maze_hash": checkpoint.maze_hash,
                "episode_number": checkpoint.episode_number,
                "total_episodes": checkpoint.total_episodes,
                "episodes_completed": checkpoint.episodes_completed,
                "success_rate": checkpoint.success_rate,
                "epsilon": checkpoint.epsilon,
                "agent_state": checkpoint.agent_state,
                "q_table": checkpoint.q_table,
                "training_history": training_history,
                "config": {
                    "learning_rate": checkpoint.config.learning_rate,
                    "discount_factor": checkpoint.config.discount_factor,
                    "epsilon": checkpoint.config.epsilon,
                    "epsilon_decay": checkpoint.config.epsilon_decay,
                    "epsilon_min": checkpoint.config.epsilon_min,
                    "max_episodes": checkpoint.config.max_episodes,
                    "max_steps_per_episode": checkpoint.config.max_steps_per_episode,
                    "reward_goal": checkpoint.config.reward_goal,
                    "reward_wall": checkpoint.config.reward_wall,
                    "reward_step": checkpoint.config.reward_step,
                    "use_smart_rewards": checkpoint.config.use_smart_rewards,
                    "reward_progress": checkpoint.config.reward_progress,
                    "reward_exploration": checkpoint.config.reward_exploration,
                    "reward_dead_end": checkpoint.config.reward_dead_end,
                    "reward_revisit": checkpoint.config.reward_revisit,
                    "reward_stuck": checkpoint.config.reward_stuck,
                    "reward_backward": checkpoint.config.reward_backward,
                    "use_distance_guidance": checkpoint.config.use_distance_guidance,
                    "detect_dead_ends": checkpoint.config.detect_dead_ends,
                    "stuck_threshold": checkpoint.config.stuck_threshold
                },
                "grid_info": checkpoint.grid_info
            }
            
            json.dump(checkpoint_dict, f, indent=2)
        
        return str(checkpoint_file)
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[TrainingCheckpoint]:
        """Load a training checkpoint."""
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct training history
            training_history = []
            for ep_data in data.get("training_history", []):
                episode = Episode(
                    number=ep_data["number"],
                    steps=ep_data["steps"],
                    total_reward=ep_data["total_reward"],
                    reached_goal=ep_data["reached_goal"],
                    epsilon_used=ep_data["epsilon_used"],
                    elapsed_time=ep_data.get("elapsed_time", 0.0)
                )
                training_history.append(episode)
            
            # Reconstruct config
            config_data = data["config"]
            config = RLConfig(
                learning_rate=config_data["learning_rate"],
                discount_factor=config_data["discount_factor"],
                epsilon=config_data["epsilon"],
                epsilon_decay=config_data["epsilon_decay"],
                epsilon_min=config_data["epsilon_min"],
                max_episodes=config_data["max_episodes"],
                max_steps_per_episode=config_data["max_steps_per_episode"],
                reward_goal=config_data["reward_goal"],
                reward_wall=config_data["reward_wall"],
                reward_step=config_data["reward_step"],
                use_smart_rewards=config_data.get("use_smart_rewards", True),
                reward_progress=config_data.get("reward_progress", 5.0),
                reward_exploration=config_data.get("reward_exploration", 1.0),
                reward_dead_end=config_data.get("reward_dead_end", -15.0),
                reward_revisit=config_data.get("reward_revisit", -2.0),
                reward_stuck=config_data.get("reward_stuck", -10.0),
                reward_backward=config_data.get("reward_backward", -1.0),
                use_distance_guidance=config_data.get("use_distance_guidance", True),
                detect_dead_ends=config_data.get("detect_dead_ends", True),
                stuck_threshold=config_data.get("stuck_threshold", 5)
            )
            
            # Create checkpoint
            checkpoint = TrainingCheckpoint(
                checkpoint_id=data["checkpoint_id"],
                timestamp=data["timestamp"],
                maze_name=data["maze_name"],
                maze_file_path=data["maze_file_path"],
                maze_hash=data["maze_hash"],
                episode_number=data["episode_number"],
                total_episodes=data["total_episodes"],
                episodes_completed=data["episodes_completed"],
                success_rate=data["success_rate"],
                epsilon=data["epsilon"],
                agent_state=data["agent_state"],
                q_table=data["q_table"],
                training_history=training_history,
                config=config,
                grid_info=data["grid_info"]
            )
            
            return checkpoint
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading checkpoint {checkpoint_id}: {e}")
            return None
    
    def get_checkpoints_for_maze(self, maze_hash: str) -> List[CheckpointMetadata]:
        """Get all checkpoints for a specific maze."""
        checkpoints = []
        
        for checkpoint_file in self.checkpoints_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                if data.get("maze_hash") == maze_hash:
                    metadata = CheckpointMetadata(
                        checkpoint_id=data["checkpoint_id"],
                        maze_name=data["maze_name"],
                        maze_hash=data["maze_hash"],
                        timestamp=data["timestamp"],
                        episode_number=data["episode_number"],
                        success_rate=data["success_rate"],
                        completion_percentage=(data["episodes_completed"] / data["total_episodes"]) * 100,
                        file_path=str(checkpoint_file),
                        file_size=checkpoint_file.stat().st_size,
                        is_compatible=True
                    )
                    checkpoints.append(metadata)
                    
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Sort by episode number
        checkpoints.sort(key=lambda x: x.episode_number)
        return checkpoints
    
    def apply_checkpoint_to_grid(self, checkpoint: TrainingCheckpoint, grid: Grid) -> bool:
        """Apply checkpoint Q-values to the grid."""
        try:
            # Clear existing Q-values
            grid.reset_q_values()
            
            # Apply Q-values from checkpoint
            for coord_str, q_values_dict in checkpoint.q_table.items():
                node = grid.nodes.get(coord_str)
                if node:
                    node.q_values = QValues(
                        up=q_values_dict.get("up", 0.0),
                        down=q_values_dict.get("down", 0.0),
                        left=q_values_dict.get("left", 0.0),
                        right=q_values_dict.get("right", 0.0)
                    )
            
            return True
            
        except Exception as e:
            print(f"Error applying checkpoint to grid: {e}")
            return False
    
    def validate_checkpoint_compatibility(self, checkpoint: TrainingCheckpoint, 
                                        grid: Grid, start: Coord, target: Coord) -> bool:
        """Validate that a checkpoint is compatible with the current maze."""
        current_hash = self.generate_maze_hash(grid, start, target)
        return current_hash == checkpoint.maze_hash
    
    def cleanup_old_checkpoints(self, keep_count: int = 10):
        """Clean up old checkpoints, keeping only the most recent ones per maze."""
        maze_checkpoints = {}
        
        # Group checkpoints by maze
        for checkpoint_file in self.checkpoints_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                maze_hash = data.get("maze_hash")
                if maze_hash:
                    if maze_hash not in maze_checkpoints:
                        maze_checkpoints[maze_hash] = []
                    
                    maze_checkpoints[maze_hash].append({
                        "file": checkpoint_file,
                        "episode": data["episode_number"],
                        "timestamp": data["timestamp"]
                    })
                    
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Keep only the most recent checkpoints for each maze
        for maze_hash, checkpoints in maze_checkpoints.items():
            checkpoints.sort(key=lambda x: x["episode"], reverse=True)
            
            # Remove old checkpoints
            for checkpoint in checkpoints[keep_count:]:
                try:
                    checkpoint["file"].unlink()
                except OSError:
                    pass


# Global instance
checkpoint_manager = CheckpointManager()