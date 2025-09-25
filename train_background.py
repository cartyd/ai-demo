#!/usr/bin/env python3
"""
Background training script for RL pathfinding with checkpoint saving.
This script trains an agent in the background and saves checkpoints at regular intervals.
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from rl.domain.qlearning import QLearningAgent
from rl.domain.types import RLConfig
from rl.utils.grid_factory import create_empty_grid, generate_maze_grid, place_start_and_target
from rl.utils.checkpoint_manager import checkpoint_manager
from utils.maze_serialization import load_maze, apply_maze_to_grid


def create_checkpoint_callback(maze_name: str, grid, start, target, config):
    """Create a callback function for saving checkpoints."""
    def save_checkpoint(episode_number: int, agent_state: dict):
        try:
            checkpoint_id = f"{maze_name}_ep{episode_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checkpoint_path = checkpoint_manager.create_checkpoint(
                checkpoint_id=checkpoint_id,
                maze_name=maze_name,
                grid=grid,
                start=start,
                target=target,
                agent_state=agent_state,
                config=config,
                episode_number=episode_number,
                total_episodes=config.max_episodes
            )
            print(f"✅ Checkpoint saved: {checkpoint_id} -> {checkpoint_path}")
        except Exception as e:
            print(f"❌ Failed to save checkpoint at episode {episode_number}: {e}")
    
    return save_checkpoint


def load_maze_from_file(maze_file: str):
    """Load maze from saved file."""
    try:
        maze_data = load_maze(maze_file)
        if not maze_data:
            raise ValueError(f"Could not load maze from {maze_file}")
        
        # Create grid and apply maze
        grid = create_empty_grid(maze_data.width, maze_data.height)
        apply_maze_to_grid(maze_data, grid)
        
        return grid, maze_data.start, maze_data.target, maze_data.name
    except Exception as e:
        print(f"Error loading maze: {e}")
        return None, None, None, None


def main():
    parser = argparse.ArgumentParser(description="Background RL training with checkpoints")
    parser.add_argument("--maze", type=str, help="Path to saved maze file")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to train")
    parser.add_argument("--checkpoint-interval", type=int, default=200, help="Episodes between checkpoints")
    parser.add_argument("--maze-size", type=int, default=15, help="Size for generated maze (if no maze file)")
    parser.add_argument("--load-checkpoint", type=str, help="Checkpoint ID to continue from")
    
    args = parser.parse_args()
    
    print("🧠 RL Background Training with Checkpoints")
    print("=" * 50)
    
    # Load or generate maze
    if args.maze:
        print(f"📁 Loading maze from: {args.maze}")
        grid, start, target, maze_name = load_maze_from_file(args.maze)
        if grid is None:
            print("❌ Failed to load maze. Exiting.")
            return 1
    else:
        print(f"🎲 Generating new maze: {args.maze_size}x{args.maze_size}")
        grid, start, target = generate_maze_grid(args.maze_size, args.maze_size)
        maze_name = f"Generated_Maze_{args.maze_size}x{args.maze_size}"
    
    print(f"🏷️  Maze: {maze_name}")
    print(f"📐 Grid: {grid.width}x{grid.height}")
    print(f"🎯 Start: {start} → Target: {target}")
    
    # Create agent with optimized config for background training
    config = RLConfig(
        learning_rate=0.15,
        discount_factor=0.98,
        epsilon=0.4,
        epsilon_decay=0.995,
        epsilon_min=0.03,
        max_episodes=args.episodes,
        max_steps_per_episode=500,
        training_mode="background",
        use_smart_rewards=True,
        enable_early_stopping=True,
        early_stop_patience=200,
        min_improvement_threshold=0.02
    )
    
    agent = QLearningAgent(config)
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"📂 Loading checkpoint: {args.load_checkpoint}")
        checkpoint = checkpoint_manager.load_checkpoint(args.load_checkpoint)
        if checkpoint:
            # Validate compatibility
            if checkpoint_manager.validate_checkpoint_compatibility(checkpoint, grid, start, target):
                # Apply checkpoint to agent and grid
                agent.load_agent_state(checkpoint.agent_state)
                checkpoint_manager.apply_checkpoint_to_grid(checkpoint, grid)
                print(f"✅ Checkpoint loaded: Episode {checkpoint.episode_number}, Success rate: {checkpoint.success_rate:.1%}")
            else:
                print("❌ Checkpoint is not compatible with current maze. Starting fresh.")
        else:
            print("❌ Failed to load checkpoint. Starting fresh.")
    
    # Create checkpoint callback
    checkpoint_callback = create_checkpoint_callback(maze_name, grid, start, target, config)
    
    # Display training configuration
    print(f"\n⚙️  Training Configuration:")
    print(f"   Episodes: {args.episodes}")
    print(f"   Checkpoint interval: {args.checkpoint_interval}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Epsilon: {config.epsilon} → {config.epsilon_min}")
    print(f"   Starting from episode: {agent.episodes_completed}")
    
    # Start training
    print(f"\n🚀 Starting background training...")
    try:
        result = agent.train_background_with_checkpoints(
            grid=grid,
            start=start,
            target=target,
            episodes=args.episodes,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_callback=checkpoint_callback
        )
        
        # Display results
        print(f"\n🎉 Training completed!")
        print(f"   Total episodes: {result.total_episodes}")
        print(f"   Successful episodes: {result.successful_episodes}")
        print(f"   Success rate: {result.success_rate:.1%}")
        print(f"   Average reward: {result.average_reward:.2f}")
        print(f"   Final epsilon: {result.final_epsilon:.3f}")
        print(f"   Converged: {result.converged}")
        
        if result.early_stopped:
            print(f"   Early stopped: {result.stopping_reason}")
        
        # Test final policy
        print(f"\n🧪 Testing final policy...")
        path_result = agent.find_path(grid, start, target)
        if path_result.success:
            print(f"✅ Path found! Length: {path_result.path_length}")
        else:
            print(f"❌ No path found with final policy")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())