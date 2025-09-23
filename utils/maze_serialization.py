"""
Maze serialization utilities for saving and loading mazes.
Supports both A* and RL maze formats with metadata.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class MazeData:
    """Container for maze data with metadata."""
    
    def __init__(self, width: int, height: int, walls: List[Tuple[int, int]], 
                 start: Tuple[int, int], target: Tuple[int, int],
                 name: str = "", description: str = "", generation_method: str = ""):
        self.width = width
        self.height = height
        self.walls = walls
        self.start = start
        self.target = target
        self.name = name
        self.description = description
        self.generation_method = generation_method
        self.created_at = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert maze data to dictionary for serialization."""
        return {
            'width': self.width,
            'height': self.height,
            'walls': self.walls,
            'start': self.start,
            'target': self.target,
            'name': self.name,
            'description': self.description,
            'generation_method': self.generation_method,
            'created_at': self.created_at,
            'version': '1.0'
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MazeData':
        """Create maze data from dictionary."""
        maze = cls(
            width=data['width'],
            height=data['height'],
            walls=data['walls'],
            start=tuple(data['start']),
            target=tuple(data['target']),
            name=data.get('name', ''),
            description=data.get('description', ''),
            generation_method=data.get('generation_method', '')
        )
        maze.created_at = data.get('created_at', datetime.now().isoformat())
        return maze


def extract_maze_from_grid(grid, start_coord: Optional[Tuple[int, int]] = None, 
                          target_coord: Optional[Tuple[int, int]] = None) -> MazeData:
    """Extract maze data from a grid object (works with both A* and RL grids)."""
    walls = []
    
    # Extract wall positions
    for node_id, node in grid.nodes.items():
        if not node.walkable or node.state == "wall":
            walls.append(node.coord)
    
    # Find start and target if not provided
    if start_coord is None or target_coord is None:
        for node in grid.nodes.values():
            if node.state == "start" and start_coord is None:
                start_coord = node.coord
            elif node.state == "target" and target_coord is None:
                target_coord = node.coord
    
    # Default positions if still not found
    if start_coord is None:
        start_coord = (0, 0)
    if target_coord is None:
        target_coord = (grid.width - 1, grid.height - 1)
    
    return MazeData(
        width=grid.width,
        height=grid.height,
        walls=walls,
        start=start_coord,
        target=target_coord
    )


def apply_maze_to_grid(maze_data: MazeData, grid):
    """Apply maze data to a grid object (works with both A* and RL grids)."""
    # First, clear the grid - reset all nodes to empty
    for node in grid.nodes.values():
        node.walkable = True
        node.state = "empty"
        if hasattr(node, 'reset_rl_data'):
            node.reset_rl_data()
    
    # Apply walls
    for wall_coord in maze_data.walls:
        node = grid.get_node(wall_coord)
        if node:
            node.walkable = False
            node.state = "wall"
    
    # Set start position
    start_node = grid.get_node(maze_data.start)
    if start_node:
        start_node.state = "start"
        start_node.walkable = True
    
    # Set target position
    target_node = grid.get_node(maze_data.target)
    if target_node:
        target_node.state = "target"
        target_node.walkable = True


def save_maze(maze_data: MazeData, filepath: str) -> bool:
    """Save maze data to a JSON file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(maze_data.to_dict(), f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving maze: {e}")
        return False


def load_maze(filepath: str) -> Optional[MazeData]:
    """Load maze data from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return MazeData.from_dict(data)
    except Exception as e:
        print(f"Error loading maze: {e}")
        return None


def get_mazes_directory() -> str:
    """Get the default directory for saved mazes."""
    project_dir = Path(__file__).parent.parent
    mazes_dir = project_dir / "saved_mazes"
    mazes_dir.mkdir(exist_ok=True)
    return str(mazes_dir)


def list_saved_mazes() -> List[Tuple[str, MazeData]]:
    """List all saved mazes with their metadata."""
    mazes_dir = get_mazes_directory()
    mazes = []
    
    try:
        for filepath in Path(mazes_dir).glob("*.json"):
            maze_data = load_maze(str(filepath))
            if maze_data:
                mazes.append((str(filepath), maze_data))
    except Exception as e:
        print(f"Error listing mazes: {e}")
    
    return sorted(mazes, key=lambda x: x[1].created_at, reverse=True)


def generate_maze_filename(maze_data: MazeData) -> str:
    """Generate a filename for a maze based on its metadata."""
    # Clean the name for filesystem use
    safe_name = "".join(c for c in maze_data.name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not safe_name:
        safe_name = f"maze_{maze_data.width}x{maze_data.height}"
    
    # Add timestamp to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.json"
    
    return os.path.join(get_mazes_directory(), filename)


def delete_maze(filepath: str) -> bool:
    """Delete a saved maze file."""
    try:
        os.remove(filepath)
        return True
    except Exception as e:
        print(f"Error deleting maze: {e}")
        return False