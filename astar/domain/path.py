"""Path reconstruction and direction utilities for A* pathfinding."""

from typing import List, Optional
from .types import Coord, Grid
from .neighbors import get_direction_vector


def reconstruct_path(target_coord: Coord, grid: Grid) -> List[Coord]:
    """
    Reconstruct the path from target back to start using parent pointers.
    Returns the path from start to target (reversed from parent chain).
    """
    path = []
    current_coord = target_coord
    
    while current_coord is not None:
        path.append(current_coord)
        
        # Get the current node
        current_node = grid.get_node(current_coord)
        if not current_node or not current_node.parent:
            break
            
        # Find parent coordinate from parent ID
        parent_coord = _coord_from_id(current_node.parent)
        if parent_coord is None:
            break
            
        current_coord = parent_coord
    
    # Reverse to get path from start to target
    return list(reversed(path))


def calculate_path_cost(path: List[Coord], grid: Grid) -> float:
    """Calculate the total cost of a path."""
    if len(path) < 2:
        return 0.0
    
    total_cost = 0.0
    for i in range(1, len(path)):
        from_coord = path[i-1]
        to_coord = path[i]
        
        # Calculate movement cost
        from .neighbors import get_movement_cost
        total_cost += get_movement_cost(from_coord, to_coord, grid)
    
    return total_cost


def get_path_directions(path: List[Coord]) -> List[tuple[int, int]]:
    """
    Get direction vectors for each segment of the path.
    Returns list of (dx, dy) tuples representing movement directions.
    """
    if len(path) < 2:
        return []
    
    directions = []
    for i in range(1, len(path)):
        from_coord = path[i-1]
        to_coord = path[i]
        direction = get_direction_vector(from_coord, to_coord)
        directions.append(direction)
    
    return directions


def smooth_path_directions(directions: List[tuple[int, int]]) -> List[tuple[int, int]]:
    """
    Smooth path directions to avoid rapid direction changes.
    For visualization purposes, helps mouse rotation look more natural.
    """
    if len(directions) <= 1:
        return directions
    
    smoothed = [directions[0]]
    
    for i in range(1, len(directions)):
        prev_dir = smoothed[-1]
        curr_dir = directions[i]
        
        # If directions are opposite, keep the previous direction for smoother rotation
        if (prev_dir[0] == -curr_dir[0] and prev_dir[1] == -curr_dir[1] and 
            prev_dir != (0, 0) and curr_dir != (0, 0)):
            smoothed.append(prev_dir)
        else:
            smoothed.append(curr_dir)
    
    return smoothed


def get_path_waypoints(path: List[Coord], grid: Grid) -> List[tuple[Coord, tuple[int, int]]]:
    """
    Get path waypoints with their movement directions.
    Returns list of (coordinate, direction) tuples.
    Useful for animating the mouse movement.
    """
    if len(path) < 2:
        return [(path[0], (0, 0))] if path else []
    
    directions = get_path_directions(path)
    smoothed_directions = smooth_path_directions(directions)
    
    waypoints = []
    
    # First waypoint uses first direction (or looks toward target if no movement)
    if smoothed_directions:
        waypoints.append((path[0], smoothed_directions[0]))
    else:
        waypoints.append((path[0], (0, 0)))
    
    # Middle waypoints
    for i in range(1, len(path) - 1):
        coord = path[i]
        direction = smoothed_directions[i-1] if i-1 < len(smoothed_directions) else (0, 0)
        waypoints.append((coord, direction))
    
    # Last waypoint
    if len(path) > 1:
        last_direction = smoothed_directions[-1] if smoothed_directions else (0, 0)
        waypoints.append((path[-1], last_direction))
    
    return waypoints


def _coord_from_id(node_id: str) -> Optional[Coord]:
    """Convert node ID back to coordinate."""
    try:
        parts = node_id.split(',')
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except ValueError:
        pass
    return None


def validate_path(path: List[Coord], grid: Grid) -> bool:
    """
    Validate that a path is walkable and connected.
    Returns True if path is valid.
    """
    if not path:
        return False
    
    # Check that all nodes in path are passable
    for coord in path:
        node = grid.get_node(coord)
        if not node:
            return False
        # Allow start and target nodes even if they have special states
        if node.state not in ["start", "target"] and not node.is_passable():
            return False
    
    # Check that path segments are valid moves (adjacent cells)
    for i in range(1, len(path)):
        from_coord = path[i-1]
        to_coord = path[i]
        
        dx = abs(to_coord[0] - from_coord[0])
        dy = abs(to_coord[1] - from_coord[1])
        
        # Valid moves: orthogonal (dx=1,dy=0 or dx=0,dy=1) or diagonal (dx=1,dy=1)
        if not ((dx == 1 and dy == 0) or (dx == 0 and dy == 1) or (dx == 1 and dy == 1)):
            return False
    
    return True