"""Neighbor generation for A* pathfinding with movement rules."""

import math
from typing import List, Tuple
from .types import Coord, Grid, AlgoConfig


def get_neighbors(coord: Coord, grid: Grid, config: AlgoConfig) -> List[Tuple[Coord, float]]:
    """
    Get valid neighbors for a coordinate with their movement costs.
    Returns list of (neighbor_coord, cost) tuples.
    """
    x, y = coord
    neighbors = []
    
    # Define movement directions and their costs
    if config.allow_diagonal:
        # 8-directional movement
        directions = [
            # Orthogonal moves (cost = 1.0)
            ((-1, 0), 1.0), ((1, 0), 1.0), ((0, -1), 1.0), ((0, 1), 1.0),
            # Diagonal moves (cost = √2 ≈ 1.414)
            ((-1, -1), math.sqrt(2)), ((-1, 1), math.sqrt(2)), 
            ((1, -1), math.sqrt(2)), ((1, 1), math.sqrt(2))
        ]
    else:
        # 4-directional movement only
        directions = [
            ((-1, 0), 1.0), ((1, 0), 1.0), ((0, -1), 1.0), ((0, 1), 1.0)
        ]
    
    for (dx, dy), base_cost in directions:
        new_x, new_y = x + dx, y + dy
        new_coord = (new_x, new_y)
        
        # Check bounds
        if not grid.is_valid_coord(new_coord):
            continue
            
        neighbor_node = grid.get_node(new_coord)
        if not neighbor_node or not neighbor_node.is_passable():
            continue
            
        # Check corner cutting for diagonal moves
        if config.allow_diagonal and abs(dx) + abs(dy) == 2:  # Diagonal move
            if not config.corner_cutting and _is_corner_blocked(coord, (dx, dy), grid):
                continue
        
        # Calculate actual cost including node weight
        actual_cost = base_cost * neighbor_node.weight
        neighbors.append((new_coord, actual_cost))
    
    return neighbors


def _is_corner_blocked(coord: Coord, direction: Tuple[int, int], grid: Grid) -> bool:
    """
    Check if a diagonal move is blocked by adjacent walls (corner cutting).
    Returns True if the diagonal move should be blocked.
    """
    x, y = coord
    dx, dy = direction
    
    # Check the two orthogonal neighbors that form the "corner"
    side1 = (x + dx, y)
    side2 = (x, y + dy)
    
    side1_blocked = False
    side2_blocked = False
    
    if grid.is_valid_coord(side1):
        side1_node = grid.get_node(side1)
        side1_blocked = not side1_node or not side1_node.is_passable()
    else:
        side1_blocked = True  # Out of bounds counts as blocked
        
    if grid.is_valid_coord(side2):
        side2_node = grid.get_node(side2)
        side2_blocked = not side2_node or not side2_node.is_passable()
    else:
        side2_blocked = True  # Out of bounds counts as blocked
    
    # Block diagonal if both orthogonal sides are blocked
    return side1_blocked and side2_blocked


def get_movement_cost(from_coord: Coord, to_coord: Coord, grid: Grid) -> float:
    """
    Calculate the movement cost between two adjacent coordinates.
    Includes node weight of the destination.
    """
    dx = abs(to_coord[0] - from_coord[0])
    dy = abs(to_coord[1] - from_coord[1])
    
    # Determine base movement cost
    if dx + dy == 1:  # Orthogonal
        base_cost = 1.0
    elif dx == 1 and dy == 1:  # Diagonal
        base_cost = math.sqrt(2)
    else:
        raise ValueError(f"Invalid movement from {from_coord} to {to_coord}")
    
    # Apply destination node weight
    to_node = grid.get_node(to_coord)
    if to_node:
        return base_cost * to_node.weight
    return base_cost


def get_direction_vector(from_coord: Coord, to_coord: Coord) -> Tuple[int, int]:
    """Get the direction vector between two coordinates."""
    dx = to_coord[0] - from_coord[0]
    dy = to_coord[1] - from_coord[1]
    
    # Normalize to -1, 0, or 1
    if dx != 0:
        dx = 1 if dx > 0 else -1
    if dy != 0:
        dy = 1 if dy > 0 else -1
        
    return (dx, dy)


def get_rotation_angle(direction: Tuple[int, int]) -> float:
    """
    Get rotation angle in degrees for the given direction vector.
    Used for rotating the mouse sprite to face movement direction.
    """
    dx, dy = direction
    
    # Map direction to angle (0° = right, 90° = down, etc.)
    direction_angles = {
        (1, 0): 0,      # Right
        (1, 1): 45,     # Down-Right
        (0, 1): 90,     # Down
        (-1, 1): 135,   # Down-Left
        (-1, 0): 180,   # Left
        (-1, -1): 225,  # Up-Left
        (0, -1): 270,   # Up
        (1, -1): 315,   # Up-Right
    }
    
    return direction_angles.get(direction, 0)