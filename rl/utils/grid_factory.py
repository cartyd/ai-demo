"""Grid factory for creating, resetting, and randomizing grids for RL pathfinding."""

from typing import Optional, Tuple, List
from ..domain.types import Grid, GridNode, Coord, QValues
from .rng import SeededRNG, default_rng


def create_empty_grid(width: int, height: int) -> Grid:
    """
    Create a new empty grid with the specified dimensions.
    
    Args:
        width: Grid width (must be > 0)
        height: Grid height (must be > 0)
        
    Returns:
        New Grid instance with all empty nodes
        
    Raises:
        ValueError: If width or height <= 0
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Grid dimensions must be positive, got {width}x{height}")
    
    nodes = {}
    
    for y in range(height):
        for x in range(width):
            coord = (x, y)
            node_id = f"{x},{y}"
            
            node = GridNode(
                id=node_id,
                coord=coord,
                q_values=QValues(),
                walkable=True,
                weight=1.0,
                state="empty",
                visit_count=0,
                last_reward=0.0
            )
            
            nodes[node_id] = node
    
    return Grid(width=width, height=height, nodes=nodes)


def reset_grid(grid: Grid, preserve_walls: bool = False) -> None:
    """
    Reset a grid to empty state.
    
    Args:
        grid: Grid to reset
        preserve_walls: If True, keep wall positions
    """
    for node in grid.nodes.values():
        if preserve_walls and node.state == "wall":
            continue
            
        node.state = "empty"
        node.walkable = True
        node.weight = 1.0
        node.reset_rl_data()


def add_random_walls(grid: Grid, density: float, rng: Optional[SeededRNG] = None) -> None:
    """
    Add random walls to the grid.
    
    Args:
        grid: Grid to modify
        density: Wall density (0.0 to 1.0, where 1.0 = all walls)
        rng: Random number generator to use (uses default if None)
    """
    if not (0.0 <= density <= 1.0):
        raise ValueError(f"Density must be between 0.0 and 1.0, got {density}")
    
    if rng is None:
        rng = default_rng
    
    # Calculate number of walls to place
    total_nodes = len(grid.nodes)
    num_walls = int(total_nodes * density)
    
    # Get all empty node coordinates
    empty_coords = []
    for node in grid.nodes.values():
        if node.state == "empty":
            empty_coords.append(node.coord)
    
    # Randomly select coordinates for walls
    if num_walls > len(empty_coords):
        num_walls = len(empty_coords)
    
    wall_coords = rng.sample(empty_coords, num_walls)
    
    # Place walls
    for coord in wall_coords:
        grid.set_node_state(coord, "wall")
        node = grid.get_node(coord)
        if node:
            node.walkable = False


def place_start_and_target(grid: Grid, start: Optional[Coord] = None, 
                          target: Optional[Coord] = None, 
                          rng: Optional[SeededRNG] = None) -> Tuple[Coord, Coord]:
    """
    Place start and target positions on the grid.
    
    Args:
        grid: Grid to modify
        start: Specific start coordinate (random if None)
        target: Specific target coordinate (random if None)
        rng: Random number generator to use
        
    Returns:
        Tuple of (start_coord, target_coord)
        
    Raises:
        ValueError: If no valid positions available or positions overlap
    """
    if rng is None:
        rng = default_rng
    
    # Get all passable coordinates
    passable_coords = []
    for node in grid.nodes.values():
        if node.is_passable() and node.state == "empty":
            passable_coords.append(node.coord)
    
    if len(passable_coords) < 2:
        raise ValueError("Not enough passable positions for start and target")
    
    # Select start position
    if start is None:
        start = rng.choice(passable_coords)
    elif not grid.is_valid_coord(start):
        raise ValueError(f"Start position {start} is out of bounds")
    elif not grid.get_node(start) or not grid.get_node(start).is_passable():
        raise ValueError(f"Start position {start} is not passable")
    
    # Select target position (ensure it's different from start)
    available_for_target = [coord for coord in passable_coords if coord != start]
    if not available_for_target:
        raise ValueError("No valid target positions available")
    
    if target is None:
        target = rng.choice(available_for_target)
    elif not grid.is_valid_coord(target):
        raise ValueError(f"Target position {target} is out of bounds")
    elif target == start:
        raise ValueError("Start and target positions cannot be the same")
    elif not grid.get_node(target) or not grid.get_node(target).is_passable():
        raise ValueError(f"Target position {target} is not passable")
    
    # Place start and target
    grid.set_node_state(start, "start")
    grid.set_node_state(target, "target")
    
    return start, target


def create_preset_grid(width: int, height: int, preset: str, 
                      seed: Optional[int] = None) -> Tuple[Grid, Coord, Coord]:
    """
    Create a preset grid configuration.
    
    Args:
        width: Grid width
        height: Grid height
        preset: Preset name ("empty", "maze", "sparse", "dense")
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (grid, start_coord, target_coord)
    """
    rng = SeededRNG(seed)
    grid = create_empty_grid(width, height)
    
    if preset == "empty":
        # No walls
        pass
    elif preset == "sparse":
        # 15% walls
        add_random_walls(grid, 0.15, rng)
    elif preset == "dense":
        # 35% walls
        add_random_walls(grid, 0.35, rng)
    elif preset == "maze":
        # Create a maze-like pattern with 45% walls
        add_random_walls(grid, 0.45, rng)
    else:
        raise ValueError(f"Unknown preset: {preset}")
    
    # Place start and target
    start, target = place_start_and_target(grid, rng=rng)
    
    return grid, start, target


def generate_solvable_grid(width: int, height: int, wall_density: float = 0.3, 
                          seed: Optional[int] = None) -> Tuple[Grid, Coord, Coord]:
    """
    Generate a grid that is guaranteed to be solvable.
    
    Args:
        width: Grid width
        height: Grid height
        wall_density: Density of walls to place
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (grid, start_coord, target_coord)
    """
    rng = SeededRNG(seed)
    
    # Create grid and add walls
    grid = create_empty_grid(width, height)
    add_random_walls(grid, wall_density, rng)
    
    # Place start and target
    start, target = place_start_and_target(grid, rng=rng)
    
    return grid, start, target


def generate_maze_grid(width: int, height: int, 
                      seed: Optional[int] = None) -> Tuple[Grid, Coord, Coord]:
    """
    Generate a maze-like grid using recursive backtracking.
    
    Args:
        width: Grid width (should be odd for proper maze generation)
        height: Grid height (should be odd for proper maze generation)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (grid, start_coord, target_coord)
    """
    rng = SeededRNG(seed)
    
    # Make dimensions odd for proper maze generation
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1
    
    # Create grid filled with walls
    grid = create_empty_grid(width, height)
    for node in grid.nodes.values():
        node.state = "wall"
        node.walkable = False
    
    # Recursive backtracking maze generation
    def carve_passages(x: int, y: int, visited: set):
        visited.add((x, y))
        grid.set_node_state((x, y), "empty")
        if grid.get_node((x, y)):
            grid.get_node((x, y)).walkable = True
        
        # Randomize directions
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        rng.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < width and 0 <= ny < height and 
                (nx, ny) not in visited):
                # Carve wall between current and next cell
                wall_x, wall_y = x + dx // 2, y + dy // 2
                grid.set_node_state((wall_x, wall_y), "empty")
                if grid.get_node((wall_x, wall_y)):
                    grid.get_node((wall_x, wall_y)).walkable = True
                
                carve_passages(nx, ny, visited)
    
    # Start maze generation from top-left corner (1, 1)
    visited = set()
    carve_passages(1, 1, visited)
    
    # Place start and target in empty spaces
    start, target = place_start_and_target(grid, rng=rng)
    
    return grid, start, target