"""Grid factory for creating, resetting, and randomizing grids."""

from typing import Optional, Tuple, List
from ..domain.types import Grid, GridNode, Coord, Costs
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
                cost=Costs(),
                walkable=True,
                weight=1.0,
                state="empty",
                parent=None
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
        node.reset_costs()


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
        # TODO: Could implement proper maze generation algorithm here
    else:
        raise ValueError(f"Unknown preset: {preset}")
    
    # Place start and target
    start, target = place_start_and_target(grid, rng=rng)
    
    return grid, start, target


def get_passable_neighbors(grid: Grid, coord: Coord) -> List[Coord]:
    """Get all passable neighbor coordinates (8-connected)."""
    x, y = coord
    neighbors = []
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            
            neighbor_coord = (x + dx, y + dy)
            if (grid.is_valid_coord(neighbor_coord) and 
                grid.get_node(neighbor_coord) and
                grid.get_node(neighbor_coord).is_passable()):
                neighbors.append(neighbor_coord)
    
    return neighbors


def ensure_path_exists(grid: Grid, start: Coord, target: Coord, allow_diagonal: bool = False) -> bool:
    """
    Check if a path exists between start and target using flood fill.
    
    Args:
        grid: Grid to check
        start: Start coordinate
        target: Target coordinate
        allow_diagonal: Whether diagonal movement is allowed
        
    Returns:
        True if path exists, False otherwise
    """
    if not grid.is_valid_coord(start) or not grid.is_valid_coord(target):
        return False
    
    start_node = grid.get_node(start)
    target_node = grid.get_node(target)
    
    if (not start_node or not start_node.is_passable() or
        not target_node or not target_node.is_passable()):
        return False
    
    if start == target:
        return True
    
    # Flood fill from start using appropriate movement
    visited = {start}
    queue = [start]
    
    # Define movement directions
    if allow_diagonal:
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-directional
    
    while queue:
        current = queue.pop(0)
        
        if current == target:
            return True
        
        x, y = current
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            
            if (neighbor not in visited and
                grid.is_valid_coord(neighbor)):
                neighbor_node = grid.get_node(neighbor)
                if neighbor_node and neighbor_node.is_passable():
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    return False


def add_contiguous_walls(grid: Grid, density: float, rng: Optional[SeededRNG] = None) -> None:
    """
    Add contiguous wall structures to create more interesting maze-like patterns.
    
    Args:
        grid: Grid to modify
        density: Wall density (0.0 to 1.0)
        rng: Random number generator to use
    """
    if rng is None:
        rng = default_rng
    
    # Calculate number of walls to place
    total_nodes = len(grid.nodes)
    num_walls = int(total_nodes * density)
    walls_placed = 0
    
    # Create contiguous wall structures
    attempts = 0
    max_attempts = num_walls * 3
    
    while walls_placed < num_walls and attempts < max_attempts:
        attempts += 1
        
        # Pick a random starting point
        start_x = rng.randint(0, grid.width - 1)
        start_y = rng.randint(0, grid.height - 1)
        start_coord = (start_x, start_y)
        
        start_node = grid.get_node(start_coord)
        if not start_node or start_node.state != "empty":
            continue
        
        # Choose wall structure type (favor corridors and L-shapes for more maze-like patterns)
        structure_type = rng.choice(["line", "L_shape", "L_shape", "block", "corridor", "corridor"])
        
        if structure_type == "line":
            walls_placed += _create_line_wall(grid, start_coord, rng, num_walls - walls_placed)
        elif structure_type == "L_shape":
            walls_placed += _create_l_shape_wall(grid, start_coord, rng, num_walls - walls_placed)
        elif structure_type == "block":
            walls_placed += _create_block_wall(grid, start_coord, rng, num_walls - walls_placed)
        elif structure_type == "corridor":
            walls_placed += _create_corridor_walls(grid, start_coord, rng, num_walls - walls_placed)


def _create_line_wall(grid: Grid, start: Coord, rng: SeededRNG, max_walls: int) -> int:
    """Create a straight line of walls."""
    direction = rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
    length = min(rng.randint(5, 12), max_walls)  # Longer walls for more obstacles
    
    walls_created = 0
    current = start
    
    for _ in range(length):
        if grid.is_valid_coord(current):
            node = grid.get_node(current)
            if node and node.state == "empty":
                grid.set_node_state(current, "wall")
                node.walkable = False
                walls_created += 1
        
        current = (current[0] + direction[0], current[1] + direction[1])
        if not grid.is_valid_coord(current):
            break
    
    return walls_created


def _create_l_shape_wall(grid: Grid, start: Coord, rng: SeededRNG, max_walls: int) -> int:
    """Create an L-shaped wall structure."""
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    dir1 = rng.choice(directions)
    dir2 = rng.choice([d for d in directions if d != dir1 and d != (-dir1[0], -dir1[1])])
    
    length1 = min(rng.randint(4, 8), max_walls // 2)  # Longer L-shapes
    length2 = min(rng.randint(4, 8), max_walls - length1)
    
    walls_created = 0
    
    # First leg of L
    current = start
    for _ in range(length1):
        if grid.is_valid_coord(current):
            node = grid.get_node(current)
            if node and node.state == "empty":
                grid.set_node_state(current, "wall")
                node.walkable = False
                walls_created += 1
        current = (current[0] + dir1[0], current[1] + dir1[1])
    
    # Second leg of L
    for _ in range(length2):
        if grid.is_valid_coord(current):
            node = grid.get_node(current)
            if node and node.state == "empty":
                grid.set_node_state(current, "wall")
                node.walkable = False
                walls_created += 1
        current = (current[0] + dir2[0], current[1] + dir2[1])
    
    return walls_created


def _create_block_wall(grid: Grid, start: Coord, rng: SeededRNG, max_walls: int) -> int:
    """Create a rectangular block of walls."""
    width = min(rng.randint(2, 4), int(max_walls ** 0.5) + 1)
    height = min(rng.randint(2, 4), max_walls // width)
    
    walls_created = 0
    
    for dy in range(height):
        for dx in range(width):
            coord = (start[0] + dx, start[1] + dy)
            if grid.is_valid_coord(coord):
                node = grid.get_node(coord)
                if node and node.state == "empty":
                    grid.set_node_state(coord, "wall")
                    node.walkable = False
                    walls_created += 1
    
    return walls_created


def _create_corridor_walls(grid: Grid, start: Coord, rng: SeededRNG, max_walls: int) -> int:
    """Create corridor-like walls that form dead ends."""
    # Create a U-shaped or dead-end corridor
    direction = rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
    perpendicular = (direction[1], direction[0]) if direction[0] == 0 else (-direction[1], direction[0])
    
    length = min(rng.randint(6, 12), max_walls // 3)  # Longer corridors
    walls_created = 0
    
    # Create walls on both sides of a corridor
    for i in range(length):
        for side in [-1, 1]:
            wall_coord = (
                start[0] + i * direction[0] + side * perpendicular[0],
                start[1] + i * direction[1] + side * perpendicular[1]
            )
            
            if grid.is_valid_coord(wall_coord):
                node = grid.get_node(wall_coord)
                if node and node.state == "empty":
                    grid.set_node_state(wall_coord, "wall")
                    node.walkable = False
                    walls_created += 1
    
    # Close off the end to create a dead end (most of the time)
    if rng.random() < 0.85:  # 85% chance to create dead end
        end_coord = (
            start[0] + length * direction[0],
            start[1] + length * direction[1]
        )
        if grid.is_valid_coord(end_coord):
            node = grid.get_node(end_coord)
            if node and node.state == "empty":
                grid.set_node_state(end_coord, "wall")
                node.walkable = False
                walls_created += 1
    
    return walls_created


def generate_maze_grid(width: int, height: int, seed: Optional[int] = None) -> Tuple[Grid, Coord, Coord]:
    """
    Generate a proper maze using recursive backtracking algorithm.
    Creates traditional maze-like patterns with guaranteed solvability.
    
    Args:
        width: Grid width (minimum 7, should be odd for best results)
        height: Grid height (minimum 7, should be odd for best results)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (grid, start_coord, target_coord)
        
    Raises:
        ValueError: If width or height is less than 7
    """
    if width < 7 or height < 7:
        raise ValueError(f"Maze dimensions must be at least 7x7 for proper maze generation, got {width}x{height}")
    
    rng = SeededRNG(seed)
    
    # Ensure odd dimensions for proper maze generation
    maze_width = width if width % 2 == 1 else width - 1
    maze_height = height if height % 2 == 1 else height - 1
    
    # Create grid filled with walls
    grid = create_empty_grid(maze_width, maze_height)
    
    # Fill entire grid with walls initially
    for node in grid.nodes.values():
        node.state = "wall"
        node.walkable = False
    
    # Generate maze using recursive backtracking
    _generate_maze_recursive(grid, rng)
    
    # Place start and target in open areas
    start, target = _place_start_target_in_maze(grid, rng)
    
    return grid, start, target


def _generate_maze_recursive(grid: Grid, rng: SeededRNG) -> None:
    """
    Generate maze using recursive backtracking algorithm.
    This creates the classic maze structure with guaranteed connectivity.
    """
    # Start from position (1,1) to ensure walls around edges
    start_x, start_y = 1, 1
    
    # Mark starting cell as passage
    start_coord = (start_x, start_y)
    grid.set_node_state(start_coord, "empty")
    node = grid.get_node(start_coord)
    if node:
        node.walkable = True
    
    # Stack for backtracking
    stack = [start_coord]
    visited = {start_coord}
    
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # Move by 2 to maintain wall structure
    
    while stack:
        current = stack[-1]
        
        # Get unvisited neighbors
        neighbors = []
        for dx, dy in directions:
            next_x = current[0] + dx
            next_y = current[1] + dy
            next_coord = (next_x, next_y)
            
            if (grid.is_valid_coord(next_coord) and 
                next_coord not in visited and
                next_x > 0 and next_y > 0 and 
                next_x < grid.width - 1 and next_y < grid.height - 1):
                neighbors.append((next_coord, (current[0] + dx // 2, current[1] + dy // 2)))
        
        if neighbors:
            # Choose random neighbor
            next_cell, wall_between = rng.choice(neighbors)
            
            # Mark neighbor as passage
            grid.set_node_state(next_cell, "empty")
            node = grid.get_node(next_cell)
            if node:
                node.walkable = True
            
            # Remove wall between current and neighbor
            grid.set_node_state(wall_between, "empty")
            wall_node = grid.get_node(wall_between)
            if wall_node:
                wall_node.walkable = True
            
            visited.add(next_cell)
            stack.append(next_cell)
        else:
            # Backtrack
            stack.pop()


def _place_start_target_in_maze(grid: Grid, rng: SeededRNG) -> Tuple[Coord, Coord]:
    """
    Place start and target positions in open areas of the maze.
    Ensures they are far apart for interesting pathfinding.
    """
    # Get all empty (passable) cells
    empty_cells = []
    for node in grid.nodes.values():
        if node.state == "empty" and node.walkable:
            empty_cells.append(node.coord)
    
    if len(empty_cells) < 2:
        raise ValueError("Not enough empty cells for start and target")
    
    # Choose start from one corner area
    corner_cells = [coord for coord in empty_cells 
                   if coord[0] < grid.width // 3 and coord[1] < grid.height // 3]
    
    if corner_cells:
        start = rng.choice(corner_cells)
    else:
        start = rng.choice(empty_cells)
    
    # Choose target from opposite area, ensuring distance
    far_cells = [coord for coord in empty_cells 
                if abs(coord[0] - start[0]) + abs(coord[1] - start[1]) > min(grid.width, grid.height) // 2]
    
    if far_cells:
        target = rng.choice(far_cells)
    else:
        # Fallback: choose any cell far from start
        available_cells = [coord for coord in empty_cells if coord != start]
        target = max(available_cells, key=lambda c: abs(c[0] - start[0]) + abs(c[1] - start[1]))
    
    # Set the states
    grid.set_node_state(start, "start")
    grid.set_node_state(target, "target")
    
    return start, target


def generate_solvable_grid(width: int, height: int, wall_density: float,
                          max_attempts: int = 10, seed: Optional[int] = None) -> Tuple[Grid, Coord, Coord]:
    """
    Generate a grid with interesting maze-like structures that is guaranteed to have a path.
    
    Args:
        width: Grid width
        height: Grid height
        wall_density: Desired wall density (0.0 to 1.0)
        max_attempts: Maximum attempts to generate a solvable grid
        seed: Random seed
        
    Returns:
        Tuple of (grid, start_coord, target_coord)
        
    Raises:
        ValueError: If unable to generate solvable grid within max_attempts
    """
    rng = SeededRNG(seed)
    
    for attempt in range(max_attempts):
        try:
            grid = create_empty_grid(width, height)
            
            # Use contiguous walls for more interesting patterns
            add_contiguous_walls(grid, wall_density, rng)
            
            # Place start and target
            start, target = place_start_and_target(grid, rng=rng)
            
            # Ensure solvability by checking path exists (use 4-dir for conservative check)
            if ensure_path_exists(grid, start, target, allow_diagonal=False):
                return grid, start, target
            else:
                # If no path exists, remove some strategic walls to create one
                _ensure_basic_path(grid, start, target, rng)
                if ensure_path_exists(grid, start, target, allow_diagonal=False):
                    return grid, start, target
                
        except ValueError:
            # Try again with different configuration
            continue
    
    # Fallback: create a simple grid with guaranteed path
    grid = create_empty_grid(width, height)
    add_random_walls(grid, min(wall_density, 0.4), rng)  # Higher density for more challenge
    start, target = place_start_and_target(grid, rng=rng)
    return grid, start, target




def generate_multipath_maze(width: int, height: int, seed: Optional[int] = None) -> Tuple[Grid, Coord, Coord]:
    """
    Generate a maze with multiple paths to the goal.
    Creates a more interesting pathfinding scenario by having several viable routes.
    
    Args:
        width: Grid width (minimum 7)
        height: Grid height (minimum 7)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (grid, start_coord, target_coord)
        
    Raises:
        ValueError: If width or height is less than 7
    """
    if width < 7 or height < 7:
        raise ValueError(f"Multipath maze dimensions must be at least 7x7, got {width}x{height}")
    rng = SeededRNG(seed)
    
    # Start with a perfect maze (single path)
    grid, start, target = generate_maze_grid(width, height, seed)
    
    # Add multiple paths by selectively removing walls
    _create_multiple_paths(grid, start, target, rng)
    
    return grid, start, target


def _create_multiple_paths(grid: Grid, start: Coord, target: Coord, rng: SeededRNG) -> None:
    """
    Modify a perfect maze to create multiple paths to the goal.
    """
    # Find walls that could be removed to create alternate paths
    wall_removal_candidates = []
    
    for node in grid.nodes.values():
        if node.state == "wall":
            coord = node.coord
            
            # Check if removing this wall would create a useful alternate path
            if _would_create_alternate_path(grid, coord):
                wall_removal_candidates.append(coord)
    
    # Remove a portion of candidate walls to create multiple paths
    num_walls_to_remove = min(len(wall_removal_candidates) // 3, 
                             max(3, len(wall_removal_candidates) // 4))
    
    walls_to_remove = rng.sample(wall_removal_candidates, 
                                min(num_walls_to_remove, len(wall_removal_candidates)))
    
    for wall_coord in walls_to_remove:
        # Verify removing this wall still maintains maze structure
        if _safe_to_remove_wall(grid, wall_coord):
            grid.set_node_state(wall_coord, "empty")
            node = grid.get_node(wall_coord)
            if node:
                node.walkable = True


def _would_create_alternate_path(grid: Grid, wall_coord: Coord) -> bool:
    """
    Check if removing a wall would create a useful alternate path.
    """
    x, y = wall_coord
    
    # Check neighbors - we want walls that connect two different passage areas
    neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    passage_neighbors = []
    
    for nx, ny in neighbors:
        if grid.is_valid_coord((nx, ny)):
            neighbor_node = grid.get_node((nx, ny))
            if neighbor_node and neighbor_node.walkable:
                passage_neighbors.append((nx, ny))
    
    # We want walls that have exactly 2 passage neighbors (connecting two areas)
    return len(passage_neighbors) == 2


def _safe_to_remove_wall(grid: Grid, wall_coord: Coord) -> bool:
    """
    Check if it's safe to remove a wall without creating too open an area.
    """
    x, y = wall_coord
    
    # Count empty neighbors in 3x3 area around the wall
    empty_count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            check_coord = (x + dx, y + dy)
            if grid.is_valid_coord(check_coord):
                node = grid.get_node(check_coord)
                if node and (node.walkable or node.state in ["start", "target"]):
                    empty_count += 1
    
    # Don't remove wall if it would create too large an open area
    return empty_count <= 4


def generate_branching_maze(width: int, height: int, seed: Optional[int] = None) -> Tuple[Grid, Coord, Coord]:
    """
    Generate a maze with deliberate branching paths and multiple routes.
    Creates a tree-like structure with many decision points.
    
    Args:
        width: Grid width (minimum 7)
        height: Grid height (minimum 7)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (grid, start_coord, target_coord)
        
    Raises:
        ValueError: If width or height is less than 7
    """
    if width < 7 or height < 7:
        raise ValueError(f"Branching maze dimensions must be at least 7x7, got {width}x{height}")
    rng = SeededRNG(seed)
    
    max_attempts = 5
    for attempt in range(max_attempts):
        # Create empty grid
        grid = create_empty_grid(width, height)
        
        # Fill with walls initially
        for node in grid.nodes.values():
            node.state = "wall"
            node.walkable = False
        
        # Create branching path structure
        _create_branching_paths(grid, rng)
        
        try:
            # Place start and target
            start, target = _place_start_target_in_maze(grid, rng)
            
            # Verify the maze is solvable (use 4-dir for conservative check) 
            if ensure_path_exists(grid, start, target, allow_diagonal=False):
                return grid, start, target
            else:
                # If not solvable, try adding more paths
                _add_connecting_paths(grid, rng)
                if ensure_path_exists(grid, start, target, allow_diagonal=False):
                    return grid, start, target
                    
        except ValueError:
            # Not enough empty cells, try creating more paths
            _create_additional_paths(grid, rng)
            continue
    
    # Fallback: return a regular maze if branching generation fails
    return generate_maze_grid(width, height, seed)


def _create_branching_paths(grid: Grid, rng: SeededRNG) -> None:
    """
    Create a branching path structure with multiple routes.
    This creates a network more similar in density to the other maze types.
    """
    # Use a different approach - create multiple connected areas
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Create several seed points across the grid
    seed_points = []
    grid_quarters = [
        (grid.width // 4, grid.height // 4),
        (3 * grid.width // 4, grid.height // 4),
        (grid.width // 4, 3 * grid.height // 4),
        (3 * grid.width // 4, 3 * grid.height // 4),
        (grid.width // 2, grid.height // 2),  # Center
    ]
    
    # Create initial paths from each seed point
    for seed_x, seed_y in grid_quarters:
        if grid.is_valid_coord((seed_x, seed_y)):
            # Create paths in multiple directions from each seed
            grid.set_node_state((seed_x, seed_y), "empty")
            node = grid.get_node((seed_x, seed_y))
            if node:
                node.walkable = True
            
            # Create 2-3 paths from each seed point
            num_paths = rng.randint(2, 4)
            chosen_directions = rng.sample(directions, num_paths)
            
            for direction in chosen_directions:
                length = rng.randint(8, 15)
                _create_straight_path_with_branches(grid, (seed_x, seed_y), direction, rng, length)
    
    # Add connecting paths between different areas
    _add_connecting_corridors(grid, rng)
    
    # Fill in some additional random paths to increase density
    _add_density_paths(grid, rng)


def _create_branching_path_recursive(grid: Grid, start: Coord, direction: Tuple[int, int], 
                                   rng: SeededRNG, continue_prob: float, max_length: int = 15) -> None:
    """
    Create a single branching path with random turns and branches.
    """
    current = start
    length = 0
    
    while length < max_length and rng.random() < continue_prob:
        # Move in current direction
        next_pos = (current[0] + direction[0], current[1] + direction[1])
        
        if not grid.is_valid_coord(next_pos) or not _is_safe_path_cell(grid, next_pos):
            # Try a different direction if blocked
            alternative_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            alternative_directions.remove(direction)
            
            found_alternative = False
            for alt_dir in alternative_directions:
                alt_pos = (current[0] + alt_dir[0], current[1] + alt_dir[1])
                if grid.is_valid_coord(alt_pos) and _is_safe_path_cell(grid, alt_pos):
                    next_pos = alt_pos
                    direction = alt_dir
                    found_alternative = True
                    break
            
            if not found_alternative:
                break
            
        # Create the path cell
        grid.set_node_state(next_pos, "empty")
        node = grid.get_node(next_pos)
        if node:
            node.walkable = True
        
        current = next_pos
        length += 1
        
        # More frequent branching to create denser network
        if rng.random() < 0.4:  # 40% chance to turn or branch (increased from 30%)
            if rng.random() < 0.6:  # 60% chance to branch vs turn (increased branching)
                # Create a branch
                branch_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                branch_directions.remove(direction)
                branch_direction = rng.choice(branch_directions)
                
                # Less probability decay for deeper branches
                _create_branching_path_recursive(grid, current, branch_direction, 
                                               rng, continue_prob * 0.8, max_length // 2 + 2)
            else:
                # Turn to a new direction
                new_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                new_directions.remove(direction)
                direction = rng.choice(new_directions)


def _is_safe_path_cell(grid: Grid, coord: Coord) -> bool:
    """
    Check if it's safe to place a path cell at this location.
    Prevents creating too large open areas while allowing good connectivity.
    """
    x, y = coord
    
    # Don't place path if the cell is already a path
    node = grid.get_node(coord)
    if node and node.walkable:
        return False
    
    # Check if there are too many empty neighbors (relaxed from 3 to 5)
    empty_neighbors = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            check_coord = (x + dx, y + dy)
            if grid.is_valid_coord(check_coord):
                neighbor_node = grid.get_node(check_coord)
                if neighbor_node and neighbor_node.walkable:
                    empty_neighbors += 1
    
    return empty_neighbors <= 5  # Allow up to 5 empty neighbors for better connectivity








def _add_connecting_paths(grid: Grid, rng: SeededRNG) -> None:
    """
    Add additional connecting paths to improve maze connectivity.
    """
    empty_cells = [node.coord for node in grid.nodes.values() if node.walkable]
    
    if len(empty_cells) < 2:
        return
    
    # Add a few more paths between existing empty areas
    for _ in range(min(3, len(empty_cells) // 10)):
        start_cell = rng.choice(empty_cells)
        target_cell = rng.choice(empty_cells)
        
        if start_cell != target_cell:
            _create_simple_path(grid, start_cell, target_cell, rng)


def _create_additional_paths(grid: Grid, rng: SeededRNG) -> None:
    """
    Create additional paths when there aren't enough empty cells.
    """
    # Add some random paths from the center outward
    center_x, center_y = grid.width // 2, grid.height // 2
    center = (center_x, center_y)
    
    # Ensure center is empty
    grid.set_node_state(center, "empty")
    center_node = grid.get_node(center)
    if center_node:
        center_node.walkable = True
    
    # Add paths in cardinal directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for direction in directions:
        _create_simple_path_from_center(grid, center, direction, rng)


def _create_simple_path(grid: Grid, start: Coord, target: Coord, rng: SeededRNG) -> None:
    """
    Create a simple path between two points.
    """
    current = start
    max_steps = abs(target[0] - start[0]) + abs(target[1] - start[1]) + 5
    steps = 0
    
    while current != target and steps < max_steps:
        # Move toward target
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        
        if abs(dx) > abs(dy) and dx != 0:
            next_coord = (current[0] + (1 if dx > 0 else -1), current[1])
        elif dy != 0:
            next_coord = (current[0], current[1] + (1 if dy > 0 else -1))
        else:
            break
        
        if grid.is_valid_coord(next_coord):
            grid.set_node_state(next_coord, "empty")
            node = grid.get_node(next_coord)
            if node:
                node.walkable = True
            current = next_coord
        
        steps += 1


def _create_simple_path_from_center(grid: Grid, center: Coord, direction: Tuple[int, int], rng: SeededRNG) -> None:
    """
    Create a simple path from center in a given direction.
    """
    current = center
    length = rng.randint(3, min(8, grid.width // 3))
    
    for _ in range(length):
        next_coord = (current[0] + direction[0], current[1] + direction[1])
        
        if grid.is_valid_coord(next_coord):
            grid.set_node_state(next_coord, "empty")
            node = grid.get_node(next_coord)
            if node:
                node.walkable = True
            current = next_coord
        else:
            break


def _create_straight_path_with_branches(grid: Grid, start: Coord, direction: Tuple[int, int], 
                                       rng: SeededRNG, length: int) -> None:
    """
    Create a straight path with occasional branches.
    """
    current = start
    
    for i in range(length):
        # Move in the main direction
        next_pos = (current[0] + direction[0], current[1] + direction[1])
        
        if not grid.is_valid_coord(next_pos):
            break
            
        # Create the path cell
        node = grid.get_node(next_pos)
        if node and node.state == "wall":
            grid.set_node_state(next_pos, "empty")
            node.walkable = True
        
        current = next_pos
        
        # Occasionally create a branch (every 3-5 steps)
        if i > 2 and i % rng.randint(3, 6) == 0:
            # Create a short branch
            branch_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            branch_directions.remove(direction)
            
            for branch_dir in rng.sample(branch_directions, rng.randint(1, 2)):
                branch_length = rng.randint(3, 8)
                _create_simple_straight_path(grid, current, branch_dir, branch_length)


def _create_simple_straight_path(grid: Grid, start: Coord, direction: Tuple[int, int], length: int) -> None:
    """
    Create a simple straight path without branches.
    """
    current = start
    
    for _ in range(length):
        next_pos = (current[0] + direction[0], current[1] + direction[1])
        
        if not grid.is_valid_coord(next_pos):
            break
            
        node = grid.get_node(next_pos)
        if node and node.state == "wall":
            grid.set_node_state(next_pos, "empty")
            node.walkable = True
        
        current = next_pos


def _add_connecting_corridors(grid: Grid, rng: SeededRNG) -> None:
    """
    Add corridors to connect different areas of the maze.
    """
    # Find empty cells in different quadrants
    quadrants = [
        [], [], [], []  # top-left, top-right, bottom-left, bottom-right
    ]
    
    for node in grid.nodes.values():
        if node.walkable:
            x, y = node.coord
            if x < grid.width // 2 and y < grid.height // 2:
                quadrants[0].append(node.coord)
            elif x >= grid.width // 2 and y < grid.height // 2:
                quadrants[1].append(node.coord)
            elif x < grid.width // 2 and y >= grid.height // 2:
                quadrants[2].append(node.coord)
            else:
                quadrants[3].append(node.coord)
    
    # Create connections between quadrants
    connections = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (1, 2)]  # Various quadrant pairs
    
    for q1, q2 in rng.sample(connections, min(3, len(connections))):
        if quadrants[q1] and quadrants[q2]:
            start = rng.choice(quadrants[q1])
            end = rng.choice(quadrants[q2])
            _create_simple_path(grid, start, end, rng)


def _add_density_paths(grid: Grid, rng: SeededRNG) -> None:
    """
    Add additional paths to increase overall maze density.
    """
    # Get all empty cells
    empty_cells = [node.coord for node in grid.nodes.values() if node.walkable]
    
    if len(empty_cells) < 10:
        return
    
    # Add random paths to increase density
    num_additional = min(15, len(empty_cells) // 5)
    
    for _ in range(num_additional):
        start = rng.choice(empty_cells)
        direction = rng.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        length = rng.randint(5, 12)
        _create_simple_straight_path(grid, start, direction, length)


def _ensure_basic_path(grid: Grid, start: Coord, target: Coord, rng: SeededRNG) -> None:
    """
    Ensure a basic path exists by clearing strategic walls.
    
    Creates a simple path between start and target by clearing walls
    along a route, but adds some randomness to keep it interesting.
    """
    current = start
    
    while current != target:
        # Move toward target
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        
        # Choose direction (prefer moving toward target but add some randomness)
        if abs(dx) > abs(dy):
            next_coord = (current[0] + (1 if dx > 0 else -1), current[1])
        else:
            next_coord = (current[0], current[1] + (1 if dy > 0 else -1))
        
        # Add some randomness to path (20% chance to take alternate route)
        if rng.random() < 0.2 and abs(dx) > 1 and abs(dy) > 1:
            if abs(dx) > abs(dy):
                next_coord = (current[0], current[1] + (1 if dy > 0 else -1))
            else:
                next_coord = (current[0] + (1 if dx > 0 else -1), current[1])
        
        # Clear the path
        if grid.is_valid_coord(next_coord):
            node = grid.get_node(next_coord)
            if node and node.state == "wall":
                grid.set_node_state(next_coord, "empty")
                node.walkable = True
        
        current = next_coord
        
        # Safety check to avoid infinite loops
        if abs(current[0] - start[0]) + abs(current[1] - start[1]) > grid.width + grid.height:
            break
