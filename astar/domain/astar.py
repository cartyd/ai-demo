"""Core A* pathfinding algorithm implementation."""

from typing import Optional, Set, Callable
from .types import Coord, Grid, AlgoConfig, PathfindingResult, Costs
from .priority_queue import PriorityQueue
from .heuristics import get_heuristic
from .neighbors import get_neighbors
from .path import reconstruct_path, calculate_path_cost


class AStarAlgorithm:
    """
    A* pathfinding algorithm implementation.
    Framework-agnostic pure Python implementation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the algorithm state."""
        self.open_set = PriorityQueue()
        self.closed_set: Set[str] = set()
        self.start_coord: Optional[Coord] = None
        self.target_coord: Optional[Coord] = None
        self.config = AlgoConfig()
        self.nodes_explored = 0
        self.current_node_id: Optional[str] = None
    
    def initialize(self, start: Coord, target: Coord, grid: Grid, config: AlgoConfig):
        """Initialize the algorithm with start and target positions."""
        if not grid.is_valid_coord(start):
            raise ValueError(f"Start coordinate {start} is out of bounds")
        if not grid.is_valid_coord(target):
            raise ValueError(f"Target coordinate {target} is out of bounds")
        
        start_node = grid.get_node(start)
        target_node = grid.get_node(target)
        
        if not start_node or not start_node.is_passable():
            raise ValueError(f"Start position {start} is not passable")
        if not target_node or not target_node.is_passable():
            raise ValueError(f"Target position {target} is not passable")
        
        if start == target:
            raise ValueError("Start and target positions are the same")
        
        self.reset()
        self.start_coord = start
        self.target_coord = target
        self.config = config
        
        # Reset grid pathfinding states
        grid.reset_pathfinding_states()
        
        # Initialize start node
        start_node.cost = Costs(g=0.0, h=self._calculate_heuristic(start), f=0.0)
        start_node.cost.f = start_node.cost.g + start_node.cost.h
        start_node.parent = None
        
        # Add start to open set
        start_id = self._coord_to_id(start)
        self.open_set.put(
            start_id, 
            start_node.cost.f, 
            start_node.cost.h, 
            start_node.cost.g, 
            start
        )
    
    def step(self, grid: Grid) -> Optional[PathfindingResult]:
        """
        Execute one step of the A* algorithm.
        Returns PathfindingResult if algorithm is complete, None otherwise.
        """
        if not self.start_coord or not self.target_coord:
            raise ValueError("Algorithm not initialized")
        
        # Check if we have nodes to explore
        if self.open_set.is_empty():
            return PathfindingResult(found=False, nodes_explored=self.nodes_explored)
        
        # Get the node with lowest f-cost
        result = self.open_set.get()
        if not result:
            return PathfindingResult(found=False, nodes_explored=self.nodes_explored)
        
        current_id, current_coord = result
        self.current_node_id = current_id
        self.nodes_explored += 1
        
        # Move current from open to closed
        self.closed_set.add(current_id)
        current_node = grid.get_node(current_coord)
        if not current_node:
            return PathfindingResult(found=False, nodes_explored=self.nodes_explored)
        
        # Update visual state
        if current_coord != self.start_coord and current_coord != self.target_coord:
            current_node.state = "current"
        
        # Check if we reached the target
        if current_coord == self.target_coord:
            path = reconstruct_path(self.target_coord, grid)
            path_cost = calculate_path_cost(path, grid)
            return PathfindingResult(
                path=path,
                path_cost=path_cost,
                found=True,
                nodes_explored=self.nodes_explored
            )
        
        # Explore neighbors
        neighbors = get_neighbors(current_coord, grid, self.config)
        for neighbor_coord, move_cost in neighbors:
            neighbor_id = self._coord_to_id(neighbor_coord)
            
            # Skip if already evaluated
            if neighbor_id in self.closed_set:
                continue
            
            neighbor_node = grid.get_node(neighbor_coord)
            if not neighbor_node:
                continue
            
            # Calculate tentative g cost
            tentative_g = current_node.cost.g + move_cost
            
            # Check if this is a better path to the neighbor
            if (not self.open_set.contains(neighbor_id) or 
                tentative_g < neighbor_node.cost.g):
                
                # Update neighbor costs
                h_cost = self._calculate_heuristic(neighbor_coord)
                f_cost = tentative_g + h_cost
                
                neighbor_node.cost = Costs(g=tentative_g, h=h_cost, f=f_cost)
                neighbor_node.parent = current_id
                
                # Add to open set
                self.open_set.put(
                    neighbor_id, f_cost, h_cost, tentative_g, neighbor_coord
                )
                
                # Update visual state
                if (neighbor_coord != self.start_coord and 
                    neighbor_coord != self.target_coord):
                    neighbor_node.state = "open"
        
        # Mark current as closed (visually)
        if current_coord != self.start_coord and current_coord != self.target_coord:
            current_node.state = "closed"
        
        return None  # Algorithm continues
    
    def run_complete(self, grid: Grid) -> PathfindingResult:
        """
        Run the complete A* algorithm until completion.
        Returns the final PathfindingResult.
        """
        max_iterations = grid.width * grid.height * 2  # Safety limit
        iterations = 0
        
        while iterations < max_iterations:
            result = self.step(grid)
            if result is not None:
                return result
            iterations += 1
        
        # Algorithm didn't complete within reasonable iterations
        return PathfindingResult(found=False, nodes_explored=self.nodes_explored)
    
    def _calculate_heuristic(self, coord: Coord) -> float:
        """Calculate heuristic cost from coordinate to target."""
        if not self.target_coord:
            return 0.0
        
        heuristic_func = get_heuristic(self.config.heuristic)
        return heuristic_func(coord, self.target_coord)
    
    def _coord_to_id(self, coord: Coord) -> str:
        """Convert coordinate to string ID."""
        return f"{coord[0]},{coord[1]}"
    
    def get_open_set_coords(self) -> list[Coord]:
        """Get all coordinates currently in the open set."""
        coords = []
        for item_id, _, coord in self.open_set.get_all_items():
            coords.append(coord)
        return coords
    
    def get_closed_set_coords(self, grid: Grid) -> list[Coord]:
        """Get all coordinates currently in the closed set."""
        coords = []
        for node_id in self.closed_set:
            coord = self._id_to_coord(node_id)
            if coord:
                coords.append(coord)
        return coords
    
    def _id_to_coord(self, node_id: str) -> Optional[Coord]:
        """Convert string ID back to coordinate."""
        try:
            parts = node_id.split(',')
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
        except ValueError:
            pass
        return None
    
    def is_complete(self) -> bool:
        """Check if the algorithm has completed (success or failure)."""
        return self.open_set.is_empty()
    
    def get_current_node_coord(self) -> Optional[Coord]:
        """Get the coordinate of the currently being processed node."""
        if self.current_node_id:
            return self._id_to_coord(self.current_node_id)
        return None


def find_path(start: Coord, target: Coord, grid: Grid, config: AlgoConfig) -> PathfindingResult:
    """
    Convenience function to run A* pathfinding from start to finish.
    
    Args:
        start: Starting coordinate
        target: Target coordinate  
        grid: Grid to search in
        config: Algorithm configuration
        
    Returns:
        PathfindingResult with path and statistics
    """
    algorithm = AStarAlgorithm()
    try:
        algorithm.initialize(start, target, grid, config)
        return algorithm.run_complete(grid)
    except ValueError as e:
        return PathfindingResult(found=False, nodes_explored=0)