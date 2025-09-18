"""Core type definitions for the A* pathfinding algorithm."""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict

# Coordinate type for grid positions
Coord = Tuple[int, int]

# Node states for visualization
NodeState = Literal[
    "empty", "wall", "start", "target",
    "open", "closed", "path", "current"
]

# Heuristic function identifiers
HeuristicId = Literal["manhattan", "euclidean", "diagonal", "octile"]

# Movement types
MovementType = Literal["4-dir", "8-dir"]


@dataclass
class Costs:
    """Stores G, H, and F costs for A* algorithm."""
    g: float = 0.0  # Cost from start
    h: float = 0.0  # Heuristic estimate to target
    f: float = 0.0  # Total cost (g + h)

    def __post_init__(self):
        """Ensure f = g + h."""
        self.f = self.g + self.h


@dataclass
class GridNode:
    """Represents a single node in the pathfinding grid."""
    id: str
    coord: Coord
    cost: Costs
    walkable: bool = True
    weight: float = 1.0
    state: NodeState = "empty"
    parent: Optional[str] = None

    def reset_costs(self):
        """Reset all costs to zero."""
        self.cost = Costs()
        self.parent = None

    def is_passable(self) -> bool:
        """Check if this node can be traversed."""
        return self.walkable and self.state not in ["wall"]


@dataclass
class Grid:
    """Represents the entire pathfinding grid."""
    width: int
    height: int
    nodes: Dict[str, GridNode]

    def get_node(self, coord: Coord) -> Optional[GridNode]:
        """Get node at coordinate, returns None if out of bounds."""
        node_id = f"{coord[0]},{coord[1]}"
        return self.nodes.get(node_id)

    def set_node_state(self, coord: Coord, state: NodeState):
        """Set the state of a node at given coordinate."""
        node = self.get_node(coord)
        if node:
            node.state = state

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if coordinate is within grid bounds."""
        x, y = coord
        return 0 <= x < self.width and 0 <= y < self.height

    def reset_pathfinding_states(self):
        """Reset all nodes to remove pathfinding visualization states."""
        for node in self.nodes.values():
            if node.state in ["open", "closed", "path", "current"]:
                node.state = "empty"
            node.reset_costs()


@dataclass
class AlgoConfig:
    """Configuration for the A* algorithm."""
    movement: MovementType = "4-dir"
    corner_cutting: bool = False
    heuristic: HeuristicId = "manhattan"
    step_mode: bool = False

    @property
    def allow_diagonal(self) -> bool:
        """Whether diagonal movement is allowed."""
        return self.movement == "8-dir"


@dataclass
class PathfindingResult:
    """Result of a pathfinding operation."""
    path: Optional[list[Coord]] = None
    path_cost: float = 0.0
    nodes_explored: int = 0
    found: bool = False
    
    @property
    def success(self) -> bool:
        """Whether pathfinding was successful."""
        return self.found and self.path is not None and len(self.path) > 0