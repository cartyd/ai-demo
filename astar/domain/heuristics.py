"""Heuristic functions for A* pathfinding algorithm."""

import math
from typing import Callable
from .types import Coord, HeuristicId


def manhattan_distance(start: Coord, target: Coord) -> float:
    """
    Manhattan (L1) distance heuristic.
    Admissible for 4-directional movement.
    """
    return abs(start[0] - target[0]) + abs(start[1] - target[1])


def euclidean_distance(start: Coord, target: Coord) -> float:
    """
    Euclidean (L2) distance heuristic.
    Admissible for any movement but may be too optimistic for grid-based movement.
    """
    dx = start[0] - target[0]
    dy = start[1] - target[1]
    return math.sqrt(dx * dx + dy * dy)


def diagonal_distance(start: Coord, target: Coord) -> float:
    """
    Diagonal (Chebyshev/L∞) distance heuristic.
    Admissible for 8-directional movement where diagonal cost = orthogonal cost.
    """
    return max(abs(start[0] - target[0]), abs(start[1] - target[1]))


def octile_distance(start: Coord, target: Coord) -> float:
    """
    Octile distance heuristic for 8-directional movement.
    Assumes diagonal moves cost √2 ≈ 1.414, orthogonal moves cost 1.
    More accurate than diagonal distance for typical grid pathfinding.
    """
    dx = abs(start[0] - target[0])
    dy = abs(start[1] - target[1])
    
    # Cost: min(dx, dy) diagonal moves + |dx - dy| orthogonal moves
    return min(dx, dy) * math.sqrt(2) + abs(dx - dy)


# Mapping from heuristic IDs to functions
HEURISTICS: dict[HeuristicId, Callable[[Coord, Coord], float]] = {
    "manhattan": manhattan_distance,
    "euclidean": euclidean_distance,
    "diagonal": diagonal_distance,
    "octile": octile_distance,
}


def get_heuristic(heuristic_id: HeuristicId) -> Callable[[Coord, Coord], float]:
    """Get heuristic function by ID."""
    return HEURISTICS[heuristic_id]


def is_admissible(heuristic_id: HeuristicId, allow_diagonal: bool) -> bool:
    """
    Check if a heuristic is admissible for the given movement rules.
    An admissible heuristic never overestimates the true cost.
    """
    if allow_diagonal:
        # For 8-directional movement, octile is most accurate, 
        # diagonal is admissible, manhattan may overestimate
        return heuristic_id in ["octile", "diagonal", "euclidean"]
    else:
        # For 4-directional movement, manhattan is exact,
        # others may underestimate
        return heuristic_id in ["manhattan"]


def get_recommended_heuristic(allow_diagonal: bool) -> HeuristicId:
    """Get the recommended heuristic for the movement type."""
    return "octile" if allow_diagonal else "manhattan"