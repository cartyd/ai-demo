"""Priority queue implementation for A* algorithm with proper tie-breaking."""

import heapq
from typing import List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class PriorityItem:
    """
    Item in the priority queue with proper comparison for tie-breaking.
    
    Comparison order:
    1. f_cost (lower is better)
    2. h_cost (lower is better - favor nodes closer to target)
    3. g_cost (lower is better - favor nodes closer to start)
    4. item_id (lexicographical order for determinism)
    """
    f_cost: float
    h_cost: float
    g_cost: float
    item_id: str
    data: Any
    
    def __lt__(self, other: 'PriorityItem') -> bool:
        """Define comparison for heap ordering."""
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        if self.h_cost != other.h_cost:
            return self.h_cost < other.h_cost
        if self.g_cost != other.g_cost:
            return self.g_cost < other.g_cost
        return self.item_id < other.item_id


class PriorityQueue:
    """
    Priority queue optimized for A* algorithm.
    Uses binary heap with proper tie-breaking for deterministic behavior.
    """
    
    def __init__(self):
        self._heap: List[PriorityItem] = []
        self._entry_finder: dict[str, PriorityItem] = {}
        self._removed_marker = object()  # Sentinel for removed items
        
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._entry_finder) == 0
    
    def size(self) -> int:
        """Get the number of items in the queue."""
        return len(self._entry_finder)
    
    def put(self, item_id: str, f_cost: float, h_cost: float, g_cost: float, data: Any):
        """
        Add an item to the queue or update its priority.
        If the item already exists with lower f_cost, it won't be updated.
        """
        # Check if we already have this item with better or equal cost
        if item_id in self._entry_finder:
            existing = self._entry_finder[item_id]
            if existing.data is not self._removed_marker and existing.f_cost <= f_cost:
                # Keep the existing better entry
                return
            # Mark the existing entry as removed
            existing.data = self._removed_marker
        
        # Create new entry
        entry = PriorityItem(f_cost, h_cost, g_cost, item_id, data)
        self._entry_finder[item_id] = entry
        heapq.heappush(self._heap, entry)
    
    def get(self) -> Optional[Tuple[str, Any]]:
        """
        Remove and return the item with lowest f_cost.
        Returns None if queue is empty.
        """
        while self._heap:
            entry = heapq.heappop(self._heap)
            if entry.data is not self._removed_marker:
                del self._entry_finder[entry.item_id]
                return (entry.item_id, entry.data)
        return None
    
    def peek(self) -> Optional[Tuple[str, float, Any]]:
        """
        Look at the next item without removing it.
        Returns (item_id, f_cost, data) or None if empty.
        """
        while self._heap:
            entry = self._heap[0]
            if entry.data is not self._removed_marker:
                return (entry.item_id, entry.f_cost, entry.data)
            # Remove stale entry and continue
            heapq.heappop(self._heap)
        return None
    
    def contains(self, item_id: str) -> bool:
        """Check if an item is in the queue."""
        return item_id in self._entry_finder and \
               self._entry_finder[item_id].data is not self._removed_marker
    
    def get_cost(self, item_id: str) -> Optional[float]:
        """Get the f_cost of an item in the queue, or None if not present."""
        if self.contains(item_id):
            return self._entry_finder[item_id].f_cost
        return None
    
    def clear(self):
        """Remove all items from the queue."""
        self._heap.clear()
        self._entry_finder.clear()
    
    def get_all_items(self) -> List[Tuple[str, float, Any]]:
        """
        Get all items in the queue without removing them.
        Returns list of (item_id, f_cost, data) tuples.
        Useful for visualization.
        """
        items = []
        for entry in self._heap:
            if entry.data is not self._removed_marker:
                items.append((entry.item_id, entry.f_cost, entry.data))
        return items