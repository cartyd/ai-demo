"""Random number generation utilities for RL pathfinding."""

import random
import numpy as np
from typing import Optional


class SeededRNG:
    """Seeded random number generator for reproducible results."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def random(self) -> float:
        """Generate random float in [0, 1)."""
        return random.random()
    
    def randint(self, a: int, b: int) -> int:
        """Generate random integer in [a, b]."""
        return random.randint(a, b)
    
    def choice(self, seq):
        """Choose random element from sequence."""
        return random.choice(seq)
    
    def sample(self, population, k: int):
        """Sample k elements from population without replacement."""
        return random.sample(population, k)
    
    def shuffle(self, seq):
        """Shuffle sequence in place."""
        random.shuffle(seq)
    
    def uniform(self, a: float, b: float) -> float:
        """Generate random float in [a, b)."""
        return random.uniform(a, b)


# Default RNG instance
default_rng = SeededRNG()


def set_global_seed(seed: Optional[int]):
    """Set global random seed for reproducibility."""
    global default_rng
    default_rng = SeededRNG(seed)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)