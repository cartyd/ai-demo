"""Seeded random number generator for reproducible demos."""

import random
from typing import Optional


class SeededRNG:
    """Seeded random number generator for reproducible results."""
    
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._seed = seed
    
    @property
    def seed(self) -> Optional[int]:
        """Get the current seed."""
        return self._seed
    
    def set_seed(self, seed: Optional[int]):
        """Set a new seed."""
        self._seed = seed
        self._rng.seed(seed)
    
    def random(self) -> float:
        """Generate a random float in [0.0, 1.0)."""
        return self._rng.random()
    
    def randint(self, a: int, b: int) -> int:
        """Generate a random integer N such that a <= N <= b."""
        return self._rng.randint(a, b)
    
    def choice(self, seq):
        """Choose a random element from a non-empty sequence."""
        return self._rng.choice(seq)
    
    def shuffle(self, seq) -> None:
        """Shuffle the sequence in place."""
        self._rng.shuffle(seq)
    
    def sample(self, population, k: int):
        """Choose k unique random elements from the population."""
        return self._rng.sample(population, k)
    
    def uniform(self, a: float, b: float) -> float:
        """Generate a random float N such that a <= N <= b."""
        return self._rng.uniform(a, b)
    
    def gauss(self, mu: float, sigma: float) -> float:
        """Generate a random float with Gaussian distribution."""
        return self._rng.gauss(mu, sigma)


# Global instance for convenience
default_rng = SeededRNG()


def set_global_seed(seed: Optional[int]):
    """Set the seed for the global RNG instance."""
    default_rng.set_seed(seed)


def get_global_seed() -> Optional[int]:
    """Get the seed of the global RNG instance."""
    return default_rng.seed