"""Distance table computation using lazy breadth-first search.

This module provides the DistTable class for efficiently computing shortest path
distances from a goal location to all other locations in a grid.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .mapf_utils import Coord, Grid, get_neighbors, is_valid_coord


@dataclass
class DistTable:
    """Distance table for computing shortest paths from a goal location.

    Uses lazy BFS evaluation: distances are computed on-demand when first requested,
    with subsequent queries returning cached values. This is efficient when only
    a subset of distances are needed.

    Attributes:
        grid: The 2D grid map.
        goal: The goal coordinate from which distances are computed.
        Q: BFS queue for lazy evaluation (internal use).
        table: 2D array storing computed distances (internal use).
        NIL: Sentinel value indicating uncomputed/unreachable distances (internal use).
    """

    grid: Grid
    goal: Coord
    Q: deque[Coord] = field(init=False)
    table: np.ndarray = field(init=False)  # distance matrix
    NIL: int = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the distance table with the goal location at distance 0."""
        self.NIL = self.grid.size
        self.Q = deque([self.goal])
        self.table = np.full(self.grid.shape, self.NIL, dtype=int)
        self.table[self.goal] = 0

    def get(self, target: Coord) -> int:
        """Get the shortest distance from the goal to the target coordinate.

        Uses lazy BFS evaluation: if the distance is not yet computed, continues
        BFS from where it left off until the target is reached or determined
        unreachable.

        Args:
            target: The coordinate whose distance from goal to compute.

        Returns:
            The shortest path distance (number of steps) from goal to target.
            Returns grid.size if target is invalid or unreachable.
        """
        # check valid input
        if not is_valid_coord(self.grid, target):
            return self.grid.size

        # distance has been known
        if self.table[target] < self.table.size:
            return int(self.table[target])

        # BFS with lazy evaluation
        while len(self.Q) > 0:
            u = self.Q.popleft()
            d = int(self.table[u])
            for v in get_neighbors(self.grid, u):
                if d + 1 < self.table[v]:
                    self.table[v] = d + 1
                    self.Q.append(v)
            if u == target:
                return d

        return self.NIL
