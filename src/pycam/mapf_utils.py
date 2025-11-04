"""Utility functions and data structures for Multi-Agent Path Finding (MAPF).

This module provides core data structures (Grid, Coord, Config, Deadline) and
utility functions for loading MAPF problem instances, validating solutions, and
computing solution costs.
"""

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Grid: TypeAlias = NDArray[np.bool_]
"""2D boolean array representing a grid map where True indicates passable cells."""

Coord: TypeAlias = tuple[int, int]
"""Coordinate tuple (y, x) representing a position in the grid."""


@dataclass
class Config:
    """Configuration representing positions of all agents at a specific timestep.

    A configuration is essentially a list of coordinates, one for each agent.
    It supports list-like access patterns and hashing for use in search algorithms.

    Attributes:
        positions: List of agent positions as (y, x) coordinates.
    """

    positions: list[Coord] = field(default_factory=lambda: [])

    def __getitem__(self, k: int) -> Coord:
        """Get the position of agent k.

        Args:
            k: Agent index.

        Returns:
            The (y, x) coordinate of agent k.
        """
        return self.positions[k]

    def __setitem__(self, k: int, coord: Coord) -> None:
        """Set the position of agent k.

        Args:
            k: Agent index.
            coord: New (y, x) coordinate for agent k.
        """
        self.positions[k] = coord

    def __len__(self) -> int:
        """Get the number of agents in this configuration.

        Returns:
            Number of agents.
        """
        return len(self.positions)

    def __hash__(self) -> int:
        """Compute hash for use in sets and dictionaries.

        Returns:
            Hash value based on agent positions.
        """
        return hash(tuple(self.positions))

    def __eq__(self, other: object) -> bool:
        """Check equality with another configuration.

        Args:
            other: Object to compare with.

        Returns:
            True if other is a Config with identical positions.
        """
        if not isinstance(other, Config):
            return NotImplemented
        return self.positions == other.positions

    def append(self, coord: Coord) -> None:
        """Add a new agent position to this configuration.

        Args:
            coord: The (y, x) coordinate to add.
        """
        self.positions.append(coord)

    def __iter__(self) -> Iterator[Coord]:
        """Iterate over agent positions.

        Returns:
            Iterator over agent positions.
        """
        return iter(self.positions)


Configs: TypeAlias = list[Config]
"""List of configurations representing a solution path over time."""


@dataclass
class Deadline:
    """Time limit manager for search algorithms.

    Tracks elapsed time and checks whether a time limit has been exceeded.

    Attributes:
        time_limit_ms: Maximum allowed time in milliseconds.
    """

    time_limit_ms: int

    def __post_init__(self) -> None:
        """Initialize the start time when the deadline is created."""
        self.start_time = time.time()

    @property
    def elapsed(self) -> float:
        """Get elapsed time since deadline creation.

        Returns:
            Elapsed time in milliseconds.
        """
        return (time.time() - self.start_time) * 1000

    @property
    def is_expired(self) -> bool:
        """Check if the time limit has been exceeded.

        Returns:
            True if elapsed time exceeds the time limit.
        """
        return self.elapsed > self.time_limit_ms


def get_grid(map_file: str | Path) -> Grid:
    """Load a grid map from a file in MAPF benchmark format.

    The map file format is:
    - Lines starting with "width" and "height" specify dimensions
    - Grid rows use '.' for passable cells and other characters for obstacles

    Args:
        map_file: Path to the map file.

    Returns:
        A 2D boolean numpy array where True indicates passable cells and
        False indicates obstacles. Shape is (height, width).

    Raises:
        AssertionError: If the map format is invalid or dimensions don't match.
    """
    width, height = 0, 0
    with open(map_file, "r") as f:
        # retrieve map size
        for row in f:
            # get width
            res = re.match(r"width\s(\d+)", row)
            if res:
                width = int(res.group(1))

            # get height
            res = re.match(r"height\s(\d+)", row)
            if res:
                height = int(res.group(1))

            if width > 0 and height > 0:
                break

        # retrieve map
        grid = np.zeros((height, width), dtype=bool)
        y = 0
        for row in f:
            row = row.strip()
            if len(row) == width and row != "map":
                grid[y] = [s == "." for s in row]
                y += 1

    # simple error check
    assert y == height, f"map format seems strange, check {map_file}"

    # grid[y, x] -> True: available, False: obstacle
    return grid


def get_scenario(scen_file: str | Path, N: int | None = None) -> tuple[Config, Config]:
    """Load start and goal configurations from a scenario file.

    The scenario file format follows MAPF benchmark conventions with tab-separated
    values including start/goal coordinates.

    Args:
        scen_file: Path to the scenario file.
        N: Optional maximum number of agents to load. If None, loads all agents.

    Returns:
        A tuple (starts, goals) where:
        - starts: Configuration of starting positions
        - goals: Configuration of goal positions
    """
    with open(scen_file, "r") as f:
        starts, goals = Config(), Config()
        for row in f:
            res = re.match(
                r"\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+", row
            )
            if res:
                x_s, y_s, x_g, y_g = [int(res.group(k)) for k in range(1, 5)]
                starts.append((y_s, x_s))  # align with grid
                goals.append((y_g, x_g))

                # check the number of agents
                if (N is not None) and len(starts) >= N:
                    break

    return starts, goals


def is_valid_coord(grid: Grid, coord: Coord) -> bool:
    """Check if a coordinate is valid and passable in the grid.

    Args:
        grid: The grid map.
        coord: The (y, x) coordinate to check.

    Returns:
        True if the coordinate is within bounds and represents a passable cell,
        False otherwise.
    """
    y, x = coord
    if y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1] or not grid[coord]:
        return False
    return True


def get_neighbors(grid: Grid, coord: Coord) -> list[Coord]:
    """Get all valid neighboring coordinates (4-connected).

    Args:
        grid: The grid map.
        coord: The (y, x) coordinate whose neighbors to find.

    Returns:
        List of valid neighboring coordinates. Returns empty list if coord
        itself is invalid.
    """
    # coord: y, x
    neigh: list[Coord] = []

    # check valid input
    if not is_valid_coord(grid, coord):
        return neigh

    y, x = coord

    if x > 0 and grid[y, x - 1]:
        neigh.append((y, x - 1))

    if x < grid.shape[1] - 1 and grid[y, x + 1]:
        neigh.append((y, x + 1))

    if y > 0 and grid[y - 1, x]:
        neigh.append((y - 1, x))

    if y < grid.shape[0] - 1 and grid[y + 1, x]:
        neigh.append((y + 1, x))

    return neigh


def save_configs_for_visualizer(configs: Configs, filename: str | Path) -> None:
    """Save solution configurations to a file for visualization tools.

    The output format is compatible with mapf-visualizer tools.

    Args:
        configs: List of configurations representing the solution path.
        filename: Output file path. Parent directories will be created if needed.
    """
    output_dirname = Path(filename).parent
    if not output_dirname.exists():
        output_dirname.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        for t, config in enumerate(configs):
            row = f"{t}:" + "".join([f"({x},{y})," for (y, x) in config]) + "\n"
            f.write(row)


def validate_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> None:
    """Validate a MAPF solution for correctness.

    Checks that:
    - Solution starts with the start configuration
    - Solution ends with the goal configuration
    - All transitions are valid (agents move to adjacent cells or stay)
    - No vertex collisions (two agents at same location)
    - No edge collisions (two agents swap positions)

    Args:
        grid: The grid map.
        starts: Starting configuration.
        goals: Goal configuration.
        solution: List of configurations representing the solution path.

    Raises:
        AssertionError: If any validation check fails.
    """
    # starts
    assert all(
        [u == v for (u, v) in zip(starts, solution[0])]
    ), "invalid solution, check starts"

    # goals
    assert all(
        [u == v for (u, v) in zip(goals, solution[-1])]
    ), "invalid solution, check goals"

    T = len(solution)
    N = len(starts)

    for t in range(T):
        for i in range(N):
            v_i_now = solution[t][i]
            v_i_pre = solution[max(t - 1, 0)][i]

            # check continuity
            assert v_i_now in [v_i_pre] + get_neighbors(
                grid, v_i_pre
            ), "invalid solution, check connectivity"

            # check collision
            for j in range(i + 1, N):
                v_j_now = solution[t][j]
                v_j_pre = solution[max(t - 1, 0)][j]
                assert not (v_i_now == v_j_now), "invalid solution, vertex collision"
                assert not (
                    v_i_now == v_j_pre and v_i_pre == v_j_now
                ), "invalid solution, edge collision"


def is_valid_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> bool:
    """Check if a MAPF solution is valid without raising exceptions.

    Args:
        grid: The grid map.
        starts: Starting configuration.
        goals: Goal configuration.
        solution: List of configurations representing the solution path.

    Returns:
        True if the solution is valid, False otherwise.
    """
    try:
        validate_mapf_solution(grid, starts, goals, solution)
        return True
    except Exception as e:
        print(e)
        return False


def get_sum_of_loss(configs: Configs) -> int:
    """Calculate the sum of loss (total number of non-goal moves) in a solution.

    For each timestep and agent, counts 1 if the agent is not at its goal or
    moved from its goal. This is a common MAPF solution quality metric.

    Args:
        configs: List of configurations representing the solution path.
        The last configuration is assumed to be the goal configuration.

    Returns:
        Total sum of loss across all timesteps and agents.
    """
    cost = 0
    for t in range(1, len(configs)):
        cost += sum(
            [
                not (v_from == v_to == goal)
                for (v_from, v_to, goal) in zip(configs[t - 1], configs[t], configs[-1])
            ]
        )
    return cost
