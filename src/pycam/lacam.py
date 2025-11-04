"""LaCAM* algorithm implementation for Multi-Agent Path Finding.

This module implements a simplified version of the LaCAM* algorithm
(Lazy Constraints Addition search for MAPF), an anytime search-based
algorithm for solving MAPF problems with eventual optimality guarantees.

Algorithm Structure:
    LaCAM* uses a two-level search approach:

    - **High-level search**: Explores the configuration space (complete states
      of all agents' positions) using best-first search guided by heuristic costs.
      Configurations are evaluated using f = g + h, where g is the actual cost
      from the start and h is a heuristic lower bound to the goal.

    - **Low-level search**: For each high-level node, explores different orderings
      and constraints on agent movements to generate valid successor configurations
      without collisions.

Key Properties:
    - **Anytime algorithm**: Can be interrupted at any time with a valid solution
    - **Eventually optimal**: Given sufficient time, converges to optimal solutions
      for sum-of-costs objective with cumulative transition costs
    - **Complete**: Always finds a solution if one exists

Implementation Notes:
    This is a **minimal educational implementation** using random action selection
    instead of PIBT for simplicity. While this maintains the core algorithmic
    structure and theoretical properties, it significantly reduces practical
    performance compared to the full LaCAM3 implementation with PIBT integration.

    For production use cases requiring maximum scalability (10k+ agents), consider:
    - LaCAM with PIBT: https://github.com/Kei18/py-lacam/tree/pibt
    - LaCAM3 (C++): https://github.com/Kei18/lacam3

References:
    - Okumura, K. LaCAM: Search-Based Algorithm for Quick Multi-Agent Pathfinding.
      AAAI. 2023. https://ojs.aaai.org/index.php/AAAI/article/view/26377
    - Okumura, K. Improving LaCAM for Scalable Eventually Optimal Multi-Agent Pathfinding.
      IJCAI. 2023. https://www.ijcai.org/proceedings/2023/28
    - Okumura, K. Engineering LaCAM*: Towards Real-Time, Large-Scale, and Near-Optimal
      Multi-Agent Pathfinding. AAMAS. 2024. https://arxiv.org/abs/2308.04292
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from .dist_table import DistTable
from .mapf_utils import (
    Config,
    Configs,
    Coord,
    Deadline,
    Grid,
    get_neighbors,
    is_valid_coord,
)

NO_AGENT: int = np.iinfo(np.int32).max
"""Sentinel value indicating no agent occupies a location."""

NO_LOCATION: Coord = (np.iinfo(np.int32).max, np.iinfo(np.int32).max)
"""Sentinel coordinate indicating an unassigned location."""

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]  # d_y, d_x
"""Possible actions: up, right, down, left, stay."""


@dataclass
class LowLevelNode:
    """Low-level search node representing partial agent assignments.

    In LaCAM*, low-level nodes represent constraints on which agents must move
    to which locations. The low-level tree explores different orderings and
    assignments to find collision-free configurations.

    Attributes:
        who: List of agent IDs with assigned next locations.
        where: List of next locations for the corresponding agents.
        depth: Number of agents with assigned locations (len(who) == len(where)).
    """

    who: list[int] = field(default_factory=lambda: [])
    where: list[Coord] = field(default_factory=lambda: [])
    depth: int = 0

    def get_child(self, who: int, where: Coord) -> LowLevelNode:
        """Create a child node with one additional agent assignment.

        Args:
            who: Agent ID to assign.
            where: Location to assign to the agent.

        Returns:
            New LowLevelNode with the additional assignment.
        """
        return LowLevelNode(
            who=self.who + [who],
            where=self.where + [where],
            depth=self.depth + 1,
        )


@dataclass
class HighLevelNode:
    """High-level search node representing a complete configuration.

    High-level nodes form the main search space, where each node represents
    a complete configuration (positions of all agents). The search performs
    best-first search with lower bounds to find optimal or near-optimal solutions.

    Attributes:
        Q: Current configuration (positions of all agents).
        order: Order in which agents are assigned locations in low-level search.
        parent: Parent node in the search tree (for solution reconstruction).
        tree: Low-level search tree for this node (list of constraint nodes).
        g: Actual cost from start to this configuration.
        h: Heuristic estimate of cost from this configuration to goal.
        f: Total estimated cost (g + h).
        neighbors: Set of neighboring configurations generated from this node.
    """

    Q: Config
    order: list[int]
    parent: HighLevelNode | None = None
    tree: deque[LowLevelNode] = field(default_factory=lambda: deque([LowLevelNode()]))
    g: int = 0
    h: int = 0
    f: int = field(init=False)
    neighbors: set[HighLevelNode] = field(default_factory=lambda: set())

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass initialization."""
        self.f = self.g + self.h

    def __eq__(self, other: object) -> bool:
        """Check equality based on configuration.

        Args:
            other: Object to compare with.

        Returns:
            True if other is a HighLevelNode with the same configuration.
        """
        if isinstance(other, HighLevelNode):
            return self.Q == other.Q
        return False

    def __hash__(self) -> int:
        """Compute hash based on configuration for use in sets/dicts.

        Returns:
            Hash value of the configuration.
        """
        return self.Q.__hash__()


class LaCAM:
    """LaCAM* solver for Multi-Agent Path Finding problems.

    LaCAM* is an anytime search-based algorithm that performs a two-level search
    in the configuration space to find collision-free paths for multiple agents.

    Algorithm Overview:
        **High-level search**: Best-first search over configurations (complete
        states of all agents). Each configuration is evaluated using f = g + h,
        where g is the actual cost and h is a heuristic lower bound.

        **Low-level search**: For each high-level configuration, explores different
        agent orderings and movement constraints to generate valid successor
        configurations without collisions.

    Solution Modes:
        - **Anytime mode (flg_star=True)**: Continues refining the solution
          after finding an initial solution, eventually converging to optimal
          (given sufficient time). This is the default mode.
        - **First-solution mode (flg_star=False)**: Returns immediately after
          finding the first valid solution (faster but suboptimal).

    Optimality Guarantee:
        When run in anytime mode with sufficient time, LaCAM* is **eventually
        optimal** for the sum-of-costs objective, provided that:
        1. Solution costs are cumulative transition costs
        2. The search is not terminated prematurely

    Performance Characteristics:
        - **Completeness**: Always finds a solution if one exists
        - **Scalability**: This simplified implementation handles dozens to hundreds
          of agents. For 10k+ agents, use LaCAM3 with PIBT integration.
        - **Time complexity**: Dependent on problem size and time limit

    Example:
        >>> from pycam import LaCAM, get_grid, get_scenario
        >>> grid = get_grid("map.map")
        >>> starts, goals = get_scenario("scenario.scen", N=4)
        >>> planner = LaCAM()
        >>>
        >>> # Anytime mode: Get eventually optimal solution
        >>> solution = planner.solve(
        ...     grid=grid,
        ...     starts=starts,
        ...     goals=goals,
        ...     time_limit_ms=5000,
        ...     flg_star=True,  # Enable refinement
        ...     verbose=1
        ... )
        >>>
        >>> # Fast mode: Get first solution quickly
        >>> solution = planner.solve(
        ...     grid=grid,
        ...     starts=starts,
        ...     goals=goals,
        ...     time_limit_ms=1000,
        ...     flg_star=False,  # Disable refinement
        ...     verbose=1
        ... )
    """

    def __init__(self) -> None:
        """Initialize the LaCAM* solver."""
        pass

    def solve(
        self,
        grid: Grid,
        starts: Config,
        goals: Config,
        time_limit_ms: int = 3000,
        deadline: Deadline | None = None,
        flg_star: bool = True,
        seed: int = 0,
        verbose: int = 1,
    ) -> Configs:
        """Solve a MAPF problem instance.

        Args:
            grid: The 2D grid map.
            starts: Starting configuration (initial positions of all agents).
            goals: Goal configuration (target positions of all agents).
            time_limit_ms: Time limit in milliseconds (default: 3000).
            deadline: Optional Deadline object (if None, created from time_limit_ms).
            flg_star: If True, refine solution for optimality (default: True).
                     If False, return first found solution (suboptimal).
            seed: Random seed for tie-breaking and action ordering (default: 0).
            verbose: Verbosity level (0: silent, 1: basic, 2+: detailed) (default: 1).

        Returns:
            List of configurations representing the solution path from starts to goals.
            Returns empty list if no solution found within time limit.
        """
        # set problem
        self.num_agents: int = len(starts)
        self.grid: Grid = grid
        self.starts: Config = starts
        self.goals: Config = goals
        self.deadline: Deadline = (
            deadline if deadline is not None else Deadline(time_limit_ms)
        )
        # set hyper parameters
        self.flg_star: bool = flg_star
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)
        self.verbose = verbose
        return self._solve()

    def _solve(self) -> Configs:
        """Internal method performing the main LaCAM* search algorithm.

        Returns:
            Solution path as a list of configurations.
        """
        self.info(1, "start solving MAPF")
        # set cache, used for collision check
        self.occupied_from: np.ndarray = np.full(self.grid.shape, NO_AGENT, dtype=int)
        self.occupied_to: np.ndarray = np.full(self.grid.shape, NO_AGENT, dtype=int)

        # set distance tables
        self.dist_tables: list[DistTable] = [
            DistTable(self.grid, goal) for goal in self.goals
        ]

        # set search scheme
        OPEN: deque[HighLevelNode] = deque([])
        EXPLORED: dict[Config, HighLevelNode] = {}
        N_goal: HighLevelNode | None = None

        # set initial node
        Q_init = self.starts
        N_init = HighLevelNode(
            Q=Q_init, order=self.get_order(Q_init), h=self.get_h_value(Q_init)
        )
        OPEN.appendleft(N_init)
        EXPLORED[N_init.Q] = N_init

        # main loop
        while len(OPEN) > 0 and not self.deadline.is_expired:
            N: HighLevelNode = OPEN[0]

            # goal check
            if N_goal is None and N.Q == self.goals:
                N_goal = N
                self.info(1, f"initial solution found, cost={N_goal.g}")
                # no refinement -> terminate
                if not self.flg_star:
                    break

            # lower bound check
            if N_goal is not None and N_goal.g <= N.f:
                OPEN.popleft()
                continue

            # low-level search end
            if len(N.tree) == 0:
                OPEN.popleft()
                continue

            # low-level search
            C: LowLevelNode = N.tree.popleft()  # constraints
            if C.depth < self.num_agents:
                i = N.order[C.depth]
                v = N.Q[i]
                cands = [v] + get_neighbors(self.grid, v)
                self.rng.shuffle(cands)
                for u in cands:
                    N.tree.append(C.get_child(i, u))

            # generate the next configuration
            Q_to = self.configuration_generator(N, C)
            if Q_to is None:
                # invalid configuration
                continue
            elif Q_to in EXPLORED.keys():
                # known configuration
                N_known = EXPLORED[Q_to]
                N.neighbors.add(N_known)
                OPEN.appendleft(N_known)  # typically helpful
                # rewrite, Dijkstra update
                D = deque([N])
                while len(D) > 0:
                    N_from = D.popleft()
                    for N_to in N_from.neighbors:
                        g = N_from.g + self.get_edge_cost(N_from.Q, N_to.Q)
                        if g < N_to.g:
                            if N_goal is not None and N_to is N_goal:
                                self.info(2, f"cost update: {N_goal.g:4d} -> {g:4d}")
                            N_to.g = g
                            N_to.f = N_to.g + N_to.h
                            N_to.parent = N_from
                            D.append(N_to)
                            if N_goal is not None and N_to.f < N_goal.g:
                                OPEN.appendleft(N_to)
            else:
                # new configuration
                N_new = HighLevelNode(
                    Q=Q_to,
                    parent=N,
                    order=self.get_order(Q_to),
                    g=N.g + self.get_edge_cost(N.Q, Q_to),
                    h=self.get_h_value(Q_to),
                )
                N.neighbors.add(N_new)
                OPEN.appendleft(N_new)
                EXPLORED[Q_to] = N_new

        # categorize result
        if N_goal is not None and len(OPEN) == 0:
            self.info(1, f"reach optimal solution, cost={N_goal.g}")
        elif N_goal is not None:
            self.info(1, f"suboptimal solution, cost={N_goal.g}")
        elif len(OPEN) == 0:
            self.info(1, "detected unsolvable instance")
        else:
            self.info(1, "failure due to timeout")
        return self.backtrack(N_goal)

    @staticmethod
    def backtrack(_N: HighLevelNode | None) -> Configs:
        """Reconstruct solution path by following parent pointers.

        Args:
            _N: Goal node (or None if no solution found).

        Returns:
            List of configurations from start to goal. Returns empty list if _N is None.
        """
        configs: Configs = []
        N = _N
        while N is not None:
            configs.append(N.Q)
            N = N.parent
        configs.reverse()
        return configs

    def get_edge_cost(self, Q_from: Config, Q_to: Config) -> int:
        """Calculate the cost of transitioning between two configurations.

        Cost is the number of agents that are not at their goal or moved from their goal.

        Args:
            Q_from: Source configuration.
            Q_to: Destination configuration.

        Returns:
            Transition cost (sum of agents not staying at goal).
        """
        # e.g., \sum_i | not (Q_from[i] == Q_to[k] == g_i) |
        cost = 0
        for i in range(self.num_agents):
            if not (self.goals[i] == Q_from[i] == Q_to[i]):
                cost += 1
        return cost

    def get_h_value(self, Q: Config) -> int:
        """Calculate heuristic value (lower bound on remaining cost).

        Uses sum of individual shortest path distances to goals.

        Args:
            Q: Configuration to evaluate.

        Returns:
            Heuristic value (sum of distances to goals for all agents).
            Returns maximum int value if any agent cannot reach its goal.
        """
        # e.g., \sum_i dist(Q[i], g_i)
        cost = 0
        for agent_idx, loc in enumerate(Q):
            c = self.dist_tables[agent_idx].get(loc)
            # Note: DistTable.get() always returns int, no None check needed
            if c >= self.grid.size:
                return np.iinfo(np.int32).max
            cost += c
        return cost

    def get_order(self, Q: Config) -> list[int]:
        """Determine the order in which agents are assigned in low-level search.

        Agents are ordered by descending distance to goal (with random tie-breaking).
        This heuristic prioritizes agents that are farther from their goals.

        Args:
            Q: Configuration to determine agent ordering for.

        Returns:
            List of agent indices in processing order.
        """
        # e.g., by descending order of dist(Q[i], g_i)
        order = list(range(self.num_agents))
        self.rng.shuffle(order)
        order.sort(key=lambda i: self.dist_tables[i].get(Q[i]), reverse=True)
        return order

    def configuration_generator(
        self, N: HighLevelNode, C: LowLevelNode
    ) -> Config | None:
        """Generate a successor configuration from constraints.

        Uses the low-level node constraints to assign positions, then randomly
        assigns remaining agents. Checks for collisions during generation.

        Args:
            N: Current high-level node.
            C: Low-level constraint node specifying partial assignments.

        Returns:
            A valid successor configuration, or None if generation fails due to collisions.
        """
        Q_to = Config([NO_LOCATION for _ in range(self.num_agents)])

        # set constraints to Q_to
        for k in range(C.depth):
            Q_to[C.who[k]] = C.where[k]

        # generate configuration
        flg_success = True
        for i in range(self.num_agents):
            v_i_from = N.Q[i]
            self.occupied_from[v_i_from] = i

            # set next position by random choice when without constraint
            if Q_to[i] == NO_LOCATION:
                a = self.rng.choice(ACTIONS)
                v = (v_i_from[0] + a[0], v_i_from[1] + a[1])
                if is_valid_coord(self.grid, v):
                    Q_to[i] = v
                else:
                    flg_success = False
                    break

            v_i_to: Coord = Q_to[i]
            # check vertex collision
            if self.occupied_to[v_i_to] != NO_AGENT:
                flg_success = False
                break
            # check edge collision
            j = self.occupied_from[v_i_to]
            if j != NO_AGENT and j != i and Q_to[j] == v_i_from:
                flg_success = False
                break
            self.occupied_to[v_i_to] = i

        # cleanup cache used for collision checking
        for i in range(self.num_agents):
            v_i_from = N.Q[i]
            self.occupied_from[v_i_from] = NO_AGENT
            v_i_next = Q_to[i]
            if v_i_next != NO_LOCATION:
                self.occupied_to[v_i_next] = NO_AGENT

        return Q_to if flg_success else None

    def info(self, level: int, msg: str) -> None:
        """Log an informational message if verbosity level is sufficient.

        Args:
            level: Minimum verbosity level required to display this message.
            msg: Message to log.
        """
        if self.verbose < level:
            return
        logger.debug(f"{int(self.deadline.elapsed):4d}ms  {msg}")
