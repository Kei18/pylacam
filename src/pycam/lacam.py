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
NO_LOCATION: Coord = (np.iinfo(np.int32).max, np.iinfo(np.int32).max)
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]  # d_y, d_x


@dataclass
class LowLevelNode:
    who: list[int] = field(default_factory=lambda: [])
    where: list[Coord] = field(default_factory=lambda: [])
    depth: int = 0

    def get_child(self, who: int, where: Coord) -> LowLevelNode:
        return LowLevelNode(
            who=self.who + [who],
            where=self.where + [where],
            depth=self.depth + 1,
        )


@dataclass
class HighLevelNode:
    Q: Config
    order: list[int]
    parent: HighLevelNode | None = None
    tree: deque[LowLevelNode] = field(default_factory=lambda: deque([LowLevelNode()]))
    g: int = 0
    h: int = 0
    f: int = g + h
    neighbors: set[HighLevelNode] = field(default_factory=lambda: set())

    def __eq__(self, other) -> bool:
        if isinstance(other, HighLevelNode):
            return self.Q == other.Q
        return False

    def __hash__(self) -> int:
        return self.Q.__hash__()


class LaCAM:
    def __init__(self) -> None:
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
        configs: Configs = []
        N = _N
        while N is not None:
            configs.append(N.Q)
            N = N.parent
        configs.reverse()
        return configs

    def get_edge_cost(self, Q_from: Config, Q_to: Config) -> int:
        # e.g., \sum_i | not (Q_from[i] == Q_to[k] == g_i) |
        cost = 0
        for i in range(self.num_agents):
            if not (self.goals[i] == Q_from[i] == Q_to[i]):
                cost += 1
        return cost

    def get_h_value(self, Q: Config) -> int:
        # e.g., \sum_i dist(Q[i], g_i)
        cost = 0
        for agent_idx, loc in enumerate(Q):
            c = self.dist_tables[agent_idx].get(loc)
            if c is None:
                return np.iinfo(np.int32).max
            cost += c
        return cost

    def get_order(self, Q: Config) -> list[int]:
        # e.g., by descending order of dist(Q[i], g_i)
        order = list(range(self.num_agents))
        self.rng.shuffle(order)
        order.sort(key=lambda i: self.dist_tables[i].get(Q[i]), reverse=True)
        return order

    def configuration_generator(
        self, N: HighLevelNode, C: LowLevelNode
    ) -> Config | None:
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
        if self.verbose < level:
            return
        logger.debug(f"{int(self.deadline.elapsed):4d}ms  {msg}")
