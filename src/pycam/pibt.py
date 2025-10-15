"""
a toy PIBT implementation taken from
https://github.com/Kei18/pypibt
"""

import numpy as np

from .dist_table import DistTable
from .mapf_utils import Config, Coord, get_neighbors


class PIBT:
    def __init__(self, dist_tables: list[DistTable], seed: int = 0) -> None:
        self.N = len(dist_tables)
        assert self.N > 0
        self.dist_tables = dist_tables
        self.grid = self.dist_tables[0].grid

        # cache
        self.NIL = self.N  # meaning \bot
        self.NIL_COORD: Coord = self.grid.shape  # meaning \bot
        self.occupied_now = np.full(self.grid.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(self.grid.shape, self.NIL, dtype=int)

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

    def funcPIBT(self, Q_from: Config, Q_to: Config, i: int) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        C = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize
        C = sorted(C, key=lambda u: self.dist_tables[i].get(u))

        # vertex assignment
        for v in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False

    def step(
        self,
        Q_from: Config,
        Q_to: Config,
        order: list[int],
    ) -> bool:
        flg_success = True

        # setup
        for i, (v_i_from, v_i_to) in enumerate(zip(Q_from, Q_to)):
            self.occupied_now[v_i_from] = i
            if v_i_to != self.NIL_COORD:
                #  check vertex collision
                if self.occupied_nxt[v_i_to] != self.NIL:
                    flg_success = False
                    break
                # check edge collision
                j = self.occupied_now[v_i_to]
                if j != self.NIL and j != i and Q_to[j] == v_i_from:
                    flg_success = False
                    break
                self.occupied_nxt[v_i_to] = i

        # perform PIBT
        if flg_success:
            for i in order:
                if Q_to[i] == self.NIL_COORD:
                    flg_success = self.funcPIBT(Q_from, Q_to, i)
                    if not flg_success:
                        break

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            if q_to != self.NIL_COORD:
                self.occupied_nxt[q_to] = self.NIL

        return flg_success
