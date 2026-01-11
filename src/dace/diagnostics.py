import numpy as np
import dace
from gt4py.cartesian.gtscript import stencil
from typing import Tuple

from sl_dace.stencils.gradient import c2_x, c2_y


class LipschitzDiag:

    def __init__(self, grid: Tuple[int]):

        # todo: a grid class which takes
        self.grid = grid

        self.c2_x = stencil(definition=c2_x, name="c2_x")
        self.c2_y = stencil(definition=c2_y, name="c2_y")

    def __call__(self,
                 u,
                 v,
                 du_dx,
                 du_dy,
                 dv_dx,
                 dv_dy,
                 dx,
                 dy,
                 dth):

        self.c2_x(u, du_dx, dx, origin=(1, 0, 0), domain=self.grid)
        self.c2_x(v, dv_dx, dx, origin=(1, 0, 0), domain=self.grid)

        self.c2_y(u, du_dy, dy, origin=(0, 1, 0), domain=self.grid)
        self.c2_y(v, dv_dy, dy, origin=(0, 1, 0), domain=self.grid)

        # dace.reduce for performances
        return dth * np.max(np.maximum(du_dx, du_dy), np.maximum(dv_dx, dv_dy))

