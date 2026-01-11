import logging

from config import Config
from sl_dace.diagnostics import diagnostic_lipschitz
from sl_dace.interpolation.interpolation_2d import (
    interpolate_lin_2d,
)
from sl_dace.elarche import Elarche
from sl_dace.dims import I, J, K
import dace

logging.getLogger(__name__)


class SmiLagXY:

    def __init__(self, grid: Tuple[int], nitmp: int = 4):
        self.nitmp = nitmp
        self.grid = grid

        self.elarche = Elarche(self.grid)

    def __call__(self):

        lx_d, ly_d, i_d, j_d = self.elarche(
            I=I,
            J=J,
            dx=dx,
            dy=dy,
            dth=dth,
            bcx_kind=bcx_kind,
            bcy_kind=bcy_kind,
            vx_e=vx_e,
            vy_e=vy_e,
            vx=vx,
            vy=vy,
            nitmp=nitmp,
        )

        # Interpolate
        tracer_e = self.elarche.d_interpolate_lin_2d(
            tracer,
            lx_d,
            ly_d,
            i_d,
            j_d,
            I,
            J,
        )

