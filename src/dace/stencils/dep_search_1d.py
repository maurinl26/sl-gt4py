from gt4py.cartesian.gtscript import (
    stencil,
    Field,
    IJK,
    floor,
    computation,
    PARALLEL,
    interval,
)
import numpy as np

def dep_search_1d(
    vx_e: Field[IJK, np.float32],
    vx_tmp: Field[IJK, np.float32],
    i_a: Field[IJK, np.int32],
    i_d: Field[IJK, np.int32],
    lx: Field[IJK, np.float32],
    dx: np.float32,
    dth: np.float32,
):
    with computation(PARALLEL), interval(...):
        trajx = -dth * (vx_e + vx_tmp) / dx  # dth = dt / 2
        i_d = i_a + floor(trajx)
        lx = trajx - floor(trajx)
