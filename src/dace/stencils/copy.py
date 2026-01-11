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

def copy(
    vx_tmp: Field[IJK, np.float32],
    vy_tmp: Field[IJK, np.float32],
    vx: Field[IJK, np.float32],
    vy: Field[IJK, np.float32]
):

    with computation(PARALLEL), interval(...):
        vx_tmp = vx
        vy_tmp = vy