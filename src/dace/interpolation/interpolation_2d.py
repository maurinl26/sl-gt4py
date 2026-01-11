import numpy as np
import dace
from sl_dace.dims import I, J, K, H

# to dace sdfg
def interpolate_lin_2d(
    lx: dace.float32[I, J, K],
    ly: dace.float32[I, J, K],
    i_dep: dace.int32[I + H, J + H, K],
    j_dep: dace.int32[I + H, J + H, K],
    psi_dep: dace.float32[I + H, J + H, K],
    psi: dace.float32[I, J, K],
):
    """Perform a 2d linear interpolation on a regular horizontal plane.

    Note: H stands for the size of the halo (corresponding to max CFL)

    Args:
        psi (np.ndarray): field to interpolate
        lx (np.ndarray): fractional shift from departure point on axis x
        ly (np.ndarray): fractional shift from departure point on axis y
        i_dep (np.ndarray): index of departure point on axis x
        j_dep (np.ndarray): index of departure point on axis y
    """

    # Lookup
    for i, j, k in dace.map[0:I, 0:J, 0:K]:

            wx = lx[i, j, k]
            wy = ly[i, j, k]

            lower_left = psi_dep[i_dep[i, j, k], j_dep[i, j, k], k]
            lower_right = psi_dep[i_dep[i, j, k] + 1, j_dep[i, j, k], k]
            upper_left = psi_dep[i_dep[i, j, k], j_dep[i, j, k] + 1, k]
            upper_right = psi_dep[i_dep[i, j, k] + 1, j_dep[i, j, k] + 1, k]

            first_line = (
                    (1 - wx ) * lower_left
                    + wx * lower_right
            )

            second_line = (
                    (1 - wx) * upper_left
                    + wx * upper_right
            )

            psi[i, j, k] = (1 - wy) * first_line + wy * second_line


