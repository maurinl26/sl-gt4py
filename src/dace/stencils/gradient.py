from gt4py.cartesian.gtscript import computation, interval, PARALLEL, Field, IJK

def c2_x(f: Field[IJK, float], df_dx: Field[IJK, float], dx: float):

    with computation(PARALLEL), interval(...):
        df_dx = (f[-1, 0, 0] - f[-1, 0, 0]) / (2 * dx)


def c2_y(f: Field[IJK, float], df_dy: Field[IJK, float], dy: float):

    with computation(PARALLEL), interval(...):
        df_dy = (f[0, -1, 0] - f[0, 1, 0]) / (2 * dy)
