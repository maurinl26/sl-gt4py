from itertools import repeat
from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import sqrt, log, cos, atan, I, J, K
from gt4py.storage import empty
from sl_gt4py.gt4py_config import backend, backend_opts, dtype, dtype_int
from sl_gt4py.config import Config
from sl_gt4py.utils.storage import managed_temporary

def blossey_velocity(
    xcr: gtscript.Field[dtype], 
    ycr: gtscript.Field[dtype], 
    u: gtscript.Field[dtype],
    v: gtscript.Field[dtype],
    dx: dtype, 
    dy: dtype,
    mt: dtype, 
    config: Config
    ):
    
    with managed_temporary(config.indices, *repeat((I, J, K), 1)) as (stf):

        stf = _blossey_stf(xcr, ycr, mt)
        u, v = _stream_function_xy(stf, dx, dy)

    return u, v

def init_blossey(
    xcr: gtscript.Field[dtype],
    ycr: gtscript.Field[dtype],
    mt: dtype,
    dt: dtype, 
    dx: dtype, 
    dy: dtype, 
    nx: dtype_int, 
    ny: dtype_int):
    # Vitesses
    vx, vy = blossey_velocity(xcr, ycr, mt, dx, dy)
    vx_p, vy_p = blossey_velocity(xcr, ycr, mt - dt, dx, dy)
    vx_e, vy_e = empty((nx, ny), aligned_index=(0, 0)), empty((nx, ny), aligned_index=(0, 0))

    return vx, vy, vx_p, vy_p, vx_e, vy_e

@gtscript.stencil(backend, **backend_opts)
def _blossey_stf(
    xcr: gtscript.Field[dtype],
    ycr: gtscript.Field[dtype],
    mt: gtscript.Field[dtype],
    stf: gtscript.Field[dtype]
):
    
    with computation(PARALLEL), interval(...):
        rsq = (xcr - 0.5) ** 2.0 + (ycr - 0.5) ** 2.0

        term1 = 0.5 * rsq
        term2 = log(1.0 - 16.0 * rsq + 256.0 * rsq**2.0) / 96.0
        term3 = -log(1.0 + 16.0 * rsq) / 48.0
        term4 = -sqrt(3.0) * atan((-1.0 + 32.0 * rsq) / sqrt(3.0)) / 48.0

        stf = (
        4.0
        * 3.1416
        * (term1 + cos(2.0 * 3.1416 * mt) * (term1 + term2 + term3 + term4))
    )

        
@gtscript.stencil(backend, **backend_opts)
def _stream_function_xy(stf: gtscript.Field[dtype], 
                        u: gtscript.Field[dtype],
                        v: gtscript.Field[dtype],
                        dx: dtype, 
                        dy: dtype):
    """Computes velocity components based on stream function

    Args:
        stf (np.ndarray): stream function

    Returns:
        Tuple[np.ndarray]: u (velocity on x axis), v (velocity on y axis)
    """
    
    with computation(PARALLEL), interval(...):
    
        # TODO : implementation du gradient a la main
        dstf_dx = c2_gradient_x(stf, dx)
        dstf_dy = c2_gradient_y(stf, dy, axis=1)
        u = dstf_dy
        v = - dstf_dx

    return u, v


@gtscript.function
def c2_gradient_x(psi: gtscript.Field, dx: dtype):
    return (psi[0, 1] - psi[0, -1]) / (2 * dx)

@gtscript.function
def c2_gradient_e(psi: gtscript.Field, dx: dtype):
    return (psi[0, 1] - psi[0, 0]) / dx

@gtscript.function
def c2_gradient_w(psi: gtscript.Field, dx: dtype):
    return (psi[0, 0] - psi[0, -1]) / dx

@gtscript.function
def c2_gradient_y(psi: gtscript.Field, dy: dtype):
    return (psi[1, 0] - psi[-1, 0]) / (2 * dy)

@gtscript.function
def c2_gradient_s(psi: gtscript.Field, dy: dtype):
    return (psi[0, 1] - psi[0, 0]) / dy

@gtscript.function
def c2_gradient_n(psi: gtscript.Field, dy: dtype):
    return (psi[0, 0] - psi[0, -1]) / dy