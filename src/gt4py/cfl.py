from gt4py.cartesian import gtscript
from sl_gt4py.gt4py_config import backend, backend_opts, dtype

@gtscript.stencil(backend, **backend_opts)
def _cfl_1d(
    u: gtscript.Field[dtype],
    dt: dtype,
    dx: dtype
):
    with computation(PARALLEL), interval(...):
        cfl = u * dt / dx