from gt4py.cartesian import gtscript

from sl_gt4py.gt4py_config import backend, backend_opts, dtype


@gtscript.stencil(backend=backend, **backend_opts)
def copy(
    vx: gtscript.Field[dtype],
    vx_tmp: gtscript.Field[dtype], 
    vy: gtscript.Field[dtype],
    vy_tmp: gtscript.Field[dtype],
):
    """Copy velocity fields into temporary fields.

    Args:
        vx (gtscript.Field[dtype]): velocity on x 
        vx_tmp (gtscript.Field[dtype]): temporary velocity on x
        vy (gtscript.Field[dtype]): velocity on y
        vy_tmp (gtscript.Field[dtype]): temporary velocity on y
    """
    with computation(PARALLEL), interval(...):
        vx_tmp[0, 0, 0] = vx[0, 0, 0]
        vy_tmp[0, 0, 0] = vy[0, 0, 0]