import logging

from gt4py.cartesian import gtscript

from sl_gt4py.gt4py_config import backend, backend_opts, dtype

logging.getLogger(__name__)

@gtscript.stencil(backend=backend, **backend_opts)
def sl_init(
    vx_e: gtscript.Field[dtype],
    vy_e: gtscript.Field[dtype],
    vx: gtscript.Field[dtype],
    vy: gtscript.Field[dtype],
    vx_p: gtscript.Field[dtype],
    vy_p: gtscript.Field[dtype],
    lsettls: bool = True,
):
    """Initialize draft velocities with either 
    LSETTLS method : 2 fields for velocity (at t and t - dt)
    LNESN (not LSETTLS) method : 1 field for velocity (at t)

    Args:
        vx_e (gtscript.Field[dtype]): outlined velocity at t + dt on x
        vy_e (gtscript.Field[dtype]): outlined velocity at t + dt on y
        vx (gtscript.Field[dtype]): velocity at t on x
        vy (gtscript.Field[dtype]): velocity at t on y
        vx_p (gtscript.Field[dtype]): velocity at t - dt on x
        vy_p (gtscript.Field[dtype]): velcoity at t - dt on y 
        lsettls (bool, optional): LSETTLS or LNESC. Defaults to True.

    Returns:
        Tuple[gtscript.Field[dtype]]: velocities at t and t + dt
    """
    
    # LSETTLS
    with computation(PARALLEL), interval(...):
        if lsettls:
            vx_e[0, 0, 0] = vx[0, 0, 0]
            vy_e[0, 0, 0] = vy[0, 0, 0]

            vx[0, 0, 0] = 2 * vx[0, 0, 0] - vx_p[0, 0, 0]
            vy[0, 0, 0] = 2 * vy[0, 0, 0] - vy_p[0, 0, 0]
            
        else:
                
            vx_e[0, 0, 0] = vx[0, 0, 0]
            vy_e[0, 0, 0] = vy[0, 0, 0]
                
                
                