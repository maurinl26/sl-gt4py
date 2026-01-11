from gt4py.cartesian import gtscript

from sl_gt4py.gt4py_config import backend, backend_opts, dtype


@gtscript.stencil(backend=backend, **backend_opts)
def backup(
    tracer: gtscript.Field[dtype],
    tracer_e: gtscript.Field[dtype],
):
    """Copy fields for next iteration.
    Ex : vx_e becomes vx at next model time step

    Args:
        vx (gtscript.Field[dtype]): x velocity
        vy (gtscript.Field[dtype]): y velocity
        vx_e (gtscript.Field[dtype]): ebauche vx
        vy_e (gtscript.Field[dtype]): ebauche vy
        tracer (gtscript.Field[dtype]): tracer field
        tracer_e (gtscript.Field[dtype]): ebauche at t + dt for tracer field

    Returns:
        Tuple[gtscript.Field[dtype]]: copy for fields
    """
    
    with computation(PARALLEL), interval(...):

        # Copie des champs
        tracer[0, 0, 0] = tracer_e[0, 0, 0] 

