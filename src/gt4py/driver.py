import logging
import numpy as np
from gt4py.cartesian import gtscript

from config import Config
from sl_gt4py.gt4py_config import dtype, dtype_int
from sl_gt4py.sl_2D import sl_xy
from sl_gt4py.smilag_init import sl_init
from sl_gt4py.backup import backup
from sl_gt4py.blossey_velocity import blossey_velocity
from sl_gt4py.cfl import _cfl_1d

# Driver

# TODO: dace program
def sl_driver(
    config: Config,
    vx: gtscript.Field[dtype],
    vy: gtscript.Field[dtype],
    vx_e: gtscript.Field[dtype],
    vy_e: gtscript.Field[dtype],
    vx_p: gtscript.Field[dtype],
    vy_p: gtscript.Field[dtype],
    vx_tmp: gtscript.Field[dtype],
    vy_tmp: gtscript.Field[dtype],
    tracer: gtscript.Field[dtype],
    tracer_e: gtscript.Field[dtype],
    I: gtscript.Field[dtype],
    J: gtscript.Field[dtype], 
    I_d: gtscript.Field[dtype],
    J_d: gtscript.Field[dtype],
    lx: gtscript.Field[dtype],
    ly: gtscript.Field[dtype],
    lsettls: bool,
    model_starttime: float,
    model_endtime: float,
):
    """Driver for SL tracer advection in GT4Py.

    Args:
        config (Config): _description_
        vx (gtscript.Field[dtype]): _description_
        vy (gtscript.Field[dtype]): _description_
        vx_e (gtscript.Field[dtype]): _description_
        vy_e (gtscript.Field[dtype]): _description_
        vx_p (gtscript.Field[dtype]): _description_
        vy_p (gtscript.Field[dtype]): _description_
        vx_tmp (gtscript.Field[dtype]): _description_
        vy_tmp (gtscript.Field[dtype]): _description_
        tracer (gtscript.Field[dtype]): _description_
        tracer_e (gtscript.Field[dtype]): _description_
        I (_type_): _description_
        J (_type_): _description_
        I_d (_type_): _description_
        J_d (_type_): _description_
        lx (_type_): _description_
        ly (_type_): _description_
        lsettls (bool): _description_
        model_starttime (float): _description_
        model_endtime (float): _description_
    """
    tracer_ref = tracer.copy()

    # Advection
    sl_init(
        vx_e=vx_e, vy_e=vy_e, vx=vx, vy=vy, vx_p=vx_p, vy_p=vy_p, lsettls=lsettls
    )

    t = model_starttime
    jstep = 0
    
    # TODO dace loop
    while t < model_endtime:
        
        jstep += 1
        t += config.dt
        logging.info(f"Step : {jstep}")
        logging.info(f"Time : {100*t/(model_endtime - model_starttime):.02f}%")

        # TODO: shift blossey velocity to GT4Py stencil
        vx_2d, vy_2d = blossey_velocity(config.xcr, config.ycr, t, config.dx, config.dy)
        vx_e_2d, vy_e_2d = blossey_velocity(
            config.xcr, config.ycr, t + config.dt, config.dx, config.dy
        )
        

        # Estimations
        tracer_e = sl_xy(
            config=config,
            vx=vx,
            vy=vy,
            vx_e=vx_e,
            vy_e=vy_e,
            tracer=tracer,
            tracer_e=tracer_e,
            vx_tmp=vx_tmp,
            vy_tmp=vy_tmp,
            I=I,
            J=J,
            I_d=I_d,
            J_d=J_d,
            lx=lx,
            ly=ly,
            nitmp=4,
        )

        backup(tracer=tracer, tracer_e=tracer_e)

        # Diagnostics and outputs
        courant_xmax = np.max(_cfl_1d(vx_e, config.dx, config.dt))
        courant_ymax = np.max(_cfl_1d(vy_e, config.dy, config.dt))

        logging.info(f"Maximum courant number : {max(courant_xmax, courant_ymax):.02f}")