from typing import Tuple
import numpy as np
import logging
import dace
from gt4py.cartesian import gtscript

from config import Config
from sl_gt4py.copy import copy
from sl_gt4py.departure_search import _dep_search_1d
from sl_gt4py.gt4py_config import dtype, dtype_int
from sl_gt4py.interpolation.dace_interpolation import dace_interpolate_lin_2d

logging.getLogger(__name__)

NITMP = dace.symbol("NITMP")

# ELARCHE
@dace.program
def lagrangian_search(
    config: Config,
    vx: gtscript.Field[dtype],
    vy: gtscript.Field[dtype],
    vx_e: gtscript.Field[dtype],
    vy_e: gtscript.Field[dtype],
    I: gtscript.Field[dtype],
    J: gtscript.Field[dtype],
    I_d: gtscript.Field[dtype],
    J_d: gtscript.Field[dtype],
    lx: gtscript.Field[dtype],
    ly: gtscript.Field[dtype],
    vx_tmp: gtscript.Field[dtype],
    vy_tmp: gtscript.Field[dtype],
    nitmp: int = 4,
):
    """Research departure point for a given grid and velocity field.
    Terminates on nsiter iterations.

    Args:
        x (gtscript.Field[dtype]): grid of arrival points
        v (gtscript.Field[dtype]): velocity fields
        nsiter (int, optional): number of iterations. Defaults to 10.

    Returns:
        gtscript.Field[dtype]: departure point
    """    
    copy(vx, vx_tmp, vy, vy_tmp)

    for l in range(NITMP):
                
        _dep_search_1d(I, vx_e, vx_tmp, lx, I_d, config.dx, config.dth)
        _dep_search_1d(J, vy_e, vy_tmp, ly, J_d, config.dy, config.dth)

        # Hors stencil
        ####### Interpolation for fields ########
        vx_tmp = dace_interpolate_lin_2d(
            vx,
            lx, 
            ly,
            I_d, 
            J_d,
            config.bcx_kind,
            config.bcy_kind,
            config.nx, 
            config.ny,
            config.nz
        )
        
        vy_tmp = dace_interpolate_lin_2d(
            vy,
            lx, 
            ly,
            I_d, 
            J_d,
            config.bcx_kind,
            config.bcy_kind,
            config.nx, 
            config.ny,
            config.nz
        )

    return lx, ly, I_d, J_d


def sl_xy(
    config: Config,
    vx: gtscript.Field[dtype],
    vy: gtscript.Field[dtype],
    vx_e: gtscript.Field[dtype],
    vy_e: gtscript.Field[dtype],
    tracer: gtscript.Field[dtype],
    tracer_e: gtscript.Field[dtype],
    I: gtscript.Field[dtype_int], 
    J: gtscript.Field[dtype_int], 
    I_d: gtscript.Field[dtype_int],
    J_d: gtscript.Field[dtype_int],
    lx: gtscript.Field[dtype],
    ly: gtscript.Field[dtype],
    vx_tmp: gtscript.Field[dtype],
    vy_tmp: gtscript.Field[dtype],
    nitmp: int,
    filter: bool = True,
):
    """Performs tracer advection with 2D semi lagrangian.
    1: search for departure point
    2: interpolate tracer field

    Args:
        config (Config): grid configuration
        vx (gtscript.Field[dtype]): velocity on x 
        vy (gtscript.Field[dtype]): velocity on y 
        vx_e (gtscript.Field[dtype]): velocity on x at t + dt
        vy_e (gtscript.Field[dtype]): velocity on y at t + dt
        tracer (gtscript.Field[dtype]): tracer field 
        tracer_e (gtscript.Field[dtype]): tracer at t + dt (ebauche)
        interpolation_function (callable): linear or cubic interpolation
        nitmp (int): number of iterations for departure search
        
    Returns:
        gtscript.Field[dtype]: tracer outline (ebauche) at t + dt 
    """
        
    # Recherche semi lag
    lx_d, ly_d, i_d, j_d = lagrangian_search(
        config=config,
        vx_e=vx_e,
        vy_e=vy_e,
        vx=vx,
        vy=vy,
        I=I,
        J=J,
        I_d=I_d,
        J_d=J_d,
        lx=lx,
        ly=ly,
        vx_tmp=vx_tmp,
        vy_tmp=vy_tmp,
        nitmp=nitmp,
    )
    

    # Interpolate
    tracer_e = dace_interpolate_lin_2d(
        tracer,
        lx_d, 
        ly_d,
        i_d, 
        j_d,
        config.bcx_kind,
        config.bcy_kind,
        config.nx, 
        config.ny,
        config.nz
    )
        
    return tracer_e



