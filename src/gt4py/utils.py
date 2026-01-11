import gt4py.cartesian as gtscript
import numpy as np
from itertools import product

from sl_gt4py.gt4py_config import dtype, backend
from sl_gt4py.geometry.coordinates import Grid

@gtscript.function(backend)
def copya2b(
    a: gtscript.Field[dtype],
    b: gtscript.Field[dtype]
):
    b[0, 0, 0] = a[0, 0, 0]
    return b
        
@gtscript.stencil(backend)     
def backup(
    vx: gtscript.Field[dtype],
    vy: gtscript.Field[dtype],
    vx_e: gtscript.Field[dtype],
    vy_e: gtscript.Field[dtype],
    tracer: gtscript.Field[dtype],
    tracer_e: gtscript.Field[dtype]
):
    with computation(PARALLEL), interval(...):
        vx = copya2b(vx_e, vx)
        vy = copya2b(vy_e, vy)
        tracer = copya2b(tracer_e, tracer)
 

    
        
        
