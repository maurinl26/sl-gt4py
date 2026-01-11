from gt4py.cartesian import gtscript
from sl_gt4py.gt4py_config import dtype

@gtscript.function
def floor(x: dtype):
    floor_x = x * 10 // 10
    return floor_x