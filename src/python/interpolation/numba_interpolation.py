import numba

from sl_gt4py.interpolation.interpolation import interpolate_lin_2d, interpolate_cub_2d

numba_interpolate_lin_2d = numba.jit(interpolate_lin_2d)
numba_interpolate_cub_2d = numba.jit(interpolate_cub_2d)