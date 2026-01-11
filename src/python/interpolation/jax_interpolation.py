import jax

from sl_gt4py.interpolation.interpolation import interpolate_cub_2d, interpolate_lin_2d

jax_interpolate_lin_2d = jax.jit(interpolate_lin_2d)
jax_interpolate_cub_2d = jax.jit(interpolate_cub_2d)