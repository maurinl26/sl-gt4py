# -*- coding: utf-8 -*-
"""
Defines the stretching of the levels in the vertical direction.

.. plot::

   from inspect import getmembers, isfunction
   import matplotlib.pyplot as plt
   import numpy as np

   import fvms.geometry.zstretch
   from fvms.model.config import Config

   # set parameters for plot
   nz = 81
   top = 25000.0
   dz_near_surface = 100.0

   # init config
   # TODO(tehrengruber): use default Config instead of unrelated yml file
   config = Config.from_file("../../config/thermal.yml", nz=nz, zmax=top)

   # plot all functions from this module
   fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
   for name, func in getmembers(fvms.geometry.zstretch, isfunction):
       if name == "none":
           zstr = func(config)
       elif name in ["tanh", "tanh_cst_dz", "sine", "cosine", "tan_upper", "tan_surface"]:
           zstr = func(config, top=top, dz_near_surface=dz_near_surface)
       else:
           continue

       ax0.plot(zstr, np.arange(nz), label=name, linestyle="--")
       ax1.plot(
           np.diff(zstr), 0.5 * (np.arange(nz)[1:] + np.arange(nz)[:-1]), linestyle="--"
       )
   ax0.legend()
   ax0.set(xlabel="Height [m]", ylabel="Model Level")
   ax1.set(xlim=(0, None), ylim=(0, nz - 1), xlabel="Level Thickness [m]")

Example of the different stretching functions for 81 vertical levels and a domain height of 25km.
All `zstretch` variants are plotted with `dz_near_surface = 100.0`.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve


def none(zc: np.ndarray, bottom: float, top: float) -> np.ndarray:
    return zc - bottom


def tanh(zc: np.ndarray, bottom: float, top: float, *, dz_near_surface: float) -> np.ndarray:

    depth = top - bottom
    zcn = zc - bottom

    def scale_func(s):
        return (
            s / np.tanh(s) * (1.0 - np.tanh(s) ** 2.0) - dz_near_surface * (zcn.size - 1.0) / depth
        )

    scale = fsolve(scale_func, 1.2)

    zstretch = depth * (1.0 + (np.tanh(scale * (zcn / depth - 1.0))) / (np.tanh(scale)))
    return zstretch


def tanh_cst_dz(zc: np.ndarray, bottom: float, top: float, *, dz_near_surface: float) -> np.ndarray:
    """
    Stretching with constant layer thickness at the surface and at the domain top

    This stretching function is based on the tanh function above with the condition of constant layer thickness
    at the surface and the domain top.
    """

    depth = top - bottom
    zcn = zc - bottom

    def scale_func(s):
        return (s / np.tanh(s) + 2.0 / 3.0 * s**2.0) * (
            1.0 - np.tanh(s) ** 2.0
        ) - dz_near_surface * (zcn.size - 1.0) / depth

    scale = fsolve(scale_func, 1.2)

    zstretch = depth * (
        1.0 + (np.tanh(scale * (zcn / depth - 1.0))) / (np.tanh(scale))
    ) - 2.0 / depth * scale**2.0 * (1.0 - np.tanh(scale) ** 2.0) * (
        -1.0 / 3.0 * zcn * depth + 0.5 * zcn**2.0 - 1.0 / 6.0 * zcn**3.0 / depth
    )
    return zstretch


def sine(zc: np.ndarray, bottom: float, top: float, *, dz_near_surface: float) -> np.ndarray:
    """
    Stretching function based on a wave-like perturbation of the level thickness.

    This stretching functions leads to a constant level thickness at the surface and at the model top.
    It is characterised by a steady increase in the layer thickness along almost the entire depths of the domain.
    """

    depth = top - bottom
    zcn = zc - bottom

    scale = (1.0 - dz_near_surface * (zcn.size - 1.0) / depth) / np.pi

    zstretch = zcn - scale * depth * np.sin(zcn / depth * np.pi)
    return zstretch


def cosine(zc: np.ndarray, bottom: float, top: float, *, dz_near_surface: float) -> np.ndarray:
    """
    Stretching function based on a wave-like perturbation of the level thickness.

    This stretching functions leads to a constant level thickness at the surface and at the model top.
    It is characterised by a shorter range of steady increase in the layer thickness (compared to 'sine' above).
    """

    depth = top - bottom
    zcn = zc - bottom

    scale = 1.0 - (dz_near_surface * (zcn.size - 1.0) / depth)

    zstretch = (
        (1.0 - scale) * zcn
        + scale * zcn**2 / depth
        + scale * depth / (2.0 * np.pi**2) * (np.cos(2.0 * np.pi * zcn / depth) - 1.0)
    )
    return zstretch


def tan_upper(zc: np.ndarray, bottom: float, top: float, *, dz_near_surface: float) -> np.ndarray:
    """
    Stretching function with rather small maximum layer thickness but many layers with this thickness.

    This stretching functions leads to a constant level thickness at the surface and at the model top.
    It is characterized by a few levels with almost constant dz and rapid increase to a moderate large dz.
    This layer thickness at upper levels stays constant for about the upper half of the model levels.
    """

    depth = top - bottom
    zcn = zc - bottom

    scale = 1.0 - (dz_near_surface * (zcn.size - 1.0) / depth)

    zstretch = (
        scale * depth
        + (1.0 - scale) * zcn
        + scale * depth * np.tanh(np.tan((zcn / depth - 1.0) * 0.5 * np.pi))
    )
    return zstretch


def tan_surface(zc: np.ndarray, bottom: float, top: float, *, dz_near_surface: float) -> np.ndarray:
    """
    Stretching function with many levels with small level thickness close to the surface.

    This stretching functions leads to a constant level thickness at the surface and at the model top.
    As a consequence of the many levels with small thickness, the layer thickness increases
    rapidly and to large values above.
    """

    depth = top - bottom
    zcn = zc - bottom

    scale = 2.0 * (1.0 - (dz_near_surface * (zcn.size - 1) / depth)) / (np.pi - 2.0)

    zstretch = (1.0 + scale) * zcn - scale * depth * np.tanh(np.tan(zcn / depth * 0.5 * np.pi))
    return zstretch
