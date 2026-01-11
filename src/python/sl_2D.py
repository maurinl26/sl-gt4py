from typing import Tuple
import numpy as np
import logging

from config import Config
from sl_python.diagnostics import diagnostic_lipschitz
from sl_python.filter import overshoot_filter, undershoot_filter
from sl_python.interpolation.interpolation_2d import (
    interpolate_lin_2d,
    max_interpolator_2d,
    min_interpolator_2d,
)
from sl_python.periodic_filters import (
    periodic_overshoot_filter,
    periodic_undershoot_filter,
)

logging.getLogger(__name__)


def dep_search_1d(
    i: np.ndarray, vx_e: np.ndarray, vx_tmp: np.ndarray, dx: np.ndarray, dth: float
) -> Tuple[np.ndarray]:
    """Compute departure point coordinate (1d)

    Args:
        I (np.ndarray): _description_
        vx_e (np.ndarray): velocity at arrival point (t + dt)
        vx_tmp (np.ndarray): estimate of velocity at departure point
        dx (np.ndarray): grid spacing
        dth (float): half model time step

    Returns:
        Tuple[np.ndarray]:
            i_d: indice of departure point on grid
            lx: adimensionned spacing of departure point from ref grid point
    """

    # Deplacement
    trajx = -dth * (vx_e[:,:,:] + vx_tmp[:,:,:]) / dx  # dth = dt / 2
    i_d = (i + np.floor(trajx)).astype(int)
    lx = trajx[:,:,:] - np.floor(trajx[:,:,:])

    return lx, i_d

def sl_init(
    vx_e: np.ndarray,
    vy_e: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vx_p: np.ndarray,
    vy_p: np.ndarray,
    lsettls: bool = True,
) -> Tuple[np.ndarray]:
    """Initialize draft velocities with either
    LSETTLS method : 2 fields for velocity (at t and t - dt)
    LNESN (not LSETTLS) method : 1 field for velocity (at t)

    Args:
        vx_e (np.ndarray): outlined velocity at t + dt on x
        vy_e (np.ndarray): outlined velocity at t + dt on y
        vx (np.ndarray): velocity at t on x
        vy (np.ndarray): velocity at t on y
        vx_p (np.ndarray): velocity at t - dt on x
        vy_p (np.ndarray): velcoity at t - dt on y
        lsettls (bool, optional): LSETTLS or LNESC. Defaults to True.

    Returns:
        Tuple[np.ndarray]: velocities at t and t + dt
    """
    # LSETTLS
    if lsettls:
        vx_e = vx.copy()
        vy_e = vy.copy()

        vx = 2 * vx - vx_p
        vy = 2 * vy - vy_p

    # LNESC
    else:
        vx_e = vx.copy()
        vy_e = vy.copy()

    return vx, vy, vx_e, vy_e

def backup(
    vx: np.ndarray,
    vy: np.ndarray,
    vx_e: np.ndarray,
    vy_e: np.ndarray,
    tracer: np.ndarray,
    tracer_e: np.ndarray,
) -> Tuple[np.ndarray]:
    """Copy fields for next iteration.
    Ex : vx_e becomes vx at next model time step

    Args:
        vx (np.ndarray): x velocity
        vy (np.ndarray): y velocity
        vx_e (np.ndarray): ebauche vx
        vy_e (np.ndarray): ebauche vy
        tracer (np.ndarray): tracer field
        tracer_e (np.ndarray): ebauche at t + dt for tracer field

    Returns:
        Tuple[np.ndarray]: copy for fields
    """

    # Copie des champs
    tracer = tracer_e.copy()
    vx = vx_e.copy()
    vy = vy_e.copy()

    return vx, vy, tracer


# ELARCHE
def lagrangian_search(
    config: Config,
    vx: np.ndarray,
    vy: np.ndarray,
    vx_e: np.ndarray,
    vy_e: np.ndarray,
    nitmp: int = 4,
) -> Tuple[np.ndarray]:
    """Research departure point for a given grid and velocity field.
    Terminates on nsiter iterations.

    Args:
        x (np.ndarray): grid of arrival points
        v (np.ndarray): velocity fields
        nsiter (int, optional): number of iterations. Defaults to 10.

    Returns:
        np.ndarray: departure point
    """

    vx_tmp = vx.copy()
    vy_tmp = vy.copy()

    # Array declaration
    for l in range(nitmp):
        lx, i_d = dep_search_1d(config.I, vx_e, vx_tmp, config.dx, config.dth)
        ly, j_d = dep_search_1d(config.J, vy_e, vy_tmp, config.dy, config.dth)

        lipschitz = diagnostic_lipschitz(
            vx_tmp, vy_tmp, config.dx, config.dy, config.dth
        )

        ####### Interpolation for fields ########
        vx_tmp = interpolate_lin_2d(
            vx, lx, ly, i_d, j_d, config.bcx_kind, config.bcy_kind, config.nx, config.ny
        )

        vy_tmp = interpolate_lin_2d(
            vy, lx, ly, i_d, j_d, config.bcx_kind, config.bcy_kind, config.nx, config.ny
        )

    return lx, ly, i_d, j_d

def sl_xy(
    config: Config,
    vx: np.ndarray,
    vy: np.ndarray,
    vx_e: np.ndarray,
    vy_e: np.ndarray,
    tracer: np.ndarray,
    tracer_e: np.ndarray,
    nitmp: int,
) -> np.ndarray:
    """Performs tracer advection with 2D semi lagrangian.
    1: search for departure point
    2: interpolate tracer field

    Args:
        config (Config): grid configuration
        vx (np.ndarray): velocity on x
        vy (np.ndarray): velocity on y
        vx_e (np.ndarray): velocity on x at t + dt
        vy_e (np.ndarray): velocity on y at t + dt
        tracer (np.ndarray): tracer field
        tracer_e (np.ndarray): tracer at t + dt (ebauche)
        nitmp (int): number of iterations for departure search

    Returns:
        np.ndarray: tracer outline (ebauche) at t + dt
    """

    # Recherche semi lag
    lx_d, ly_d, i_d, j_d = lagrangian_search(
        config=config,
        vx_e=vx_e,
        vy_e=vy_e,
        vx=vx,
        vy=vy,
        nitmp=nitmp,
    )

    # Interpolate
    tracer_e = interpolate_lin_2d(
        tracer,
        lx_d,
        ly_d,
        i_d,
        j_d,
        config.bcx_kind,
        config.bcy_kind,
        config.nx,
        config.ny,
    )

    # Max et min locaux pour filtage
    tracer_sup = max_interpolator_2d(
        tracer, i_d, j_d, config.bcx_kind, config.bcy_kind, config.nx, config.ny
    )

    tracer_inf = min_interpolator_2d(
        tracer, i_d, j_d, config.bcx_kind, config.bcy_kind, config.nx, config.ny
    )

    if config.filter:
        if config.bcx_kind == 1 and config.bcy_kind == 1:
            tracer_e = periodic_overshoot_filter(
                tracer_e, tracer_sup, config.nx, config.ny
            )
            tracer_e = periodic_undershoot_filter(
                tracer_e, tracer_inf, config.nx, config.ny
            )

        if config.bcx_kind == 0 and config.bcy_kind == 0:
            tracer_e = overshoot_filter(tracer_e, tracer_sup, config.nx, config.ny)
            tracer_e = undershoot_filter(tracer_e, tracer_inf, config.nx, config.ny)

    return tracer_e

