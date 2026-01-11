from typing import Tuple
import jax.numpy as jnp
import jax
from jax import lax
import logging

from config import Config
from sl_jax.diagnostics import diagnostic_lipschitz
from sl_jax.filter import overshoot_filter, undershoot_filter
from sl_jax.interpolation.interpolation_2d import (
    interpolate_lin_2d,
    max_interpolator_2d,
    min_interpolator_2d,
)
from sl_jax.periodic_filters import (
    periodic_overshoot_filter,
    periodic_undershoot_filter,
)

logging.getLogger(__name__)


def dep_search_1d(
    i: jnp.ndarray, vx_e: jnp.ndarray, vx_tmp: jnp.ndarray, dx: jnp.ndarray, dth: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute departure point coordinate (1d) using JAX

    Args:
        i (jnp.ndarray): grid indices
        vx_e (jnp.ndarray): velocity at arrival point (t + dt)
        vx_tmp (jnp.ndarray): estimate of velocity at departure point
        dx (jnp.ndarray): grid spacing
        dth (float): half model time step

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            lx: adimensionned spacing of departure point from ref grid point
            i_d: indice of departure point on grid
    """

    # Displacement
    trajx = -dth * (vx_e[:,:,:] + vx_tmp[:,:,:]) / dx  # dth = dt / 2
    i_d = (i + jnp.floor(trajx)).astype(int)
    lx = trajx[:,:,:] - jnp.floor(trajx[:,:,:])

    return lx, i_d


def sl_init(
    vx_e: jnp.ndarray,
    vy_e: jnp.ndarray,
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    vx_p: jnp.ndarray,
    vy_p: jnp.ndarray,
    lsettls: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize draft velocities with either
    LSETTLS method : 2 fields for velocity (at t and t - dt)
    LNESN (not LSETTLS) method : 1 field for velocity (at t)

    Args:
        vx_e (jnp.ndarray): outlined velocity at t + dt on x
        vy_e (jnp.ndarray): outlined velocity at t + dt on y
        vx (jnp.ndarray): velocity at t on x
        vy (jnp.ndarray): velocity at t on y
        vx_p (jnp.ndarray): velocity at t - dt on x
        vy_p (jnp.ndarray): velocity at t - dt on y
        lsettls (bool, optional): LSETTLS or LNESC. Defaults to True.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: velocities at t and t + dt
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
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    vx_e: jnp.ndarray,
    vy_e: jnp.ndarray,
    tracer: jnp.ndarray,
    tracer_e: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Copy fields for next iteration.
    Ex : vx_e becomes vx at next model time step

    Args:
        vx (jnp.ndarray): x velocity
        vy (jnp.ndarray): y velocity
        vx_e (jnp.ndarray): ebauche vx
        vy_e (jnp.ndarray): ebauche vy
        tracer (jnp.ndarray): tracer field
        tracer_e (jnp.ndarray): ebauche at t + dt for tracer field

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: copy for fields
    """

    # Copy fields
    tracer = tracer_e.copy()
    vx = vx_e.copy()
    vy = vy_e.copy()

    return vx, vy, tracer


def lagrangian_search(
    config: Config,
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    vx_e: jnp.ndarray,
    vy_e: jnp.ndarray,
    nitmp: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Research departure point for a given grid and velocity field using JAX.
    Terminates on nitmp iterations. Optimized with lax.fori_loop.

    Args:
        config (Config): configuration object
        vx (jnp.ndarray): velocity field on x
        vy (jnp.ndarray): velocity field on y
        vx_e (jnp.ndarray): velocity at t+dt on x
        vy_e (jnp.ndarray): velocity at t+dt on y
        nitmp (int, optional): number of iterations. Defaults to 4.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: departure point coordinates and indices
    """

    def iteration_body(l, state):
        """Body function for fori_loop iteration."""
        vx_tmp, vy_tmp, lx, ly, i_d, j_d = state
        
        # Compute departure points
        lx, i_d = dep_search_1d(config.I, vx_e, vx_tmp, config.dx, config.dth)
        ly, j_d = dep_search_1d(config.J, vy_e, vy_tmp, config.dy, config.dth)

        # Diagnostic (computed but not used in state)
        lipschitz = diagnostic_lipschitz(
            vx_tmp, vy_tmp, config.dx, config.dy, config.dth
        )

        # Interpolation for fields
        vx_tmp = interpolate_lin_2d(
            vx, lx, ly, i_d, j_d, config.bcx_kind, config.bcy_kind, config.nx, config.ny
        )

        vy_tmp = interpolate_lin_2d(
            vy, lx, ly, i_d, j_d, config.bcx_kind, config.bcy_kind, config.nx, config.ny
        )
        
        return (vx_tmp, vy_tmp, lx, ly, i_d, j_d)

    # Initialize state
    vx_tmp = vx.copy()
    vy_tmp = vy.copy()
    lx = jnp.zeros_like(vx)
    ly = jnp.zeros_like(vy)
    i_d = jnp.zeros_like(vx, dtype=jnp.int32)
    j_d = jnp.zeros_like(vy, dtype=jnp.int32)
    
    init_state = (vx_tmp, vy_tmp, lx, ly, i_d, j_d)
    
    # Run iterations using lax.fori_loop for better performance
    final_state = lax.fori_loop(0, nitmp, iteration_body, init_state)
    
    _, _, lx, ly, i_d, j_d = final_state
    
    return lx, ly, i_d, j_d


def sl_xy(
    config: Config,
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    vx_e: jnp.ndarray,
    vy_e: jnp.ndarray,
    tracer: jnp.ndarray,
    tracer_e: jnp.ndarray,
    nitmp: int,
) -> jnp.ndarray:
    """Performs tracer advection with 2D semi lagrangian using JAX.
    1: search for departure point
    2: interpolate tracer field

    Args:
        config (Config): grid configuration
        vx (jnp.ndarray): velocity on x
        vy (jnp.ndarray): velocity on y
        vx_e (jnp.ndarray): velocity on x at t + dt
        vy_e (jnp.ndarray): velocity on y at t + dt
        tracer (jnp.ndarray): tracer field
        tracer_e (jnp.ndarray): tracer at t + dt (ebauche)
        nitmp (int): number of iterations for departure search

    Returns:
        jnp.ndarray: tracer outline (ebauche) at t + dt
    """

    # Semi-lagrangian search
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

    # Local max and min for filtering
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
