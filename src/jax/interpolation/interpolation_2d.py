import jax.numpy as jnp
import jax
from jax import vmap

from sl_jax.boundaries import boundaries


def interpolate_lin_2d(
    psi: jnp.ndarray,
    lx: jnp.ndarray,
    ly: jnp.ndarray,
    i_d: jnp.ndarray,
    j_d: jnp.ndarray,
    bcx_kind: int,
    bcy_kind: int,
    nx: int,
    ny: int,
):
    """Perform a 2d linear interpolation using JAX

    Args:
        psi (jnp.ndarray): field to interpolate
        lx (jnp.ndarray): x interpolation weights
        ly (jnp.ndarray): y interpolation weights
        i_d (jnp.ndarray): x departure indices
        j_d (jnp.ndarray): y departure indices
        bcx_kind (int): boundary condition type for x
        bcy_kind (int): boundary condition type for y
        nx (int): grid size in x
        ny (int): grid size in y

    Returns:
        jnp.ndarray: interpolated field
    """
    # Lagrange polynomials
    p0 = lambda l: 1 - l
    p1 = lambda l: l

    # Interpolation weights
    px = jnp.array([p0(lx), p1(lx)])
    py = jnp.array([p0(ly), p1(ly)])

    # Compute boundary indices
    id_0 = boundaries(i_d, nx, bcx_kind)
    id_p1 = boundaries(i_d + 1, nx, bcx_kind)
    
    jd_0 = boundaries(j_d, ny, bcy_kind)
    jd_p1 = boundaries(j_d + 1, ny, bcy_kind)

    # Gather values at interpolation stencil points
    psi_00 = psi[id_0, jd_0]
    psi_10 = psi[id_p1, jd_0]
    psi_01 = psi[id_0, jd_p1]
    psi_11 = psi[id_p1, jd_p1]

    # Interpolate in x direction
    psi_x0 = px[0] * psi_00 + px[1] * psi_10
    psi_x1 = px[0] * psi_01 + px[1] * psi_11

    # Interpolate in y direction
    psi_d = py[0] * psi_x0 + py[1] * psi_x1

    return psi_d


def interpolate_cub_2d(
    psi: jnp.ndarray,
    lx: jnp.ndarray,
    ly: jnp.ndarray,
    i_d: jnp.ndarray,
    j_d: jnp.ndarray,
    bcx_kind: int,
    bcy_kind: int,
    nx: int,
    ny: int,
):
    """Perform 2d cubic interpolation using JAX

    Args:
        psi (jnp.ndarray): field to interpolate
        lx (jnp.ndarray): x interpolation weights
        ly (jnp.ndarray): y interpolation weights
        i_d (jnp.ndarray): x departure indices
        j_d (jnp.ndarray): y departure indices
        bcx_kind (int): boundary condition type for x
        bcy_kind (int): boundary condition type for y
        nx (int): grid size in x
        ny (int): grid size in y

    Returns:
        jnp.ndarray: interpolated field
    """
    # Lagrange polynomials
    p_1 = lambda l: (1 / 6) * l * (l - 1) * (2 - l)
    p0 = lambda l: (1 - l**2) * (1 - l / 2)
    p1 = lambda l: (1 / 2) * l * (l + 1) * (2 - l)
    p2 = lambda l: (1 / 6) * l * (l**2 - 1)

    px = jnp.array([p_1(lx), p0(lx), p1(lx), p2(lx)])
    py = jnp.array([p_1(ly), p0(ly), p1(ly), p2(ly)])

    # Compute boundary indices
    id_m1 = boundaries(i_d - 1, nx, bcx_kind)
    id_0 = boundaries(i_d, nx, bcx_kind)
    id_p1 = boundaries(i_d + 1, nx, bcx_kind)
    id_p2 = boundaries(i_d + 2, nx, bcx_kind)
    
    jd_m1 = boundaries(j_d - 1, ny, bcy_kind)
    jd_0 = boundaries(j_d, ny, bcy_kind)
    jd_p1 = boundaries(j_d + 1, ny, bcy_kind)
    jd_p2 = boundaries(j_d + 2, ny, bcy_kind)

    # Gather values at all 16 stencil points
    psi_vals = jnp.array([
        psi[id_m1, jd_m1], psi[id_0, jd_m1], psi[id_p1, jd_m1], psi[id_p2, jd_m1],
        psi[id_m1, jd_0], psi[id_0, jd_0], psi[id_p1, jd_0], psi[id_p2, jd_0],
        psi[id_m1, jd_p1], psi[id_0, jd_p1], psi[id_p1, jd_p1], psi[id_p2, jd_p1],
        psi[id_m1, jd_p2], psi[id_0, jd_p2], psi[id_p1, jd_p2], psi[id_p2, jd_p2],
    ])

    # Interpolate in x direction (4 y-levels)
    psi_y = jnp.array([
        px[0] * psi_vals[0] + px[1] * psi_vals[1] + px[2] * psi_vals[2] + px[3] * psi_vals[3],
        px[0] * psi_vals[4] + px[1] * psi_vals[5] + px[2] * psi_vals[6] + px[3] * psi_vals[7],
        px[0] * psi_vals[8] + px[1] * psi_vals[9] + px[2] * psi_vals[10] + px[3] * psi_vals[11],
        px[0] * psi_vals[12] + px[1] * psi_vals[13] + px[2] * psi_vals[14] + px[3] * psi_vals[15],
    ])

    # Interpolate in y direction
    psi_d = py[0] * psi_y[0] + py[1] * psi_y[1] + py[2] * psi_y[2] + py[3] * psi_y[3]

    return psi_d


def max_interpolator_2d(
    psi: jnp.ndarray,
    i_d: jnp.ndarray,
    j_d: jnp.ndarray,
    bcx_kind: int,
    bcy_kind: int,
    nx: int,
    ny: int,
):
    """Compute maximum value in the 2x2 stencil around departure point

    Args:
        psi (jnp.ndarray): field
        i_d (jnp.ndarray): x departure indices
        j_d (jnp.ndarray): y departure indices
        bcx_kind (int): boundary condition type for x
        bcy_kind (int): boundary condition type for y
        nx (int): grid size in x
        ny (int): grid size in y

    Returns:
        jnp.ndarray: maximum values at each point
    """
    
    id_0 = boundaries(i_d, nx, bcx_kind)
    id_p1 = boundaries(i_d + 1, nx, bcx_kind)
    
    jd_0 = boundaries(j_d, ny, bcy_kind)
    jd_p1 = boundaries(j_d + 1, ny, bcy_kind)
    
    # Gather the 4 corner values
    vals = jnp.stack([
        psi[id_0, jd_0],
        psi[id_0, jd_p1],
        psi[id_p1, jd_0],
        psi[id_p1, jd_p1],
    ], axis=0)
    
    # Return maximum across the stencil
    psi_d = jnp.max(vals, axis=0)
    
    return psi_d


def min_interpolator_2d(
    psi: jnp.ndarray,
    i_d: jnp.ndarray,
    j_d: jnp.ndarray,
    bcx_kind: int,
    bcy_kind: int,
    nx: int,
    ny: int,
):
    """Compute minimum value in the 2x2 stencil around departure point

    Args:
        psi (jnp.ndarray): field
        i_d (jnp.ndarray): x departure indices
        j_d (jnp.ndarray): y departure indices
        bcx_kind (int): boundary condition type for x
        bcy_kind (int): boundary condition type for y
        nx (int): grid size in x
        ny (int): grid size in y

    Returns:
        jnp.ndarray: minimum values at each point
    """
    
    id_0 = boundaries(i_d, nx, bcx_kind)
    id_p1 = boundaries(i_d + 1, nx, bcx_kind)
    
    jd_0 = boundaries(j_d, ny, bcy_kind)
    jd_p1 = boundaries(j_d + 1, ny, bcy_kind)
    
    # Gather the 4 corner values
    vals = jnp.stack([
        psi[id_0, jd_0],
        psi[id_0, jd_p1],
        psi[id_p1, jd_0],
        psi[id_p1, jd_p1],
    ], axis=0)
    
    # Return minimum across the stencil
    psi_d = jnp.min(vals, axis=0)
    
    return psi_d
