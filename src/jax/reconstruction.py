"""FFSL (Flux-Form Semi-Lagrangian) Reconstruction with PPM.

This module implements Piecewise Parabolic Method (PPM) reconstruction
for FFSL schemes using convolution operations with JAX.

The PPM method provides third-order accurate reconstruction with
monotonicity-preserving limiters to prevent spurious oscillations.

References:
    - Colella & Woodward (1984): The Piecewise Parabolic Method (PPM)
    - Lin & Rood (1996): Multidimensional Flux-Form Semi-Lagrangian Transport Schemes
"""

import jax.numpy as jnp
import jax
from jax import lax


def ppm_reconstruction_1d(
    q: jnp.ndarray,
    axis: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute PPM reconstruction for a 1D profile along specified axis.

    Reconstructs left (qL) and right (qR) edge values for each cell
    using parabolic interpolation.

    Args:
        q: Field to reconstruct (shape can be 1D, 2D, or 3D)
        axis: Axis along which to perform reconstruction (0, 1, or 2)

    Returns:
        tuple containing:
            - qL: Left edge values (same shape as q)
            - qR: Right edge values (same shape as q)
    """

    # Step 1: Compute cell-averaged slopes using 4th-order interpolation
    # delta_q[i] represents the slope in cell i
    delta_q = compute_slopes(q, axis)

    # Step 2: Compute edge values using parabolic interpolation
    qL, qR = compute_edge_values(q, delta_q, axis)

    # Step 3: Apply monotonicity limiters
    qL, qR = apply_monotonicity_limiter(q, qL, qR, axis)

    return qL, qR


def compute_slopes(q: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Compute cell-averaged slopes using 4th-order interpolation.

    Uses a centered difference formula:
    delta_q[i] = (q[i+1] - q[i-1]) / 2

    With corrections at boundaries using one-sided differences.

    Args:
        q: Field values
        axis: Axis along which to compute slopes

    Returns:
        delta_q: Slopes in each cell
    """

    # Shift arrays to get neighboring values
    q_plus = jnp.roll(q, -1, axis=axis)
    q_minus = jnp.roll(q, 1, axis=axis)

    # Centered difference
    delta_q = 0.5 * (q_plus - q_minus)

    return delta_q


def compute_edge_values(
    q: jnp.ndarray,
    delta_q: jnp.ndarray,
    axis: int = 0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute left and right edge values from cell averages and slopes.

    Uses 4th-order accurate reconstruction:
    qL[i] = q[i] - delta_q[i] / 2 + (q[i+1] - q[i] - delta_q[i+1]) / 6
    qR[i] = qL[i+1]

    Args:
        q: Cell-averaged values
        delta_q: Cell slopes
        axis: Axis along which to compute edges

    Returns:
        tuple containing:
            - qL: Left edge values
            - qR: Right edge values
    """

    # Get shifted arrays
    q_plus = jnp.roll(q, -1, axis=axis)
    delta_q_plus = jnp.roll(delta_q, -1, axis=axis)

    # Compute left edge values using 4th-order formula
    qL = q + 0.5 * (q - q_plus) + (delta_q + delta_q_plus) / 6.0

    # Right edge of cell i is left edge of cell i+1
    qR = jnp.roll(qL, -1, axis=axis)

    return qL, qR


def apply_monotonicity_limiter(
    q: jnp.ndarray,
    qL: jnp.ndarray,
    qR: jnp.ndarray,
    axis: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply monotonicity-preserving limiters to edge values.

    Ensures that:
    1. Edge values lie between neighboring cell averages
    2. No new extrema are created
    3. Monotonicity is preserved

    This implements the PPM limiters from Colella & Woodward (1984).

    Args:
        q: Cell-averaged values
        qL: Left edge values (before limiting)
        qR: Right edge values (before limiting)
        axis: Axis along which to apply limiters

    Returns:
        tuple containing:
            - qL: Limited left edge values
            - qR: Limited right edge values
    """

    # Get neighboring cell values
    q_minus = jnp.roll(q, 1, axis=axis)
    q_plus = jnp.roll(q, -1, axis=axis)

    # Compute local min and max from neighbors
    q_min = jnp.minimum(jnp.minimum(q, q_minus), q_plus)
    q_max = jnp.maximum(jnp.maximum(q, q_minus), q_plus)

    # Check if edge values violate monotonicity
    # If qL or qR are outside [q_min, q_max], we need to limit

    # Clip edge values to local bounds
    qL = jnp.clip(qL, q_min, q_max)
    qR = jnp.clip(qR, q_min, q_max)

    # Check if parabola has local extremum within cell
    # If (qR - q) * (q - qL) <= 0, cell contains extremum
    has_extremum = (qR - q) * (q - qL) <= 0.0

    # If extremum exists, flatten to cell average
    qL = jnp.where(has_extremum, q, qL)
    qR = jnp.where(has_extremum, q, qR)

    # Additional limiter: ensure parabola doesn't overshoot too much
    # Limit deviation from cell average
    da = qR - qL
    a6 = 6.0 * (q - 0.5 * (qL + qR))

    # If the parabola is too steep, reduce slopes
    condition = da * a6 > da * da
    qL = jnp.where(condition, 3.0 * q - 2.0 * qR, qL)

    condition = da * a6 < -da * da
    qR = jnp.where(condition, 3.0 * q - 2.0 * qL, qR)

    return qL, qR


def ppm_reconstruction_2d(
    q: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform 2D PPM reconstruction on a field.

    Computes left and right edge values in both x and y directions.

    Args:
        q: 2D field to reconstruct (shape: [nx, ny] or [nx, ny, nz])

    Returns:
        tuple containing:
            - qL_x: Left edge values in x direction
            - qR_x: Right edge values in x direction
            - qL_y: Left edge values in y direction
            - qR_y: Right edge values in y direction
    """

    # Reconstruct in x direction (axis 0)
    qL_x, qR_x = ppm_reconstruction_1d(q, axis=0)

    # Reconstruct in y direction (axis 1)
    qL_y, qR_y = ppm_reconstruction_1d(q, axis=1)

    return qL_x, qR_x, qL_y, qR_y


def convolution_weights(lx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute convolution weights for PPM integration.

    For a parabolic profile q(s) = qL + s*(qR - qL) + s*(1-s)*a6
    where a6 = 6*(q_avg - 0.5*(qL + qR))

    The integral from 0 to lx is:
    Q = qL*lx + 0.5*(qR - qL)*lx^2 + a6*(lx^2/2 - lx^3/3)

    This can be written as:
    Q = w0*qL + w1*qR + w2*q_avg

    Args:
        lx: Fractional distance (0 to 1) representing CFL number

    Returns:
        tuple containing:
            - w0: Weight for left edge value
            - w1: Weight for right edge value
            - w2: Weight for cell average
    """

    lx2 = lx * lx
    lx3 = lx2 * lx

    # Weights derived from integrating the parabola
    w0 = lx - 0.5 * lx2 + lx2 - 2.0 * lx3
    w1 = 0.5 * lx2 - lx2 + 2.0 * lx3
    w2 = 2.0 * (lx2 - lx3)

    return w0, w1, w2


def ffsl_convolution_1d(
    q: jnp.ndarray,
    qL: jnp.ndarray,
    qR: jnp.ndarray,
    courant: jnp.ndarray,
    axis: int = 0,
) -> jnp.ndarray:
    """Compute FFSL flux using convolution with PPM reconstruction.

    This function integrates the parabolic profile over the upwind region
    defined by the Courant number (CFL condition).

    Args:
        q: Cell-averaged values
        qL: Left edge values from PPM reconstruction
        qR: Right edge values from PPM reconstruction
        courant: Courant number (CFL = v * dt / dx)
        axis: Axis along which to compute flux

    Returns:
        flux: Flux computed using PPM reconstruction
    """

    # For positive velocities, we integrate from the left
    # For negative velocities, we integrate from the right

    # Get absolute Courant number and sign
    abs_courant = jnp.abs(courant)
    sign_courant = jnp.sign(courant)

    # Compute convolution weights
    w0, w1, w2 = convolution_weights(abs_courant)

    # Compute flux using weighted combination
    # For positive flow: integrate from left
    flux_pos = w0 * qL + w1 * qR + w2 * q

    # For negative flow: need to shift and reverse
    qL_shifted = jnp.roll(qL, 1, axis=axis)
    qR_shifted = jnp.roll(qR, 1, axis=axis)
    q_shifted = jnp.roll(q, 1, axis=axis)

    flux_neg = w0 * qR_shifted + w1 * qL_shifted + w2 * q_shifted

    # Select based on flow direction
    flux = jnp.where(sign_courant >= 0, flux_pos, flux_neg)

    return flux


def ffsl_reconstruction_2d(
    q: jnp.ndarray,
    courant_x: jnp.ndarray,
    courant_y: jnp.ndarray,
) -> jnp.ndarray:
    """Perform 2D FFSL reconstruction with PPM and convolution.

    This is the main interface function that:
    1. Performs PPM reconstruction in both directions
    2. Computes fluxes using convolution
    3. Updates the field using flux-form advection

    Args:
        q: 2D field to advect (shape: [nx, ny] or [nx, ny, nz])
        courant_x: Courant number in x direction (CFL_x = vx * dt / dx)
        courant_y: Courant number in y direction (CFL_y = vy * dt / dy)

    Returns:
        q_new: Updated field after FFSL advection
    """

    # Perform 2D PPM reconstruction
    qL_x, qR_x, qL_y, qR_y = ppm_reconstruction_2d(q)

    # Compute fluxes in x direction
    flux_x = ffsl_convolution_1d(q, qL_x, qR_x, courant_x, axis=0)

    # Compute fluxes in y direction
    flux_y = ffsl_convolution_1d(q, qL_y, qR_y, courant_y, axis=1)

    # Update using flux-form advection
    # q_new = q - (flux_x[i] - flux_x[i-1]) - (flux_y[j] - flux_y[j-1])
    flux_x_left = jnp.roll(flux_x, 1, axis=0)
    flux_y_left = jnp.roll(flux_y, 1, axis=1)

    q_new = q - (flux_x - flux_x_left) - (flux_y - flux_y_left)

    return q_new
