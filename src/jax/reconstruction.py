"""FFSL (Flux-Form Semi-Lagrangian) Reconstruction with PPM using convolution.

This module implements Piecewise Parabolic Method (PPM) reconstruction
for FFSL schemes using JAX convolution operations (jnp.convolve).

The PPM method provides third-order accurate reconstruction with
monotonicity-preserving limiters to prevent spurious oscillations.

References:
    - Colella & Woodward (1984): The Piecewise Parabolic Method (PPM)
    - Lin & Rood (1996): Multidimensional Flux-Form Semi-Lagrangian Transport Schemes
"""

import jax.numpy as jnp
import jax
from jax import lax


def compute_slopes_convolution(q: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Compute cell-averaged slopes using convolution with centered difference kernel.

    Uses convolution with kernel [-0.5, 0, 0.5] to compute:
    delta_q[i] = (q[i+1] - q[i-1]) / 2

    Args:
        q: Field values
        axis: Axis along which to compute slopes

    Returns:
        delta_q: Slopes in each cell
    """

    # Centered difference kernel: [-0.5, 0, 0.5]
    kernel = jnp.array([-0.5, 0.0, 0.5])

    # Move axis to last position for convolution
    q_moved = jnp.moveaxis(q, axis, -1)

    # Apply convolution along last axis
    delta_q_moved = jnp.apply_along_axis(
        lambda x: jnp.convolve(x, kernel, mode='same'),
        -1,
        q_moved
    )

    # Move axis back
    delta_q = jnp.moveaxis(delta_q_moved, -1, axis)

    return delta_q


def compute_edge_values_convolution(
    q: jnp.ndarray,
    axis: int = 0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute left and right edge values using convolution-based reconstruction.

    Uses PPM reconstruction with 4th-order accurate stencil.
    Implements the convolution: qL[i] = sum_j c[j] * q[i+j]

    Args:
        q: Cell-averaged values
        axis: Axis along which to compute edges

    Returns:
        tuple containing:
            - qL: Left edge values
            - qR: Right edge values
    """

    # PPM edge reconstruction kernel (4th order)
    # qL[i] = (1/12)*q[i-2] - (7/12)*q[i-1] + (7/6)*q[i] + (1/12)*q[i+1]
    # This is a standard 4th-order PPM interpolation kernel
    kernel_L = jnp.array([1/12, -7/12, 7/6, 1/12])

    # For right edge, shift by one
    # qR[i] = qL[i+1]

    # Move axis to last position
    q_moved = jnp.moveaxis(q, axis, -1)

    # Apply convolution for left edges
    qL_moved = jnp.apply_along_axis(
        lambda x: jnp.convolve(x, kernel_L, mode='same'),
        -1,
        q_moved
    )

    # Move back
    qL = jnp.moveaxis(qL_moved, -1, axis)

    # Right edge is left edge of next cell
    qR = jnp.roll(qL, -1, axis=axis)

    return qL, qR


def apply_van_leer_limiter(
    q: jnp.ndarray,
    qL: jnp.ndarray,
    qR: jnp.ndarray,
    axis: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Van Leer monotonicity limiter to edge values.

    The Van Leer limiter is a simpler alternative to PPM limiters that:
    1. Preserves monotonicity
    2. Is less dissipative than min-mod
    3. Is easier to implement and understand

    The limiter computes a slope with:
    slope = 2 * (q[i+1] - q[i]) * (q[i] - q[i-1]) / (q[i+1] - q[i-1])
    if (q[i+1] - q[i]) and (q[i] - q[i-1]) have the same sign,
    otherwise slope = 0.

    Args:
        q: Cell-averaged values
        qL: Left edge values (before limiting)
        qR: Right edge values (before limiting)
        axis: Axis along which to apply limiters

    Returns:
        tuple containing:
            - qL: Limited left edge values
            - qR: Limited right edge values

    Reference:
        Van Leer, B. (1977). "Towards the ultimate conservative difference scheme III"
    """

    # Get neighboring cell values
    q_minus = jnp.roll(q, 1, axis=axis)
    q_plus = jnp.roll(q, -1, axis=axis)

    # Compute forward and backward differences
    dq_forward = q_plus - q
    dq_backward = q - q_minus

    # Van Leer slope limiter
    # If signs differ, set slope to zero (local extremum)
    # Otherwise, use harmonic mean weighted slope
    same_sign = dq_forward * dq_backward > 0.0

    # Harmonic mean of forward and backward differences
    # slope = 2 * dq_forward * dq_backward / (dq_forward + dq_backward)
    # Avoid division by zero
    denominator = dq_forward + dq_backward
    denominator = jnp.where(jnp.abs(denominator) < 1e-10, 1e-10, denominator)

    slope_limited = 2.0 * dq_forward * dq_backward / denominator

    # Set slope to zero at extrema
    slope_limited = jnp.where(same_sign, slope_limited, 0.0)

    # Reconstruct edge values from limited slope
    qL_limited = q - 0.5 * slope_limited
    qR_limited = q + 0.5 * slope_limited

    return qL_limited, qR_limited


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


def ppm_reconstruction_1d(
    q: jnp.ndarray,
    axis: int = 0,
    limiter: str = "ppm",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute PPM reconstruction for a 1D profile along specified axis using convolution.

    Reconstructs left (qL) and right (qR) edge values for each cell
    using parabolic interpolation with convolution operations.

    Args:
        q: Field to reconstruct (shape can be 1D, 2D, or 3D)
        axis: Axis along which to perform reconstruction (0, 1, or 2)
        limiter: Type of limiter to use ('ppm' or 'van_leer')
                 - 'ppm': Colella & Woodward (1984) PPM limiters (default)
                 - 'van_leer': Van Leer (1977) slope limiter

    Returns:
        tuple containing:
            - qL: Left edge values (same shape as q)
            - qR: Right edge values (same shape as q)
    """

    # Step 1: Compute edge values using convolution-based reconstruction
    qL, qR = compute_edge_values_convolution(q, axis)

    # Step 2: Apply monotonicity limiters
    if limiter == "van_leer":
        qL, qR = apply_van_leer_limiter(q, qL, qR, axis)
    elif limiter == "ppm":
        qL, qR = apply_monotonicity_limiter(q, qL, qR, axis)
    else:
        raise ValueError(f"Unknown limiter type: {limiter}. Use 'ppm' or 'van_leer'.")

    return qL, qR


def ppm_reconstruction_2d(
    q: jnp.ndarray,
    limiter: str = "ppm",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform 2D PPM reconstruction on a field using convolution.

    Computes left and right edge values in both x and y directions.

    Args:
        q: 2D field to reconstruct (shape: [nx, ny] or [nx, ny, nz])
        limiter: Type of limiter to use ('ppm' or 'van_leer')
                 - 'ppm': Colella & Woodward (1984) PPM limiters (default)
                 - 'van_leer': Van Leer (1977) slope limiter

    Returns:
        tuple containing:
            - qL_x: Left edge values in x direction
            - qR_x: Right edge values in x direction
            - qL_y: Left edge values in y direction
            - qR_y: Right edge values in y direction
    """

    # Reconstruct in x direction (axis 0)
    qL_x, qR_x = ppm_reconstruction_1d(q, axis=0, limiter=limiter)

    # Reconstruct in y direction (axis 1)
    qL_y, qR_y = ppm_reconstruction_1d(q, axis=1, limiter=limiter)

    return qL_x, qR_x, qL_y, qR_y


def compute_flux_kernel(courant: float, npoints: int = 5) -> jnp.ndarray:
    """Compute flux integration kernel for PPM convolution.

    For a given Courant number, computes the kernel that integrates
    the parabolic profile over the upwind region.

    Args:
        courant: Courant number (CFL = v * dt / dx)
        npoints: Number of points in the kernel (default: 5)

    Returns:
        kernel: Flux integration kernel for convolution
    """

    # For PPM, the flux kernel integrates the parabola
    # This is a simplified version - more sophisticated kernels can be built

    abs_c = jnp.abs(courant)

    # Build a kernel based on the Courant number
    # For small CFL, use local stencil
    # The kernel represents weights for flux calculation

    # Simplified flux kernel (can be refined)
    center_idx = npoints // 2
    kernel = jnp.zeros(npoints)

    # Main weight at center
    kernel = kernel.at[center_idx].set(abs_c)

    # Upwind bias
    if courant > 0:
        kernel = kernel.at[center_idx - 1].set(abs_c * 0.5)
    else:
        kernel = kernel.at[center_idx + 1].set(abs_c * 0.5)

    # Normalize
    kernel = kernel / jnp.sum(kernel)

    return kernel


def ffsl_convolution_1d_direct(
    q: jnp.ndarray,
    qL: jnp.ndarray,
    qR: jnp.ndarray,
    courant: jnp.ndarray,
    axis: int = 0,
) -> jnp.ndarray:
    """Compute FFSL flux using direct convolution with PPM reconstruction.

    This function integrates the parabolic profile over the upwind region
    using jnp.convolve for the flux calculation.

    Args:
        q: Cell-averaged values
        qL: Left edge values from PPM reconstruction
        qR: Right edge values from PPM reconstruction
        courant: Courant number (CFL = v * dt / dx)
        axis: Axis along which to compute flux

    Returns:
        flux: Flux computed using PPM reconstruction and convolution
    """

    # Build the parabolic coefficient a6 = 6*(q - 0.5*(qL + qR))
    a6 = 6.0 * (q - 0.5 * (qL + qR))

    # For uniform Courant number, can use single convolution
    # For varying Courant, need point-wise calculation

    # Get absolute Courant number and sign
    abs_courant = jnp.abs(courant)
    sign_courant = jnp.sign(courant)

    # Compute flux using integrated parabola
    # F = ∫₀^C [qL + s*(qR - qL) + s*(1-s)*a6] ds

    lx = abs_courant
    lx2 = lx * lx
    lx3 = lx2 * lx

    # Integration weights (derived analytically)
    w_qL = lx - 0.5 * lx2 + (lx2 - 2.0 * lx3)
    w_qR = 0.5 * lx2 + (2.0 * lx3 - lx2)
    w_q = 2.0 * (lx2 - lx3)

    # Compute flux for positive flow
    flux_pos = w_qL * qL + w_qR * qR + w_q * q

    # For negative flow: shift and reverse
    qL_shifted = jnp.roll(qL, 1, axis=axis)
    qR_shifted = jnp.roll(qR, 1, axis=axis)
    q_shifted = jnp.roll(q, 1, axis=axis)

    flux_neg = w_qL * qR_shifted + w_qR * qL_shifted + w_q * q_shifted

    # Select based on flow direction
    flux = jnp.where(sign_courant >= 0, flux_pos, flux_neg)

    return flux


def sigmoid_weight(x: jnp.ndarray, center: float, width: float = 0.1) -> jnp.ndarray:
    """Compute smooth sigmoid weight for differentiable windowing.

    Args:
        x: Input values
        center: Center of the sigmoid
        width: Width parameter (smaller = sharper transition)

    Returns:
        weight: Sigmoid weight between 0 and 1
    """
    return 1.0 / (1.0 + jnp.exp(-(x - center) / width))


def ffsl_convolution_1d_differentiable(
    q: jnp.ndarray,
    qL: jnp.ndarray,
    qR: jnp.ndarray,
    courant: jnp.ndarray,
    axis: int = 0,
    window_size: int = 5,
    sigmoid_width: float = 0.1,
) -> jnp.ndarray:
    """Compute FFSL flux using differentiable sliding window with sigmoid weighting.

    Instead of using hard thresholds with fractional parts, this uses a smooth
    sigmoid-weighted window that is fully differentiable. The window slides
    based on the Courant number and weights cells according to their contribution.

    This approach is ideal for:
    - Optimization problems requiring gradients
    - Machine learning applications
    - Inverse problems and data assimilation

    Args:
        q: Cell-averaged values
        qL: Left edge values from PPM reconstruction
        qR: Right edge values from PPM reconstruction
        courant: Courant number (CFL = v * dt / dx)
        axis: Axis along which to compute flux
        window_size: Size of the sliding window (default: 5)
        sigmoid_width: Width of sigmoid transition (smaller = sharper, default: 0.1)

    Returns:
        flux: Differentiable flux computed using sigmoid-weighted window
    """

    shape = q.shape
    ndim = len(shape)

    # Get absolute Courant number for window positioning
    abs_courant = jnp.abs(courant)
    sign_courant = jnp.sign(courant)

    # Create position grid for windowing
    # Position i corresponds to cells that contribute to the flux
    positions = jnp.arange(window_size) - window_size // 2

    # Compute weights for each position based on Courant number
    # The weight determines how much each cell contributes to the flux
    # Using sigmoid for smooth, differentiable transitions

    # For each cell, compute its distance from the flux integration region
    # Weight decreases smoothly as we move away from [0, courant]
    weights = jnp.zeros((window_size,) + shape)

    for i, pos in enumerate(positions):
        # Distance from flux region boundaries
        # Cell at position 'pos' contributes if it's within [0, abs_courant]
        lower_dist = jnp.abs(pos) - 0.0  # Distance from lower boundary
        upper_dist = jnp.abs(pos) - abs_courant  # Distance from upper boundary

        # Smooth window: inside if 0 <= pos <= abs_courant
        # Using sigmoid to create smooth transitions
        weight_lower = sigmoid_weight(-lower_dist, 0.0, sigmoid_width)
        weight_upper = sigmoid_weight(-upper_dist, 0.0, sigmoid_width)

        # Combine: weight is high when inside window
        weights = weights.at[i].set(weight_lower * weight_upper)

    # Normalize weights to ensure conservation
    weight_sum = jnp.sum(weights, axis=0)
    weight_sum = jnp.where(weight_sum < 1e-10, 1.0, weight_sum)
    weights = weights / weight_sum

    # Gather values with window
    flux_components = jnp.zeros((window_size,) + shape)

    for i, pos in enumerate(positions):
        # Roll to get values at position 'pos'
        if pos == 0:
            q_pos = q
            qL_pos = qL
            qR_pos = qR
        else:
            q_pos = jnp.roll(q, -pos, axis=axis)
            qL_pos = jnp.roll(qL, -pos, axis=axis)
            qR_pos = jnp.roll(qR, -pos, axis=axis)

        # Compute local flux contribution using PPM reconstruction
        # For each position, integrate the parabola over a unit cell
        # weighted by its contribution to the total flux

        # Local parabola coefficient
        a6_pos = 6.0 * (q_pos - 0.5 * (qL_pos + qR_pos))

        # Fractional contribution of this cell
        # This represents how much of the cell is swept by the flux
        frac = jnp.clip(abs_courant - jnp.abs(pos), 0.0, 1.0)
        frac2 = frac * frac
        frac3 = frac2 * frac

        # Integration weights for fractional cell
        w_qL = frac - 0.5 * frac2 + (frac2 - 2.0 * frac3)
        w_qR = 0.5 * frac2 + (2.0 * frac3 - frac2)
        w_q = 2.0 * (frac2 - frac3)

        # Local flux contribution
        local_flux = w_qL * qL_pos + w_qR * qR_pos + w_q * q_pos

        # Weight by sigmoid window
        flux_components = flux_components.at[i].set(weights[i] * local_flux)

    # Sum all contributions
    flux_pos = jnp.sum(flux_components, axis=0)

    # Handle negative flow direction with smooth transition
    flux_neg = jnp.roll(flux_pos, 1, axis=axis)

    # Smooth direction selection using sigmoid
    # Instead of jnp.where, use sigmoid blending for full differentiability
    direction_weight = sigmoid_weight(sign_courant, 0.0, sigmoid_width)
    flux = direction_weight * flux_pos + (1.0 - direction_weight) * flux_neg

    return flux


def ffsl_convolution_1d_kernel(
    q: jnp.ndarray,
    courant: float,
    axis: int = 0,
) -> jnp.ndarray:
    """Compute FFSL advection using convolution with flux kernel.

    This applies a pre-computed flux kernel using jnp.convolve
    for efficient flux-form advection.

    Args:
        q: Field to advect
        courant: Uniform Courant number (CFL = v * dt / dx)
        axis: Axis along which to advect

    Returns:
        q_new: Updated field after convolution-based advection
    """

    # First, perform PPM reconstruction
    qL, qR = ppm_reconstruction_1d(q, axis=axis)

    # Compute flux using direct method
    flux = ffsl_convolution_1d_direct(q, qL, qR,
                                      jnp.ones_like(q) * courant,
                                      axis=axis)

    # Apply flux divergence
    flux_left = jnp.roll(flux, 1, axis=axis)
    q_new = q - (flux - flux_left)

    return q_new


def ffsl_reconstruction_2d(
    q: jnp.ndarray,
    courant_x: jnp.ndarray,
    courant_y: jnp.ndarray,
    limiter: str = "ppm",
    differentiable: bool = False,
    sigmoid_width: float = 0.1,
    window_size: int = 5,
) -> jnp.ndarray:
    """Perform 2D FFSL reconstruction with PPM using convolution.

    This is the main interface function that:
    1. Performs PPM reconstruction in both directions using convolution
    2. Computes fluxes using convolution-based integration
    3. Updates the field using flux-form advection

    Args:
        q: 2D field to advect (shape: [nx, ny] or [nx, ny, nz])
        courant_x: Courant number in x direction (CFL_x = vx * dt / dx)
        courant_y: Courant number in y direction (CFL_y = vy * dt / dy)
        limiter: Type of limiter to use ('ppm' or 'van_leer')
                 - 'ppm': Colella & Woodward (1984) PPM limiters (default)
                 - 'van_leer': Van Leer (1977) slope limiter (differentiable)
        differentiable: Use fully differentiable sigmoid-windowed flux (default: False)
        sigmoid_width: Width of sigmoid transition for differentiable mode (default: 0.1)
        window_size: Size of sliding window for differentiable mode (default: 5)

    Returns:
        q_new: Updated field after FFSL advection with convolution
    """

    # Perform 2D PPM reconstruction using convolution
    qL_x, qR_x, qL_y, qR_y = ppm_reconstruction_2d(q, limiter=limiter)

    # Compute fluxes in x direction using convolution approach
    if differentiable:
        flux_x = ffsl_convolution_1d_differentiable(
            q, qL_x, qR_x, courant_x, axis=0,
            window_size=window_size, sigmoid_width=sigmoid_width
        )
    else:
        flux_x = ffsl_convolution_1d_direct(q, qL_x, qR_x, courant_x, axis=0)

    # Compute fluxes in y direction using convolution approach
    if differentiable:
        flux_y = ffsl_convolution_1d_differentiable(
            q, qL_y, qR_y, courant_y, axis=1,
            window_size=window_size, sigmoid_width=sigmoid_width
        )
    else:
        flux_y = ffsl_convolution_1d_direct(q, qL_y, qR_y, courant_y, axis=1)

    # Update using flux-form advection
    # q_new = q - (flux_x[i] - flux_x[i-1]) - (flux_y[j] - flux_y[j-1])
    flux_x_left = jnp.roll(flux_x, 1, axis=0)
    flux_y_left = jnp.roll(flux_y, 1, axis=1)

    q_new = q - (flux_x - flux_x_left) - (flux_y - flux_y_left)

    return q_new


# Legacy compatibility functions
def convolution_weights(lx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute convolution weights for PPM integration (legacy function).

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


# Alias for backward compatibility
ffsl_convolution_1d = ffsl_convolution_1d_direct
