"""Example: FFSL Reconstruction with PPM and Convolution.

This example demonstrates how to use the FFSL (Flux-Form Semi-Lagrangian)
reconstruction with Piecewise Parabolic Method (PPM) for advection.

The FFSL method combines:
1. High-order PPM reconstruction for spatial accuracy
2. Flux-form advection for exact mass conservation
3. Monotonicity limiters to prevent spurious oscillations
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from sl_jax.reconstruction import (
    ffsl_reconstruction_2d,
    ppm_reconstruction_2d,
)


def example_1d_advection():
    """Example 1: 1D advection of different profiles."""
    print("Example 1: 1D Advection")
    print("-" * 50)

    # Setup
    nx = 100
    x = jnp.linspace(0, 1, nx)
    dx = x[1] - x[0]

    # Test different initial conditions
    profiles = {
        "Gaussian": jnp.exp(-((x - 0.5) / 0.1) ** 2),
        "Square wave": jnp.where((x > 0.3) & (x < 0.7), 1.0, 0.0),
        "Sine wave": jnp.sin(2 * jnp.pi * x),
    }

    # Advection parameters
    velocity = 0.5
    dt = 0.01
    courant = velocity * dt / dx

    print(f"Grid points: {nx}")
    print(f"CFL number: {courant:.3f}")
    print(f"Time step: {dt}")
    print()

    for name, q0 in profiles.items():
        # Expand to 2D for compatibility (1D profile along x-axis)
        q_2d = q0[:, jnp.newaxis]
        courant_x = jnp.ones_like(q_2d) * courant
        courant_y = jnp.zeros_like(q_2d)

        # Perform one advection step
        q_new = ffsl_reconstruction_2d(q_2d, courant_x, courant_y)

        # Extract 1D profile
        q1 = q_new[:, 0]

        # Check conservation
        mass0 = jnp.sum(q0) * dx
        mass1 = jnp.sum(q1) * dx
        error = jnp.abs(mass1 - mass0) / mass0 * 100

        print(f"{name}:")
        print(f"  Initial mass: {mass0:.6f}")
        print(f"  Final mass:   {mass1:.6f}")
        print(f"  Relative error: {error:.4f}%")
        print()


def example_2d_advection():
    """Example 2: 2D advection of a vortex."""
    print("Example 2: 2D Advection")
    print("-" * 50)

    # Setup grid
    nx, ny = 64, 64
    x = jnp.linspace(-1, 1, nx)
    y = jnp.linspace(-1, 1, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Initial condition: Gaussian bump
    q0 = jnp.exp(-5 * (X**2 + Y**2))

    # Velocity field (solid body rotation)
    vx = -Y
    vy = X

    # Time step
    dt = 0.01
    courant_x = vx * dt / dx
    courant_y = vy * dt / dy

    print(f"Grid: {nx} x {ny}")
    print(f"Max CFL_x: {jnp.max(jnp.abs(courant_x)):.3f}")
    print(f"Max CFL_y: {jnp.max(jnp.abs(courant_y)):.3f}")
    print()

    # Perform advection
    q1 = ffsl_reconstruction_2d(q0, courant_x, courant_y)

    # Compute diagnostics
    mass0 = jnp.sum(q0) * dx * dy
    mass1 = jnp.sum(q1) * dx * dy
    conservation_error = jnp.abs(mass1 - mass0) / mass0 * 100

    min_val = jnp.min(q1)
    max_val = jnp.max(q1)

    print("Results:")
    print(f"  Initial mass: {mass0:.6f}")
    print(f"  Final mass:   {mass1:.6f}")
    print(f"  Conservation error: {conservation_error:.4f}%")
    print(f"  Min value: {min_val:.6f}")
    print(f"  Max value: {max_val:.6f}")
    print()


def example_ppm_reconstruction():
    """Example 3: Understanding PPM reconstruction."""
    print("Example 3: PPM Reconstruction")
    print("-" * 50)

    # Create a test field with various features
    nx, ny = 32, 32
    x = jnp.linspace(0, 2 * jnp.pi, nx)
    y = jnp.linspace(0, 2 * jnp.pi, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Field with multiple scales
    q = (
        1.0
        + 0.5 * jnp.sin(X)
        + 0.3 * jnp.cos(2 * Y)
        + 0.2 * jnp.sin(X) * jnp.cos(Y)
    )

    # Perform PPM reconstruction
    qL_x, qR_x, qL_y, qR_y = ppm_reconstruction_2d(q)

    # Analyze reconstruction
    print("Field statistics:")
    print(f"  Cell average - mean: {jnp.mean(q):.4f}, std: {jnp.std(q):.4f}")
    print()
    print("X-direction reconstruction:")
    print(f"  Left edges - mean: {jnp.mean(qL_x):.4f}, std: {jnp.std(qL_x):.4f}")
    print(f"  Right edges - mean: {jnp.mean(qR_x):.4f}, std: {jnp.std(qR_x):.4f}")
    print()
    print("Y-direction reconstruction:")
    print(f"  Left edges - mean: {jnp.mean(qL_y):.4f}, std: {jnp.std(qL_y):.4f}")
    print(f"  Right edges - mean: {jnp.mean(qR_y):.4f}, std: {jnp.std(qR_y):.4f}")
    print()

    # Check edge consistency (qR[i] should be related to qL[i+1])
    qR_x_shifted = jnp.roll(qL_x, -1, axis=0)
    edge_difference = jnp.mean(jnp.abs(qR_x - qR_x_shifted))
    print(f"Edge consistency check: {edge_difference:.6f}")
    print("(Small values indicate consistent reconstruction)")
    print()


def example_limiting():
    """Example 4: Effect of monotonicity limiters."""
    print("Example 4: Monotonicity Limiters")
    print("-" * 50)

    # Create field with sharp gradients
    nx = 50
    x = jnp.linspace(0, 1, nx)

    # Step function (challenging for limiters)
    q = jnp.where(x < 0.5, 0.0, 1.0)

    # Expand to 2D
    q_2d = q[:, jnp.newaxis]

    # Reconstruct
    qL_x, qR_x, _, _ = ppm_reconstruction_2d(q_2d)

    # Extract 1D
    qL = qL_x[:, 0]
    qR = qR_x[:, 0]

    # Check for overshoots/undershoots
    q_min = jnp.min(q)
    q_max = jnp.max(q)

    overshoots = jnp.sum((qL > q_max) | (qR > q_max))
    undershoots = jnp.sum((qL < q_min) | (qR < q_min))

    print("Step function test:")
    print(f"  Original range: [{q_min:.4f}, {q_max:.4f}]")
    print(f"  Left edges range: [{jnp.min(qL):.4f}, {jnp.max(qL):.4f}]")
    print(f"  Right edges range: [{jnp.min(qR):.4f}, {jnp.max(qR):.4f}]")
    print(f"  Overshoots: {overshoots}")
    print(f"  Undershoots: {undershoots}")
    print()

    if overshoots == 0 and undershoots == 0:
        print("✓ Limiters successfully prevented all overshoots/undershoots!")
    else:
        print("⚠ Some overshoots/undershoots detected (may be at boundaries)")
    print()


def example_convergence_study():
    """Example 5: Convergence study with grid refinement."""
    print("Example 5: Convergence Study")
    print("-" * 50)

    # Smooth initial condition
    def analytical_solution(x, y, t, vx=1.0, vy=0.5):
        """Analytical solution for constant velocity advection."""
        return jnp.exp(-((x - vx * t) ** 2 + (y - vy * t) ** 2) / 0.1)

    # Test different grid resolutions
    resolutions = [32, 64, 128]
    errors = []

    vx, vy = 1.0, 0.5
    t_final = 0.1

    print(f"Advection test: vx={vx}, vy={vy}, t={t_final}")
    print()

    for nx in resolutions:
        ny = nx
        x = jnp.linspace(0, 2, nx)
        y = jnp.linspace(0, 2, ny)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # Initial condition
        q0 = analytical_solution(X, Y, 0.0, vx, vy)

        # Time stepping
        dt = 0.5 * dx / max(abs(vx), abs(vy))  # CFL ~ 0.5
        nsteps = int(t_final / dt)
        dt = t_final / nsteps  # Adjust to hit t_final exactly

        courant_x = jnp.ones((nx, ny)) * vx * dt / dx
        courant_y = jnp.ones((nx, ny)) * vy * dt / dy

        # Advect
        q = q0.copy()
        for step in range(nsteps):
            q = ffsl_reconstruction_2d(q, courant_x, courant_y)

        # Compare with analytical solution
        q_exact = analytical_solution(X, Y, t_final, vx, vy)
        error = jnp.sqrt(jnp.mean((q - q_exact) ** 2))
        errors.append(error)

        print(f"Resolution {nx}x{ny}:")
        print(f"  Time steps: {nsteps}")
        print(f"  CFL: {jnp.max(jnp.abs(courant_x)):.3f}")
        print(f"  L2 error: {error:.6e}")
        print()

    # Estimate convergence rate
    if len(errors) > 1:
        rate = jnp.log(errors[0] / errors[1]) / jnp.log(2)
        print(f"Estimated convergence rate: {rate:.2f}")
        print("(Expected: ~2-3 for PPM method)")
    print()


def main():
    """Run all examples."""
    print("=" * 60)
    print("FFSL Reconstruction with PPM - Examples")
    print("=" * 60)
    print()

    # Run examples
    example_1d_advection()
    example_2d_advection()
    example_ppm_reconstruction()
    example_limiting()
    example_convergence_study()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
