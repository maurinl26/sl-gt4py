"""Example: Differentiable FFSL Flux with Sigmoid Windowing.

This example demonstrates the fully differentiable flux computation using
sigmoid-weighted sliding windows instead of hard thresholds with fractional parts.

Key advantages:
1. Fully differentiable for optimization
2. Smooth transitions instead of discontinuities
3. Ideal for machine learning and inverse problems
4. Compatible with jax.grad and jax.jacobian
"""

import jax.numpy as jnp
import jax
import sys
import importlib.util

# Import reconstruction module directly
spec = importlib.util.spec_from_file_location('reconstruction', 'src/jax/reconstruction.py')
reconstruction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconstruction)


def example_1_sigmoid_weighting():
    """Example 1: Understand sigmoid weighting function."""
    print("=" * 60)
    print("Example 1: Sigmoid Weighting Function")
    print("=" * 60)

    # Test sigmoid at different positions
    x = jnp.linspace(-2, 2, 100)

    print("\nSigmoid weight function: w(x) = 1 / (1 + exp(-(x - center)/width))")
    print("\nDifferent widths:")

    for width in [0.01, 0.1, 0.5]:
        w = reconstruction.sigmoid_weight(x, center=0.0, width=width)
        print(f"\n  Width = {width}:")
        print(f"    w(-1) = {w[25]:.4f}")
        print(f"    w(0)  = {w[50]:.4f}")
        print(f"    w(+1) = {w[75]:.4f}")
        print(f"    Smoothness: {'Sharp' if width < 0.1 else 'Smooth'}")

    print("\nNote: Smaller width → sharper transition (closer to step function)")
    print("      Larger width → smoother transition (more gradual)")
    print()


def example_2_standard_vs_differentiable():
    """Example 2: Compare standard and differentiable flux computation."""
    print("=" * 60)
    print("Example 2: Standard vs Differentiable Flux")
    print("=" * 60)

    # Simple 1D profile
    q = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
    qL, qR = reconstruction.ppm_reconstruction_1d(q, axis=0, limiter="van_leer")

    courant = jnp.ones_like(q) * 0.3

    print("\nProfile:")
    print(f"  q = {q}")
    print(f"  Courant = {courant[0]:.2f}")

    # Standard flux
    flux_std = reconstruction.ffsl_convolution_1d_direct(q, qL, qR, courant, axis=0)

    # Differentiable flux
    flux_diff = reconstruction.ffsl_convolution_1d_differentiable(
        q, qL, qR, courant, axis=0,
        window_size=5, sigmoid_width=0.1
    )

    print("\nStandard flux:")
    print(f"  {flux_std}")

    print("\nDifferentiable flux:")
    print(f"  {flux_diff}")

    diff = jnp.mean(jnp.abs(flux_std - flux_diff))
    print(f"\nMean absolute difference: {diff:.6f}")
    print("Note: Small difference indicates similar accuracy")
    print()


def example_3_gradient_computation():
    """Example 3: Compute gradients with differentiable flux."""
    print("=" * 60)
    print("Example 3: Gradient Computation")
    print("=" * 60)

    # Define a loss function that uses FFSL advection
    def loss_function(q_initial, courant_x, courant_y, q_target):
        """Loss: MSE between advected field and target."""
        q_advected = reconstruction.ffsl_reconstruction_2d(
            q_initial, courant_x, courant_y,
            limiter="van_leer",
            differentiable=True,
            sigmoid_width=0.1,
            window_size=5
        )
        return jnp.mean((q_advected - q_target)**2)

    # Setup
    nx, ny = 16, 16
    q_initial = jnp.ones((nx, ny))
    courant_x = jnp.ones((nx, ny)) * 0.3
    courant_y = jnp.ones((nx, ny)) * 0.2
    q_target = jnp.ones((nx, ny)) * 1.1  # Slightly different

    print(f"\nGrid: {nx} x {ny}")
    print(f"Initial field mean: {jnp.mean(q_initial):.4f}")
    print(f"Target field mean: {jnp.mean(q_target):.4f}")

    # Compute loss
    loss = loss_function(q_initial, courant_x, courant_y, q_target)
    print(f"\nLoss (MSE): {loss:.6f}")

    # Compute gradient with respect to initial condition
    print("\nComputing gradient w.r.t. initial condition...")
    grad_q = jax.grad(loss_function, argnums=0)(q_initial, courant_x, courant_y, q_target)

    print(f"✓ Gradient computed successfully!")
    print(f"  Gradient shape: {grad_q.shape}")
    print(f"  Gradient mean: {jnp.mean(grad_q):.6f}")
    print(f"  Gradient std: {jnp.std(grad_q):.6f}")

    # Compute gradient with respect to Courant numbers
    print("\nComputing gradient w.r.t. Courant number X...")
    grad_cx = jax.grad(loss_function, argnums=1)(q_initial, courant_x, courant_y, q_target)

    print(f"✓ Gradient computed successfully!")
    print(f"  Gradient mean: {jnp.mean(grad_cx):.6f}")

    print("\nNote: Full differentiability enables gradient-based optimization")
    print()


def example_4_optimization_problem():
    """Example 4: Simple optimization using differentiable FFSL."""
    print("=" * 60)
    print("Example 4: Optimization Problem")
    print("=" * 60)

    # Problem: find optimal Courant number to match target after advection

    def objective(courant_value):
        """Objective: match target distribution after advection."""
        nx, ny = 16, 16

        # Initial Gaussian
        x = jnp.linspace(-2, 2, nx)
        y = jnp.linspace(-2, 2, ny)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        q_init = jnp.exp(-(X**2 + Y**2))

        # Target: shifted Gaussian
        q_target = jnp.exp(-((X - 0.5)**2 + (Y - 0.3)**2))

        # Advect with given Courant
        courant_x = jnp.ones((nx, ny)) * courant_value
        courant_y = jnp.ones((nx, ny)) * courant_value * 0.6

        q_advected = reconstruction.ffsl_reconstruction_2d(
            q_init, courant_x, courant_y,
            limiter="van_leer",
            differentiable=True,
            sigmoid_width=0.1
        )

        # MSE loss
        return jnp.mean((q_advected - q_target)**2)

    print("\nProblem: Find optimal Courant number to match target distribution")
    print("\nInitial guess: CFL = 0.2")

    # Initial guess
    courant_init = 0.2
    loss_init = objective(courant_init)
    print(f"  Initial loss: {loss_init:.6f}")

    # Compute gradient
    grad_fn = jax.grad(objective)

    # Simple gradient descent
    learning_rate = 0.01
    courant_opt = courant_init

    print("\nOptimization (10 steps of gradient descent):")
    for step in range(10):
        grad = grad_fn(courant_opt)
        courant_opt = courant_opt - learning_rate * grad
        loss = objective(courant_opt)

        if step % 2 == 0:
            print(f"  Step {step}: CFL = {courant_opt:.4f}, Loss = {loss:.6f}")

    print(f"\nFinal optimized CFL: {courant_opt:.4f}")
    print(f"Final loss: {objective(courant_opt):.6f}")
    print(f"Improvement: {(1 - objective(courant_opt)/loss_init)*100:.1f}%")
    print()


def example_5_jacobian_computation():
    """Example 5: Compute full Jacobian matrix."""
    print("=" * 60)
    print("Example 5: Jacobian Computation")
    print("=" * 60)

    # Small grid for demonstration
    nx, ny = 8, 8

    def advection_operator(q):
        """FFSL advection as a function of the field."""
        courant_x = jnp.ones((nx, ny)) * 0.3
        courant_y = jnp.ones((nx, ny)) * 0.2

        return reconstruction.ffsl_reconstruction_2d(
            q, courant_x, courant_y,
            limiter="van_leer",
            differentiable=True,
            sigmoid_width=0.1
        )

    # Test field
    q = jnp.ones((nx, ny))

    print(f"\nGrid: {nx} x {ny}")
    print("Computing Jacobian of advection operator...")

    # Compute Jacobian: dq_out/dq_in
    jacobian = jax.jacobian(lambda x: advection_operator(x).flatten())(q)

    print(f"✓ Jacobian computed successfully!")
    print(f"  Jacobian shape: {jacobian.shape}")
    print(f"  Expected: ({nx*ny}, {nx*ny})")

    # Analyze Jacobian structure
    jac_2d = jacobian.reshape(nx, ny, nx, ny)
    print(f"\n Jacobian properties:")
    print(f"  Mean: {jnp.mean(jacobian):.6f}")
    print(f"  Std: {jnp.std(jacobian):.6f}")
    print(f"  Sparsity: {jnp.sum(jnp.abs(jacobian) < 1e-6) / jacobian.size * 100:.1f}%")

    print("\nNote: Jacobian reveals how output depends on each input cell")
    print()


def example_6_2d_advection_comparison():
    """Example 6: 2D advection with different modes."""
    print("=" * 60)
    print("Example 6: 2D Advection - All Modes")
    print("=" * 60)

    # Create 2D Gaussian
    nx, ny = 32, 32
    x = jnp.linspace(-2, 2, nx)
    y = jnp.linspace(-2, 2, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    q = jnp.exp(-(X**2 + Y**2))

    courant_x = jnp.ones((nx, ny)) * 0.3
    courant_y = jnp.ones((nx, ny)) * 0.2

    print(f"\nGrid: {nx} x {ny}")
    print(f"Initial mass: {jnp.sum(q):.6f}")

    # Mode 1: Standard with PPM limiter
    q1 = reconstruction.ffsl_reconstruction_2d(
        q, courant_x, courant_y,
        limiter="ppm", differentiable=False
    )

    print("\nMode 1: Standard PPM")
    print(f"  Final mass: {jnp.sum(q1):.6f}")
    print(f"  Conservation error: {abs(jnp.sum(q1) - jnp.sum(q)):.8f}")

    # Mode 2: Van Leer limiter (differentiable reconstruction)
    q2 = reconstruction.ffsl_reconstruction_2d(
        q, courant_x, courant_y,
        limiter="van_leer", differentiable=False
    )

    print("\nMode 2: Van Leer (differentiable reconstruction)")
    print(f"  Final mass: {jnp.sum(q2):.6f}")
    print(f"  Conservation error: {abs(jnp.sum(q2) - jnp.sum(q)):.8f}")

    # Mode 3: Fully differentiable (Van Leer + sigmoid flux)
    q3 = reconstruction.ffsl_reconstruction_2d(
        q, courant_x, courant_y,
        limiter="van_leer", differentiable=True,
        sigmoid_width=0.1, window_size=5
    )

    print("\nMode 3: Fully Differentiable (Van Leer + sigmoid flux)")
    print(f"  Final mass: {jnp.sum(q3):.6f}")
    print(f"  Conservation error: {abs(jnp.sum(q3) - jnp.sum(q)):.8f}")

    # Compare modes
    print("\nComparison:")
    print(f"  |Mode1 - Mode2|: {jnp.mean(jnp.abs(q1 - q2)):.8f}")
    print(f"  |Mode2 - Mode3|: {jnp.mean(jnp.abs(q2 - q3)):.8f}")
    print(f"  |Mode1 - Mode3|: {jnp.mean(jnp.abs(q1 - q3)):.8f}")

    print("\nNote: All modes give similar results, but Mode 3 is fully differentiable")
    print()


def example_7_sigmoid_width_sensitivity():
    """Example 7: Effect of sigmoid width parameter."""
    print("=" * 60)
    print("Example 7: Sigmoid Width Sensitivity")
    print("=" * 60)

    # Simple profile
    nx, ny = 16, 16
    q = jnp.ones((nx, ny))
    courant_x = jnp.ones((nx, ny)) * 0.3
    courant_y = jnp.ones((nx, ny)) * 0.2

    print(f"\nGrid: {nx} x {ny}")
    print(f"Testing different sigmoid widths:\n")

    widths = [0.01, 0.05, 0.1, 0.2, 0.5]

    for width in widths:
        q_advected = reconstruction.ffsl_reconstruction_2d(
            q, courant_x, courant_y,
            limiter="van_leer", differentiable=True,
            sigmoid_width=width
        )

        conservation_error = abs(jnp.sum(q_advected) - jnp.sum(q))

        print(f"  Width = {width:5.2f}: conservation error = {conservation_error:.8f}")

    print("\nRecommendation:")
    print("  • Use width ~ 0.1 for good balance")
    print("  • Smaller width → sharper, closer to standard method")
    print("  • Larger width → smoother, more regularized")
    print()


def main():
    """Run all differentiable flux examples."""
    print("\n" + "=" * 60)
    print("Differentiable FFSL Flux with Sigmoid Windowing")
    print("=" * 60 + "\n")

    example_1_sigmoid_weighting()
    example_2_standard_vs_differentiable()
    example_3_gradient_computation()
    example_4_optimization_problem()
    example_5_jacobian_computation()
    example_6_2d_advection_comparison()
    example_7_sigmoid_width_sensitivity()

    print("=" * 60)
    print("Summary: Differentiable FFSL Features")
    print("=" * 60)
    print("\n✓ Fully differentiable flux computation")
    print("✓ Sigmoid-weighted sliding windows")
    print("✓ Compatible with jax.grad and jax.jacobian")
    print("✓ Ideal for optimization and ML applications")
    print("✓ Smooth transitions instead of discontinuities")
    print("✓ Comparable accuracy to standard methods")
    print("\nUse cases:")
    print("  • Gradient-based optimization")
    print("  • Machine learning with physics")
    print("  • Inverse problems and data assimilation")
    print("  • Parameter estimation")
    print("  • Sensitivity analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()
