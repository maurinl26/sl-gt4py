"""Example: Comparing PPM and Van Leer Limiters.

This example demonstrates the differences between:
1. PPM limiters (Colella & Woodward 1984)
2. Van Leer slope limiters (Van Leer 1977)

Key differences:
- PPM: More accurate on smooth profiles, more complex
- Van Leer: Simpler, more robust, fully differentiable
"""

import jax.numpy as jnp
import jax
import sys
import importlib.util

# Import reconstruction module directly
spec = importlib.util.spec_from_file_location('reconstruction', 'src/jax/reconstruction.py')
reconstruction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconstruction)


def example_1_smooth_profile():
    """Example 1: Limiters on smooth sinusoidal profile."""
    print("=" * 60)
    print("Example 1: Smooth Profile (Sinusoid)")
    print("=" * 60)

    # Smooth sinusoidal profile
    x = jnp.linspace(0, 2*jnp.pi, 64)
    q = 1.0 + 0.5 * jnp.sin(x)

    print("\nProfile: q = 1.0 + 0.5*sin(x)")
    print(f"Grid points: {len(q)}")

    # Reconstruct with both limiters
    qL_ppm, qR_ppm = reconstruction.ppm_reconstruction_1d(q, axis=0, limiter="ppm")
    qL_vl, qR_vl = reconstruction.ppm_reconstruction_1d(q, axis=0, limiter="van_leer")

    print("\nPPM Limiter:")
    print(f"  Mean qL: {jnp.mean(qL_ppm):.6f}")
    print(f"  Mean qR: {jnp.mean(qR_ppm):.6f}")
    print(f"  Max deviation from q: {jnp.max(jnp.abs(qL_ppm - q)):.6f}")

    print("\nVan Leer Limiter:")
    print(f"  Mean qL: {jnp.mean(qL_vl):.6f}")
    print(f"  Mean qR: {jnp.mean(qR_vl):.6f}")
    print(f"  Max deviation from q: {jnp.max(jnp.abs(qL_vl - q)):.6f}")

    print("\nNote: On smooth profiles, PPM should be more accurate")
    print()


def example_2_discontinuous_profile():
    """Example 2: Limiters on discontinuous (step) profile."""
    print("=" * 60)
    print("Example 2: Discontinuous Profile (Step Function)")
    print("=" * 60)

    # Step function
    x = jnp.linspace(0, 1, 50)
    q = jnp.where(x < 0.5, 0.0, 1.0)

    print("\nProfile: Step function at x=0.5")
    print(f"Grid points: {len(q)}")

    # Reconstruct with both limiters
    qL_ppm, qR_ppm = reconstruction.ppm_reconstruction_1d(q, axis=0, limiter="ppm")
    qL_vl, qR_vl = reconstruction.ppm_reconstruction_1d(q, axis=0, limiter="van_leer")

    print("\nPPM Limiter:")
    overshoots_ppm = jnp.sum((qL_ppm > 1.0) | (qR_ppm > 1.0) | (qL_ppm < 0.0) | (qR_ppm < 0.0))
    print(f"  Overshoots/undershoots: {overshoots_ppm}")
    print(f"  Max value: {jnp.max(jnp.maximum(qL_ppm, qR_ppm)):.6f}")
    print(f"  Min value: {jnp.min(jnp.minimum(qL_ppm, qR_ppm)):.6f}")

    print("\nVan Leer Limiter:")
    overshoots_vl = jnp.sum((qL_vl > 1.0) | (qR_vl > 1.0) | (qL_vl < 0.0) | (qR_vl < 0.0))
    print(f"  Overshoots/undershoots: {overshoots_vl}")
    print(f"  Max value: {jnp.max(jnp.maximum(qL_vl, qR_vl)):.6f}")
    print(f"  Min value: {jnp.min(jnp.minimum(qL_vl, qR_vl)):.6f}")

    print("\nNote: Both limiters should prevent overshoots")
    print()


def example_3_extrema_preservation():
    """Example 3: Behavior at local extrema."""
    print("=" * 60)
    print("Example 3: Local Extrema Preservation")
    print("=" * 60)

    # Profile with local maximum
    q = jnp.array([1.0, 2.0, 3.0, 5.0, 3.0, 2.0, 1.0])

    print("\nProfile with local maximum at index 3:")
    print(f"  q = {q}")

    # Reconstruct with both limiters
    qL_ppm, qR_ppm = reconstruction.ppm_reconstruction_1d(q, axis=0, limiter="ppm")
    qL_vl, qR_vl = reconstruction.ppm_reconstruction_1d(q, axis=0, limiter="van_leer")

    idx = 3  # Index of local maximum
    print(f"\nAt maximum (index {idx}, q={q[idx]:.1f}):")

    print("\nPPM Limiter:")
    print(f"  qL[{idx}] = {qL_ppm[idx]:.4f}")
    print(f"  qR[{idx}] = {qR_ppm[idx]:.4f}")
    print(f"  Flattened: {jnp.abs(qL_ppm[idx] - q[idx]) < 0.01 and jnp.abs(qR_ppm[idx] - q[idx]) < 0.01}")

    print("\nVan Leer Limiter:")
    print(f"  qL[{idx}] = {qL_vl[idx]:.4f}")
    print(f"  qR[{idx}] = {qR_vl[idx]:.4f}")
    print(f"  Flattened: {jnp.abs(qL_vl[idx] - q[idx]) < 0.01 and jnp.abs(qR_vl[idx] - q[idx]) < 0.01}")

    print("\nNote: Both should flatten at local extrema")
    print()


def example_4_differentiability():
    """Example 4: Test differentiability with jax.grad."""
    print("=" * 60)
    print("Example 4: Differentiability Test")
    print("=" * 60)

    # Define a loss function that uses reconstruction
    def loss_ppm(q):
        qL, qR = reconstruction.ppm_reconstruction_1d(q, axis=0, limiter="ppm")
        return jnp.sum((qL - qR)**2)

    def loss_van_leer(q):
        qL, qR = reconstruction.ppm_reconstruction_1d(q, axis=0, limiter="van_leer")
        return jnp.sum((qL - qR)**2)

    # Test input
    q = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    print("\nTest profile:")
    print(f"  q = {q}")

    # Try to compute gradients
    print("\nVan Leer Limiter:")
    try:
        grad_vl = jax.grad(loss_van_leer)(q)
        print(f"  ✓ Gradient computed successfully")
        print(f"  Gradient: {grad_vl}")
    except Exception as e:
        print(f"  ✗ Gradient computation failed: {e}")

    print("\nPPM Limiter:")
    try:
        grad_ppm = jax.grad(loss_ppm)(q)
        print(f"  ✓ Gradient computed successfully")
        print(f"  Gradient: {grad_ppm}")
    except Exception as e:
        print(f"  ✗ Gradient computation failed: {e}")

    print("\nNote: Van Leer is fully differentiable")
    print("      PPM uses jnp.where which may have gradient issues")
    print()


def example_5_2d_advection_comparison():
    """Example 5: 2D advection with both limiters."""
    print("=" * 60)
    print("Example 5: 2D Advection Comparison")
    print("=" * 60)

    # Create 2D Gaussian
    nx, ny = 32, 32
    x = jnp.linspace(-2, 2, nx)
    y = jnp.linspace(-2, 2, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    q = jnp.exp(-(X**2 + Y**2))

    print("\n2D Gaussian profile:")
    print(f"  Grid: {nx} x {ny}")
    print(f"  Initial mass: {jnp.sum(q):.6f}")
    print(f"  Initial max: {jnp.max(q):.6f}")

    # Courant numbers
    courant_x = jnp.ones((nx, ny)) * 0.3
    courant_y = jnp.ones((nx, ny)) * 0.2

    # Advect with PPM limiter
    q_ppm = reconstruction.ffsl_reconstruction_2d(q, courant_x, courant_y, limiter="ppm")

    print("\nAfter advection with PPM limiter:")
    print(f"  Final mass: {jnp.sum(q_ppm):.6f}")
    print(f"  Conservation error: {abs(jnp.sum(q_ppm) - jnp.sum(q)):.8f}")
    print(f"  Final max: {jnp.max(q_ppm):.6f}")

    # Advect with Van Leer limiter
    q_vl = reconstruction.ffsl_reconstruction_2d(q, courant_x, courant_y, limiter="van_leer")

    print("\nAfter advection with Van Leer limiter:")
    print(f"  Final mass: {jnp.sum(q_vl):.6f}")
    print(f"  Conservation error: {abs(jnp.sum(q_vl) - jnp.sum(q)):.8f}")
    print(f"  Final max: {jnp.max(q_vl):.6f}")

    # Compare results
    difference = jnp.mean(jnp.abs(q_ppm - q_vl))
    print(f"\nMean absolute difference: {difference:.8f}")
    print("Note: Small difference indicates similar performance")
    print()


def example_6_multiple_steps():
    """Example 6: Multiple advection steps."""
    print("=" * 60)
    print("Example 6: Multiple Advection Steps")
    print("=" * 60)

    # Initial Gaussian
    x = jnp.linspace(0, 2*jnp.pi, 64)
    q0 = 1.0 + 0.5 * jnp.sin(x)

    print("\nInitial profile: q = 1.0 + 0.5*sin(x)")
    print(f"Initial mass: {jnp.sum(q0):.6f}")

    nsteps = 10
    courant = 0.3

    # Advect with PPM
    q_ppm = q0.copy()
    for _ in range(nsteps):
        q_2d = q_ppm[:, jnp.newaxis]
        courant_x = jnp.ones_like(q_2d) * courant
        courant_y = jnp.zeros_like(q_2d)
        q_2d_new = reconstruction.ffsl_reconstruction_2d(q_2d, courant_x, courant_y, limiter="ppm")
        q_ppm = q_2d_new[:, 0]

    # Advect with Van Leer
    q_vl = q0.copy()
    for _ in range(nsteps):
        q_2d = q_vl[:, jnp.newaxis]
        courant_x = jnp.ones_like(q_2d) * courant
        courant_y = jnp.zeros_like(q_2d)
        q_2d_new = reconstruction.ffsl_reconstruction_2d(q_2d, courant_x, courant_y, limiter="van_leer")
        q_vl = q_2d_new[:, 0]

    print(f"\nAfter {nsteps} steps:")

    print("\nPPM Limiter:")
    print(f"  Final mass: {jnp.sum(q_ppm):.6f}")
    print(f"  Mass conservation error: {abs(jnp.sum(q_ppm) - jnp.sum(q0)):.8f}")
    print(f"  Max value: {jnp.max(q_ppm):.6f} (initial: {jnp.max(q0):.6f})")

    print("\nVan Leer Limiter:")
    print(f"  Final mass: {jnp.sum(q_vl):.6f}")
    print(f"  Mass conservation error: {abs(jnp.sum(q_vl) - jnp.sum(q0)):.8f}")
    print(f"  Max value: {jnp.max(q_vl):.6f} (initial: {jnp.max(q0):.6f})")

    print("\nNote: Accumulated errors show long-term stability")
    print()


def example_7_performance_comparison():
    """Example 7: Performance comparison."""
    print("=" * 60)
    print("Example 7: Performance Comparison")
    print("=" * 60)

    # Large 2D field
    nx, ny = 128, 128
    q = jnp.ones((nx, ny))
    courant_x = jnp.ones((nx, ny)) * 0.3
    courant_y = jnp.ones((nx, ny)) * 0.2

    print(f"\nGrid size: {nx} x {ny}")

    # JIT compile both versions
    ffsl_ppm = jax.jit(lambda q, cx, cy: reconstruction.ffsl_reconstruction_2d(q, cx, cy, limiter="ppm"))
    ffsl_vl = jax.jit(lambda q, cx, cy: reconstruction.ffsl_reconstruction_2d(q, cx, cy, limiter="van_leer"))

    # Warmup
    _ = ffsl_ppm(q, courant_x, courant_y).block_until_ready()
    _ = ffsl_vl(q, courant_x, courant_y).block_until_ready()

    # Timing
    import time

    # PPM timing
    start = time.time()
    for _ in range(10):
        _ = ffsl_ppm(q, courant_x, courant_y).block_until_ready()
    time_ppm = (time.time() - start) / 10

    # Van Leer timing
    start = time.time()
    for _ in range(10):
        _ = ffsl_vl(q, courant_x, courant_y).block_until_ready()
    time_vl = (time.time() - start) / 10

    print(f"\nPPM Limiter: {time_ppm*1000:.3f} ms per step")
    print(f"Van Leer Limiter: {time_vl*1000:.3f} ms per step")
    print(f"Speedup: {time_ppm/time_vl:.2f}x")

    print("\nNote: Van Leer is typically faster due to simplicity")
    print()


def main():
    """Run all limiter comparison examples."""
    print("\n" + "=" * 60)
    print("PPM vs Van Leer Limiter Comparison")
    print("=" * 60 + "\n")

    example_1_smooth_profile()
    example_2_discontinuous_profile()
    example_3_extrema_preservation()
    example_4_differentiability()
    example_5_2d_advection_comparison()
    example_6_multiple_steps()
    example_7_performance_comparison()

    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    print("\nPPM Limiters:")
    print("  ✓ More accurate on smooth profiles")
    print("  ✓ Well-tested in atmospheric models")
    print("  ✗ More complex implementation")
    print("  ✗ May have gradient issues with jax.grad")

    print("\nVan Leer Limiters:")
    print("  ✓ Simpler and more robust")
    print("  ✓ Fully differentiable")
    print("  ✓ Faster execution")
    print("  ✗ Slightly more dissipative on smooth profiles")

    print("\nRecommendation:")
    print("  - Use PPM for production runs with smooth data")
    print("  - Use Van Leer for optimization/inverse problems")
    print("  - Use Van Leer for ML applications requiring gradients")
    print("=" * 60)


if __name__ == "__main__":
    main()
