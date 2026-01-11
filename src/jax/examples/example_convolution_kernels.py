"""Example: Understanding Convolution Kernels in FFSL/PPM Reconstruction.

This example demonstrates how jnp.convolve is used in the FFSL implementation
for efficient PPM reconstruction.

The key insight is that PPM reconstruction can be expressed as convolutions
with specific kernels, making the implementation both elegant and efficient.
"""

import jax.numpy as jnp
import sys
import importlib.util

# Import reconstruction module directly
spec = importlib.util.spec_from_file_location('reconstruction', 'src/jax/reconstruction.py')
reconstruction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconstruction)


def example_1_centered_difference_kernel():
    """Example 1: Centered difference as convolution."""
    print("=" * 60)
    print("Example 1: Centered Difference with Convolution")
    print("=" * 60)

    # The centered difference kernel: [-0.5, 0, 0.5]
    # This computes: delta[i] = 0.5 * (q[i+1] - q[i-1])

    q = jnp.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
    kernel = jnp.array([-0.5, 0.0, 0.5])

    # Apply convolution
    delta = jnp.convolve(q, kernel, mode='same')

    print("\nInput field q:")
    print(f"  {q}")
    print("\nCentered difference kernel: [-0.5, 0, 0.5]")
    print("\nComputed slopes (delta):")
    print(f"  {delta}")
    print("\nInterpretation:")
    print("  delta[i] = (q[i+1] - q[i-1]) / 2")
    print("  This estimates the local slope in each cell")
    print()


def example_2_ppm_edge_kernel():
    """Example 2: PPM edge reconstruction kernel."""
    print("=" * 60)
    print("Example 2: PPM Edge Reconstruction Kernel")
    print("=" * 60)

    # PPM 4th-order edge reconstruction kernel
    # qL[i] = (1/12)*q[i-2] - (7/12)*q[i-1] + (7/6)*q[i] + (1/12)*q[i+1]

    q = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    kernel_L = jnp.array([1/12, -7/12, 7/6, 1/12])

    # Apply convolution
    qL = jnp.convolve(q, kernel_L, mode='same')

    print("\nInput cell averages q:")
    print(f"  {q}")
    print("\nPPM edge kernel: [1/12, -7/12, 7/6, 1/12]")
    print("\nLeft edge values qL:")
    print(f"  {qL}")
    print("\nFor a linear field q[i] = i:")
    print("  qL[i] should be approximately q[i] - 0.5")
    print(f"  qL[4] = {qL[4]:.3f}, expected ≈ {q[4] - 0.5:.3f}")
    print()


def example_3_convolution_properties():
    """Example 3: Properties of convolution kernels."""
    print("=" * 60)
    print("Example 3: Convolution Kernel Properties")
    print("=" * 60)

    # Different kernels have different properties

    kernels = {
        "Centered difference": jnp.array([-0.5, 0.0, 0.5]),
        "PPM edge (4th order)": jnp.array([1/12, -7/12, 7/6, 1/12]),
        "Simple average": jnp.array([1/3, 1/3, 1/3]),
        "Gaussian-like": jnp.array([0.25, 0.5, 0.25]),
    }

    for name, kernel in kernels.items():
        print(f"\n{name}:")
        print(f"  Kernel: {kernel}")
        print(f"  Sum: {jnp.sum(kernel):.3f}")
        print(f"  Symmetry: {'Yes' if jnp.allclose(kernel, kernel[::-1]) else 'No'}")

        # Apply to constant field
        q_const = jnp.ones(10)
        result = jnp.convolve(q_const, kernel, mode='same')
        print(f"  On constant field: {jnp.mean(result):.3f} (should be 1.0 for conservative)")
    print()


def example_4_mode_comparison():
    """Example 4: Convolution modes (same, valid, full)."""
    print("=" * 60)
    print("Example 4: Convolution Modes")
    print("=" * 60)

    q = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    kernel = jnp.array([-0.5, 0.0, 0.5])

    print("\nInput field q:")
    print(f"  {q}")
    print(f"  Length: {len(q)}")
    print("\nKernel:")
    print(f"  {kernel}")
    print(f"  Length: {len(kernel)}")

    # Different modes
    result_same = jnp.convolve(q, kernel, mode='same')
    result_valid = jnp.convolve(q, kernel, mode='valid')
    result_full = jnp.convolve(q, kernel, mode='full')

    print("\nMode='same' (same length as input):")
    print(f"  {result_same}")
    print(f"  Length: {len(result_same)}")

    print("\nMode='valid' (no boundary effects):")
    print(f"  {result_valid}")
    print(f"  Length: {len(result_valid)}")

    print("\nMode='full' (complete convolution):")
    print(f"  {result_full}")
    print(f"  Length: {len(result_full)}")

    print("\nNote: We use 'same' for periodic/continuous domains")
    print()


def example_5_ppm_reconstruction_with_convolution():
    """Example 5: Complete PPM reconstruction using convolution."""
    print("=" * 60)
    print("Example 5: Full PPM Reconstruction with Convolution")
    print("=" * 60)

    # Create a smooth profile
    x = jnp.linspace(0, 2*jnp.pi, 32)
    q = 1.0 + 0.5 * jnp.sin(x)

    print("\nInput: Sinusoidal profile")
    print(f"  q = 1.0 + 0.5*sin(x)")
    print(f"  Grid points: {len(q)}")
    print(f"  Mean value: {jnp.mean(q):.4f}")

    # Perform PPM reconstruction
    qL, qR = reconstruction.ppm_reconstruction_1d(q, axis=0)

    print("\nAfter PPM reconstruction:")
    print(f"  Left edges (qL) mean: {jnp.mean(qL):.4f}")
    print(f"  Right edges (qR) mean: {jnp.mean(qR):.4f}")
    print(f"  Cell averages (q) mean: {jnp.mean(q):.4f}")

    # Check consistency: qR[i] = qL[i+1] for smooth fields
    qL_shifted = jnp.roll(qL, -1)
    edge_consistency = jnp.mean(jnp.abs(qR - qL_shifted))
    print(f"\nEdge consistency: {edge_consistency:.6f}")
    print("  (Small value indicates consistent reconstruction)")

    # Check that edges bracket the cell average
    print("\nMonotonicity check:")
    bracketed = jnp.logical_and(
        jnp.logical_or(qL <= q, qR <= q),
        jnp.logical_or(qL >= q, qR >= q)
    )
    print(f"  Cells with q between qL and qR: {jnp.sum(bracketed)}/{len(q)}")
    print()


def example_6_flux_computation_with_convolution():
    """Example 6: Flux computation using convolution."""
    print("=" * 60)
    print("Example 6: Flux Computation with Convolution")
    print("=" * 60)

    # Create a simple profile
    q = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0])

    print("\nInput profile q:")
    print(f"  {q}")

    # Reconstruct
    qL, qR = reconstruction.ppm_reconstruction_1d(q, axis=0)

    print("\nReconstructed edges:")
    print(f"  qL: {qL}")
    print(f"  qR: {qR}")

    # Compute flux with different Courant numbers
    courant_values = [0.1, 0.3, 0.5, 0.7]

    print("\nFlux computation for different CFL:")
    for cfl in courant_values:
        courant = jnp.ones_like(q) * cfl
        flux = reconstruction.ffsl_convolution_1d_direct(q, qL, qR, courant, axis=0)
        print(f"  CFL = {cfl}: flux mean = {jnp.mean(flux):.4f}")

    print("\nNote: Flux increases with Courant number")
    print("      Flux ≈ CFL * q for constant q")
    print()


def example_7_2d_convolution():
    """Example 7: 2D FFSL with convolution."""
    print("=" * 60)
    print("Example 7: 2D FFSL Reconstruction")
    print("=" * 60)

    # Create 2D Gaussian
    nx, ny = 32, 32
    x = jnp.linspace(-2, 2, nx)
    y = jnp.linspace(-2, 2, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    q = jnp.exp(-(X**2 + Y**2))

    print("\n2D Gaussian bump:")
    print(f"  Grid: {nx} x {ny}")
    print(f"  Initial mass: {jnp.sum(q):.4f}")
    print(f"  Max value: {jnp.max(q):.4f}")

    # Apply 2D reconstruction
    qL_x, qR_x, qL_y, qR_y = reconstruction.ppm_reconstruction_2d(q)

    print("\nAfter 2D PPM reconstruction:")
    print(f"  X-direction edges computed via convolution")
    print(f"  Y-direction edges computed via convolution")

    # Advect with uniform velocity
    courant_x = jnp.ones((nx, ny)) * 0.3
    courant_y = jnp.ones((nx, ny)) * 0.2

    q_new = reconstruction.ffsl_reconstruction_2d(q, courant_x, courant_y)

    print("\nAfter FFSL advection:")
    print(f"  Final mass: {jnp.sum(q_new):.4f}")
    print(f"  Conservation error: {abs(jnp.sum(q_new) - jnp.sum(q)):.6f}")
    print(f"  Max value: {jnp.max(q_new):.4f}")
    print()


def example_8_kernel_visualization():
    """Example 8: Visualize convolution kernels."""
    print("=" * 60)
    print("Example 8: Convolution Kernel Visualization")
    print("=" * 60)

    print("\nCentered Difference Kernel:")
    kernel_diff = jnp.array([-0.5, 0.0, 0.5])
    print("  Position:  i-1   i   i+1")
    print(f"  Weight:    {kernel_diff[0]:.2f}  {kernel_diff[1]:.2f}  {kernel_diff[2]:.2f}")
    print("  Effect: Approximates derivative")

    print("\nPPM Edge Reconstruction Kernel:")
    kernel_ppm = jnp.array([1/12, -7/12, 7/6, 1/12])
    print("  Position:  i-2    i-1    i     i+1")
    print(f"  Weight:    {kernel_ppm[0]:.3f}  {kernel_ppm[1]:.3f}  {kernel_ppm[2]:.3f}  {kernel_ppm[3]:.3f}")
    print("  Effect: 4th-order accurate edge interpolation")

    print("\nKey Properties:")
    print("  • Sum of weights determines conservation")
    print("  • Symmetry determines odd/even function behavior")
    print("  • Width determines stencil size and accuracy")
    print("  • Central weighting determines smoothness")
    print()


def main():
    """Run all convolution examples."""
    print("\n" + "=" * 60)
    print("FFSL/PPM Reconstruction with jnp.convolve")
    print("Understanding Convolution Kernels")
    print("=" * 60 + "\n")

    example_1_centered_difference_kernel()
    example_2_ppm_edge_kernel()
    example_3_convolution_properties()
    example_4_mode_comparison()
    example_5_ppm_reconstruction_with_convolution()
    example_6_flux_computation_with_convolution()
    example_7_2d_convolution()
    example_8_kernel_visualization()

    print("=" * 60)
    print("All convolution examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
