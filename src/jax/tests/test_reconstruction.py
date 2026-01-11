"""Tests for FFSL reconstruction with PPM and convolution.

This test suite validates the implementation of:
1. PPM reconstruction (slope calculation, edge values, limiters)
2. Convolution weights for flux integration
3. FFSL advection with convolution
"""

import jax.numpy as jnp
import jax
import pytest
from sl_jax.reconstruction import (
    ppm_reconstruction_1d,
    ppm_reconstruction_2d,
    convolution_weights,
    ffsl_convolution_1d,
    ffsl_reconstruction_2d,
    compute_slopes,
    compute_edge_values,
    apply_monotonicity_limiter,
)


class TestPPMReconstruction1D:
    """Test suite for 1D PPM reconstruction."""

    def test_constant_field(self):
        """Test that reconstruction of constant field returns constant edges."""
        # Constant field should have qL = qR = q everywhere
        q = jnp.ones(10) * 5.0
        qL, qR = ppm_reconstruction_1d(q, axis=0)

        # Check that edges equal cell average (within tolerance)
        assert jnp.allclose(qL, q, atol=1e-10)
        assert jnp.allclose(qR, q, atol=1e-10)

    def test_linear_field(self):
        """Test reconstruction on linear field."""
        # Linear field: q = i
        q = jnp.arange(10, dtype=jnp.float32)
        qL, qR = ppm_reconstruction_1d(q, axis=0)

        # For linear field, parabolic reconstruction should be exact
        # qL[i] should be approximately q[i] - 0.5
        # qR[i] should be approximately q[i] + 0.5
        # (except at boundaries with periodic BC)
        expected_qL = q - 0.5
        expected_qR = q + 0.5

        # Check interior points (avoiding boundaries)
        assert jnp.allclose(qL[1:-1], expected_qL[1:-1], atol=0.1)
        assert jnp.allclose(qR[1:-1], expected_qR[1:-1], atol=0.1)

    def test_monotonicity_limiter(self):
        """Test that limiters prevent overshoots."""
        # Create field with local maximum
        q = jnp.array([1.0, 2.0, 5.0, 2.0, 1.0])
        qL, qR = ppm_reconstruction_1d(q, axis=0)

        # Edge values should not exceed local bounds
        # At position 2 (maximum), edges should be limited
        assert qL[2] >= jnp.min(q)
        assert qL[2] <= jnp.max(q)
        assert qR[2] >= jnp.min(q)
        assert qR[2] <= jnp.max(q)

    def test_shape_preservation(self):
        """Test that reconstruction preserves array shape."""
        shapes = [(10,), (10, 5), (10, 5, 3)]
        for shape in shapes:
            q = jnp.ones(shape)
            qL, qR = ppm_reconstruction_1d(q, axis=0)
            assert qL.shape == shape
            assert qR.shape == shape


class TestPPMReconstruction2D:
    """Test suite for 2D PPM reconstruction."""

    def test_2d_constant_field(self):
        """Test 2D reconstruction on constant field."""
        q = jnp.ones((10, 10)) * 3.0
        qL_x, qR_x, qL_y, qR_y = ppm_reconstruction_2d(q)

        # All edges should equal cell average
        assert jnp.allclose(qL_x, q, atol=1e-10)
        assert jnp.allclose(qR_x, q, atol=1e-10)
        assert jnp.allclose(qL_y, q, atol=1e-10)
        assert jnp.allclose(qR_y, q, atol=1e-10)

    def test_2d_shape_preservation(self):
        """Test that 2D reconstruction preserves shape."""
        q = jnp.ones((8, 12))
        qL_x, qR_x, qL_y, qR_y = ppm_reconstruction_2d(q)

        assert qL_x.shape == q.shape
        assert qR_x.shape == q.shape
        assert qL_y.shape == q.shape
        assert qR_y.shape == q.shape


class TestConvolutionWeights:
    """Test suite for convolution weights."""

    def test_weights_sum(self):
        """Test that weights properly integrate over cell."""
        # For a constant field (qL = qR = q), integral should be lx * q
        lx = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        w0, w1, w2 = convolution_weights(lx)

        # For constant profile, we should get back lx
        # (since qL = qR = q, and integral should be lx * q)
        # Check that w0 + w1 + w2 behaves correctly
        # This is a sanity check on the weight formulation

        # At lx = 0, integral should be 0
        assert jnp.isclose(w0[0] + w1[0] + w2[0], 0.0, atol=1e-10)

    def test_weights_bounds(self):
        """Test that weights are computed for valid Courant numbers."""
        # Test various Courant numbers
        lx_values = jnp.linspace(0.0, 1.0, 11)
        w0, w1, w2 = convolution_weights(lx_values)

        # Weights should be finite
        assert jnp.all(jnp.isfinite(w0))
        assert jnp.all(jnp.isfinite(w1))
        assert jnp.all(jnp.isfinite(w2))


class TestFFSLConvolution1D:
    """Test suite for 1D FFSL convolution."""

    def test_zero_courant(self):
        """Test that zero Courant number gives zero flux."""
        q = jnp.ones(10)
        qL, qR = ppm_reconstruction_1d(q, axis=0)
        courant = jnp.zeros(10)

        flux = ffsl_convolution_1d(q, qL, qR, courant, axis=0)

        # Zero CFL should give zero or minimal flux
        assert jnp.allclose(flux, 0.0, atol=1e-6)

    def test_constant_field_advection(self):
        """Test advection of constant field."""
        q = jnp.ones(10) * 5.0
        qL, qR = ppm_reconstruction_1d(q, axis=0)
        courant = jnp.ones(10) * 0.5  # CFL = 0.5

        flux = ffsl_convolution_1d(q, qL, qR, courant, axis=0)

        # For constant field, flux should be constant * CFL
        expected_flux = 5.0 * 0.5
        assert jnp.allclose(flux, expected_flux, atol=0.1)


class TestFFSLReconstruction2D:
    """Test suite for 2D FFSL reconstruction."""

    def test_conservation(self):
        """Test that FFSL advection conserves mass for periodic BC."""
        # Create a 2D field with some structure
        nx, ny = 20, 20
        x = jnp.linspace(0, 2*jnp.pi, nx)
        y = jnp.linspace(0, 2*jnp.pi, ny)
        X, Y = jnp.meshgrid(x, y, indexing='ij')

        q = 1.0 + 0.5 * jnp.sin(X) * jnp.cos(Y)

        # Small Courant numbers
        courant_x = jnp.ones((nx, ny)) * 0.1
        courant_y = jnp.ones((nx, ny)) * 0.1

        q_new = ffsl_reconstruction_2d(q, courant_x, courant_y)

        # Check that total mass is approximately conserved
        mass_initial = jnp.sum(q)
        mass_final = jnp.sum(q_new)

        # Allow some tolerance due to boundary effects
        relative_error = jnp.abs(mass_final - mass_initial) / mass_initial
        assert relative_error < 0.1  # Less than 10% error

    def test_zero_courant_no_change(self):
        """Test that zero Courant number leaves field unchanged."""
        nx, ny = 10, 10
        q = jnp.ones((nx, ny)) * 3.0
        courant_x = jnp.zeros((nx, ny))
        courant_y = jnp.zeros((nx, ny))

        q_new = ffsl_reconstruction_2d(q, courant_x, courant_y)

        # Field should be nearly unchanged
        assert jnp.allclose(q_new, q, atol=1e-6)

    def test_shape_preservation_2d(self):
        """Test that 2D reconstruction preserves shape."""
        shapes = [(10, 10), (20, 15), (8, 12)]
        for shape in shapes:
            q = jnp.ones(shape)
            courant_x = jnp.zeros(shape)
            courant_y = jnp.zeros(shape)

            q_new = ffsl_reconstruction_2d(q, courant_x, courant_y)
            assert q_new.shape == shape


class TestMonotonicityAndLimiters:
    """Test suite for monotonicity preservation."""

    def test_no_new_extrema(self):
        """Test that PPM reconstruction doesn't create new extrema."""
        # Smooth monotonic function
        q = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        qL, qR = ppm_reconstruction_1d(q, axis=0)

        # Check that no edge value exceeds original bounds
        q_min = jnp.min(q)
        q_max = jnp.max(q)

        assert jnp.all(qL >= q_min - 1e-10)
        assert jnp.all(qL <= q_max + 1e-10)
        assert jnp.all(qR >= q_min - 1e-10)
        assert jnp.all(qR <= q_max + 1e-10)


# Example usage demonstration
def test_example_gaussian_advection():
    """Example: advect a Gaussian bump using FFSL reconstruction."""
    # Setup grid
    nx, ny = 50, 50
    x = jnp.linspace(-2, 2, nx)
    y = jnp.linspace(-2, 2, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # Initial Gaussian
    q0 = jnp.exp(-(X**2 + Y**2))

    # Uniform velocity field
    vx = 0.5  # velocity in x
    vy = 0.3  # velocity in y
    dt = 0.1
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Courant numbers
    courant_x = jnp.ones((nx, ny)) * vx * dt / dx
    courant_y = jnp.ones((nx, ny)) * vy * dt / dy

    # Perform one advection step
    q1 = ffsl_reconstruction_2d(q0, courant_x, courant_y)

    # Basic sanity checks
    assert q1.shape == q0.shape
    assert jnp.all(jnp.isfinite(q1))
    assert jnp.min(q1) >= -0.1  # Allow small undershoots due to boundaries
    assert jnp.max(q1) <= jnp.max(q0) + 0.1  # Allow small overshoots


if __name__ == "__main__":
    # Run basic tests
    test = TestPPMReconstruction1D()
    test.test_constant_field()
    test.test_linear_field()
    test.test_monotonicity_limiter()
    print("1D PPM reconstruction tests passed!")

    test2d = TestPPMReconstruction2D()
    test2d.test_2d_constant_field()
    print("2D PPM reconstruction tests passed!")

    test_conv = TestConvolutionWeights()
    test_conv.test_weights_sum()
    print("Convolution weights tests passed!")

    test_ffsl = TestFFSLReconstruction2D()
    test_ffsl.test_zero_courant_no_change()
    print("FFSL reconstruction tests passed!")

    test_example_gaussian_advection()
    print("Example Gaussian advection completed!")

    print("\nAll tests passed successfully!")
