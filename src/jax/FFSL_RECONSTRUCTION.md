# FFSL Reconstruction with PPM and Convolution

This document describes the implementation of Flux-Form Semi-Lagrangian (FFSL) reconstruction using Piecewise Parabolic Method (PPM) with convolution operations in JAX.

## Overview

The FFSL method combines:
1. **High-order spatial reconstruction** using PPM for third-order accuracy
2. **Flux-form advection** for exact mass conservation
3. **Monotonicity limiters** to prevent spurious oscillations
4. **Convolution-based flux integration** for efficient computation

## Mathematical Background

### PPM Reconstruction

For each cell `i`, we reconstruct a parabolic profile:

```
q(s) = qL[i] + s*(qR[i] - qL[i]) + s*(1-s)*a6[i]
```

where:
- `s ∈ [0, 1]` is the position within the cell
- `qL[i]` is the left edge value
- `qR[i]` is the right edge value
- `a6[i] = 6*(q[i] - 0.5*(qL[i] + qR[i]))` is the curvature coefficient

#### Edge Value Computation

Left edge values are computed using fourth-order interpolation:

```
qL[i] = q[i] + 0.5*(q[i] - q[i+1]) + (δq[i] + δq[i+1])/6
```

where `δq[i] = (q[i+1] - q[i-1])/2` is the cell-averaged slope.

Right edge values satisfy:
```
qR[i] = qL[i+1]
```

#### Monotonicity Limiters

To prevent oscillations, we apply limiters that ensure:

1. **Bounds preservation**: Edge values lie between neighboring cell averages
   ```
   q_min = min(q[i-1], q[i], q[i+1])
   q_max = max(q[i-1], q[i], q[i+1])
   qL[i] = clip(qL[i], q_min, q_max)
   qR[i] = clip(qR[i], q_min, q_max)
   ```

2. **No new extrema**: If the parabola has a local extremum within the cell, flatten to cell average
   ```
   if (qR[i] - q[i]) * (q[i] - qL[i]) ≤ 0:
       qL[i] = qR[i] = q[i]
   ```

3. **Curvature limiting**: Prevent excessive steepness
   ```
   if da * a6 > da²:
       qL[i] = 3*q[i] - 2*qR[i]
   if da * a6 < -da²:
       qR[i] = 3*q[i] - 2*qL[i]
   ```

### Convolution-Based Flux Integration

For a given Courant number `C = v*dt/dx`, the flux is computed by integrating the parabolic profile:

```
F[i] = ∫₀^C q(s) ds = w0*qL[i] + w1*qR[i] + w2*q[i]
```

where the convolution weights are:

```
w0 = C - 0.5*C² + C² - 2*C³
w1 = 0.5*C² - C² + 2*C³
w2 = 2*(C² - C³)
```

## Implementation Details

### Module Structure

The implementation is in `src/jax/reconstruction.py` with the following key functions:

- `ppm_reconstruction_1d(q, axis)`: Performs 1D PPM reconstruction along specified axis
- `ppm_reconstruction_2d(q)`: Performs 2D PPM reconstruction in both x and y directions
- `convolution_weights(lx)`: Computes weights for flux integration
- `ffsl_convolution_1d(q, qL, qR, courant, axis)`: Computes 1D flux using convolution
- `ffsl_reconstruction_2d(q, courant_x, courant_y)`: Main 2D FFSL advection interface

### Performance Considerations

The implementation leverages JAX features for performance:

1. **JIT compilation**: All functions can be JIT-compiled with `jax.jit`
2. **Vectorization**: Operations are fully vectorized using JAX array operations
3. **Automatic differentiation**: Compatible with `jax.grad` for sensitivity analysis
4. **GPU acceleration**: Runs on GPU when JAX is configured for CUDA

## Usage Examples

### Basic 2D Advection

```python
import jax.numpy as jnp
from sl_jax.reconstruction import ffsl_reconstruction_2d

# Setup
nx, ny = 64, 64
q = jnp.ones((nx, ny))  # Initial field

# Courant numbers (CFL condition)
vx, vy = 0.5, 0.3  # velocities
dt = 0.1
dx, dy = 0.01, 0.01
courant_x = jnp.ones((nx, ny)) * vx * dt / dx
courant_y = jnp.ones((nx, ny)) * vy * dt / dy

# Perform FFSL advection
q_new = ffsl_reconstruction_2d(q, courant_x, courant_y)
```

### Time Integration Loop

```python
# Time stepping with FFSL
nsteps = 100
for step in range(nsteps):
    q = ffsl_reconstruction_2d(q, courant_x, courant_y)
```

### With JIT Compilation

```python
import jax

# JIT compile for performance
ffsl_jit = jax.jit(ffsl_reconstruction_2d)

# Use compiled version
q_new = ffsl_jit(q, courant_x, courant_y)
```

## Testing

Run the test suite to validate the implementation:

```bash
cd src/jax/tests
python test_reconstruction.py
```

Or with pytest:

```bash
pytest src/jax/tests/test_reconstruction.py -v
```

### Test Coverage

The test suite includes:

1. **Constant field tests**: Verify reconstruction of uniform fields
2. **Linear field tests**: Check accuracy on linear profiles
3. **Monotonicity tests**: Ensure limiters prevent overshoots
4. **Conservation tests**: Verify mass conservation in flux-form
5. **Convergence tests**: Validate spatial accuracy order

## Examples

Run the comprehensive examples:

```bash
python src/jax/examples/example_ffsl_reconstruction.py
```

This includes:
- 1D advection of different profiles (Gaussian, square wave, sine)
- 2D vortex advection
- PPM reconstruction analysis
- Monotonicity limiter effectiveness
- Grid convergence study

## Properties

### Advantages

1. **Exact mass conservation**: Flux-form guarantees conservation
2. **High accuracy**: Third-order spatial accuracy with PPM
3. **Monotonicity**: Limiters prevent spurious oscillations
4. **Efficiency**: Convolution avoids expensive particle tracking
5. **Unconditionally stable**: No CFL restriction for linear advection

### Limitations

1. **Boundary treatment**: Requires special handling at domain boundaries
2. **Multidimensional splitting**: Sequential application in x and y directions
3. **Non-periodic domains**: May require boundary condition modifications
4. **Computational cost**: Higher than first-order methods

## CFL Condition

While FFSL is unconditionally stable for linear advection, accuracy degrades for large Courant numbers. Recommended:

```
CFL = |v| * dt / dx < 1
```

For best accuracy with PPM:
```
CFL < 0.5
```

## References

1. **Colella, P., & Woodward, P. R. (1984)**. "The Piecewise Parabolic Method (PPM) for gas-dynamical simulations." *Journal of Computational Physics*, 54(1), 174-201.

2. **Lin, S. J., & Rood, R. B. (1996)**. "Multidimensional flux-form semi-Lagrangian transport schemes." *Monthly Weather Review*, 124(9), 2046-2070.

3. **Zerroukat, M., Wood, N., & Staniforth, A. (2002)**. "SLICE: A Semi-Lagrangian Inherently Conserving and Efficient scheme for transport problems." *Quarterly Journal of the Royal Meteorological Society*, 128(586), 2801-2820.

## Future Enhancements

Potential improvements for future versions:

1. **Higher-order reconstruction**: Implement quintic or WENO schemes
2. **Adaptive limiting**: Dynamic limiter strength based on local smoothness
3. **Dimensionally-unsplit advection**: True 2D reconstruction
4. **Divergent flows**: Extension to compressible/divergent velocity fields
5. **Vertical coordinate**: Irregular grid spacing in z-direction

## Contact

For questions or issues related to FFSL reconstruction:
- Check the test suite for usage examples
- Review the inline documentation in `reconstruction.py`
- Run the examples in `examples/example_ffsl_reconstruction.py`
