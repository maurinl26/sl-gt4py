# Installing JAX for sl_jax

This guide explains how to install JAX to use the `sl_jax` module.

## Prerequisites

- Python 3.9 or later
- pip or conda package manager

## Installation Options

### Option 1: CPU-only JAX (Quick Start)

For CPU-only execution and testing:

```bash
pip install jax jaxlib
```

### Option 2: GPU Support (CUDA)

For NVIDIA GPU support with CUDA 12:

```bash
pip install -U "jax[cuda12]"
```

For CUDA 11:

```bash
pip install -U "jax[cuda11]"
```

### Option 3: Using Conda

```bash
conda install -c conda-forge jax jaxlib
```

For GPU support with conda:

```bash
conda install -c conda-forge jaxlib=*=*cuda* jax cuda-nvcc
```

## Verifying Installation

After installation, verify JAX is working:

```python
import jax
import jax.numpy as jnp

# Check if GPU is available
print(f"JAX devices: {jax.devices()}")

# Simple test
x = jnp.array([1, 2, 3])
print(f"JAX array: {x}")
print(f"Sum: {jnp.sum(x)}")
```

## Running Tests

Once JAX is installed, run the functional tests:

```bash
# Run all tests
pytest tests/functional/test_sl_jax.py -v

# Run specific test
pytest tests/functional/test_sl_jax.py::test_sl_jax_basic -v

# Run tests and show output
pytest tests/functional/test_sl_jax.py -v -s
```

Or run directly:

```bash
python tests/functional/test_sl_jax.py
```

## Using sl_jax

After JAX is installed, you can use sl_jax:

```python
import jax.numpy as jnp
from jax import jit
from config import Config
from sl_jax import sl_xy

# Create configuration
config = Config(nx=100, ny=100, dx=1.0, dy=1.0, dt=0.1)

# Initialize fields as JAX arrays
vx = jnp.zeros((100, 100))
vy = jnp.zeros((100, 100))
tracer = jnp.ones((100, 100))

# JIT compile for performance
sl_xy_jit = jit(sl_xy, static_argnums=(0, 6))

# Run advection
tracer_new = sl_xy_jit(config, vx, vy, vx, vy, tracer, tracer, nitmp=4)
```

## Troubleshooting

### ImportError: No module named 'jax'

JAX is not installed. Follow installation instructions above.

### CUDA errors on GPU

Ensure you have:
1. Compatible NVIDIA GPU driver
2. Correct CUDA toolkit version
3. Matching JAX/jaxlib version for your CUDA

Check your CUDA version:
```bash
nvcc --version
```

### Memory errors

For large grids, you may need to:
1. Reduce grid size
2. Use CPU instead of GPU
3. Increase GPU memory allocation

## Performance Tips

1. **Always JIT compile hot functions**:
   ```python
   from jax import jit
   my_func_jit = jit(my_func)
   ```

2. **Keep data on device**: Minimize transfers between CPU and GPU

3. **Use appropriate precision**: `jax.config.update("jax_enable_x64", True)` for 64-bit

4. **Profile your code**:
   ```python
   with jax.profiler.trace("/tmp/jax-trace"):
       result = my_func(data)
   ```

## Further Reading

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GitHub](https://github.com/google/jax)
- [JAX Installation Guide](https://github.com/google/jax#installation)
- [JAX Tutorial](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
