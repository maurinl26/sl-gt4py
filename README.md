# semi-lagrangian-advection

Semi-Lagrangian advection schemes in plain python, DaCe and Jax.


## Semi-Lagragian implementation in python, Jax and DaCe

- [sl_dace](./src/sl_dace): point-wise semi-lagrangian in DaCe
  - [interpolation](./src/sl_dace/interpolation/):
    - [interpolation_2d](./src/sl_dace/interpolation/interpolation_2d.py) : 2d linear interpolation for classical SL
    - [flux_integral](./src/sl_dace/interpolation/flux_integral.py) : flux integrals along the trajectory for FFSL
  
  - [stencils](./src/sl_dace/stencils/):
      - [ppm.py](./src/sl_dace/stencils/ppm.py) : ppm reconstruction and limiter 
      - [ffsl.py](./src/sl_dace/stencils/ffsl.py) : stencils for 1d FFSL
      - [dep_search_1d.py](./src/sl_dace/stencils/dep_search_1d.py) : depature search for classical SL
- [sl_jax](./src/sl_jax): point-wise semi-lagrangian in Jax,
- [sl_python](./src/sl_python): raw python version (no performance).

Functional tests:
- Uniform advection on a plate :

```bash
# setup precision (by default double)
export PRECISION = double | simple

# run tests
uv run python test/functional/test_uniform.py
```
- Blossey shear test on a plate :

```bash
# setup precision (by default double)
export PRECISION = double | simple

# run tests
uv run python test/functional/test_uniform.py
```  

## Setup

The project dependencies are managed with uv

To install uv :

```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
```

To create a virtual environment :

```bash
    uv init                     # init project
    uv venv --python 3.12                     # create virtual environment
    source .venv/bin/activate   # activate virtual environment
    uv sync                     # load and synchronize project dependencies
```

## sl_python

sl_python implements the classical sl scheme in 2d.

## sl_dace



- **utils/**
  - typingx.py
 
  Set precision 

   ```bash
   
   ```

  - **dims.py/**
    I : number of points on x axis (plain levels)
    J : number of points on y axis (plain levels)
    K : number of points on z axis

    To define a field on Half-Level : dtype_float[I, J, K + 1]
  

## Build the doc 

```bash
   uv run sphinx-autobuild -M html docs/ docbuild/
```

## WIP

- ffsl_x.py / ffsl_y.py : ffsl 1d orchestration
- (WIP) ffsl_xy.py : ffsl 2d with swift splitting 
- elarche.py : departure search for classical SL
- (WIP) sl_init.py : SETTLS or NESC init orchestration
- sl_xy.py : classical SL orchestration
- (WIP) sl_driver.py : driver for full sl or ffsl schemes in dace

