import pytest
import numpy as np
from sl_dace.interpolation.interpolation_2d import interpolate_lin_2d
import dace
from sl_dace.dims import I, J, K


def simple_interpolation(
        psi: dace.float32[I, J],
        psi_int: dace.float32[I, J],
        i_dep: dace.uint32[I, J]
    ):
    for i, j in dace.map[0:I, 0:J]:

            psi[i, j] = psi[i_dep[i, j], j]


            # todo : check index resolution


def test_simple_interpolation():

    c_simple_interpolation = (
        dace.program(simple_interpolation)
        .to_sdfg()
        .compile()
    )

    psi = np.ones((10, 10), dtype=np.float32)
    psi_int = np.ones((10, 10), dtype=np.float32)
    i_dep = np.ones((10, 10), dtype=np.int32)

    c_simple_interpolation(
        psi=psi,
        psi_int=psi_int,
        i_dep=i_dep,
        I=10,
        J=10
    )

def test_interpolate_lin_2d():

    nx = 100
    ny = 100
    nz = 90

    psi = np.ones((nx, ny, nz), dtype=np.float32)
    lx = np.zeros((nx, ny, nz), dtype=np.float32)
    ly = np.zeros((nx, ny, nz), dtype=np.float32)

    i_d = np.ones((nx, ny, nz), dtype=np.int32)
    j_d = np.ones((nx, ny, nz), dtype=np.int32)

    psi_dep = np.zeros((nx, ny, nz), dtype=np.float32)

    h = 0

    c_interpolate_lin_2d = (
        dace.program(interpolate_lin_2d)
        .to_sdfg()
        .compile()
    )

    c_interpolate_lin_2d(
            psi=psi,
            lx=lx,
            ly=ly,
            i_dep=i_d,
            j_dep=j_d,
            psi_dep=psi_dep,
            I=nx,
            J=ny,
            K=nz,
            H=h
    )

    assert psi.mean() == 0.0