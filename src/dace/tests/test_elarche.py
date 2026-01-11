import pytest
from sl_dace.elarche import Elarche
import numpy as np
import dace

@pytest.fixture(name="grid", scope="module")
def grid_fixture():
    return (50, 50, 15)


def test_elarche(grid):

    dx = np.float32(0.01)
    dy = np.float32(0.01)
    dth = np.float32(0.004)

    idx = np.ones(shape=grid, dtype=np.int32)
    jdx = np.ones(shape=grid, dtype=np.int32)

    vx = np.zeros(shape=grid, dtype=np.float32)
    vy = np.zeros(shape=grid, dtype=np.float32)
    vx_tmp = np.zeros(shape=grid, dtype=np.float32)
    vy_tmp = np.zeros(shape=grid, dtype=np.float32)
    vx_e = np.zeros(shape=grid, dtype=np.float32)
    vy_e = np.zeros(shape=grid, dtype=np.float32)

    # Outputs
    lx = np.zeros(shape=grid, dtype=np.float32)
    ly = np.zeros(shape=grid, dtype=np.float32)
    i_dep = np.ones(shape=grid, dtype=np.int32)
    j_dep = np.zeros(shape=grid, dtype=np.int32)

    Elarche(grid, nitmp=1)(
        dx=dx,
        dy=dy,
        dth=dth,
        idx=idx,
        jdx=jdx,
        vx=vx,
        vy=vy,
        vx_tmp=vx_tmp,
        vy_tmp=vy_tmp,
        vx_e=vx_e,
        vy_e=vy_e,
        # Outputs
        lx=lx,
        ly=ly,
        i_dep=i_dep,
        j_dep=j_dep,
    )

