from gt4py.cartesian.gtscript import stencil
import numpy as np
import pytest
import logging

from sl_dace.stencils.dep_search_1d import dep_search_1d
from typing import Tuple


@pytest.mark.parametrize("backend", ["numpy", "gt:cpu_ifirst", "dace:cpu"] )
def test_dep_search_1d(backend: str, dtypes: dict, inner_domain: Tuple[int], origin: Tuple[int]):

    logging.info(f"Backend : {backend}")

    stencil_dep_search_1d = stencil(backend=backend,
                                    definition=dep_search_1d,
                                    name="dep_search_1d",
                                    dtypes=dtypes,
                                    build_info={},
                                    externals={},
                                    rebuild=True,
                                    )

    # Departures
    vx_e = np.ones((50, 50, 10), dtype=dtypes[float])
    vx_tmp = np.ones((50, 50, 10), dtype=dtypes[float])
    i_a = np.ones((50, 50, 10), dtype=dtypes[int])

    # Outputs
    i_d = np.zeros((50, 50, 10), dtype=dtypes[int])
    lx = np.zeros((50, 50, 10), dtype=dtypes[float])

    dx = dtypes[float](1)
    dth = dtypes[float](0.5)

    stencil_dep_search_1d(
        vx_e=vx_e,
        vx_tmp=vx_tmp,
        i_a=i_a,
        i_d=i_d,
        lx=lx,
        dx=dx,
        dth=dth,
        domain=inner_domain,
        origin=origin
    )

    print(i_d.mean())

    # one cell left shift
    assert i_d.mean() == 0

