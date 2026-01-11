# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import TYPE_CHECKING

from gt4py.cartesian import gtscript

from fvms.build_config import backend, backend_opts, dtype
from fvms.utils.dims import I, J, K
from fvms.utils.stencil import stencil

if TYPE_CHECKING:
    from fvms.geometry.coordinates import Grid
    from fvms.utils.storage import Field


def delta_zcr(grid: Grid, dzcr: Field, zcr: Field) -> None:
    _delta_zcr(
        grid.indices,
        (I, J, K),
        dzcr=dzcr,
        zcr=zcr,
        subdomain="covering",
    )


# todo(ckuehnlein): there may be other specifcations depending on vertical discretization
@stencil(backend=backend, **backend_opts)
def _delta_zcr(dzcr: gtscript.Field[dtype], zcr: gtscript.Field[dtype]):
    with computation(PARALLEL):
        with interval(0, 1):
            dzcr[0, 0, 0] = zcr[0, 0, 1] - zcr[0, 0, 0]
        with interval(1, -1):
            dzcr[0, 0, 0] = 0.5 * (zcr[0, 0, 1] - zcr[0, 0, -1])
        with interval(-1, None):
            dzcr[0, 0, 0] = zcr[0, 0, 0] - zcr[0, 0, -1]
