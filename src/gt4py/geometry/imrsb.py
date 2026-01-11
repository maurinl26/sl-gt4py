# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from fvms.build_config import dtype
from fvms.operators.basic import set_a2cst
from fvms.utils.dims import I, J, K
from fvms.utils.storage import zeros

if TYPE_CHECKING:
    from fvms.geometry.coordinates import Coordinates, Grid
    from fvms.model.config import ImmersedBoundaries


class Imrsb:
    def __init__(self, grid: Grid, coordinates: Coordinates, imrsb: ImmersedBoundaries) -> None:
        self.enabled = imrsb.enabled
        self.orog = imrsb.orog
        self.hybrid = imrsb.hybrid
        self.relax = imrsb.relax
        self.noslip = imrsb.noslip
        self.qsbm = imrsb.qsbm
        self.tscale = imrsb.tscale
        self.tam = zeros(grid.indices, (I, J, K))
        self.atm = zeros(grid.indices, (I, J, K))
        set_a2cst(grid.indices, (I, J, K), self.atm, cst=dtype(1.0), subdomain="covering")

        zcr = coordinates.zcr
        zorog = coordinates.zorog

        self.tam["covering"][:, :, :] = np.where(
            zcr["covering"][:, :, :] <= zorog["covering"][:, :, np.newaxis], 1.0, 0.0
        )
        self.atm["covering"][:, :, :] = np.where(
            zcr["covering"][:, :, :] <= zorog["covering"][:, :, np.newaxis], 0.0, 1.0
        )
