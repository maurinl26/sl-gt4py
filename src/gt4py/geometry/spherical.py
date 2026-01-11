# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from fvms.utils.dims import I, J
from fvms.utils.storage import to_numpy, zeros

if TYPE_CHECKING:
    from fvms.geometry.curvilinear import Coordinates, Grid
    from fvms.model.config import Model


class Spherical:
    def __init__(self, grid: Grid, coordinates: Coordinates, model: Model) -> None:
        indices = grid.indices
        self.sphere = model.sphere
        self.coslat = zeros(indices, (I, J))
        self.sinlat = zeros(indices, (I, J))
        if self.sphere:
            self.coslat["covering"] = np.cos(to_numpy(coordinates.ycr["covering"][:, :, 0]))
            self.sinlat["covering"] = np.sin(to_numpy(coordinates.ycr["covering"][:, :, 0]))
        else:
            self.coslat["covering"] = 1.0
