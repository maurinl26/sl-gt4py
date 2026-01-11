# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import TYPE_CHECKING

from fvms.build_config import dtype
from fvms.utils.dims import I, J, K
from fvms.utils.storage import zeros

if TYPE_CHECKING:
    from fvms.geometry.spherical import Spherical
    from fvms.geometry.coordinates import Coordinates, Grid
    from fvms.geometry.curvilinear import Metric
    from fvms.model.config import Constants, Model

"""
Define fundamental rhs momentum equation forcing fields
"""


class Forces:
    def __init__(
        self,
        grid: Grid,
        coordinates: Coordinates,
        metric: Metric,
        spherical: Spherical,
        model: Model,
        constants: Constants,
    ) -> None:
        indices = grid.indices
        self.gravity_x = 0
        self.gravity_y = 0
        self.gravity_z = zeros(indices, (I, J, K))
        self.coriolis_x = 0
        self.coriolis_y = zeros(indices, (I, J))
        self.coriolis_z = zeros(indices, (I, J))
        self.gravity_z["covering"] = -constants.gravity0 / (metric.gmm["covering"] ** 2)
        ycr0 = 0.5 * (grid.ymin + grid.ymax)
        if model.sphere:
            self.coriolis_y["covering"] = (
                model.deep * dtype(constants.fcoriolis0) * spherical.coslat["covering"]
            )
            self.coriolis_z["covering"] = dtype(constants.fcoriolis0) * spherical.sinlat["covering"]
        else:
            self.coriolis_y["covering"] = 0
            self.coriolis_z["covering"] = dtype(constants.fcr0) + int(model.beta_plane) * dtype(
                constants.beta0
            ) * (coordinates.ycr["covering"][:, :, 0] - ycr0)
