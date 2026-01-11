# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from functools import cached_property
import numpy as np
from typing import TYPE_CHECKING

from sl_gt4py.config import dtype
from sl_gt4py.geometry.zdelta import delta_zcr
from sl_gt4py.utils.dims import I, J, K
from sl_gt4py.utils.index_space import ProductSet
from sl_gt4py.utils.periodic_bcs import update_periodic_layers
from sl_gt4py.utils.storage import translate_to_memory_index_convention, zeros

if TYPE_CHECKING:
    from typing import Optional

    from sl_gt4py.utils.indices import Indices
    from sl_gt4py.utils.typingx import Pair, Triple
    from sl_gt4py.utils.storage import Field


class Grid:
    """Representation of the three-dimensional regular computational grid."""

    # todo(ckuehnlein, stubbiali, tehrengruber): resolve inconsist units of coordinates and spacings
    # in Grid at a later stage when analytical model options and form of generalised coordinates have
    # been devised.

    indices: Indices
    bounds: Triple[Pair[dtype]]
    sphere: bool
    radius_sphere: dtype
    deg2rad_sphere: dtype
    rad2deg_sphere: dtype

    def __init__(
        self, indices: Indices, bounds: Triple[Pair[float]], sphere: bool, radius_sphere: dtype
    ) -> None:
        """
        The grid consists of the points contained in ``indices`` and spans the control volume
        whose bounds along each direction are specified by ``bounds``.
        """
        self.indices = indices
        self.bounds = tuple((dtype(bound[0]), dtype(bound[1])) for bound in bounds)
        self.sphere = sphere
        self.radius_sphere = radius_sphere
        self.deg2rad_sphere = dtype(math.pi / 180.0 * self.sphere + (1.0 - self.sphere))
        self.rad2deg_sphere = dtype(180.0 / math.pi * self.sphere + (1.0 - self.sphere))

    @cached_property
    def xc(self) -> np.ndarray:
        return np.linspace(self.xmin, self.xmax, self.shape[0])

    @cached_property
    def yc(self) -> np.ndarray:
        return np.linspace(self.ymin, self.ymax, self.shape[1])

    @cached_property
    def zc(self) -> np.ndarray:
        return np.linspace(self.zmin, self.zmax, self.shape[2])

    @cached_property
    def xmin(self) -> dtype:
        return self.bounds[0][0]

    @cached_property
    def ymin(self) -> dtype:
        return self.bounds[1][0]

    @cached_property
    def zmin(self) -> dtype:
        return self.bounds[2][0]

    @cached_property
    def xmax(self) -> dtype:
        return self.bounds[0][1]

    @cached_property
    def ymax(self) -> dtype:
        return self.bounds[1][1]

    @cached_property
    def zmax(self) -> dtype:
        return self.bounds[2][1]

    @cached_property
    def shape(self) -> Triple[int]:
        """Number of points along each dimension."""
        assert isinstance(self.indices.index_spaces[I, J, K].subset["definition"], ProductSet)
        return self.indices.index_spaces[I, J, K].subset["definition"].shape

    @cached_property
    def spacings(self):
        return tuple(
            [
                dtype(
                    ((self.bounds[0][1] - self.bounds[0][0]) / (self.shape[0] - 1))
                    * (1 - self.sphere + self.radius_sphere * self.sphere)
                ),
                dtype(
                    ((self.bounds[1][1] - self.bounds[1][0]) / (self.shape[1] - 1))
                    * (1 - self.sphere + self.radius_sphere * self.sphere)
                ),
                dtype(((self.bounds[2][1] - self.bounds[2][0]) / (self.shape[2] - 1))),
            ]
        )


# The horizontal coordinates supported by the model (at least for now) are
#  two dimensional. To emphasize the dependency on this invariant elsewhere,
#  we wrap the two coordinate arrays in this class.
class HorizontalCoordinates:
    xcr: np.ndarray  # 2d
    ycr: np.ndarray  # 2d

    def __init__(self, xcr: np.ndarray, ycr: np.ndarray):
        self.xcr = xcr
        self.ycr = ycr


class Coordinates:
    """
    Representation of the horizontal and vertical physical coordinates, as well as orography heights
    and vertical stretching.
    """

    zorog: Field
    zorog_smooth: Optional[Field]

    xcr: Field
    ycr: Field
    zcr: Field

    def __init__(
        self,
        grid: Grid,
        horizontal_coordinates: HorizontalCoordinates,
        zcr: np.ndarray,
        zorog: np.ndarray,
        zorog_smooth: Optional[np.ndarray] = None,
    ) -> None:
        self.zorog = zeros(grid.indices, (I, J))
        self.zorog["definition"] = zorog
        update_periodic_layers(grid.indices, self.zorog)
        # todo(ckuehnlein, n-krieger): Re-think if zorog_smooth should be kept, we currently tend to remove it entirely as not needed after SLEVE is specified
        if zorog_smooth is not None:
            self.zorog_smooth = zeros(grid.indices, (I, J))
            self.zorog_smooth["definition"] = zorog_smooth
            update_periodic_layers(grid.indices, self.zorog_smooth)

        self.xcr = zeros(grid.indices, ("non_periodic", (I, J, K)))
        self.ycr = zeros(grid.indices, ("non_periodic", (I, J, K)))
        self.zcr = zeros(grid.indices, ("non_periodic", (I, J, K)))
        self.xcr["definition"] = horizontal_coordinates.xcr[:, :, np.newaxis]
        self.ycr["definition"] = horizontal_coordinates.ycr[:, :, np.newaxis]
        self.zcr["definition"] = zcr
        self.correct_periodic_layer_values(grid, self.xcr, self.ycr, self.zcr)

        self.delta_zcr = zeros(grid.indices, (I, J, K))
        delta_zcr(grid, self.delta_zcr, self.zcr)

    @staticmethod
    def correct_periodic_layer_values(
        grid: Grid,
        xcr: Field,
        ycr: Field,
        zcr: Field,
    ) -> None:
        update_periodic_layers(grid.indices, xcr.with_altered_index_space((I, J, K)))
        update_periodic_layers(grid.indices, ycr.with_altered_index_space((I, J, K)))
        update_periodic_layers(grid.indices, zcr.with_altered_index_space((I, J, K)))

        for image_id, image_index_space in grid.indices.periodic_images[I, J, K].items():
            periodic_subset_mem = translate_to_memory_index_convention(
                grid.indices.index_spaces[I, J, K],
                image_index_space.subset["periodic_layers"],
            )
            periodic_subset_mem_slice = tuple(
                slice(range_.start, range_.stop) for range_ in periodic_subset_mem.args
            )
            xcr.data[periodic_subset_mem_slice] += image_id.dir[0] * (grid.xmax - grid.xmin)
            ycr.data[periodic_subset_mem_slice] += image_id.dir[1] * (grid.ymax - grid.ymin)
            zcr.data[periodic_subset_mem_slice] += image_id.dir[2] * (grid.zmax - grid.zmin)
