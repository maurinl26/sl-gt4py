# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from fvms.build_config import dtype

if TYPE_CHECKING:
    from fvms.geometry.coordinates import Grid


def identity(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    return dtype(np.meshgrid(grid.xc, grid.yc, sparse=False, indexing="ij"))


def clustered(grid: Grid, *, focus_x: float, focus_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Cluster the mesh points around the focal point (`focus_x`, `focus_y`) using parabolas."""

    # warning: the current model formulation does not support deformed xy coordinates
    # todo(ckuehnlein, stubbiali): add the necessary steps to make the model work on deformed horizontal meshes

    if not (grid.xmin < focus_x < grid.xmax and grid.ymin < focus_y < grid.ymax):
        raise ValueError("Focal point must lie inside the grid boundaries.")

    x, y = dtype(np.meshgrid(grid.xc, grid.yc, sparse=False, indexing="ij"))

    xcr = np.zeros_like(x)
    ycr = np.zeros_like(y)

    lx1 = focus_x - grid.xmin
    mx1 = x <= focus_x
    xn1 = (x[mx1] - grid.xmin) / lx1
    xcr[mx1] = grid.xmin + lx1 * (xn1 * (2 - xn1))

    lx2 = grid.xmax - focus_x
    mx2 = x > focus_x
    xn2 = (x[mx2] - focus_x) / lx2
    xcr[mx2] = focus_x + lx2 * xn2**2

    ly1 = focus_y - grid.ymin
    my1 = y <= focus_y
    yn1 = (y[my1] - grid.ymin) / ly1
    ycr[my1] = grid.ymin + ly1 * (yn1 * (2 - yn1))

    ly2 = grid.ymax - focus_y
    my2 = y > focus_y
    yn2 = (y[my2] - focus_y) / ly2
    ycr[my2] = focus_y + ly2 * yn2**2

    return xcr, ycr
