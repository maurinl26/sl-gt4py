# -*- coding: utf-8 -*-
from __future__ import annotations
import logging
import netCDF4 as nc
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    from fvms.geometry.coordinates import Grid


log = logging.getLogger(__name__)


def from_file(
    grid: Grid,
    zstretch: np.ndarray,
    zs: np.ndarray,
    bottom: float,
    zsmooth: Optional[np.ndarray],
    *,
    filename: str,
    varname: str = "zcr",
) -> np.ndarray:
    with nc.Dataset(filename, mode="r") as ds:
        zcr = ds.variables[varname][...]

    if grid.shape != zcr.shape:
        raise RuntimeError("Dimension mismatch for coordinates to load from file")
    if not np.allclose(zcr[:, :, -1], grid.zmax):
        raise RuntimeError("Top mismatch for config and coordinates to load from file")
    if not np.allclose(zcr[:, :, 0], zs + bottom):
        raise RuntimeError("Surface mismatch for orography and coordinates to load from file")

    return zcr


def btf_levels(
    grid: Grid,
    zstretch: np.ndarray,
    zs: np.ndarray,
    bottom: float,
    zsmooth: Optional[np.ndarray],
) -> np.ndarray:

    """Basic terrain-following levels."""
    zcr = np.zeros(grid.shape)
    bb = 1.0 - zstretch / zstretch[-1]
    zcr[:, :, :] = (
        zstretch[np.newaxis, np.newaxis, :]
        + bb[np.newaxis, np.newaxis, :] * zs[:, :, np.newaxis]
        + bottom
    )
    return zcr


def htf_levels(
    grid: Grid,
    zstretch: np.ndarray,
    zs: np.ndarray,
    bottom: float,
    zsmooth: Optional[np.ndarray],
    *,
    z_hybrid_transition: float,  # z_hybrid_transition is absolute height
) -> np.ndarray:

    """Hybrid terrain-following levels."""
    zcr = np.zeros(grid.shape)
    z_trans = z_hybrid_transition - bottom
    bb = np.where(zstretch <= z_trans, np.cos(0.5 * np.pi * zstretch / z_trans) ** 6.0, 0.0)
    zcr[:, :, :] = (
        zstretch[np.newaxis, np.newaxis, :]
        + bb[np.newaxis, np.newaxis, :] * zs[:, :, np.newaxis]
        + bottom
    )
    return zcr


def sleve_levels(
    grid: Grid,
    zstretch: np.ndarray,
    zs: np.ndarray,
    bottom: float,
    zsmooth: np.ndarray,
    *,
    z_hybrid_transition: float,  # z_hybrid_transition is absolute height
) -> np.ndarray:

    """Hybrid SLEVE terrain-following levels."""
    if zsmooth is None:
        raise ValueError(
            f"The SLEVE terrain-following level specification requires a smoothed orography."
        )
    zcr = np.zeros(grid.shape)
    z_trans = z_hybrid_transition - bottom
    s1 = 5000.0
    s2 = 2000.0
    en = 1.35
    d1 = np.sinh((z_trans / s1) ** en)
    d2 = np.sinh((z_trans / s2) ** en)
    bb1 = np.where(
        zstretch <= z_trans, np.sinh((z_trans / s1) ** en - (zstretch / s1) ** en) / d1, 0.0
    )
    bb2 = np.where(
        zstretch <= z_trans, np.sinh((z_trans / s2) ** en - (zstretch / s2) ** en) / d2, 0.0
    )
    zcr[:, :, :] = (
        zstretch[np.newaxis, np.newaxis, :]
        + bb1[np.newaxis, np.newaxis, :] * zsmooth[:, :, np.newaxis]
        + bb2[np.newaxis, np.newaxis, :] * (zs[:, :, np.newaxis] - zsmooth[:, :, np.newaxis])
        + bottom
    )
    return zcr


def imrsb_levels(
    grid: Grid,
    zstretch: np.ndarray,
    zs: Optional[np.ndarray],
    bottom: float,
    zsmooth: Optional[np.ndarray],
) -> np.ndarray:

    """Levels for pure immersed boundaries."""
    zcr = np.zeros(grid.shape)
    zcr[:, :, :] = zstretch[np.newaxis, np.newaxis, :] + bottom
    return zcr
