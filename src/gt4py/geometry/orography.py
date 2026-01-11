# -*- coding: utf-8 -*-
from __future__ import annotations
import netCDF4 as nc
import numpy as np
from typing import TYPE_CHECKING

from fvms.build_config import dtype

if TYPE_CHECKING:
    from typing import Optional

    from fvms.geometry.coordinates import Grid, HorizontalCoordinates


def from_file(
    grid: Grid,
    horizontal_coordinates: HorizontalCoordinates,
    *,
    filename: str,
    varname: str = "orog",
) -> np.ndarray:
    with nc.Dataset(filename, mode="r") as ds:
        zorog = ds.variables[varname][...]

    if grid.shape[:2] != zorog.shape:
        raise RuntimeError("Dimension mismatch for grid and orography.")
    return zorog.astype(dtype)


def flat(
    grid: Grid,
    horizontal_coordinates: HorizontalCoordinates,
    *,
    xcenter: Optional[float] = None,
    ycenter: Optional[float] = None,
    amplitude: float,
) -> np.ndarray:
    x = horizontal_coordinates.xcr
    y = horizontal_coordinates.ycr
    xcenter = xcenter if xcenter is not None else 0.5 * (grid.xmax + grid.xmin)
    ycenter = ycenter if ycenter is not None else 0.5 * (grid.ymax + grid.ymin)
    rad = np.sqrt((x - xcenter) ** 2 + (y - ycenter) ** 2)
    zorog = amplitude * rad
    return zorog


# TODO(): replace parameters "zonal" and "meridional" with "direction: Optional[str]" which could be either "zonal" or "meridional"
# This affects all functions with the arguments "zonal: bool" and "meridional: bool"
# See #13 for more details
def cosine_compact(
    grid: Grid,
    horizontal_coordinates: HorizontalCoordinates,
    *,
    xcenter: Optional[float] = None,
    ycenter: Optional[float] = None,
    amplitude: float,
    halfwidth: float,
    zonal: bool = False,
    meridional: bool = False,
) -> np.ndarray:
    x = horizontal_coordinates.xcr
    y = horizontal_coordinates.ycr
    xcenter = xcenter if xcenter is not None else 0.5 * (grid.xmax + grid.xmin)
    ycenter = ycenter if ycenter is not None else 0.5 * (grid.ymax + grid.ymin)
    if zonal:
        rad = np.sqrt((x - xcenter) ** 2)
    elif meridional:
        rad = np.sqrt((y - ycenter) ** 2)
    else:
        rad = np.sqrt((x - xcenter) ** 2 + (y - ycenter) ** 2)
    zorog = 0.5 * amplitude * (1.0 + np.cos(np.pi * rad / halfwidth))
    zorog = np.where(rad < halfwidth, zorog, 0.0)
    return zorog


def gaussian_cosine(
    grid: Grid,
    horizontal_coordinates: HorizontalCoordinates,
    *,
    xcenter: Optional[float] = None,
    ycenter: Optional[float] = None,
    amplitude: float,
    halfwidth_large: float,
    halfwidth_small: float,
    zonal: bool = False,
    meridional: bool = False,
) -> np.ndarray:
    x = horizontal_coordinates.xcr
    y = horizontal_coordinates.ycr
    xcenter = xcenter if xcenter is not None else 0.5 * (grid.xmax + grid.xmin)
    ycenter = ycenter if ycenter is not None else 0.5 * (grid.ymax + grid.ymin)
    if zonal:
        rad = np.sqrt((x - xcenter) ** 2)
    elif meridional:
        rad = np.sqrt((y - ycenter) ** 2)
    else:
        rad = np.sqrt((x - xcenter) ** 2 + (y - ycenter) ** 2)
    zorog = (
        amplitude
        * np.exp(-((rad / halfwidth_large) ** 2))
        * np.cos(np.pi * rad / halfwidth_small) ** 2
    )
    return zorog


def double_cosine(
    grid: Grid,
    horizontal_coordinates: HorizontalCoordinates,
    *,
    xcenter: Optional[float] = None,
    ycenter: Optional[float] = None,
    amplitude: float,
    halfwidth_large: float,
    halfwidth_small: float,
    zonal: bool = False,
    meridional: bool = False,
) -> np.ndarray:
    x = horizontal_coordinates.xcr
    y = horizontal_coordinates.ycr
    xcenter = xcenter if xcenter is not None else 0.5 * (grid.xmax + grid.xmin)
    ycenter = ycenter if ycenter is not None else 0.5 * (grid.ymax + grid.ymin)
    if zonal:
        rad = np.sqrt((x - xcenter) ** 2)
    elif meridional:
        rad = np.sqrt((y - ycenter) ** 2)
    else:
        rad = np.sqrt((x - xcenter) ** 2 + (y - ycenter) ** 2)
    h_star = np.cos(np.pi * rad / (2 * halfwidth_large)) ** 2
    h_star = np.where(rad < halfwidth_large, h_star, 0.0)
    zorog = amplitude * np.cos(np.pi * rad / halfwidth_small) ** 2 * h_star
    return zorog


def gaussian(
    grid: Grid,
    horizontal_coordinates: HorizontalCoordinates,
    *,
    xcenter: Optional[float] = None,
    ycenter: Optional[float] = None,
    amplitude: float,
    halfwidth: float,
    zonal: bool = False,
    meridional: bool = False,
) -> np.ndarray:
    x = horizontal_coordinates.xcr
    y = horizontal_coordinates.ycr
    xcenter = xcenter if xcenter is not None else 0.5 * (grid.xmax + grid.xmin)
    ycenter = ycenter if ycenter is not None else 0.5 * (grid.ymax + grid.ymin)
    if zonal:
        rad = np.sqrt((x - xcenter) ** 2)
    elif meridional:
        rad = np.sqrt((y - ycenter) ** 2)
    else:
        rad = np.sqrt((x - xcenter) ** 2 + (y - ycenter) ** 2)
    zorog = amplitude * np.exp(-((rad / halfwidth) ** 2))
    return zorog


def agnesi(
    grid: Grid,
    horizontal_coordinates: HorizontalCoordinates,
    *,
    xcenter: Optional[float] = None,
    ycenter: Optional[float] = None,
    amplitude: float,
    halfwidth: float,
    zonal: bool = False,
    meridional: bool = False,
) -> np.ndarray:
    x = horizontal_coordinates.xcr
    y = horizontal_coordinates.ycr
    xcenter = xcenter if xcenter is not None else 0.5 * (grid.xmax + grid.xmin)
    ycenter = ycenter if ycenter is not None else 0.5 * (grid.ymax + grid.ymin)
    if zonal:
        rad = np.sqrt((x - xcenter) ** 2)
    elif meridional:
        rad = np.sqrt((y - ycenter) ** 2)
    else:
        rad = np.sqrt((x - xcenter) ** 2 + (y - ycenter) ** 2)
    zorog = amplitude / (1.0 + (rad / halfwidth) ** 2)
    return zorog
