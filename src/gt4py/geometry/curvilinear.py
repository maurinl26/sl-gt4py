# -*- coding: utf-8 -*-
from __future__ import annotations
from itertools import repeat
from typing import TYPE_CHECKING

from gt4py.cartesian import gtscript

from fvms.build_config import backend, backend_opts, dtype
from fvms.geometry.spherical import Spherical
from fvms.operators.basic import div_acst, mult_acst
from fvms.operators.nabla import Nabla
from fvms.utils.dims import I, J, K
from fvms.utils.stencil import stencil
from fvms.utils.storage import managed_temporary, zeros

if TYPE_CHECKING:
    from fvms.geometry.coordinates import Coordinates, Grid
    from fvms.model import config
    from fvms.utils.boundary import BoundaryDomainKind
    from fvms.utils.storage import Field
    from fvms.utils.typingx import Triple


@stencil(backend=backend, **backend_opts)
def _kronecker(
    dxdxc: gtscript.Field[dtype],
    dxdyc: gtscript.Field[dtype],
    dydxc: gtscript.Field[dtype],
    dydyc: gtscript.Field[dtype],
    dzdxc: gtscript.Field[dtype],
    dzdyc: gtscript.Field[dtype],
    dzdzc: gtscript.Field[dtype],
    g11: gtscript.Field[dtype],
    g12: gtscript.Field[dtype],
    g13: gtscript.Field[dtype],
    g21: gtscript.Field[dtype],
    g22: gtscript.Field[dtype],
    g23: gtscript.Field[dtype],
    g33: gtscript.Field[dtype],
    gac: gtscript.Field[dtype],
    gmm: gtscript.Field[dtype],
    zcr: gtscript.Field[dtype],
    coslat: gtscript.Field[gtscript.IJ, dtype],
    radius: dtype,
    sphere: bool,
    deep: bool,
):
    with computation(PARALLEL), interval(...):
        gxy = dxdxc[0, 0, 0] * dydyc[0, 0, 0] - dxdyc[0, 0, 0] * dydxc[0, 0, 0]
        gxyi = 1.0 / gxy
        g11[0, 0, 0] = dydyc[0, 0, 0] * gxyi
        g12[0, 0, 0] = -dxdyc[0, 0, 0] * gxyi
        g21[0, 0, 0] = -dydxc[0, 0, 0] * gxyi
        g22[0, 0, 0] = dxdxc[0, 0, 0] * gxyi
        gac[0, 0, 0] = gxy * dzdzc[0, 0, 0]
        gaci = 1.0 / gac[0, 0, 0]
        g13[0, 0, 0] = (dydxc[0, 0, 0] * dzdyc[0, 0, 0] - dydyc[0, 0, 0] * dzdxc[0, 0, 0]) * gaci
        g23[0, 0, 0] = (dxdyc[0, 0, 0] * dzdxc[0, 0, 0] - dxdxc[0, 0, 0] * dzdyc[0, 0, 0]) * gaci
        g33[0, 0, 0] = 1.0 / dzdzc[0, 0, 0]
        gmm[0, 0, 0] = 1.0 + (sphere and deep) * zcr[0, 0, 0] / radius
        go11 = 1.0 / (gmm[0, 0, 0] * coslat[0, 0])
        go22 = 1.0 / gmm[0, 0, 0]
        go33 = 1.0
        g11[0, 0, 0] = g11[0, 0, 0] * go11
        g12[0, 0, 0] = g12[0, 0, 0] * go11
        g21[0, 0, 0] = g21[0, 0, 0] * go22
        g22[0, 0, 0] = g22[0, 0, 0] * go22
        g13[0, 0, 0] = g13[0, 0, 0] * go11
        g23[0, 0, 0] = g23[0, 0, 0] * go22
        g33[0, 0, 0] = g33[0, 0, 0] * go33
        gac[0, 0, 0] = gac[0, 0, 0] * gmm[0, 0, 0] ** 2.0 * coslat[0, 0]


@stencil(backend=backend, **backend_opts)
def _geometric_coefficients(
    g11: gtscript.Field[dtype],
    g12: gtscript.Field[dtype],
    g13: gtscript.Field[dtype],
    g21: gtscript.Field[dtype],
    g22: gtscript.Field[dtype],
    g23: gtscript.Field[dtype],
    g33: gtscript.Field[dtype],
    gc11: gtscript.Field[dtype],
    gc12: gtscript.Field[dtype],
    gc13: gtscript.Field[dtype],
    gc21: gtscript.Field[dtype],
    gc22: gtscript.Field[dtype],
    gc23: gtscript.Field[dtype],
    gc31: gtscript.Field[dtype],
    gc32: gtscript.Field[dtype],
    gc33: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):
        gc11[0, 0, 0] = g11[0, 0, 0] * g11[0, 0, 0] + g21[0, 0, 0] * g21[0, 0, 0]
        gc12[0, 0, 0] = g11[0, 0, 0] * g12[0, 0, 0] + g21[0, 0, 0] * g22[0, 0, 0]
        gc13[0, 0, 0] = g11[0, 0, 0] * g13[0, 0, 0] + g21[0, 0, 0] * g23[0, 0, 0]
        gc21[0, 0, 0] = g12[0, 0, 0] * g11[0, 0, 0] + g22[0, 0, 0] * g21[0, 0, 0]
        gc22[0, 0, 0] = g12[0, 0, 0] * g12[0, 0, 0] + g22[0, 0, 0] * g22[0, 0, 0]
        gc23[0, 0, 0] = g12[0, 0, 0] * g13[0, 0, 0] + g22[0, 0, 0] * g23[0, 0, 0]
        gc31[0, 0, 0] = g13[0, 0, 0] * g11[0, 0, 0] + g23[0, 0, 0] * g21[0, 0, 0]
        gc32[0, 0, 0] = g13[0, 0, 0] * g12[0, 0, 0] + g23[0, 0, 0] * g22[0, 0, 0]
        gc33[0, 0, 0] = (
            g13[0, 0, 0] * g13[0, 0, 0] + g23[0, 0, 0] * g23[0, 0, 0] + g33[0, 0, 0] * g33[0, 0, 0]
        )


@stencil(backend=backend, **backend_opts)
def _transform_velocity(
    uvelx: gtscript.Field[dtype],
    uvely: gtscript.Field[dtype],
    uvelz: gtscript.Field[dtype],
    g11: gtscript.Field[dtype],
    g12: gtscript.Field[dtype],
    g13: gtscript.Field[dtype],
    g21: gtscript.Field[dtype],
    g22: gtscript.Field[dtype],
    g23: gtscript.Field[dtype],
    g33: gtscript.Field[dtype],
    velx: gtscript.Field[dtype],
    vely: gtscript.Field[dtype],
    velz: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):
        velx[0, 0, 0] = g11[0, 0, 0] * uvelx[0, 0, 0] + g21[0, 0, 0] * uvely[0, 0, 0]
        vely[0, 0, 0] = g12[0, 0, 0] * uvelx[0, 0, 0] + g22[0, 0, 0] * uvely[0, 0, 0]
        velz[0, 0, 0] = (
            g13[0, 0, 0] * uvelx[0, 0, 0]
            + g23[0, 0, 0] * uvely[0, 0, 0]
            + g33[0, 0, 0] * uvelz[0, 0, 0]
        )


@stencil(backend=backend, **backend_opts)
def _itransform_velocity(
    velx: gtscript.Field[dtype],
    vely: gtscript.Field[dtype],
    velz: gtscript.Field[dtype],
    g11: gtscript.Field[dtype],
    g12: gtscript.Field[dtype],
    g13: gtscript.Field[dtype],
    g21: gtscript.Field[dtype],
    g22: gtscript.Field[dtype],
    g23: gtscript.Field[dtype],
    g33: gtscript.Field[dtype],
    uvelx: gtscript.Field[dtype],
    uvely: gtscript.Field[dtype],
    uvelz: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):
        gxyi = 1.0 / (g11[0, 0, 0] * g22[0, 0, 0] - g12[0, 0, 0] * g21[0, 0, 0])
        uvelx[0, 0, 0] = gxyi * (g22[0, 0, 0] * velx[0, 0, 0] - g21[0, 0, 0] * vely[0, 0, 0])
        uvely[0, 0, 0] = gxyi * (g11[0, 0, 0] * vely[0, 0, 0] - g12[0, 0, 0] * velx[0, 0, 0])
        uvelz[0, 0, 0] = (
            velz[0, 0, 0] - g13[0, 0, 0] * velx[0, 0, 0] - g23[0, 0, 0] * vely[0, 0, 0]
        ) / g33[0, 0, 0]


@stencil(backend=backend, **backend_opts)
def _transform_nabla(
    nabla_x: gtscript.Field[dtype],
    nabla_y: gtscript.Field[dtype],
    nabla_z: gtscript.Field[dtype],
    g11: gtscript.Field[dtype],
    g12: gtscript.Field[dtype],
    g13: gtscript.Field[dtype],
    g21: gtscript.Field[dtype],
    g22: gtscript.Field[dtype],
    g23: gtscript.Field[dtype],
    g33: gtscript.Field[dtype],
    grad_x: gtscript.Field[dtype],
    grad_y: gtscript.Field[dtype],
    grad_z: gtscript.Field[dtype],
):
    with computation(PARALLEL), interval(...):
        grad_x[0, 0, 0] = (
            nabla_x[0, 0, 0] * g11[0, 0, 0]
            + nabla_y[0, 0, 0] * g12[0, 0, 0]
            + nabla_z[0, 0, 0] * g13[0, 0, 0]
        )
        grad_y[0, 0, 0] = (
            nabla_x[0, 0, 0] * g21[0, 0, 0]
            + nabla_y[0, 0, 0] * g22[0, 0, 0]
            + nabla_z[0, 0, 0] * g23[0, 0, 0]
        )
        grad_z[0, 0, 0] = nabla_z[0, 0, 0] * g33[0, 0, 0]


class Metric:
    def __init__(
        self,
        boundary_domain_kinds: Triple[BoundaryDomainKind],
        grid: Grid,
        coordinates: Coordinates,
        spherical: Spherical,
        nabla_config: config.Nabla,
        radius_sphere: dtype,
        sphere: bool,
        deep: bool,
    ) -> None:
        self.indices = grid.indices
        self.radius_sphere = radius_sphere
        self.sphere = sphere
        self.deep = deep
        self.g11 = zeros(self.indices, (I, J, K))
        self.g12 = zeros(self.indices, (I, J, K))
        self.g13 = zeros(self.indices, (I, J, K))
        self.g21 = zeros(self.indices, (I, J, K))
        self.g22 = zeros(self.indices, (I, J, K))
        self.g23 = zeros(self.indices, (I, J, K))
        self.g33 = zeros(self.indices, (I, J, K))
        self.gac = zeros(self.indices, (I, J, K))
        self.gmm = zeros(self.indices, (I, J, K))
        self.gc11 = zeros(self.indices, (I, J, K))
        self.gc12 = zeros(self.indices, (I, J, K))
        self.gc13 = zeros(self.indices, (I, J, K))
        self.gc21 = zeros(self.indices, (I, J, K))
        self.gc22 = zeros(self.indices, (I, J, K))
        self.gc23 = zeros(self.indices, (I, J, K))
        self.gc31 = zeros(self.indices, (I, J, K))
        self.gc32 = zeros(self.indices, (I, J, K))
        self.gc33 = zeros(self.indices, (I, J, K))

        self.nabla = Nabla(
            boundary_domain_kinds,
            grid,
            method_x=nabla_config.method_x,
            method_y=nabla_config.method_y,
            method_z=nabla_config.method_z,
        )

        with managed_temporary(self.indices, *repeat((I, J, K), 7)) as (
            nabla_xcr_x,
            nabla_xcr_y,
            nabla_ycr_x,
            nabla_ycr_y,
            nabla_zcr_x,
            nabla_zcr_y,
            nabla_zcr_z,
        ):

            if self.sphere:
                mult_acst(
                    self.indices,
                    (I, J, K),
                    coordinates.xcr,
                    cst=radius_sphere,
                    subdomain="covering",
                )
                mult_acst(
                    self.indices,
                    (I, J, K),
                    coordinates.ycr,
                    cst=radius_sphere,
                    subdomain="covering",
                )
            self.nabla(coordinates.xcr, pdx=nabla_xcr_x, pdy=nabla_xcr_y)
            self.nabla(coordinates.ycr, pdx=nabla_ycr_x, pdy=nabla_ycr_y)
            self.nabla(coordinates.zcr, pdx=nabla_zcr_x, pdy=nabla_zcr_y, pdz=nabla_zcr_z)
            if self.sphere:
                div_acst(
                    self.indices,
                    (I, J, K),
                    coordinates.xcr,
                    cst=radius_sphere,
                    subdomain="covering",
                )
                div_acst(
                    self.indices,
                    (I, J, K),
                    coordinates.ycr,
                    cst=radius_sphere,
                    subdomain="covering",
                )

            _kronecker(
                self.indices,
                (I, J, K),
                nabla_xcr_x,
                nabla_xcr_y,
                nabla_ycr_x,
                nabla_ycr_y,
                nabla_zcr_x,
                nabla_zcr_y,
                nabla_zcr_z,
                self.g11,
                self.g12,
                self.g13,
                self.g21,
                self.g22,
                self.g23,
                self.g33,
                self.gac,
                self.gmm,
                coordinates.zcr,
                coslat=spherical.coslat,
                radius=radius_sphere,
                sphere=sphere,
                deep=deep,
                subdomain="covering",
            )

            _geometric_coefficients(
                self.indices,
                (I, J, K),
                self.g11,
                self.g12,
                self.g13,
                self.g21,
                self.g22,
                self.g23,
                self.g33,
                self.gc11,
                self.gc12,
                self.gc13,
                self.gc21,
                self.gc22,
                self.gc23,
                self.gc31,
                self.gc32,
                self.gc33,
                subdomain="covering",
            )

    def transform_velocity(self, uvel: Field, vel: Field, inverse: bool = False) -> None:
        uvelx, uvely, uvelz = uvel
        velx, vely, velz = vel

        if not inverse:
            _transform_velocity(
                self.indices,
                (I, J, K),
                uvelx,
                uvely,
                uvelz,
                self.g11,
                self.g12,
                self.g13,
                self.g21,
                self.g22,
                self.g23,
                self.g33,
                velx,
                vely,
                velz,
                subdomain="covering",
            )
        else:
            _itransform_velocity(
                self.indices,
                (I, J, K),
                velx,
                vely,
                velz,
                self.g11,
                self.g12,
                self.g13,
                self.g21,
                self.g22,
                self.g23,
                self.g33,
                uvelx,
                uvely,
                uvelz,
                subdomain="covering",
            )

    def transform_nabla(
        self,
        nabla_x: Field,
        nabla_y: Field,
        nabla_z: Field,
        grad_x: Field,
        grad_y: Field,
        grad_z: Field,
    ) -> None:
        _transform_nabla(
            self.indices,
            (I, J, K),
            nabla_x,
            nabla_y,
            nabla_z,
            self.g11,
            self.g12,
            self.g13,
            self.g21,
            self.g22,
            self.g23,
            self.g33,
            grad_x,
            grad_y,
            grad_z,
            subdomain="covering",
        )
