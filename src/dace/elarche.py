import dace
from gt4py.cartesian.gtscript import stencil
from typing import Tuple

# stencils
from sl_dace.interpolation.interpolation_2d import interpolate_lin_2d
from sl_dace.stencils.copy import copy
from sl_dace.stencils.dep_search_1d import dep_search_1d
from sl_dace.dims import I, J, K


class Elarche:

    def __init__(self, grid: Tuple[int], halo: int = 0, nitmp: int = 4):
        self.grid = grid
        self.symbol_mapping = {
                "I": grid[0],
                "J": grid[1],
                "K": grid[2],
                "H": halo
            }
        self.nitmp = nitmp

        # stencils
        self.copy = stencil(
            backend="dace:cpu",
            definition=copy,
            name="copy"
        )
        self.dep_search_1d = stencil(
            backend="dace:cpu",
            definition=dep_search_1d,
            name="dep_search_1d"
        )

        # dace
        # interpolate_lin_2d
        self.d_interpolate_lin_2d = (
            dace.program(interpolate_lin_2d)
            .to_sdfg()
            .compile()
        )


    def __call__(self,
                 dx: dace.float32,
                 dy: dace.float32,
                 dth: dace.float32,
                 idx: dace.int32[I, J, K],
                 jdx: dace.int32[I, J, K],
                 vx: dace.float32[I, J, K],
                 vy: dace.float32[I, J, K],
                 vx_tmp: dace.float32[I, J, K],
                 vy_tmp: dace.float32[I, J, K],
                 vx_e: dace.float32[I, J, K],
                 vy_e: dace.float32[I, J, K],
                 # Outputs
                 lx: dace.float32[I, J, K],
                 ly: dace.float32[I, J, K],
                 i_dep: dace.int32[I, J, K],
                 j_dep: dace.int32[I, J, K],
        ):

        # Temporaries
        self.copy(
            vx=vx,
            vy=vy,
            vx_tmp=vx_tmp,
            vy_tmp=vy_tmp,
            domain=self.grid,
            origin=(0, 0, 0)
        )

        # Array declaration
        for l in range(self.nitmp):
            self.dep_search_1d(
                vx_e=vx_e,
                vx_tmp=vx_tmp,
                i_a=idx,
                i_d=i_dep,
                lx=lx,
                dx=dx,
                dth=dth,
                domain=self.grid,
                origin=(0, 0, 0)
            )
            self.dep_search_1d(
                vx_e=vy_e,
                vx_tmp=vy_tmp,
                i_a=jdx,
                i_d=jdx,
                lx=ly,
                dx=dy,
                dth=dth,
                domain=self.grid,
                origin=(0, 0, 0)
            )

            # todo: add LipschitzDiag
            ####### Interpolation for fields ########
            self.d_interpolate_lin_2d(
                psi_dep=vx,
                psi=vx_tmp,
                lx=lx,
                ly=ly,
                i_dep=i_dep,
                j_dep=j_dep,
                **self.symbol_mapping
            )

            self.d_interpolate_lin_2d(
                psi_dep=vy,
                psi=vy_tmp,
                lx=lx,
                ly=ly,
                i_dep=i_dep,
                j_dep=j_dep,
                **self.symbol_mapping
            )

        return lx, ly, i_dep, j_dep


