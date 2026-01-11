# -*- coding: utf-8 -*-
from __future__ import annotations
from copy import copy
from typing import TYPE_CHECKING

from gt4py.cartesian import gtscript

from sl_gt4py.build import (
    backend,
    backend_opts,
    dtype,
    update_periodic_layers_using_copy_stencil,
)
from sl_gt4py.config import update
from sl_gt4py.utils.stencil import stencil, stencil_args

if TYPE_CHECKING:
    from sl_gt4py.utils.indices import Indices
    from sl_gt4py.utils.storage import Field


@stencil(backend=backend, **backend_opts)
def copy_stencil(in_field: gtscript.Field[dtype], out_field: gtscript.Field[dtype]):
    with computation(PARALLEL), interval(...):
        out_field[0, 0, 0] = in_field[0, 0, 0]


def update_periodic_layers(indices: Indices, field: Field) -> None:
    for image_id, index_space in indices.periodic_images[field.index_space].items():
        # define a new field backed by the same memory, but posed on the index space of the periodic image
        periodic_image_field = copy(field)  # create a shallow copy
        assert image_id.id_ == periodic_image_field.index_space
        periodic_image_field.index_space = image_id

        if not update_periodic_layers_using_copy_stencil:
            # update all values in the periodic image
            for offsets, lengths in stencil_args(
                indices,
                image_id,
                {"periodic_image_field": periodic_image_field, "field": field},
                subdomain="periodic_layers",
            ):
                slice_field = tuple(
                    slice(offset, offset + lengths[i]) for i, offset in enumerate(offsets["field"])
                )
                slice_periodic_image_field = tuple(
                    slice(offset, offset + lengths[i])
                    for i, offset in enumerate(offsets["periodic_image_field"])
                )
                field.data[slice_field] = periodic_image_field.data[slice_periodic_image_field]
        else:
            copy_stencil(
                indices,
                image_id,
                periodic_image_field,
                field,
                subdomain="periodic_layers",
                update_pbcs=False,
            )
