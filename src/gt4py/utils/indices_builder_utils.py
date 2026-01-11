# -*- coding: utf-8 -*-
from __future__ import annotations
import copy
from typing import TYPE_CHECKING

from sl_gt4py.utils.boundary import BoundaryDomainKind
from sl_gt4py.utils.dims import DimSymbol
from sl_gt4py.utils.index_space import IndexSpace, ProductSet, UnitRange, union
from sl_gt4py.utils.indices import PeriodicImageId, RemovedDimension

if TYPE_CHECKING:
    from typing import Union, Literal
    from collections.abc import Callable, Hashable, Sequence

    from sl_gt4py.utils.typingx import Pair, Triple


def get_staggered_field_shape(
    shape: tuple[int, ...],
    boundary_domain_kinds: tuple[BoundaryDomainKind, ...],
    staggered_dim: int,
) -> tuple[int, ...]:
    return tuple(
        size
        # one edge more than vertices in the staggered dimension
        + (1 if dim == staggered_dim else 0)
        # one edge less than vertices for periodic boundary conditions
        # as the last vertex is already a periodic one
        + (-1 if bc == BoundaryDomainKind.PERIODIC else 0)
        for dim, (size, bc) in enumerate(zip(shape, boundary_domain_kinds))
    )


def create_index_space(
    shape: tuple[int, ...],
    boundary_domain_kinds: tuple[BoundaryDomainKind, ...],
    boundary_domain_widths: tuple[Pair[int], ...],
    periodic_layers: tuple[Union[int, Pair[int]], ...],
    has_boundary: tuple[bool, ...],
    inclusive_periodic_boundary: bool = True,
    padding: tuple[Union[int, Pair[int]], ...] = None,
    *,
    subdomain_label_from_dir: Callable[[tuple[int, ...]], str] = lambda dir_: str(dir_),
) -> IndexSpace:
    def slices_from_dir(dir_: tuple[int, ...]) -> tuple[slice, ...]:
        def slices_from_dir_l(dir_l, size):
            # todo(stubbiali): the width of the slices should be dictated by boundary_domain_widths
            dir_l_to_slice = (slice(0, 1), slice(1, size - 1), slice(size - 1, size))
            if size == 1:
                assert dir_l == 0
                return slice(0, None)
            return dir_l_to_slice[dir_l + 1]

        return tuple(slices_from_dir_l(dir_l, size) for dir_l, size in zip(dir_, shape))

    def is_prescribed_boundary(dir_: Sequence[int, ...]) -> bool:
        return any(
            has_boundary[dim] and dir_l in [-1, 1] and bc == BoundaryDomainKind.PRESCRIBED
            for dim, (dir_l, bc) in enumerate(zip(dir_, boundary_domain_kinds))
        )

    def is_periodic_boundary(dir_: Sequence[int, ...]) -> bool:
        return any(
            inclusive_periodic_boundary
            and has_boundary[dim]
            and dir_l == 1
            and bc == BoundaryDomainKind.PERIODIC
            for dim, (dir_l, bc) in enumerate(zip(dir_, boundary_domain_kinds))
        )

    def is_interior(dir_):
        return not is_prescribed_boundary(dir_) and not is_periodic_boundary(dir_)

    def is_part_of(dir_, candidate_dir):
        return all(
            candidate_dir_l == dir_l or candidate_dir_l == 0
            for candidate_dir_l, dir_l in zip(candidate_dir, dir_)
        )

    # used to skip if any non-center direction has a size of 1,
    # e.g. top and bottom for pseudo horizontal
    def is_off_center_in_collapsed_dim(dir_: Sequence[int, ...]):
        # sanity check to ensure there was no boundary specified in a
        # collapsed dimension
        # assert not any(
        #     dir_l != 0 and size == 1 and hb
        #     for dir_l, size, hb in zip(dir_, shape, has_boundary)
        # )
        return any(dir_l != 0 and size == 1 for dir_l, size in zip(dir_, shape))

    ndims = len(shape)
    if not padding:
        padding = tuple([0] * ndims)

    index_space_subsets = {"definition": ProductSet(*(UnitRange(0, l) for l in shape))}

    empty_set = ProductSet(*([UnitRange(0, 0)] * ndims))
    interior = empty_set
    periodic_boundary = empty_set

    for dir_ in ProductSet(*([UnitRange(-1, 2)] * ndims)):
        part = index_space_subsets["definition"][slices_from_dir(dir_)]

        if is_interior(dir_):
            interior = union(interior, part, simplify=False)

        if is_periodic_boundary(dir_):
            periodic_boundary = union(periodic_boundary, part, simplify=False)

    index_space_subsets["interior"] = interior
    index_space_subsets["periodic_boundary"] = periodic_boundary
    index_space_subsets["physical"] = index_space_subsets["definition"]
    index_space_subsets["physical"] = index_space_subsets["definition"].without(
        index_space_subsets["periodic_boundary"]
    )
    index_space_subsets["periodic_layers"] = (
        index_space_subsets["physical"]
        .extend(
            *(
                el if bc == BoundaryDomainKind.PERIODIC else 0
                for bc, el in zip(boundary_domain_kinds, periodic_layers)
            )
        )
        .without(index_space_subsets["physical"])
    )

    prescribed_boundary = empty_set
    prescribed_boundary_parts = {
        dir_: empty_set
        for dir_ in ProductSet(*([UnitRange(-1, 2)] * ndims))
        if is_prescribed_boundary(dir_)  # and not is_corner(dir_)
        and (all(has_boundary[dim] or dir_l == 0 for dim, dir_l in enumerate(dir_)))
        and not any(dir_l != 0 and size == 1 for dir_l, size in zip(dir_, shape))
    }
    for dir_ in ProductSet(*([UnitRange(-1, 2)] * ndims)):
        if is_off_center_in_collapsed_dim(dir_):
            continue

        part = index_space_subsets["physical"][slices_from_dir(dir_)]

        if is_prescribed_boundary(dir_):
            prescribed_boundary = union(prescribed_boundary, part, simplify=False)
            for candidate_dir in prescribed_boundary_parts.keys():
                if is_part_of(dir_, candidate_dir):
                    prescribed_boundary_parts[candidate_dir] = union(
                        prescribed_boundary_parts[candidate_dir], part
                    )

    for dir_, part in prescribed_boundary_parts.items():
        index_space_subsets[("prescribed_boundary", subdomain_label_from_dir(dir_))] = part
    index_space_subsets["prescribed_boundary"] = prescribed_boundary

    # used for stencils with an extent larger than 1
    # todo(tehrengruber): using translate is not a great way to derive those. revisit.
    for dim in range(ndims):
        for i_dir_l, dir_l in enumerate([-1, 1]):
            dir_ = tuple(dir_l if l == dim else 0 for l in range(0, ndims))
            for offset in range(0, boundary_domain_widths[dim][i_dir_l]):
                offset_translation = tuple(-offset * dir_l for dir_l in dir_)
                label = ("prescribed_boundary", subdomain_label_from_dir(dir_))
                if label in index_space_subsets:
                    label_with_offset = (
                        "prescribed_boundary",
                        subdomain_label_from_dir(dir_),
                        offset,
                    )
                    index_space_subsets[label_with_offset] = index_space_subsets[label].translate(
                        *offset_translation
                    )

    # used for point-wise stencils, might change in the future
    covering = index_space_subsets["covering"] = union(*index_space_subsets.values())

    # used for stencils expecting a west+east, north+south, bottom+top neighbor,
    # e.g. centered diff
    # todo(tehrengruber, stubbiali): more meaningful name
    if all(size > 1 for size in shape):  # not meaningful for pseudo horizontal/vertical
        for dim in range(ndims):
            for extent_l in range(1, max(boundary_domain_widths[dim]) + 1):
                if 2 * extent_l > shape[dim]:
                    continue
                extent = tuple(extent_l if l == dim else 0 for l in range(0, ndims))
                margin = tuple(-el for el in extent)
                index_space_subsets[("interior", extent)] = covering.extend(*margin)

    assert (
        union(interior, prescribed_boundary, periodic_boundary) == index_space_subsets["definition"]
    )
    assert (
        union(index_space_subsets["physical"], periodic_boundary)
        == index_space_subsets["definition"]
    )

    if any(p != 0 for p in padding):
        index_space_subsets["_padding"] = union(*index_space_subsets.values()).extend(*padding)

    return IndexSpace({k: v.simplify() for k, v in index_space_subsets.items()})


def add_periodic_images(
    index_spaces: dict[Hashable, IndexSpace],
    index_conventions: dict[Hashable, dict[Hashable, Triple[int]]],
) -> None:
    ids = tuple(index_spaces.keys())
    for id_ in ids:
        index_space = index_spaces[id_]
        periodic_images = {}

        if "periodic_layers" not in index_space.subset:
            continue

        for dir in ProductSet(*([UnitRange(-1, 2)] * index_space.dim)):
            if all(dir_el == 0 for dir_el in dir):
                continue

            offsets = tuple(
                dir_i * l for dir_i, l in zip(dir, index_space.subset["physical"].shape)
            )

            image_id = PeriodicImageId(id_, dir)
            image_indices = index_space.bounds.translate(*offsets)
            image_index_space_subsets = {"definition": image_indices}
            image_index_space_subsets["periodic_layers"] = (
                index_space.subset["physical"]
                .translate(*offsets)
                .intersect(index_space.subset["periodic_layers"])
            )

            if image_index_space_subsets["periodic_layers"].empty:
                continue

            image_index_space = index_spaces[image_id] = IndexSpace(image_index_space_subsets)
            index_conventions[image_id] = {
                image_id: (0,) * index_space.dim,
                id_: (0,) * index_space.dim,
            }
            periodic_images[image_id] = image_index_space

        # ensure periodic images cover entire boundary
        assert (len(periodic_images) == 0 and index_space.subset["periodic_layers"].empty) or union(
            *(
                image_index_space.subset["periodic_layers"]
                for image_index_space in periodic_images.values()
            )
        ) == index_space.subset["periodic_layers"]


def subdomain_label_from_dir_ij(dir_: tuple[int, int, ...]) -> str:
    labels = [["south", "north"], ["west", "east"], ["bottom", "top"]]
    # meridional direction comes first
    permuted_dir = (dir_[1], dir_[0], *dir_[2:])
    return "-".join(
        labels[l][0 if dir_l == -1 else 1] for l, dir_l in enumerate(permuted_dir) if dir_l != 0
    )


def subdomain_label_from_dir_k(dir_: tuple[int]) -> str:
    return subdomain_label_from_dir_ij((0, 0, *dir_))


def deduce_lower_dim_index_space_construction_args(
    construction_args: dict[Triple[DimSymbol], dict]
) -> dict[tuple[DimSymbol, ...], dict]:
    """Deduce construction arguments of horizontal and vertical index spaces."""
    new_construction_args = {}
    for id_, args in construction_args.items():
        assert args["subdomain_label_from_dir"] == subdomain_label_from_dir_ij

        # horizontal
        if id_[-1].offset == 0:
            new_construction_args[id_[0:2]] = {
                **{k: v[0:2] for k, v in args.items() if k != "subdomain_label_from_dir"},
                "subdomain_label_from_dir": subdomain_label_from_dir_ij,
            }

        # vertical
        if id_[0].offset == 0 and id_[1].offset == 0:
            new_construction_args[id_[-1:]] = {
                **{k: v[-1:] for k, v in args.items() if k != "subdomain_label_from_dir"},
                "subdomain_label_from_dir": subdomain_label_from_dir_k,
            }
    return new_construction_args


def deduce_lower_dim_index_conventions(
    conventions: dict[Triple[DimSymbol], Triple[int]]
) -> dict[tuple[DimSymbol, ...], Triple[Union[int, Literal[RemovedDimension]]]]:
    """Deduce index conventions of horizontal and vertical index spaces."""
    new_conventions = copy.deepcopy(conventions)
    for convention, new_convention in zip(conventions.values(), new_conventions.values()):
        for id_, conv in convention.items():
            # skip everything that is not a triple of DimSymbols, e.g. non-periodic
            if not all(isinstance(el, DimSymbol) for el in id_):
                continue

            # horizontal
            if id_[-1].offset == 0:
                new_convention[id_[0:2]] = (*conv[0:2], RemovedDimension)

            # vertical
            if id_[0].offset == 0 and id_[1].offset == 0:
                new_convention[id_[-1:]] = (RemovedDimension, RemovedDimension, *conv[-1:])
    return new_conventions


def remove_periodic_layers_from_index_space(index_space: IndexSpace) -> IndexSpace:
    """Return a new index space with all periodic layers removed."""
    return IndexSpace({k: v for k, v in index_space.subset.items() if k != "periodic_layers"})


def deduce_non_periodic_index_conventions(
    conventions: dict[Triple[DimSymbol], Triple[int]],
    non_periodic_index_space_ids: list[Triple[int]],
) -> dict[tuple[DimSymbol, ...], Triple[Union[int, Literal[RemovedDimension]]]]:
    new_conventions = copy.deepcopy(conventions)
    for convention, new_convention in zip(conventions.values(), new_conventions.values()):
        for id_ in non_periodic_index_space_ids:
            # inherit convention from non-periodic counter-part
            new_convention[("non_periodic", id_)] = convention[id_]
    return new_conventions
