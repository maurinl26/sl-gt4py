# -*- coding: utf-8 -*-
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from sl_gt4py.utils.boundary import BoundaryDomainKind
from sl_gt4py.utils.dims import I, J, K
from sl_gt4py.utils.indices import Indices
from sl_gt4py.utils.indices_builder_utils import (
    add_periodic_images,
    create_index_space,
    deduce_lower_dim_index_conventions,
    deduce_lower_dim_index_space_construction_args,
    deduce_non_periodic_index_conventions,
    get_staggered_field_shape,
    remove_periodic_layers_from_index_space,
    subdomain_label_from_dir_ij,
)

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Union
    from sl_gt4py.utils.index_space import IndexSpace
    from sl_gt4py.utils.typingx import Pair, Triple

from sl_gt4py.utils.index_space import IndexSpace


class IndicesBuilder(ABC):
    """
    Pursue different strategies to create index spaces and the corresponding
    index conventions, and collect them in ``Indices``.
    """

    shape: Triple[int]
    boundary_domain_kinds: Triple[BoundaryDomainKind]
    boundary_domain_widths: Triple[Pair[int]]

    def __init__(
        self,
        shape: Triple[int],
        boundary_domain_kinds: Triple[BoundaryDomainKind],
        boundary_domain_widths: Triple[Union[int, Pair[int]]],
    ) -> None:
        assert all(size >= 3 for size in shape)

        self.shape = shape
        self.boundary_domain_kinds = boundary_domain_kinds
        self.boundary_domain_widths = tuple(
            (item, item) if isinstance(item, int) else item for item in boundary_domain_widths
        )

    def get_indices(self) -> Indices:
        index_spaces = self.get_index_spaces()
        index_conventions = self.get_index_conventions()
        add_periodic_images(index_spaces, index_conventions)
        return Indices(index_spaces, index_conventions)

    @abstractmethod
    def get_index_spaces(self) -> dict[Hashable, IndexSpace]:
        ...

    @abstractmethod
    def get_index_conventions(self) -> dict[Hashable, dict[Hashable, Triple[int]]]:
        ...


class MedianDual(IndicesBuilder):
    """
    Along any direction, the number of mass points is *smaller* than the
    number of staggered points by a unity.
    """

    def get_index_spaces(self):
        construction_args = {
            (I, J, K): {
                "shape": self.shape,
                "boundary_domain_kinds": self.boundary_domain_kinds,
                "boundary_domain_widths": self.boundary_domain_widths,
                "periodic_layers": (2, 2, 2),
                "has_boundary": (True, True, True),
                "subdomain_label_from_dir": subdomain_label_from_dir_ij,
            },
            (I - 1 / 2, J, K): {
                "shape": get_staggered_field_shape(self.shape, self.boundary_domain_kinds, 0),
                "boundary_domain_kinds": self.boundary_domain_kinds,
                "boundary_domain_widths": self.boundary_domain_widths,
                "periodic_layers": ((0, 1), 0, 0),
                "has_boundary": (True, False, False),
                "subdomain_label_from_dir": subdomain_label_from_dir_ij,
            },
            (I, J - 1 / 2, K): {
                "shape": get_staggered_field_shape(self.shape, self.boundary_domain_kinds, 1),
                "boundary_domain_kinds": self.boundary_domain_kinds,
                "boundary_domain_widths": self.boundary_domain_widths,
                "periodic_layers": (0, (0, 1), 0),
                "has_boundary": (False, True, False),
                "subdomain_label_from_dir": subdomain_label_from_dir_ij,
            },
            (I, J, K - 1 / 2): {
                "shape": get_staggered_field_shape(self.shape, self.boundary_domain_kinds, 2),
                "boundary_domain_kinds": self.boundary_domain_kinds,
                "boundary_domain_widths": self.boundary_domain_widths,
                "periodic_layers": (0, 0, (0, 1)),
                "has_boundary": (False, False, True),
                "subdomain_label_from_dir": subdomain_label_from_dir_ij,
            },
        }

        # alter construction args to lower dimensional index spaces
        construction_args = {
            **construction_args,
            **deduce_lower_dim_index_space_construction_args(construction_args),
        }

        # actually construct the index spaces
        index_spaces = {k: create_index_space(**args) for k, args in construction_args.items()}

        # add non-periodic index spaces (derived from existing ones)
        #  useful for fields that are defined everywhere, but are not periodic
        #  e.g. the coordinate fields xcr, ycr, zcr
        index_spaces = {
            **index_spaces,
            ("non_periodic", (I, J, K)): remove_periodic_layers_from_index_space(
                index_spaces[(I, J, K)]
            ),
        }

        return index_spaces

    def get_index_conventions(self):
        conventions = {
            (I, J, K): {
                (I, J, K): (0, 0, 0),
                (I - 1 / 2, J, K): (0, 0, 0),
                (I, J - 1 / 2, K): (0, 0, 0),
                (I, J, K - 1 / 2): (0, 0, 0),
            },
            (I - 1 / 2, J, K): {(I - 1 / 2, J, K): (0, 0, 0), (I, J, K): (-1, 0, 0)},
            (I, J - 1 / 2, K): {(I, J - 1 / 2, K): (0, 0, 0), (I, J, K): (0, -1, 0)},
            (I, J, K - 1 / 2): {(I, J, K - 1 / 2): (0, 0, 0), (I, J, K): (0, 0, -1)},
        }
        conventions = deduce_lower_dim_index_conventions(conventions)
        conventions = deduce_non_periodic_index_conventions(conventions, [(I, J, K)])
        return conventions


class VertexCentered(IndicesBuilder):
    """
    Along any direction, the number of mass points is *greater* than the
    number of staggered points by a unity.
    """

    def get_index_spaces(self):
        construction_args = {
            (I, J, K): {
                "shape": self.shape,
                "boundary_domain_kinds": self.boundary_domain_kinds,
                "boundary_domain_widths": self.boundary_domain_widths,
                "periodic_layers": (2, 2, 2),
                "has_boundary": (True, True, True),
                "subdomain_label_from_dir": subdomain_label_from_dir_ij,
            },
            (I - 1 / 2, J, K): {
                "shape": (self.shape[0] - 1, self.shape[1], self.shape[2]),
                "boundary_domain_kinds": self.boundary_domain_kinds,
                "boundary_domain_widths": self.boundary_domain_widths,
                "periodic_layers": ((0, 1), 0, 0),
                "has_boundary": (True, False, False),
                "subdomain_label_from_dir": subdomain_label_from_dir_ij,
            },
            (I, J - 1 / 2, K): {
                "shape": (self.shape[0], self.shape[1] - 1, self.shape[2]),
                "boundary_domain_kinds": self.boundary_domain_kinds,
                "boundary_domain_widths": self.boundary_domain_widths,
                "periodic_layers": (0, (0, 1), 0),
                "has_boundary": (False, True, False),
                "subdomain_label_from_dir": subdomain_label_from_dir_ij,
            },
            (I, J, K - 1 / 2): {
                "shape": (self.shape[0], self.shape[1], self.shape[2] - 1),
                "boundary_domain_kinds": self.boundary_domain_kinds,
                "boundary_domain_widths": self.boundary_domain_widths,
                "periodic_layers": (0, 0, (0, 1)),
                "has_boundary": (False, False, True),
                "subdomain_label_from_dir": subdomain_label_from_dir_ij,
            },
        }

        # alter construction args to lower dimensional index spaces
        construction_args = {
            **construction_args,
            **deduce_lower_dim_index_space_construction_args(construction_args),
        }

        # actually construct the index spaces
        index_spaces = {k: create_index_space(**args) for k, args in construction_args.items()}

        # add non-periodic index spaces (derived from existing ones)
        #  useful for fields that are defined everywhere, but are not periodic
        #  e.g. the coordinate fields xcr, ycr, zcr
        index_spaces = {
            **index_spaces,
            ("non_periodic", (I, J, K)): remove_periodic_layers_from_index_space(
                index_spaces[I, J, K]
            ),
        }

        return index_spaces

    def get_index_conventions(self):
        conventions = {
            (I, J, K): {
                (I, J, K): (0, 0, 0),
                (I - 1 / 2, J, K): (0, 0, 0),
                (I, J - 1 / 2, K): (0, 0, 0),
                (I, J, K - 1 / 2): (0, 0, 0),
            },
            (I - 1 / 2, J, K): {(I - 1 / 2, J, K): (0, 0, 0), (I, J, K): (-1, 0, 0)},
            (I, J - 1 / 2, K): {(I, J - 1 / 2, K): (0, 0, 0), (I, J, K): (0, -1, 0)},
            (I, J, K - 1 / 2): {(I, J, K - 1 / 2): (0, 0, 0), (I, J, K): (0, 0, -1)},
        }
        conventions = deduce_lower_dim_index_conventions(conventions)
        conventions = deduce_non_periodic_index_conventions(conventions, [(I, J, K)])
        return conventions
