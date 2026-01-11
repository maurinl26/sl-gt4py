# -*- coding: utf-8 -*-
from __future__ import annotations
from cached_property import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable
    from typing import Literal, Union

    from sl_gt4py.utils.index_space import CartesianSet, IndexSpace
    from sl_gt4py.utils.typingx import Triple

    DirType = Literal[-1, 0, 1]

# dummy type signifying that a dimension has been removed
RemovedDimension = type("RemovedDimension", (), {})


class PeriodicImageId:
    """Identifier of a periodic image."""

    id_: Hashable
    dir: Triple[DirType]  # direction, e.g. (-1, -1, 0) is north-west

    def __init__(self, id_: Hashable, dir: Triple[DirType]) -> None:
        self.id_ = id_
        self.dir = dir

    def __repr__(self):
        return f"PeriodicImageId(id_={self.id_}, dir={self.dir})"


class Indices:
    """A collection of index spaces and index conventions."""

    index_spaces: dict[Hashable, IndexSpace]
    index_conventions: dict[Hashable, dict[Hashable, Triple[int]]]

    def __init__(
        self,
        index_spaces: dict[Hashable, IndexSpace],
        index_conventions: dict[Hashable, dict[Hashable, Triple[int]]],
    ) -> None:
        assert all(id_ in index_spaces for id_ in index_conventions)

        self.index_spaces = index_spaces
        self.index_conventions = index_conventions

        with open("index.txt", "w") as f:
            for name, space in self.index_spaces.items():
                print("################################################", file=f)
                print(name, file=f)
                print(space, file=f)

    @cached_property
    def periodic_images(self) -> dict[Hashable, dict[PeriodicImageId, IndexSpace]]:
        out = {}
        for key, index_space in self.index_spaces.items():
            if isinstance(key, PeriodicImageId):
                periodic_images = out.setdefault(key.id_, {})
                periodic_images[key] = index_space
            else:
                out.setdefault(key, {})
        return out

    def transform(
        self, transformer: Callable[[IndexSpace], IndexSpace], ignore_periodic_images: bool = False
    ) -> Indices:
        new_index_spaces = {
            key: transformer(index_space)
            for key, index_space in self.index_spaces.items()
            if not (ignore_periodic_images and isinstance(key, PeriodicImageId))
        }
        return Indices(new_index_spaces, self.index_conventions)

    def __getitem__(self, args: Triple[Union[int, slice]]) -> Indices:
        """Create a new `Indices` by slicing all index spaces."""
        return self.transform(lambda index_space: index_space[args], ignore_periodic_images=True)

    def intersect(self, mask: CartesianSet) -> Indices:
        """
        Create a new `Indices` by intersecting all index spaces with `mask`.
        """
        return self.transform(
            lambda index_space: index_space.intersect(mask), ignore_periodic_images=True
        )

    def translate(self, *offset: int) -> Indices:
        """Translate all index spaces by `offset`."""
        return self.transform(
            lambda index_space: index_space.translate(*offset), ignore_periodic_images=False
        )
