# -*- coding: utf-8 -*-
from __future__ import annotations
from contextlib import contextmanager
import numpy as np
from typing import TYPE_CHECKING

try:
    import cupy as cp
except ImportError:
    cp = None

import gt4py as gt

from sl_gt4py.build import backend, dtype
from sl_gt4py.index_space import CartesianSet, IndexSpace, ProductSet
from sl_gt4py.typingx import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Optional, Union

    from sl_gt4py.utils.indices import Indices


def translate_to_memory_index_convention(
    field_or_idx_space: Union[Field, IndexSpace],
    arg: Union[CartesianSet, tuple[int]],
    allow_undefined: bool = False,
) -> Union[CartesianSet, tuple[int]]:
    """
    Translate set or point to memory index convention

    The memory index convention is defined to have its origin point at 0, 0, 0 and map to the origin of the fields
    index space
    """
    # todo: enable asserts conditionally in a "debug" mode (they are really slow)
    if isinstance(field_or_idx_space, Field):
        field = field_or_idx_space
        index_space = field.indices.index_spaces[field.index_space]
    elif isinstance(field_or_idx_space, IndexSpace):
        index_space = field_or_idx_space
    else:
        raise ValueError("Expected a Field or an IndexSpace.")

    if isinstance(arg, CartesianSet):
        # note(stubbiali): the following assert is commented out as it is too expensive
        # assert arg.issubset(index_space.covering if not allow_undefined else index_space.bounds)
        return arg.translate(*(-idx for idx in index_space.bounds[(0,) * index_space.dim]))
    if isinstance(arg, tuple) and all(isinstance(i, int) for i in arg):
        # note(stubbiali): the following assert is commented out as it is too expensive
        # assert arg in (index_space.covering if not allow_undefined else index_space.bounds)
        return tuple(
            idx - oidx for idx, oidx in zip(arg, index_space.bounds[(0,) * index_space.dim])
        )

    raise ValueError()


class Field:
    indices: Indices
    index_space: Hashable
    data: ArrayLike

    def __init__(
        self,
        indices: Indices,
        index_space: Hashable,
        buffer: Optional[ArrayLike] = None,
    ) -> None:
        self.indices = indices
        self.index_space = index_space

        if buffer is None:
            shape = indices.index_spaces[index_space].shape
            buffer = gt.storage.zeros(
                backend=backend, dtype=dtype, shape=shape, aligned_index=(0,) * len(shape)
            )
        self.data = buffer

    def __copy__(self) -> Field:
        return Field(self.indices, self.index_space, self.data)

    def _slice_for_subdomain(self, subdomain: str) -> tuple[slice]:
        index_space = self.indices.index_spaces[self.index_space]
        set_ = index_space.subset[subdomain]
        if not isinstance(set_, ProductSet):
            raise ValueError("Argument to view must be cubical.")

        mem_indices = translate_to_memory_index_convention(self, set_)
        slices = tuple(slice(arg.start, arg.stop) for arg in mem_indices.args)

        return slices

    def with_altered_index_space(self, index_space: Hashable) -> Field:
        assert (
            self.indices.index_spaces[self.index_space].covering
            == self.indices.index_spaces[index_space].covering
        )
        return Field(self.indices, index_space, self.data)

    def __getitem__(self, subdomain: Union[tuple[slice], str]) -> ArrayLike:
        if isinstance(subdomain, tuple) and all(s == slice(None) for s in subdomain):
            return self.data[...]
        else:
            return self.data[self._slice_for_subdomain(subdomain)]

    def __setitem__(self, subdomain: Union[tuple[slice], str], val: ArrayLike) -> None:
        if cp is not None:
            if isinstance(val, np.ndarray) and isinstance(self.data, cp.ndarray):
                print("Warning: assignment from host buffer to device buffer is slow.")
                val = cp.array(val)
            elif isinstance(val, cp.ndarray) and isinstance(self.data, np.ndarray):
                print("Warning: assignment from device buffer to host buffer is slow.")
                val = cp.asnumpy(val)

        if isinstance(subdomain, tuple) and all(s == slice(None) for s in subdomain):
            self.data[...] = val
        else:
            self.data[self._slice_for_subdomain(subdomain)] = val


def zeros(indices: Indices, index_space: Hashable) -> Field:
    return Field(indices, index_space)


def to_numpy(arr, *, writeable: bool = False) -> np.ndarray:
    """
    Convert a numpy or cupy like array into a base-class numpy array

    Mainly used for functions expecting numpy arrays, e.g. pyplot, that don't work with cupy. The returned value
    is either read-only and/or a copy. This function is not suitable if you need to change the original data in which
    case you either need to support cupy arrays or write back the changes manually after modification.
    """
    if hasattr(arr, "__cuda_array_interface__"):
        data = cp.asnumpy(arr)
    elif hasattr(arr, "__array_interface__"):
        # copy as returning a writable numpy array without copying would break consistency of CPU and GPU code
        data = np.array(arr, copy=writeable)
        # create shallow copy so that we can set writeability without affecting the original array
        data = data[...]
    else:
        raise ValueError("Expected a numpy or cupy array like argument.")

    data.flags.writeable = writeable

    return data


class _StoragePool:
    pool: dict[Hashable, list[Field]]

    def __init__(self) -> None:
        self.pool = {}

    def allocate(self, indices: Indices, index_space: Hashable) -> Field:
        if index_space not in self.pool:
            self.pool[index_space] = []
        if len(self.pool[index_space]) > 0:
            res = self.pool[index_space].pop()
            return res

        return zeros(indices, index_space)

    def release(self, field: Field) -> None:
        self.pool[field.index_space].append(field)


_storage_pools: dict[int, _StoragePool] = {}


@contextmanager
def managed_temporary(indices: Indices, *index_spaces: Hashable):
    with managed_temporary_pool(indices, _implicit=True) as pool:
        pool = _storage_pools[hash(indices)]
        storages = tuple(pool.allocate(indices, index_space) for index_space in index_spaces)
        try:
            yield storages
        finally:
            for storage in storages:
                pool.release(storage)


_implicit_pool_constructions = 0
_implicit_pool_constructions_did_warn = False


@contextmanager
def managed_temporary_pool(indices: Indices, _implicit: bool = False):
    indices_hash = hash(indices)
    if not indices_hash in _storage_pools:
        if _implicit:
            global _implicit_pool_constructions, _implicit_pool_constructions_did_warn
            _implicit_pool_constructions += 1
            if _implicit_pool_constructions > 100:
                _implicit_pool_constructions_did_warn = True
                print(
                    "Warning: Detected many implicit storage pool constructions. If you get this warning"
                    "you have probably forgotten to wrap your code in a `managed_temporary_pool` context."
                    "A common place is around the main time-stepping loop such that temporary storages"
                    "are shared for the entire simulation."
                )
        try:
            # allocate new pool
            _storage_pools[indices_hash] = pool = _StoragePool()
            yield pool
        finally:
            del _storage_pools[indices_hash]
    else:
        # if there is already a pool for the grid use that one
        yield _storage_pools[indices_hash]
