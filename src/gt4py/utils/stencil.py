# -*- coding: utf-8 -*-
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import logging
import multiprocessing
import threading
from typing import TYPE_CHECKING

import gt4py.cartesian as gt

from sl_gt4py.config import config
from sl_gt4py.utils.index_space import ProductSet, UnionCartesian
from sl_gt4py.utils.indices import Indices, RemovedDimension
from sl_gt4py.utils.storage import Field, translate_to_memory_index_convention

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable
    from typing import Union

    from sl_gt4py.utils.index_space import IndexSpace

log = logging.getLogger(__name__)

_thread_pool = None
_thread_pool_lock = threading.Lock()
_pending_compilation_tasks = set()

_stencil_args_cache = {}


def stencil_args(
    indices: Indices,
    index_space: Hashable,
    fields: dict[str, Field],
    *,
    subdomain: Union[str, tuple[str]] = "definition",
) -> list[tuple[dict[str, tuple[int, int, int]], tuple[int, int, int]]]:
    """Compute stencil origin and domain"""
    cache_key = hash(
        (
            indices,
            index_space,
            tuple((name, field.indices, field.index_space) for name, field in fields.items()),
            subdomain,
        )
    )
    if cache_key in _stencil_args_cache:
        return _stencil_args_cache[cache_key]

    # retrieve index convention of the stencil
    idx_conv = indices.index_conventions[index_space]

    # restrict index space to subdomain to obtain stencil indices in definition convention
    def_indices = indices.index_spaces[index_space].subset[subdomain]

    if isinstance(def_indices, ProductSet):
        def_indices_components = [def_indices]
    elif isinstance(def_indices, UnionCartesian):
        assert def_indices.disjoint
        def_indices_components = def_indices.args
    else:
        raise RuntimeError()

    result = []

    for def_indices_component in def_indices_components:
        stencil_origins = {}
        stencil_domain = tuple(bound.size for bound in def_indices_component.bounds.args)

        for name, field in fields.items():
            assert field.indices == indices

            if field.index_space not in idx_conv:
                raise ValueError(
                    f"No index convention for stencil on {index_space} "
                    f"available to access field defined on {field.index_space}."
                )
            field_idx_conv = idx_conv[field.index_space]

            def filter_removed_dim(arr):
                return type(arr)(
                    el
                    for conv_el, el in zip(field_idx_conv, arr)
                    if conv_el is not RemovedDimension
                )

            # remove dimensions from definition indices on which the field is not defined
            def_indices_component_restriced = ProductSet(
                *filter_removed_dim(def_indices_component.args)
            )
            field_idx_conv_restricted = filter_removed_dim(field_idx_conv)

            # translate indices
            #  to definition index convention
            field_def_indices = def_indices_component_restriced.translate(
                *field_idx_conv_restricted
            )
            #  to memory index convention
            field_mem_indices = translate_to_memory_index_convention(field, field_def_indices)

            # stencil origin
            stencil_origins[name] = tuple(bound.start for bound in field_mem_indices.bounds.args)
            assert filter_removed_dim(stencil_domain) == tuple(
                bound.size for bound in field_mem_indices.bounds.args
            )

        result.append((stencil_origins, stencil_domain))

    _stencil_args_cache[cache_key] = result

    return result


def _hash_args(fields: dict[str, Field], args, kwargs) -> int:
    """Hash all stencil arguments (used for debugging purposes)"""
    import numpy as np

    hashes = []
    for arg in [*args, kwargs.values()]:
        if not isinstance(arg, gt.storage.storage.Storage):
            hashes.append(arg)
    for field in fields.values():
        hashes.append(np.average(np.asarray(field["definition"])))
    return hash(tuple(hashes))


def stencil(backend: str, **kwargs) -> Callable[[Indices, Hashable, ...], None]:
    def _decorator(definition):
        # compile stencil object
        stencil_obj = gt.gtscript.stencil(backend, definition, **kwargs)

        # extract mapping from argument position to name
        field_names = list(
            arg_info.name
            for arg_info, arg_annotation in zip(
                definition._gtscript_["api_signature"], definition._gtscript_["api_annotations"]
            )
            if isinstance(arg_annotation, gt.gtscript._FieldDescriptor)
        )

        # extract field names that are modified during a stencil call
        rw_fields = [
            field
            for field, field_info in stencil_obj.field_info.items()
            if field_info
            and field_info.access
            in [gt.definitions.AccessKind.WRITE, gt.definitions.AccessKind.READ_WRITE]
        ]

        # define wrapper computing origin and domain of the regular gt4py stencil api and passing
        #  it to the compiled stencil object
        def _wrapper(
            indices, index_space, *args, subdomain="definition", update_pbcs=None, **kwargs
        ):
            assert "domain" not in kwargs and "origin" not in kwargs

            # abort if the target subdomain is empty
            if indices.index_spaces[index_space].subset[subdomain].empty:
                return

            fields = {}
            # positional arguments
            for arg, arg_info, arg_annotation in zip(
                args,
                definition._gtscript_["api_signature"],
                definition._gtscript_["api_annotations"],
            ):
                fields[arg_info.name] = arg
            # keyword arguments
            for field_name in field_names:
                if field_name in kwargs:
                    fields[field_name] = kwargs[field_name]
            if not len(fields) == len(field_names):
                raise ValueError(f"Expected {len(field_names)} arguments, but got {len(fields)}.")

            # convert fields into storages
            args = tuple(arg.data if isinstance(arg, Field) else arg for arg in args)
            kwargs = {k: v.data if isinstance(v, Field) else v for k, v in kwargs.items()}

            if build_config.debug_stencil_calls:
                print(
                    f"Stencil call: {definition.__module__} {definition.__name__}, "
                    f"state: enter, hash: {_hash_args(fields, args, kwargs)}"
                )

            for stencil_origins, stencil_domain in stencil_args(
                indices, index_space, fields, subdomain=subdomain
            ):
                stencil_obj(
                    *args,
                    origin=stencil_origins,
                    domain=stencil_domain,
                    validate_args=build_config.validate_stencil_args,
                    **kwargs,
                )

            # update periodic layers
            if (
                update_pbcs == None and subdomain != "covering"
            ) or update_pbcs:  # if the stencil is running everywhere, the periodic layers don't need to be updated
                for rw_field in rw_fields:
                    from .periodic_bcs import update_periodic_layers

                    update_periodic_layers(indices, fields[rw_field])

            if build_config.debug_stencil_calls:
                print(
                    f"Stencil call: {definition.__module__} {definition.__name__}, "
                    f"state: exit, hash: {_hash_args(fields, args, kwargs)}"
                )

        return _wrapper

    def parallel_compilation_decorator(definition):
        assert threading.current_thread() is threading.main_thread()
        global _thread_pool

        future = None

        def compilation_task():
            # compile
            result = None
            try:
                result = _decorator(definition)
            except Exception as e:
                with _thread_pool_lock:
                    import traceback

                    print(
                        f"An error occurred compiling stencil "
                        f"`{definition.__module__}.{definition.__name__}`:"
                    )
                    traceback.print_exc()

            # cleanup
            with _thread_pool_lock:
                _pending_compilation_tasks.remove(future)
                if len(_pending_compilation_tasks) == 0:
                    global _thread_pool
                    _thread_pool.shutdown(wait=False)
                    _thread_pool = None
            return result

        with _thread_pool_lock:
            if _thread_pool is None:
                _thread_pool = ThreadPoolExecutor(multiprocessing.cpu_count())

            future = _thread_pool.submit(compilation_task)
            _pending_compilation_tasks.add(future)

        def _wrapper(*args, **kwargs):
            global _thread_pool
            assert threading.current_thread() is threading.main_thread()

            result = future.result()
            if __debug__:
                if not result:
                    raise RuntimeError(
                        f"Unable to execute stencil call to `{definition.__name__}` as compilation "
                        f"failed. Check previous errors.`"
                    )

            return result(*args, **kwargs)

        # with _thread_pool_lock:
        #    print(
        #        f"Queued compilation of `{definition.__module__}.{definition.__name__}` "
        #        f"stencil (queue length: {len(_pending_compilation_tasks)})"
        #    )

        return _wrapper

    # todo(tehrengruber): useful, but sometimes throws an error
    if build_config.parallel_compilation:
        return parallel_compilation_decorator
    else:
        return _decorator


def wait_for_compilation_completion():
    if build_config.parallel_compilation:
        logging.info("Waiting for stencil compilation to finish.")

        thread_pool = None
        # use lock to retrieve thread pool avoiding race-condition
        with _thread_pool_lock:
            global _thread_pool
            thread_pool = _thread_pool
        if thread_pool:
            thread_pool.shutdown(wait=True)
