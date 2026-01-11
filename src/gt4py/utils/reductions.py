# -*- coding: utf-8 -*-
from collections.abc import Callable
import numpy as np
import sys

from sl_gt4py.utils.storage import Field


def generate_wrapper(np_func_name: str) -> Callable[[Field, ...], ...]:
    np_func = getattr(np, np_func_name)

    def _wrapper(field: Field, *args, **kwargs):
        assert isinstance(field, Field)
        result = np_func(field["physical"], *args, **kwargs)
        # cupy will return a 0-dim array, use item() to get a cpu scalar
        return result.item()

    return _wrapper


for func in ["min", "max", "sum", "average"]:
    setattr(sys.modules[__name__], func, generate_wrapper(func))
