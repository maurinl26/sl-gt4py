# -*- coding: utf-8 -*-
import numpy as np
from typing import TypeVar, Union, Tuple

try:
    import cupy as cp
except ImportError:
    cp = None


T = TypeVar("T")

Pair = Tuple[T, T]
Triple = Tuple[T, T, T]

if cp is not None:
    ArrayLike = Union[np.ndarray, cp.ndarray]
else:
    ArrayLike = np.ndarray
