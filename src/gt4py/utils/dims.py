# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal


# small helper metaclass that ensures equal DimSymbols have the same id
#  resulting in faster __hash__ and __eq__ calls
class Singleton(type):
    _instances = {}
    def __call__(cls, *args):
        key = hash((cls, args))
        if key not in cls._instances:
            cls._instances[key] = super(Singleton, cls).__call__(*args)
        return cls._instances[key]


class DimSymbol(metaclass=Singleton):
    """Symbol representing a dimension, e.g. I or I - 1/2."""

    name: str
    offset: Literal[0, -1/2]

    # careful: no default arguments are allowed as this breaks the Singleton
    #  pattern
    def __init__(self, name: str, offset: Literal[0, -1/2]):
        self.name = name
        self.offset = offset

    def __add__(self, offset: float):
        assert self.offset == 0 or self.offset+offset == 0
        return DimSymbol(self.name, self.offset + offset)

    def __sub__(self, other: float):
        return self+(-other)

    def __repr__(self):
        if self.offset == 0:
            return self.name
        elif self.offset > 0:
            return f"{self.name} + {self.offset}"
        elif self.offset < 0:
            return f"{self.name} - {abs(self.offset)}"
        raise RuntimeError()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


# convenient definitions
I = DimSymbol("I", 0)
J = DimSymbol("J", 0)
K = DimSymbol("K", 0)
