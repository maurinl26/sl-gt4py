import pytest
from typing import Union
import numpy as np

@pytest.fixture(name="dtypes", scope="module")
def dtypes_fixtures(precision: Union["single", "double"] = 'single'):
    if precision == "single":
        return {
            float: np.float32,
            int: np.int32,
        }
    elif precision == "double":
        return {
            float: np.float64,
            int: np.int64,
        }
    else:
        raise KeyError("precision not in 'single' or 'double'")

@pytest.fixture(name="origin", scope="module")
def origin_with_halo_fixture(h: int = 5):
    return (h, h, 0)

@pytest.fixture(name="domain_with_halo", scope="module")
def domain_with_halo_fixture(nx: int = 50, ny: int = 50, nz: int = 10):
    return nx, ny, nz

@pytest.fixture(name="inner_domain", scope="module")
def inner_domain_fixture(nx: int = 50, ny: int = 50, nz: int = 10, h: int = 5):
    return nx - 2*h, ny - 2*h, nz