import numpy as np
from typing import Tuple

# Pointwise functions
def r(x, y):
    return np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)


def r_tilde(x, y):
    return 5 * np.sqrt((x - 0.3) ** 2 + (y - 0.5) ** 2)


def theta(x, y):
    return np.arctan((y - 0.5) / (x - 0.5))


def u(r, t, T):
    return (4 * np.pi * r / T) * (
        1 + np.cos(2 * np.pi * t / T) * (1 - (4 * r) ** 6) / (1 + (4 * r) ** 6)
    )


def tracer_shape(x: float, y: float, tracer0: float):
    r_t = r_tilde(x, y)
    return np.where(r_t < 1, tracer0 + ((1 + np.cos(np.pi * r_t)) / 2) ** 2, tracer0)


# Field functions
def polar_coordinates(xcr: np.ndarray, ycr: np.ndarray):
    R = r(xcr, ycr)
    Theta = theta(xcr, ycr)
    return R, Theta


# Taken from Christian Kühnlein
def blossey_stf(xcr: np.ndarray, ycr: np.ndarray, mt: float) -> np.ndarray:
    """Compute blossey stream function as described by
    Durran & Blossey (Selective monotonicity preservation in scalar advection)

    Args:
        t (float): _description_
        T (float): _description_
        r (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    rsq = (xcr - 0.5) ** 2.0 + (ycr - 0.5) ** 2.0

    term1 = 0.5 * rsq
    term2 = np.log(1.0 - 16.0 * rsq + 256.0 * rsq**2.0) / 96.0
    term3 = -np.log(1.0 + 16.0 * rsq) / 48.0
    term4 = -np.sqrt(3.0) * np.arctan((-1.0 + 32.0 * rsq) / np.sqrt(3.0)) / 48.0

    stf = (
        4.0
        * np.pi
        * (term1 + np.cos(2.0 * np.pi * mt) * (term1 + term2 + term3 + term4))
    )

    return stf


# Taken from FVM Slim (Christian)
def stream_function_xy(stf: np.ndarray, dx, dy) -> Tuple[np.ndarray]:
    """Computes velocity components based on stream function

    Args:
        stf (np.ndarray): stream function

    Returns:
        Tuple[np.ndarray]: u (velocity on x axis), v (velocity on y axis)
    """
    
    # TODO : implementation du gradient a la main
    dstf_dx = np.gradient(stf, dx, axis=0)
    dstf_dy = np.gradient(stf, dy, axis=1)
    u = dstf_dy
    v = - dstf_dx

    return u, v


def blossey_velocity(xcr: np.ndarray, ycr: np.ndarray, mt, dx, dy) -> Tuple[np.ndarray]:
    stf = blossey_stf(xcr, ycr, mt)
    u, v = stream_function_xy(stf, dx, dy)

    return u, v


def init_blossey(xcr: np.ndarray, ycr: np.ndarray, mt, dt, dx, dy, nx: int, ny: int):
    # Vitesses
    vx, vy = blossey_velocity(xcr, ycr, mt, dx, dy)
    vx_p, vy_p = blossey_velocity(xcr, ycr, mt - dt, dx, dy)
    vx_e, vy_e = np.empty((nx, ny)), np.empty((nx, ny))

    return vx, vy, vx_p, vy_p, vx_e, vy_e


def blossey_tracer(xcr, ycr):
    # Tracer
    tracer = tracer_shape(xcr, ycr, 0)
    tracer_e = tracer.copy()
    return tracer, tracer_e
