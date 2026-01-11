import numpy as np
import matplotlib.pyplot as plt


def diagnostic_interpolation(field: np.ndarray, i: int, j: int, j_step: int, dx: float):
    """Trace les polynomes de lagrange associés

    Args:
        tracer (np.ndarray): _description_
        i (int): _description_
        j (int): _description_
    """
    
    lx = np.linspace(-1, 2, 30)
    ly = np.linspace(-1, 2, 30)
    
    # Polynomes de lagrange d'ordre 3
    p_1 = lambda l: (1/6) * l * (l - 1) * (2 - l) 
    p0 = lambda l: (1 - l**2) * (1 - l / 2)
    p1 = lambda l: (1/2) * l * (l + 1) * (2 - l)
    p2 = lambda l: (1/6) * l * (l**2 - 1) 
    
    g = lambda l: 20 * np.exp(-(l + (i - 6 - j_step * 1.5))**2 / 4)

    p0_x = [p_1(l)*field[i-1, j] + p0(l)*field[i, j] + p1(l)*field[i+1, j] + p2(l)*field[i+2, j] for l in lx]
    g_x = [g(l) for l in lx]
    
    plt.plot(lx, p0_x)
    plt.plot(lx, g_x)
    plt.show()
    
    
def diagnostic_lipschitz(u: np.ndarray, v: np.ndarray, dx: float, dy: float, dth: float):
    """Diagnostic for Semi-Lagrangian Research stability

    Args:
        u (np.ndarray): velocity on x 
        v (np.ndarray): velocity on y
        dx (float): spacing on x
        dy (float): spacing on y
        dth (float): half time step

    Returns:
        float: lipschitz condition on stability
    """
    
    dudx = (1/dx) * np.gradient(u, axis=0)
    dudy = (1/dy) * np.gradient(u, axis=1)
    dvdx = (1/dx) * np.gradient(v, axis=0)
    dvdy = (1/dy) * np.gradient(v, axis=1)
    
    return dth * np.maximum(np.maximum(dudx, dudy), np.maximum(dvdx, dvdy))


def diagnostic_overshoot(
    tracer_e: np.ndarray,
    tracer_sup: np.ndarray,
)-> np.ndarray:
    """Compute overshoots

    Args:
        tracer_e (np.ndarray): interpolated field
        tracer_sup (np.ndarray): sup field

    Returns:
        np.ndarray: overshoots
    """
    
    return np.maximum(tracer_e - tracer_sup, np.finfo(np.float64).tiny)

def diagnostic_undershoot(
    tracer_e: np.ndarray,
    tracer_inf: np.ndarray
):
    """Compute undershoots

    Args:
        tracer_e (np.ndarray): interpolated field
        tracer_inf (np.ndarray): inf field
    """
    
    return np.maximum(tracer_inf - tracer_e, np.finfo(np.float64).tiny)

