import numpy as np

def periodic_overshoot_filter(
    tracer_e: np.ndarray,
    tracer_sup: np.ndarray,
    nx: int,
    ny: int
):
    
    overshoot = np.maximum(tracer_e - tracer_sup, 0)
    
    # Inner domain
    for i in range(0, nx):
        for j in range(0, ny):
            # Distribution en cas d'overshoot
            slot_1 = max(tracer_sup[i % nx, j % ny] - tracer_e[((i - 1) % nx) % nx, ((j - 1) % ny) % ny], 0)
            slot_2 = max(tracer_sup[i % nx, j % ny] - tracer_e[i % nx, ((j - 1) % ny) % ny], 0)
            slot_3 = max(tracer_sup[i % nx, j % ny] - tracer_e[((i + 1) % nx) % nx, ((j - 1) % ny) % ny], 0)
            slot_4 = max(tracer_sup[i % nx, j % ny] - tracer_e[((i + 1) % nx) % nx, j % ny], 0)
            slot_5 = max(tracer_sup[i % nx, j % ny] - tracer_e[((i + 1) % nx) % nx, ((j + 1) % ny) % ny], 0)
            slot_6 = max(tracer_sup[i % nx, j % ny] - tracer_e[i %  nx, ((j + 1) % ny) % ny], 0)
            slot_7 = max(tracer_sup[i % nx, j % ny] - tracer_e[((i - 1) % nx) % nx, ((j + 1) % ny) % ny], 0)
            slot_8 = max(tracer_sup[i % nx, j % ny] - tracer_e[((i - 1) % nx) % nx, j % ny], 0)

            total_disp = (
                slot_1 + slot_2 + slot_3 + slot_4 + slot_5 + slot_6 + slot_7 + slot_8
            )

            # inner domain
            if total_disp > overshoot[i, j]:
                tracer_e[i, j] -= overshoot[i, j]
                tracer_e[((i - 1) % nx) % nx, ((j - 1) % ny) % ny] += (slot_1 / total_disp) * overshoot[i, j]
                tracer_e[i % nx, ((j - 1) % ny) % ny] += (slot_2 / total_disp) * overshoot[i, j]
                tracer_e[((i + 1) % nx) % nx, ((j - 1) % ny) % ny] += (slot_3 / total_disp) * overshoot[i, j]
                tracer_e[((i + 1) % nx) % nx, j % ny] += (slot_4 / total_disp) * overshoot[i, j]
                tracer_e[((i + 1) % nx) % nx, ((j + 1) % ny) % ny] += (slot_5 / total_disp) * overshoot[i, j]
                tracer_e[i % nx, ((j + 1) % ny) % ny] += (slot_6 / total_disp) * overshoot[i, j]
                tracer_e[((i - 1) % nx) % nx, ((j + 1) % ny) % ny] += (slot_7 / total_disp) * overshoot[i, j]
                tracer_e[((i - 1) % nx) % nx, j % ny] += (slot_8 / total_disp) * overshoot[i, j]
                
    return tracer_e

def periodic_undershoot_filter(
    tracer_e: np.ndarray,
    tracer_inf: np.ndarray,
    nx: int,
    ny: int
):
    
    undershoot = np.maximum(tracer_inf - tracer_e, np.finfo(np.float64).tiny)
    
    # Inner domain
    for i in range(0, nx):
        for j in range(0, ny):
            # Distribution en cas d'undershoot
            slot_1 = max(tracer_e[(i - 1) % nx, (j - 1) % ny] - tracer_inf[i, j], 0)
            slot_2 = max(tracer_e[i % nx, (j - 1) % ny] - tracer_inf[i, j], 0)
            slot_3 = max(tracer_e[(i + 1) % nx, (j - 1) % ny] - tracer_inf[i, j], 0)
            slot_4 = max(tracer_e[(i + 1) % nx, j % ny] - tracer_inf[i, j], 0)
            slot_5 = max(tracer_e[(i + 1) % nx, (j + 1) % ny] - tracer_inf[i, j], 0)
            slot_6 = max(tracer_e[i % nx, (j + 1) % ny] - tracer_inf[i, j], 0)
            slot_7 = max(tracer_e[(i - 1) % nx, (j + 1) % ny] - tracer_inf[i, j], 0)
            slot_8 = max(tracer_e[(i - 1) % nx, j % ny] - tracer_inf[i, j], 0)

            total_disp = (
                slot_1 + slot_2 + slot_3 + slot_4 + slot_5 + slot_6 + slot_7 + slot_8
            )

            # inner domain
            if total_disp > undershoot[i, j]:
                tracer_e[i, j]  += undershoot[i, j]
                tracer_e[(i - 1) % nx, (j - 1) % ny] -= (slot_1 / total_disp) * undershoot[i, j]
                tracer_e[i % nx, (j - 1) % ny] -= (slot_2 / total_disp) * undershoot[i, j]
                tracer_e[(i + 1) % nx, (j - 1) % ny] -= (slot_3 / total_disp) * undershoot[i, j]
                tracer_e[(i + 1) % nx, j % ny] -= (slot_4 / total_disp) * undershoot[i, j]
                tracer_e[(i + 1) % nx, (j + 1) % ny] -= (slot_5 / total_disp) * undershoot[i, j]
                tracer_e[i % nx, (j + 1) % ny] -= (slot_6 / total_disp) * undershoot[i, j]
                tracer_e[(i - 1) % nx, (j + 1) % ny] -= (slot_7 / total_disp) * undershoot[i, j]
                tracer_e[(i - 1) % nx, j % ny] -= (slot_8 / total_disp) * undershoot[i, j]
                
    return tracer_e
