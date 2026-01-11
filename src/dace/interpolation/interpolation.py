import numpy as np
import dace

from sl_python.boundaries import boundaries

def p0_lin(l: float):
    return 1 - l

def p1_lin(l: float):
    return l

def interpolate_lin_2d(
    psi: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
    i_d: np.ndarray,
    j_d: np.ndarray,
    bcx_kind: int,
    bcy_kind: int,
    nx: int,
    ny: int,
    nz: int
):
    """Perform a 1d linear interpolation

    Args:
        lx (np.ndarray): _description_
        ly (np.ndarray): _description_
        psi (np.ndarray): _description_
        i_d (np.ndarray): _description_
        j_d (np.ndarray): _description_
        bcx_kind (int): _description_
        bcy_kind (int): _description_
        nx (int): _description_
        ny (int): _description_
    """

    # Interp selon x -> field_hat_x
    px0 = p0_lin(lx)
    px1 = p1_lin(lx)
    py0 = p0_lin(ly)
    py1 = p1_lin(ly)

    # 1. Construire les tableaux d'indices i_d0, i_d1 / j_d0, j_d1
    # Non periodique
    id_0 = boundaries(i_d, nx, bcx_kind)
    id_p1 = boundaries(i_d + 1, nx, bcx_kind)
    
    jd_0 = boundaries(j_d, ny, bcy_kind)
    jd_p1 = boundaries(j_d + 1, ny, bcy_kind)

    # Lookup
    psi_d_i = np.zeros((4, nx, ny, nz))
    for k in range(nz):
        for i in range(nx):
            for j in range(ny):
                psi_d_i[0, i, j, k] = psi[id_0[i, j, k], jd_0[i, j, k]]
                psi_d_i[1, i, j, k] = psi[id_p1[i, j, k], jd_0[i, j, k]]

                psi_d_i[2, i, j, k] = psi[id_0[i, j, k], jd_p1[i, j, k]]
                psi_d_i[3, i, j, k] = psi[id_p1[i, j, k], jd_p1[i, j, k]]

    psi_d_j = np.zeros((2, nx, ny, nz))
    psi_d_j[0] = px0 * psi_d_i[0] + px1 * psi_d_i[1]
    psi_d_j[1] = px0 * psi_d_i[2] + px1 * psi_d_i[3]

    psi_d = py0 * psi_d_j[0] + py1 * psi_d_j[1]

    return psi_d

# Described as arrays
def interpolate_cub_2d(
    psi: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
    i_d: np.ndarray,
    j_d: np.ndarray,
    bcx_kind: int,
    bcy_kind: int,
    nx: int,
    ny: int,
    nz: int,
):
    """_summary_

    Args:
        psi (np.ndarray): _description_
        lx (np.ndarray): _description_
        ly (np.ndarray): _description_
        i_d (np.ndarray): _description_
        j_d (np.ndarray): _description_
        bcx_kind (int): _description_
        bcy_kind (int): _description_
        nx (int): _description_
        ny (int): _description_

    Returns:
        _type_: _description_
    """
    # Polynomes de Lagrange
    p_1 = lambda l: (1 / 6) * l * (l - 1) * (2 - l)
    p0 = lambda l: (1 - l**2) * (1 - l / 2)
    p1 = lambda l: (1 / 2) * l * (l + 1) * (2 - l)
    p2 = lambda l: (1 / 6) * l * (l**2 - 1)

    px = np.array([p_1(lx), p0(lx), p1(lx), p2(lx)])
    py = np.array([p_1(ly), p0(ly), p1(ly), p2(ly)])

    # 1. Tableaux d'indices
    # Non periodique
    
    id_m1 = boundaries(i_d - 1, nx, bcx_kind)
    id_0 = boundaries(i_d, nx, bcx_kind)
    id_p1 = boundaries(i_d + 1, nx, bcx_kind)
    id_p2 = boundaries(i_d + 2, nx, bcx_kind)
    
    jd_m1 = boundaries(j_d - 1, ny, bcy_kind)
    jd_0 = boundaries(j_d, ny, bcy_kind)
    jd_p1 = boundaries(j_d + 1, ny, bcy_kind)
    jd_p2 = boundaries(j_d + 2, ny, bcy_kind)
    

    # Lookup
    psi_d_i = np.zeros((16, nx, ny, nz))
    for k in range(nz):
        for i in range(nx):
            for j in range(ny):
                psi_d_i[0, i, j, k] = psi[id_m1[i, j, k], jd_m1[i, j, k]]
                psi_d_i[1, i, j, k] = psi[id_0[i, j, k], jd_m1[i, j, k]]
                psi_d_i[2, i, j, k] = psi[id_p1[i, j, k], jd_m1[i, j, k]]
                psi_d_i[3, i, j, k] = psi[id_p2[i, j, k], jd_m1[i, j, k]]

                psi_d_i[4, i, j, k] = psi[id_m1[i, j, k], jd_0[i, j, k]]
                psi_d_i[5, i, j, k] = psi[id_0[i, j, k], jd_0[i, j, k]]
                psi_d_i[6, i, j, k] = psi[id_p1[i, j, k], jd_0[i, j, k]]
                psi_d_i[7, i, j, k] = psi[id_p2[i, j, k], jd_0[i, j, k]]

                psi_d_i[8, i, j, k] = psi[id_m1[i, j, k], jd_p1[i, j, k]]
                psi_d_i[9, i, j, k] = psi[id_0[i, j, k], jd_p1[i, j, k]]
                psi_d_i[10, i, j, k] = psi[id_p1[i, j, k], jd_p1[i, j, k]]
                psi_d_i[11, i, j, k] = psi[id_p2[i, j, k], jd_p1[i, j, k]]

                psi_d_i[12, i, j, k] = psi[id_m1[i, j, k], jd_p2[i, j, k]]
                psi_d_i[13, i, j, k] = psi[id_0[i, j, k], jd_p2[i, j, k]]
                psi_d_i[14, i, j, k] = psi[id_p1[i, j, k], jd_p2[i, j, k]]
                psi_d_i[15, i, j, k] = psi[id_p2[i, j, k], jd_p2[i, j, k]]

    psi_d_j = np.zeros((4, nx, ny, nz))
    psi_d_j[0] = (
        px[0] * psi_d_i[0]
        + px[1] * psi_d_i[1]
        + px[2] * psi_d_i[2]
        + px[3] * psi_d_i[3]
    )
    psi_d_j[1] = (
        px[0] * psi_d_i[4]
        + px[1] * psi_d_i[5]
        + px[2] * psi_d_i[6]
        + px[3] * psi_d_i[7]
    )
    psi_d_j[2] = (
        px[0] * psi_d_i[8]
        + px[1] * psi_d_i[9]
        + px[2] * psi_d_i[10]
        + px[3] * psi_d_i[11]
    )
    psi_d_j[3] = (
        px[0] * psi_d_i[12]
        + px[1] * psi_d_i[13]
        + px[2] * psi_d_i[14]
        + px[3] * psi_d_i[15]
    )

    psi_d = (
        py[0] * psi_d_j[0]
        + py[1] * psi_d_j[1]
        + py[2] * psi_d_j[2]
        + py[3] * psi_d_j[3]
    )

    return psi_d