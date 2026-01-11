import numpy as np

from sl_python.boundaries import boundaries

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
    # Polynômes de lagrange
    p0 = lambda l: 1 - l
    p1 = lambda l: l

    # Interp selon x -> field_hat_x
    px = np.array([p0(lx), p1(lx)])
    py = np.array([p0(ly), p1(ly)])

    # 1. Construire les tableaux d'indices i_d0, i_d1 / j_d0, j_d1
    # Non periodique
    id_0 = boundaries(i_d, nx, bcx_kind)
    id_p1 = boundaries(i_d + 1, nx, bcx_kind)
    
    jd_0 = boundaries(j_d, ny, bcy_kind)
    jd_p1 = boundaries(j_d + 1, ny, bcy_kind)

    # Lookup
    psi_d_i = np.zeros((4, nx, ny))
    for i in range(nx):
        for j in range(ny):
            psi_d_i[0, i, j] = psi[id_0[i, j], jd_0[i, j]]
            psi_d_i[1, i, j] = psi[id_p1[i, j], jd_0[i, j]]

            psi_d_i[2, i, j] = psi[id_0[i, j], jd_p1[i, j]]
            psi_d_i[3, i, j] = psi[id_p1[i, j], jd_p1[i, j]]

    psi_d_j = np.zeros((2, nx, ny))
    psi_d_j[0] = px[0] * psi_d_i[0] + px[1] * psi_d_i[1]
    psi_d_j[1] = px[0] * psi_d_i[2] + px[1] * psi_d_i[3]

    psi_d = py[0] * psi_d_j[0] + py[1] * psi_d_j[1]

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
    psi_d_i = np.zeros((16, nx, ny))
    for i in range(nx):
        for j in range(ny):
            psi_d_i[0, i, j] = psi[id_m1[i, j], jd_m1[i, j]]
            psi_d_i[1, i, j] = psi[id_0[i, j], jd_m1[i, j]]
            psi_d_i[2, i, j] = psi[id_p1[i, j], jd_m1[i, j]]
            psi_d_i[3, i, j] = psi[id_p2[i, j], jd_m1[i, j]]

            psi_d_i[4, i, j] = psi[id_m1[i, j], jd_0[i, j]]
            psi_d_i[5, i, j] = psi[id_0[i, j], jd_0[i, j]]
            psi_d_i[6, i, j] = psi[id_p1[i, j], jd_0[i, j]]
            psi_d_i[7, i, j] = psi[id_p2[i, j], jd_0[i, j]]

            psi_d_i[8, i, j] = psi[id_m1[i, j], jd_p1[i, j]]
            psi_d_i[9, i, j] = psi[id_0[i, j], jd_p1[i, j]]
            psi_d_i[10, i, j] = psi[id_p1[i, j], jd_p1[i, j]]
            psi_d_i[11, i, j] = psi[id_p2[i, j], jd_p1[i, j]]

            psi_d_i[12, i, j] = psi[id_m1[i, j], jd_p2[i, j]]
            psi_d_i[13, i, j] = psi[id_0[i, j], jd_p2[i, j]]
            psi_d_i[14, i, j] = psi[id_p1[i, j], jd_p2[i, j]]
            psi_d_i[15, i, j] = psi[id_p2[i, j], jd_p2[i, j]]

    psi_d_j = np.zeros((4, nx, ny))
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


def interpolate_cub_irregular_2D(
    zcr: np.ndarray,
    psi: np.ndarray,
    lx: np.ndarray,
    lz: np.ndarray,
    i: np.ndarray,
    k: np.ndarray,
    bcx_kind: int,
    nx: int,
    nz: int,
):
    # Polynomes de Lagrange
    p_1 = lambda l: (
        ((l + zcr[i, k] - zcr[i, k]) / (zcr[i, k - 1] - zcr[i, k]))
        * ((l + zcr[i, k] - zcr[i, k + 1]) / (zcr[i, k - 1] - zcr[i, k + 1]))
        * ((l + zcr[i, k] - zcr[i, k + 2]) / (zcr[i, k - 1] - zcr[i, k + 2]))
    )
    p0 = lambda l: (
        ((l + zcr[i, k] - zcr[i, k - 1]) / (zcr[i, k] - zcr[i, k - 1]))
        * ((l + zcr[i, k] - zcr[i, k + 1]) / (zcr[i, k] - zcr[i, k + 1]))
        * ((l + zcr[i, k] - zcr[i, k + 2]) / (zcr[i, k] - zcr[i, k + 2]))
    )
    p1 = lambda l: (
        ((l + zcr[i, k] - zcr[i, k - 1]) / (zcr[i, k + 1] - zcr[i, k - 1]))
        * ((l + zcr[i, k] - zcr[i, k]) / (zcr[i, k + 1] - zcr[i, k]))
        * ((l + zcr[i, k] - zcr[i, k + 2]) / (zcr[i, k + 1] - zcr[i, k + 2]))
    )
    p2 = lambda l: (
        ((l + zcr[i, k] - zcr[i, k - 1]) / (zcr[i, k + 2] - zcr[i, k - 1]))
        * ((l + zcr[i, k] - zcr[i, k]) / (zcr[i, k + 2] - zcr[i, k]))
        * ((l + zcr[i, k] - zcr[i, k + 1]) / (zcr[i, k + 2] - zcr[i, k + 1]))
    )

    kd_m1 = np.where(k - 1 < nx, k - 1, nx)
    kd_m1 = np.where(kd_m1 >= 0, kd_m1, 0)

    kd_0 = np.where(k < nx, k, nx)
    kd_0 = np.where(kd_0 >= 0, kd_0, 0)

    kd_p1 = np.where(k + 1 < nx, k + 1, nx)
    kd_p1 = np.where(kd_p1 >= 0, kd_p1, 0)

    kd_p2 = np.where(k + 2 < nx, k + 2, nx)
    kd_p2 = np.where(kd_p2 >= 0, kd_p2, 0)

    pz_m1 = p_1(lz)
    pz_0 = p0(lz)
    pz_p1 = p1(lz)
    pz_p2 = p2(lz)
    
    p_1x = lambda l: (1 / 6) * l * (l - 1) * (2 - l)
    p0x = lambda l: (1 - l**2) * (1 - l / 2)
    p1x = lambda l: (1 / 2) * l * (l + 1) * (2 - l)
    p2x = lambda l: (1 / 6) * l * (l**2 - 1)
    
    px_m1 = p_1x(lx)
    px_0 = p0x(lx)
    px_p1 = p1x(lx)
    px_p2 = p2x(lx)
    
    # 1. Tableaux d'indices
    # Non periodique
    id_m1 = boundaries(i - 1, nx, bcx_kind)
    id_0 = boundaries(i, nx, bcx_kind)
    id_p1 = boundaries(i + 1, nx, bcx_kind)
    id_p2 = boundaries(i + 2, nx, bcx_kind)


    # Lookup
    psi_d_i = np.zeros((16, nx, nz))
    for i in range(nx):
        for j in range(nz):
            psi_d_i[0, i, j] = psi[id_m1[i, j], kd_m1[i, j]]
            psi_d_i[1, i, j] = psi[id_0[i, j], kd_m1[i, j]]
            psi_d_i[2, i, j] = psi[id_p1[i, j], kd_m1[i, j]]
            psi_d_i[3, i, j] = psi[id_p2[i, j], kd_m1[i, j]]

            psi_d_i[4, i, j] = psi[id_m1[i, j], kd_0[i, j]]
            psi_d_i[5, i, j] = psi[id_0[i, j], kd_0[i, j]]
            psi_d_i[6, i, j] = psi[id_p1[i, j], kd_0[i, j]]
            psi_d_i[7, i, j] = psi[id_p2[i, j], kd_0[i, j]]

            psi_d_i[8, i, j] = psi[id_m1[i, j], kd_p1[i, j]]
            psi_d_i[9, i, j] = psi[id_0[i, j], kd_p1[i, j]]
            psi_d_i[10, i, j] = psi[id_p1[i, j], kd_p1[i, j]]
            psi_d_i[11, i, j] = psi[id_p2[i, j], kd_p1[i, j]]

            psi_d_i[12, i, j] = psi[id_m1[i, j], kd_p2[i, j]]
            psi_d_i[13, i, j] = psi[id_0[i, j], kd_p2[i, j]]
            psi_d_i[14, i, j] = psi[id_p1[i, j], kd_p2[i, j]]
            psi_d_i[15, i, j] = psi[id_p2[i, j], kd_p2[i, j]]

    psi_d_j = np.zeros((4, nx, nz))
    psi_d_j[0] = (
        pz_m1 * psi_d_i[0]
        + pz_0 * psi_d_i[1]
        + pz_p1 * psi_d_i[2]
        + pz_p2 * psi_d_i[3]
    )
    psi_d_j[1] = (
        pz_m1 * psi_d_i[4]
        + pz_0 * psi_d_i[5]
        + pz_p1 * psi_d_i[6]
        + pz_p2 * psi_d_i[7]
    )
    psi_d_j[2] = (
        pz_m1 * psi_d_i[8]
        + pz_0 * psi_d_i[9]
        + pz_p1 * psi_d_i[10]
        + pz_p2 * psi_d_i[11]
    )
    psi_d_j[3] = (
        pz_m1 * psi_d_i[12]
        + pz_0 * psi_d_i[13]
        + pz_p1 * psi_d_i[14]
        + pz_p2 * psi_d_i[15]
    )

    psi_d = (
        px_m1 * psi_d_j[0]
        + px_0 * psi_d_j[1]
        + px_p1 * psi_d_j[2]
        + px_p2 * psi_d_j[3]
    )

    return psi_d


def max_interpolator_2d(
    psi: np.ndarray,
    i_d: np.ndarray,
    j_d: np.ndarray,
    bcx_kind: int,
    bcy_kind: int,
    nx: int,
    ny: int,
):
    
    id_0 = boundaries(i_d, nx, bcx_kind)
    id_p1 = boundaries(i_d + 1, nx, bcx_kind)
    
    jd_0 = boundaries(j_d, ny, bcy_kind)
    jd_p1 = boundaries(j_d + 1, ny, bcy_kind)
    
    psi_d = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            psi_d[i, j] = max(
                psi[id_0[i, j], jd_0[i, j]],
                psi[id_0[i, j], jd_p1[i, j]],
                psi[id_p1[i, j], jd_0[i, j]],
                psi[id_p1[i, j], jd_p1[i, j]],
            )
    
    return psi_d

def min_interpolator_2d(
    psi: np.ndarray,
    i_d: np.ndarray,
    j_d: np.ndarray,
    bcx_kind: int,
    bcy_kind: int,
    nx: int,
    ny: int,
):
    
    id_0 = boundaries(i_d, nx, bcx_kind)
    id_p1 = boundaries(i_d + 1, nx, bcx_kind)
    
    jd_0 = boundaries(j_d, ny, bcy_kind)
    jd_p1 = boundaries(j_d + 1, ny, bcy_kind)
    
    psi_d = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            psi_d[i, j] = min(
                psi[id_0[i, j], jd_0[i, j]],
                psi[id_0[i, j], jd_p1[i, j]],
                psi[id_p1[i, j], jd_0[i, j]],
                psi[id_p1[i, j], jd_p1[i, j]],
            )
    
    return psi_d