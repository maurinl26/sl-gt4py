import dace
import numpy as np

NX = dace.symbol("NX")
NY = dace.symbol("NX")
NZ = dace.symbol("NZ")

@dace.function
def p0_lin(l: dace.float64):
    return 1 - l

@dace.function
def p1_lin(l: dace.float64):
    return l

@dace.function
def boundaries(
    indice: dace.float64,
    n: dace.int32,
    bc_kind: dace.bool
):
    if bc_kind == 0:
        
        if indice >= n:
            indice = n - 1
        elif indice <= 0:
            indice = 0
        else:
            indice  = indice

    else:
        indice = indice % n
        
    return indice

@dace.function
def dace_interpolate_lin_2d(
    psi: dace.float64[NX, NY, NY],
    lx: dace.float64[NX, NY, NY],
    ly: dace.float64[NX, NY, NY],
    i_d: dace.float64[NX, NY, NY],
    j_d: dace.float64[NX, NY, NY],
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
    
    # Lookup
    psi_d_i = np.zeros((4, nx, ny, nz))
    for k in range(NZ):
        for i in range(NX):
            for j in range(NY):
                
                id_0 = boundaries(i_d[i, j, k], nx, bcx_kind)
                id_p1 = boundaries(i_d[i, j, k] + 1, nx, bcx_kind)
    
                jd_0 = boundaries(j_d[i, j, k], ny, bcy_kind)
                jd_p1 = boundaries(j_d[i, j, k] + 1, ny, bcy_kind)
                
                psi_d_i[0, i, j, k] = psi[id_0, jd_0]
                psi_d_i[1, i, j, k] = psi[id_p1, jd_0]

                psi_d_i[2, i, j, k] = psi[id_0, jd_p1]
                psi_d_i[3, i, j, k] = psi[id_p1, jd_p1]

    psi_d_j = np.zeros((2, nx, ny, nz))
    psi_d_j[0] = px0 * psi_d_i[0] + px1 * psi_d_i[1]
    psi_d_j[1] = px0 * psi_d_i[2] + px1 * psi_d_i[3]

    psi_d = py0 * psi_d_j[0] + py1 * psi_d_j[1]

    return psi_d
