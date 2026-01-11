import jax.numpy as jnp


def boundaries(
    indices: jnp.ndarray,
    n: int,
    bc_kind: int
) -> jnp.ndarray:
    """Apply boundary conditions
    1: periodic
    0: fixed

    Args:
        indices (jnp.ndarray): list of indices to shrink
        n (int): size of spatial domain
        bc_kind (int): type of boundaries 

    Returns:
        jnp.ndarray: processed indices
    """
    
    if bc_kind == 0:
        id_m1 = jnp.where(indices < n, indices, n - 1)
        id_m1 = jnp.where(id_m1 >= 0, id_m1, 0)

    else:
        id_m1 = jnp.where(indices < n, indices, indices % n)
        id_m1 = jnp.where(id_m1 >= 0, id_m1, id_m1 % n)

    return id_m1
