import jax.numpy as jnp


def diagnostic_lipschitz(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float, dth: float):
    """Diagnostic for Semi-Lagrangian Research stability using JAX

    Args:
        u (jnp.ndarray): velocity on x 
        v (jnp.ndarray): velocity on y
        dx (float): spacing on x
        dy (float): spacing on y
        dth (float): half time step

    Returns:
        float: lipschitz condition on stability
    """
    
    dudx = (1/dx) * jnp.gradient(u, axis=0)
    dudy = (1/dy) * jnp.gradient(u, axis=1)
    dvdx = (1/dx) * jnp.gradient(v, axis=0)
    dvdy = (1/dy) * jnp.gradient(v, axis=1)
    
    return dth * jnp.maximum(jnp.maximum(dudx, dudy), jnp.maximum(dvdx, dvdy))


def diagnostic_overshoot(
    tracer_e: jnp.ndarray,
    tracer_sup: jnp.ndarray,
) -> jnp.ndarray:
    """Compute overshoots using JAX

    Args:
        tracer_e (jnp.ndarray): interpolated field
        tracer_sup (jnp.ndarray): sup field

    Returns:
        jnp.ndarray: overshoots
    """
    
    return jnp.maximum(tracer_e - tracer_sup, jnp.finfo(jnp.float64).tiny)


def diagnostic_undershoot(
    tracer_e: jnp.ndarray,
    tracer_inf: jnp.ndarray
) -> jnp.ndarray:
    """Compute undershoots using JAX

    Args:
        tracer_e (jnp.ndarray): interpolated field
        tracer_inf (jnp.ndarray): inf field

    Returns:
        jnp.ndarray: undershoots
    """
    
    return jnp.maximum(tracer_inf - tracer_e, jnp.finfo(jnp.float64).tiny)
