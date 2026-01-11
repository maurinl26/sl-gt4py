import jax.numpy as jnp
import jax
from jax import lax


def periodic_overshoot_filter(
    tracer_e: jnp.ndarray,
    tracer_sup: jnp.ndarray,
    nx: int,
    ny: int
):
    """Filter overshoots on interpolated field with periodic boundaries using JAX.
    Optimized with lax.fori_loop for sequential cell updates.

    Args:
        tracer_e (jnp.ndarray): Grid point tracer at t + dt
        tracer_sup (jnp.ndarray): Maximum tracer values
        nx (int): grid size in x
        ny (int): grid size in y

    Returns:
        jnp.ndarray: filtered tracer field
    """
    
    overshoot = jnp.maximum(tracer_e - tracer_sup, 0)
    
    def process_cell(idx, tracer):
        """Process a single cell for overshoot filtering."""
        i = idx // ny
        j = idx % ny
        
        # Calculate available slots in 8 neighbors
        slot_1 = jnp.maximum(tracer_sup[i % nx, j % ny] - tracer[(i - 1) % nx, (j - 1) % ny], 0)
        slot_2 = jnp.maximum(tracer_sup[i % nx, j % ny] - tracer[i % nx, (j - 1) % ny], 0)
        slot_3 = jnp.maximum(tracer_sup[i % nx, j % ny] - tracer[(i + 1) % nx, (j - 1) % ny], 0)
        slot_4 = jnp.maximum(tracer_sup[i % nx, j % ny] - tracer[(i + 1) % nx, j % ny], 0)
        slot_5 = jnp.maximum(tracer_sup[i % nx, j % ny] - tracer[(i + 1) % nx, (j + 1) % ny], 0)
        slot_6 = jnp.maximum(tracer_sup[i % nx, j % ny] - tracer[i % nx, (j + 1) % ny], 0)
        slot_7 = jnp.maximum(tracer_sup[i % nx, j % ny] - tracer[(i - 1) % nx, (j + 1) % ny], 0)
        slot_8 = jnp.maximum(tracer_sup[i % nx, j % ny] - tracer[(i - 1) % nx, j % ny], 0)
        
        total_disp = slot_1 + slot_2 + slot_3 + slot_4 + slot_5 + slot_6 + slot_7 + slot_8
        
        # Apply redistribution if total available > overshoot
        overshoot_val = overshoot[i, j]
        apply_filter = total_disp > overshoot_val
        
        # Update tracer array
        tracer = tracer.at[i, j].add(-overshoot_val * apply_filter)
        tracer = tracer.at[(i - 1) % nx, (j - 1) % ny].add((slot_1 / (total_disp + 1e-10)) * overshoot_val * apply_filter)
        tracer = tracer.at[i % nx, (j - 1) % ny].add((slot_2 / (total_disp + 1e-10)) * overshoot_val * apply_filter)
        tracer = tracer.at[(i + 1) % nx, (j - 1) % ny].add((slot_3 / (total_disp + 1e-10)) * overshoot_val * apply_filter)
        tracer = tracer.at[(i + 1) % nx, j % ny].add((slot_4 / (total_disp + 1e-10)) * overshoot_val * apply_filter)
        tracer = tracer.at[(i + 1) % nx, (j + 1) % ny].add((slot_5 / (total_disp + 1e-10)) * overshoot_val * apply_filter)
        tracer = tracer.at[i % nx, (j + 1) % ny].add((slot_6 / (total_disp + 1e-10)) * overshoot_val * apply_filter)
        tracer = tracer.at[(i - 1) % nx, (j + 1) % ny].add((slot_7 / (total_disp + 1e-10)) * overshoot_val * apply_filter)
        tracer = tracer.at[(i - 1) % nx, j % ny].add((slot_8 / (total_disp + 1e-10)) * overshoot_val * apply_filter)
        
        return tracer
    
    # Process all cells sequentially using fori_loop
    result = lax.fori_loop(0, nx * ny, process_cell, tracer_e)
    
    return result


def periodic_undershoot_filter(
    tracer_e: jnp.ndarray,
    tracer_inf: jnp.ndarray,
    nx: int,
    ny: int
):
    """Filter undershoots on interpolated field with periodic boundaries using JAX.
    Optimized with lax.fori_loop for sequential cell updates.

    Args:
        tracer_e (jnp.ndarray): Grid point tracer at t + dt
        tracer_inf (jnp.ndarray): Minimum tracer values
        nx (int): grid size in x
        ny (int): grid size in y

    Returns:
        jnp.ndarray: filtered tracer field
    """
    
    undershoot = jnp.maximum(tracer_inf - tracer_e, jnp.finfo(jnp.float64).tiny)
    
    def process_cell(idx, tracer):
        """Process a single cell for undershoot filtering."""
        i = idx // ny
        j = idx % ny
        
        # Calculate available slots in 8 neighbors
        slot_1 = jnp.maximum(tracer[(i - 1) % nx, (j - 1) % ny] - tracer_inf[i, j], 0)
        slot_2 = jnp.maximum(tracer[i % nx, (j - 1) % ny] - tracer_inf[i, j], 0)
        slot_3 = jnp.maximum(tracer[(i + 1) % nx, (j - 1) % ny] - tracer_inf[i, j], 0)
        slot_4 = jnp.maximum(tracer[(i + 1) % nx, j % ny] - tracer_inf[i, j], 0)
        slot_5 = jnp.maximum(tracer[(i + 1) % nx, (j + 1) % ny] - tracer_inf[i, j], 0)
        slot_6 = jnp.maximum(tracer[i % nx, (j + 1) % ny] - tracer_inf[i, j], 0)
        slot_7 = jnp.maximum(tracer[(i - 1) % nx, (j + 1) % ny] - tracer_inf[i, j], 0)
        slot_8 = jnp.maximum(tracer[(i - 1) % nx, j % ny] - tracer_inf[i, j], 0)
        
        total_disp = slot_1 + slot_2 + slot_3 + slot_4 + slot_5 + slot_6 + slot_7 + slot_8
        
        # Apply redistribution if total available > undershoot
        undershoot_val = undershoot[i, j]
        apply_filter = total_disp > undershoot_val
        
        # Update tracer array
        tracer = tracer.at[i, j].add(undershoot_val * apply_filter)
        tracer = tracer.at[(i - 1) % nx, (j - 1) % ny].add(-(slot_1 / (total_disp + 1e-10)) * undershoot_val * apply_filter)
        tracer = tracer.at[i % nx, (j - 1) % ny].add(-(slot_2 / (total_disp + 1e-10)) * undershoot_val * apply_filter)
        tracer = tracer.at[(i + 1) % nx, (j - 1) % ny].add(-(slot_3 / (total_disp + 1e-10)) * undershoot_val * apply_filter)
        tracer = tracer.at[(i + 1) % nx, j % ny].add(-(slot_4 / (total_disp + 1e-10)) * undershoot_val * apply_filter)
        tracer = tracer.at[(i + 1) % nx, (j + 1) % ny].add(-(slot_5 / (total_disp + 1e-10)) * undershoot_val * apply_filter)
        tracer = tracer.at[i % nx, (j + 1) % ny].add(-(slot_6 / (total_disp + 1e-10)) * undershoot_val * apply_filter)
        tracer = tracer.at[(i - 1) % nx, (j + 1) % ny].add(-(slot_7 / (total_disp + 1e-10)) * undershoot_val * apply_filter)
        tracer = tracer.at[(i - 1) % nx, j % ny].add(-(slot_8 / (total_disp + 1e-10)) * undershoot_val * apply_filter)
        
        return tracer
    
    # Process all cells sequentially using fori_loop
    result = lax.fori_loop(0, nx * ny, process_cell, tracer_e)
    
    return result
