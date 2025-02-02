import jax
import jax.numpy as jnp

def radius(state):
    """
    Return the radial position of the particle in the state.
    """
    
    return jnp.linalg.norm(state[:, 0], axis=1)
    

def center_of_mass(state, mass):
    """
    Return the center of mass of the system.
    """
    
    return jnp.sum(state[:, 0] * mass[:, jnp.newaxis], axis=0) / jnp.sum(mass)

