from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from jdgsim.dynamics import DIRECT_ACC, direct_acc
from jdgsim.potentials import combined_external_acceleration

    
@jit
def center_of_mass(state, mass):
    """
    Return the center of mass of the system.

    Parameters
    ----------
    state : jnp.ndarray
        Array of shape (N_particles, 6) representing the positions and velocities of the particles.
    mass : jnp.ndarray

    Returns
    -------
    jnp.ndarray
        The center of mass of the system.
    """
    
    return jnp.sum(state[:, 0] * mass[:, jnp.newaxis], axis=0) / jnp.sum(mass)

@jit
def E_kin(state, mass):
    """
    Return the kinetic energy of the system.

    Parameters
    ----------
    state : jnp.ndarray
        Array of shape (N_particles, 6) representing the positions and velocities of the particles.
    mass : jnp.ndarray
        Array of shape (N_particles,) representing the masses of the particles.

    Returns
    -------
    float
        The kinetic energy of the system.

    """
    
    return 0.5 * jnp.sum(jnp.sum(state[:, 1]**2, axis=1) * mass)

@partial(jax.jit, static_argnames=['config'])
def E_pot(state, mass, config, params):
    """
    Return the potential energy of the system.

    Parameters
    ----------
    state : jnp.ndarray
        Array of shape (N_particles, 6) representing the positions and velocities of the particles.
    mass : jnp.ndarray
        Array of shape (N_particles,) representing the masses of the particles.

    Returns
    -------
    float
        The potential energy of the system.
    """
    
    # return - jnp.sum(jnp.sum(config.acceleration_scheme(state, mass, config, params) * state[:, 0], axis=1) * mass)
    if config.acceleration_scheme == DIRECT_ACC:
        _, pot = direct_acc(state, mass, config, params, return_potential=True)
        self_Epot = jnp.sum(pot*mass)
    
    external_Epot = 0.
    if len(config.external_accelerations) > 0:
        _, pot = combined_external_acceleration(state, config, params, return_potential=True)
        external_Epot = jnp.sum(pot*mass)
        
    return self_Epot + external_Epot

@partial(jax.jit, static_argnames=['config'])
def E_tot(state, mass, config, params):
    """
    Return the total energy of the system.

    Parameters
    ----------
    state : jnp.ndarray
        Array of shape (N_particles, 6) representing the positions and velocities of the particles.
    mass : jnp.ndarray
        Array of shape (N_particles,) representing the masses of the particles.

    Returns
    -------
    float
        The total energy of the system.
    """
    
    return E_kin(state, mass) + E_pot(state, mass, config, params)

def Angular_momentum(state, mass):
    """
    Return the angular momentum of the system.

    Parameters
    ----------
    state : jnp.ndarray
        Array of shape (N_particles, 6) representing the positions and velocities of the particles.
    mass : jnp.ndarray
        Array of shape (N_particles,) representing the masses of the particles.

    Returns
    -------
    jnp.ndarray
        The angular momentum of the system.
    """
    
    return jnp.sum(jnp.cross(state[:, 0], state[:, 1]) * mass[:, jnp.newaxis], axis=0)

