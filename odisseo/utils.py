from typing import Union, NamedTuple
from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from odisseo.dynamics import direct_acc, direct_acc_laxmap, direct_acc_matrix, direct_acc_for_loop, direct_acc_sharding
from odisseo.potentials import combined_external_acceleration, combined_external_acceleration_vmpa_switch
from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.option_classes import DIRECT_ACC, DIRECT_ACC_LAXMAP, DIRECT_ACC_MATRIX, DIRECT_ACC_FOR_LOOP, DIRECT_ACC_SHARDING

from jaxtyping import jaxtyped
from beartype import beartype as typechecker

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, )    
def center_of_mass(state: jnp.ndarray, 
                   mass: jnp.ndarray) -> jnp.ndarray:
    """
    Return the center of mass of the system.

    Args:
        state (jnp.ndarray): Array of shape (N_particles, 6) representing the positions and velocities of the particles.
        mass (jnp.ndarray): Array of masses for each particle.
    Returns:
        jnp.ndarray: The center of mass position

    """
    
    return jnp.sum(state[:, 0] * mass[:, jnp.newaxis], axis=0) / jnp.sum(mass)

@jaxtyped(typechecker=typechecker)
@jit
def E_kin(state: jnp.ndarray, 
          mass: jnp.ndarray) -> jnp.ndarray:
    """
    Return the kinetic energy of the system.

   Args:
        state (jnp.ndarray): Array of shape (N_particles, 6) representing the positions and velocities of the particles.
        mass (jnp.ndarray): Array of masses for each particle.
    Returns:
        jnp.ndarray: Kinetic energy of the particles in the system
    """

    return 0.5 * (jnp.sum(state[:, 1]**2, axis=1) * mass)
    

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def E_pot(state: jnp.ndarray,
        mass: jnp.ndarray,
        config: SimulationConfig,
        params: SimulationParams, ) -> jnp.ndarray:
    """
    Return the potential energy of the system.

    Args:
        state (jnp.ndarray): Array of shape (N_particles, 6) representing the positions and velocities of the particles.
        mass (jnp.ndarray): Array of shape (N_particles,) representing the masses of the particles.
        config (SimulationConfig): Configuration object containing simulation parameters.
        params (SimulationParams): Parameters object containing physical parameters for the simulation.
    
    Returns:
        E_tot: The potential energy of each particle in the system.

    """
    
    if config.acceleration_scheme == DIRECT_ACC:
        _, pot = direct_acc(state, mass, config, params, return_potential=True)
    elif config.acceleration_scheme == DIRECT_ACC_LAXMAP:
        _, pot = direct_acc_laxmap(state, mass, config, params, return_potential=True)
    elif config.acceleration_scheme == DIRECT_ACC_MATRIX:
        _, pot = direct_acc_matrix(state, mass, config, params, return_potential=True)
    elif config.acceleration_scheme == DIRECT_ACC_FOR_LOOP:
        pot = direct_acc_for_loop(state, mass, config, params, return_potential=True)
    elif config.acceleration_scheme == DIRECT_ACC_SHARDING:
        pot = direct_acc_sharding(state, mass, config, params, return_potential=True)
    
    self_Epot = pot*mass

    external_Epot = 0.
    if len(config.external_accelerations) > 0:
        _, external_pot = combined_external_acceleration_vmpa_switch(state, config, params, return_potential=True)
        external_Epot = external_pot*mass
        
    return self_Epot + external_Epot

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def E_tot(state: jnp.ndarray,
        mass: jnp.ndarray,
        config: SimulationConfig,
        params: SimulationParams, ) -> jnp.ndarray:
    """
    Return the total energy of the system.

    Args:
        state (jnp.ndarray): Array of shape (N_particles,2, 3) representing the positions and velocities of the particles.
        mass (jnp.ndarray): Array of shape (N_particles,) representing the masses of the particles.
        config (SimulationConfig): Configuration object containing simulation parameters.
        params (SimulationParams): Parameters object containing physical parameters for the simulation.    

    Returns:
        float: The total energy of each particle in the system

    """
    
    return E_kin(state, mass) + E_pot(state, mass, config, params)

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, )
def Angular_momentum(state: jnp.ndarray, 
                     mass: jnp.ndarray) -> jnp.ndarray:
    """
    Return the angular momentum of the system.

    Args:
        state (jnp.ndarray): Array of shape (N_particles, 6) representing the positions and velocities of the particles.
        mass (jnp.ndarray): Array of shape (N_particles,) representing the masses of the particles.
    Returns:
        jnp.ndarray: The angular momentum of each particle in the system

    """
    
    return jnp.cross(state[:, 0], state[:, 1]) * mass[:, jnp.newaxis]

