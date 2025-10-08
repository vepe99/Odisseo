from beartype.typing import Optional, Tuple, Callable, Union, List, NamedTuple
from functools import partial
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

import jax
import jax.numpy as jnp
from jax import vmap, jit, pmap
from jax import random
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

import equinox as eqx

from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.option_classes import DIRECT_ACC, DIRECT_ACC_LAXMAP, DIRECT_ACC_MATRIX, DIRECT_ACC_FOR_LOOP, DIRECT_ACC_SHARDING, NO_SELF_GRAVITY



@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def single_body_acc(particle_i: jnp.ndarray, 
                    particle_j: jnp.ndarray, 
                    mass_i: jnp.ndarray, 
                    mass_j: jnp.ndarray, 
                    config: SimulationConfig, 
                    params: SimulationParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute acceleration of particle_i due to particle_j.
    
    Args:
        particle_i: Position and velocity of particle_i.
        particle_j: Position and velocity of particle_j.
        mass_i: Mass of particle_i.
        mass_j: Mass of particle_j.
        config: Configuration parameters.
        params: Simulation parameters.
    
    Returns:
        The acceleration of particle_i due to particle_j, and the potential felt particle_i due to particle_j.
    """

    r_ij = jax.lax.stop_gradient(particle_i[0, :] - particle_j[0, :])
    condtion = jnp.all(r_ij == 0.0)

    def same_position():
        return jnp.zeros(3), 0.0
    def different_position():
        r_mag = jnp.linalg.norm(r_ij)
        acc = - params.G * mass_j * (r_ij/(r_mag**2 + config.softening**2)**(3/2))
        pot = - params.G * mass_j / (r_mag**2 + config.softening**2)**(1/2)
        return acc, pot
    return jax.lax.cond(condtion, same_position, different_position)
    
    
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential'])
def direct_acc(state: jnp.ndarray, 
               mass: jnp.ndarray, 
               config: SimulationConfig, 
               params: SimulationParams, 
               return_potential=False):
    """
    Compute acceleration of all particles due to all other particles by vmap of the single_body_acc function.

    Args:
        state: Array of shape (N, 2, 3) containing the positions and velocities of the particles.
        mass: Array of shape (N,) containing the masses of the particles.
        config: Configuration object containing the number of particles (N_particles) and softening parameter.
        params: Parameters object containing the gravitational constant (G).
        return_potential: If True, also return the potential energy. Defaults to False.

    Returns:
        Array of shape (N, 3) containing the accelerations of the particles.
        Array of shape (N,) containing the potential energy of the particles, if return_potential is True.
    
    """

    def net_force_on_body(particle_i, mass_i):
        
        acc, potential = vmap(lambda particle_j, mass_j: single_body_acc(particle_i, particle_j, mass_i, mass_j, config, params))(state, mass)
        if return_potential:
            return jnp.sum(acc, axis=0), jnp.sum(potential, )
        else:
            return jnp.sum(acc, axis=0)

    return vmap(net_force_on_body)(state, mass)


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential'])
def direct_acc_laxmap(state: jnp.ndarray,
                       mass: jnp.ndarray,
                       config: SimulationConfig,
                       params: SimulationParams,
                       return_potential=False):
    """
    Compute acceleration of all particles due to all other particles by using lax.map of the single_body_acc function.
    If config.double_map is True, lax.map uses lax.map for both loops, otherwise the inner loop is vectorized using vmap.
    Memory usage is reduced by using lax.map instead of vmap thanks to batching.

    Args:
        state: Array of shape (N, 2, 3) containing the positions and velocities of the particles.
        mass: Array of shape (N,) containing the masses of the particles.
        config: Configuration object containing the number of particles (N_particles) and softening parameter.
        params: Parameters object containing the gravitational constant (G).
        return_potential: If True, also return the potential energy. Defaults to False.

    Returns:
        Array of shape (N, 3) containing the accelerations of the particles.
        Array of shape (N,) containing the potential energy of the particles, if return_potential is True.
    """

    def net_force_on_body(state_and_mass):
        particle_i, mass_i = state_and_mass

        if config.double_map:
            @partial(jax.jit)
            def single_body_acc_lax(state_and_mass_j):
                particle_j, mass_j = state_and_mass_j
                return single_body_acc(particle_i, particle_j, mass_i, mass_j, config, params)
            acc, potential = jax.lax.map(single_body_acc_lax, (state, mass), batch_size=config.batch_size)
        else:
            acc, potential = vmap(lambda particle_j, mass_j: single_body_acc(particle_i, particle_j, mass_i, mass_j, config, params))(state, mass)

        if return_potential:
            return jnp.sum(acc, axis=0), jnp.sum(potential, )
        else:
            return jnp.sum(acc, axis=0)

    return jax.lax.map(net_force_on_body, (state, mass), batch_size=config.batch_size)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit(donate='all')
def direct_acc_matrix(state: jnp.ndarray, 
                      mass: jnp.ndarray, 
                      config: SimulationConfig, 
                      params: SimulationParams, 
                     return_potential: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Compute the direct acceleration matrix for a system of particles. Uses matrix operations.

    Args:
        state: Array of shape (N, 2, 3) containing the positions and velocities of the particles.
        mass: Array of shape (N,) containing the masses of the particles.
        config: Configuration object containing the number of particles (N_particles) and softening parameter.
        params: Parameters object containing the gravitational constant (G).
        return_potential: If True, also return the potential energy. Defaults to False.

    Returns:
        Array of shape (N, 3) containing the accelerations of the particles.
        Array of shape (N,) containing the potential energy of the particles, if return_potential is True.
    """
    pos = state[:, 0, :]  # Extract positions (N, 3)

    # Compute pairwise differences
    dpos = jax.lax.stop_gradient(pos[:, None, :] - pos[None, :, :])  # Shape: (N, N, 3)

    eye = jax.lax.stop_gradient(jnp.eye(config.N_particles))

    # Compute squared distances with softening plus avoid self interaction
    r2_safe = jnp.sum(dpos**2, axis=-1) + config.softening**2  # Shape: (N, N)

    # Compute 1/r^3 safely
    inv_r3 = r2_safe**-1.5 * (1.0 - eye)  # Diagonal is zero

    # Compute acceleration
    acc = - params.G * jnp.sum((mass[:, None] * dpos) * inv_r3[:, :, None], axis=1)

    if return_potential:
        # Compute potential energy (only sum interactions once)
        inv_r = r2_safe**-0.5 * (1.0 - eye)  # Diagonal is zero
        pot = -params.G * jnp.sum(mass[:, None] * inv_r, axis=1)
        return acc, pot
    else:
        return acc
    
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential'])
def direct_acc_for_loop(state: jnp.ndarray, 
                      mass: jnp.ndarray, 
                      config: SimulationConfig, 
                      params: SimulationParams, 
                     return_potential: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Compute the direct acceleration matrix for a system of particles. Uses a double for loop and Newton's third low to reduce the 
    computation from O(N^2) to O(N^2 /2).

    Args:
        state: Array of shape (N, 2, 3) containing the positions and velocities of the particles.
        mass: Array of shape (N,) containing the masses of the particles.
        config: Configuration object containing the number of particles (N_particles) and softening parameter.
        params: Parameters object containing the gravitational constant (G).
        return_potential: If True, also return the potential energy. Defaults to False.

    Returns:
        Array of shape (N, 3) containing the accelerations of the particles.
        Array of shape (N,) containing the potential energy of the particles, if return_potential is True.
    """

    def compute_acc(carry, pos):
        if return_potential:
            pot = carry
        else:
            acc =  carry
        r = jax.lax.stop_gradient(pos[None, :] - positions)
        r2 = jnp.sum(r**2, axis=1) + config.softening**2
        if return_potential:
            inv_r = jnp.where(r2 == 0., 0., r2**(-1/2))
            pot = jnp.sum(-params.G * mass * inv_r, keepdims=True)
            return pot, pot
        else:
            inv_r3 = jnp.where(r2 == 0., 0., r2**(-3/2))
            acc = jnp.sum(-params.G * mass[:, None] * r * inv_r3[:, None], axis=0)
            return acc, acc        

    positions =  state[:, 0]
    if return_potential:
        initial_pot = jnp.array([0.], )
        _, pot = jax.lax.scan(compute_acc, initial_pot, positions)
        return pot
    else:
        initial_acc = jnp.zeros_like(positions[0],)
        _, acc = jax.lax.scan(compute_acc, initial_acc, positions)
        return acc

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit(donate='all')
def direct_acc_sharding(state: jnp.ndarray, 
                      mass: jnp.ndarray, 
                      config: SimulationConfig, 
                      params: SimulationParams, 
                     return_potential: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Compute the direct acceleration matrix for a system of particles. Shard the positions to allow for parallel computation.
    CURRENTLY NOT WORKING.


    Args:
        state: Array of shape (N, 2, 3) containing the positions and velocities of the particles.
        mass: Array of shape (N,) containing the masses of the particles.
        config: Configuration object containing the number of particles (N_particles) and softening parameter.
        params: Parameters object containing the gravitational constant (G).
        return_potential: If True, also return the potential energy. Defaults to False.

    Returns:
        Array of shape (N, 3) containing the accelerations of the particles.
        Array of shape (N,) containing the potential energy of the particles, if return_potential is True.
    """
    
    pos = state[:, 0]
    # Create a mesh from all devices
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('N_particles',))
    # Define sharding strategy - shard along axis 0
    sharding = NamedSharding(mesh, P('N_particles', None))
    in_specs = P('N_particles', None)
    out_specs = P('N_particles', None)
    @jit
    def put_on_device(positions):
        positions_sharded = jax.device_put(positions, sharding)
        return positions_sharded
    pos_sharded = put_on_device(pos.copy())
    @jit 
    def pairwise_diff(pos):
        return pos[0, None] - pos_sharded
    @jit
    def lax_map_pairwise_diff(pos):
        return jax.lax.map(pairwise_diff, pos, batch_size=config.batch_size)
    dpos = jax.lax.stop_gradient(shard_map(lax_map_pairwise_diff, 
                                           mesh=mesh, 
                                           in_specs=in_specs, 
                                           out_specs=out_specs)(pos))
    eye = jax.lax.stop_gradient(jnp.eye(config.N_particles))
    r2_safe = jnp.sum(dpos**2, axis=-1) + config.softening**2 + eye # Shape: (N, N)
    if return_potential:
        inv_r = r2_safe**-0.5 * (1.0 - eye)
        return jax.device_put(jnp.sum(-params.G * jnp.sum(mass[:, None] * inv_r, axis=1), axis=0), devices[0])
    else:
        inv_r3 = r2_safe**-1.5 * (1.0 - eye)
        return jax.device_put(jnp.sum(-params.G * jnp.sum((mass[:, None] * dpos) * inv_r3[:, :, None], axis=1), axis=0), devices[0])


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config', 'return_potential'])
def no_self_gravity(state: jnp.ndarray, 
                    mass: jnp.ndarray, 
                    config: SimulationConfig, 
                    params: SimulationParams, 
                    return_potential=False):
    """
    Remove the self interaction between particles.

    Args:
        state: Array of shape (N, 2, 3) containing the positions and velocities of the particles.
        mass: Array of shape (N,) containing the masses of the particles.
        config: Configuration object containing the number of particles (N_particles) and softening parameter.
        params: Parameters object containing the gravitational constant (G).
        return_potential: If True, also return the potential energy. Defaults to False.

    Returns:
        Array of shape (N, 3) containing the accelerations of the particles.
        Array of shape (N,) containing the potential energy of the particles, if return_potential is True.
    
    """
        
    if return_potential:
        return jnp.zeros((config.N_particles, 3)), jnp.zeros((config.N_particles,))
    else:
        return jnp.zeros((config.N_particles, 3))






