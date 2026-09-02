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
from jax import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import equinox as eqx

from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.option_classes import DIRECT_ACC, DIRECT_ACC_LAXMAP, DIRECT_ACC_MATRIX, DIRECT_ACC_FOR_LOOP, DIRECT_ACC_SHARDING, NO_SELF_GRAVITY



def _resolve_batch_size(raw_batch_size) -> int:
    """Normalize lax.map batch_size to a positive Python int."""
    if isinstance(raw_batch_size, tuple):
        if len(raw_batch_size) != 1:
            raise ValueError(f"batch_size tuple must have length 1, got {raw_batch_size}")
        raw_batch_size = raw_batch_size[0]
    if isinstance(raw_batch_size, list):
        if len(raw_batch_size) != 1:
            raise ValueError(f"batch_size list must have length 1, got {raw_batch_size}")
        raw_batch_size = raw_batch_size[0]

    batch_size = int(raw_batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return batch_size


@partial(jax.jit, static_argnames=['config'])
@jaxtyped(typechecker=typechecker)
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
    dtype = r_ij.dtype
    g_const = jnp.asarray(params.G, dtype=dtype)
    softening_sq = jnp.asarray(config.softening, dtype=dtype) ** 2

    def same_position():
        return jnp.zeros((3,), dtype=dtype), jnp.asarray(0.0, dtype=dtype)

    def different_position():
        r_mag = jnp.linalg.norm(r_ij)
        denom = (r_mag**2 + softening_sq) ** (jnp.asarray(1.5, dtype=dtype))
        acc = -g_const * mass_j * (r_ij / denom)
        pot = -g_const * mass_j / jnp.sqrt(r_mag**2 + softening_sq)
        return acc, pot

    return jax.lax.cond(condtion, same_position, different_position)
    
    
@partial(jax.jit, static_argnames=['config', 'return_potential'])
@jaxtyped(typechecker=typechecker)
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


@partial(jax.jit, static_argnames=['config', 'return_potential'])
@jaxtyped(typechecker=typechecker)
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

    batch_size = _resolve_batch_size(config.batch_size)

    def net_force_on_body(state_and_mass):
        particle_i, mass_i = state_and_mass

        if config.double_map:
            @partial(jax.jit)
            def single_body_acc_lax(state_and_mass_j):
                particle_j, mass_j = state_and_mass_j
                return single_body_acc(particle_i, particle_j, mass_i, mass_j, config, params)
            acc, potential = jax.lax.map(single_body_acc_lax, (state, mass), batch_size=batch_size)
        else:
            acc, potential = vmap(lambda particle_j, mass_j: single_body_acc(particle_i, particle_j, mass_i, mass_j, config, params))(state, mass)

        if return_potential:
            return jnp.sum(acc, axis=0), jnp.sum(potential, )
        else:
            return jnp.sum(acc, axis=0)

    return jax.lax.map(net_force_on_body, (state, mass), batch_size=batch_size)


@eqx.filter_jit(donate='all')
@jaxtyped(typechecker=typechecker)
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
        # mass must be indexed by j (the summed axis), as in the acceleration above.
        pot = -params.G * jnp.sum(mass[None, :] * inv_r, axis=1)
        return acc, pot
    else:
        return acc
    
@partial(jax.jit, static_argnames=['config', 'return_potential'])
@jaxtyped(typechecker=typechecker)
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

@eqx.filter_jit(donate='all')
@jaxtyped(typechecker=typechecker)
def direct_acc_sharding(state: jnp.ndarray, 
                      mass: jnp.ndarray, 
                      config: SimulationConfig, 
                      params: SimulationParams, 
                     return_potential: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Compute the direct acceleration matrix for a system of particles. Shard the positions to allow for parallel computation.

    The rows of the pairwise separation matrix are distributed over ``jax.devices()``
    while the columns are replicated on every shard. ``N_particles`` need not be a
    multiple of the device count: it is padded with zero-mass particles internally.


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
    
    batch_size = _resolve_batch_size(config.batch_size)
    pos = state[:, 0]
    N = config.N_particles

    # Create a mesh from all devices
    devices = jax.devices()
    n_devices = len(devices)
    mesh = Mesh(devices, axis_names=('N_particles',))

    # shard_map requires the sharded axis to divide evenly over the mesh, so pad
    # up to a multiple of the device count with zero-mass particles. They exert no
    # force (mass 0) and their rows are dropped from the result below.
    N_padded = -(-N // n_devices) * n_devices
    n_pad = N_padded - N
    if n_pad:
        pos = jnp.concatenate([pos, jnp.zeros((n_pad, 3), dtype=pos.dtype)], axis=0)
        mass = jnp.concatenate([mass, jnp.zeros((n_pad,), dtype=mass.dtype)], axis=0)

    # Positions are needed in two layouts: sharded over the mesh to split the rows
    # of the pairwise matrix, and replicated so every shard sees all the columns.
    pos_sharded = jax.device_put(pos, NamedSharding(mesh, P('N_particles', None)))
    pos_replicated = jax.device_put(pos, NamedSharding(mesh, P(None, None)))

    def pairwise_diff_shard(pos_local, pos_all):
        # pos_local: (N_padded // n_devices, 3) rows owned by this shard.
        # pos_all:   (N_padded, 3) every particle, replicated on this shard.
        def pairwise_diff(particle_i):
            return particle_i[None, :] - pos_all  # (N_padded, 3)
        return jax.lax.map(pairwise_diff, pos_local, batch_size=batch_size)

    dpos = jax.lax.stop_gradient(shard_map(pairwise_diff_shard,
                                           mesh=mesh,
                                           in_specs=(P('N_particles', None), P(None, None)),
                                           out_specs=P('N_particles', None))(pos_sharded,
                                                                             pos_replicated))
    eye = jax.lax.stop_gradient(jnp.eye(N_padded))
    r2_safe = jnp.sum(dpos**2, axis=-1) + config.softening**2 + eye # Shape: (N_padded, N_padded)
    inv_r3 = r2_safe**-1.5 * (1.0 - eye)  # Diagonal is zero
    acc = - params.G * jnp.sum((mass[:, None] * dpos) * inv_r3[:, :, None], axis=1)
    acc = jax.device_put(acc[:N], devices[0])
    if return_potential:
        inv_r = r2_safe**-0.5 * (1.0 - eye)  # Diagonal is zero
        # mass must be indexed by j (the summed axis), as in the acceleration above;
        # this is also what zeroes out the zero-mass padding particles.
        pot = - params.G * jnp.sum(mass[None, :] * inv_r, axis=1)
        return acc, jax.device_put(pot[:N], devices[0])
    else:
        return acc


@partial(jax.jit, static_argnames=['config', 'return_potential'])
@jaxtyped(typechecker=typechecker)
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
