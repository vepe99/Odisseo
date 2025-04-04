from typing import Optional, Tuple, Callable, Union, List, NamedTuple
from functools import partial
from jaxtyping import jaxtyped, Array, Float, Scalar
from beartype import beartype as typechecker

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax import random
from odisseo.potentials import combined_external_acceleration, combined_external_acceleration_vmpa_switch
from odisseo.dynamics import direct_acc, direct_acc_laxmap, direct_acc_matrix, direct_acc_for_loop, direct_acc_sharding

from odisseo.option_classes import DIRECT_ACC, DIRECT_ACC_LAXMAP, DIRECT_ACC_MATRIX, DIRECT_ACC_FOR_LOOP, DIRECT_ACC_SHARDING
from odisseo.option_classes import SimulationConfig, SimulationParams


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def leapfrog(state: jnp.ndarray,
             mass: jnp.ndarray,
             dt: Scalar,
             config: SimulationConfig,
             params: SimulationParams):
    """
    Simple implementation of a symplectic Leapfrog (Verlet) integrator for N-body simulations.

    Args:
        state (jax.numpy.ndarray): The state of the particles, where the first column represents positions and the second column represents velocities.
        mass (jax.numpy.ndarray): The mass of the particles.
        dt (float): Time-step for current integration.
        config (object): Configuration object containing the acceleration scheme and external accelerations.
        params (dict): Additional parameters for the acceleration functions.
    Returns:
        jax.numpy.ndarray: The updated state of the particles.
    """
    if config.acceleration_scheme == DIRECT_ACC:
        acc_func = direct_acc
    
    elif config.acceleration_scheme == DIRECT_ACC_LAXMAP:
        acc_func = direct_acc_laxmap

    elif config.acceleration_scheme == DIRECT_ACC_MATRIX:
        acc_func = direct_acc_matrix
    
    elif config.acceleration_scheme == DIRECT_ACC_FOR_LOOP:
        acc_func = direct_acc_for_loop
    
    elif config.acceleration_scheme == DIRECT_ACC_SHARDING:
        acc_func = direct_acc_sharding

    add_external_acceleration = len(config.external_accelerations) > 0
    
    acc = acc_func(state, mass, config, params)

    # Check additional accelerations
    if add_external_acceleration:
        acc = acc + combined_external_acceleration_vmpa_switch(state, config, params)
            
    # removing half-step velocity
    state = state.at[:, 0].set(state[:, 0] + state[:, 1]*dt + 0.5*acc*(dt**2))

    acc2 = acc_func(state, mass, config, params)

    if add_external_acceleration:
        acc2 = acc2 + combined_external_acceleration_vmpa_switch(state, config, params)
         
    state = state.at[:, 1].set(state[:, 1] + 0.5*(acc + acc2)*dt)
    
    return state

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def RungeKutta4(state: jnp.ndarray,
             mass: jnp.ndarray,
             dt: Scalar,
             config: SimulationConfig,
             params: SimulationParams):
    """
    Simple implementation of a 4th order Runge-Kutta integrator for N-body simulations.

    Args:
        state (jax.numpy.ndarray): The state of the particles, where the first column represents positions and the second column represents velocities.
        mass (jax.numpy.ndarray): The mass of the particles.
        dt (float): Time-step for current integration.
        config (object): Configuration object containing the acceleration scheme and external accelerations.
        params (dict): Additional parameters for the acceleration functions.
    Returns:
        jax.numpy.ndarray: The updated state of the particles.
    """
    if config.acceleration_scheme == DIRECT_ACC:
        acc_func = direct_acc
    
    elif config.acceleration_scheme == DIRECT_ACC_LAXMAP:
        acc_func = direct_acc_laxmap

    elif config.acceleration_scheme == DIRECT_ACC_MATRIX:
        acc_func = direct_acc_matrix

    add_external_acceleration = len(config.external_accelerations) > 0

    k1r = state[:, 1] * dt
    k1v = acc_func(state, mass, config, params) * dt

    state_2 = state.copy()
    state_2 = state_2.at[:, 0].set(state[:, 0] + 0.5*k1r)
    acc2 = acc_func(state_2, mass, config, params)
    if add_external_acceleration:
        acc2 = acc2 + combined_external_acceleration_vmpa_switch(state, config, params)

    k2r = (state[:, 1] + 0.5*k1v) * dt
    k2v = acc2 * dt

    state_3 = state.copy()
    state_3 = state_3.at[:, 0].set(state[:, 0] + 0.5*k2r)
    acc3 = acc_func(state_3, mass, config, params)
    if add_external_acceleration:
        acc3 = acc3 + combined_external_acceleration_vmpa_switch(state, config, params)

    k3r = (state[:, 1] + 0.5*k2v) * dt
    k3v = acc3 * dt

    state_4 = state.copy()
    state_4 = state_4.at[:, 0].set(state[:, 0] + k3r)
    acc4 = acc_func(state_4, mass, config, params)
    if add_external_acceleration:
        acc4 = acc4 + combined_external_acceleration_vmpa_switch(state, config, params)

    k4r = (state[:, 1] + k3v) * dt
    k4v = acc4 * dt

    state = state.at[:, 0].set(state[:, 0] + (k1r + 2*k2r + 2*k3r + k4r)/6)
    state = state.at[:, 1].set(state[:, 1] + (k1v + 2*k2v + 2*k3v + k4v)/6)

    return state




