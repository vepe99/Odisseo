from typing import Optional, Tuple, Callable, Union, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
from jax import random

NFW_POTENTIAL = 0
POINT_MASS = 1
MN_POTENTIAL = 2


@partial(jax.jit, static_argnames=['config', 'return_potential'])
def combined_external_acceleration(state, config, params, return_potential=False):
    #TO BE IMPLEMENTED, VECTORIZE THE SUM OVER ALL THE EXTERNAL ACCELERATIONS FUNCTIONS

    total_external_acceleration = jnp.zeros_like(state[:, 0])
    total_external_potential = jnp.zeros_like(config.N_particles)
    if return_potential:
        if NFW_POTENTIAL in config.external_accelerations:
            acc_NFW, pot_NFW = NFW(state, config, params, return_potential=True)
            total_external_acceleration = total_external_acceleration + acc_NFW
            total_external_potential = total_external_potential +   pot_NFW
        return total_external_acceleration, total_external_potential
    else:
        if NFW_POTENTIAL in config.external_accelerations:
            total_external_acceleration = total_external_acceleration + NFW(state, config, params)
        return total_external_acceleration
    
@partial(jax.jit, static_argnames=['config', 'return_potential'])    
def combined_external_acceleration_vmpa_switch(state, config, params, return_potential=False):
    total_external_acceleration = jnp.zeros_like(state[:, 0])
    total_external_potential = jnp.zeros_like(config.N_particles)
    state_tobe_vmap  = jnp.repeat(state[jnp.newaxis, ...], repeats=len(config.external_accelerations), axis=0)
    if return_potential:
        # The POTENTIAL_LIST NEEDS TO BE IN THE SAME ORDER AS THE INTEGER VALUES 
        POTENTIAL_LIST = [lambda state: NFW(state, config=config, params=params, return_potential=True), 
                        lambda state: point_mass(state, config=config, params=params, return_potential=True),
                        lambda state: MyamotoNagai(state, config=config, params=params, return_potential=True)]
        vmap_function = vmap(lambda i, state: lax.switch(i, POTENTIAL_LIST, state))
        external_acc, external_pot = vmap_function(jnp.array(config.external_accelerations), state_tobe_vmap)
        total_external_acceleration = jnp.sum(external_acc, axis=0)
        total_external_potential = jnp.sum(external_pot, axis=0)
        return total_external_acceleration, total_external_potential
    else:
        POTENTIAL_LIST = [lambda state: NFW(state, config=config, params=params, return_potential=False),
                          lambda state: point_mass(state, config=config, params=params, return_potential=False),
                          lambda state: MyamotoNagai(state, config=config, params=params, return_potential=False)]
        vmap_function = vmap(lambda i, state: lax.switch(i, POTENTIAL_LIST, state))
        external_acc = vmap_function(jnp.array(config.external_accelerations), state_tobe_vmap)
        total_external_acceleration = jnp.sum(external_acc, axis=0)
        return total_external_acceleration


@partial(jax.jit, static_argnames=['config', 'return_potential'])
def NFW(state, config, params, return_potential=False):
    """
    Compute acceleration of all particles due to a NFW profile.

    Parameters
    ----------
    state : jnp.ndarray
        Array of shape (N_particles, 6) representing the positions and velocities of the particles. 
    config: NamedTuple
        Configuration parameters.
    params: NamedTuple
        Simulation parameters.
    return_potential: bool
        If True, also returns the potential energy of the NFW profile.
    
    Returns
    -------
    Tuple
        - Acceleration: jnp.ndarray 
            Acecleration of all particles due to NFW external potential
        - Potential: jnp.ndarray
            Potential energy of all particles due to NFW external potential
            Returned only if return_potential is True.   
    """
    
    params_NFW = params.NFW_params
    
    r  = jnp.linalg.norm(state[:, 0], axis=1)

    NUM = (params_NFW.r_s+r)*jnp.log(1+r/params_NFW.r_s) - r
    DEN = r*r*r*(params_NFW.r_s+r)*params_NFW.d_c

    acc =  - params.G * params_NFW.Mvir*NUM[:, jnp.newaxis]/DEN[:, jnp.newaxis] * state[:, 0]
    pot = - params.G * params_NFW.Mvir*jnp.log(1+r/params_NFW.r_s)/(r*params_NFW.d_c)

    if return_potential:
        return acc, pot
    else:
        return acc
    

@partial(jax.jit, static_argnames=['config', 'return_potential'])
def point_mass(state, config, params, return_potential=False):
    """
    Compute acceleration of all particles due to a point mass.

    Parameters
    ----------
    state : jnp.ndarray
        Array of shape (N_particles, 6) representing the positions and velocities of the particles. 
    config: NamedTuple
        Configuration parameters.
    params: NamedTuple
        Simulation parameters.
    return_potential: bool
        If True, also returns the potential energy of the point mass.
    
    Returns
    -------
    Tuple
        - Acceleration: jnp.ndarray 
            Acecleration of all particles due to point mass external potential
        - Potential: jnp.ndarray
            Potential energy of all particles due to point mass external potential
            Returned only if return_potential is True.   
    """
    params_point_mass = params.PointMass_params
    
    r  = jnp.linalg.norm(state[:, 0], axis=1)
    
    acc = - params.G * params_point_mass.M * state[:, 0] / (r**3)[:, None]
    pot = - params.G * params_point_mass.M / r
    
    if return_potential:
        return acc, pot
    else:
        return acc
    

@partial(jax.jit, static_argnames=['config', 'return_potential'])
def MyamotoNagai(state, config, params, return_potential=False):
    """
    Compute acceleration of all particles due to a Miyamoto-Nagai profile.

    Parameters
    ----------
    state : jnp.ndarray
        Array of shape (N_particles, 6) representing the positions and velocities of the particles. 
    config: NamedTuple
        Configuration parameters.
    params: NamedTuple
        Simulation parameters.
    return_potential: bool
        If True, also returns the potential energy of the Miyamoto-Nagai profile.
    
    Returns
    -------
    Tuple
        - Acceleration: jnp.ndarray 
            Acecleration of all particles due to Miyamoto-Nagai external potential
        - Potential: jnp.ndarray
            Potential energy of all particles due to Miyamoto-Nagai external potential
            Returned only if return_potential is True.   
    """
    params_MN = params.MN_params
    
    z2 = state[:, 0, 2]**2
    b = params_MN.b
    a = params_MN.a

    Dz = (a+(z2+b**2)**0.5)
    D = jnp.sum(state[:, 0, :2]**2, axis=1) + Dz**2
    K = params.G * params_MN.M / D**(3/2)
    ax = -K * state[:, 0, 0]
    ay = -K * state[:, 0, 1]
    az = -K * state[:, 0, 2] * Dz / (z2 + b**2)**0.5
    acc = jnp.stack([ax, ay, az], axis=1)

    pot = - params.G * params_MN.M / jnp.sqrt(D)
    
    if return_potential:
        return acc, pot
    else:
        return acc