from typing import Optional, Tuple, Callable, Union, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax import random
from jdgsim.potentials import combined_external_acceleration, combined_external_acceleration_vmpa_switch
from jdgsim.dynamics import DIRECT_ACC, direct_acc, DIRECT_ACC_LAXMAP, direct_acc_laxmap
LEAPFROG = 0

@partial(jax.jit, static_argnames=['config'])
def leapfrog(state, mass, dt, config, params):
    """
    Simple implementation of a symplectic Leapfrog (Verlet) integrator for N-body simulations.

    Parameters
    ----------
    state : jax.numpy.ndarray
        The state of the particles, where the first column represents positions and the second column represents velocities.
    mass : jax.numpy.ndarray
        The mass of the particles.
    dt : float
        Time-step for current integration.
    config : object
        Configuration object containing the acceleration scheme and external accelerations.
    params : dict
        Additional parameters for the acceleration functions.

    Returns
    -------
    jax.numpy.ndarray
        The updated state of the particles.
    """
    if config.acceleration_scheme == DIRECT_ACC:
        acc_func = direct_acc
    
    elif config.acceleration_scheme == DIRECT_ACC_LAXMAP:
        acc_func = direct_acc_laxmap
    
    acc = acc_func(state, mass, config, params)

    # Check additional accelerations
    if len(config.external_accelerations) > 0:
        acc = acc + combined_external_acceleration_vmpa_switch(state, config, params)
            
    # removing half-step velocity
    state = state.at[:, 0].set(state[:, 0] + state[:, 1]*dt + 0.5*acc*(dt**2))

    acc2 = acc_func(state, mass, config, params)

    if len(config.external_accelerations) > 0:
        acc2 = acc2 + combined_external_acceleration_vmpa_switch(state, config, params)
         
    state = state.at[:, 1].set(state[:, 1] + 0.5*(acc + acc2)*dt)
    
    return state





