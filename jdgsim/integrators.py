from typing import Optional, Tuple, Callable, Union, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax import random
from jdgsim.potentials import combined_external_acceleration
from jdgsim.dynamics import direct_acc

@partial(jax.jit, static_argnames=['dt', 'config'])
def leapfrog(state, mass, dt, config, params):
    """
    Simple implementation of a symplectic Leapfrog (Verlet) integrator for N-body simulations.
    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param dt: Times-step for current integration (notice some methods can use smaller sub-time step to
    achieve the final result
    :param config.acceleration_scheme: It needs to be a function from the module (:mod:`fireworks.nbodylib.dynamics`)
    following the input/output style of the template function  (:func:`fireworks.nbodylib.dynamics.acceleration_estimate_template`).
    :param config.softening: config.softening parameter for the acceleration estimate, can use 0 as default value
    :param external_accelerations: a list of additional force estimators (e.g. an external potential field) to
    consider to estimate the final acceleration (and if available jerk) on the particles
    :return: A tuple with 5 elements:

        - The updated particles instance
        - dt, the effective timestep evolved in the simulation (for some integrator this can be
            different wrt the input dt)
    """
    if config.acceleration_scheme == 'direct_acc':
        aff_func = direct_acc
    
    acc = aff_func(state, mass, config, params)

    # Check additional accelerations
    if len(config.external_accelerations) > 0:
        acc = acc + combined_external_acceleration(state, config, params)
            
    # removing half-step velocity
    state = state.at[:, 0].set(state[:, 0] + state[:, 1]*dt + 0.5*acc*(dt**2))

    acc2 = aff_func(state, mass, config, params)

    if len(config.external_accelerations) > 0:
        acc2 = acc2 + combined_external_acceleration(state, config, params)
         
    state = state.at[:, 1].set(state[:, 1] + 0.5*(acc + acc2)*dt)
    
    return state
