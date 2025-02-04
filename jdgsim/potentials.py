from typing import Optional, Tuple, Callable, Union, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax import random

NFW_POTENTIAL = 0

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
def NFW(state, config, params, return_potential=False):
    
    params_NFW = params.NFW_params
    
    if params_NFW['c'] is not None:
        params_NFW['d_c'] = jnp.log(1+params_NFW['c']) - params_NFW['c']/(1+params_NFW['c']) #critical density
        
    r  = jnp.linalg.norm(state[:, 0], axis=1)

    NUM = (params_NFW['r_s']+r)*jnp.log(1+r/params_NFW['r_s']) - r
    DEN = r*r*r*(params_NFW['r_s']+r)*params_NFW['d_c']

    acc = - params.G * params_NFW['Mvir']*NUM[:, jnp.newaxis]/DEN[:, jnp.newaxis] * state[:, 0]
    pot = -params.G * params_NFW['Mvir']*jnp.log(1+r/params_NFW['r_s'])/(r*params_NFW['d_c'])

    if return_potential:
        return acc, pot
    else:
        return acc
