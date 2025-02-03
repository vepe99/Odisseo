from typing import Optional, Tuple, Callable, Union, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax import random

from jdgsim.utils import radius

NFW_POTENTIAL = 0

@partial(jax.jit, static_argnames=['config'])
def combined_external_acceleration(state, config, params):

    total_external_acceleration = jnp.zeros_like(state[:, 0])
    if NFW_POTENTIAL in config.external_accelerations:
        total_external_acceleration = total_external_acceleration + NFW(state, config, params)
        
    #TO BE IMPLEMENTED, VECTORIZE THE SUM OVER ALL THE EXTERNAL ACCELERATIONS FUNCTIONS
    
    return total_external_acceleration 


@partial(jax.jit, static_argnames=['config'])
def NFW(state, config, params):
    
    params_NFW = params.NFW_params
    
    if params_NFW['c'] is not None:
        params_NFW['d_c'] = jnp.log(1+params_NFW['c']) - params_NFW['c']/(1+params_NFW['c']) #critical density
        
    r  = radius(state)

    NUM = (params_NFW['r_s']+r)*jnp.log(1+r/params_NFW['r_s']) - r
    DEN = r*r*r*(params_NFW['r_s']+r)*params_NFW['d_c']

    acc = - params.G * params_NFW['Mvir']*NUM[:, jnp.newaxis]/DEN[:, jnp.newaxis] * state[:, 0]
    # pot = -G * params_NFW['Mvir']*jnp.log(1+r/params_NFW['r_s'])/(r*params_NFW['d_c'])

    return acc
