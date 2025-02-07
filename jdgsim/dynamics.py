from typing import Optional, Tuple, Callable, Union, List
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax import random

DIRECT_ACC = 0


def single_body_acc(particle_i, particle_j, mass_i, mass_j, config, params) -> jnp.ndarray:
    """Compute the acceleration and potential of particle_i due to particle_j"""
    
    r_ij = particle_i[0, :] - particle_j[0, :]
    r_mag = jnp.linalg.norm(r_ij)   # Avoid division by zero and close encounter with config.softening
    # acc_mag = params.G  * mass_j / (r_mag + config.softening)**2
    # return - (acc_mag / (r_mag+config.softening)) * r_ij, - acc_mag*r_mag

    return - params.G * (mass_j) * (r_ij/(r_mag**2 + config.softening**2)**(3/2)), - params.G * mass_j / (r_mag + config.softening)
    

@partial(jax.jit, static_argnames=['config', 'return_potential'])
def direct_acc(state, mass, config, params, return_potential=False):
    """Compute net force acceleration and potential of each body using JAX's vmap."""

    def net_force_on_body(particle_i, mass_i):
        
        acc, potential = vmap(lambda particle_j, mass_j: single_body_acc(particle_i, particle_j, mass_i, mass_j, config, params))(state, mass)
        if return_potential:
            return jnp.sum(acc, axis=0), jnp.sum(potential, )
        else:
            return jnp.sum(acc, axis=0)

    return vmap(net_force_on_body)(state, mass)
