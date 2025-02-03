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
    acc_ij = params.G  * mass_j / (r_mag + config.softening)**2
    return - (acc_ij / (r_mag+config.softening)) * r_ij, - acc_ij*r_mag
    

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




# def acceleration_direct(particles: Particles, config.softening: float = 0):
    
#     acceleration = jnp.zeros_like(particles.position)
    
#     for i in range(N-1):
#         for j in range(i+1, N):
#             #distance matrix
#             rij = particles.position[i] - particles.position[j]
#             #distance vector
#             r = jnp.linalg.norm(rij) + config.softening
#             #acceleration caused by the j-th particle on the i-th particle
#             a = -rij * particles.mass[j] / r**3
#             acceleration = acceleration.at[i].add(a)
#             #Newton's 3rd law, accelaration caused by the i-th particle on the j-th particle
#             acceleration = acceleration.at[j].add(-(particles.mass[i]/particles.mass[j])*a) 
            
#             return acceleration
        
# def acceleration_vectorized(particles: Particles, config.softening: float = 0):
    
#     N = particles.position.shape[0]
#     acceleration = jnp.zeros_like(particles.position)
    
#     #distance matrix
#     rij = particles.position[:, jnp.newaxis, :] - particles.position[jnp.newaxis, :, :]
#     #distance vector
#     r = jnp.linalg.norm(rij, axis=2) + config.softening
#     #acceleration caused by the j-th particle on the i-th particle
#     a = -rij * particles.mass[j] / r**3
#     acceleration = jnp.sum(a, axis=1)
    
#     return acceleration 