from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, random
from jax.lax import while_loop
import numpy as np
from multiprocessing import Pool


from jdgsim.utils import E_pot
from jdgsim.dynamics import direct_acc
from jdgsim import construct_initial_state

# @partial(jax.jit, static_argnames=['config', 'rejection_samples'])
# def Plummer_sphere_jax(key, mass, params, config, rejection_samples= 10_000):

    # Plummer_Mtot = jnp.sum(mass)
    # r = jnp.sqrt( params.Plummer_a / (random.uniform(key, shape=config.N_particles)**(-3/2) -1))
    # phi = random.uniform(key, shape=config.N_particles, minval=0, maxval=jnp.pi) 
    # sin_i = random.uniform(key, shape=config.N_particles, minval=-1, maxval=1)
    
    # positions = jnp.array([r*jnp.cos(jnp.arcsin(sin_i))*jnp.cos(phi), r*jnp.cos(jnp.arcsin(sin_i))*jnp.sin(phi), r*sin_i]).T
    # potential = - params.G * Plummer_Mtot / jnp.sqrt( jnp.linalg.norm(positions, axis=1)**2 + params.Plummer_a**2)
    
    # def generate_velocity(key, potential_i):
    #     velocity_i = random.uniform(key, shape=(rejection_samples, 3), minval=-jnp.sqrt(2*potential_i), maxval=jnp.sqrt(2*potential_i))
    #     escape_velocity_mask = jnp.sum(velocity_i**2, axis=1) <= 2*potential_i
    #     isotropic_velocity_mask = random.uniform(key, shape=rejection_samples) <= ((0.5 * jnp.sum(velocity_i**2, axis=1) + potential_i ) / potential_i)**(7/2)
    #     valid_mask = jnp.logical_and(escape_velocity_mask, isotropic_velocity_mask)
    #     return jnp.where(valid_mask, velocity_i[valid_mask], jnp.zeros(3))
        
    # velocities = vmap(generate_velocity)(random.split(key, potential.shape[0]), potential)   
        
    # Plummer_Mtot = jnp.sum(mass)
    # r = jnp.sqrt( params.Plummer_a / (random.uniform(key, shape=config.N_particles)**(-3/2) -1))
    # phi = random.uniform(key, shape=config.N_particles, minval=0, maxval=jnp.pi) 
    # sin_i = random.uniform(key, shape=config.N_particles, minval=-1, maxval=1)
    
    # positions = jnp.array([r*jnp.cos(jnp.arcsin(sin_i))*jnp.cos(phi), r*jnp.cos(jnp.arcsin(sin_i))*jnp.sin(phi), r*sin_i]).T
    # potential = - params.G * Plummer_Mtot / jnp.sqrt( jnp.linalg.norm(positions, axis=1)**2 + params.Plummer_a**2)
    
    # def generate_velocity(key, potential_i):        
    #     # while True:
    #     #     velocity_i = random.uniform(key, shape=3, minval=-jnp.sqrt(2*potential_i), maxval=jnp.sqrt(2*potential_i))
    #     #     if jnp.sum(velocity_i**2) <= 2*potential_i:
    #     #         u = random.uniform(key)
    #     #         f = ((0.5 * jnp.sum(velocity_i**2) + potential_i ) / potential_i)**(7/2)
    #     #         if u <= f:
    #     #             return velocity_i
                
    #     def cond_func(velocity_i):
    #         return (jnp.sum(velocity_i**2) > 2*potential_i)|(random.uniform(key) > ((0.5 * jnp.sum(velocity_i**2) + potential_i ) / potential_i)**(7/2) )
    #     def body_func(velocity_i):
    #         velocity_i = random.uniform(key, shape=3, minval=-jnp.sqrt(2*potential_i), maxval=jnp.sqrt(2*potential_i))
    #         return velocity_i
    #     initial_val = jnp.zeros(3)
    #     while_loop(cond_fun=cond_func, body_fun=body_func, init_val=initial_val)
    
    # velocities = vmap(generate_velocity)(random.split(key, potential.shape), potential)
    
    # return positions, velocities
 
def generate_velocity_Plummer(potential_i, rejection_samples=1000):
        velocity_i = np.random.uniform(size=(rejection_samples, 3), low=-np.sqrt(-2*potential_i), high=np.sqrt(-2*potential_i))
        escape_velocity_mask = np.sum(velocity_i**2, axis=1) <= - 2*potential_i
        isotropic_velocity_mask = np.random.uniform(size=rejection_samples) <= ((0.5 * np.sum(velocity_i**2, axis=1) + potential_i ) / potential_i)**(7/2)
        return velocity_i[(escape_velocity_mask)&(isotropic_velocity_mask)][0]
    
def Plummer_sphere(mass, params, config,):
    Plummer_Mtot = np.sum(mass)
    r = np.sqrt( params.Plummer_a / (np.random.uniform(size=config.N_particles)**(-3/2) -1))
    phi = np.random.uniform(size=config.N_particles, low=0, high=np.pi) 
    sin_i = np.random.uniform(size=config.N_particles, low=-1, high=1)
    
    positions = np.array([r*np.cos(np.arcsin(sin_i))*np.cos(phi), r*np.cos(np.arcsin(sin_i))*np.sin(phi), r*sin_i]).T
    potential = - params.G * Plummer_Mtot / np.sqrt( np.linalg.norm(positions, axis=1)**2 + params.Plummer_a**2)
    with Pool(processes=1) as pool:
        velocities = pool.map(generate_velocity_Plummer, potential)
    return jnp.array(positions), jnp.array(velocities)


# def Plummer_sphere(key, params, config, mass=1.0, ):
#     """
#     Generate particle samples from a Plummer sphere distribution using JAX.
    
#     :param key: JAX random key.
#     """
        
    
#     # Generate positions
#     def generate_position(key):
#         while True:
#             key, subkey = random.split(key)
#             x1, x2, x3 = random.uniform(subkey, (3,), minval=-1, maxval=1)
#             r2 = x1**2 + x2**2 + x3**2
#             if r2 < 1:
#                 break
#         r = (r2**(-2/3) - 1)**(-0.5)
#         return r * jnp.array([x1, x2, x3]), key

#     positions = []
#     for _ in range(config.N_particles):
        
#         pos, key = generate_position(key)
#         positions.append(pos)
#     positions = jnp.array(positions)

#     # Generate velocities
#     def generate_velocity(key, position):
#         while True:
#             key, subkey = random.split(key)
#             x1, x2, x3 = random.uniform(subkey, (3,), minval=-1, maxval=1)
#             v2 = x1**2 + x2**2 + x3**2
#             if v2 < 1:
#                 break
#         v = jnp.sqrt(2 * params.G * mass / (1 + jnp.sum(position**2))**0.5)
#         return v * jnp.array([x1, x2, x3]), key

#     velocities = []
#     for pos in positions:
#         vel, key = generate_velocity(key, pos)
#         velocities.append(vel)
#     velocities = jnp.array(velocities)
    
#     mass = jnp.ones(config.N_particles)
#     return positions, velocities, mass
