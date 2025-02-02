import jax
import jax.numpy as jnp
from jax import random


def Plummer_sphere(key, params, config, mass=1.0, ):
    """
    Generate particle samples from a Plummer sphere distribution using JAX.
    
    :param key: JAX random key.
    """
    
    # Generate positions
    def generate_position(key):
        while True:
            key, subkey = random.split(key)
            x1, x2, x3 = random.uniform(subkey, (3,), minval=-1, maxval=1)
            r2 = x1**2 + x2**2 + x3**2
            if r2 < 1:
                break
        r = (r2**(-2/3) - 1)**(-0.5)
        return r * jnp.array([x1, x2, x3]), key

    positions = []
    for _ in range(config.N_particles):
        
        pos, key = generate_position(key)
        positions.append(pos)
    positions = jnp.array(positions)

    # Generate velocities
    def generate_velocity(key, position):
        while True:
            key, subkey = random.split(key)
            x1, x2, x3 = random.uniform(subkey, (3,), minval=-1, maxval=1)
            v2 = x1**2 + x2**2 + x3**2
            if v2 < 1:
                break
        v = jnp.sqrt(2 * params.G * mass / (1 + jnp.sum(position**2))**0.5)
        return v * jnp.array([x1, x2, x3]), key

    velocities = []
    for pos in positions:
        vel, key = generate_velocity(key, pos)
        velocities.append(vel)
    velocities = jnp.array(velocities)
    
    mass = jnp.ones(config.N_particles)
    return positions, velocities, mass
