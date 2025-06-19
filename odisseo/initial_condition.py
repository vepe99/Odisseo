from typing import Optional, Tuple, Callable, Union, List, NamedTuple
from functools import partial
from jaxtyping import jaxtyped, PRNGKeyArray
from beartype import beartype as typechecker

import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
from multiprocessing import Pool

from odisseo.option_classes import SimulationConfig, SimulationParams

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['config'])
def Plummer_sphere(key: PRNGKeyArray,
                    config: SimulationConfig,
                    params: SimulationParams) -> Tuple:
    """
    Create initial conditions for a Plummer sphere. The sampling of velocities 
    is done by inverse fitting the cumulative distribution function of the 
    Plummer sphere.

    Args:
        key (jax.random.PRNGKey): Random key.
        config (NamedTuple): Configuration NamedTuple containing the number of particles (N_particles).
        params (NamedTuple): Parameters NamedTuple containing:
        
    Returns:
        tuple: A tuple containing:
            positions (jnp.array): Array of shape (N_particles, 3) representing the positions of the particles.
            velocities (jnp.array): Array of shape (N_particles, 3) representing the velocities of the particles.
            masses (jnp.array): Array of shape (N_particles,) representing the masses of the particles.
    """
    
    key_r, key_phi, key_sin_i, key_u, key_phi_v, key_sin_i_v= random.split(key, 6)
    r = jnp.sqrt( params.Plummer_params.a**2 / (random.uniform(key=key_r, shape=(config.N_particles,))**(-2/3) -1  )   )
    phi = random.uniform(key=key_phi, shape=(config.N_particles,), minval=0, maxval=2*jnp.pi) 
    sin_i = random.uniform(key=key_sin_i, shape=(config.N_particles,), minval=-1, maxval=1)
    positions = jnp.array([r*jnp.cos(jnp.arcsin(sin_i))*jnp.cos(phi), 
                           r*jnp.cos(jnp.arcsin(sin_i))*jnp.sin(phi), 
                           r*sin_i]).T
    potential = - params.G * params.Plummer_params.Mtot / jnp.sqrt( jnp.linalg.norm(positions, axis=1)**2 + params.Plummer_params.a**2)
    velocities_escape = jnp.sqrt(-2*potential )


    def G(q):
        """
        Normalize Cumulative distribution function of q=v/v_escape for a Plummer sphere.
        The assosiate unormalized probability distribution function assosiated with it is
        g(q) = (1-q)**(7/2) * q**2

        Args:
            q: (float) Velocity ratio v/v_escape.

        Returns:
            float: Normalized cumulative distribution function.
        """
        # return 1287/16 * ((-2*(1-q)**(9/2))*(99*q**2+36*q+8)/1287 +16/1287)
        return 1/(jnp.pi*7/512) * (q*jnp.sqrt(1 - q**2)*(-384*q**8 + 1488*q**6 - 2104*q**4 + 1210*q**2 - 105) + 105*jnp.asin(q))/3840
    
    # Invere fitting
    q = jnp.linspace(0, 1, 500)
    y = G(q)

    u = random.uniform(key=key_u, shape=(config.N_particles,))
    samples = jnp.interp(u, y, q)
    velocities_modulus = samples * velocities_escape

    # Generate random angles for the velocity
    phi_v = random.uniform(key=key_phi_v, shape=(config.N_particles,), minval=0, maxval=2*jnp.pi) 
    sin_i_v = random.uniform(key=key_sin_i_v, shape=(config.N_particles,), minval=-1, maxval=1)
    velocities = velocities_modulus[:, None]*jnp.array([jnp.cos(jnp.arcsin(sin_i_v))*jnp.cos(phi_v), 
                                                jnp.cos(jnp.arcsin(sin_i_v))*jnp.sin(phi_v), 
                                                sin_i_v]).T


    # return jnp.array(positions), jnp.array(velocities), params.Plummer_params.Mtot/config.N_particles*jnp.ones(config.N_particles)
    return jnp.array(positions), jnp.array(velocities), 1/config.N_particles*jnp.ones(config.N_particles)
     
 
  
def Plummer_sphere_multiprocess(mass, config, params):
    """
    Generate initial conditions for a Plummer sphere using rejection sampling.
    The calculation is done using numpy.

    Args:
        mass (float): The total mass of the Plummer sphere.
        config (NamedTuple): Configuration NamedTuple containing the number of particles (N_particles).
        params (NamedTuple): Parameters NamedTuple containing:
    
    Returns:
        tuple: A tuple containing:
            positions (jnp.array): Array of shape (N_particles, 3) representing the positions of the particles.
            velocities (jnp.array): Array of shape (N_particles, 3) representing the velocities of the particles.
            masses (jnp.array): Array of shape (N_particles,) representing the masses of the particles.
    """
    Plummer_Mtot = params.Plummer_params.Mtot
    r = np.sqrt( params.Plummer_params.a / (np.random.uniform(size=config.N_particles)**(-3/2) -1))
    phi = np.random.uniform(size=config.N_particles, low=0, high=2*np.pi) 
    sin_i = np.random.uniform(size=config.N_particles, low=-1, high=1)
    
    positions = np.array([r*np.cos(np.arcsin(sin_i))*np.cos(phi), 
                          r*np.cos(np.arcsin(sin_i))*np.sin(phi), 
                          r*sin_i]).T
    potential = - params.G * Plummer_Mtot / (r**2 + params.Plummer_params.a**2)

    def generate_velocity_Plummer(potential_i, rejection_samples=1000):
            velocity_i = np.random.uniform(size=(rejection_samples, 3), low=-np.sqrt(-2*potential_i), high=np.sqrt(-2*potential_i))
            escape_velocity_mask = np.sum(velocity_i**2, axis=1) <= - 2*potential_i
            isotropic_velocity_mask = np.random.uniform(size=rejection_samples) <= ((0.5 * np.sum(velocity_i**2, axis=1) + potential_i ) / potential_i)**(7/2)
            return velocity_i[(escape_velocity_mask)&(isotropic_velocity_mask)][0]
    
    with Pool(processes=1) as pool:
        velocities = pool.map(generate_velocity_Plummer, potential)
    return jnp.array(positions), jnp.array(velocities), 100/config.N_particles*jnp.ones(config.N_particles)

@partial(jax.jit)
def ic_two_body(mass1: Union[float, jnp.ndarray],
                mass2: Union[float, jnp.ndarray],
                rp: Union[float, jnp.ndarray],
                e: Union[float, jnp.ndarray],
                params: SimulationParams) -> Tuple:
    """
    Create initial conditions for a two-body system.
    
    By default, the two bodies will be placed along the x-axis at the
    closest distance rp. Depending on the input eccentricity, the two 
    bodies can be in a circular (e < 1), parabolic (e = 1), or hyperbolic 
    orbit (e > 1).

    Args:
        mass1 (float): Mass of the first body [nbody units].
        mass2 (float): Mass of the second body [nbody units].
        rp (float): Closest orbital distance [nbody units].
        e (float): Eccentricity.
        config (NamedTuple): Configuration NamedTuple.
        params (NamedTuple): Parameters NamedTuple.
    Returns:
        tuple: A tuple containing:
            - pos (jnp.ndarray): Positions of the particles.
            - vel (jnp.ndarray): Velocities of the particles.
            - mass (jnp.ndarray): Masses of the particles.
    """

    Mtot=mass1+mass2

    # if e==1.:
    #     vrel=jnp.sqrt(params.G * 2*Mtot/rp)
    # else:
    #     a=rp/(1-e)
    #     vrel=jnp.sqrt(params.G * Mtot*(2./rp-1./a))

    def circular_orbit():
        return jnp.sqrt(params.G * 2*Mtot/rp)
    def elliptical_orbit():
        a=rp/(1-e)
        return jnp.sqrt(params.G * Mtot*(2./rp-1./a))

    vrel = jax.lax.cond(
            e == 1.,
            circular_orbit,
            elliptical_orbit,
            )

    v1 = -params.G*mass2/Mtot * vrel
    v2 = params.G*mass1/Mtot * vrel

    pos  = jnp.array([[0.,0.,0.],[rp,0.,0.]])
    vel  = jnp.array([[0.,v1,0.],[0.,v2,0.]])
    mass = jnp.array([mass1, mass2])

    return pos, vel, mass
    
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_samples'])
def sample_position_on_sphere(key: PRNGKeyArray,
                              r_p: float,
                              num_samples: int = 1):
    """
    Sample uniform positions on a sphere of radius r_p.

    Args:
        key (jax.random.PRNGKey): JAX random key for sampling.
        r_p (float): Radius of the sphere.
        num_samples (int): Number of samples to generate. Deafult is 1.

    Returns:
        jnp.ndarray: Sampled positions (num_samples, 3).
    """
    subkey1, subkey2 = random.split(key)
    
    # Sample phi uniformly in [0, 2π]
    phi = random.uniform(subkey1, shape=(num_samples,), minval=0, maxval=2*jnp.pi)
    
    # Sample cos(theta) uniformly in [-1, 1] to ensure uniform distribution on the sphere
    costheta = random.uniform(subkey2, shape=(num_samples,), minval=-1, maxval=1)
    theta = jnp.arccos(costheta)  # Convert to theta
    
    # Convert to Cartesian coordinates
    x = r_p * jnp.sin(theta) * jnp.cos(phi)
    y = r_p * jnp.sin(theta) * jnp.sin(phi)
    z = r_p * jnp.cos(theta)

    return jnp.stack([x, y, z], axis=-1)

@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['num_samples'])
def sample_position_on_circle(key: PRNGKeyArray,
                             r_p: float,
                             num_samples: int =1):
    """
    Sample uniform positions on a sphere of radius r_p.

    Args:
        key (jax.random.PRNGKey): JAX random key for sampling.
        r_p (float): Radius of the sphere.
        num_samples (int): Number of samples to generate.

    Returns:
        jnp.ndarray: Sampled positions (num_samples, 3).
    """
    subkey1, subkey2 = random.split(key)
    
    # Sample phi uniformly in [0, 2π]
    phi = random.uniform(subkey1, shape=(num_samples,), minval=0, maxval=2*jnp.pi)
    
    # Sample cos(theta) uniformly in [-1, 1] to ensure uniform distribution on the sphere
    theta = jnp.radians(90) # Convert to theta
    
    # Convert to Cartesian coordinates
    x = r_p * jnp.sin(theta) * jnp.cos(phi)
    y = r_p * jnp.sin(theta) * jnp.sin(phi)
    z = jnp.zeros_like(x)

    return jnp.stack([x, y, z], axis=-1)

@jaxtyped(typechecker=typechecker)
@partial(jax.jit,)
def inclined_position(position: jnp.ndarray,
                     inclination: jnp.ndarray):
    """
    Convert position on the xy-plane to an inclined orbit.

   Args:
        position (jnp.ndarray): (x, y, z) position of the Plummer sphere.
        inclination (jnp.ndarray): Inclination angle in radians.

    Returns:
        jnp.ndarray: (x, y, z) position of the Plummer sphere after inclination.
    """
    x, y, z = position.T
    phi = jnp.arctan2(y, x)[0]  # Azimuthal angle

    # Rotation matrix around x-axis by inclination
    R_x = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(inclination), -jnp.sin(inclination)],
        [0, jnp.sin(inclination), jnp.cos(inclination)]
    ])

    # Rotate position around x-axis
    rotated_position = R_x @ position.T

    return rotated_position.T

@jaxtyped(typechecker=typechecker)
@partial(jax.jit,)
def inclined_circular_velocity(position: jnp.ndarray, 
                               v_c: jnp.ndarray, 
                               inclination: jnp.ndarray):
    """
    Convert circular velocity module on the xy plane to an inclined orbit Cartesian components.
    
    Args:
        position (jnp.ndarray): (x, y, z) position of the Plummer sphere.
        v_c (float): Circular velocity (km/s).
        inclination (float): Inclination angle in radians.
    Returns:
        jnp.ndarray: (v_x, v_y, v_z) velocity components.
    """
    x, y, z = position.T
    phi = jnp.arctan2(y, x) # Azimuthal angle
    # Initial velocity vector
    velocity = jnp.zeros(3)
    velocity = velocity.at[1].set(v_c[0])
    # Rotation matrix around z-axis by phi
    R_z = jnp.array([
        [jnp.cos(phi[0]), -jnp.sin(phi[0]), 0],
        [jnp.sin(phi[0]), jnp.cos(phi[0]), 0],
        [0, 0, 1]
    ])

    # Rotate velocity around z-axis
    velocity = R_z @ velocity

    # Rotation matrix around x-axis by inclination
    R_x = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(inclination), -jnp.sin(inclination)],
        [0, jnp.sin(inclination), jnp.cos(inclination)]
    ])

    # Rotate velocity around x-axis
    rotated_velocity = R_x @ velocity

    return rotated_velocity