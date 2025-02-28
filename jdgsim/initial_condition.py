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

def Plummer_sphere(key, config, params):
    """
    Create initial conditions for a Plummer sphere. The sampling of velocities is done by inverse fitting 
    the cumulative distribution function of the Plummer sphere.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    config : NamedTuple
        Configuration NamedTuple containing the number of particles (N_particles).
    params : NamedTuple
        Parameters NamedTuple containing:
        - Plummer_a : float
            Scale length of the Plummer sphere.
        - G : float
            Gravitational constant.
    Returns
    -------
    tuple
        A tuple containing:
        - positions : jnp.array
            Array of shape (N_particles, 3) representing the positions of the particles.
        - velocities : jnp.array
            Array of shape (N_particles, 3) representing the velocities of the particles.
        - masses : jnp.array
            Array of shape (N_particles,) representing the masses of the particles.
    """
    
    Plummer_Mtot = params.Plummer_params.Mtot
    key_r, key_phi, key_sin_i, key_u, key_phi_v, key_sin_i_v= random.split(key, 6)
    r = jnp.sqrt( params.Plummer_params.a / (random.uniform(key=key_r, shape=(config.N_particles,))**(-3/2) -1))
    phi = random.uniform(key=key_phi, shape=(config.N_particles,), minval=0, maxval=jnp.pi) 
    sin_i = random.uniform(key=key_sin_i, shape=(config.N_particles,), minval=-1, maxval=1)
    
    positions = jnp.array([r*jnp.cos(jnp.arcsin(sin_i))*jnp.cos(phi), r*jnp.cos(jnp.arcsin(sin_i))*jnp.sin(phi), r*sin_i]).T
    potential = - params.G * Plummer_Mtot / jnp.sqrt( jnp.linalg.norm(positions, axis=1)**2 + params.Plummer_params.a**2)
    velocities_escape = jnp.sqrt(-2*potential)


    def G(q):
        """
        Normalize Cumulative distribution function of q=v/v_escape for a Plummer sphere.
        The assosiate unormalized probability distribution function assosiated with it is
        g(q) = (1-q)**(7/2) * q**2

        Parameters
        ----------
        q : float
            Velocity ratio v/v_escape.
        """
        return 1287/16 * ((-2*(1-q)**(9/2))*(99*q**2+36*q+8)/1287 +16/1287)
    
    # Invere fitting
    q = jnp.linspace(0, 1, 100_000)
    y = G(q)

    u = random.uniform(key=key_u, shape=(config.N_particles,))
    samples = jnp.interp(u, y, q)
    velocities_modulus = samples * velocities_escape

    # Generate random angles for the velocity
    phi_v = random.uniform(key=key_phi_v, shape=(config.N_particles,), minval=0, maxval=jnp.pi) 
    sin_i_v = random.uniform(key=key_sin_i_v, shape=(config.N_particles,), minval=-1, maxval=1)
    velocities = jnp.array([velocities_modulus*jnp.cos(jnp.arcsin(sin_i_v))*jnp.cos(phi_v), velocities_modulus*jnp.cos(jnp.arcsin(sin_i_v))*jnp.sin(phi_v), velocities_modulus*sin_i_v]).T


    return jnp.array(positions), jnp.array(velocities), 1/config.N_particles*jnp.ones(config.N_particles)
     
 
  
def Plummer_sphere_multiprocess(mass, config, params):
    """
    Parameters
    ----------
    mass : float
        The total mass of the Plummer sphere.
    config : NamedTuple
        Configuration NamedTuple containing the number of particles (N_particles).
    params : NamedTuple
        Parameters NamedTuple containing:
        - Plummer_a : float
            Scale length of the Plummer sphere.
        - G : float
            Gravitational constant.
    Returns
    -------
    tuple
        A tuple containing:
        - positions : jnp.array
            Array of shape (N_particles, 3) representing the positions of the particles.
        - velocities : jnp.array
            Array of shape (N_particles, 3) representing the velocities of the particles.
        - masses : jnp.array
            Array of shape (N_particles,) representing the masses of the particles.
    """
    Plummer_Mtot = params.Plummer_params.Mtot
    r = np.sqrt( params.Plummer_params.a / (np.random.uniform(size=config.N_particles)**(-3/2) -1))
    phi = np.random.uniform(size=config.N_particles, low=0, high=np.pi) 
    sin_i = np.random.uniform(size=config.N_particles, low=-1, high=1)
    
    positions = np.array([r*np.cos(np.arcsin(sin_i))*np.cos(phi), r*np.cos(np.arcsin(sin_i))*np.sin(phi), r*sin_i]).T
    potential = - params.G * Plummer_Mtot / np.sqrt( np.linalg.norm(positions, axis=1)**2 + params.Plummer_params.a**2)

    def generate_velocity_Plummer(potential_i, rejection_samples=1000):
            velocity_i = np.random.uniform(size=(rejection_samples, 3), low=-np.sqrt(-2*potential_i), high=np.sqrt(-2*potential_i))
            escape_velocity_mask = np.sum(velocity_i**2, axis=1) <= - 2*potential_i
            isotropic_velocity_mask = np.random.uniform(size=rejection_samples) <= ((0.5 * np.sum(velocity_i**2, axis=1) + potential_i ) / potential_i)**(7/2)
            return velocity_i[(escape_velocity_mask)&(isotropic_velocity_mask)][0]
    
    with Pool(processes=1) as pool:
        velocities = pool.map(generate_velocity_Plummer, potential)
    return jnp.array(positions), jnp.array(velocities), 1/config.N_particles*jnp.ones(config.N_particles)

def ic_two_body(mass1: float, mass2: float, rp: float, e: float, config, params):
    """
    Create initial conditions for a two-body system.
    
    By default, the two bodies will be placed along the x-axis at the
    closest distance rp. Depending on the input eccentricity, the two 
    bodies can be in a circular (e < 1), parabolic (e = 1), or hyperbolic 
    orbit (e > 1).

    Parameters
    ----------
    mass1 : float
        Mass of the first body [nbody units].
    mass2 : float
        Mass of the second body [nbody units].
    rp : float
        Closest orbital distance [nbody units].
    e : float
        Eccentricity.
    config : NamedTuple
        Configuration NamedTuple.
    params : NamedTuple
        Parameters NamedTuple.

    Returns
    -------
    pos : jnp.ndarray
        Positions of the particles.
    vel : jnp.ndarray
        Velocities of the particles.
    mass : jnp.ndarray
        Masses of the particles.
    """

    Mtot=mass1+mass2

    if e==1.:
        vrel=jnp.sqrt(params.G * 2*Mtot/rp)
    else:
        a=rp/(1-e)
        vrel=jnp.sqrt(params.G * Mtot*(2./rp-1./a))

    v1 = -params.G*mass2/Mtot * vrel
    v2 = params.G*mass1/Mtot * vrel

    pos  = jnp.array([[0.,0.,0.],[rp,0.,0.]])
    vel  = jnp.array([[0.,v1,0.],[0.,v2,0.]])
    mass = jnp.array([mass1, mass2])

    return pos, vel, mass
    
