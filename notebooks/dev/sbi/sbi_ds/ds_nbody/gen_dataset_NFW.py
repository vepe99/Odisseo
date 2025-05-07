# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"
from autocvd import autocvd
autocvd(num_gpus = 1)

import time

from typing import NamedTuple

import numpyro 
import numpyro.distributions as dist

from numpyro.handlers import condition, reparam, seed, trace
from numpyro.infer.reparam import LocScaleReparam, TransformReparam

from math import pi

from typing import Optional, Tuple, Callable, Union, List
from functools import partial

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import vmap, jit, pmap
from jax import random

# jax.config.update("jax_enable_x64", True)

import numpy as np
from astropy import units as u
from astropy import constants as c

import odisseo
from odisseo import construct_initial_state
from odisseo.integrators import leapfrog
from odisseo.dynamics import direct_acc, DIRECT_ACC, DIRECT_ACC_LAXMAP, DIRECT_ACC_FOR_LOOP, DIRECT_ACC_MATRIX
from odisseo.option_classes import SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams, MN_POTENTIAL, NFW_POTENTIAL
from odisseo.initial_condition import Plummer_sphere, ic_two_body, sample_position_on_sphere, inclined_circular_velocity, sample_position_on_circle, inclined_position
from odisseo.utils import center_of_mass
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.visualization import create_3d_gif, create_projection_gif, energy_angular_momentum_plot
from odisseo.potentials import MyamotoNagai, NFW
from odisseo.option_classes import DIFFRAX_BACKEND, DOPRI5, TSIT5, SEMIIMPLICITEULER, LEAPFROGMIDPOINT, REVERSIBLEHEUN


plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 15,
})


from jax.scipy.integrate import trapezoid
from galpy.potential import MiyamotoNagaiPotential



@jit
def mass_enclosed_NFW(R, params):
    """
    Compute the mass of the NFW potential at a given radius R.

    ref: wikipedia
    """
    c = params.NFW_params.c
    Mvir = params.NFW_params.Mvir
    r_s = params.NFW_params.r_s
    rho_0 = (Mvir / (4*jnp.pi * r_s**3)) * (jnp.log(1+c) - c/(1+c))**-1

    return 4*jnp.pi*rho_0*r_s**3 * (jnp.log(1 + R/r_s) - R/(r_s + R))


@partial(jit, static_argnames=("code_units",))
def to_observable(state, code_units):

    """
    Convert the state vector to observable quantities.
    The return is a 6D vector with the following elements:
    - parallax
    - b (galactic latitude)
    - l (galactic longitude)
    - v_r (radial velocity)
    - mu_b (proper motion in b direction)
    - mu_l (proper motion in l direction)
    """
    X = state[:, 0] * code_units.code_length.to(u.kpc)
    X_sun = X.at[:, 0].set(8. - X[:, 0] )
    r = jnp.linalg.norm(X_sun, axis=1)
    parallax = 1 / r
    b = jnp.arcsin(X_sun[:, 2] / r)
    l = jnp.arctan2(X_sun[:, 1], X_sun[:, 0])

    state = state.at[:, 1].set(state[:, 1]* code_units.code_velocity.to(u.km/u.s)) 
    v_x, v_y, v_z = state[:, 1, 0], state[:, 1, 1], state[:, 1, 2]
    v_l = -v_x * jnp.sin(l) + v_y * jnp.cos(l)
    v_b = -v_x * jnp.cos(l) * jnp.sin(b) - v_y * jnp.sin(l) * jnp.sin(b) + v_z * jnp.cos(b)
    v_r = v_x * jnp.cos(l) * jnp.cos(b) + v_y * jnp.sin(l) * jnp.cos(b) + v_z * jnp.sin(b)
    mu_l = v_l / (4.74047 * r)
    mu_b = v_b / (4.74047 * r)
    return jnp.stack([parallax, b, l, v_r, mu_b, mu_l], axis=1 )

code_length = 10.0 * u.kpc
code_mass = 1e4 * u.Msun
G = 1 
code_units = CodeUnits(code_length, code_mass, G=G)


def run_simulation(key,
                   sigma: jnp.ndarray,
                   with_noise: bool = True,
                   ):

    #config param, this cannot be differentiate 
    config = SimulationConfig(N_particles = 10_000, 
                          return_snapshots = False, 
                          num_timesteps = 1000, 
                          external_accelerations=(NFW_POTENTIAL, ), 
                          acceleration_scheme = DIRECT_ACC_MATRIX,
                          softening = (0.1 * u.kpc).to(code_units.code_length).value) #default values
    
    # simulation parameters to be sampled 
    t_end = numpyro.sample("t_end", dist.Uniform(0.500, 10.0))
    Mtot_plummer = numpyro.sample("Mtot_plummer", dist.Uniform(1e3, 1e5))
    a_plummer = numpyro.sample("a_plummer", dist.Uniform(0.1, 2.0))
    Mtot_NFW = numpyro.sample("M_NFW", dist.Uniform(5e11, 1.5e12))
    r_s = numpyro.sample("r_s", dist.Uniform(1, 20.0))

    params = SimulationParams(t_end = t_end * u.Myr.to(code_units.code_time),  
                          Plummer_params= PlummerParams(Mtot=Mtot_plummer * u.Msun.to(code_units.code_mass),
                                                        a=a_plummer * u.kpc.to(code_units.code_length)),
                          NFW_params= NFWParams(Mvir=Mtot_NFW * u.Msun.to(code_units.code_mass),
                                               r_s= r_s * u.kpc.to(code_units.code_length),
                                               c = 8.0),                           
                          G=G, ) 
    
    # initial conditions
    #set up the particles in the initial state
    positions, velocities, mass = Plummer_sphere(key=key, params=params, config=config)

    #put the Plummer sphere in a ciruclar orbit around the NFW halo
    ra = 200*u.kpc.to(code_units.code_length)
    e = numpyro.sample("e", dist.Uniform(0.0, 0.7)) #nuance parameter 
    rp = (1-e)/(1+e) * ra
    # sample the position of the center of mass
    # Sample phi uniformly in [0, 2π]
    phi = numpyro.sample("phi", dist.Uniform(0, 2*pi)) #nuance parameter
    
    # Sample cos(theta) uniformly in [-1, 1] to ensure uniform distribution on the sphere
    costheta = numpyro.sample("costheta", dist.Uniform(-1, 1)) #nuance parameter
    theta = jnp.arccos(costheta)  # Convert to theta
    
    # Convert to Cartesian coordinates
    x = rp * jnp.sin(theta) * jnp.cos(phi)
    y = rp * jnp.sin(theta) * jnp.sin(phi)
    z = rp * jnp.cos(theta)

    pos_com = jnp.stack([x, y, z], axis=-1)

    inclination = jnp.pi/2 - jnp.acos(z/rp)

    # mass1 = mass_enclosed_MN(rp, z, params, code_units) + mass_enclosed_NFW(rp, params)
    mass1 = mass_enclosed_NFW(rp, params)
    mass2 = params.Plummer_params.Mtot 
    _, bulk_velocity, _ = ic_two_body(mass1=mass1,
                                    mass2=mass2,
                                    rp=rp,
                                    e=e,
                                    params=params)
    bulk_velocity_modulus = bulk_velocity[1, 1].reshape((1))
    vel_com = inclined_circular_velocity(pos_com, bulk_velocity_modulus, inclination)

    # Add the center of mass position and velocity to the Plummer sphere particles
    positions = positions + pos_com
    velocities = velocities + vel_com

    #initialize the initial state
    initial_state = construct_initial_state(positions, velocities)

    #time integration
    final_state = time_integration(primitive_state=initial_state, mass=mass,  params=params, config=config)

    #to observable space (parallax, b, l, v_r, mu_b, mu_l), shape (config.N_particles, 6)
    final_state = to_observable(final_state, code_units)

    if with_noise is True:
    #   separate the observables
        parallax = numpyro.sample("parallax", dist.Normal(final_state[:, 0], sigma[0]))
        b = numpyro.sample("b", dist.Normal(final_state[:, 1], sigma[1]))

        cos_b = jnp.cos(final_state[:, 1])
        l = numpyro.sample("l", dist.Normal(final_state[:, 2], sigma[2]/cos_b))
        v_r = numpyro.sample("v_r", dist.Normal(final_state[:, 3], sigma[3]))
        mu_b = numpyro.sample("mu_b", dist.Normal(final_state[:, 4], sigma[4]))
        mu_l = numpyro.sample("mu_l", dist.Normal(final_state[:, 5], sigma[5]/cos_b))

    else:
        x = numpyro.deterministic("y", final_state)

def get_samples_and_scores(
    model,
    key,
    batch_size=1,
    score_type="conditional",
    distribute = False,
    thetas=None,
    with_noise=True,
):
    """Handling function sampling and computing the score from the model.

    Parameters
    ----------
    model : numpyro model
    key : PRNG Key
    batch_size : int, optional
        size of the batch to sample, by default 64
    score_type : str, optional
        'density' for nabla_theta log p(theta | y, z) or
        'conditional' for nabla_theta log p(y | z, theta), by default 'conditional'
    thetas : Array (batch_size, 2), optional
        thetas used to sample simulations or
        'None' sample thetas from the model, by default None
    with_noise : bool, optional
        add noise in simulations, by default True
        note: if no noise the score is only nabla_theta log p(theta, z)
        and log_prob log p(theta, z)

    Returns
    -------
    Array
        (log_prob, sample), score
    """

    params_name = ["t_end", "Mtot_plummer", "a_plummer", "M_NFW", "r_s"]

    def log_prob_fn(theta, key):
        cond_model = seed(model, key)
        cond_model = condition(
            cond_model,
            {
                "t_end": theta[0],
                "Mtot_plummer": theta[1],
                "a_plummer": theta[2],
                "M_NFW": theta[3],
                "r_s": theta[4],
            },
        )
        model_trace = trace(cond_model).get_trace()


        sample = {
            "theta": jnp.stack(
                [model_trace[name]["value"] for name in params_name], axis=-1
            ),
            "y": jnp.stack( [model_trace["parallax"]["value"], 
                             model_trace["b"]["value"],
                             model_trace["l"]["value"],
                             model_trace["v_r"]["value"],
                             model_trace["mu_b"]["value"],
                             model_trace["mu_l"]["value"]], axis=1)
        }

        if score_type == "density":
            #this is for npe
            logp = 0
            for name in params_name:
                logp += model_trace[name]["fn"].log_prob(model_trace[name]["value"])

        elif score_type == "conditional":
            #this is for nle
            logp = 0

        if with_noise:
            logp_ = 0.
            for name in ["parallax", "b", "l", "v_r", "mu_b", "mu_l"]:
                logp_ += (
                    model_trace[name]["fn"]
                    .log_prob(jax.lax.stop_gradient(model_trace[name]["value"]))
                    .sum()
                ) #take the product over the observables 
            
            #add to the total likelihood
            logp += logp_
            
        for name in ["e", "costheta", "phi"]:
            logp += model_trace[name]["fn"].log_prob(model_trace[name]["value"]).sum()

        return logp, sample

    # Split the key by batch
    keys = jax.random.split(key, batch_size)

    if distribute and jax.device_count() ==2:
        # Distribute the keys across devices
        devices = jax.devices()
        shards = [keys[0], keys[1] ]
        keys = jax.device_put_sharded(shards, devices=jax.devices())

    # Sample theta from the model
    if thetas is None:

        @jax.vmap
        def get_params(key):
            with jax.checking_leaks():
                model_trace = trace(seed(model, key)).get_trace()
                thetas = jnp.stack(
                    [model_trace[name]["value"] for name in params_name], axis=-1
                )
                return thetas

        
        thetas = get_params(keys)

    return jax.vmap(jax.value_and_grad(log_prob_fn, has_aux=True))(thetas, keys)


model = partial(
    run_simulation,
    key = random.PRNGKey(0),
    sigma=jnp.ones((6,)) * 0.1,
    with_noise=True,
)

print('Beginning sampling...')
start_time = time.time()
batch_size = 1
num_chunks = 100_000
name_str = 90_000
for i in range(name_str, num_chunks, batch_size):
    (log_prob, sample), score = get_samples_and_scores(
                                    model,
                                    batch_size=batch_size,
                                    key=random.PRNGKey(i),   
                                )
    for j in range(batch_size):
        # Save the samples and scores
        np.savez_compressed(f"./data/data_NFW/chunk_{name_str:06d}.npz",
                             theta=sample["theta"][j],
                               x=sample["y"][j],
                                 score=score[j])
        name_str += 1
        print('chunk', name_str-1)
end_time = time.time()
print("Time taken to sample in seconds:", end_time - start_time)