from autocvd import autocvd
autocvd(num_gpus = 1)

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # Set to the 0 for tmux 6
import time

import jax 
import jax.numpy as jnp
from jax import jit, random
import equinox as eqx
from jax.sharding import Mesh, PartitionSpec, NamedSharding
jax.config.update("jax_enable_x64", True)

from tqdm import tqdm
import numpy as np
from astropy import units as u

from odisseo import construct_initial_state
from odisseo.dynamics import DIRECT_ACC_MATRIX
from odisseo.option_classes import SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams, PSPParams
from odisseo.option_classes import MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL
from odisseo.initial_condition import Plummer_sphere
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.utils import projection_on_GD1

#import flowjax, use for the loss function
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data

#optimization
from jaxopt import ScipyBoundedMinimize, LBFGS
import optax

from chainconsumer import Chain, ChainConsumer, Truth, make_sample
import pandas as pd


import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 15,
})


code_length = 10 * u.kpc
code_mass = 1e4 * u.Msun
code_time = 3 * u.Gyr
code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  

#set the config, we cannot differentiate with respect to the config
config = SimulationConfig(N_particles = 5_000, 
                          return_snapshots = False, 
                          num_timesteps = 1000, 
                          external_accelerations=(NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL), 
                          acceleration_scheme = DIRECT_ACC_MATRIX,
                          softening = (0.1 * u.pc).to(code_units.code_length).value,) #default values

# set the simulation parameters, we can differentiate with respect to these parameters
params = SimulationParams(t_end = (3 * u.Gyr).to(code_units.code_time).value,  
                          Plummer_params= PlummerParams(Mtot=(10**4.05 * u.Msun).to(code_units.code_mass).value,
                                                        a=(8 * u.pc).to(code_units.code_length).value),
                           MN_params= MNParams(M = (68_193_902_782.346756 * u.Msun).to(code_units.code_mass).value,
                                              a = (3.0 * u.kpc).to(code_units.code_length).value,
                                              b = (0.280 * u.kpc).to(code_units.code_length).value),
                          NFW_params= NFWParams(Mvir=(4.3683325e11 * u.Msun).to(code_units.code_mass).value,
                                               r_s= (16.0 * u.kpc).to(code_units.code_length).value,),      
                          PSP_params= PSPParams(M = 4501365375.06545 * u.Msun.to(code_units.code_mass),
                                                alpha = 1.8, 
                                                r_c = (1.9*u.kpc).to(code_units.code_length).value),                    
                          G=code_units.G, ) 

#the center of mass has the same config and params as the main simulation but it needs to be integrated backwards in time 
config_com = config._replace(N_particles=1,)
params_com = params._replace(t_end=-params.t_end,)

#random key for JAX
key = random.PRNGKey(0)
key_Plummer_true, key_selection_true, key_background_selection, key_background_true, key_noise_true, key_flow = random.split(key, 6)

#Final position and velocity of the center of mass
pos_com_final = jnp.array([[11.8, 0.79, 6.4]]) * u.kpc.to(code_units.code_length)
vel_com_final = jnp.array([[109.5,-254.5,-90.3]]) * (u.km/u.s).to(code_units.code_velocity)
mass_com = jnp.array([params.Plummer_params.Mtot]) 

#we construmt the initial state of the com 
initial_state_com = construct_initial_state(pos_com_final, vel_com_final,)
#we run the simulation backwards in time for the center of mass
final_state_com = time_integration(initial_state_com, mass_com, config=config_com, params=params_com)
#we calculate the final position and velocity of the center of mass
pos_com = final_state_com[:, 0]
vel_com = final_state_com[:, 1]

#we construct the initial state of the Plummer sphere
positions, velocities, mass = Plummer_sphere(key=key_Plummer_true, params=params, config=config)
#we add the center of mass position and velocity to the Plummer sphere particles
positions = positions + pos_com
velocities = velocities + vel_com
#initialize the initial state
initial_state_stream = construct_initial_state(positions, velocities, )
#run the simulation
final_state = time_integration(initial_state_stream, mass, config=config, params=params)

#projection on the GD1 stream
stream = projection_on_GD1(final_state, code_units=code_units,)


#Bimodal sampling
@jit
def selection_function(stream_star, p, key):
    # Apply selection criteria
    return jnp.where(jax.random.uniform(key=key, shape=1) < p, stream_star, jnp.nan)

# Generate a random background star position
@jit 
def background_assignement(key):
    return jax.random.uniform(key=key, shape=(6,),
                              minval=jnp.array([6, -120, -8, -250, -2., -0.10]),
                             maxval=jnp.array([20, 70, 2, 250, 1.0, 0.10]))

# Select stars from the stream based on Bimodal sampling
keys = random.split(key_selection_true, stream.shape[0])
p = jnp.ones(shape=(stream.shape[0]))* 0.95
selected_stream = jax.vmap(selection_function, )(stream, p, keys)

# Nbackground star contamination
N_background = int(1e6)
#Generate the probability of selectin a background star
background_selected_probability = jnp.where(jax.random.uniform(key=key_background_selection, shape=(N_background,)) < 1e-3, 1.0, 0.0)
keys = random.split(key_background_true, N_background)
selected_background = jax.vmap(lambda key, background_star_probability: jnp.where(background_star_probability, background_assignement(key), jnp.nan))(keys, background_selected_probability)

# Combine the selected stream and background stars
stream = jnp.concatenate((selected_stream, selected_background), axis=0)

#add gaussian noise to the stream, same as in Albatross paper (https://arxiv.org/pdf/2304.02032)
noise_std = jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0])
stream = stream + jax.random.normal(key=key_noise_true, shape=stream.shape) * noise_std

stream_mean = jnp.nanmean(stream, axis=0)
stream_std = jnp.nanstd(stream, axis=0)
stream = (stream - stream_mean) / stream_std  # Standardize the data

stream_target = stream[~jnp.isnan(stream)].reshape(-1, 6)  # Flatten the stream for training


# for now we will only use the last snapshot to caluclate the loss and the gradient
config =  config._replace(return_snapshots=False,)
config_com = config_com._replace(return_snapshots=False,)

@jit
def rbf_kernel(x, y, sigma):
    """RBF kernel optimized for 6D astronomical data"""
    return jnp.exp(-jnp.sum((x - y)**2) / (2 * sigma**2))


@jit
def time_integration_fix_position_grad(t_end, 
                                       M_plummer,
                                       a_plummer,
                                       M_NFW,
                                       r_s_NFW,
                                       M_MN,
                                       a_MN,
                                       key):

    #Creation of the Plummer sphere requires a key 
    # key = random.PRNGKey(key)
    key_Plummer, key_selection, key_background, key_noise = random.split(key, 4)
    
    #we set up the parameters of the simulations, changing only the parameter that we want to optimize
    #parameters of the stream
    new_params = params._replace(
                t_end = t_end,
                Plummer_params=params.Plummer_params._replace(
                    Mtot=10**M_plummer,
                    a=a_plummer
                ),
                NFW_params=params.NFW_params._replace(
                    Mvir=10**M_NFW,
                    r_s=r_s_NFW
                ),
                MN_params=params.MN_params._replace(
                    M=10**M_MN,
                    a=a_MN
                ))
    #parameters of the center of mass
    #we set the t_end to be negative, so we run the simulation backwards in time
    new_params_com = new_params._replace(t_end=-t_end,)

    #Final position and velocity of the center of mass
    pos_com_final = jnp.array([[11.8, 0.79, 6.4]]) * u.kpc.to(code_units.code_length)
    vel_com_final = jnp.array([[109.5,-254.5,-90.3]]) * (u.km/u.s).to(code_units.code_velocity)
    mass_com = jnp.array([params.Plummer_params.Mtot]) 
    
    #we construmt the initial state of the com 
    initial_state_com = construct_initial_state(pos_com_final, vel_com_final,)
    #we run the simulation backwards in time for the center of mass
    final_state_com = time_integration(initial_state_com, mass_com, config=config_com, params=new_params_com)
    #we calculate the final position and velocity of the center of mass
    pos_com = final_state_com[:, 0]
    vel_com = final_state_com[:, 1]

    #we construct the initial state of the Plummer sphere
    positions, velocities, mass = Plummer_sphere(key=key_Plummer, params=new_params, config=config)
    #we add the center of mass position and velocity to the Plummer sphere particles
    positions = positions + pos_com
    velocities = velocities + vel_com
    #initialize the initial state
    initial_state_stream = construct_initial_state(positions, velocities, )
    #run the simulation
    final_state = time_integration(initial_state_stream, mass, config=config, params=new_params)

    #projection on the GD1 stream
    stream = projection_on_GD1(final_state, code_units=code_units,)

    #Stream selection success
    keys_selection = random.split(key_selection, stream.shape[0])
    p = jnp.ones(shape=(stream.shape[0]))* 0.95
    # selected_stream = jax.vmap(selection_function, )(stream, p, keys_selection)
    selected_stream = stream

    # #background contamination
    N_background = int(1e6)
    # #Generate the probability of selectin a background star, this is computationally expensive, so we just add 1_000 background stars
    # background_selected = jnp.where(jax.random.uniform(key=key_background, shape=(N_background,)) < 1e-3, 1.0, 0.0)
    # keys_background = random.split(key_background, N_background)
    # selected_background = jax.vmap(lambda key, background_star: jnp.where(background_star, background_assignement(key), jnp.nan))(keys_background, background_selected)
    N_background = int(N_background * 1e-3)
    selected_background = jax.vmap(background_assignement, )(random.split(key=key_background, num=N_background))

    # Combine the selected stream and background stars
    stream = jnp.concatenate((selected_stream, selected_background), axis=0)

    #add gaussian noise to the stream
    noise_std = jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0])
    stream = stream + jax.random.normal(key=key_noise, shape=stream.shape) * noise_std
    #we calculate the loss as the negative log likelihood of the stream
    bounds = jnp.array([
        [6, 20],        # R [kpc]
        [-120, 70],     # phi1 [deg]  
        [-8, 2],        # phi2 [deg]
        [-250, 250],    # vR [km/s]
        [-2., 1.0],     # v1_cosphi2 [mas/yr]
        [-0.10, 0.10]   # v2 [mas/yr]
    ])
        
    def normalize_stream(stream):
        # Normalize each dimension to [0,1]
        return (stream - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    
    sim_norm = normalize_stream(stream)
    target_norm = normalize_stream(stream_target)
    
    # Adaptive bandwidth for 6D data
    n_sim, n_target = len(stream), len(stream_target)
    sigma = 0.5 * jnp.power(n_sim + n_target, -1/(6+4))  # Rule of thumb for 6D
    
    # # Compute MMD terms
    # xx = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda xj: rbf_kernel(xi, xj, sigma))(sim_norm))(sim_norm))
    # yy = jnp.mean(jax.vmap(lambda yi: jax.vmap(lambda yj: rbf_kernel(yi, yj, sigma))(target_norm))(target_norm))
    # xy = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda yj: rbf_kernel(xi, yj, sigma))(target_norm))(sim_norm))
    
    # return xx + yy - 2 * xy

    @jit 
    def compute_mmd(sim_norm, target_norm, sigmas):
        xx = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda xj: rbf_kernel(xi, xj, sigmas))(sim_norm))(sim_norm))
        yy = jnp.mean(jax.vmap(lambda yi: jax.vmap(lambda yj: rbf_kernel(yi, yj, sigmas))(target_norm))(target_norm))
        xy = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda yj: rbf_kernel(xi, yj, sigmas))(target_norm))(sim_norm))
        return xx + yy - 2 * xy

    distances = jax.vmap(lambda x: jax.vmap(lambda y: jnp.linalg.norm(x - y))(target_norm))(sim_norm)
    distance_flat = distances.flatten()

    # # Use percentiles as natural scales
    sigmas = jnp.array([
        jnp.percentile(distance_flat, 10),   # Fine scale
        jnp.percentile(distance_flat, 25),   # Small scale  
        jnp.percentile(distance_flat, 50),   # Medium scale (median)
        jnp.percentile(distance_flat, 75),   # Large scale
        jnp.percentile(distance_flat, 90),   # Very large scale
    ])

    # Adaptive weights based on scale separation
    # scale_weights = jnp.array([0.15, 0.2, 0.3, 0.25, 0.1])
    scale_weights = jnp.ones_like(sigmas)  # Equal weights for simplicity

    # Compute MMD with multiple kernels
    mmd_total = jnp.sum(scale_weights * jax.vmap(lambda sigma: compute_mmd(sim_norm, target_norm, sigma))(sigmas))
    
    return mmd_total / len(sigmas)


@jit
def time_integration_fix_position_grad_ScipyMinimize(param, key):
    t_end, M_plummer, a_plummer, Mvir, r_s_NFW, M_MN, a_MN = 10**param
    return time_integration_fix_position_grad(t_end, 
                                              M_plummer,
                                              a_plummer,
                                              Mvir,
                                              r_s_NFW,
                                              M_MN,
                                              a_MN,
                                              key)

optimizer = ScipyBoundedMinimize(
     method="l-bfgs-b", 
     dtype=jnp.float64,
     fun=time_integration_fix_position_grad_ScipyMinimize, 
     tol=1e-8, 
    )


# key = random.PRNGKey(42) #compgpu8 tmux 0
# key = random.PRNGKey(43) #compgpu8 tmux 1
key = random.PRNGKey(44) #compgpu8 tmux 2
# key = random.PRNGKey(45) #compgpu8 tmux 1

parameter_value = jax.random.uniform(key=key, 
                                    shape=(1000, 7), 
                                    minval=jnp.array([np.log10(0.5 * u.Gyr.to(code_units.code_time)).item(), # t_end in Gyr
                                                    np.log10(10**3.0 * u.Msun.to(code_units.code_mass)).item(), # Plummer mass
                                                    np.log10(params.Plummer_params.a*(1/4)).item(),
                                                    np.log10(np.log10(params.NFW_params.Mvir*(1/4)).item()),
                                                    np.log10(params.NFW_params.r_s*(1/4)).item(), 
                                                    np.log10(params.MN_params.M*(1/4)).item(), 
                                                    np.log10(params.MN_params.a*(1/4)).item(),]), 
                                                    
                                    maxval=jnp.array([np.log10(5 * u.Gyr.to(code_units.code_time)).item(), # t_end in Gyr
                                                    np.log10(10**4.5 * u.Msun.to(code_units.code_mass)).item(), #Plummer mass
                                                    np.log10(params.Plummer_params.a*(8/4)).item(),
                                                    np.log10(params.NFW_params.Mvir*(8/4)).item(), 
                                                    np.log10(params.NFW_params.r_s*(8/4)).item(), 
                                                    np.log10(params.MN_params.M*(8/4)).item(), 
                                                    np.log10(params.MN_params.a*(8/4)).item(),])) 
print('Start sampling with ScipyMinimize')
start_time = time.time()
i = 1000
for p, k in tqdm(zip(parameter_value, random.split(key, parameter_value.shape[0]) ) ):
    sol = optimizer.run(init_params=p, 
                        key=k,
                        bounds = jnp.array([[np.log10(0.5 * u.Gyr.to(code_units.code_time)).item(), 
                                     np.log10(10**3.0 * u.Msun.to(code_units.code_mass)).item(), 
                                     np.log10(params.Plummer_params.a*(1/4)).item(),
                                     np.log10(params.NFW_params.Mvir*(1/4)).item(),
                                     np.log10(params.NFW_params.r_s*(1/4)).item(), 
                                     np.log10(params.MN_params.M*(1/4)).item(), 
                                     np.log10(params.MN_params.a*(1/4)).item()],
                                    [np.log10(5 * u.Gyr.to(code_units.code_time)).item(), 
                                     np.log10(10**4.5 * u.Msun.to(code_units.code_mass)).item(), 
                                     np.log10(params.Plummer_params.a*(8/4)).item(),
                                     np.log10(params.NFW_params.Mvir*(8/4)).item(), 
                                     np.log10(params.NFW_params.r_s*(8/4)).item(), 
                                     np.log10(params.MN_params.M*(8/4)).item(), 
                                     np.log10(params.MN_params.a*(8/4)).item()]]))
    
    np.savez(f'./sampling_ScipyMinimize/ScipyBoundedMinimize/new_loss/sample_{i}.npz', 
             sample=np.array(sol.params),
             loss=np.array(sol.state.fun_val), )
    i += 1

end_time = time.time()
print("Time taken to sample in seconds:", end_time - start_time)