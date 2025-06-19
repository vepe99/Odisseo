
from autocvd import autocvd
autocvd(num_gpus = 1)
# import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # Set the GPU to use, change as needed
from tqdm import tqdm
import jax 
import jax.numpy as jnp
from jax import jit, random
import equinox as eqx
from jax.sharding import Mesh, PartitionSpec, NamedSharding
jax.config.update("jax_enable_x64", True)


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
from jaxopt import ScipyMinimize, LBFGS
import optax
import optimistix as optx

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
code_mass = 1e5 * u.Msun
code_time = 3 * u.Gyr
code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  

#set the config, we cannot differentiate with respect to the config
config = SimulationConfig(N_particles = 1_000, 
                          return_snapshots = False, 
                          num_timesteps = 500, 
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
key = random.PRNGKey(1)
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
positions, velocities, mass = Plummer_sphere(key=key, params=params, config=config)
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
key = random.PRNGKey(42)
keys = random.split(key, stream.shape[0])
p = jnp.ones(shape=(stream.shape[0]))* 0.95
selected_stream = jax.vmap(selection_function, )(stream, p, keys)

# Nbackground star contamination
N_background = int(1e6)
#Generate the probability of selectin a background star
background_selected_probability = jnp.where(jax.random.uniform(key=key, shape=(N_background,)) < 1e-3, 1.0, 0.0)
keys = random.split(key, N_background)
selected_background = jax.vmap(lambda key, background_star_probability: jnp.where(background_star_probability, background_assignement(key), jnp.nan))(keys, background_selected_probability)

# Combine the selected stream and background stars
stream = jnp.concatenate((selected_stream, selected_background), axis=0)

#add gaussian noise to the stream, same as in Albatross paper (https://arxiv.org/pdf/2304.02032)
noise_std = jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0])
stream = stream + jax.random.normal(key=jax.random.key(0), shape=stream.shape) * noise_std

stream_mean = jnp.nanmean(stream, axis=0)
stream_std = jnp.nanstd(stream, axis=0)
stream = (stream - stream_mean) / stream_std  # Standardize the data

stream_target = stream[~jnp.isnan(stream)].reshape(-1, 6)  # Remove NaN values for training

# Assign a new out of the observational windows value to the NaN values, and use constrain support NF, not implemented yet
# stream = jax.vmap(lambda stream_star: jnp.where(jnp.isnan(stream_star), jnp.ones((6))*100, stream_star))(stream)


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
                    Mtot=M_plummer,
                    a=a_plummer
                ),
                NFW_params=params.NFW_params._replace(
                    Mvir=M_NFW,
                    r_s=r_s_NFW
                ),
                MN_params=params.MN_params._replace(
                    M=M_MN,
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
    selected_stream = stream

    # #background contamination
    N_background = int(1e6)
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
    # n_sim, n_target = len(stream), len(stream_target)
    # sigma = 0.5 * jnp.power(n_sim + n_target, -1/(6+4))  # Rule of thumb for 6D

    #use the median of the pairwise distance as the bandwidth
    @jit
    def compute_median_distance(X, Y):
        """Compute median pairwise distance between datasets"""
        # Sample subset for efficiency (important for large datasets)

        X_sample = X
        Y_sample = Y
        
        # Compute all pairwise distances
        distances = jax.vmap(lambda x: jax.vmap(lambda y: jnp.linalg.norm(x - y))(Y_sample))(X_sample)
        
        # Return median distance
        return jnp.median(distances.flatten())
    
    @jit 
    def compute_mmd(sim_norm, target_norm, sigmas):
        xx = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda xj: rbf_kernel(xi, xj, sigmas))(sim_norm))(sim_norm))
        yy = jnp.mean(jax.vmap(lambda yi: jax.vmap(lambda yj: rbf_kernel(yi, yj, sigmas))(target_norm))(target_norm))
        xy = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda yj: rbf_kernel(xi, yj, sigmas))(target_norm))(sim_norm))
        return xx + yy - 2 * xy
    
    # Method 1: Median multiscale
    # median_dist = compute_median_distance(sim_norm, target_norm)
    # sigma = median_dist / jnp.sqrt(2)  # Common scaling factor
    # sigmas = jnp.array([sigma/4, sigma/2, sigma, sigma*2, sigma*4]) 
    

    distances = jax.vmap(lambda x: jax.vmap(lambda y: jnp.linalg.norm(x - y))(target_norm))(sim_norm)
    distance_flat = distances.flatten()

    # Use percentiles as natural scales
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
    



# Define parameter bounds
# Define parameter bounds in code units
param_bounds = jnp.array([
    [jnp.log10((0.5 * u.Gyr).to(code_units.code_time).value), 
     jnp.log10((5.0 * u.Gyr).to(code_units.code_time).value)],           # t_end (0.5-5 Gyr)
    
    [jnp.log10((10**3.0 * u.Msun).to(code_units.code_mass).value), 
     jnp.log10((10**4.5 * u.Msun).to(code_units.code_mass).value)],      # log10(M_plummer) (10^3 - 10^4.5 Msun)
    
    [jnp.log10(params.Plummer_params.a*0.25), 
     jnp.log10(params.Plummer_params.a*2)],                              # log10(a_plummer) - already in code units
    
    [jnp.log10((params.NFW_params.Mvir*0.25 * u.Msun).to(code_units.code_mass).value), 
     jnp.log10((params.NFW_params.Mvir*2 * u.Msun).to(code_units.code_mass).value)],         # log10(M_NFW) (10^10 - 10^12 Msun)
    
    [jnp.log10((params.NFW_params.r_s * 0.25 * u.kpc).to(code_units.code_length).value), 
     jnp.log10((params.NFW_params.r_s * 2 * u.kpc).to(code_units.code_length).value)],        # log10(r_s_NFW) (5-50 kpc)
    
    [jnp.log10((params.MN_params.M * 0.25 * u.Msun).to(code_units.code_mass).value), 
     jnp.log10((params.MN_params.M * 2 * u.Msun).to(code_units.code_mass).value)],         # log10(M_MN)
    
    [jnp.log10((params.MN_params.a * 0.25 * u.kpc).to(code_units.code_length).value), 
     jnp.log10((params.MN_params.a * 2 * u.kpc).to(code_units.code_length).value)],        # log10(a_MN) (1-10 kpc)
])

def tanh_transform(unbounded_params, bounds):
    """
    Transform unbounded parameters to bounded log-space parameters using tanh
    Better than sigmoid: more symmetric, better gradients, more stable
    """
    lower, upper = bounds[:, 0], bounds[:, 1]
    # tanh maps (-∞, +∞) to (-1, +1), then we scale to (lower, upper)
    return lower + (upper - lower) * (jnp.tanh(unbounded_params) + 1) / 2

def inverse_tanh_transform(bounded_log_params, bounds):
    """
    Convert bounded log-space parameters back to unbounded space
    Much more stable than inverse sigmoid
    """
    lower, upper = bounds[:, 0], bounds[:, 1]
    # Normalize to (-1, 1)
    normalized = 2 * (bounded_log_params - lower) / (upper - lower) - 1
    # Clamp to avoid numerical issues (tanh is more forgiving)
    normalized = jnp.clip(normalized, -0.999, 0.999)
    return jnp.arctanh(normalized)

@jit
def time_integration_fix_position_grad_ScipyMinimize(param, key):
    # param = inverse_tanh_transform(param, param_bounds)
    # param = sigmoid_transform(param, param_bounds)
    t_end, M_plummer, a_plummer, Mvir, r_s_NFW, M_MN, a_MN = 10**param
    # t_end = 10**t_end
    # M_plummer = 10**M_plummer
    # a_plummer = 10**a_plummer
    # Mvir = 10**Mvir
    # r_s_NFW = 10**r_s_NFW
    # M_MN = 10**M_MN
    # a_MN = 10**a_MN
    
    return time_integration_fix_position_grad(t_end, 
                                              M_plummer,
                                              a_plummer,
                                              Mvir,
                                              r_s_NFW,
                                              M_MN,
                                              a_MN,
                                              key)


# We pick gradient descent for pedagogical and visualization reasons.
# In practice one would use e.g. Levenberg-Marquardt from the
# optimistix package.

# from functools import partial
# @partial(jit, static_argnames=('func','learning_rate', 'tol', 'max_iter'))
def gradient_descent_optimization(func, x_init, key, learning_rate=20, max_iter=2000):
    # xlist = jnp.zeros((max_iter + 1, x_init.shape[0]))
    xlist = []
    x = x_init
    loss_list = []
    # loss_list = jnp.zeros(max_iter + 1)

    xlist.append(x)
    # xlist = xlist.at[0].set(x)  # Initialize the first element with x_init

    # ADAM optimizer
    # optimizer = optax.adam(learning_rate=learning_rate)
    # optimizer_state = optimizer.init(x)

    ##SCHEDULE FREE ADAM
    # learning_rate_fn = optax.warmup_constant_schedule(peak_value=learning_rate, warmup_steps=10, init_value=0.0)
    # optimizer = optax.adam(learning_rate_fn, b1=0.)
    # optimizer = optax.contrib.schedule_free(optimizer, learning_rate_fn, b1=0.9)
    # optimizer_state = optimizer.init(x)

    #SCHEDULE FREE ADAMW
    optimizer = optax.contrib.schedule_free_adamw(learning_rate)
    optimizer_state = optimizer.init(x)


    for _ in range(max_iter):
        # Compute the function value and its gradient
        loss, f_grad = jax.value_and_grad(func)(x, key)
        loss_list.append(loss)
        # loss_list = loss_list.at[_].set(loss)
        
        # Update the parameter
        updates, optimizer_state = optimizer.update(f_grad, optimizer_state, x)
        x = optax.apply_updates(x, updates)
        key = random.split(key, 1)[0]  # Update the key for the next iteration
        xlist.append(x)
        # xlist = xlist.at[_ + 1].set(x)

    
    return x, xlist, loss_list

# key = random.PRNGKey(42) #compgpu 4 tmux 0
# key = random.PRNGKey(43) #compgpu 4 tmux 1
# key = random.PRNGKey(44) #compgpu 4 tmux 2
# key = random.PRNGKey(45) #compgpu 4 tmux 3
key = random.PRNGKey(46) #compgpu 4 tmux 7
parameter_value = jax.random.uniform(key=key, 
                                    shape=(500, 7), 
                                    minval=jnp.array([0.5 * u.Gyr.to(code_units.code_time), # t_end in Gyr
                                                    10**3.0 * u.Msun.to(code_units.code_mass), # Plummer mass
                                                    params.Plummer_params.a*(1/4),
                                                    params.NFW_params.Mvir*(1/4),
                                                    params.NFW_params.r_s*(1/4), 
                                                    params.MN_params.M*(1/4), 
                                                    params.MN_params.a*(1/4),]), 
                                                    
                                    maxval=jnp.array([5 * u.Gyr.to(code_units.code_time), # t_end in Gyr
                                                    10**4.5 * u.Msun.to(code_units.code_mass), #Plummer mass
                                                    params.Plummer_params.a*(8/4),
                                                    params.NFW_params.Mvir*(8/4), 
                                                    params.NFW_params.r_s*(8/4), 
                                                    params.MN_params.M*(8/4), 
                                                    params.MN_params.a*(8/4),])) 

i = 2000
for initial_guess in tqdm(parameter_value):
    key = random.PRNGKey(i)  # Use a different key for each optimization
    x1, xlist, loss_list = gradient_descent_optimization(
        time_integration_fix_position_grad_ScipyMinimize, 
        jnp.log10(initial_guess),  
        key, 
        learning_rate=0.01, 
        max_iter= 50)
    # Convert to numpy arrays for easier handling
    np.savez(f'./sampling_gradient_descend/loss_gradient_descending_{i}.npz',
            xlist=xlist, 
            loss_list=loss_list, 
            parameter_value=initial_guess)
    i += 1


