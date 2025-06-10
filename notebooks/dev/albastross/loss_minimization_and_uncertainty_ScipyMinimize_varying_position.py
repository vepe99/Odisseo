from autocvd import autocvd
autocvd(num_gpus = 1)
# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '3' #set the GPU to use, if you have multiple GPUs, you can change this to the desired GPU

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
#for memory reason we will only select 1_000 background stars
# N_background = int(1e6 * 1e-3)  # Reduce the number of background stars for memory efficiency
# selected_background = jax.vmap(background_assignement, )(random.split(key=key_background_true, num=N_background))

# Combine the selected stream and background stars
stream = jnp.concatenate((selected_stream, selected_background), axis=0)

#add gaussian noise to the stream, same as in Albatross paper (https://arxiv.org/pdf/2304.02032)
noise_std = jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0])
stream = stream + jax.random.normal(key=key_noise_true, shape=stream.shape) * noise_std

stream_mean = jnp.nanmean(stream, axis=0)
stream_std = jnp.nanstd(stream, axis=0)
stream = (stream - stream_mean) / stream_std  # Standardize the data

# Assign a new out of the observational windows value to the NaN values, and use constrain support NF, not implemented yet
# stream = jax.vmap(lambda stream_star: jnp.where(jnp.isnan(stream_star), jnp.ones((6))*100, stream_star))(stream)


# create the flow
subkey, rng = jax.random.split(key_flow)
flow = masked_autoregressive_flow(
    subkey,
    base_dist=Normal(jnp.zeros(stream.shape[1])),
    transformer=RationalQuadraticSpline(knots=8, interval=4),
)

#we train only on the non NaN values of the stream
key, subkey = jax.random.split(key_flow)
flow, losses = fit_to_data(subkey, flow, stream[~jnp.isnan(stream)].reshape(-1, 6), learning_rate=1e-3)


# for now we will only use the last snapshot to caluclate the loss and the gradient
config =  config._replace(return_snapshots=False,)
config_com = config_com._replace(return_snapshots=False,)

@jit
def time_integration_varying_position_grad(t_end, 
                                            M_plummer,
                                            a_plummer,
                                            M_NFW,
                                            r_s_NFW,
                                            M_MN,
                                            a_MN,
                                            x,
                                            y,
                                            z,
                                            vx,
                                            vy,
                                            vz,
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
    pos_com_final = jnp.array([[x, y, z]]) * u.kpc.to(code_units.code_length)
    vel_com_final = jnp.array([[vx, vy, vz]]) * (u.km/u.s).to(code_units.code_velocity)
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
    selected_stream = jax.vmap(selection_function, )(stream, p, keys_selection)

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
    log_prob = eqx.filter_jit(flow.log_prob)((stream-stream_mean)/stream_std)  # Subtract the mean and divde by the std for normalization

    loss = - jnp.sum( jnp.where( jnp.isinf(log_prob), 0., log_prob  )  ) #if the NaN value are passed the value of the log_prob is -inf, we set it to 0 to not contribute to the loss

    return loss


@jit
def time_integration_varying_position_grad_ScipyMinimize(param, key):
    t_end, M_plummer, a_plummer, Mvir, r_s_NFW, M_MN, a_MN, x, y, z, vx, vy, vz = param
    return time_integration_varying_position_grad(t_end, 
                                                M_plummer,
                                                a_plummer,
                                                Mvir,
                                                r_s_NFW,
                                                M_MN,
                                                a_MN,
                                                x,
                                                y,
                                                z,
                                                vx,
                                                vy,
                                                vz,
                                                key)

optimizer = ScipyBoundedMinimize(
     method="l-bfgs-b", 
     dtype=jnp.float64,
     fun=time_integration_varying_position_grad_ScipyMinimize, 
     tol=1e-6, 
    )


# key = random.PRNGKey(42) #tmux 5 0-500
# key = random.PRNGKey(43) #tmux 6
# key = random.PRNGKey(44) #tmux 7
key = random.PRNGKey(45) #tmux 8
parameter_value = jax.random.uniform(key=key, 
                                    shape=(500, 13), 
                                    minval=jnp.array([0.5 * u.Gyr.to(code_units.code_time), # t_end in Gyr
                                                    np.log10(10**3.0 * u.Msun.to(code_units.code_mass)).item(), # Plummer mass
                                                    params.Plummer_params.a*(1/4),
                                                    np.log10(params.NFW_params.Mvir*(1/4)).item(),
                                                    params.NFW_params.r_s*(1/4), 
                                                    np.log10(params.MN_params.M*(1/4)).item(), 
                                                    params.MN_params.a*(1/4),
                                                    10.0, #x can be left in kpc
                                                    0.1, #y
                                                    6.0, #z
                                                    90.0, #vx can be left in km/s
                                                    -280.0, #vy
                                                    -120.0]), #vz
                                                    
                                    maxval=jnp.array([5 * u.Gyr.to(code_units.code_time), # t_end in Gyr
                                                    np.log10(10**4.5 * u.Msun.to(code_units.code_mass)).item(), #Plummer mass
                                                    params.Plummer_params.a*(8/4),
                                                    np.log10(params.NFW_params.Mvir*(8/4)).item(), 
                                                    params.NFW_params.r_s*(8/4), 
                                                    np.log10(params.MN_params.M*(8/4)).item(), 
                                                    params.MN_params.a*(8/4),
                                                    14.0, #x
                                                    2.5,  #y
                                                    8.0,  #z
                                                    115.0, #vx
                                                    -230.0, #vy
                                                    -80.0])) #vz) 
print('Start sampling with ScipyMinimize')
start_time = time.time()
i = 1500
for p, k in tqdm(zip(parameter_value, random.split(key, parameter_value.shape[0]) ) ):
    sol = optimizer.run(init_params=p, 
                        key=k,
                        bounds = jnp.array([[0.5 * u.Gyr.to(code_units.code_time), 
                                     np.log10(10**3.0 * u.Msun.to(code_units.code_mass)).item(), 
                                     params.Plummer_params.a*(1/4),
                                     np.log10(params.NFW_params.Mvir*(1/4)).item(),
                                     params.NFW_params.r_s*(1/4), 
                                     np.log10(params.MN_params.M*(1/4)).item(), 
                                     params.MN_params.a*(1/4),
                                     10.0, #x can be left in kpc
                                    0.1, #y
                                    6.0, #z
                                    90.0, #vx can be left in km/s
                                    -280.0, #vy
                                    -120.0],
                                    [5 * u.Gyr.to(code_units.code_time), 
                                     np.log10(10**4.5 * u.Msun.to(code_units.code_mass)).item(), 
                                     params.Plummer_params.a*(8/4),
                                     np.log10(params.NFW_params.Mvir*(8/4)).item(), 
                                     params.NFW_params.r_s*(8/4), 
                                     np.log10(params.MN_params.M*(8/4)).item(), 
                                     params.MN_params.a*(8/4),
                                     14.0, #x
                                    2.5,  #y
                                    8.0,  #z
                                    115.0, #vx
                                    -230.0, #vy
                                    -80.0]]))
    np.savez(f'./sampling_ScipyMinimize_varying_position/ScipyBoundedMinimize/sample_{i}.npz', 
             sample=np.array(sol.params),
             loss=np.array(sol.state.fun_val),)

    i += 1

end_time = time.time()
print("Time taken to sample in seconds:", end_time - start_time)