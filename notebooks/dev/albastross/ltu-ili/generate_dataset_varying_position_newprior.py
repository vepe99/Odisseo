from autocvd import autocvd
autocvd(num_gpus = 1)
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # Set to the 0 for tmux 6

import time

import jax
import jax.numpy as jnp
from jax import jit, random

jax.config.update("jax_enable_x64", True)

import numpy as np
from astropy import units as u

from odisseo import construct_initial_state
from odisseo.dynamics import  DIRECT_ACC_MATRIX
from odisseo.option_classes import SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams, PSPParams, MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL
from odisseo.initial_condition import Plummer_sphere
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.utils import projection_on_GD1

code_length = 10.0 * u.kpc
code_mass = 1e4 * u.Msun
code_time = 3 * u.Gyr
code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  


config = SimulationConfig(N_particles = 1_000,
                          return_snapshots = False, 
                          num_timesteps = 1000, 
                          external_accelerations=(NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL), 
                          acceleration_scheme = DIRECT_ACC_MATRIX,
                          softening = (0.1 * u.pc).to(code_units.code_length).value,) #default values


@jit
def run_simulation(rng_key, 
                   params, 
                   position_velocity):
    
    #the center of mass needs to be integrated backwards in time first 
    config_com = config._replace(N_particles=1,)
    params_com = params._replace(t_end=-params.t_end,)

    #this is the final position of the cluster, we need to integrate backwards in time 
    pos_com_final = jnp.array([[position_velocity[0], position_velocity[1], position_velocity[2]]]) * u.kpc.to(code_units.code_length)
    vel_com_final = jnp.array([[position_velocity[3], position_velocity[4], position_velocity[5]]]) * (u.km/u.s).to(code_units.code_velocity)
    mass_com = jnp.array([params.Plummer_params.Mtot]) 
    
    #we construmt the initial state of the com 
    initial_state_com = construct_initial_state(pos_com_final, vel_com_final,)
    #we run the simulation backwards in time for the center of mass
    final_state_com = time_integration(initial_state_com, mass_com, config=config_com, params=params_com)
    #we calculate the final position and velocity of the center of mass
    pos_com = final_state_com[:, 0]
    vel_com = final_state_com[:, 1]

    #we construct the initial state of the Plummer sphere
    positions, velocities, mass = Plummer_sphere(key=random.PRNGKey(rng_key), params=params, config=config)
    #we add the center of mass position and velocity to the Plummer sphere particles
    positions = positions + pos_com
    velocities = velocities + vel_com
    #initialize the initial state
    initial_state_stream = construct_initial_state(positions, velocities, )
    #run the simulation
    final_state = time_integration(initial_state_stream, mass, config=config, params=params)

    #projection on the GD1 stream
    stream = projection_on_GD1(final_state, code_units=code_units,)

    return stream



print('Beginning sampling...')

params_true = SimulationParams(t_end = (3 * u.Gyr).to(code_units.code_time).value,  
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

b_code_units = params_true.MN_params.b * u.kpc.to(code_units.code_length)
M_PSP_units = params_true.PSP_params.M * u.Msun.to(code_units.code_mass)
alpha_PSP_units = params_true.PSP_params.alpha
r_c_PSP_units = params_true.PSP_params.r_c * u.kpc.to(code_units.code_length)


@jit
def vmapped_run_simulation(rng_key, params_values):
    params_samples = SimulationParams(t_end = params_values[0],
                          Plummer_params = PlummerParams(Mtot=params_values[1],
                                                        a=params_values[2] ),
                          NFW_params = NFWParams(Mvir=params_values[3] ,
                                               r_s= params_values[4]),
                          MN_params = MNParams(M = params_values[5] ,
                                              a = params_values[6] ,
                                              b = b_code_units ),
                          PSP_params = PSPParams(M = M_PSP_units,
                                                alpha = 1.8, 
                                                r_c = r_c_PSP_units),
                          G = code_units.G, )
    position_velocity = params_values[7:]
    return run_simulation(rng_key, params_samples, position_velocity=position_velocity)


def sample_mixed_prior(rng_key, batch_size, params_true):
    """
    Sample with improved priors for masses/scales but uniform for position/velocity
    """
    keys = jax.random.split(rng_key, 13)
    
    # 1. Time evolution - uniform as before
    t_end = jax.random.uniform(keys[0], (batch_size,), minval=0.5, maxval=5.0)
    
    # 2. Plummer mass - log-uniform (better for masses)
    log_M_plummer = jax.random.uniform(keys[1], (batch_size,), 
                                       minval=jnp.log10(10**3.0), 
                                       maxval=jnp.log10(10**4.5))
    M_plummer = 10**log_M_plummer
    
    # 3-7. Physical parameters - log-normal around true values
    # For log-normal: sample from normal, then transform
    log_a_plummer = jax.random.normal(keys[2], (batch_size,)) * 0.5 + jnp.log(params_true.Plummer_params.a)
    a_plummer = jnp.exp(log_a_plummer)
    
    log_M_NFW = jax.random.normal(keys[3], (batch_size,)) * 0.3 + jnp.log(params_true.NFW_params.Mvir)
    M_NFW = jnp.exp(log_M_NFW)
    
    log_r_s_NFW = jax.random.normal(keys[4], (batch_size,)) * 0.4 + jnp.log(params_true.NFW_params.r_s)
    r_s_NFW = jnp.exp(log_r_s_NFW)
    
    log_M_MN = jax.random.normal(keys[5], (batch_size,)) * 0.2 + jnp.log(params_true.MN_params.M)
    M_MN = jnp.exp(log_M_MN)
    
    log_a_MN = jax.random.normal(keys[6], (batch_size,)) * 0.3 + jnp.log(params_true.MN_params.a)
    a_MN = jnp.exp(log_a_MN)

    
    # 8-13. Position and velocity - UNIFORM as in your original code
    x_pos = jax.random.uniform(keys[7], (batch_size,), minval=10.0, maxval=14.0)
    y_pos = jax.random.uniform(keys[8], (batch_size,), minval=0.1, maxval=2.5)
    z_pos = jax.random.uniform(keys[9], (batch_size,), minval=6.0, maxval=8.0)
    vx = jax.random.uniform(keys[10], (batch_size,), minval=90.0, maxval=115.0)
    vy = jax.random.uniform(keys[11], (batch_size,), minval=-280.0, maxval=-230.0)
    vz = jax.random.uniform(keys[12], (batch_size,), minval=-120.0, maxval=-80.0)
    
    return jnp.stack([t_end, M_plummer, a_plummer, M_NFW, r_s_NFW, 
                      M_MN, a_MN, x_pos, y_pos, z_pos, vx, vy, vz], axis=1)
#11500
start_time = time.time()
batch_size = 500
num_chunks = 115000
name_str = 0
for i in range(name_str, num_chunks, batch_size):
    rng_key = random.PRNGKey(i)
    # Use mixed prior (improved for masses, uniform for positions/velocities)
    parameter_value = sample_mixed_prior(rng_key, batch_size, params_true)
    
    parameter_value_code_units = jnp.array([parameter_value[:, 0] * u.Gyr.to(code_units.code_time),
                                            parameter_value[:, 1] * u.Msun.to(code_units.code_mass),
                                            parameter_value[:, 2],
                                            parameter_value[:, 3],
                                            parameter_value[:, 4],
                                            parameter_value[:, 5],
                                            parameter_value[:, 6],
                                            parameter_value[:, 7],
                                            parameter_value[:, 8],
                                            parameter_value[:, 9],
                                            parameter_value[:, 10],
                                            parameter_value[:, 11],
                                            parameter_value[:, 12]]).T
    
    stream_samples = jax.vmap(vmapped_run_simulation, )(random.split(rng_key, batch_size)[:, 0], parameter_value_code_units)
    for j in range(batch_size):
        np.savez_compressed(f"/export/data/vgiusepp/odisseo_data/data_varying_position_newprior/file_{name_str:06d}.npz",
                            x = stream_samples[j],
                            theta = parameter_value[j],)
        name_str += 1
        print('chunk', name_str-1)

end_time = time.time()
print("Time taken to sample in seconds:", end_time - start_time)