from autocvd import autocvd
autocvd(num_gpus = 1)
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set to the 0 for tmux 6

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


config = SimulationConfig(N_particles = 5_000,
                          return_snapshots = False, 
                          num_timesteps = 1000, 
                          external_accelerations=(NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL), 
                          acceleration_scheme = DIRECT_ACC_MATRIX,
                          softening = (0.1 * u.pc).to(code_units.code_length).value,) #default values


@jit
def run_simulation(rng_key, 
                   params):
    
    #the center of mass needs to be integrated backwards in time first 
    config_com = config._replace(N_particles=1,)
    params_com = params._replace(t_end=-params.t_end,)

    #this is the final position of the cluster, we need to integrate backwards in time 
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
    return run_simulation(rng_key, params_samples)



start_time = time.time()
batch_size = 60
num_chunks = 40_000
name_str = 22_960
for i in range(name_str, num_chunks, batch_size):
    rng_key = random.PRNGKey(i)
    parameter_value = jax.random.uniform(rng_key, 
                                         shape=(batch_size, 13), 
                                         minval=jnp.array([0.5, # t_end in Gyr
                                                           10**3.0, # Plummer mass
                                                           params_true.Plummer_params.a*(1/4),
                                                           params_true.NFW_params.Mvir*(1/4),
                                                           params_true.NFW_params.r_s*(1/4), 
                                                           params_true.MN_params.M*(1/4), 
                                                           params_true.MN_params.a*(1/4), 
                                                           10.0, #x
                                                           0.1, #y
                                                           6.0, #z
                                                           90.0, #vx
                                                           -280.0, #vy
                                                           -120.0]), #vz
                                                           
                                         maxval=jnp.array([5, # t_end in Gyr
                                                           10**4.5, #Plummer mass
                                                           params_true.Plummer_params.a*(8/4),
                                                           params_true.NFW_params.Mvir*(8/4), 
                                                           params_true.NFW_params.r_s*(8/4), 
                                                           params_true.MN_params.M*(8/4), 
                                                           params_true.MN_params.a*(8/4),
                                                           14.0, #x
                                                           2.5,  #y
                                                           8.0,  #z
                                                           115.0, #vx
                                                           -230.0, #vy
                                                           -80.0])) #vz
    
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
        np.savez_compressed(f"/export/data/vgiusepp/odisseo_data/data_varying_position/file_{name_str:06d}.npz",
                            x = stream_samples[j],
                            theta = parameter_value[j],)
        name_str += 1
        print('chunk', name_str-1)

end_time = time.time()
print("Time taken to sample in seconds:", end_time - start_time)