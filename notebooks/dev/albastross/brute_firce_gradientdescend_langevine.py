import os

from autocvd import autocvd
autocvd(num_gpus = 2)

# os.environ['CUDA_VISIBLE_DEVICES'] = '4, 6, 9, 7'  # Set the visible GPUs

import jax 
import jax.numpy as jnp
from jax import jit, random
import equinox as eqx
from jax.sharding import Mesh, PartitionSpec, NamedSharding
# jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision

import matplotlib.pyplot as plt


import numpy as np
from astropy import units as u
from astropy import constants as c

import odisseo
from odisseo import construct_initial_state
from odisseo.integrators import leapfrog
from odisseo.dynamics import direct_acc, DIRECT_ACC, DIRECT_ACC_LAXMAP, DIRECT_ACC_FOR_LOOP, DIRECT_ACC_MATRIX
from odisseo.option_classes import SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams, PSPParams, MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL
from odisseo.initial_condition import Plummer_sphere, ic_two_body, sample_position_on_sphere, inclined_circular_velocity, sample_position_on_circle, inclined_position
from odisseo.utils import center_of_mass
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.visualization import create_3d_gif, create_projection_gif, energy_angular_momentum_plot
from odisseo.potentials import MyamotoNagai, NFW

from odisseo.utils import halo_to_gd1_velocity_vmap, halo_to_gd1_vmap, projection_on_GD1


plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 15,
})

code_length = 10 * u.kpc
code_mass = 1e4 * u.Msun
G = 1
code_time = 3 * u.Gyr
code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
# code_units = CodeUnits(code_length, code_mass, G=G)  # default values


config = SimulationConfig(N_particles = 1000, 
                          return_snapshots = True, 
                          num_snapshots = 500, 
                          num_timesteps = 1000, 
                          external_accelerations=(NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL), 
                          acceleration_scheme = DIRECT_ACC_MATRIX,
                          softening = (0.1 * u.pc).to(code_units.code_length).value,) #default values

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


key = random.PRNGKey(1)

#set up the particles in the initial state
positions, velocities, mass = Plummer_sphere(key=key, params=params, config=config)


#the center of mass needs to be integrated backwards in time first 
config_com = config._replace(N_particles=1,return_snapshots=True,)
params_com = params._replace(t_end=-params.t_end,)

#this is the final position of the cluster, we need to integrate backwards in time 
pos_com_final = jnp.array([[11.8, 0.79, 6.4]]) * u.kpc.to(code_units.code_length)
vel_com_final = jnp.array([[109.5,-254.5,-90.3]]) * (u.km/u.s).to(code_units.code_velocity)
# pos_com_final = jnp.array([[12.4, 1.5, 7.1]]) * u.kpc.to(code_units.code_length)
# vel_com_final = jnp.array([[107.0, -243.0, -105.0]]) * (u.km/u.s).to(code_units.code_velocity)


mass_com = jnp.array([params_com.Plummer_params.Mtot])
final_state_com = construct_initial_state(pos_com_final, vel_com_final)

snapshots_com = time_integration(final_state_com, mass_com, config_com, params_com)
pos_com, vel_com = snapshots_com.states[-1, :, 0], snapshots_com.states[-1, :, 1]

# Add the center of mass position and velocity to the Plummer sphere particles
positions = positions + pos_com
velocities = velocities + vel_com

#initialize the initial state
initial_state_stream = construct_initial_state(positions, velocities)

#run the simulation
snapshots = time_integration(initial_state_stream, mass, config, params)

#project on GD1
stream_target = projection_on_GD1(snapshots.states[-1, :], code_units=code_units ) # Reshape to (N_particles, 6)


# Gradient on t_end 
# for now we will only use the last snapshot to caluclate the loss and the gradient
config =  config._replace(return_snapshots=False,)
config_com = config_com._replace(return_snapshots=False,)
stream_target = s

def normalize_data(X):
    """Z-score normalization"""
    mean = jnp.mean(X, axis=0)
    std = jnp.std(X, axis=0)
    return (X - mean) / std, mean, std

# Your stream data: [R, phi1, phi2, vR, v1_cosphi2, v2]
def normalize_stream_data(stream):
    """Normalize stellar stream data for RBF"""
    
    # Remove NaN values first
    mask = ~jnp.isnan(stream).any(axis=1)
    stream_clean = stream[mask]
    
    # Normalize each dimension
    stream_norm, mean, std = normalize_data(stream_clean)
    
    return stream_norm, mean, std

target_norm, mean, std = normalize_stream_data(stream_target)

@jit
def rbf_kernel(x, y, sigma):
    """RBF kernel optimized for 6D astronomical data"""
    return jnp.exp(-jnp.sum((x - y)**2) / (2 * sigma**2))

@jit 
def compute_mmd(sim_norm, target_norm, sigmas):
    xx = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda xj: rbf_kernel(xi, xj, sigmas))(sim_norm))(sim_norm))
    yy = jnp.mean(jax.vmap(lambda yi: jax.vmap(lambda yj: rbf_kernel(yi, yj, sigmas))(target_norm))(target_norm))
    xy = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda yj: rbf_kernel(xi, yj, sigmas))(target_norm))(sim_norm))
    return xx + yy - 2 * xy

@jit
def time_integration_grad(t_end, 
                          M_plummer,
                          a_plummer,
                          M_NFW, 
                          rs_NFW,
                          M_MN,
                          a_MN,
                          key):


    key_Plummer, key_noise = jax.random.split(key, 2)
    #we set up the parameters of the simulations, changing only the parameter that we want to optimize
    new_params = params._replace(t_end = t_end,)
    new_params = new_params._replace(NFW_params = params.NFW_params._replace(Mvir = M_NFW, r_s = rs_NFW,))
    new_params = new_params._replace(MN_params = params.MN_params._replace(M = M_MN, a = a_MN,))
    new_params = new_params._replace(Plummer_params = params.Plummer_params._replace(Mtot = M_plummer, a = a_plummer,))

    new_params_com = params_com._replace(t_end = -t_end,)
    new_params_com = new_params_com._replace(NFW_params = params_com.NFW_params._replace(Mvir = M_NFW, r_s = rs_NFW,))
    new_params_com = new_params_com._replace(MN_params = params_com.MN_params._replace(M = M_MN, a = a_MN,))
    new_params_com = new_params_com._replace(Plummer_params = params_com.Plummer_params._replace(Mtot = M_plummer, a = a_plummer,))
    

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
    #add gaussian noise to the stream
    noise_std = jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0])
    stream = stream + jax.random.normal(key=key_noise, shape=stream.shape) * noise_std
    #we calculate the loss as the negative log likelihood of the stream

     # Normalize to standard ranges for each dimension
    bounds = jnp.array([
        [6, 20],        # R [kpc]
        [-120, 70],     # phi1 [deg]  
        [-8, 2],        # phi2 [deg]
        [-250, 250],    # vR [km/s]
        [-2., 1.0],     # v1_cosphi2 [mas/yr]
        [-0.10, 0.10]   # v2 [mas/yr]
    ])

    
    sim_norm = (stream - mean)/std
    
    # Adaptive bandwidth for 6D data
    # n_sim, n_target = len(stream), len(stream_target)
    # sigma = 0.5 * jnp.power(n_sim + n_target, -1/(6+4))  # Rule of thumb for 6D
    
    # # Compute MMD terms
    # xx = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda xj: rbf_kernel(xi, xj, sigma))(sim_norm))(sim_norm))
    # yy = jnp.mean(jax.vmap(lambda yi: jax.vmap(lambda yj: rbf_kernel(yi, yj, sigma))(target_norm))(target_norm))
    # xy = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda yj: rbf_kernel(xi, yj, sigma))(target_norm))(sim_norm))
    
    # return xx + yy - 2 * xy

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
    scale_weights = jnp.array([0.1, 0.1, 0.1, 0.1, 0.6])
    scale_weights = jnp.ones_like(sigmas)  # Equal weights for simplicity

    # Compute MMD with multiple kernels
    mmd_total = jnp.sum(scale_weights * jax.vmap(lambda sigma: compute_mmd(sim_norm, target_norm, sigma))(sigmas))
    
    return mmd_total / len(sigmas)


bounds_mass = jnp.log10(np.array([1e11, 1e12]) * u.Msun.to(code_units.code_mass)) # in code units
bound_time = jnp.log10(np.array([5e-3, 5]) * u.Gyr.to(code_units.code_time)) # in code units

@jit 
def normalize_Mvir_and_t_end(Mvir_and_t_end):
    # Mvir, t_end = Mvir_and_t_end
    Mvir, t_end = jnp.log10(Mvir_and_t_end)
    # Normalize Mvir to [-1, 1]
    Mvir_norm = (Mvir - bounds_mass[0]) / (bounds_mass[1] - bounds_mass[0])
    # Normalize t_end to [-1, 1]
    t_end_norm = (t_end - bound_time[0]) / (bound_time[1] - bound_time[0]) 
    return jnp.array([Mvir_norm, t_end_norm])

@jit
def de_normalize_Mvir_and_t_end(Mvir_and_t_end_norm):
    Mvir_norm, t_end_norm = Mvir_and_t_end_norm
    # De-normalize Mvir to original scale
    Mvir = Mvir_norm * (bounds_mass[1] - bounds_mass[0]) + bounds_mass[0]
    Mvir = jnp.power(10, Mvir)
    # De-normalize t_end to original scale
    t_end = t_end_norm * (bound_time[1] - bound_time[0]) + bound_time[0]
    t_end = jnp.power(10, t_end)
    return jnp.array([Mvir, t_end])

    
@jit
def time_integration_for_gradient_descend(params, key):
    # Mvir, t_end = de_normalize_Mvir_and_t_end(Mvir_and_t_end)
    t_end, M_plummer, a_plummer, M_NFW, rs_NFW, M_MN, a_MN = params
    t_end = 10**t_end
    M_plummer = 10**M_plummer
    a_plummer = 10**a_plummer
    M_NFW = 10**M_NFW
    M_MN = 10**M_MN 
    rs_NFW = 10**rs_NFW
    a_MN = 10**a_MN
    return time_integration_grad(t_end,
                                   M_plummer,
                                   a_plummer,
                                   M_NFW, 
                                   rs_NFW,
                                   M_MN,
                                   a_MN,
                                   key)


# Calculate the value of the function and the gradient wrt the total mass of the plummer sphere
t_end = (params.t_end * (5/4) * u.Gyr).to(code_units.code_time).value  # Example: 25% increase in t_end
M_plummer = (params.Plummer_params.Mtot * (3/4) * u.Msun).to(code_units.code_mass).value
a_plummer = (params.Plummer_params.a * (3/4) * u.kpc).to(code_units.code_length).value
M_NFW = (params.NFW_params.Mvir * (3/4) * u.Msun).to(code_units.code_mass).value
rs_NFW = (params.NFW_params.r_s * (3/4) * u.kpc).to(code_units.code_length).value
M_MN = (params.MN_params.M * (3/4) * u.Msun).to(code_units.code_mass).value
a_MN = (params.MN_params.a * (3/4) * u.kpc).to(code_units.code_length).value
key = random.PRNGKey(0)
loss, grad = jax.value_and_grad(lambda t_end, M_plummer, a_plummer, M_NFW, rs_NFW, M_MN, a_MN, key: time_integration_grad(jnp.log10(t_end),
                                                                                  jnp.log10(M_plummer),
                                                                                  jnp.log10(a_plummer),
                                                                                  jnp.log10(M_NFW),
                                                                                  jnp.log10(rs_NFW),
                                                                                  jnp.log10(M_MN),
                                                                                  jnp.log10(a_MN),
                                                                                  key),
                                 argnums=(0,1,2,3,4,5,6))(t_end, M_plummer, a_plummer, M_NFW, rs_NFW, M_MN, a_MN, key)
print("Gradient of the total mass of the Mvir of NFW:\n", grad)
print("Loss:\n", loss)


### GRID SEARCH 
n_sim = 2

# t_end_values = jnp.linspace(params.t_end * (1/4), params.t_end * (8/4), n_sim-1)   # Adjust range based on expected timescales
# M_plummer_values = jnp.linspace(params.Plummer_params.Mtot*(1/4), params.Plummer_params.Mtot*(8/4), n_sim-1)  # Adjust range based on expected values
# a_plummer_values = jnp.linspace(params.Plummer_params.a*(1/4), params.Plummer_params.a*(8/4), n_sim-1)  # Adjust range based on expected values
# M_NFW_values = jnp.linspace(params.NFW_params.Mvir*(1/4), params.NFW_params.Mvir*(8/4), n_sim-1)  # Adjust range based on expected values
# rs_NFW_values = jnp.linspace(params.NFW_params.r_s*(1/4), params.NFW_params.r_s*(8/4), n_sim-1)  # Adjust range based on expected values
# M_MN_values = jnp.linspace(params.MN_params.M*(1/4), params.MN_params.M*(8/4), n_sim-1)  # Adjust range based on expected values
# a_MN_values = jnp.linspace(params.MN_params.a*(1/4), params.MN_params.a*(8/4), n_sim-1)  # Adjust range based on expected values
t_end_values = jax.random.uniform(random.PRNGKey(0), shape=(n_sim-1,), minval=params.t_end * (1/4), maxval=params.t_end * (8/4))  # Random values in the range
M_plummer_values = jax.random.uniform(random.PRNGKey(1), shape=(n_sim-1,), minval=params.Plummer_params.Mtot*(1/4), maxval=params.Plummer_params.Mtot*(8/4))  # Random values in the range
a_plummer_values = jax.random.uniform(random.PRNGKey(2), shape=(n_sim-1,), minval=params.Plummer_params.a*(1/4), maxval=params.Plummer_params.a*(8/4))  # Random values in the range
M_NFW_values = jax.random.uniform(random.PRNGKey(3), shape=(n_sim-1,), minval=params.NFW_params.Mvir*(1/4), maxval=params.NFW_params.Mvir*(8/4))  # Random values in the range
rs_NFW_values = jax.random.uniform(random.PRNGKey(4), shape=(n_sim-1,), minval=params.NFW_params.r_s*(1/4), maxval=params.NFW_params.r_s*(8/4))  # Random values in the range
M_MN_values = jax.random.uniform(random.PRNGKey(5), shape=(n_sim-1,), minval=params.MN_params.M*(1/4), maxval=params.MN_params.M*(8/4))  # Random values in the range
a_MN_values = jax.random.uniform(random.PRNGKey(6), shape=(n_sim-1,), minval=params.MN_params.a*(1/4), maxval=params.MN_params.a*(8/4))  # Random values in the range
# Append the true values to the arrays
t_end_values = jnp.concatenate([t_end_values, jnp.array([params.t_end])])  # Append the true t_end value
M_plummer_values = jnp.concatenate([M_plummer_values, jnp.array([params.Plummer_params.Mtot])])  # Append the true Mtot value
a_plummer_values = jnp.concatenate([a_plummer_values, jnp.array([params.Plummer_params.a])])  # Append the true a value
M_NFW_values = jnp.concatenate([M_NFW_values, jnp.array([params.NFW_params.Mvir])])  # Append the true Mvir value
rs_NFW_values = jnp.concatenate([rs_NFW_values, jnp.array([params.NFW_params.r_s])])  # Append the true r_s value
M_MN_values = jnp.concatenate([M_MN_values, jnp.array([params.MN_params.M])])  # Append the true Mtot value
a_MN_values = jnp.concatenate([a_MN_values, jnp.array([params.MN_params.a])])  # Append the true a value
# Ensure all arrays are sorted
t_end_values = jnp.sort(t_end_values)
M_plummer_values = jnp.sort(M_plummer_values)
a_plummer_values = jnp.sort(a_plummer_values)
M_NFW_values = jnp.sort(M_NFW_values)
rs_NFW_values = jnp.sort(rs_NFW_values)
M_MN_values = jnp.sort(M_MN_values)
a_MN_values = jnp.sort(a_MN_values)
# Create a meshgrid
M_plummer_grid, a_plummer_grid, t_end_grid, M_NFW_grid, rs_NFW_grid, M_MN_grid, a_MN_grid = jnp.meshgrid(
    M_plummer_values, a_plummer_values, t_end_values, M_NFW_values, rs_NFW_values, M_MN_values, a_MN_values, indexing="ij")
# Flatten the grid for vectorized computation
M_plummer_flat = M_plummer_grid.flatten()
a_plummer_flat = a_plummer_grid.flatten()
t_end_flat = t_end_grid.flatten()
M_NFW_flat = M_NFW_grid.flatten()
rs_NFW_flat = rs_NFW_grid.flatten()
M_MN_flat = M_MN_grid.flatten()
a_MN_flat = a_MN_grid.flatten()
keys_flat = jax.random.split(random.PRNGKey(0), len(M_plummer_flat))  # Create a flat array of keys
mesh = Mesh(np.array(jax.devices()), ("i",))
M_plummer_sharded = jax.device_put(M_plummer_flat, NamedSharding(mesh, PartitionSpec("i")))
a_plummer_sharded = jax.device_put(a_plummer_flat, NamedSharding(mesh, PartitionSpec("i")))
t_end_sharded = jax.device_put(t_end_flat, NamedSharding(mesh, PartitionSpec("i")))
M_NFW_sharded = jax.device_put(M_NFW_flat, NamedSharding(mesh, PartitionSpec("i")))
rs_NFW_sharded = jax.device_put(rs_NFW_flat, NamedSharding(mesh, PartitionSpec("i")))
M_MN_sharded = jax.device_put(M_MN_flat, NamedSharding(mesh, PartitionSpec("i")))   
a_MN_sharded = jax.device_put(a_MN_flat, NamedSharding(mesh, PartitionSpec("i")))
keys_sharded = jax.device_put(keys_flat, NamedSharding(mesh, PartitionSpec("i")))

@jit
def time_integration_for_laxmap(input):
    M_plummer, a_plummer, t_end, M_NFW, rs_NFW, M_MN, a_MN, key = input
    params = jnp.log10(jnp.array([M_plummer, a_plummer, t_end, M_NFW, rs_NFW, M_MN, a_MN]))
    return jax.value_and_grad(time_integration_for_gradient_descend)(params, key)

loss, grad = jax.lax.map(f=time_integration_for_laxmap,
                         xs=(M_plummer_sharded, a_plummer_sharded, t_end_sharded,
                              M_NFW_sharded, rs_NFW_sharded, M_MN_sharded, a_MN_sharded, keys_sharded),
                         batch_size=15)


loss_min, min_index = jnp.min(loss), jnp.argmin(loss)
M_plummer_min = M_plummer_values[min_index // (n_sim**6)]
a_plummer_min = a_plummer_values[(min_index // n_sim) % n_sim]
t_end_min = t_end_values[(min_index // (n_sim**5)) % n_sim]
M_NFW_min = M_NFW_values[(min_index // (n_sim**4)) % n_sim]
rs_NFW_min = rs_NFW_values[(min_index // (n_sim**3)) % n_sim]
M_MN_min = M_MN_values[(min_index // (n_sim**2)) % n_sim]
a_MN_min = a_MN_values[min_index % n_sim]
print(f"Minimum loss: {loss_min}, M_plummer: {M_plummer_min * code_units.code_mass.to(u.Msun)}, a_plummer: {a_plummer_min * code_units.code_length.to(u.kpc)}, t_end: {t_end_min * code_units.code_time.to(u.Gyr)}, M_NFW: {M_NFW_min * code_units.code_mass.to(u.Msun)}, rs_NFW: {rs_NFW_min * code_units.code_length.to(u.kpc)}, M_MN: {M_MN_min * code_units.code_mass.to(u.Msun)}, a_MN: {a_MN_min * code_units.code_length.to(u.kpc)}")

np.savez('./brute_force_gradientdescend_langevine/grid_search_results.npz',
         loss=loss, 
         M_plummer=M_plummer_values, 
         a_plummer=a_plummer_values, 
         t_end=t_end_values, 
         M_NFW=M_NFW_values, 
         rs_NFW=rs_NFW_values, 
         M_MN=M_MN_values, 
         a_MN=a_MN_values,
         M_plummer_min=M_plummer_min,
         a_plummer_min=a_plummer_min,
         t_end_min=t_end_min,
         M_NFW_min=M_NFW_min,
         rs_NFW_min=rs_NFW_min,
         M_MN_min=M_MN_min,
         a_MN_min=a_MN_min)

## Gradient descend
from jaxopt import ScipyMinimize, LBFGS
import optax
import optimistix as optx
from tqdm import tqdm


# from functools import partial
# @partial(jit, static_argnames=('func','learning_rate', 'tol', 'max_iter'))
def gradient_descent_optimization(func, x_init, key, learning_rate=20, tol=0.5, max_iter=2000):
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

    #SCHEDULE FREE ADAM
    # learning_rate_fn = optax.warmup_constant_schedule(peak_value=learning_rate, warmup_steps=10, init_value=0.0)
    # optimizer = optax.adam(learning_rate_fn, b1=0.)
    # optimizer = optax.contrib.schedule_free(optimizer, learning_rate_fn, b1=0.9)
    # optimizer_state = optimizer.init(x)

    #SCHEDULE FREE ADAMW
    optimizer = optax.contrib.schedule_free_adamw(learning_rate)
    optimizer_state = optimizer.init(x)


    for _ in tqdm(range(max_iter)):
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
        
        # Check convergence
        if jnp.linalg.norm(updates) < tol:
            break
    
    return x, xlist, loss_list
x1, xlist, loss_list = gradient_descent_optimization(
    time_integration_for_gradient_descend, 
    jnp.array([jnp.log10(M_plummer_min), jnp.log10(a_plummer_min), jnp.log10(t_end_min), 
               jnp.log10(M_NFW_min), jnp.log10(rs_NFW_min), jnp.log10(M_MN_min), jnp.log10(a_MN_min)]),
    random.PRNGKey(0), 
    learning_rate=0.01,
    tol=1e-12, 
    max_iter=100)

x1 = 10**x1  # Convert back to original scale
xlist = 10**jnp.array(xlist)  # Convert back to original scale

np.savez('./brute_force_gradientdescend_langevine/gradient_descent_results.npz',
            xlist=xlist, 
            loss_list=loss_list, 
            M_plummer=x1[0], 
            a_plummer=x1[1], 
            t_end=x1[2], 
            M_NFW=x1[3], 
            rs_NFW=x1[4], 
            M_MN=x1[5], 
            a_MN=x1[6])


@jit 
def time_integration_for_langevin(params, key):
    t_end = 10**params['t_end']
    M_plummer = 10**params['M_plummer']
    a_plummer = 10**params['a_plummer']
    M_NFW = 10**params['M_NFW']
    rs_NFW = 10**params['rs_NFW']
    M_MN = 10**params['M_MN']
    a_MN = 10**params['a_MN']
    key = random.PRNGKey(0)

    return time_integration_grad(t_end,
                                   M_plummer,
                                   a_plummer,
                                   M_NFW, 
                                   rs_NFW,
                                   M_MN,
                                   a_MN,
                                   key)

from tqdm import tqdm 

def langevin_sampler(initial_params, num_samples, step_size, rng_key):
    samples = []
    params = initial_params
    for i in tqdm(range(num_samples)):
        grads = jax.jacfwd(time_integration_for_langevin)(params, rng_key)
        noise = jax.random.normal(rng_key, shape=jnp.array(list(params.values())).shape)
        params = {k: v + 0.5 * step_size * grads[k] + jnp.sqrt(step_size) * noise[idx] for idx, (k, v) in enumerate(params.items())}
        samples.append(params)
        rng_key, _ = jax.random.split(rng_key)
    return samples


params_MLE = {'t_end': jnp.log10(t_end_min),
              'M_plummer': jnp.log10(M_plummer_min), 
              'a_plummer': jnp.log10(a_plummer_min),
              'M_NFW': jnp.log10(M_NFW_min), 
              'rs_NFW': jnp.log10(rs_NFW_min),
              'M_MN': jnp.log10(M_MN_min), 
              'a_MN': jnp.log10(a_MN_min)}
rng_key = random.PRNGKey(42)
out_samps = langevin_sampler(params_MLE, 1000, 1e-7, rng_key)

np.savez('./brute_force_gradientdescend_langevine/langevin_samples.npz',
            out_samps=out_samps,
            params_MLE=params_MLE)

from chainconsumer import Chain, ChainConsumer, Truth
import pandas as pd

df = pd.DataFrame(out_samps, columns=['M_plummer', 'a_plummer', 't_end', 'M_NFW', 'rs_NFW', 'M_MN', 'a_MN'])
df['M_plummer'] = 10**df['M_plummer'] * code_units.code_mass.to(u.Msun)
df['a_plummer'] = 10**df['a_plummer'] * code_units.code_length.to(u.kpc)
df['t_end'] = 10**df['t_end'] * code_units.code_time.to(u.Gyr)
df['M_NFW'] = 10**df['M_NFW'] * code_units.code_mass.to(u.Msun) 
df['rs_NFW'] = 10**df['rs_NFW'] * code_units.code_length.to(u.kpc)
df['M_MN'] = 10**df['M_MN'] * code_units.code_mass.to(u.Msun)
df['a_MN'] = 10**df['a_MN'] * code_units.code_length.to(u.kpc)

df['M_plummer'] = df['M_plummer'].astype(float)
df['a_plummer'] = df['a_plummer'].astype(float)
df['t_end'] = df['t_end'].astype(float)
df['M_NFW'] = df['M_NFW'].astype(float)
df['rs_NFW'] = df['rs_NFW'].astype(float)
df['M_MN'] = df['M_MN'].astype(float)
df['a_MN'] = df['a_MN'].astype(float)
c = ChainConsumer()
c.add_chain(Chain(samples=df, name='Langevin samples'))
c.add_truth(Truth(location={'M_plummer': params.Plummer_params.Mtot * code_units.code_mass.to(u.Msun),
                            'a_plummer': params.Plummer_params.a * code_units.code_length.to(u.kpc),
                            't_end': params.t_end * code_units.code_time.to(u.Gyr),
                            'M_NFW': params.NFW_params.Mvir * code_units.code_mass.to(u.Msun),
                            'rs_NFW': params.NFW_params.r_s * code_units.code_length.to(u.kpc),
                            'M_MN': params.MN_params.M * code_units.code_mass.to(u.Msun),
                            'a_MN': params.MN_params.a * code_units.code_length.to(u.kpc)}), )
fig = c.plotter.plot()

fig.savefig('./brute_force_gradientdescend_langevine/langevin_samples.pdf', bbox_inches='tight')