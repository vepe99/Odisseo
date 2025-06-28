import os

# from autocvd import autocvd
# autocvd(num_gpus = 2)
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 7'  # Set the GPU to use, if available
from functools import partial

import jax 
import jax.numpy as jnp
from jax import jit, random
import equinox as eqx
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.scipy.stats import gaussian_kde
# jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision

import matplotlib.pyplot as plt


import numpy as np
from astropy import units as u
from astropy import constants as c
import pickle


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

code_length = 10 * u.kpc
code_mass = 1e4 * u.Msun
G = 1
code_time = 3 * u.Gyr
code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  

config = SimulationConfig(N_particles = 1000, 
                          return_snapshots = False, 
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


# Create the TARGET stream for the GD-1 stream
key = random.PRNGKey(43)
#set up the particles in the initial state
positions, velocities, mass = Plummer_sphere(key=key, params=params, config=config)
#the center of mass needs to be integrated backwards in time first 
config_com = config._replace(N_particles=1,)
params_com = params._replace(t_end=-params.t_end,)

#this is the final position of the cluster, we need to integrate backwards in time 
pos_com_final = jnp.array([[11.8, 0.79, 6.4]]) * u.kpc.to(code_units.code_length)
vel_com_final = jnp.array([[109.5,-254.5,-90.3]]) * (u.km/u.s).to(code_units.code_velocity)
mass_com = jnp.array([params_com.Plummer_params.Mtot])
final_state_com = construct_initial_state(pos_com_final, vel_com_final)

final_state_com = time_integration(final_state_com, mass_com, config_com, params_com)
pos_com, vel_com = final_state_com[:, 0], final_state_com[:, 1]
# Add the center of mass position and velocity to the Plummer sphere particles
positions = positions + pos_com
velocities = velocities + vel_com

#initialize the initial state
initial_state_stream = construct_initial_state(positions, velocities)
final_state = time_integration(initial_state_stream, mass, config, params)
target_stream_clean = projection_on_GD1(final_state, code_units=code_units ) # Reshape to (N_particles, 6)
noise_std = jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0])
# We are going to use a different noiy realization of the target stream for each run of gradient descent 

# Normalization function
@jit
def normalize_data(X):
    """Z-score normalization"""
    mean = jnp.mean(X, axis=0)
    std = jnp.std(X, axis=0)
    return (X - mean) / std, mean, std

@jit
def normalize_stream_data(stream):
    """Normalize stellar stream data for RBF"""
    
    # Remove NaN values first
    # mask = ~jnp.isnan(stream).any(axis=1)
    # stream_clean = stream[mask]
    
    # Normalize each dimension
    stream_norm, mean, std = normalize_data(stream)
    
    return stream_norm, mean, std


#loss (MMD) function
@jit 
def percintile_based_mmd(sim_norm, target_norm, scale_weights = jnp.array([0.1, 0.1, 0.3, 0.25, 0.25])):
    
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

    """MMD using percentiles as natural scales"""
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
    mmd = jnp.sum(scale_weights * jax.vmap(lambda sigma: compute_mmd(sim_norm, target_norm, sigma))(sigmas))/len(sigmas)
    return mmd 

@partial(jax.jit, static_argnames=('nsteps', 'noise_std'))
def compute_minima(key, target_stream_clean=target_stream_clean, noise_std=noise_std, nsteps = 10):
    key_noise_target, key_simulation, key_sampling_values, key_gradascend = random.split(key, 4)
    target_stream = target_stream_clean + random.normal(key_noise_target, shape=target_stream_clean.shape) * noise_std
    target_norm, mean, std = normalize_stream_data(target_stream)

    # time_integration function 
    @jit
    def simulation_time_integration(params_input, key, ):
        #we are going to sample on the log space so to avoid negative values
        params_input = {key: 10**value for key, value in params_input.items()}
        key_Plummer, key_noise = jax.random.split(key, 2)
        #we set up the parameters of the simulations, changing only the parameter that we want to optimize for N-body
        new_params = params._replace(t_end = params_input['t_end'],)
        new_params = new_params._replace(NFW_params = params.NFW_params._replace(Mvir = params_input['M_NFW'], ))
        # new_params = new_params._replace(NFW_params = params.NFW_params._replace(r_s = params_input['rs_NFW'], ))
        new_params = new_params._replace(MN_params = params.MN_params._replace(M = params_input['M_MN'], ))
        # new_params = new_params._replace(MN_params = params.MN_params._replace(a = params_input['a_MN'], ))
        new_params = new_params._replace(Plummer_params = params.Plummer_params._replace(Mtot = params_input['M_plummer'], ))
        # new_params = new_params._replace(Plummer_params = params.Plummer_params._replace(a = params_input['a_plummer'], ))
        #change only the parameters for the center of mass
        new_params_com = params_com._replace(t_end = -params_input['t_end'],)
        new_params_com = new_params_com._replace(NFW_params = params_com.NFW_params._replace(Mvir = params_input['M_NFW'],))
        # new_params_com = new_params_com._replace(NFW_params = params_com.NFW_params._replace(r_s = params_input['rs_NFW'],))
        new_params_com = new_params_com._replace(MN_params = params_com.MN_params._replace(M = params_input['M_MN'], ))
        # new_params_com = new_params_com._replace(MN_params = params_com.MN_params._replace(a = params_input['a_MN'], ))
        new_params_com = new_params_com._replace(Plummer_params = params_com.Plummer_params._replace(Mtot = params_input['M_plummer'],))
        # new_params_com = new_params_com._replace(Plummer_params = params_com.Plummer_params._replace(a = params_input['a_plummer'],))
        
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
        sim_norm = (stream - mean)/std

        #return the negative MMD loss, which will consider as the pseudo log likelihood
        return - percintile_based_mmd(sim_norm, target_norm, ) 
    
    n_sim_grid_search = 10
    t_end_values = jnp.log10(jnp.linspace(params.t_end * (1/4), params.t_end * (8/4), n_sim_grid_search))  # Logarithmic values in the range
    M_plummer_values = jnp.log10(jnp.linspace(params.Plummer_params.Mtot * (1/4), params.Plummer_params.Mtot * (8/4), n_sim_grid_search))  # Logarithmic values in the range
    # a_plummer_values = jnp.log10(jnp.linspace(params.Plummer_params.a * (1/4), params.Plummer_params.a * (8/4), n_sim_grid_search))  # Logarithmic values in the range
    M_NFW_values = jnp.log10(jnp.linspace(params.NFW_params.Mvir * (1/4), params.NFW_params.Mvir * (8/4), n_sim_grid_search))  # Logarithmic values in the range
    # rs_NFW_values = jnp.log10(jnp.linspace(params.NFW_params.r_s * (1/4), params.NFW_params.r_s * (8/4), n_sim_grid_search))  # Logarithmic values in the range
    M_MN_values = jnp.log10(jnp.linspace(params.MN_params.M * (1/4), params.MN_params.M * (8/4), n_sim_grid_search))  # Logarithmic values in the range
    # a_MN_values = jnp.log10(jnp.linspace(params.MN_params.a * (1/4), params.MN_params.a * (8/4), n_sim_grid_search))  # Logarithmic values in the range
    
    ## Create a meshgrid
    # M_plummer_grid, a_plummer_grid, t_end_grid, M_NFW_grid, rs_NFW_grid, M_MN_grid, a_MN_grid = jnp.meshgrid(
    #     M_plummer_values, a_plummer_values, t_end_values, M_NFW_values, rs_NFW_values, M_MN_values, a_MN_values, indexing="ij")

    M_plummer_grid, t_end_grid, M_NFW_grid, M_MN_grid = jnp.meshgrid(
        M_plummer_values, t_end_values, M_NFW_values, M_MN_values, indexing="ij")
    
    # Flatten the grid for vectorized computation
    M_plummer_flat = M_plummer_grid.flatten()
    # a_plummer_flat = a_plummer_grid.flatten()
    t_end_flat = t_end_grid.flatten()
    M_NFW_flat = M_NFW_grid.flatten()
    # rs_NFW_flat = rs_NFW_grid.flatten()
    M_MN_flat = M_MN_grid.flatten()
    # a_MN_flat = a_MN_grid.flatten()
    keys_flat = jax.random.split(key_sampling_values, len(M_plummer_flat))  # Create a flat array of keys
    mesh = Mesh(np.array(jax.devices()), ("i",))
    M_plummer_sharded = jax.device_put(M_plummer_flat, NamedSharding(mesh, PartitionSpec("i")))
    # a_plummer_sharded = jax.device_put(a_plummer_flat, NamedSharding(mesh, PartitionSpec("i")))
    t_end_sharded = jax.device_put(t_end_flat, NamedSharding(mesh, PartitionSpec("i")))
    M_NFW_sharded = jax.device_put(M_NFW_flat, NamedSharding(mesh, PartitionSpec("i")))
    # rs_NFW_sharded = jax.device_put(rs_NFW_flat, NamedSharding(mesh, PartitionSpec("i")))
    M_MN_sharded = jax.device_put(M_MN_flat, NamedSharding(mesh, PartitionSpec("i")))   
    # a_MN_sharded = jax.device_put(a_MN_flat, NamedSharding(mesh, PartitionSpec("i")))
    keys_sharded = jax.device_put(keys_flat, NamedSharding(mesh, PartitionSpec("i")))

    # Create parameter dictionary with all flattened arrays
    params_grid = {
        'M_plummer': M_plummer_sharded,
        # 'a_plummer': a_plummer_sharded, 
        't_end': t_end_sharded,
        'M_NFW': M_NFW_sharded,
        # 'rs_NFW': rs_NFW_sharded,
        'M_MN': M_MN_sharded,
        # 'a_MN': a_MN_sharded,
    }

    # Vectorized simulation function
    @jit
    def simulation_vectorized(params_dict, keys):
        """Vectorized version that processes all parameter combinations"""
        def single_simulation(params_arrays_key):
            params, key = params_arrays_key

            # Convert single row to dictionary
            param_dict = {param_name: params[i] 
                        for i, param_name in enumerate(params_dict.keys())}
            return simulation_time_integration(param_dict, key)
        
        # Stack all parameters into a single array for vmapping
        param_arrays = jnp.stack([params_dict[key] for key in params_dict.keys()], axis=1)
        params_arrays_keys = (param_arrays, keys)
        
        return jax.lax.map(single_simulation, xs=params_arrays_keys, batch_size=550)
        # return jax.vmap(single_simulation, )(params_arrays_keys)

    # Run the vectorized computation
    pseudo_loglikelihood = simulation_vectorized(params_grid, keys_sharded)
    max_index = jnp.argmax(pseudo_loglikelihood)
    M_plummer_min = M_plummer_flat[max_index]
    # a_plummer_min = a_plummer_flat[max_index]
    t_end_min = t_end_flat[max_index]
    M_NFW_min = M_NFW_flat[max_index]
    # rs_NFW_min = rs_NFW_flat[max_index]
    M_MN_min = M_MN_flat[max_index]
    # a_MN_min = a_MN_flat[max_index]

    ## gradient descent 
    ### Gradient of ln-likelihood is simple as
    grad_func = jax.jit(jax.grad(simulation_time_integration))

    @partial(jax.jit, static_argnames=('num_iterations',))
    def gradient_ascent(params, learning_rates, num_iterations=nsteps):
        trajectory = jnp.zeros((num_iterations, len(params.keys())))
        loglike = jnp.zeros((num_iterations))
        keys = jnp.zeros((num_iterations,2))  # Initialize keys as an empty array
        key = key_gradascend  # Initialize a random key for reproducibility
        # for i in range(num_iterations):
        #     grads = grad_func(params, key)
        #     params = {k: v + learning_rates[k] * grads[k] for k, v in params.items()}
        #     ll = simulation_time_integration(params, key)
        #     keys = keys.at[i].set(key)
        #     key, _ = random.split(key)
        #     trajectory = trajectory.at[i].set(jnp.array([v for v in params.values()]))
        #     loglike = loglike.at[i].set(ll)
        #     # if i % 10 == 0:  # Print progress every 10 iterations
        #         # arr = jnp.asarray(loglike)
        #         # print(f"Iteration {i}, max lnlikelihood {str(arr.max())}")
        # return params, trajectory, loglike, keys

        def loop_body(i, carry):
            params, key, trajectory, loglike, keys = carry

            grads = grad_func(params, key)
            # Update parameters
            params = {k: v + learning_rates[k] * grads[k] for k, v in params.items()}

            # Evaluate likelihood
            ll = simulation_time_integration(params, key)

            # Save current key
            keys = keys.at[i].set(key)

            # Split key
            key, _ = random.split(key)

            # Save params trajectory
            trajectory = trajectory.at[i].set(jnp.array([v for v in params.values()]))

            # Save log-likelihood
            loglike = loglike.at[i].set(ll)

            return (params, key, trajectory, loglike, keys)

        # Initialize carry
        init_carry = (params, key, trajectory, loglike, keys)

        # Run fori_loop
        params, key, trajectory, loglike, keys = jax.lax.fori_loop(0, num_iterations, loop_body, init_carry)
        return params, trajectory,  loglike, keys

    lr = 1e-4
    learning_rates = {
        't_end': lr,  # Learning rate for t_end
        'M_plummer': lr,  # Learning rate for M_plummer
        # 'a_plummer': lr,  # Learning rate for a_plummer
        'M_NFW': lr,  # Learning rate for M_NFW
        # 'rs_NFW': lr,  # Learning rate for rs_NFW
        'M_MN': lr,  # Learning rate for M_MN
        # 'a_MN': lr,  # Learning rate for a_MN
    }

    best_fit = {'M_plummer': M_plummer_min,
                # 'a_plummer': a_plummer_min,
                't_end': t_end_min,
                'M_NFW': M_NFW_min,
                # 'rs_NFW': rs_NFW_min,
                'M_MN': M_MN_min,}
                # 'a_MN': a_MN_min,}
    
    params_final, traj, loglike_traj, keys = gradient_ascent(best_fit, learning_rates, nsteps) 
    # return params_final 
    return {k: v for k, v in zip(params_final.keys(), traj[jnp.argmax(loglike_traj)])}

if __name__ == "__main__":
    import time
    for i in range(200):
        start = time.time()
        key = random.PRNGKey(i)
        params_final = compute_minima(key)
        params_final['t_end'] = 10**params_final['t_end'] * code_units.code_time.to(u.Gyr)
        params_final['M_plummer'] = 10**params_final['M_plummer'] * code_units.code_mass.to(u.Msun)
        # params_final['a_plummer'] = 10**params_final['a_plummer'] * code_units.code_length.to(u.pc)
        params_final['M_NFW'] = 10**params_final['M_NFW'] * code_units.code_mass.to(u.Msun)
        # params_final['rs_NFW'] = 10**params_final['rs_NFW'] * code_units.code_length.to(u.kpc)
        params_final['M_MN'] = 10**params_final['M_MN'] * code_units.code_mass.to(u.Msun)
        # params_final['a_MN'] = 10**params_final['a_MN'] * code_units.code_length.to(u.kpc)
        # params_final['a_plummer'] = 10**params_final['a_plummer'] * code_units.code_length.to(u.pc)
        # params_final['rs_NFW'] = 10**params_final['rs_NFW'] * code_units.code_length.to(u.kpc)
        # params_final['a_MN'] = 10**params_final['a_MN'] * code_units.code_length.to(u.kpc)

        print(f"Run {i}: Best fit parameters: {params_final}")
        end = time.time()
        print(f"Time taken for run {i}: {end - start} seconds")
        with open(f'./sampling_target_error_resampling/params_final_{i}.pkl', 'wb') as f:
            pickle.dump(params_final, f)
        # np.save(f'./sampling_target_error_resampling/params_final_{i}.npy', params_final)
