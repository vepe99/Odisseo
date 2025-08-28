import os

from autocvd import autocvd
autocvd(num_gpus = 1)
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # Set the GPU to use, if available

import blackjax
from blackjax.progress_bar import gen_scan_fn

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
snapshots = time_integration(initial_state_stream, mass, config, params)

final_state = snapshots.states[-1].copy()
s = projection_on_GD1(final_state, code_units=code_units,)
target_stream = s

# Gradient on the loss for 2 parameters
# for now we will only use the last snapshot to caluclate the loss and the gradient
config =  config._replace(return_snapshots=False,
                         N_particles = 1000, )
config_com = config_com._replace(return_snapshots=False,)

config =  config._replace(return_snapshots=False,)
config_com = config_com._replace(return_snapshots=False,)
stream_target = s


@jit
def time_integration_NFWM_tend_grad(Mvir, t_end, key):

    #Creation of the Plummer sphere requires a key 
    key = random.PRNGKey(key)

    #we set up the parameters of the simulations, changing only the parameter that we want to optimize
    new_params = params._replace(
                NFW_params=params.NFW_params._replace(
                    Mvir=Mvir 
                ))
    new_params = new_params._replace(
                t_end=t_end,  # Update the t_end parameter
                )
    
    new_params_com = params_com._replace(
                NFW_params=params_com.NFW_params._replace(
                    Mvir=Mvir 
                ))
    
    #we also update the t_end parameter for the center of mass
    new_params_com = new_params_com._replace(
                t_end=-t_end,  # Update the t_end parameter for the center of mass
                )
    
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
    positions, velocities, mass = Plummer_sphere(key=key, params=new_params, config=config)
    #we add the center of mass position and velocity to the Plummer sphere particles
    positions = positions + pos_com
    velocities = velocities + vel_com
    #initialize the initial state
    initial_state_stream = construct_initial_state(positions, velocities, )
    #run the simulation
    final_state = time_integration(initial_state_stream, mass, config=config, params=new_params)

    #projection on the GD1 stream
    stream = projection_on_GD1(final_state, code_units=code_units,)
    noise_std = jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.00001])

    def log_diag_multivariate_normal(x, mean, sigma):
        """
        Log PDF of a multivariate Gaussian with diagonal covariance.
        
        Parameters
        ----------
        x : (D,)
        mean : (D,)
        sigma : (D,)  # standard deviations for each dimension
        """
        diff = (x - mean) / sigma
        D = x.shape[0]
        log_det = 2.0 * jnp.sum(jnp.log(sigma))
        norm_const = -0.5 * (D * jnp.log(2 * jnp.pi) + log_det)
        exponent = -0.5 * jnp.sum(diff**2)
        return norm_const + exponent

    def stream_likelihood_diag(model_stream, obs_stream, obs_errors, ):
        """
        Log-likelihood of observed stars given simulated stream (diagonal covariance).
        
        Parameters
        ----------
        model_stream : (N_model, D)
        obs_stream : (N_obs, D)
        obs_errors : (N_obs, D)   # per-dimension standard deviations
        tau : float
            Stream membership fraction
        p_field : float
            Background probability density
        """
        def obs_log_prob(obs, sigma):
            def model_log_prob(model_point):
                return log_diag_multivariate_normal(obs, model_point, sigma)

            # Compute log_probs for all model points
            log_probs = jax.vmap(model_log_prob)(model_stream)
            
            # Numerically stable average: log(mean(exp(log_probs)))
            log_p_stream = jax.scipy.special.logsumexp(log_probs) - jnp.log(model_stream.shape[0])
            
            # Mixture model
            # p_total = tau * jnp.exp(log_p_stream) + (1 - tau) * p_field
            p_total = jnp.exp(log_p_stream)
            return jnp.log(p_total + 1e-30)

        # Vectorize over observations
        logL_values = jax.vmap(obs_log_prob)(obs_stream, jnp.repeat(obs_errors, obs_stream.shape[0]).reshape(-1, 6))
        return jnp.sum(logL_values)
    
    return stream_likelihood_diag(model_stream=stream,
                             obs_stream=stream_target,
                             obs_errors=noise_std)


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
def time_integration_for_gradient_descend(Mvir_and_t_end, key):
    # Mvir, t_end = de_normalize_Mvir_and_t_end(Mvir_and_t_end)
    Mvir, t_end = Mvir_and_t_end
    Mvir = 10**Mvir  # Convert back to original scale
    t_end = 10**t_end
    return (time_integration_NFWM_tend_grad)(Mvir, t_end, key)

# Calculate the value of the function and the gradient wrt the total mass of the plummer sphere
Mvir = (params.NFW_params.Mvir*(3/4) * u.Msun).to(code_units.code_mass).value
t_end = (params.t_end * (5/4) * u.Gyr).to(code_units.code_time).value  # Example: 25% increase in t_end
key = 0
loss, grad = jax.value_and_grad(lambda Mvir, t_end, key: time_integration_NFWM_tend_grad(jnp.log10(Mvir), jnp.log10(t_end), key), argnums=(0,1))(Mvir, t_end, key)
print("Gradient of the total mass of the Mvir of NFW:\n", grad)
print("Loss:\n", loss)  

n_sim = 10

# M_tot_values = jnp.linspace(params.NFW_params.Mvir*(1/4), params.NFW_params.Mvir*(8/4), n_sim-1) # Adjust range based on expected values
# t_end_values = jnp.linspace(params.t_end * (1/4), params.t_end * (8/4), n_sim-1)   # Adjust range based on expected timescales

M_tot_values = jax.random.uniform(random.PRNGKey(0), shape=(n_sim,), minval=params.NFW_params.Mvir*(1/4), maxval=params.NFW_params.Mvir*(8/4))  # Random values in the range
t_end_values = jax.random.uniform(random.PRNGKey(0), shape=(n_sim,), minval=params.t_end * (1/4), maxval=params.t_end * (8/4))  # Random values in the range

# M_tot_values = jnp.concatenate([M_tot_values, jnp.array([params.NFW_params.Mvir])])  # Append the true Mvir value
# t_end_values = jnp.concatenate([t_end_values, jnp.array([params.t_end])])  # Append the true t_end value
# Ensure both arrays are sorted
M_tot_values = jnp.sort(M_tot_values)
t_end_values = jnp.sort(t_end_values)


# Create a meshgrid
M_tot_grid, t_end_grid,  = jnp.meshgrid(M_tot_values, t_end_values, indexing="ij")

# Flatten the grid for vectorized computation
Mvir_flat = M_tot_grid.flatten()
t_end_flat = t_end_grid.flatten()
keys_flat = jnp.arange(len(Mvir_flat))  # Create a flat array of keys

mesh = Mesh(np.array(jax.devices()), ("i",))
Mvir_sharded = jax.device_put(Mvir_flat, NamedSharding(mesh, PartitionSpec("i")))
t_end_sharded = jax.device_put(t_end_flat, NamedSharding(mesh, PartitionSpec("i")))
keys_sharded = jax.device_put(keys_flat, NamedSharding(mesh, PartitionSpec("i")))


@jit
def time_integration_for_laxmap(input):
    Mvir, t_end, key = input
    return jax.value_and_grad(time_integration_for_gradient_descend,)((jnp.log10(Mvir), jnp.log10(t_end)), key)

loss, grad = jax.lax.map(f=time_integration_for_laxmap, 
                         xs=(Mvir_sharded, t_end_sharded, keys_sharded), 
                         batch_size=3)


loss_min, min_index = jnp.min(loss), jnp.argmin(loss)
Mvir_min = Mvir_flat[jnp.argmin(loss)]
t_end_min = t_end_flat[jnp.argmin(loss)]
print(f"Minimum loss: {loss_min}, Mvir: {Mvir_min * code_units.code_mass.to(u.Msun)}, t_end: {t_end_min * code_units.code_time.to(u.Gyr)}")
loss = loss.reshape(M_tot_grid.shape)

loss_max, max_index = jnp.max(loss), jnp.argmax(loss)
Mvir_max = M_tot_grid.flatten()[max_index]
t_end_max = t_end_grid.flatten()[max_index]
print(f"Maximum loss: {loss_max}, Mvir: {Mvir_max * code_units.code_mass.to(u.Msun)}, t_end: {t_end_max * code_units.code_time.to(u.Gyr)}")


@jit 
def time_integration_NFWM_tend_gradascend(params, key):
    t_end = 10**params['t_end']
    M_NFW = 10**params['M_NFW']
    return time_integration_NFWM_tend_grad(M_NFW, t_end, key)

### Gradient of ln-likelihood is simple as
grad_func = jax.grad(time_integration_NFWM_tend_gradascend)

def gradient_ascent(params, learning_rates, num_iterations):
    trajectory = []
    loglike = []
    keys = []
    key = 1  # Initialize a random key for reproducibility
    for i in range(num_iterations):
        grads = grad_func(params, key)
        params = {k: v + learning_rates[k] * grads[k] for k, v in params.items()}
        ll = time_integration_NFWM_tend_gradascend(params, key)
        trajectory.append(params)
        loglike.append(ll)
        keys.append(key)
        key += 1
        if i % 10 == 0:  # Print progress every 10 iterations
            arr = jnp.asarray(loglike)
            print(f"Iteration {i}, max lnlikelihood {str(arr.max())}")
    return params, trajectory, loglike, keys


lr = 1e-10
learning_rates = {
    't_end': lr,  # Learning rate for t_end
    'M_NFW': lr, 
}

best_fit = {'t_end':jnp.log10(t_end_max).item(), 'M_NFW': jnp.log10(Mvir_max).item()} 
nsteps = 40
params_final, traj, loglike_traj, keys = gradient_ascent(best_fit,learning_rates,nsteps) 

loglike_traj = jnp.asarray(loglike_traj)
id_max = jnp.argmax(loglike_traj)
params_MLE = traj[id_max]

print('t_end', 10**params_MLE['t_end']*code_units.code_time.to(u.Gyr), "Gyr")
print('M_NFW', 10**params_MLE['M_NFW']*code_units.code_mass.to(u.Msun), "Msun")


# -----------------------------
# Your priors & logdensity
# -----------------------------
min_prior_t_end = params.t_end * (1/4)
difference_min_max_tend = params.t_end * (8/4) - min_prior_t_end

min_prior_Mvir = params.NFW_params.Mvir * (1/4)
difference_max_min_Mvir = params.NFW_params.Mvir * (8/4) - min_prior_Mvir

@jit
def log_prob(theta, key):
    Mvir = 10.0 ** theta["M_NFW"]
    t_end = 10.0 ** theta["t_end"]
    return time_integration_NFWM_tend_grad(Mvir, t_end, key)

@jit
def log_prior(theta):  # FIXED: Return log probabilities directly
    Mvir = 10.0 ** theta["M_NFW"]
    t_end = 10.0 ** theta["t_end"]
    lp_M = jax.scipy.stats.uniform.logpdf(Mvir, loc=min_prior_Mvir, scale=difference_max_min_Mvir)
    lp_t = jax.scipy.stats.uniform.logpdf(t_end, loc=min_prior_t_end, scale=difference_min_max_tend)
    
    # Add Jacobian correction for log-space sampling
    jacobian_M = theta["M_NFW"] * jnp.log(10.0)
    jacobian_t = theta["t_end"] * jnp.log(10.0)
    
    return lp_M + lp_t + jacobian_M + jacobian_t

@jit
def logdensity(theta):
    loglik = log_prob(theta, key=0)
    logprior = log_prior(theta)
    
    # Safe handling of NaN/inf
    return jnp.where(
        jnp.isfinite(loglik) & jnp.isfinite(logprior),
        loglik + logprior,
        -jnp.inf
    )
# -----------------------------
# HMC/NUTS settings
# -----------------------------
print('start sampling')
num_chains  = 1
num_warmup  = 50
num_samples = 1_000

# -----------------------------
# Init positions (one per chain)
# -----------------------------
rng_key   = random.PRNGKey(0)
chain_keys = random.split(rng_key, num_chains)

init_positions = {
    "t_end":  params_MLE["t_end"],  # Jitter
    "M_NFW": params_MLE["M_NFW"] 
}
    


# -----------------------------
# Run warmup per chain
# -----------------------------

warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
(state, parameters), _ = warmup.run(warmup_key, init_positions, num_steps=num_warmup)

print('finish warmup')

def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    # progress_bar_scan = gen_scan_fn(num_samples, progress_bar=True,)
    # _, states = progress_bar_scan(one_step, initial_state, keys)
    _, states = jax.lax.scan(one_step, initial_state, keys)


    return states


kernel = blackjax.nuts(logdensity, **parameters).step
states = inference_loop(sample_key, kernel, state, num_samples)

mcmc_samples = states.position


# Convert list to NumPy array
np.savez('./NUTS_newlikelihood/samples.npz', 
         M_NFW = mcmc_samples['M_NFW'],
         t_end = mcmc_samples['t_end'] )

