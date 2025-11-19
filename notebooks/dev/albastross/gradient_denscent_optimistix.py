# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7' 
from autocvd import autocvd
autocvd(num_gpus = 1)
import pandas as pd
from equinox import filter_jit

import jax 
import jax.numpy as jnp
from jax import jit, random
import equinox as eqx
from jax.sharding import Mesh, PartitionSpec, NamedSharding

# jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from functools import partial

import numpy as np
from astropy import units as u
from astropy import constants as c

import odisseo
from odisseo import construct_initial_state
from odisseo.integrators import leapfrog
from odisseo.dynamics import direct_acc, DIRECT_ACC, DIRECT_ACC_LAXMAP, DIRECT_ACC_FOR_LOOP, DIRECT_ACC_MATRIX, NO_SELF_GRAVITY
from odisseo.option_classes import SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams, PSPParams, MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL, DIFFRAX_BACKEND, LEAPFROG
from odisseo.option_classes import SEMIIMPLICITEULER, TSIT5
from odisseo.option_classes import RECURSIVECHECKPOINTADJOING, FORWARDMODE
from odisseo.initial_condition import Plummer_sphere, Plummer_sphere_reparam
from odisseo.utils import center_of_mass
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.visualization import create_3d_gif, create_projection_gif, energy_angular_momentum_plot
from odisseo.potentials import MyamotoNagai, NFW

from odisseo.utils import halo_to_gd1_velocity_vmap, halo_to_gd1_vmap, projection_on_GD1
from jax.test_util import check_grads

plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 15,
})

plt.style.use('default')


code_length = 10 * u.kpc
code_mass = 1e4 * u.Msun
G = 1
code_time = 3 * u.Gyr
code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  


config = SimulationConfig(N_particles = 1_000, 
                          return_snapshots = True, 
                          num_snapshots = 1000, 
                          num_timesteps = 1000, 
                          external_accelerations=(NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL), 
                          acceleration_scheme = DIRECT_ACC_MATRIX,
                          softening = (0.1 * u.pc).to(code_units.code_length).value,
                          integrator = DIFFRAX_BACKEND,
                          differentation_mode=TSIT5,
                          fixed_timestep=False,
                          ) #default values

params = SimulationParams(t_end = (3 * u.Gyr).to(code_units.code_time).value,  
                          Plummer_params= PlummerParams(Mtot=(1.12 * 10**4 * u.Msun).to(code_units.code_mass).value,
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
config_com = config._replace(N_particles=1,)
params_com = params._replace(t_end=-params.t_end,)

#this is the final position of the cluster, we need to integrate backwards in time 
pos_com_final = jnp.array([[11.8, 0.79, 6.4]]) * u.kpc.to(code_units.code_length)
vel_com_final = jnp.array([[109.5,-254.5,-90.3]]) * (u.km/u.s).to(code_units.code_velocity)


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

final_state = snapshots.states[-1]
stream_data = projection_on_GD1(final_state, code_units=code_units,)
print("Simulated GD1")



# LET'S TRY OPTIMIZING IT
config =  config._replace(return_snapshots=False,)
config_com = config_com._replace(return_snapshots=False,)

config =  config._replace(return_snapshots=False,)
config_com = config_com._replace(return_snapshots=False,)
stream_target = stream_data

@jit
def rbf_kernel(x, y, sigma):
    """RBF kernel optimized for 6D astronomical data"""
    return jnp.exp(-jnp.sum((x - y)**2) / (2 * sigma**2))


@filter_jit
def run_simulation( y, args):

    t_end, M_Plummer, Mvir, r_s = y
    t_end = 10**t_end
    M_plummer = 10**M_Plummer
    Mvir = 10**Mvir
    r_s = 10**r_s

    #Creation of the Plummer sphere requires a key 
    key = random.PRNGKey(0)

    #we set up the parameters of the simulations, changing only the parameter that we want to optimize
    new_params = params._replace(
                NFW_params=params.NFW_params._replace(
                    Mvir=Mvir * u.Msun.to(code_units.code_mass),
                    r_s=r_s * u.kpc.to(code_units.code_length),
                ),
                Plummer_params=params.Plummer_params._replace(
                    Mtot=M_plummer * u.Msun.to(code_units.code_mass)
                ),
                t_end=t_end * u.Gyr.to(code_units.code_time),
                )
    
    #we also update the t_end parameter for the center of mass
    new_params_com = new_params._replace(
                t_end=-t_end * u.Gyr.to(code_units.code_time),  # Update the t_end parameter for the center of mass
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
    #add gaussian noise to the stream
    # noise_std = jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0])
    # stream = stream + jax.random.normal(key=jax.random.key(0), shape=stream.shape) * noise_std
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
        
    def normalize_stream(stream):
        # Normalize each dimension to [0,1]
        return (stream - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    
    sim_norm = normalize_stream(stream)
    target_norm = normalize_stream(stream_target)
    
    # Adaptive bandwidth for 6D data
    n_sim, n_target = len(stream), len(stream_target)


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
        # jnp.percentile(distance_flat, 10),   # Fine scale
        # jnp.percentile(distance_flat, 25),   # Small scale  
        # jnp.percentile(distance_flat, 50),   # Medium scale (median)
        jnp.percentile(distance_flat, 75),   # Large scale
        jnp.percentile(distance_flat, 90),   # Very large scale
    ])

    # Adaptive weights based on scale separation
    # scale_weights = jnp.array([0.15, 0.2, 0.3, 0.25, 0.1])
    scale_weights = jnp.ones_like(sigmas)  # Equal weights for simplicity

    # Compute MMD with multiple kernels
    mmd_total = jnp.sum(scale_weights * jax.vmap(lambda sigma: compute_mmd(sim_norm, target_norm, sigma))(sigmas))
    
    return mmd_total / len(sigmas)

print("beginning gradient descent")

from optimistix import minimise, BestSoFarMinimiser
import optimistix
from optax import adamw



# Shape will be (4, 4)
def sample_initial_conditions(key, n_samples, params, code_units):
    """
    Sample initial conditions from a uniform prior.
    
    Parameters are sampled uniformly in log-space between 0.5x and 2x the true values.
    
    Args:
        key: JAX random key
        n_samples: Number of samples to generate
        params: SimulationParams object with true parameter values
        code_units: CodeUnits object for unit conversions
    
    Returns:
        Array of shape (n_samples, 5) with log10 of [Mvir, M_MN, r_s, a, b]
    """
    # Get true values in physical units
    true_M_Plummer = params.Plummer_params.Mtot * code_units.code_mass.to(u.Msun)
    true_Mvir = params.NFW_params.Mvir * code_units.code_mass.to(u.Msun)
    true_r_s = params.NFW_params.r_s * code_units.code_length.to(u.kpc)
    true_t_end = params.t_end * code_units.code_time.to(u.Gyr)
    
    # Define bounds: [0.5 * true_value, 2.0 * true_value]
    min_factor = 0.5
    max_factor = 2.0
    
    # Stack true values
    true_values = jnp.array([true_M_Plummer, true_Mvir, true_r_s, true_t_end])
    
    # Calculate min and max in log space
    # log_min = jnp.log10(true_values * min_factor)
    # log_max = jnp.log10(true_values * max_factor)
     
    min_vals = true_values * min_factor
    min_vals = min_vals.at[0].set(1e3)  # M_Plummer minimum
    min_vals = min_vals.at[3].set(0.5)
    max_vals = true_values * max_factor
    max_vals = max_vals.at[0].set(10**5)  # M_Plummer maximum
    max_vals = max_vals.at[3].set(5.0)
    # Sample uniformly in log space
    keys = random.split(key, 5)
    samples = []
    
    for i in range(4):
        param_samples = random.uniform(
            keys[i], 
            shape=(n_samples,), 
            minval=min_vals[i], 
            maxval=max_vals[i]
        )
        samples.append(param_samples)
    
    # Stack into (n_samples, 5) array
    y0_batched = jnp.stack(samples, axis=1)
    y0_batched = jnp.log10(y0_batched)
    
    return y0_batched


y0_batched = sample_initial_conditions(random.PRNGKey(0), 1200, params, code_units)

# Shape will be (4, 4)
def minimization_vmap(y0):
    return minimise(
        fn=run_simulation,
        solver = BestSoFarMinimiser(optimistix.OptaxMinimiser(optim = adamw(learning_rate=1e-3,),  rtol=1e-3, atol=1e-4, )),
        # solver = BestSoFarMinimiser(optimistix.LBFGS(rtol=1e-8, atol=1e-8)),
        y0=y0,
        max_steps=100
    ).value

print(y0_batched.shape)
# values = jax.vmap(minimization_vmap)(y0_batched)
values = jax.lax.map(minimization_vmap, y0_batched, batch_size=150)
values = 10**values

np.savez("gradient_descent_optimistix_results.npz", values=np.array(values))

true_M_Plummer = params.Plummer_params.Mtot * code_units.code_mass.to(u.Msun)
true_Mvir = params.NFW_params.Mvir * code_units.code_mass.to(u.Msun)
true_r_s = params.NFW_params.r_s * code_units.code_length.to(u.kpc)
true_t_end = params.t_end * code_units.code_time.to(u.Gyr)

# Stack true values
true_values = jnp.array([true_M_Plummer, true_Mvir, true_r_s, true_t_end])

# print('Minimization result:', values)
# print("True values:", true_values)


numeric_df = pd.DataFrame(np.array(values), columns=["$M_{Plummer}$", "$M^{NFW}$", "$r^{NFW}_s$", "$t_{end}$"])
from chainconsumer import Chain, ChainConsumer, Truth

c = ChainConsumer()
c.add_chain(Chain(samples=numeric_df, name="Gradient descend", ), )

c.add_truth(Truth(location = {"$M_{Plummer}$": params.Plummer_params.Mtot * code_units.code_mass.to(u.Msun),
                              "$M^{NFW}$": params.NFW_params.Mvir * code_units.code_mass.to(u.Msun), 
                              "$r^{NFW}_s$": params.NFW_params.r_s * code_units.code_length.to(u.kpc),
                              "$t_{end}$": params.t_end * code_units.code_time.to(u.Gyr),
                              }, color='red', name="True value"))

fig = c.plotter.plot()
fig.savefig("Gradient_descent.png", dpi=300)
