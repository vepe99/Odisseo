# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7' 
from autocvd import autocvd
autocvd(num_gpus = 5)
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



@jax.jit
def halo_to_sun(Xhalo: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from simulation frame to cartesian frame centred at Sun
    Args:
      Xhalo: 3d position (x [kpc], y [kpc], z [kpc]) in simulation frame
    Returns:
      3d position (x_s [kpc], y_s [kpc], z_s [kpc]) in Sun frame
    Examples
    --------
    >>> halo_to_sun(jnp.array([1.0, 2.0, 3.0]))
    """
    sunx = 8.0
    xsun = sunx - Xhalo[0]
    ysun = Xhalo[1]
    zsun = Xhalo[2]
    return jnp.array([xsun, ysun, zsun])


@jax.jit
def sun_to_gal(Xsun: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from sun cartesian frame to galactic co-ordinates
    Args:
      Xsun: 3d position (x_s [kpc], y_s [kpc], z_s [kpc]) in Sun frame
    Returns:
      3d position (r [kpc], b [rad], l [rad]) in galactic frame
    Examples
    --------
    >>> sun_to_gal(jnp.array([1.0, 2.0, 3.0]))
    """
    r = jnp.linalg.norm(Xsun)
    b = jnp.arcsin(Xsun[2] / r)
    l = jnp.arctan2(Xsun[1], Xsun[0])
    return jnp.array([r, b, l])


@jax.jit
def gal_to_equat(Xgal: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from galactic co-ordinates to equatorial co-ordinates
    Args:
      Xgal: 3d position (r [kpc], b [rad], l [rad]) in galactic frame
    Returns:
      3d position (r [kpc], alpha [rad], delta [rad]) in equatorial frame
    Examples
    --------
    >>> gal_to_equat(jnp.array([1.0, 2.0, 3.0]))
    """
    dNGPdeg = 27.12825118085622
    lNGPdeg = 122.9319185680026
    aNGPdeg = 192.85948
    dNGP = dNGPdeg * jnp.pi / 180.0
    lNGP = lNGPdeg * jnp.pi / 180.0
    aNGP = aNGPdeg * jnp.pi / 180.0
    r = Xgal[0]
    b = Xgal[1]
    l = Xgal[2]
    sb = jnp.sin(b)
    cb = jnp.cos(b)
    sl = jnp.sin(lNGP - l)
    cl = jnp.cos(lNGP - l)
    cs = cb * sl
    cc = jnp.cos(dNGP) * sb - jnp.sin(dNGP) * cb * cl
    alpha = jnp.arctan(cs / cc) + aNGP
    delta = jnp.arcsin(jnp.sin(dNGP) * sb + jnp.cos(dNGP) * cb * cl)
    return jnp.array([r, alpha, delta])


def transform_velocity(transform_fn, X, V):
    """
    Generic velocity transformation through coordinate mapping.

    Args:
      transform_fn: function R^3 → R^3 mapping positions to new coordinates
      X: position vector in original coordinates (3,)
      V: velocity vector in original coordinates (3,)

    Returns:
      velocity vector in transformed coordinates (3,)
    """
    J = jax.jacobian(transform_fn)(X)  # (3,3) Jacobian
    return J @ V

def halo_to_equatorial(Xhalo):
    Xsun = halo_to_sun(Xhalo)
    Xgal = sun_to_gal(Xsun)
    Xeq  = gal_to_equat(Xgal)
    return Xeq


#vamp functions
halo_to_equatorial_batch = jax.vmap(halo_to_equatorial, in_axes=(0))
transform_velocity_batch = jax.vmap(transform_velocity, in_axes=(None, 0, 0))



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


stream_target = snapshots.states[-1]
print('shape stream target before projection', stream_target.shape)

# stream_target = projection_on_GD1(stream_target, code_units=code_units,)
pos_stream_target = stream_target[:,0]
vel_stream_target = stream_target[:,1]
pos_eq_stream_target = halo_to_equatorial_batch(pos_stream_target)
vel_eq_stream_target = transform_velocity_batch(halo_to_equatorial, pos_stream_target, vel_stream_target)
stream_target = jnp.concatenate([pos_eq_stream_target, vel_eq_stream_target], axis=1)
print('shape simulated target', stream_target.shape)
print("Simulated GD1")



# LET'S TRY OPTIMIZING IT
config =  config._replace(return_snapshots=False,)
config_com = config_com._replace(return_snapshots=False,)

config =  config._replace(return_snapshots=False,)
config_com = config_com._replace(return_snapshots=False,)

@filter_jit
def run_simulation( y, args):

    M_Plummer, Mvir, r_s, t_end= y
    t_end = 10**t_end
    M_Plummer = 10**M_Plummer
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
                    Mtot=M_Plummer * u.Msun.to(code_units.code_mass)
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
    stream = time_integration(initial_state_stream, mass, config=config, params=new_params)

    pos_stream = stream[:, 0]
    vel_stream = stream[:, 1]
    pos_eq_stream = halo_to_equatorial_batch(pos_stream)
    vel_eq_stream = transform_velocity_batch(halo_to_equatorial, pos_stream, vel_stream)
    stream = jnp.concatenate([pos_eq_stream, vel_eq_stream], axis=1)

    #projection on the GD1 stream
    # stream = projection_on_GD1(final_state, code_units=code_units,)
    #add gaussian noise to the stream
    # noise_std = jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0])
    # stream = stream + jax.random.normal(key=jax.random.key(0), shape=stream.shape) * noise_std
    #we calculate the loss as the negative log likelihood of the stream


     # Normalize to standard ranges for each dimension
    # bounds = jnp.array([
    #     [6, 20],        # R [kpc]
    #     [-120, 70],     # phi1 [deg]  
    #     [-8, 2],        # phi2 [deg]
    #     [-250, 250],    # vR [km/s]
    #     [-2., 1.0],     # v1_cosphi2 [mas/yr]
    #     [-0.10, 0.10]   # v2 [mas/yr]
    # ])
        
    # def normalize_stream(stream):
    #     # Normalize each dimension to [0,1]
    #     return (stream - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    
    # sim_norm = normalize_stream(stream)
    # target_norm = normalize_stream(stream_target)
    
    # # Adaptive bandwidth for 6D data
    # n_sim, n_target = len(stream), len(stream_target)


    # @jit 
    # def compute_mmd(sim_norm, target_norm, sigmas):
    #     xx = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda xj: rbf_kernel(xi, xj, sigmas))(sim_norm))(sim_norm))
    #     yy = jnp.mean(jax.vmap(lambda yi: jax.vmap(lambda yj: rbf_kernel(yi, yj, sigmas))(target_norm))(target_norm))
    #     xy = jnp.mean(jax.vmap(lambda xi: jax.vmap(lambda yj: rbf_kernel(xi, yj, sigmas))(target_norm))(sim_norm))
    #     return xx + yy - 2 * xy

    # distances = jax.vmap(lambda x: jax.vmap(lambda y: jnp.linalg.norm(x - y))(target_norm))(sim_norm)
    # distance_flat = distances.flatten()

    # # # Use percentiles as natural scales
    # sigmas = jnp.array([
    #     # jnp.percentile(distance_flat, 10),   # Fine scale
    #     # jnp.percentile(distance_flat, 25),   # Small scale  
    #     # jnp.percentile(distance_flat, 50),   # Medium scale (median)
    #     jnp.percentile(distance_flat, 75),   # Large scale
    #     jnp.percentile(distance_flat, 90),   # Very large scale
    # ])

    # # Adaptive weights based on scale separation
    # # scale_weights = jnp.array([0.15, 0.2, 0.3, 0.25, 0.1])
    # scale_weights = jnp.ones_like(sigmas)  # Equal weights for simplicity

    # # Compute MMD with multiple kernels
    # mmd_total = jnp.sum(scale_weights * jax.vmap(lambda sigma: compute_mmd(sim_norm, target_norm, sigma))(sigmas))
    
    # return mmd_total / len(sigmas)
    # we calculate the loss as the negative log likelihood of the stream
    
    #add gaussian noise to the stream
    # noise_std = jnp.array([0.25, 0.001, 0.15, 5., 0.1, 0.0001])
    noise_std = jnp.zeros(6)  # no observational errors

   
    @jax.jit
    def log_multivariate_normal(x, mean, cov):
        """
        Log PDF of a multivariate Gaussian with full covariance.

        Parameters
        ----------
        x : (D,)
        mean : (D,)
        cov : (D, D)  # covariance matrix (must be symmetric positive definite)
        """
        D = x.shape[0]
        L = jnp.linalg.cholesky(cov)
        diff = x - mean
        solve = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
        mahal = jnp.sum(solve**2)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        norm_const = -0.5 * (D * jnp.log(2 * jnp.pi) + log_det)
        return norm_const - 0.5 * mahal


    @jax.jit
    def stream_likelihood_fullcov(model_stream, obs_stream, obs_errors, smooth_sigma, ):
        """
        Log-likelihood of observed stars given simulated stream (full covariance version).
        """
        cov = smooth_sigma

        def obs_log_prob(obs):
            def model_log_prob(model_point):
                return log_multivariate_normal(obs, model_point, cov)
            log_probs = jax.vmap(model_log_prob)(model_stream)
            return jax.scipy.special.logsumexp(log_probs) - jnp.log(model_stream.shape[0])

        logL_values = jax.vmap(obs_log_prob)(obs_stream)
        return jnp.sum(logL_values)
    
    stream_cov = 0.1*jnp.cov(stream_target.T)
    return - stream_likelihood_fullcov(model_stream=stream,
                             obs_stream=stream_target,
                             obs_errors=noise_std,
                             smooth_sigma=stream_cov)

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


y0_batched = sample_initial_conditions(random.PRNGKey(0), 500, params, code_units)

# Shape will be (4, 4)
# def minimization_vmap(y0):
#     return minimise(
#         fn=run_simulation,
#         solver = BestSoFarMinimiser(optimistix.OptaxMinimiser(optim = adamw(learning_rate=1e-2,),  rtol=1e-3, atol=1e-4, )),
#         y0=y0,
#         max_steps=50
#     ).value

import jax
import jax.numpy as jnp
import optax

# Your loss wrapper
def loss_fn(y):
    return run_simulation(y, None)

value_and_grad_loss = jax.value_and_grad(loss_fn)

# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)   # or adam, rmsprop, lion, etc.

@jax.jit
def gd_optax_steps(y0, num_steps=30):
    # Initialize optimizer state
    opt_state = optimizer.init(y0)

    # Carry: params, opt_state, best_params, best_loss
    init_best_loss = jnp.inf
    carry0 = (y0, opt_state, y0, init_best_loss)

    def body_fun(i, carry):
        y, opt_state, best_y, best_loss = carry

        loss, grad = value_and_grad_loss(y)
        updates, opt_state = optimizer.update(grad, opt_state, y)
        y_next = optax.apply_updates(y, updates)

        improved = loss < best_loss
        best_y = jnp.where(improved, y_next, best_y)
        best_loss = jnp.where(improved, loss, best_loss)

        return (y_next, opt_state, best_y, best_loss)

    _, _, best_y, _ = jax.lax.fori_loop(0, num_steps, body_fun, carry0)
    return best_y


def minimization_vmap(y0):
    return gd_optax_steps(y0, num_steps=20)

from jax.sharding import Mesh, PartitionSpec, NamedSharding
mesh = Mesh(np.array(jax.devices()), ("i",))
y0_batched = jax.lax.with_sharding_constraint(y0_batched, NamedSharding(mesh, PartitionSpec("i")))


print(y0_batched.shape)
print('/////')
print('y0: ')
print(y0_batched)
print('/////')
# values = jax.vmap(minimization_vmap)(y0_batched)
values = jax.lax.map(minimization_vmap, y0_batched, batch_size=100)
print('values: \n', values)
# values = 10**values


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
